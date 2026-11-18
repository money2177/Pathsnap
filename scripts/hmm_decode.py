"""Stage 7: HMM / Viterbi route reconstruction on top of the tuned HistGradientBoosting.

1. Out-of-fold (OOF) classifier probabilities on the 78 reliable train trips: the tuned HGB is
   refit on 4 of the 5 GroupKFold folds (same folds as stages 4 and 6) and scores the held-out
   trips, so the decoder is tuned on probabilities it has not been trained on.
2. Decoder parameters (emission weight, beta, break cost, u-turn cost, gap split) are tuned with
   Optuna TPE on the train OOF probabilities only (objective: point-level Top-1 accuracy).
3. student_test is decoded once with the tuned parameters, using the probabilities of the final
   refit model from stage 6, and compared with:
     - classifier only (argmax per point, no sequence model)
     - HMM with geometric emissions only (classic Gaussian distance, no classifier)
     - HMM with classifier emissions and the untuned default parameters
Outputs in outputs/hmm/.

Run: python scripts/hmm_decode.py [--trials 60]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, optuna, pandas as pd
from sklearn.model_selection import GroupKFold
from src.data_io import REPO_ROOT, Trip
from src.evaluation import continuity
from src.features import FEATURES
from src.hmm import (BACKWARD_TOL_M, MAX_SEARCH_M, MIN_SEARCH_M, ROUTE_FACTOR, HMMParams, RoadNetwork,
                     build_lattice, classifier_rows, path_connectivity, reconstruct_route, route_overlap, viterbi)
from src.models import SEED
from train_compare import N_FOLDS, load_train, log
from tune_hgb import make_model

OUT = os.path.join(REPO_ROOT, "outputs", "hmm")
TUNED = os.path.join(REPO_ROOT, "outputs", "models", "tuning", "tuning_result_stage2.json")
PRED = os.path.join(REPO_ROOT, "outputs", "models", "compare", "student_test_predictions.parquet")
PROB_COL = "HistGradientBoosting_tuned_stage2"
COLS = ["trip", "point_idx", "arc_id", "proj_frac", "dist", "y"]


def oof_probabilities():
    path = f"{OUT}/train_oof.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    params = json.load(open(TUNED))["best_params"]
    fit_df, ev = load_train()
    X, y = fit_df[FEATURES].to_numpy(np.float32), fit_df.y.to_numpy()
    p = np.full(len(ev), np.nan)
    for k, (tr, va) in enumerate(GroupKFold(n_splits=N_FOLDS).split(fit_df, y, fit_df.trip)):
        t0 = time.time()
        m = make_model(params).fit(X[tr], y[tr])
        mask = ev.trip.isin(fit_df.trip.iloc[va].unique()).to_numpy()
        p[mask] = m.predict_proba(ev.loc[mask, FEATURES].to_numpy(np.float32))[:, 1]
        log(f"OOF fold {k}: {mask.sum()} rows ({time.time()-t0:.0f}s)")
    assert not np.isnan(p).any()
    out = ev[COLS].assign(prob=p.astype(np.float32), fold=-1)
    for k, (_, va) in enumerate(GroupKFold(n_splits=N_FOLDS).split(fit_df, y, fit_df.trip)):
        out.loc[out.trip.isin(fit_df.trip.iloc[va].unique()), "fold"] = k
    out.to_parquet(path, index=False)
    return out


def lattices(df, partition):
    L, nets = {}, {}
    for tid, c in df.groupby("trip", sort=True):
        nets[tid] = RoadNetwork(Trip(tid, partition))
        L[tid] = build_lattice(nets[tid], c, c.prob.to_numpy())
    return L, nets


def decode_all(L, prm, method):
    """method: 'classifier' | 'hmm' | 'hmm_geometric' -> {trip: (rows, info, seconds)}."""
    res = {}
    for tid, lat in L.items():
        t0 = time.time()
        if method == "classifier":
            rows, info = classifier_rows(lat), {}
        else:
            rows, info = viterbi(lat, prm, emission="distance" if method == "hmm_geometric" else "classifier")
        res[tid] = (rows, info, time.time() - t0)
    return res


def point_acc(L, res):
    c = sum(int(L[t].y[r].sum()) for t, (r, _, _) in res.items())
    return c / sum(len(L[t].point_idx) for t in res)


def evaluate(L, nets, res, partition, prm):
    rows = []
    for tid, (r, info, sec) in res.items():
        lat, net = L[tid], nets[tid]
        arcs = lat.arc[r]
        route, holes = reconstruct_route(net, arcs)
        rows.append(dict(trip=tid, n_points=len(r), top1_accuracy=float(lat.y[r].mean()),
                         route_continuity=continuity(arcs, tid, partition),
                         network_connectivity=path_connectivity(lat, r, prm.gap_s),
                         unjoinable_transitions=holes, decode_seconds=sec, **info,
                         **route_overlap(net.g, route, net.trip.route)))
    pt = pd.DataFrame(rows)
    w = pt.n_points
    agg = {k: float(np.average(pt[k], weights=w)) for k in
           ["top1_accuracy", "route_continuity", "network_connectivity", "route_precision", "route_recall", "route_f1"]}
    agg.update(trip_mean_route_f1=float(pt.route_f1.mean()), unjoinable_transitions=int(pt.unjoinable_transitions.sum()),
               decode_ms_per_1k_points=1e6 * pt.decode_seconds.sum() / w.sum())
    return agg, pt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    oof = oof_probabilities()
    t0 = time.time(); Ltr, Ntr = lattices(oof, "train"); build_tr = time.time() - t0
    log(f"train lattices: {len(Ltr)} trips ({build_tr:.0f}s)")
    test = pd.read_parquet(os.path.join(REPO_ROOT, "outputs", "features", "student_test.parquet"), columns=COLS)
    test["prob"] = pd.read_parquet(PRED)[PROB_COL].to_numpy()
    t0 = time.time(); Lte, Nte = lattices(test, "student_test"); build_te = time.time() - t0
    log(f"student_test lattices: {len(Lte)} trips ({build_te:.1f}s)")
    fold_of = oof.groupby("trip").fold.first().to_dict()

    default = HMMParams()

    def objective(trial):
        prm = HMMParams(emission_weight=trial.suggest_float("emission_weight", 0.1, 5.0, log=True),
                        beta=trial.suggest_float("beta", 2.0, 200.0, log=True),
                        break_cost=trial.suggest_float("break_cost", 0.5, 200.0, log=True),
                        uturn_cost=trial.suggest_float("uturn_cost", 0.0, 30.0),
                        gap_s=trial.suggest_float("gap_s", 10.0, 600.0, log=True))
        res = decode_all(Ltr, prm, "hmm")
        per_fold = [point_acc(Ltr, {t: v for t, v in res.items() if fold_of[t] == k}) for k in range(N_FOLDS)]
        trial.set_user_attr("fold_top1", per_fold)
        return point_acc(Ltr, res)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=12))
    study.enqueue_trial({k: v for k, v in default.to_dict().items() if k != "prob_floor"})   # anchor: defaults
    t0 = time.time()
    study.optimize(objective, n_trials=args.trials,
                   callbacks=[lambda s, t: log(f"trial {t.number}: {t.value:.4f} {t.params}") if t.number % 10 == 0 or t.value == s.best_value else None])
    search_s = time.time() - t0
    tuned = HMMParams(**study.best_params)
    log(f"BEST train-OOF top1 {study.best_value:.4f} {study.best_params} ({search_s/60:.1f} min)")

    methods = {"classifier_only": ("classifier", default), "hmm_geometric_emission": ("hmm_geometric", tuned),
               "hmm_default_params": ("hmm", default), "hmm_tuned": ("hmm", tuned)}
    summary, per_trip = {}, []
    for part, L, N in [("train_oof", Ltr, Ntr), ("student_test", Lte, Nte)]:
        for name, (meth, prm) in methods.items():
            agg, pt = evaluate(L, N, decode_all(L, prm, meth), "train" if part == "train_oof" else "student_test", prm)
            summary[f"{part}/{name}"] = agg
            per_trip.append(pt.assign(partition=part, method=name))
            log(f"{part:12s} {name:24s} top1={agg['top1_accuracy']:.4f} cont={agg['route_continuity']:.4f} "
                f"conn={agg['network_connectivity']:.4f} routeF1={agg['route_f1']:.4f}")

    # decoded student_test predictions (one row per GPS point) for the notebook, maps and the API
    rows = []
    for tid, lat in Lte.items():
        rc, rh = classifier_rows(lat), viterbi(lat, tuned)[0]
        rows.append(pd.DataFrame({"trip": tid, "point_idx": lat.point_idx, "arc_classifier": lat.arc[rc],
                                  "arc_hmm": lat.arc[rh], "prob_classifier": lat.prob[rc], "prob_hmm_choice": lat.prob[rh],
                                  "correct_classifier": lat.y[rc], "correct_hmm": lat.y[rh], "dt": lat.dt}))
    pd.concat(rows).to_parquet(f"{OUT}/student_test_decoded.parquet", index=False)
    pd.concat(per_trip).to_csv(f"{OUT}/per_trip_results.csv", index=False)
    study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "duration")).to_csv(f"{OUT}/decoder_trials.csv", index=False)
    pd.DataFrame(summary).T.to_csv(f"{OUT}/comparison.csv")
    json.dump(dict(classifier="HistGradientBoosting (tuned, stage 6)", classifier_params=json.load(open(TUNED))["best_params"],
                   hmm_params=tuned.to_dict(), default_params=default.to_dict(),
                   constants=dict(BACKWARD_TOL_M=BACKWARD_TOL_M, ROUTE_FACTOR=ROUTE_FACTOR,
                                  MIN_SEARCH_M=MIN_SEARCH_M, MAX_SEARCH_M=MAX_SEARCH_M, gps_sigma_geometric_m=5.0),
                   search=dict(objective="point-level Top-1 accuracy on train out-of-fold probabilities (78 reliable trips)",
                               sampler="TPE(seed=42, 12 startup)", n_trials=len(study.trials), search_seconds=search_s,
                               best_value=study.best_value, best_fold_top1=study.best_trial.user_attrs["fold_top1"],
                               space={"emission_weight": "log-uniform 0.1-5", "beta": "log-uniform 2-200 m",
                                      "break_cost": "log-uniform 0.5-200 (widened from 5-200: first search hit 5)", "uturn_cost": "uniform 0-30 (widened from 0-10: first search hit 9.2)",
                                      "gap_s": "log-uniform 10-600 s"}),
                   lattice_build_seconds=dict(train=build_tr, student_test=build_te), results=summary),
              open(f"{OUT}/hmm_config.json", "w"), indent=2, default=float)
    log("done")
