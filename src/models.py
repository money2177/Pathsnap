"""Model zoo for the candidate classifier (untuned, documented initial configurations).

Every model is a scikit-learn estimator exposing predict_proba. Learned preprocessing
(StandardScaler) is part of each pipeline, so it is fitted on training folds only.
`max_train_rows` caps the training size for models whose cost grows faster than linearly
(KNN prediction, kernel SVM); the cap is applied by sampling whole GPS points, so every
sampled point keeps all its candidates.
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC

SEED = 42

MODELS = {
    "LogisticRegression": dict(
        family="Linear / probabilistic",
        config="L2, C=1.0, lbfgs, max_iter=2000, standardised",
        loss="Binary cross-entropy (log loss) + L2 penalty",
        build=lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
        max_train_rows=None,
    ),
    "GaussianNB": dict(
        family="Probabilistic (generative)",
        config="var_smoothing=1e-9, quantile-normal transformed inputs",
        loss="Maximum likelihood of class-conditional Gaussians",
        build=lambda: make_pipeline(QuantileTransformer(output_distribution="normal", n_quantiles=500,
                                                        subsample=200_000, random_state=SEED), GaussianNB()),
        max_train_rows=None,
    ),
    "KNN": dict(
        family="Instance-based",
        config="k=25, distance-weighted, Euclidean, standardised",
        loss="None (lazy learner; vote of neighbours)",
        build=lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
        max_train_rows=150_000,
    ),
    "SVM_RBF": dict(
        family="Kernel-based",
        config="RBF, C=1.0, gamma='scale', Platt-scaled probabilities, standardised",
        loss="Hinge loss + L2 (probabilities via Platt sigmoid)",
        build=lambda: make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale", probability=True,
                                                         cache_size=2000, random_state=SEED)),
        max_train_rows=20_000,
    ),
    "RandomForest": dict(
        family="Ensemble (bagging)",
        config="300 trees, max_depth=None, min_samples_leaf=5, max_features='sqrt'",
        loss="Gini impurity (greedy splits)",
        build=lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                             n_jobs=-1, random_state=SEED),
        max_train_rows=None,
    ),
    "HistGradientBoosting": dict(
        family="Boosting",
        config="lr=0.1, max_iter=300, max_leaf_nodes=31, early stopping (10% val, patience 20)",
        loss="Binary cross-entropy (log loss)",
        build=lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, max_leaf_nodes=31,
                                                     early_stopping=True, validation_fraction=0.1,
                                                     n_iter_no_change=20, scoring="loss", random_state=SEED),
        max_train_rows=None,
    ),
    "LightGBM": dict(
        family="Boosting",
        config="lr=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, colsample=0.8, "
               "<=2000 rounds, early stop on 10% trip-grouped val (patience 50, keeps best round)",
        loss="Binary cross-entropy (log loss)",
        build=lambda: LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                     random_state=SEED, n_jobs=-1, verbose=-1),
        max_train_rows=None,
        lgbm=True,
    ),
    "MLP": dict(
        family="Neural network",
        config="hidden=(64, 32), ReLU, Adam lr=1e-3, batch=512, alpha=1e-4, <=40 epochs, early stop (patience 5)",
        loss="Binary cross-entropy (log loss) + L2",
        build=lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                                    batch_size=512, learning_rate_init=1e-3,
                                                                    max_iter=1, random_state=SEED)),
        max_train_rows=None,
        epochs=True,
    ),
}


def sample_points(df, max_rows: int | None, seed: int = SEED):
    """Subsample whole GPS points until about max_rows rows remain."""
    if max_rows is None or len(df) <= max_rows:
        return df
    pts = df[["trip", "point_idx"]].drop_duplicates()
    frac = max_rows / len(df)
    keep = pts.sample(frac=frac, random_state=seed)
    return df.merge(keep, on=["trip", "point_idx"])
