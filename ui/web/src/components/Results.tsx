import { formatTripDuration } from '../utils/time';
import { ChevronDown, Check, Activity } from 'lucide-react';
import type { Prediction } from '../types';
const percent = (value?: number) => value === undefined ? '—' : (value * 100).toFixed(1) + '%';
const metricLabels = [
 ['top1_accuracy_classifier_only','Classifier-only Top-1'],['route_continuity','Route continuity'],
 ['route_precision','Route precision'],['route_recall','Route recall'],['route_f1','Route F1'],
] as const;
export default function Results({ result }: { result: Prediction }) {
 const metrics = result.metrics;
 return <section className="result-card panel" aria-label="Evaluation results">
  <div className="card-heading"><span className="eyebrow">EVALUATION RESULTS</span><span className="success-tick"><Check size={14}/></span></div>
  {metrics?.top1_accuracy !== undefined ? <>
   <div className="accuracy">{percent(metrics.top1_accuracy)}</div>
   <p className="metric-label">Point accuracy · HMM Top-1</p>
   {metrics.hmm_improvement_pp !== undefined && <div className="improvement">{metrics.hmm_improvement_pp >= 0 ? '+' : ''}{metrics.hmm_improvement_pp.toFixed(2)} pp <span>from HMM decoding</span></div>}
  </> : <p className="unscored">{metrics === null ? 'Prediction complete. No reference available on the server, so this trip is unscored.' : 'The server did not return evaluation metrics.'}</p>}
  <div className="result-stats">
   {result.route_length_m !== undefined && <div><strong>{(result.route_length_m / 1000).toFixed(2)} <small>km</small></strong><span>Route distance</span></div>}
   {result.timings?.total_ms !== undefined && <div><strong>{result.timings.total_ms.toFixed(0)} <small>ms</small></strong><span>Backend processing time</span></div>}
  </div>
  <details className="result-details"><summary>Detailed results <ChevronDown size={15}/></summary>
   <dl>
    <div><dt>GPS observations</dt><dd>{result.n_points.toLocaleString()}</dd></div>
    {result.duration_s !== undefined && <div><dt>Recorded trip duration</dt><dd>{formatTripDuration(result.duration_s)}</dd></div>}
    {result.n_route_segments !== undefined && <div><dt>Route segments</dt><dd>{result.n_route_segments}</dd></div>}
    {metricLabels.map(([key,label]) => metrics?.[key] !== undefined && <div key={key}><dt>{label}</dt><dd>{percent(metrics[key])}</dd></div>)}
   </dl>
   {result.timings && <><h3>Processing breakdown</h3><dl>{([
    ['features_ms','Feature generation'],['classifier_ms','Classifier inference'],['hmm_ms','HMM/Viterbi'],['scoring_ms','Scoring'],
   ] as const).map(([key,label]) => result.timings?.[key] !== undefined && <div key={key}><dt>{label}</dt><dd>{result.timings[key]!.toFixed(1)} ms</dd></div>)}</dl></>}
   {result.decoder && <><h3><Activity size={13}/> Decoder diagnostics</h3><dl>{([
    ['gap_splits','Gap splits'],['break_steps','Break steps'],['invalid_transitions_used','Invalid transitions'],
    ['unjoinable_transitions','Unjoinable transitions'],['network_connectivity','Network connectivity'],
   ] as const).map(([key,label]) => result.decoder?.[key] !== undefined && <div key={key}><dt>{label}</dt><dd>{key === 'network_connectivity' ? percent(result.decoder[key]) : result.decoder[key]}</dd></div>)}</dl></>}
   <div className="request-id"><span>Request ID</span><code>{result.request_id}</code><p>Matches the prediction request in the backend logs on Laptop 1.</p></div>
  </details>
 </section>;
}
