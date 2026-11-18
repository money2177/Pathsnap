export type LatLon = [number, number];
export interface Health { status: string; model: string; eval_set: string; n_trips: number; scoring_available?: boolean }
export interface Trip { trip_id: string; n_points: number; duration_s?: number }
export interface Track { trip_id: string; n_points: number; gps_points: LatLon[]; bounds?: [LatLon, LatLon]; t?: number[] }
export interface Metrics { top1_accuracy?: number; top1_accuracy_classifier_only?: number; hmm_improvement_pp?: number; route_continuity?: number; route_precision?: number; route_recall?: number; route_f1?: number }
export interface Timings { features_ms?: number; classifier_ms?: number; hmm_ms?: number; scoring_ms?: number; total_ms?: number }
export interface Prediction extends Track {
 request_id: string; eval_set?: string; duration_s?: number; n_route_segments?: number;
 route_length_m?: number; n_candidate_rows?: number; predicted_route: LatLon[][];
 classifier_only_route?: LatLon[][]; metrics?: Metrics | null; timings?: Timings;
 decoder?: { gap_splits?: number; break_steps?: number; invalid_transitions_used?: number; unjoinable_transitions?: number; network_connectivity?: number };
}
export type LayerName = 'gps' | 'hmm' | 'classifier';
export type Layers = Record<LayerName, boolean>;
