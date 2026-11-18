import type { Health, Trip, Track, Prediction } from '../types';

export class APIError extends Error {
 constructor(message: string, public connectivity = false) { super(message); }
}
export function normalizeURL(input: string): string {
 try {
  const url = new URL(input.trim());
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password ||
      url.search || url.hash || url.pathname !== '/') throw new Error();
  return url.origin;
 } catch { throw new APIError('Enter an HTTP(S) base URL with a host and optional port.'); }
}
function object(value: unknown): asserts value is Record<string, any> {
 if (!value || typeof value !== 'object' || Array.isArray(value))
  throw new APIError('The backend returned an unexpected response.');
}
function coordinate(value: unknown): boolean {
 return Array.isArray(value) && value.length === 2 &&
  value.every(n => typeof n === 'number' && Number.isFinite(n)) &&
  Math.abs(value[0]) <= 90 && Math.abs(value[1]) <= 180;
}
function numbers(data: Record<string, any>, keys: string[]) {
 for (const key of keys) {
  if (data[key] !== undefined && (typeof data[key] !== 'number' || !Number.isFinite(data[key])))
   throw new APIError('The backend returned an invalid numeric field: ' + key);
 }
}
export function validateTrack(data: unknown, id: string): Track {
 object(data);
 if (data.trip_id !== id || !Number.isInteger(data.n_points) || data.n_points < 0 ||
     !Array.isArray(data.gps_points) || !data.gps_points.every(coordinate))
  throw new APIError('The backend returned invalid trip or GPS data.');
 if (data.bounds !== undefined && (!Array.isArray(data.bounds) || data.bounds.length !== 2 ||
     !data.bounds.every(coordinate))) throw new APIError('The backend returned invalid map bounds.');
 return data as unknown as Track;
}
export function validatePrediction(data: unknown, id: string): Prediction {
 validateTrack(data, id); object(data);
 if (typeof data.request_id !== 'string' || !data.request_id || !Array.isArray(data.predicted_route))
  throw new APIError('The backend returned an incomplete prediction.');
 for (const key of ['predicted_route', 'classifier_only_route']) {
  if (data[key] !== undefined && (!Array.isArray(data[key]) ||
      !data[key].every((line: unknown) => Array.isArray(line) && line.every(coordinate))))
   throw new APIError('The backend returned invalid route coordinates.');
 }
 numbers(data, ['duration_s', 'n_route_segments', 'route_length_m', 'n_candidate_rows']);
 const fields: Record<string, string[]> = {
  metrics: ['top1_accuracy','top1_accuracy_classifier_only','hmm_improvement_pp','route_continuity','route_precision','route_recall','route_f1'],
  timings: ['features_ms','classifier_ms','hmm_ms','scoring_ms','total_ms'],
  decoder: ['gap_splits','break_steps','invalid_transitions_used','unjoinable_transitions','network_connectivity'],
 };
 for (const [key, keys] of Object.entries(fields)) {
  if (data[key] !== undefined && data[key] !== null) { object(data[key]); numbers(data[key], keys); }
 }
 return data as unknown as Prediction;
}
export class APIClient {
 base: string;
 constructor(base: string) { this.base = normalizeURL(base); }
 async request(path: string, signal: AbortSignal, body?: unknown, timeout = 10000): Promise<unknown> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal.addEventListener('abort', abort, { once: true });
  if (signal.aborted) controller.abort();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeout);
  try {
   const response = await fetch(this.base + path, {
    method: body === undefined ? 'GET' : 'POST', signal: controller.signal, cache: 'no-store',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
   });
   let data: any;
   try { data = await response.json(); } catch {
    throw new APIError('HTTP ' + response.status + ' · Backend returned invalid JSON.');
   }
   if (!response.ok) {
    const detail = typeof data?.detail === 'string' ? data.detail :
     Array.isArray(data?.detail) ? data.detail.map((x: any) => x.msg || 'Invalid request').join('; ') :
     'Check the Laptop 1 server logs.';
    throw new APIError('HTTP ' + response.status + ' · ' + detail);
   }
   return data;
  } catch (error) {
   if (signal.aborted) throw new DOMException('Cancelled', 'AbortError');
   if (timedOut) throw new APIError('The request timed out. Check the Laptop 1 logs and retry.', true);
   if (error instanceof APIError) throw error;
   throw new APIError('Could not reach Laptop 1. Check its address, server, network and firewall.', true);
  } finally { clearTimeout(timer); signal.removeEventListener('abort', abort); }
 }
 async health(signal: AbortSignal): Promise<Health> {
  const data = await this.request('/health', signal); object(data);
  if (data.status !== 'ok' || typeof data.model !== 'string' || typeof data.eval_set !== 'string' ||
      !Number.isInteger(data.n_trips)) throw new APIError('The backend is not ready or returned invalid health data.');
  return data as unknown as Health;
 }
 async trips(signal: AbortSignal): Promise<Trip[]> {
  const data = await this.request('/trips', signal); object(data);
  if (!Array.isArray(data.trips)) throw new APIError('The backend returned an invalid trip list.');
  for (const trip of data.trips) {
   object(trip);
   if (typeof trip.trip_id !== 'string' || !trip.trip_id || !Number.isInteger(trip.n_points))
    throw new APIError('The backend returned an invalid trip.');
   numbers(trip, ['duration_s']);
  }
  if (new Set(data.trips.map((t: Trip) => t.trip_id)).size !== data.trips.length)
   throw new APIError('The backend returned duplicate trip IDs.');
  return data.trips;
 }
 async track(id: string, signal: AbortSignal): Promise<Track> {
  return validateTrack(await this.request('/trips/' + encodeURIComponent(id) + '/track', signal), id);
 }
 async predict(id: string, signal: AbortSignal): Promise<Prediction> {
  return validatePrediction(await this.request('/predict', signal, { trip_id: id }, 90000), id);
 }
}
