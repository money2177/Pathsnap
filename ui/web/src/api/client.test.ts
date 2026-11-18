import { afterEach, describe, expect, it, vi } from 'vitest';
import { APIClient, normalizeURL, validatePrediction } from './client';
import { mapFeatures } from '../utils/geometry';
import type { Prediction } from '../types';

const prediction: Prediction = {
 request_id:'test-request', trip_id:'00000052', n_points:2,
 gps_points:[[12,77],[12.1,77.1]], bounds:[[12,77],[12.1,77.1]],
 predicted_route:[[[12,77],[12.05,77.05]],[[12.08,77.08],[12.1,77.1]]],
 classifier_only_route:[], metrics:null,
};
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });
describe('API contract', () => {
 it('preserves trip IDs and makes a new POST for every prediction', async () => {
  const fetch = vi.fn().mockResolvedValue({ ok:true, json:async () => prediction });
  vi.stubGlobal('fetch',fetch);
  const client = new APIClient('http://localhost:8000/');
  await client.predict('00000052',new AbortController().signal);
  await client.predict('00000052',new AbortController().signal);
  expect(fetch).toHaveBeenCalledTimes(2);
  expect(fetch.mock.calls[0][0]).toBe('http://localhost:8000/predict');
  expect(fetch.mock.calls[0][1]).toMatchObject({method:'POST',cache:'no-store',body:'{"trip_id":"00000052"}'});
 });
 it('accepts unscored and empty geometry responses', () => {
  expect(validatePrediction(prediction,'00000052').metrics).toBeNull();
  expect(validatePrediction({...prediction,gps_points:[],predicted_route:[]},'00000052').gps_points).toEqual([]);
 });
 it('rejects mismatched trips and malformed coordinates', () => {
  expect(() => validatePrediction(prediction,'00000053')).toThrow();
  expect(() => validatePrediction({...prediction,gps_points:[[95,77]]},'00000052')).toThrow();
  expect(() => validatePrediction({...prediction,predicted_route:[[[12,181]]]},'00000052')).toThrow();
 });
 it.each([404,422,500])('preserves HTTP %i error details', async status => {
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:false,status,json:async()=>({detail:'Server explanation'})}));
  await expect(new APIClient('http://localhost:8000').predict('00000052',new AbortController().signal)).rejects.toThrow('Server explanation');
 });
 it('reports network failures', async () => {
  vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
  await expect(new APIClient('http://localhost:8000').health(new AbortController().signal)).rejects.toThrow('Could not reach Laptop 1');
 });
 it('times out predictions at 90 seconds', async () => {
  vi.useFakeTimers();
  vi.stubGlobal('fetch', vi.fn((_url, options) => new Promise((_resolve,reject) => {
   options.signal.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError')));
  })));
  const result = new APIClient('http://localhost:8000').predict('00000052',new AbortController().signal);
  const assertion = expect(result).rejects.toThrow('timed out');
  await vi.advanceTimersByTimeAsync(90000);
  await assertion;
 });
 it('accepts a LAN URL and rejects credentials or endpoint paths', () => {
  expect(normalizeURL(' http://192.168.1.5:8000/ ')).toBe('http://192.168.1.5:8000');
  expect(() => normalizeURL('http://user:secret@localhost:8000')).toThrow();
  expect(() => normalizeURL('http://localhost:8000/health')).toThrow();
 });
});
describe('Map geometry', () => {
 it('swaps coordinate order and preserves independent route segments', () => {
  const data = mapFeatures(prediction,prediction);
  const lines = data.features.filter(feature=>feature.geometry.type==='LineString');
  expect(lines).toHaveLength(2);
  expect(lines[0].geometry).toEqual({type:'LineString',coordinates:[[77,12],[77.05,12.05]]});
  expect(data.features[0].properties?.terminal).toBe('start');
  expect(data.features[1].properties?.terminal).toBe('end');
 });
});
