import { useEffect, useRef, useState } from 'react';
import { APIClient, APIError, normalizeURL } from '../api/client';
import type { Health, Trip, Track, Prediction } from '../types';

const DEFAULT_URL = 'http://127.0.0.1:8000';
function savedURL() {
 try { return normalizeURL(localStorage.getItem('pathsnap.backend') || DEFAULT_URL); }
 catch { return DEFAULT_URL; }
}
export function useEvaluator() {
 const [base, setBase] = useState(savedURL);
 const [health, setHealth] = useState<Health | null>(null);
 const [trips, setTrips] = useState<Trip[]>([]);
 const [tripId, setTripId] = useState('');
 const [track, setTrack] = useState<Track | null>(null);
 const [result, setResult] = useState<Prediction | null>(null);
 const [connection, setConnection] = useState<'checking' | 'connected' | 'disconnected'>('checking');
 const [phase, setPhase] = useState<'idle' | 'loading' | 'running' | 'complete' | 'failed'>('idle');
 const [connectionError, setConnectionError] = useState('');
 const [error, setError] = useState('');
 const [revision, setRevision] = useState(0);
 const requests = useRef({ connection: new AbortController(), track: new AbortController(), prediction: new AbortController() });
 const running = useRef(false);

 function cancelTrip() {
  requests.current.track.abort(); requests.current.prediction.abort(); running.current = false;
  setTrack(null); setResult(null); setError(''); setPhase('idle');
 }
 function chooseTrip(id: string) { cancelTrip(); setTripId(id); }
 function refresh() {
  requests.current.connection.abort(); cancelTrip();
  setConnection('checking'); setRevision(value => value + 1);
 }
 function configure(value: string) {
  const url = normalizeURL(value);
  try { localStorage.setItem('pathsnap.backend', url); } catch { /* Session remains usable if storage is blocked. */ }
  if (url === base) { refresh(); return; }
  requests.current.connection.abort(); cancelTrip(); setTripId(''); setTrips([]);
  setHealth(null); setConnectionError(''); setConnection('checking'); setBase(url);
 }
 useEffect(() => {
  const controller = new AbortController(); requests.current.connection = controller;
  const api = new APIClient(base);
  setConnection('checking'); setConnectionError('');
  (async () => {
   try {
    const nextHealth = await api.health(controller.signal);
    const nextTrips = await api.trips(controller.signal);
    if (controller.signal.aborted) return;
    setHealth(nextHealth); setTrips(nextTrips); setConnection('connected');
    setTripId(current => nextTrips.some(trip => trip.trip_id === current) ? current : '');
   } catch (error) {
    if (controller.signal.aborted) return;
    setConnection('disconnected'); setConnectionError((error as Error).message);
    setHealth(null); setTrips([]); setTripId('');
   }
  })();
  return () => controller.abort();
 }, [base, revision]);
 useEffect(() => {
  if (!tripId || connection !== 'connected') return;
  const controller = new AbortController(); requests.current.track = controller;
  setPhase('loading'); setError('');
  new APIClient(base).track(tripId, controller.signal).then(data => {
   if (!controller.signal.aborted) { setTrack(data); setPhase('idle'); }
  }).catch(error => {
   if (!controller.signal.aborted) { setError(error.message); setPhase('failed'); }
  });
  return () => controller.abort();
 }, [tripId, base, revision, connection]);
 useEffect(() => () => {
  requests.current.connection.abort(); requests.current.track.abort(); requests.current.prediction.abort();
 }, []);
 async function run() {
  if (!tripId || connection !== 'connected' || running.current) return;
  running.current = true; requests.current.track.abort();
  const controller = new AbortController(); requests.current.prediction = controller;
  setResult(null); setError(''); setPhase('running');
  try {
   const data = await new APIClient(base).predict(tripId, controller.signal);
   if (!controller.signal.aborted) { setResult(data); setTrack(data); setPhase('complete'); }
  } catch (error) {
   if (!controller.signal.aborted) {
    setError((error as Error).message); setPhase('failed');
    if (error instanceof APIError && error.connectivity) {
     setConnection('disconnected'); setConnectionError(error.message);
    }
   }
  } finally { if (!controller.signal.aborted) running.current = false; }
 }
 return { base, health, trips, tripId, track, result, connection, connectionError, phase, error, chooseTrip, configure, refresh, run };
}
