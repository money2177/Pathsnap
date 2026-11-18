import { useState } from 'react';
import { ArrowUpRight, Check, ChevronDown, Layers3, LoaderCircle, LocateFixed, MapPin, Navigation, RefreshCw, Route, Settings2, WifiOff } from 'lucide-react';
import { useEvaluator } from './hooks/useEvaluator';
import RouteMap from './components/RouteMap';
import Results from './components/Results';
import Settings from './components/Settings';
import type { Layers, LayerName } from './types';

export default function App() {
 const evaluator = useEvaluator();
 const [settings, setSettings] = useState(false);
 const [layers, setLayers] = useState<Layers>({gps:true,hmm:true,classifier:false});
 const trip = evaluator.trips.find(trip => trip.trip_id === evaluator.tripId);
 const busy = evaluator.phase === 'running';
 const loading = evaluator.phase === 'loading';
 const connected = evaluator.connection === 'connected';
 function toggle(layer: LayerName) { setLayers(current => ({...current,[layer]:!current[layer]})); }
 return <main className="workspace">
  <header className="topbar">
   <a className="brand" href="#main-map" aria-label="PathSnap route workspace"><span className="brand-mark"><Route size={23} strokeWidth={1.8}/></span><span><strong>PathSnap<span className="brand-period">.</span></strong><small>Route Intelligence</small></span></a>
   <div className="workspace-label"><span/> Evaluation workspace</div>
   <div className="header-actions">
    <button className={'connection ' + evaluator.connection} onClick={() => setSettings(true)} title={evaluator.connectionError || evaluator.base}>
     <i/>{evaluator.connection === 'checking' ? 'Checking connection' : connected ? 'Backend connected' : 'Backend unavailable'}
    </button>
    <span className="header-divider"/>
    <button className="icon-button" aria-label="Open settings" title="Settings" onClick={() => setSettings(true)}><Settings2 size={19}/></button>
   </div>
  </header>
  <section className="map-workspace" id="main-map" aria-label="Route evaluation workspace">
   <RouteMap track={evaluator.track} result={evaluator.result} layers={layers}/>
   <div className="left-stack">
    <section className="trip-card panel">
     <div className="card-heading"><span className="eyebrow">ROUTE EXPLORER</span><span className="subtle-icon"><Navigation size={16}/></span></div>
     <h1>Route evaluation</h1>
     <p className="intro">Select a trip, then run a live reconstruction.</p>
     <label htmlFor="trip">Evaluation trip</label>
     <div className="select-row"><div className="select-wrap">
      <MapPin size={16}/><select id="trip" value={evaluator.tripId} onChange={e => evaluator.chooseTrip(e.target.value)} disabled={!connected || !evaluator.trips.length}>
       <option value="">Select a trip</option>{evaluator.trips.map(trip => <option key={trip.trip_id} value={trip.trip_id}>Trip {trip.trip_id}</option>)}
      </select><ChevronDown size={14}/></div>
      <button className="icon-button refresh" aria-label="Refresh trip list" title="Refresh trip list" onClick={evaluator.refresh} disabled={busy || evaluator.connection === 'checking'}><RefreshCw size={16} className={evaluator.connection === 'checking' ? 'spin' : ''}/></button>
     </div>
     {trip && <div className="trip-facts"><div><strong>{trip.n_points.toLocaleString()}</strong><span>GPS points</span></div>{trip.duration_s !== undefined && <div><strong>{trip.duration_s.toLocaleString(undefined,{maximumFractionDigits:1})}<small> s</small></strong><span>Trip duration</span></div>}</div>}
     {connected && !evaluator.trips.length && <p className="hint">No evaluation trips found. Check the folder served by Laptop 1, then refresh.</p>}
     <button className="primary run-button" onClick={evaluator.run} disabled={!connected || !trip || busy || loading}>
      {busy ? <LoaderCircle size={17} className="spin"/> : <Route size={17}/>}
      {busy ? 'Reconstructing route' : 'Run map-matching'}
      {!busy && <ArrowUpRight size={16}/>}
     </button>
     <div className="live-caption"><span/> Live predictions from Laptop 1</div>
    </section>
    {!connected && evaluator.connection !== 'checking' && <section className="connection-card panel">
     <div className="card-heading"><WifiOff size={16}/><strong>Backend unavailable</strong></div>
     <p>Connect to Laptop 1 to load evaluation trips.</p>
     <button className="text-button" onClick={() => setSettings(true)}>Connection settings <ArrowUpRight size={14}/></button>
     <details><summary>View error details</summary><p>{evaluator.connectionError}</p></details>
    </section>}
    {connected && evaluator.health && <div className="model-caption"><span className="status-dot"/><span>{evaluator.health.eval_set}<small>{evaluator.health.model}</small></span></div>}
   </div>
   {!evaluator.track && !busy && !loading && <div className="map-empty"><span className="empty-icon"><LocateFixed size={23}/></span><strong>Select a trip to explore its route.</strong><span>GPS observations appear after selection.</span></div>}
   <div className="right-stack">
    {evaluator.result && <Results result={evaluator.result}/>}
    {(busy || loading) && <div className="progress-card panel" role="status"><LoaderCircle className="spin" size={19}/><div><strong>{busy ? 'Reconstructing your route' : 'Loading GPS observations'}</strong><p>{busy ? 'Running the classifier and HMM on Laptop 1.' : 'Retrieving the selected trip.'}</p></div></div>}
    {evaluator.error && <div className="error-card panel" role="alert"><strong>Request unsuccessful</strong><p>{evaluator.error}</p><span>Check the connection or retry the request.</span></div>}
    {evaluator.result && !evaluator.result.predicted_route.some(line => line.length > 1) && <div className="error-card panel">The backend returned no reconstructed route geometry.</div>}
    {evaluator.track && !evaluator.track.gps_points.length && <div className="error-card panel">The backend returned no GPS observations.</div>}
   </div>
   <section className="layer-panel panel" aria-label="Map layers">
    <div className="layer-heading"><Layers3 size={16}/><span>Map layers</span></div>
    {([['gps','GPS observations'],['hmm','Matched route'],['classifier','Classifier only']] as const).map(([key,label]) =>
     <button key={key} className={'layer-button ' + (layers[key] ? 'active' : '')} onClick={() => toggle(key)} aria-pressed={layers[key]}><span className={'layer-swatch '+key}/>{label}<span className="layer-check">{layers[key] && <Check size={12}/>}</span></button>
    )}
    <div className="marker-legend"><span><i className="start"/> Start</span><span><i className="end"/> End</span></div>
   </section>
   <div className="map-caption"><span className="status-dot"/>{evaluator.result ? 'Live prediction · Trip ' + evaluator.result.trip_id : evaluator.track ? 'Input GPS · Trip ' + evaluator.track.trip_id : 'Geographic overview'}</div>
  </section>
  <footer className="footer"><span>PathSnap <span className="footer-separator">/</span> Route evaluation</span><span>{evaluator.result ? 'Request '+ evaluator.result.request_id : 'GPS → Classifier → HMM / Viterbi'}</span></footer>
  {settings && <Settings base={evaluator.base} error={evaluator.connectionError} checking={evaluator.connection === 'checking'} close={() => setSettings(false)} save={evaluator.configure}/>}
 </main>;
}
