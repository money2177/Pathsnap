import { useEffect, useRef, useState } from 'react';
import maplibregl, { type GeoJSONSource, type Map as MapInstance } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { Crosshair, Minus, Plus, RotateCcw } from 'lucide-react';
import type { Layers, Prediction, Track } from '../types';
import { longitudeFirst, mapFeatures } from '../utils/geometry';

interface Props { track: Track | null; result: Prediction | null; layers: Layers }
export default function RouteMap({ track, result, layers }: Props) {
 const container = useRef<HTMLDivElement>(null);
 const map = useRef<MapInstance | null>(null);
 const [ready, setReady] = useState(false);
 const [failure, setFailure] = useState('');
 const [tilesUnavailable, setTilesUnavailable] = useState(false);
 const data = useRef({ track, result, layers }); data.current = { track, result, layers };
 function fit() {
  const current = map.current;
  const selected = data.current.track;
  if (!current) return;
  if (!selected?.gps_points.length && !selected?.bounds) {
   current.easeTo({ center: [24, 28], zoom: 2.3, bearing: 0, pitch: 0, duration: 500 }); return;
  }
  const bounds = new maplibregl.LngLatBounds();
  if (selected.bounds) selected.bounds.forEach(point => bounds.extend(longitudeFirst(point)));
  else selected.gps_points.forEach(point => bounds.extend(longitudeFirst(point)));
  // Include reconstructed detours while retaining the API's observation bounds.
  for (const line of data.current.result?.predicted_route ?? [])
   for (const point of line) bounds.extend(longitudeFirst(point));
  const compact = current.getContainer().clientWidth < 800;
  current.fitBounds(bounds, { padding: compact ? 55 : { top: 100, bottom: 100, left: 330, right: 295 },
   maxZoom: 16, duration: matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 600 });
 }
 useEffect(() => {
  if (!container.current) return;
  let current: MapInstance;
  try {
   current = new maplibregl.Map({
    container: container.current, center: [24, 28], zoom: 2.3, attributionControl: false,
    style: {
     version: 8,
     sources: { basemap: { type: 'raster', tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
      tileSize: 256, maxzoom: 19, attribution: '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors' } },
     layers: [{ id: 'background', type: 'background', paint: { 'background-color': '#e6ecec' } },
      { id: 'basemap', type: 'raster', source: 'basemap', paint: { 'raster-saturation': -0.65, 'raster-contrast': -0.1, 'raster-opacity': 0.72 } }],
    },
   });
  } catch { setFailure('Interactive maps require WebGL. Try opening this page in a desktop browser with hardware acceleration enabled.'); return; }
  map.current = current;
  current.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-right');
  current.addControl(new maplibregl.ScaleControl({ maxWidth: 90 }), 'bottom-right');
  current.on('error', event => {
   if ('sourceId' in event && event.sourceId === 'basemap') setTilesUnavailable(true);
  });
  current.on('load', () => {
   current.addSource('routes', { type: 'geojson', data: mapFeatures(data.current.track, data.current.result) });
   current.addLayer({ id: 'hmm-halo', type: 'line', source: 'routes', filter: ['==', ['get','layer'], 'hmm'],
    paint: { 'line-color': '#ffffff', 'line-width': 8, 'line-opacity': 0.95 }, layout: { 'line-cap': 'round', 'line-join': 'round' } });
   for (const [id, color, width] of [['classifier', '#a088b1', 3], ['hmm', '#35795f', 4.5]] as const) {
    current.addLayer({ id, type: 'line', source: 'routes', filter: ['==', ['get','layer'], id],
     paint: { 'line-color': color, 'line-width': width }, layout: { 'line-cap': 'round', 'line-join': 'round' } });
   }
   current.addLayer({ id: 'gps', type: 'circle', source: 'routes', filter: ['==', ['get','layer'], 'gps'],
    paint: { 'circle-color': '#4685a5', 'circle-radius': ['interpolate',['linear'],['zoom'],3,2,14,3.5],
     'circle-opacity': 0.8, 'circle-stroke-color': '#ffffff', 'circle-stroke-width': 0.7 } });
   current.addLayer({ id: 'terminals', type: 'circle', source: 'routes', filter: ['all', ['==',['get','layer'],'gps'], ['!=',['get','terminal'],'']],
    paint: { 'circle-radius': 6, 'circle-color': ['match',['get','terminal'],'start','#35795f','#b95f57'],
     'circle-stroke-color': '#ffffff', 'circle-stroke-width': 2 } });
   setReady(true); fit();
  });
  const observer = new ResizeObserver(() => { current.resize(); fit(); });
  observer.observe(container.current);
  return () => { observer.disconnect(); current.remove(); map.current = null; };
 }, []);
 useEffect(() => {
  const current = map.current;
  if (!ready || !current) return;
  (current.getSource('routes') as GeoJSONSource).setData(mapFeatures(track, result));
  fit();
 }, [track, result, ready]);
 useEffect(() => {
  if (!ready || !map.current) return;
  for (const [id, visible] of Object.entries(layers))
   map.current.setLayoutProperty(id, 'visibility', visible ? 'visible' : 'none');
  map.current.setLayoutProperty('hmm-halo', 'visibility', layers.hmm ? 'visible' : 'none');
 }, [layers, ready]);
 return <>
  <div className="map-surface" ref={container} aria-label="Interactive route map" />
  {failure && <div className="map-notice" role="alert">{failure}</div>}
  {tilesUnavailable && !failure && <div className="tile-notice" role="status">Basemap unavailable. Route layers remain available.</div>}
  <div className="map-tools panel" aria-label="Map controls">
   <button aria-label="Zoom in" title="Zoom in" disabled={!ready} onClick={() => map.current?.zoomIn()}><Plus size={18}/></button>
   <button aria-label="Zoom out" title="Zoom out" disabled={!ready} onClick={() => map.current?.zoomOut()}><Minus size={18}/></button>
   <span/>
   <button aria-label="Fit route" title="Fit route" disabled={!ready || !track} onClick={fit}><Crosshair size={18}/></button>
   <button aria-label="Reset overview" title="Reset overview" disabled={!ready} onClick={() => map.current?.easeTo({center:[24,28],zoom:2.3,pitch:0,bearing:0})}><RotateCcw size={17}/></button>
  </div>
 </>;
}
