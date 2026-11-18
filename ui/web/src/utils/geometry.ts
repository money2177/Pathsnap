import type { FeatureCollection, Geometry } from 'geojson';
import type { Track, Prediction, LatLon } from '../types';
export const longitudeFirst = ([lat, lon]: LatLon): [number, number] => [lon, lat];
export function mapFeatures(track: Track | null, result: Prediction | null): FeatureCollection<Geometry> {
 const points = track?.gps_points ?? [];
 return { type: 'FeatureCollection', features: [
  ...points.map((point, index) => ({
   type: 'Feature' as const, geometry: { type: 'Point' as const, coordinates: longitudeFirst(point) },
   properties: { layer: 'gps', index, terminal: index === 0 ? 'start' : index === points.length - 1 ? 'end' : '' },
  })),
  ...(['hmm', 'classifier'] as const).flatMap(layer => {
   const lines = layer === 'hmm' ? result?.predicted_route : result?.classifier_only_route;
   return (lines ?? []).filter(line => line.length > 1).map(line => ({
    type: 'Feature' as const, properties: { layer },
    geometry: { type: 'LineString' as const, coordinates: line.map(longitudeFirst) },
   }));
  }),
 ] };
}
