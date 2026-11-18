export function formatTripDuration(seconds: number): string {
 const rounded = Math.round(seconds);
 const hours = Math.floor(rounded / 3600);
 const minutes = Math.floor((rounded % 3600) / 60);
 const remainder = rounded % 60;
 return [hours ? hours + ' h' : '', minutes || hours ? minutes + ' min' : '', remainder + ' s']
  .filter(Boolean).join(' ');
}
