import { useEffect, useRef, useState } from 'react';
import { X, ArrowUpRight, LoaderCircle } from 'lucide-react';
interface Props { base: string; error: string; checking: boolean; close: () => void; save: (url: string) => void }
export default function Settings({ base, error, checking, close, save }: Props) {
 const dialog = useRef<HTMLDialogElement>(null);
 const [value, setValue] = useState(base);
 const [invalid, setInvalid] = useState('');
 useEffect(() => { dialog.current?.showModal(); return () => dialog.current?.close(); }, []);
 return <dialog ref={dialog} className="settings-dialog" onCancel={close} onClick={event => { if (event.target === event.currentTarget) close(); }}>
  <div className="dialog-heading"><div><span className="eyebrow">WORKSPACE SETTINGS</span><h2>Backend connection</h2></div><button className="icon-button" aria-label="Close settings" onClick={close}><X size={20}/></button></div>
  <p className="muted">Connect to the FastAPI server running on Laptop 1.</p>
  <form onSubmit={event => { event.preventDefault(); try { save(value); setInvalid(''); } catch (e) { setInvalid((e as Error).message); } }}>
   <label htmlFor="api-url">API base URL</label>
   <input id="api-url" autoFocus type="text" value={value} onChange={e => setValue(e.target.value)} placeholder="http://127.0.0.1:8000" autoComplete="url"/>
   <p className="hint">Use Laptop 1’s LAN address when the devices are separate. The loopback address connects to this computer.</p>
   {(invalid || error) && <div className="inline-error" role="alert">{invalid || error}</div>}
   <button className="primary" disabled={checking}>{checking ? <LoaderCircle className="spin" size={16}/> : <ArrowUpRight size={16}/>} {checking ? 'Checking connection' : 'Save & check connection'}</button>
  </form>
  <details><summary>Connection troubleshooting</summary><p>Start FastAPI on Laptop 1, connect both laptops to the same network, and allow the API port through its firewall. Verify the server’s /health endpoint from this computer.</p><p>An HTTPS frontend requires an HTTPS API or an approved proxy. For a local evaluation, serve both over HTTP.</p></details>
 </dialog>;
}
