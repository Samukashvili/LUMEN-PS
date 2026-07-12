// REST + WebSocket helpers for LUMEN-PS.
const J = (m, url, body) => fetch(url, {
  method: m,
  headers: body ? { 'content-type': 'application/json' } : undefined,
  body: body ? JSON.stringify(body) : undefined,
}).then(async r => {
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
});

export const api = {
  device: () => J('GET', '/api/device'),
  sessions: () => J('GET', '/api/sessions'),
  newSession: (name) => J('POST', '/api/sessions', { name }),
  session: (sid) => J('GET', `/api/sessions/${sid}`),
  getConfig: (sid) => J('GET', `/api/sessions/${sid}/config`),
  setConfig: (sid, overrides) => J('PUT', `/api/sessions/${sid}/config`, { overrides }),
  capture: (sid, role) => J('POST', `/api/sessions/${sid}/capture`, { role }),
  run: (sid) => J('POST', `/api/sessions/${sid}/run`),
  job: (sid) => J('GET', `/api/sessions/${sid}/job`),
  scanURL: (sid, role, max = 1000) => `/api/sessions/${sid}/scan/${role}?max=${max}&t=${Date.now()}`,
  resultURL: (sid, name, max = 0) => `/api/sessions/${sid}/result/${name}?max=${max}&t=${Date.now()}`,
};

// Stream a job's log. onLog(lines[]), onStatus({status,result,error}). Returns a closer.
export function streamJob(sid, onLog, onStatus) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/api/sessions/${sid}/stream`);
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.type === 'log') onLog(m.lines);
    else if (m.type === 'status') onStatus(m);
  };
  ws.onerror = () => {};
  return () => { try { ws.close(); } catch (_) {} };
}
