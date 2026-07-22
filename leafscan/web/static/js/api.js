// REST + WebSocket helpers for LUMEN-PS.
const J = (m, url, body) => fetch(url, {
  method: m,
  headers: body ? { 'content-type': 'application/json' } : undefined,
  body: body ? JSON.stringify(body) : undefined,
}).then(async r => {
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
});

const upload = (url, file) => {
  const body = new FormData();
  body.append('file', file, file.name);
  return fetch(url, { method: 'POST', body }).then(async r => {
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    return r.json();
  });
};

export const api = {
  version: () => J('GET', '/api/version'),
  device: () => J('GET', '/api/device'),
  sessions: () => J('GET', '/api/sessions'),
  newSession: (name) => J('POST', '/api/sessions', { name }),
  session: (sid) => J('GET', `/api/sessions/${sid}`),
  getConfig: (sid) => J('GET', `/api/sessions/${sid}/config`),
  setConfig: (sid, overrides) => J('PUT', `/api/sessions/${sid}/config`, { overrides }),
  setOutputDir: (sid, path) => J('PUT', `/api/sessions/${sid}/output-dir`, { path }),
  chooseOutputDir: () => J('POST', '/api/choose-output-dir'),
  deleteSession: (sid, deleteFiles) => J('DELETE', `/api/sessions/${sid}`, { delete_files: deleteFiles }),
  capture: (sid, role) => J('POST', `/api/sessions/${sid}/capture`, { role }),
  importScan: (sid, role, file) => upload(`/api/sessions/${sid}/import/${role}`, file),
  removeImportedScan: (sid, role) => J('DELETE', `/api/sessions/${sid}/import/${role}`),
  resetScans: (sid) => J('POST', `/api/sessions/${sid}/reset-scans`),
  run: (sid) => J('POST', `/api/sessions/${sid}/run`),
  job: (sid) => J('GET', `/api/sessions/${sid}/job`),
  cancelJob: (sid) => J('POST', `/api/sessions/${sid}/job/cancel`),
  shutdown: () => J('POST', '/api/shutdown'),
  results: (sid) => J('GET', `/api/sessions/${sid}/results`),
  scanURL: (sid, role, max = 1000) => `/api/sessions/${sid}/scan/${role}?max=${max}&t=${Date.now()}`,
  resultURL: (sid, name, max = 0) => `/api/sessions/${sid}/result/${name}?max=${max}&t=${Date.now()}`,
  resultRawURL: (sid, name, download = false) =>
    `/api/sessions/${sid}/result/${name}?${download ? 'download=1' : 'raw=1'}`,
};

// Stream a job's log. onLog(lines[]), onStatus({status,result,error}). Returns a closer.
//
// The job keeps running server-side even if this socket drops, and the server
// tails the log by index, so a dropped connection reconnects from the last line
// already received (with backoff) instead of freezing the log + progress. Retrying
// stops once a terminal status arrives or the caller closes the stream.
export function streamJob(sid, onLog, onStatus, from = 0) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  let cursor = from;     // log index reached so far — the resume point after a drop
  let ws = null;
  let retry = 0;
  let timer = null;
  let stopped = false;   // terminal status received, or caller closed

  function connect() {
    ws = new WebSocket(`${proto}://${location.host}/api/sessions/${sid}/stream?from=${cursor}`);
    ws.onmessage = (e) => {
      let m;
      try { m = JSON.parse(e.data); } catch { return; }
      retry = 0;   // valid traffic proves the reconnected stream is healthy
      if (m.type === 'log') { cursor += m.lines.length; onLog(m.lines); }
      else if (m.type === 'status') { stopped = true; onStatus(m); }
    };
    ws.onerror = () => {};   // an error surfaces as a close; reconnect is handled there
    ws.onclose = () => {
      if (stopped) return;   // job finished or caller closed — expected, don't retry
      timer = setTimeout(connect, Math.min(500 * 2 ** retry, 8000));
      retry += 1;
    };
  }
  connect();

  return () => {
    stopped = true;
    if (timer) clearTimeout(timer);
    try { ws && ws.close(); } catch (_) {}
  };
}
