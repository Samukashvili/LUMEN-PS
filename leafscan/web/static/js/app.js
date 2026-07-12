// LUMEN-PS — front controller: boot, stages, telemetry, live log.
import { api, streamJob } from './api.js';
import { Relight } from './relight.js';

const $ = (s, r = document) => r.querySelector(s);
const LEAF = ['k0', 'k1', 'k2', 'k3'];
const STAGES = ['capture', 'configure', 'process', 'results'];
const STAGE_MAP = { '[crop]': 0, '[load]': 0, '[rigid]': 1, '[nonrigid]': 1, '[valid]': 1,
  '[calib]': 2, '[solve]': 3, '[integrate]': 4, '[out]': 5, '[qa]': 5, '[done]': 6 };
const PSTAGES = ['Load', 'Align', 'Calibrate', 'Solve', 'Integrate', 'Output'];

const S = { device: null, sid: null, meta: null, cfg: null, overrides: {},
  stage: 'capture', relight: null, closeWS: null, log: [] };

// ---- tiny helpers ---------------------------------------------------------
const getPath = (o, p) => p.split('.').reduce((a, k) => (a ? a[k] : undefined), o);
function setPath(o, p, v) {
  const ks = p.split('.'); let a = o;
  ks.slice(0, -1).forEach(k => (a = a[k] = a[k] || {}));
  a[ks.at(-1)] = v;
}
const esc = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

// ---- boot -----------------------------------------------------------------
async function boot() {
  $('#log-toggle').addEventListener('click', toggleLog);
  $('#btn-new').addEventListener('click', openNewSession);
  $('#new-session-cancel').addEventListener('click', closeNewSession);
  $('#new-session-form').addEventListener('submit', newSession);
  document.querySelectorAll('.stage').forEach(b =>
    b.addEventListener('click', () => gotoStage(b.dataset.stage)));

  try {
    S.device = await api.device();
  } catch { S.device = { connected: false, note: 'Bench API unreachable.' }; }
  paintDevice();
  await paintRecent();
}

function paintDevice() {
  const d = S.device, pill = $('#device-pill'), bd = $('#boot-device');
  pill.className = 'pill ' + (d.connected ? 'pill--ok' : 'pill--err');
  $('.pill-text', pill).textContent = d.connected
    ? `${d.name} · ${d.max_bit_depth}-bit` : 'no scanner';
  bd.querySelector('.led').style.color = d.connected ? 'var(--ok)' : 'var(--err)';
  $('.boot-device-text', bd).innerHTML = d.connected
    ? `bench online · <b>${esc(d.name)}</b> · ${d.dpi_options.join('/')} dpi · ${d.max_bit_depth}-bit`
    : esc(d.note || 'no scanner detected');
}

async function paintRecent() {
  const list = await api.sessions();
  const ul = $('#recent-list');
  if (!list.length) { ul.innerHTML = '<li class="muted">none yet</li>'; return; }
  ul.innerHTML = '';
  list.slice(0, 8).forEach(m => {
    const li = document.createElement('li');
    li.className = 'recent-item';
    const tag = m.status === 'done' ? 'tag--done' : m.status === 'ready' ? 'tag--ready' : '';
    li.innerHTML = `<span class="r-name">${esc(m.name)}</span>
      <span class="r-meta">${m.created.slice(0, 16).replace('T', ' ')}
      <span class="tag ${tag}">${m.status}</span></span>`;
    li.onclick = () => enterSession(m.id);
    ul.appendChild(li);
  });
}

function openNewSession() {
  const dialog = $('#new-session-dialog');
  dialog.hidden = false;
  const input = $('#new-session-name');
  input.focus(); input.select();
}
function closeNewSession() { $('#new-session-dialog').hidden = true; }
async function newSession(e) {
  e.preventDefault();
  const input = $('#new-session-name');
  const submit = $('#new-session-form button[type="submit"]');
  submit.disabled = true;
  try {
    const m = await api.newSession(input.value.trim() || 'Untitled leaf');
    closeNewSession();
    await enterSession(m.id);
  } catch (err) {
    logLine('[error] could not create session: ' + err.message);
  } finally {
    submit.disabled = false;
  }
}

async function enterSession(sid) {
  S.sid = sid;
  await refreshMeta();
  const bundle = await api.getConfig(sid);
  S.cfg = bundle.config; S.overrides = bundle.overrides || {};
  $('#boot').hidden = true; $('#shell').hidden = false;
  const start = S.meta.status === 'done' ? 'results'
    : S.meta.status === 'ready' ? 'configure' : 'capture';
  gotoStage(start);
}

async function refreshMeta() { S.meta = await api.session(S.sid); paintTelemetry(); }

// ---- stage routing --------------------------------------------------------
function gotoStage(name) {
  S.stage = name;
  document.querySelectorAll('.stage').forEach(b => {
    const i = STAGES.indexOf(b.dataset.stage), cur = STAGES.indexOf(name);
    b.classList.toggle('active', b.dataset.stage === name);
    b.classList.toggle('done', doneStage(b.dataset.stage));
    b.disabled = (b.dataset.stage === 'results' && S.meta.status !== 'done')
      || ((b.dataset.stage === 'configure' || b.dataset.stage === 'process') && !S.meta.ready);
  });
  ({ capture: renderCapture, configure: renderConfigure,
     process: renderProcess, results: renderResults }[name])();
}
function doneStage(name) {
  if (name === 'capture') return S.meta.ready;
  if (name === 'results') return S.meta.status === 'done';
  if (name === 'process') return S.meta.status === 'done';
  return false;
}

// ---- CAPTURE --------------------------------------------------------------
function renderCapture() {
  const m = S.meta, c = S.cfg, next = LEAF.find(r => !m.scans[r]);
  const dpiOpts = S.device?.dpi_options?.length ? S.device.dpi_options : [300, 600, 1200];
  const main = $('#stage-main');
  main.innerHTML = `
    <h2 class="stage-title">Capture</h2>
    <p class="stage-lead">Four scans of the same leaf, rotated a quarter turn between
      each. The lamp is fixed, so rotating the leaf changes the light direction —
      four rotations give four lights.</p>
    <div class="capture-settings">
      <label for="capture-dpi">Scan resolution</label>
      <select id="capture-dpi" aria-describedby="capture-dpi-note">
        ${dpiOpts.map(dpi => `<option value="${dpi}" ${dpi === c.capture.dpi ? 'selected' : ''}>${dpi} dpi</option>`).join('')}
      </select>
      <span id="capture-dpi-note">Applied to every scan in this session.</span>
      <span id="capture-dpi-status" class="mono muted" aria-live="polite"></span>
    </div>
    <div class="instr">
      <div>
        <div class="eyebrow">Protocol</div>
        <p style="margin:.4em 0 0">Place the leaf on the glass, same face up, and close the lid.
        After each scan, rotate it <span class="kbd">90° clockwise</span> about its centre and scan again.</p>
      </div>
    </div>
    <div class="scan-grid" id="slots">${LEAF.map(slot).join('')}</div>
    <details class="expander"><summary>Optional scans (advanced)</summary>
      <p class="stage-lead" style="margin-top:8px">Not required — elevation self-calibrates
      and the flat-field is fit from the background. Add these only if you want them.</p>
      <div class="scan-grid">${['flat', 'calib0', 'calib90'].map(slot).join('')}</div>
    </details>
    <div class="stage-foot">
      <button class="btn btn--primary" id="scan-btn">${next ? 'Scan ' + next : 'All scans captured'}</button>
      <button class="btn" id="to-config" ${m.ready ? '' : 'disabled'}>Continue to Configure &rarr;</button>
    </div>`;
  main.querySelectorAll('[data-scan]').forEach(b =>
    b.addEventListener('click', () => doCapture(b.dataset.scan)));
  const sb = $('#scan-btn');
  if (next) sb.addEventListener('click', () => doCapture(next)); else sb.disabled = true;
  $('#to-config').addEventListener('click', () => gotoStage('configure'));
  $('#capture-dpi').addEventListener('change', persistCaptureDpi);
}

async function persistCaptureDpi(e) {
  const select = e.currentTarget, status = $('#capture-dpi-status');
  const dpi = +select.value;
  setPath(S.overrides, 'capture.dpi', dpi);
  select.disabled = true;
  status.textContent = 'saving...';
  try {
    await save();
    status.textContent = `locked at ${dpi} dpi`;
  } catch (err) {
    status.textContent = 'could not save';
    logLine('[error] could not save capture resolution: ' + err.message);
  } finally {
    select.disabled = false;
  }
}

function slot(role) {
  const done = S.meta.scans[role];
  const label = { flat: 'flat-field', calib0: 'calib 0°', calib90: 'calib 90°' }[role] || role;
  return `<div class="slot ${done ? 'done' : ''}" id="slot-${role}">
    <div class="slot-cap"><span>${label}</span>
      <span class="led roleled"></span></div>
    <div class="slot-body">${done
      ? `<img src="${api.scanURL(S.sid, role)}" alt="${role}">`
      : `<div class="slot-hint">no scan yet<br><button class="btn btn--sm btn--ghost" data-scan="${role}">Scan</button></div>`}
    </div></div>`;
}

async function doCapture(role) {
  const el = $('#slot-' + role); if (el) { el.classList.add('busy'); el.classList.remove('done'); }
  disableActions(true);
  openLog(); logLine(`[scan] requesting ${role}…`);
  try { await api.capture(S.sid, role); } catch (e) { logLine('[error] ' + e.message); disableActions(false); return; }
  streamUntilDone(async (st) => {
    disableActions(false);
    await refreshMeta();
    if (st.status === 'done') renderCapture();
    else if (el) el.classList.remove('busy');
  });
}

// ---- CONFIGURE ------------------------------------------------------------
function renderConfigure() {
  const c = S.cfg, dpiOpts = (S.device?.dpi_options?.length ? S.device.dpi_options : [300, 600, 1200]);
  const main = $('#stage-main');
  const sel = (path, opts) => `<select data-path="${path}">${opts.map(o =>
    `<option ${String(getPath(c, path)) === String(o) ? 'selected' : ''}>${o}</option>`).join('')}</select>`;
  const num = (path, step = 1) => `<input type="number" step="${step}" data-path="${path}" value="${getPath(c, path)}">`;
  const tgl = (path) => `<button class="tgl" role="switch" data-path="${path}"
    aria-checked="${!!getPath(c, path)}"></button>`;
  const field = (t, sub, ctrl) => `<div class="field"><div class="lab"><b>${t}</b><small>${sub}</small></div>${ctrl}</div>`;

  main.innerHTML = `
    <h2 class="stage-title">Configure</h2>
    <p class="stage-lead">Defaults are tuned for a grape leaf at 600 dpi. Adjust if needed —
      every value is saved with this session.</p>
    <div class="card grid">
      ${field('Scan resolution', 'dpi · higher = finer veins, larger files', sel('capture.dpi', dpiOpts))}
      ${field('Outlier rejection', 'drop specular/shadow samples per pixel', sel('solve.rejection', ['none', 'drop_brightest', 'drop_brightest_and_darkest']))}
      ${field('Edge trim', 'px eroded to kill the background-mix boundary ring', num('output.edge.trim_px'))}
      ${field('Texture padding', 'px of bleed outside the leaf · 0 for transparent cutouts', num('output.edge.pad_px'))}
      ${field('Alpha map', 'export the leaf silhouette + RGBA copies', tgl('output.alpha.enabled'))}
      ${field('Alpha feather', 'px soften on the silhouette edge (0 = crisp)', num('output.alpha.feather_px', 0.5))}
      ${field('Non-rigid align', 'correct leaf deformation between rotations', tgl('align.nonrigid.enabled'))}
      ${field('Height map', 'integrate normals into relief (Frankot–Chellappa)', tgl('integrate.enabled'))}
      ${field('Auto-crop', 'crop to the leaf so full-res fits in memory', tgl('runtime.auto_crop'))}
      ${field('Working scale', '1.0 = full res · lower for a fast preview', num('runtime.scale', 0.05))}
    </div>
    <details class="expander"><summary>Advanced — raw overrides (JSON)</summary>
      <p class="stage-lead" style="margin-top:8px">Any config value can be overridden here. Merged over defaults.</p>
      <textarea id="ov-json">${esc(JSON.stringify(S.overrides, null, 2))}</textarea>
      <div style="margin-top:10px"><button class="btn btn--sm" id="apply-json">Apply overrides</button>
        <span id="ov-msg" class="muted mono" style="margin-left:12px"></span></div>
    </details>
    <div class="stage-foot">
      <button class="btn btn--ghost" id="back-cap">&larr; Capture</button>
      <button class="btn btn--primary" id="save-cfg">Save &amp; continue &rarr;</button>
    </div>`;

  main.querySelectorAll('[data-path]').forEach(ctrl => {
    if (ctrl.classList.contains('tgl'))
      ctrl.addEventListener('click', () => ctrl.setAttribute('aria-checked', ctrl.getAttribute('aria-checked') !== 'true'));
    ctrl.addEventListener('change', () => stageFormToOverrides(main));
  });
  $('#apply-json').addEventListener('click', async () => {
    try {
      S.overrides = JSON.parse($('#ov-json').value || '{}');
      await save(); $('#ov-msg').textContent = 'applied';
      const b = await api.getConfig(S.sid); S.cfg = b.config; renderConfigure();
    } catch (e) { $('#ov-msg').textContent = 'invalid JSON: ' + e.message; }
  });
  $('#back-cap').addEventListener('click', () => gotoStage('capture'));
  $('#save-cfg').addEventListener('click', async () => {
    stageFormToOverrides(main); await save(); gotoStage('process');
  });
}

function stageFormToOverrides(main) {
  const ov = structuredClone(S.overrides);
  main.querySelectorAll('[data-path]').forEach(ctrl => {
    const p = ctrl.dataset.path; let v;
    if (ctrl.classList.contains('tgl')) v = ctrl.getAttribute('aria-checked') === 'true';
    else if (ctrl.tagName === 'SELECT') v = isNaN(+ctrl.value) ? ctrl.value : +ctrl.value;
    else v = +ctrl.value;
    setPath(ov, p, v);
  });
  S.overrides = ov;
}
async function save() { await api.setConfig(S.sid, S.overrides); S.cfg = (await api.getConfig(S.sid)).config; }

// ---- PROCESS --------------------------------------------------------------
function renderProcess() {
  const c = S.cfg, done = S.meta.status === 'done';
  const main = $('#stage-main');
  main.innerHTML = `
    <h2 class="stage-title">Process</h2>
    <p class="stage-lead">Run the photometric-stereo solve. The live log streams every
      stage; watch the residual — low and balanced means a trustworthy result.</p>
    <div class="summary-grid" style="margin-bottom:18px">
      <div class="stat"><div class="n">${c.capture.dpi}</div><div class="l">dpi</div></div>
      <div class="stat"><div class="n">${c.runtime.scale}</div><div class="l">scale</div></div>
      <div class="stat"><div class="n">${c.solve.rejection.replace('drop_', '').replace(/_/g, ' ')}</div><div class="l">rejection</div></div>
      <div class="stat"><div class="n">${c.runtime.auto_crop ? 'on' : 'off'}</div><div class="l">auto-crop</div></div>
    </div>
    <div class="prog"><div class="prog-bar"><div class="prog-fill" id="pfill"></div></div>
      <div class="prog-stages" id="pstages">${PSTAGES.map(s => `<span class="pstage">${s}</span>`).join('')}</div>
    </div>
    <div id="run-summary" style="margin-top:18px"></div>
    <div class="stage-foot">
      <button class="btn btn--ghost" id="back-cfg">&larr; Configure</button>
      <button class="btn btn--primary" id="run-btn">${done ? 'Re-run' : 'Run reconstruction'}</button>
    </div>`;
  if (done) showSummary(S.meta.result);
  $('#back-cfg').addEventListener('click', () => gotoStage('configure'));
  $('#run-btn').addEventListener('click', doRun);
}

async function doRun() {
  disableActions(true); openLog();
  document.querySelectorAll('.pstage').forEach(p => p.classList.remove('on', 'done'));
  $('#pfill').style.width = '0%'; $('#run-summary').innerHTML = '';
  logLine('[run] starting reconstruction…');
  try { await api.run(S.sid); } catch (e) { logLine('[error] ' + e.message); disableActions(false); return; }
  streamUntilDone(async (st) => {
    disableActions(false); await refreshMeta();
    if (st.status === 'done') {
      setStageProgress(6); showSummary(st.result);
      document.querySelectorAll('.stage').forEach(b => b.classList.toggle('done', doneStage(b.dataset.stage)));
      $('#run-summary').insertAdjacentHTML('beforeend',
        `<div style="margin-top:16px"><button class="btn btn--primary" id="to-res">View results &rarr;</button></div>`);
      $('#to-res').addEventListener('click', () => gotoStage('results'));
      $('.stage[data-stage="results"]').disabled = false;
    } else {
      $('#run-summary').innerHTML = `<div class="err-box">Reconstruction failed. See the raw log.\n${esc(st.error || '')}</div>`;
    }
  });
}

function showSummary(r) {
  if (!r) return;
  $('#run-summary').innerHTML = `
    <div class="summary-grid">
      <div class="stat"><div class="n">${r.az0.toFixed(0)}°</div><div class="l">azimuth az0</div></div>
      <div class="stat"><div class="n">${r.el.toFixed(0)}°</div><div class="l">elevation</div></div>
      <div class="stat"><div class="n">${Math.max(...r.residual_means).toFixed(3)}</div><div class="l">max residual</div></div>
      <div class="stat"><div class="n">${(r.valid_px / 1e6).toFixed(1)}M</div><div class="l">solved px</div></div>
    </div>`;
}

function setStageProgress(idx) {
  const chips = document.querySelectorAll('.pstage');
  chips.forEach((c, i) => { c.classList.toggle('done', i < idx); c.classList.toggle('on', i === idx); });
  const f = $('#pfill'); if (f) f.style.width = Math.min(100, (idx / PSTAGES.length) * 100) + '%';
}

// ---- RESULTS --------------------------------------------------------------
function renderResults() {
  const sid = S.sid;
  const gallery = [
    ['normal_gl.png', 'normal'], ['albedo_srgb.png', 'albedo'], ['alpha.png', 'alpha'],
    ['height.png', 'height'], ['qa/residual_scan0.png', 'residual 0'],
    ['qa/residual_scan3.png', 'residual 3'], ['qa/mask_agreement.png', 'agreement'],
  ];
  const main = $('#stage-main');
  main.innerHTML = `
    <h2 class="stage-title">Results</h2>
    <p class="stage-lead">Drag the light across the leaf to relight it in real time —
      this is the reconstructed normal + albedo, exactly as a shader would use them.</p>
    <div class="results">
      <div class="relight">
        <canvas id="rl"></canvas>
        <div class="relight-hud">drag to move the light</div>
        <div class="relight-ctrls">
          <div class="ctrl">elevation <input type="range" id="rl-el" min="8" max="85" value="35"></div>
          <div class="ctrl">ambient <input type="range" id="rl-amb" min="0" max="60" value="12"></div>
          <div class="ctrl">convention <span class="seg" id="rl-conv">
            <button class="on" data-v="gl">GL</button><button data-v="dx">DX</button></span></div>
          <div class="ctrl">backdrop <span class="seg" id="rl-bg">
            <button class="on" data-v="1">checker</button><button data-v="0">dark</button></span></div>
        </div>
      </div>
      <div>
        <div class="eyebrow" style="margin-bottom:8px">Outputs — click to open full size</div>
        <div class="thumbs">${gallery.map(([f, l]) =>
          `<a class="thumb" href="${api.resultURL(sid, f)}" target="_blank" rel="noopener">
             <img src="${api.resultURL(sid, f, 240)}" alt="${l}"><div class="cap">${l}</div></a>`).join('')}
        </div>
      </div>
    </div>`;
  const rl = new Relight($('#rl'));
  rl.load(api.resultURL(sid, 'normal_gl.png', 2048), api.resultURL(sid, 'albedo_srgb.png', 2048),
          api.resultURL(sid, 'alpha.png', 2048)).catch(e => logLine('[error] relight: ' + e.message));
  S.relight = rl;
  $('#rl-el').addEventListener('input', e => rl.set('el', +e.target.value));
  $('#rl-amb').addEventListener('input', e => rl.set('ambient', +e.target.value / 100));
  segWire('#rl-conv', v => rl.set('dx', v === 'dx'));
  segWire('#rl-bg', v => rl.set('backdrop', +v));
}
function segWire(sel, fn) {
  const seg = $(sel);
  seg.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    seg.querySelectorAll('button').forEach(x => x.classList.remove('on'));
    b.classList.add('on'); fn(b.dataset.v);
  }));
}

// ---- telemetry + log ------------------------------------------------------
function paintTelemetry() {
  const m = S.meta, d = S.device, r = m?.result;
  const dots = LEAF.map(k => `<div class="dot ${m?.scans[k] ? 'done' : ''}">${k[1]}</div>`).join('');
  let lv = '';
  if (r) {
    const L = lightVectors(r.az0, r.el, r.thetas);
    lv = `<div class="tel-group"><div class="tel-k">Light vectors L[k]</div>
      <div class="lvec">${L.map((v, i) =>
        `<span>L${i}</span><span><span class="cx">${v.x}</span> <span class="cy">${v.y}</span> <span class="cz">${v.z}</span></span>`).join('')}</div>
      <div class="tel-row"><span class="tel-k">az0 / el</span><span class="v">${r.az0.toFixed(1)}° / ${r.el.toFixed(1)}°</span></div>
      <div class="tel-row"><span class="tel-k">residual</span><span class="v settle">${r.residual_means.map(x => x.toFixed(3)).join(' ')}</span></div>
    </div>`;
  }
  $('#tel-body').innerHTML = `
    <div class="tel-group"><div class="tel-k">Bench</div>
      <div class="tel-row"><span>scanner</span><span class="v">${d?.connected ? 'online' : 'offline'}</span></div>
      <div class="tel-row"><span>max depth</span><span class="v">${d?.max_bit_depth || '—'}-bit</span></div>
    </div>
    <div class="tel-group"><div class="tel-k">Session</div>
      <div class="tel-row"><span>name</span><span class="v">${esc(m?.name || '—')}</span></div>
      <div class="tel-row"><span>status</span><span class="v">${m?.status || '—'}</span></div>
      <div class="tel-row"><span>scans</span><span class="dots">${dots}</span></div>
    </div>${lv}`;
}

function lightVectors(az0, el, thetas) {
  const er = el * Math.PI / 180, ch = Math.cos(er), sz = Math.sin(er);
  return thetas.map(th => {
    const a = (az0 - th) * Math.PI / 180;
    return { x: (ch * Math.cos(a)).toFixed(2), y: (ch * Math.sin(a)).toFixed(2), z: sz.toFixed(2) };
  });
}

function toggleLog() {
  const c = $('#log-console'), t = $('#log-toggle');
  const open = c.hidden; c.hidden = !open; t.setAttribute('aria-expanded', String(open));
}
function openLog() { const c = $('#log-console'); if (c.hidden) toggleLog(); }
function logLine(s) {
  S.log.push(s);
  const c = $('#log-console');
  const cls = /\[error|fail/i.test(s) ? 'err' : /\[done|ok|saved/i.test(s) ? 'ok'
    : /\[calib|\[solve|residual/i.test(s) ? 'warn' : '';
  c.insertAdjacentHTML('beforeend', `<span class="${cls}">${esc(s)}</span>\n`);
  c.scrollTop = c.scrollHeight;
  for (const [pre, idx] of Object.entries(STAGE_MAP)) if (s.startsWith(pre)) setStageProgress(idx);
}

// stream the current job's log to console until it reaches a terminal state
function streamUntilDone(onDone) {
  if (S.closeWS) S.closeWS();
  S.closeWS = streamJob(S.sid,
    lines => lines.forEach(logLine),
    st => { S.closeWS && S.closeWS(); S.closeWS = null; onDone(st); });
}

function disableActions(on) {
  document.querySelectorAll('#stage-main .btn').forEach(b => { b.disabled = on; });
}

boot();
