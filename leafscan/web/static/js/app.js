// LUMEN-PS front controller: capture, persistent processing, and result inspection.
import { api, streamJob } from './api.js';
import { Relight } from './relight.js';

const $ = (s, r = document) => r.querySelector(s);
const LEAF = ['k0', 'k1', 'k2', 'k3'];
const EXPECTED_BACKEND = '2026.07-silhouette-mask-v1';
const STAGES = ['capture', 'process', 'results'];
const STAGE_MAP = { '[crop]': 0, '[load]': 0, '[rigid]': 1, '[nonrigid]': 1,
  '[valid]': 1, '[calib]': 2, '[solve]': 3, '[integrate]': 4,
  '[out]': 5, '[qa]': 5, '[done]': 6 };
const PSTAGES = ['Load', 'Align', 'Calibrate', 'Solve', 'Integrate', 'Output'];

const RESULT_META = {
  'normal_gl.png': ['OpenGL normal', 'Material map'],
  'normal_dx.png': ['DirectX normal', 'Material map'],
  'albedo_srgb.png': ['Albedo (sRGB)', 'Material map'],
  'albedo.png': ['Albedo (linear)', 'Material map'],
  'alpha.png': ['Alpha', 'Material map'],
  'height.png': ['Height', 'Material map'],
  'albedo_srgb_rgba.png': ['Albedo RGBA', 'Material map'],
  'normal_gl_rgba.png': ['Normal RGBA', 'Material map'],
  'qa/normal_preview.png': ['Normal preview', 'Diagnostics'],
  'qa/subsurface_hint.png': ['Subsurface hint', 'Diagnostics'],
  'qa/mask_agreement.png': ['Mask agreement', 'Diagnostics'],
  'qa/rejection_coverage.png': ['Rejection coverage', 'Diagnostics'],
};

const S = {
  device: null, backendCurrent: true, sid: null, meta: null, cfg: null, overrides: {},
  stage: 'capture', relight: null, closeWS: null, log: [],
  jobKind: null, jobStatus: 'idle', jobCursor: 0, jobResult: null,
  processStage: 0, resultMode: 'relight', selectedResult: null,
  removeImportRole: null,
};

// ---- helpers --------------------------------------------------------------
const getPath = (o, p) => p.split('.').reduce((a, k) => (a ? a[k] : undefined), o);
function setPath(o, p, v) {
  const ks = p.split('.'); let a = o;
  ks.slice(0, -1).forEach(k => (a = a[k] = a[k] || {}));
  a[ks.at(-1)] = v;
}
const esc = s => String(s ?? '').replace(/[&<>]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const escAttr = s => esc(s).replace(/"/g, '&quot;');
const bytes = n => n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`;

function toggleControl(path, label, note) {
  return `<div class="field"><div class="lab"><b>${label}</b><small>${note}</small></div>
    <button class="tgl" type="button" role="switch" data-path="${path}"
      aria-checked="${!!getPath(S.cfg, path)}" aria-label="${label}"></button></div>`;
}
function numberControl(path, label, note, step = 1, min = null) {
  return `<div class="field"><div class="lab"><b>${label}</b><small>${note}</small></div>
    <input type="number" data-path="${path}" step="${step}" ${min === null ? '' : `min="${min}"`}
      value="${getPath(S.cfg, path)}" aria-label="${label}"></div>`;
}
function selectControl(path, label, note, options) {
  const current = String(getPath(S.cfg, path));
  return `<div class="field"><div class="lab"><b>${label}</b><small>${note}</small></div>
    <select data-path="${path}" aria-label="${label}">${options.map(([value, text]) =>
      `<option value="${value}" ${current === String(value) ? 'selected' : ''}>${text}</option>`).join('')}
    </select></div>`;
}
function wireConfigControls(root) {
  root.querySelectorAll('.tgl[data-path]').forEach(ctrl => ctrl.addEventListener('click', () => {
    ctrl.setAttribute('aria-checked', String(ctrl.getAttribute('aria-checked') !== 'true'));
  }));
}
function formToOverrides(root) {
  const ov = structuredClone(S.overrides);
  root.querySelectorAll('[data-path]').forEach(ctrl => {
    const path = ctrl.dataset.path; let value;
    if (ctrl.classList.contains('tgl')) value = ctrl.getAttribute('aria-checked') === 'true';
    else if (ctrl.tagName === 'SELECT') value = isNaN(+ctrl.value) ? ctrl.value : +ctrl.value;
    else value = +ctrl.value;
    setPath(ov, path, value);
  });
  S.overrides = ov;
}
async function saveConfig() {
  await api.setConfig(S.sid, S.overrides);
  const bundle = await api.getConfig(S.sid);
  S.cfg = bundle.config; S.overrides = bundle.overrides || S.overrides;
}

// ---- boot -----------------------------------------------------------------
async function boot() {
  $('#log-toggle').addEventListener('click', toggleLog);
  $('#btn-new').addEventListener('click', openNewSession);
  $('#new-session-cancel').addEventListener('click', closeNewSession);
  $('#new-session-form').addEventListener('submit', newSession);
  $('#reset-scans-cancel').addEventListener('click', closeResetScans);
  $('#reset-scans-confirm').addEventListener('click', resetAllScans);
  $('#remove-import-cancel').addEventListener('click', closeRemoveImport);
  $('#remove-import-confirm').addEventListener('click', removeImportedScan);
  $('#btn-shutdown').addEventListener('click', openShutdown);
  $('#shutdown-cancel').addEventListener('click', closeShutdown);
  $('#shutdown-confirm').addEventListener('click', shutdownApp);
  $('#delete-session-cancel').addEventListener('click', closeDeleteSession);
  $('#delete-session-ui').addEventListener('click', () => deleteSession(false));
  $('#delete-session-files').addEventListener('click', () => deleteSession(true));
  $('#back-to-sessions').addEventListener('click', returnToRecent);
  document.querySelectorAll('.stage').forEach(b =>
    b.addEventListener('click', () => gotoStage(b.dataset.stage)));
  try {
    const [device, backend] = await Promise.all([api.device(), api.version()]);
    S.device = device; S.backendCurrent = backend.version === EXPECTED_BACKEND;
  } catch {
    S.device = { connected: false, note: 'Restart LUMEN-PS to load the updated scan engine.' };
    S.backendCurrent = false;
  }
  paintDevice(); await paintRecent();
}

function paintDevice() {
  const d = S.device, pill = $('#device-pill'), bd = $('#boot-device');
  const ready = d.connected && S.backendCurrent;
  pill.className = 'pill ' + (ready ? 'pill--ok' : 'pill--err');
  $('.pill-text', pill).textContent = !S.backendCurrent ? 'restart required'
    : d.connected ? `${d.name} - ${d.max_bit_depth}-bit` : 'no scanner';
  bd.querySelector('.led').style.color = ready ? 'var(--ok)' : 'var(--err)';
  $('.boot-device-text', bd).innerHTML = !S.backendCurrent
    ? 'scan engine update pending - close the old console and run <b>run.bat</b> again'
    : d.connected
    ? `bench online - <b>${esc(d.name)}</b> - ${d.dpi_options.join('/')} dpi - ${d.max_bit_depth}-bit`
    : esc(d.note || 'no scanner detected');
}
async function paintRecent() {
  const list = await api.sessions(), ul = $('#recent-list');
  if (!list.length) { ul.innerHTML = '<li class="muted">none yet</li>'; return; }
  ul.innerHTML = '';
  list.slice(0, 8).forEach(m => {
    const li = document.createElement('li'); li.className = 'recent-item';
    const tag = m.status === 'done' ? 'tag--done' : m.status === 'ready' ? 'tag--ready' : '';
    li.innerHTML = `<span class="r-name">${esc(m.name)}</span><span class="r-meta">
      ${m.created.slice(0, 16).replace('T', ' ')} <span class="tag ${tag}">${m.status}</span></span>
      <button class="recent-delete" type="button" data-delete="${escAttr(m.id)}" aria-label="Remove ${escAttr(m.name)}">Delete</button>`;
    li.onclick = () => enterSession(m.id); ul.appendChild(li);
    $('.recent-delete', li).addEventListener('click', e => { e.stopPropagation(); openDeleteSession(m); });
  });
}
function openNewSession() {
  $('#new-session-dialog').hidden = false;
  const input = $('#new-session-name'); input.focus(); input.select();
}
function closeNewSession() { $('#new-session-dialog').hidden = true; }
function openResetScans() {
  $('#reset-scans-dialog').hidden = false;
  $('#reset-scans-cancel').focus();
}
function closeResetScans() { $('#reset-scans-dialog').hidden = true; }
function openRemoveImport(role) {
  S.removeImportRole = role;
  $('#remove-import-description').textContent = `Remove imported ${role}? The other captures will stay in place.`;
  $('#remove-import-dialog').hidden = false;
  $('#remove-import-cancel').focus();
}
function closeRemoveImport() { $('#remove-import-dialog').hidden = true; S.removeImportRole = null; }
function openShutdown() { $('#shutdown-dialog').hidden = false; $('#shutdown-cancel').focus(); }
function closeShutdown() { $('#shutdown-dialog').hidden = true; }
async function shutdownApp() {
  const button = $('#shutdown-confirm');
  button.disabled = true; button.textContent = 'Shutting down...';
  try {
    await api.shutdown();
    if (S.closeWS) { S.closeWS(); S.closeWS = null; }
    disposeRelight();
    $('#shutdown-dialog').hidden = true;
    $('#shutdown-screen').hidden = false;
  } catch (err) {
    $('#shutdown-description').textContent = `Could not shut down LUMEN-PS: ${err.message}`;
    button.disabled = false; button.textContent = 'Try again';
  }
}
function openDeleteSession(meta) {
  S.deleteCandidate = meta;
  $('#delete-session-description').textContent = `Remove “${meta.name}” from Recent sessions, or permanently delete its scans and outputs${meta.output_dir ? ` in ${meta.output_dir}` : ''}.`;
  $('#delete-session-dialog').hidden = false;
  $('#delete-session-cancel').focus();
}
function closeDeleteSession() { $('#delete-session-dialog').hidden = true; S.deleteCandidate = null; }
async function deleteSession(deleteFiles) {
  const candidate = S.deleteCandidate;
  if (!candidate) return;
  const buttons = $('#delete-session-dialog').querySelectorAll('button');
  buttons.forEach(b => b.disabled = true);
  try { await api.deleteSession(candidate.id, deleteFiles); closeDeleteSession(); await paintRecent(); }
  catch (err) { $('#delete-session-description').textContent = `Could not remove this session: ${err.message}`; }
  finally { buttons.forEach(b => b.disabled = false); }
}
async function newSession(e) {
  e.preventDefault();
  const input = $('#new-session-name'), submit = $('#new-session-form button[type="submit"]');
  submit.disabled = true;
  try { const m = await api.newSession(input.value.trim() || 'Untitled leaf'); closeNewSession(); await enterSession(m.id); }
  catch (err) { logLine('[error] could not create session: ' + err.message); }
  finally { submit.disabled = false; }
}

async function enterSession(sid) {
  if (S.closeWS) { S.closeWS(); S.closeWS = null; }
  Object.assign(S, { sid, log: [], jobKind: null, jobStatus: 'idle', jobCursor: 0,
    jobResult: null, processStage: 0, selectedResult: null, resultMode: 'relight' });
  $('#log-console').innerHTML = '';
  await refreshMeta();
  const bundle = await api.getConfig(sid);
  S.cfg = bundle.config; S.overrides = bundle.overrides || {};
  $('#boot').hidden = true; $('#shell').hidden = false;
  const start = S.meta.status === 'done' ? 'results' : S.meta.ready ? 'process' : 'capture';
  gotoStage(start);
}
async function returnToRecent() {
  if (S.closeWS) { S.closeWS(); S.closeWS = null; }
  disposeRelight();
  S.sid = null; S.meta = null;
  $('#shell').hidden = true; $('#boot').hidden = false;
  await paintRecent();
}
async function refreshMeta() { S.meta = await api.session(S.sid); paintTelemetry(); paintStageNav(); }

// ---- stage routing --------------------------------------------------------
function paintStageNav() {
  if (!S.meta) return;
  document.querySelectorAll('.stage').forEach(b => {
    b.classList.toggle('active', b.dataset.stage === S.stage);
    b.classList.toggle('done', doneStage(b.dataset.stage));
    b.disabled = (b.dataset.stage === 'results' && S.meta.status !== 'done')
      || (b.dataset.stage === 'process' && !S.meta.ready);
  });
}
function disposeRelight() { if (S.relight) { try { S.relight.dispose(); } catch (_) {} S.relight = null; } }

function gotoStage(name) {
  disposeRelight();
  S.stage = name; paintStageNav();
  const render = { capture: renderCapture, process: renderProcess, results: renderResults }[name];
  Promise.resolve(render()).catch(err => {
    $('#stage-main').innerHTML = `<div class="err-box">Could not render ${esc(name)}.\n${esc(err.message)}</div>`;
    logLine('[error] ' + err.message);
  });
}
function doneStage(name) {
  if (name === 'capture') return S.meta.ready;
  return (name === 'process' || name === 'results') && S.meta.status === 'done';
}

// ---- Capture --------------------------------------------------------------
function renderCapture() {
  const m = S.meta, c = S.cfg, next = LEAF.find(r => !m.scans[r]);
  const dpiOpts = S.device?.dpi_options?.length ? S.device.dpi_options : [300, 600, 1200];
  const smart = c.capture.smart_roi || { enabled: true, preview_dpi: 75 };
  const main = $('#stage-main');
  main.innerHTML = `
    <div class="stage-heading"><div><div class="eyebrow">Stage 1 / Acquisition</div>
      <h2 class="stage-title">Capture</h2><p class="stage-lead">Rotate the subject a quarter turn between four locked-light scans.</p></div>
      <div class="capture-readout"><b>${c.capture.dpi}</b><span>detail dpi</span></div></div>
    <div class="capture-settings capture-settings--expanded">
      <div><label for="capture-dpi">Detail resolution</label>
        <select id="capture-dpi">${dpiOpts.map(d => `<option value="${d}" ${d === c.capture.dpi ? 'selected' : ''}>${d} dpi</option>`).join('')}</select></div>
      <div class="capture-speed"><button class="tgl" id="smart-roi" type="button" role="switch"
        aria-checked="${!!smart.enabled}" aria-label="Fast area scan"></button>
        <div><b>Fast area scan</b><span>${smart.preview_dpi || 75} dpi locator pass, then detail-scan only the detected area.</span></div></div>
      <span id="capture-settings-status" class="mono ${S.backendCurrent ? 'muted' : 'err'}" aria-live="polite">
        ${S.backendCurrent ? '' : 'Backend update pending. Restart LUMEN-PS before scanning.'}</span>
    </div>
    <div class="instr"><div><div class="eyebrow">Rotation protocol</div>
      <p style="margin:.4em 0 0">Keep the same face up. After each scan, rotate it <span class="kbd">90&deg; clockwise</span> about its centre.</p></div></div>
    <div class="scan-grid" id="slots">${LEAF.map(slot).join('')}</div>
    <details class="expander"><summary>Optional acquisition references</summary>
      <p class="stage-lead" style="margin-top:8px">Flat-field and corrugated calibration are available when the bench needs tighter calibration.</p>
      <div class="scan-grid scan-grid--optional">${['flat', 'calib0', 'calib90'].map(slot).join('')}</div></details>
    <div class="stage-foot"><div><button class="btn btn--danger btn--sm" id="reset-scans"
      ${Object.values(m.scans).some(Boolean) ? '' : 'disabled'}>Reset all scans</button>
      <button class="btn btn--ghost btn--sm" id="cancel-job" ${S.jobKind?.startsWith('capture:') && ['queued', 'running'].includes(S.jobStatus) ? '' : 'hidden'}>Cancel scan</button>
      <span id="capture-job-note" class="mono muted"></span></div>
      <div class="foot-actions"><button class="btn btn--primary" id="scan-btn">${next ? 'Scan ' + next : 'All scans captured'}</button>
      <button class="btn" id="to-process" ${m.ready ? '' : 'disabled'}>Continue to Process &rarr;</button></div></div>`;
  main.querySelectorAll('[data-scan]').forEach(b => b.addEventListener('click', () => doCapture(b.dataset.scan)));
  main.querySelectorAll('[data-import]').forEach(b => b.addEventListener('click', () => {
    const input = main.querySelector(`[data-import-file="${b.dataset.import}"]`);
    if (input) input.click();
  }));
  main.querySelectorAll('[data-import-file]').forEach(input => input.addEventListener('change', () => {
    const file = input.files?.[0];
    if (file) importScan(input.dataset.importFile, file);
    input.value = '';
  }));
  main.querySelectorAll('[data-remove-import]').forEach(b =>
    b.addEventListener('click', () => openRemoveImport(b.dataset.removeImport)));
  const sb = $('#scan-btn'); if (next) sb.addEventListener('click', () => doCapture(next)); else sb.disabled = true;
  if (!S.backendCurrent) sb.disabled = true;
  $('#to-process').addEventListener('click', () => gotoStage('process'));
  $('#reset-scans').addEventListener('click', openResetScans);
  $('#cancel-job')?.addEventListener('click', cancelCurrentJob);
  $('#capture-dpi').addEventListener('change', persistCaptureSettings);
  $('#smart-roi').addEventListener('click', e => {
    const b = e.currentTarget; b.setAttribute('aria-checked', String(b.getAttribute('aria-checked') !== 'true'));
    persistCaptureSettings();
  });
  if (S.jobKind?.startsWith('capture:') && ['queued', 'running'].includes(S.jobStatus)) {
    disableActions(true); $('#capture-job-note').textContent = 'Scanner job continues in the background...';
  }
}
async function persistCaptureSettings() {
  const dpi = +$('#capture-dpi').value;
  const enabled = $('#smart-roi').getAttribute('aria-checked') === 'true';
  const status = $('#capture-settings-status');
  setPath(S.overrides, 'capture.dpi', dpi); setPath(S.overrides, 'capture.smart_roi.enabled', enabled);
  status.textContent = 'saving...';
  try { await saveConfig(); status.textContent = `${dpi} dpi - ${enabled ? 'fast area scan on' : 'full bed'}`; }
  catch (err) { status.textContent = 'could not save'; logLine('[error] capture settings: ' + err.message); }
}
function slot(role) {
  const done = S.meta.scans[role];
  const primary = LEAF.includes(role);
  const imported = (S.meta.capture_sources || {})[role] === 'imported';
  const label = { flat: 'flat-field', calib0: 'calib 0&deg;', calib90: 'calib 90&deg;' }[role] || role;
  const roi = (S.meta.capture_rois || {})[role];
  return `<div class="slot ${done ? 'done' : ''}" id="slot-${role}"><div class="slot-cap"><span>${label}</span>
    ${imported ? '<span class="slot-mode slot-mode--imported">IMPORTED</span>' : roi ? '<span class="slot-mode">ROI</span>' : '<span class="led roleled"></span>'}</div>
    <div class="slot-body">${done ? `<img src="${api.scanURL(S.sid, role)}" alt="${role}">
      <div class="slot-actions"><button class="btn btn--sm" type="button" data-scan="${role}">Rescan</button>
        ${primary ? `<button class="btn btn--sm btn--ghost" type="button" data-import="${role}" aria-label="Import ${role} from file">Import</button>` : ''}
        ${imported ? `<button class="btn btn--sm btn--danger" type="button" data-remove-import="${role}" aria-label="Remove imported ${role}">Remove</button>` : ''}</div>`
      : `<div class="slot-hint">no scan yet<div class="slot-empty-actions"><button class="btn btn--sm btn--ghost" data-scan="${role}">Scan</button>
        ${primary ? `<button class="btn btn--sm btn--ghost" type="button" data-import="${role}" aria-label="Import ${role} from file">Import</button>` : ''}</div></div>`}
      ${primary ? `<input hidden type="file" data-import-file="${role}" accept=".png,.tif,.tiff,.bmp,.jpg,.jpeg,.webp,image/png,image/tiff,image/bmp,image/jpeg,image/webp">` : ''}</div></div>`;
}
async function importScan(role, file) {
  const el = $('#slot-' + role); if (el) { el.classList.add('busy'); el.classList.remove('done'); }
  disableActions(true); logLine(`[import] loading ${file.name} into ${role}...`);
  try {
    await api.importScan(S.sid, role, file);
    await refreshMeta();
    logLine(`[import] ${role}: ${file.name} imported.`);
    renderCapture();
  } catch (err) {
    logLine('[error] import failed: ' + err.message);
    renderCapture();
  }
}
async function removeImportedScan() {
  const role = S.removeImportRole;
  if (!role) return;
  const button = $('#remove-import-confirm');
  button.disabled = true;
  try {
    await api.removeImportedScan(S.sid, role);
    closeRemoveImport();
    await refreshMeta();
    logLine(`[import] ${role}: imported scan removed.`);
    renderCapture();
  } catch (err) {
    $('#remove-import-description').textContent = `Could not remove ${role}: ${err.message}`;
  } finally { button.disabled = false; }
}
async function doCapture(role) {
  if (!S.backendCurrent) {
    logLine('[error] Restart LUMEN-PS to load the smart-area scan engine.');
    return;
  }
  await persistCaptureSettings();
  const el = $('#slot-' + role); if (el) { el.classList.add('busy'); el.classList.remove('done'); }
  disableActions(true); openLog(); logLine(`[scan] requesting ${role}...`);
  try {
    const started = await api.capture(S.sid, role);
    Object.assign(S, { jobKind: started.kind, jobStatus: started.status, jobCursor: 0 });
    watchCurrentJob(0);
  } catch (err) { logLine('[error] ' + err.message); disableActions(false); }
}

async function resetAllScans() {
  const button = $('#reset-scans-confirm');
  button.disabled = true;
  try {
    await api.resetScans(S.sid);
    closeResetScans();
    await refreshMeta();
    S.jobKind = null; S.jobStatus = 'idle'; S.processStage = 0; S.jobResult = null;
    logLine('[scan] all captures reset; ready for k0.');
    renderCapture();
  } catch (err) {
    logLine('[error] could not reset scans: ' + err.message);
  } finally {
    button.disabled = false;
  }
}

// ---- Process --------------------------------------------------------------
function renderProcess() {
  const c = S.cfg, busy = S.jobKind === 'run' && ['queued', 'running'].includes(S.jobStatus);
  const done = S.meta.status === 'done';
  if (done && !busy) S.processStage = PSTAGES.length;
  const main = $('#stage-main');
  main.innerHTML = `
    <div class="stage-heading"><div><div class="eyebrow">Stage 2 / Reconstruction</div><h2 class="stage-title">Process</h2>
      <p class="stage-lead">Tune the solve, choose where outputs live, then reconstruct every material and QA map.</p></div>
      <div class="process-state ${busy ? 'is-running' : done ? 'is-done' : ''}"><span class="led"></span>
        <div><b id="process-state-title">${busy ? 'Reconstruction running' : done ? 'Results ready' : 'Ready to solve'}</b>
        <small id="process-state-note">${busy ? 'Safe to leave this page.' : done ? 'You can re-run with new settings.' : 'Four source scans detected.'}</small></div></div></div>
    <div class="process-layout">
      <section class="settings-stack" id="process-settings">
        <div class="settings-section"><div class="settings-title"><span>01</span><div><b>Photometric solve</b><small>How observations become normals.</small></div></div>
          ${selectControl('solve.rejection', 'Outlier rejection', 'Remove glare and shadow samples per pixel.', [
            ['none', 'None'], ['drop_brightest', 'Drop brightest'], ['drop_brightest_and_darkest', 'Drop brightest + darkest']])}
          ${toggleControl('align.nonrigid.enabled', 'Non-rigid alignment', 'Correct small deformations between rotations.')}
          ${toggleControl('align.mask.detect_interior_holes', 'Detect holes in subject', 'Optional for perforated subjects. Warning: this can rotoscope white parts of the object.')}</div>
        <div class="settings-section"><div class="settings-title"><span>02</span><div><b>Output maps</b><small>Edges, relief, and transparency.</small></div></div>
          ${toggleControl('integrate.enabled', 'Height map', 'Integrate normals into a relief map.')}
          ${toggleControl('output.alpha.enabled', 'Alpha map', 'Export silhouette and RGBA convenience maps.')}
          ${numberControl('output.alpha.feather_px', 'Alpha feather', 'Softness at the silhouette edge, in pixels.', 0.5, 0)}
          ${numberControl('output.edge.trim_px', 'Edge trim', 'Remove mixed leaf/background boundary pixels.', 1, 0)}
          ${numberControl('output.edge.pad_px', 'Texture padding', 'Bleed outside the silhouette for mipmaps.', 1, 0)}</div>
        <div class="settings-section"><div class="settings-title"><span>03</span><div><b>Performance</b><small>Memory and preview-quality controls.</small></div></div>
          ${toggleControl('runtime.auto_crop', 'Auto-crop in memory', 'Crop full-bed legacy scans before solving.')}
          ${numberControl('runtime.scale', 'Working scale', '1.0 is full resolution; lower values are faster.', 0.05, 0.05)}</div>
        <details class="expander"><summary>Advanced raw overrides</summary><textarea id="ov-json">${esc(JSON.stringify(S.overrides, null, 2))}</textarea>
          <button class="btn btn--sm" id="apply-json" type="button">Apply JSON</button></details>
      </section>
      <section class="process-console">
        <div class="save-destination"><div class="eyebrow">Result destination</div><label for="output-dir">Save folder</label>
          <div class="save-folder-row"><input id="output-dir" type="text" value="${escAttr(S.meta.output_dir || '')}" placeholder="Default: this session's workspace">
          <button class="btn btn--sm" id="choose-output-dir" type="button">Choose folder…</button></div>
          <small>Enter an absolute folder path. Leave empty to keep results with the session.</small></div>
        <div class="prog"><div class="prog-bar"><div class="prog-fill" id="pfill"></div></div>
          <div class="prog-stages" id="pstages">${PSTAGES.map(s => `<span class="pstage">${s}</span>`).join('')}</div></div>
        <div id="run-summary"></div>
      </section>
    </div>
    <div class="stage-foot"><button class="btn btn--ghost" id="back-cap">&larr; Capture</button>
      <div class="foot-actions"><span id="process-save-status" class="mono muted" aria-live="polite"></span>
      <button class="btn btn--ghost" id="cancel-job" ${busy ? '' : 'hidden'}>Cancel reconstruction</button>
      <button class="btn btn--primary" id="run-btn">${busy ? 'Reconstruction running...' : done ? 'Re-run reconstruction' : 'Run reconstruction'}</button></div></div>`;
  wireConfigControls(main);
  $('#back-cap').addEventListener('click', () => gotoStage('capture'));
  $('#run-btn').addEventListener('click', doRun);
  $('#cancel-job')?.addEventListener('click', cancelCurrentJob);
  $('#choose-output-dir').addEventListener('click', chooseOutputDir);
  $('#apply-json').addEventListener('click', async () => {
    const status = $('#process-save-status');
    try { S.overrides = JSON.parse($('#ov-json').value || '{}'); await saveConfig(); status.textContent = 'JSON applied'; renderProcess(); }
    catch (err) { status.textContent = 'Invalid JSON: ' + err.message; }
  });
  setStageProgress(S.processStage);
  if (done && S.meta.result) showSummary(S.meta.result, true);
  if (busy) disableProcessControls(true);
  restoreJobState();
}
function disableProcessControls(on) {
  document.querySelectorAll('#process-settings input,#process-settings select,#process-settings button,#output-dir,#choose-output-dir,#run-btn')
    .forEach(el => { el.disabled = on; });
}
async function persistProcessSettings() {
  const main = $('#stage-main'), status = $('#process-save-status');
  formToOverrides(main); status.textContent = 'saving settings...';
  await saveConfig();
  const saved = await api.setOutputDir(S.sid, $('#output-dir').value.trim() || null);
  S.meta.output_dir = saved.output_dir; status.textContent = `saving to ${saved.effective_output_dir}`;
}
async function chooseOutputDir() {
  const status = $('#process-save-status'); status.textContent = 'opening folder chooser...';
  try {
    const choice = await api.chooseOutputDir();
    if (!choice.path) { status.textContent = 'folder selection cancelled'; return; }
    $('#output-dir').value = choice.path;
    const saved = await api.setOutputDir(S.sid, choice.path);
    S.meta.output_dir = saved.output_dir;
    status.textContent = `saving to ${saved.effective_output_dir}`;
  } catch (err) { status.textContent = 'could not choose folder'; logLine('[error] folder chooser: ' + err.message); }
}
async function doRun() {
  disableProcessControls(true); openLog();
  try {
    await persistProcessSettings();
    Object.assign(S, { processStage: 0, jobKind: 'run', jobStatus: 'queued', jobCursor: 0, jobResult: null });
    setStageProgress(0); $('#run-summary').innerHTML = '';
    logLine('[run] starting reconstruction...');
    await api.run(S.sid); watchCurrentJob(0); renderProcess();
  } catch (err) { logLine('[error] ' + err.message); disableProcessControls(false); }
}
async function restoreJobState() {
  try {
    const job = await api.job(S.sid);
    if (job.kind !== 'run') return;
    S.jobKind = job.kind; S.jobStatus = job.status; S.jobResult = job.result;
    S.processStage = stageFromLines(job.log || []);
    if (S.jobCursor === 0 && job.log?.length) {
      job.log.forEach(logLine); S.jobCursor = job.log.length;
    }
    setStageProgress(S.processStage);
    if (['queued', 'running'].includes(job.status)) {
      disableProcessControls(true); if (!S.closeWS) watchCurrentJob(S.jobCursor);
    } else if (job.status === 'done' && job.result) showSummary(job.result, true);
    else if (job.status === 'error') showRunError(job.error);
  } catch (err) { logLine('[error] could not restore process state: ' + err.message); }
}
function showSummary(r, withAction = false) {
  const box = $('#run-summary'); if (!box || !r) return;
  box.innerHTML = `<div class="summary-grid"><div class="stat"><div class="n">${r.az0.toFixed(0)}&deg;</div><div class="l">azimuth</div></div>
    <div class="stat"><div class="n">${r.el.toFixed(0)}&deg;</div><div class="l">elevation</div></div>
    <div class="stat"><div class="n">${Math.max(...r.residual_means).toFixed(3)}</div><div class="l">max residual</div></div>
    <div class="stat"><div class="n">${(r.valid_px / 1e6).toFixed(1)}M</div><div class="l">solved px</div></div></div>
    ${withAction ? '<button class="btn btn--primary result-jump" id="to-results">Inspect results &rarr;</button>' : ''}`;
  $('#to-results')?.addEventListener('click', () => gotoStage('results'));
}
function showRunError(error) {
  const box = $('#run-summary'); if (box) box.innerHTML = `<div class="err-box">Reconstruction failed.\n${esc(error || '')}</div>`;
}
function stageFromLines(lines) {
  let stage = 0;
  lines.forEach(line => { for (const [prefix, index] of Object.entries(STAGE_MAP)) if (line.startsWith(prefix)) stage = Math.max(stage, index); });
  return stage;
}
function setStageProgress(index) {
  S.processStage = Math.max(S.processStage, index);
  const chips = document.querySelectorAll('.pstage');
  chips.forEach((chip, i) => { chip.classList.toggle('done', i < S.processStage || S.processStage >= PSTAGES.length);
    chip.classList.toggle('on', i === S.processStage && S.processStage < PSTAGES.length); });
  const fill = $('#pfill'); if (fill) fill.style.width = `${Math.min(100, S.processStage / PSTAGES.length * 100)}%`;
}

// ---- Results --------------------------------------------------------------
async function renderResults() {
  const main = $('#stage-main');
  main.innerHTML = `<div class="result-loading"><span class="led"></span> Loading result bench...</div>`;
  const manifest = await api.results(S.sid);
  const imageFiles = manifest.files.filter(f => f.kind === 'png');
  const material = imageFiles.filter(f => !f.name.startsWith('qa/'));
  const diagnostic = imageFiles.filter(f => f.name.startsWith('qa/'));
  const thumb = file => {
    const [label, group] = resultLabel(file.name);
    return `<button class="result-thumb" type="button" data-file="${escAttr(file.name)}" title="${escAttr(label)}">
      <img src="${api.resultURL(S.sid, file.name, 220)}" alt=""><span>${esc(label)}</span><small>${bytes(file.bytes)}</small></button>`;
  };
  main.innerHTML = `
    <div class="stage-heading result-heading"><div><div class="eyebrow">Stage 3 / Evidence bench</div><h2 class="stage-title">Results</h2>
      <p class="stage-lead">Relight the reconstruction, then inspect every exported map and QA observation without leaving the bench.</p></div>
      <div class="output-path" title="${escAttr(manifest.output_dir)}"><span>saved to</span><b>${esc(manifest.output_dir)}</b></div></div>
    <div class="result-workbench">
      <section class="result-stage">
        <div class="result-modebar"><div class="seg" id="result-mode"><button class="on" data-mode="relight">Interactive relight</button>
          <button data-mode="map">Map inspector</button></div><div id="map-title" class="map-title">Recovered material under movable light</div></div>
        <div id="relight-panel" class="relight result-display"><canvas id="rl"></canvas><div class="relight-hud">drag across the specimen to move the light</div>
          <div class="relight-ctrls"><div class="ctrl">elevation <input type="range" id="rl-el" min="8" max="85" value="35"></div>
          <div class="ctrl">ambient <input type="range" id="rl-amb" min="0" max="60" value="12"></div>
          <div class="ctrl">normal <span class="seg" id="rl-conv"><button class="on" data-v="gl">GL</button><button data-v="dx">DX</button></span></div>
          <div class="ctrl">backdrop <span class="seg" id="rl-bg"><button class="on" data-v="1">grid</button><button data-v="0">dark</button></span></div></div></div>
        <div id="map-panel" class="map-panel result-display" hidden><div class="map-canvas checker" id="map-canvas"><img id="map-image" alt="Selected result map"></div>
          <div class="map-toolbar"><div class="ctrl">zoom <input type="range" id="map-zoom" min="25" max="300" value="100"></div>
          <div class="seg" id="map-bg"><button class="on" data-v="checker">grid</button><button data-v="dark">dark</button></div>
          <a class="btn btn--sm" id="map-download" download>Download original</a></div></div>
      </section>
      <aside class="result-library"><div class="result-library-head"><div class="eyebrow">Output library</div><span>${imageFiles.length} images</span></div>
        <details open><summary>Material maps <span>${material.length}</span></summary><div class="result-thumbs">${material.map(thumb).join('')}</div></details>
        <details open><summary>Diagnostics <span>${diagnostic.length}</span></summary><div class="result-thumbs">${diagnostic.map(thumb).join('')}</div></details>
        ${manifest.files.some(f => f.name === 'qa/report.txt') ? `<a class="report-link" href="${api.resultRawURL(S.sid, 'qa/report.txt')}" download>
          <span>QA report</span><small>Download calibration and residual readout</small></a>` : ''}</aside>
    </div>`;
  wireResults(imageFiles);
}
function resultLabel(name) {
  if (RESULT_META[name]) return RESULT_META[name];
  let match = name.match(/^qa\/(observed|predicted|residual)_scan(\d)\.png$/);
  if (match) return [`${match[1][0].toUpperCase() + match[1].slice(1)} - side ${match[2]}`, 'Diagnostics'];
  const plain = name.split('/').at(-1).replace(/\.png$/i, '').replace(/_/g, ' ');
  return [plain.replace(/\b\w/g, c => c.toUpperCase()), name.startsWith('qa/') ? 'Diagnostics' : 'Material map'];
}
function wireResults(files) {
  const relight = $('#relight-panel'), map = $('#map-panel');
  const modeButtons = $('#result-mode').querySelectorAll('button');
  function setMode(mode) {
    S.resultMode = mode; relight.hidden = mode !== 'relight'; map.hidden = mode !== 'map';
    modeButtons.forEach(b => b.classList.toggle('on', b.dataset.mode === mode));
    if (mode === 'relight') $('#map-title').textContent = 'Recovered material under movable light';
  }
  modeButtons.forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
  const names = new Set(files.map(f => f.name));
  if (names.has('normal_gl.png') && names.has('albedo_srgb.png') && names.has('alpha.png')) {
    disposeRelight();
    const rl = new Relight($('#rl'));
    rl.load(api.resultURL(S.sid, 'normal_gl.png', 2048), api.resultURL(S.sid, 'albedo_srgb.png', 2048),
      api.resultURL(S.sid, 'alpha.png', 2048)).catch(err => logLine('[error] relight: ' + err.message));
    S.relight = rl;
    $('#rl-el').addEventListener('input', e => rl.set('el', +e.target.value));
    $('#rl-amb').addEventListener('input', e => rl.set('ambient', +e.target.value / 100));
    segWire('#rl-conv', v => rl.set('dx', v === 'dx')); segWire('#rl-bg', v => rl.set('backdrop', +v));
  }
  document.querySelectorAll('.result-thumb').forEach(button => button.addEventListener('click', () => {
    const file = button.dataset.file, [label, group] = resultLabel(file);
    S.selectedResult = file; setMode('map');
    $('#map-title').textContent = `${group} / ${label}`;
    $('#map-image').src = api.resultURL(S.sid, file, 2400);
    $('#map-image').style.transform = 'scale(1)'; $('#map-zoom').value = '100';
    $('#map-download').href = api.resultRawURL(S.sid, file, true);
    document.querySelectorAll('.result-thumb').forEach(x => x.classList.toggle('on', x === button));
  }));
  $('#map-zoom').addEventListener('input', e => { $('#map-image').style.transform = `scale(${+e.target.value / 100})`; });
  segWire('#map-bg', value => { $('#map-canvas').className = `map-canvas ${value}`; });
  if (S.selectedResult && names.has(S.selectedResult)) {
    document.querySelector(`.result-thumb[data-file="${CSS.escape(S.selectedResult)}"]`)?.click();
  }
}
function segWire(selector, fn) {
  const seg = $(selector); if (!seg) return;
  seg.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    seg.querySelectorAll('button').forEach(x => x.classList.remove('on')); b.classList.add('on'); fn(b.dataset.v);
  }));
}

// ---- Job persistence and live logs ---------------------------------------
function watchCurrentJob(from = 0) {
  if (S.closeWS) S.closeWS();
  S.jobCursor = from;
  S.closeWS = streamJob(S.sid, lines => {
    S.jobCursor += lines.length; lines.forEach(logLine);
  }, async status => {
    if (S.closeWS) S.closeWS(); S.closeWS = null;
    S.jobStatus = status.status; S.jobKind = status.kind; S.jobResult = status.result;
    await refreshMeta();
    if (status.kind === 'run') {
      if (status.status === 'done') S.processStage = PSTAGES.length;
      if (S.stage === 'process') renderProcess();
    } else if (status.kind?.startsWith('capture:') && S.stage === 'capture') renderCapture();
  }, from);
}
async function cancelCurrentJob() {
  const button = $('#cancel-job'); if (button) { button.disabled = true; button.textContent = 'Cancelling...'; }
  try { await api.cancelJob(S.sid); logLine('[cancel] requested by user'); }
  catch (err) { logLine('[error] could not cancel: ' + err.message); if (button) button.disabled = false; }
}
function logLine(line) {
  S.log.push(line);
  const console = $('#log-console');
  const cls = /\[error|fail/i.test(line) ? 'err' : /\[done|ok|saved/i.test(line) ? 'ok'
    : /\[calib|\[solve|residual/i.test(line) ? 'warn' : '';
  console.insertAdjacentHTML('beforeend', `<span class="${cls}">${esc(line)}</span>\n`);
  console.scrollTop = console.scrollHeight;
  for (const [prefix, index] of Object.entries(STAGE_MAP)) if (line.startsWith(prefix)) setStageProgress(index);
}
function toggleLog() {
  const console = $('#log-console'), toggle = $('#log-toggle');
  const open = console.hidden; console.hidden = !open; toggle.setAttribute('aria-expanded', String(open));
}
function openLog() { if ($('#log-console').hidden) toggleLog(); }
function disableActions(on) { document.querySelectorAll('#stage-main .btn').forEach(b => { b.disabled = on; }); }

// ---- telemetry ------------------------------------------------------------
function paintTelemetry() {
  const m = S.meta, d = S.device, r = m?.result;
  const dots = LEAF.map(k => `<div class="dot ${m?.scans[k] ? 'done' : ''}">${k[1]}</div>`).join('');
  let light = '';
  if (r) {
    const vectors = lightVectors(r.az0, r.el, r.thetas);
    light = `<div class="tel-group"><div class="tel-k">Light vectors L[k]</div><div class="lvec">${vectors.map((v, i) =>
      `<span>L${i}</span><span><span class="cx">${v.x}</span> <span class="cy">${v.y}</span> <span class="cz">${v.z}</span></span>`).join('')}</div>
      <div class="tel-row"><span class="tel-k">az0 / el</span><span class="v">${r.az0.toFixed(1)}&deg; / ${r.el.toFixed(1)}&deg;</span></div>
      <div class="tel-row"><span class="tel-k">residual</span><span class="v settle">${r.residual_means.map(x => x.toFixed(3)).join(' ')}</span></div></div>`;
  }
  $('#tel-body').innerHTML = `<div class="tel-group"><div class="tel-k">Bench</div>
    <div class="tel-row"><span>scanner</span><span class="v">${d?.connected ? 'online' : 'offline'}</span></div>
    <div class="tel-row"><span>max depth</span><span class="v">${d?.max_bit_depth || '-'}-bit</span></div></div>
    <div class="tel-group"><div class="tel-k">Session</div><div class="tel-row"><span>name</span><span class="v">${esc(m?.name || '-')}</span></div>
    <div class="tel-row"><span>status</span><span class="v">${m?.status || '-'}</span></div>
    <div class="tel-row"><span>scans</span><span class="dots">${dots}</span></div></div>${light}`;
}
function lightVectors(az0, el, thetas) {
  const er = el * Math.PI / 180, ch = Math.cos(er), sz = Math.sin(er);
  return thetas.map(theta => { const a = (az0 - theta) * Math.PI / 180;
    return { x: (ch * Math.cos(a)).toFixed(2), y: (ch * Math.sin(a)).toFixed(2), z: sz.toFixed(2) }; });
}

boot();
