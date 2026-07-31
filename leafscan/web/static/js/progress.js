export const STAGE_MAP = { '[objects]': 0, '[crop]': 0, '[load]': 0, '[rigid]': 1,
  '[nonrigid]': 1, '[valid]': 1, '[calib]': 2, '[solve]': 3, '[roughness]': 4,
  '[integrate]': 5, '[out]': 6, '[qa]': 6, '[atlas]': 6, '[done]': 7 };

export function advanceProgress(current, line) {
  const progress = { ...current };
  let stageTouched = false, stageExact = false, objectStarted = false;
  if (line.includes('[objects]')) progress.multi = true;
  const objectMatch = line.match(/\[objects\]\s+reconstructing object\s+(\d+)\/(\d+)/i);
  if (objectMatch) {
    progress.object = +objectMatch[1];
    progress.count = +objectMatch[2];
    progress.stage = 0;
    stageTouched = stageExact = objectStarted = true;
  }
  for (const [prefix, index] of Object.entries(STAGE_MAP)) {
    if (!line.includes(prefix)) continue;
    const exact = progress.multi
      && (/^\[object\s+\d+\]/i.test(line) || prefix === '[atlas]');
    progress.stage = exact ? index : Math.max(progress.stage, index);
    stageTouched = true;
    stageExact = stageExact || exact;
  }
  return { progress, stageTouched, stageExact, objectStarted };
}

export function progressFromLines(lines) {
  let progress = { stage: 0, multi: false, object: 0, count: 0 };
  lines.forEach(line => { progress = advanceProgress(progress, line).progress; });
  return progress;
}
