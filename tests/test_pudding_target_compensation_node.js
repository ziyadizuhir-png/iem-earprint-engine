/* Generic Moondrop target-level compensation regression across every Robust Target. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const sandbox = {
  window: {},
  document: { readyState: 'loading', addEventListener() {} },
  location: { pathname: '/', hostname: 'localhost' },
  console, setTimeout, clearTimeout, URL, Blob, fetch() { throw new Error('not used'); }
};
vm.runInNewContext(fs.readFileSync('app/equal_loudness.js', 'utf8'), sandbox);
vm.runInNewContext(fs.readFileSync('app/pudding_peq.js', 'utf8'), sandbox);
const api = sandbox.window.MoondropPuddingPEQ;
const t = api.__test;
const raw = t.parseCustomFR(fs.readFileSync('input/original_711/moondrop pudding fr.txt', 'utf8'), 'moondrop pudding fr.txt');
const targets = fs.readdirSync('output').filter(name => name.endsWith('__robust_target.txt')).sort();
assert(targets.length > 0, 'At least one Robust Target is required');

for (const file of targets) {
  const target = t.parseCustomFR(fs.readFileSync('output/' + file, 'utf8'), file, { walkplay: true });
  const original = target.curve;
  const solved = t.solvePuddingWithTargetCompensation(raw.curve, original, {
    solverConstraints: { maxBands: 10, minGain: -12, maxGain: 3 }
  });
  const result = solved.result;
  assert(result, `${file}: compensation produced no result`);
  assert(result.bands.length <= 10, `${file}: band limit exceeded`);
  assert(result.bands.every(b => b.freq >= 20 && b.freq <= 20000 && b.gain >= -12 && b.gain <= 3 && b.q >= 0.3 && b.q <= 10 && Math.abs(b.gain) >= 0.05), `${file}: invalid final Moondrop band`);
  assert(result.bands.every(b => Math.abs(b.gain * 10 - Math.round(b.gain * 10)) < 1e-8), `${file}: gain precision failed`);
  assert(result.bands.every(b => Math.abs(b.q * 100 - Math.round(b.q * 100)) < 1e-8), `${file}: Q precision failed`);
  assert.equal(result.metrics.deviceProfile.preampSupported, false, `${file}: fake preamp enabled`);
  assert.equal(result.metrics.deviceProfile.headroomAppliedToBands, false, `${file}: headroom altered PEQ gains`);
  assert.equal(result.metrics.outputLevelConverged, solved.metadata.outputLevelConverged, `${file}: convergence metadata mismatch`);
  assert.equal(result.metrics.targetLevelCompensationDb, solved.metadata.targetLevelCompensationDb, `${file}: target compensation metadata mismatch`);
  assert(result.metrics.outputLevel && Number.isFinite(result.metrics.outputLevel.outputLevelShiftDb), `${file}: output-level analysis missing`);

  const shifted = solved.workingTarget;
  assert.equal(shifted.length, original.length, `${file}: target point count changed`);
  for (let i = 1; i < original.length; i++) {
    const originalShape = original[i][1] - original[0][1];
    const shiftedShape = shifted[i][1] - shifted[0][1];
    assert(Math.abs(originalShape - shiftedShape) < 1e-9, `${file}: target shape changed at point ${i}`);
  }
}

console.log('Generic Moondrop target compensation regression PASS · targets:', targets.length);
