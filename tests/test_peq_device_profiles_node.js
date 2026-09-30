/* FR input and device post-processing regression tests. */
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
const appSource = fs.readFileSync('app/pudding_peq.js', 'utf8');
assert.match(appSource, /data-pudding-workflow="pudding"/);
assert.match(appSource, /data-pudding-workflow="walkplay"/);
assert.match(appSource, /5128\.\*DF\.\*Tilt/);
assert(!/id="puddingMaxBands"/.test(appSource));
assert(!/id="puddingMinGain"/.test(appSource));
assert(!/id="puddingMaxGain"/.test(appSource));

const cleanedBands = t.filterActiveMoondropBands([
  { freq: 100, gain: 2, q: 1 },
  { freq: 1000, gain: 0, q: 1 },
  { freq: 3000, gain: -1.5, q: 2 },
  { freq: 6000, gain: 0.04, q: 2 }
]);
assert.deepEqual(cleanedBands.map(b => b.gain), [2, -1.5]);
const boostLevel = t.analyzeOutputLevel([0, 0], [3, 3], [100, 1000]);
assert.equal(boostLevel.maxBoostDb, 3);
assert.equal(boostLevel.maxCutDb, 3);
const cutLevel = t.analyzeOutputLevel([0, 0, 0, 0], [-3, -2, -1, 0.5], [100, 1000, 3000, 10000]);
assert(cutLevel.meanDeltaDb < 0);
assert.notEqual(cutLevel.meanDeltaDb, (-3 - 2 - 1 + 0.5));
const flatLevel = t.analyzeOutputLevel([0, 0], [0, 0], [100, 1000]);
assert.equal(flatLevel.integratedWeightedLevelShiftDb, 0);
const conflictLevel = t.moondropOutputLevelMetadata(cutLevel, { maxBoostDb: 2, headroomRequiredDb: -2 });
assert.equal(conflictLevel.externalVolumeAdjustmentDb, null);
assert.equal(conflictLevel.externalVolumeSafe, false);
const flatMetadata = t.moondropOutputLevelMetadata(flatLevel, { maxBoostDb: 0, headroomRequiredDb: 0 });
assert.equal(flatMetadata.externalVolumeAdjustmentDb, 0);

const csv = 'Frequency (Hz),Amplitude (dB)\n20,-1\n100,0\n1000,3\n20000,-2\n';
const parsedCsv = t.parseCustomFR(csv, 'custom.csv');
assert.equal(parsedCsv.source, 'custom.csv');
assert(parsedCsv.curve.length >= 240);
assert.equal(parsedCsv.curve[0][0], 20);
assert.equal(parsedCsv.curve.at(-1)[0], 20000);
for (let i = 1; i < parsedCsv.curve.length; i++) assert(parsedCsv.curve[i][0] > parsedCsv.curve[i - 1][0]);

const parsedJson = t.parseCustomFR(JSON.stringify({ points: [
  { frequency: 20, amplitude: -2 },
  { frequency: 1000, amplitude: 1.5 },
  { frequency: 20000, amplitude: -1 }
] }), 'custom.json');
assert(parsedJson.interpolatedPoints > 0);
const sorted = t.parseCustomFR('20,0\n100,1\n50,2', 'sorted.csv');
assert.equal(sorted.sorted, true);
assert.equal(sorted.removedPoints, 0);
const overshoot = t.parseCustomFR('20,0\n100,1\n20000,-1\n20186,-2', 'overshoot.txt');
assert.equal(overshoot.originalPoints, 4);
assert.equal(overshoot.removedPoints, 1);
assert.equal(overshoot.removedReasons.above20kHz, 1);
assert(overshoot.curve.at(-1)[0] <= 20000);
const builtIn = t.parseCustomFR(fs.readFileSync('input/original_711/moondrop pudding fr.txt', 'utf8'), 'moondrop pudding fr.txt', { walkplay: true });
assert.equal(builtIn.removedPoints, 1);
assert(builtIn.curve.at(-1)[0] <= 20000);
const damagedJson = t.parseCustomFR(JSON.stringify({ points: [null, { frequency: 20, amplitude: 0 }, { frequency: 1000, amplitude: 1 }, { frequency: 1000, amplitude: 1.2 }, { frequency: 21000, amplitude: 2 }] }), 'damaged.json');
assert.equal(damagedJson.removedReasons.invalid, 1);
assert.equal(damagedJson.removedReasons.above20kHz, 1);
assert.equal(damagedJson.removedReasons.duplicate, 1);
assert.throws(() => t.parseCustomFR('20,NaN\n100,1', 'bad.txt'), /at least 2/);

const frequencies = Array.from({ length: 64 }, (_, i) => 20 * Math.pow(1000, i / 63));
const raw = frequencies.map(f => [f, 0]);
const target = frequencies.map(f => [f, 0]);
const solverResult = {
  bands: [
    { freq: 100, gain: 1.234, q: 0.456 },
    { freq: 1000, gain: 4.567, q: 1.234 },
    { freq: 3000, gain: -2.345, q: 6.789 },
    { freq: 6000, gain: 0.456, q: 2.345 },
    { freq: 9000, gain: 0.789, q: 1.876 },
    { freq: 12000, gain: -0.654, q: 2.222 },
    { freq: 15000, gain: 0.333, q: 1.111 },
    { freq: 18000, gain: -0.222, q: 1.222 },
    { freq: 19000, gain: 0.111, q: 1.333 }
  ],
  metrics: { levelOffsetDb: 0, solverConstraints: { maxBands: 10, minGain: -12, maxGain: 10 } }
};

const generic = t.applyDeviceProfile(solverResult, raw, target, 'generic');
assert.equal(generic.bands.length, 9);
assert.equal(generic.metrics.deviceProfile.preampSupported, true);
assert.equal(generic.bands[1].gain, 4.567);

const moondrop = t.applyDeviceProfile(solverResult, raw, target, 'moondrop');
assert(moondrop.bands.every(b => Math.abs(b.gain * 10 - Math.round(b.gain * 10)) < 1e-9));
assert(moondrop.bands.every(b => Math.abs(b.q * 100 - Math.round(b.q * 100)) < 1e-9));
assert(Math.abs(moondrop.metrics.deviceProfile.headroomBefore.compensationDb + moondrop.metrics.outputLevel.outputLevel.maxBoostDb) < 1e-9);
assert.equal(moondrop.metrics.deviceProfile.headroomAppliedToBands, false);
assert(Math.abs(moondrop.bands.find(b => b.freq === 100).gain - 1.2) < 1e-9);
assert.equal(moondrop.bands.find(b => b.freq === 1000).gain, 3);
assert(moondrop.bands.every(b => Math.abs(b.gain) >= 0.05));
assert(moondrop.metrics.outputLevel);
assert.equal(moondrop.metrics.outputLevel.headroomRequiredDb, moondrop.metrics.deviceProfile.headroomBefore.headroomRequiredDb);
assert.equal(JSON.parse(api.jsonPEQ(moondrop, {})).peq.length, moondrop.bands.length);
const moondropText = api.formatPEQ(moondrop);
assert.equal((moondropText.match(/^Filter /gm)||[]).length, moondrop.bands.length);
assert(!moondropText.includes('0.0 dB'));
assert(!moondropText.includes('Preamp:'));
assert(!moondropText.includes('Headroom compensation:'));

const inactiveResult = t.applyDeviceProfile({...solverResult,bands:solverResult.bands.concat({freq:500,gain:0.02,q:1.2})}, raw, target, 'moondrop');
assert(!inactiveResult.bands.some(b => b.freq === 500));
assert(inactiveResult.metrics.deviceProfile.removedInactiveBands.some(b => b.frequency === 500));

const walkplay = t.applyDeviceProfile(solverResult, raw, target, 'walkplay', { walkplayBands: 8 });
assert.equal(walkplay.bands.length, 8);
assert(walkplay.bands.every(b => b.gain >= -10 && b.gain <= 10 && b.q >= 0.1 && b.q <= 5));
assert.equal(walkplay.metrics.deviceProfile.headroom.status, 'WARN');
assert.equal(walkplay.metrics.deviceProfile.dacVolumeDb, null);
assert(Math.abs(walkplay.metrics.deviceProfile.dacRecommendation.recommendedDb + 5.6) < 1e-9);
const walkplay10 = t.applyDeviceProfile(solverResult, raw, target, 'walkplay', { walkplayBands: 10 });
assert.equal(walkplay10.bands.length, 9);

const puddingFR = t.parseCustomFR(fs.readFileSync('input/original_711/moondrop pudding fr.txt', 'utf8'), 'moondrop pudding fr.txt');
const iefTarget = t.parseCustomFR(fs.readFileSync('output/IEF2025__robust_target.txt', 'utf8'), 'IEF2025__robust_target.txt');
const iefSolver = api.optimize(puddingFR.curve, iefTarget.curve, { maxBands: 10, minGain: -12, maxGain: 3 });
const iefPudding = t.applyDeviceProfile(iefSolver, puddingFR.curve, iefTarget.curve, 'moondrop');
assert(iefPudding.bands.length <= 10);
assert(iefPudding.bands.every(b => b.gain >= -12 && b.gain <= 3 && b.q >= 0.3 && b.q <= 10 && Math.abs(b.gain) >= 0.05));
assert(iefPudding.metrics.shapeGuardAccepted !== false);
assert(iefPudding.metrics.rmseAfter <= iefSolver.metrics.rmseAfter + 0.05);
console.log('IEF2025 Pudding adapter', JSON.stringify({solverBands:iefSolver.bands.length,finalBands:iefPudding.bands.length,solverRmse:iefSolver.metrics.rmseAfter,finalRmse:iefPudding.metrics.rmseAfter,outputLevel:iefPudding.metrics.outputLevel}));
const iefCompensated = t.solvePuddingWithTargetCompensation(puddingFR.curve, iefTarget.curve, { solverConstraints: { maxBands: 10, minGain: -12, maxGain: 3 } });
console.log('IEF2025 target compensation', JSON.stringify(iefCompensated.metadata), 'finalShift=', iefCompensated.result.metrics.outputLevel.outputLevelShiftDb);

console.log('PEQ FR input and device profiles PASS');
