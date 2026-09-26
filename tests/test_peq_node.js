/* Deterministic PEQ regression tests. Run with: node tests/test_peq_node.js */
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
sandbox.window = sandbox.window;
vm.runInNewContext(fs.readFileSync('app/pudding_peq.js', 'utf8'), sandbox, { filename: 'app/pudding_peq.js' });
const api = sandbox.window.MoondropPuddingPEQ;
const t = api.__test;
api.CFG.resolution = { r0Points: 48, r1Points: 24, r2Points: 12, tonalOctaves: 1 / 6 };
api.CFG.performanceMode = 'balanced';
api.CFG.bands = 3;
api.CFG.balancedPasses = 1;
api.CFG.balancedHuberPoints = 12;
api.CFG.huberEnabled = false;

const logspace = (a, b, n) => Array.from({ length: n }, (_, i) => Math.exp(Math.log(a) + (Math.log(b) - Math.log(a)) * i / (n - 1)));
const curve = (f, peak = 0, dip = 0) => f.map(x => [x, peak * Math.exp(-0.5 * Math.pow(Math.log2(x / 1000) / 0.8, 2)) - dip * Math.exp(-0.5 * Math.pow(Math.log2(x / 4500) / 0.08, 2))]);

assert.equal(t.DOMAIN_BANDS.length, 9);
assert.deepEqual(Array.from(t.DOMAIN_BANDS[0]), [20, 100]);
assert.deepEqual(Array.from(t.DOMAIN_BANDS[8]), [12000, 20000]);
assert.deepEqual(t.resolutionGrids(20, 20000).R0.length, api.CFG.resolution.r0Points);

assert.equal(t.deadZoneHuber(0.05), 0);
assert(t.deadZoneHuber(3) > t.deadZoneHuber(1));
assert(t.huberWeight(3) < 1 && t.huberWeight(0.2) === 1);
assert.deepEqual(Array.from(t.solveLinearSystem([[2,0],[0,4]],[4,8])), [2,2]);
const fs1 = t.featureAnalysis([100, 200, 400, 800, 1600], [0, 1, 3, 1, 0]);
assert(fs1.length >= 1 && fs1[0].freq === 400);
assert(t.boostRisk({ freq: 10000, gain: 2, widthOct: .05, support: .1, isolated: true }) > 0.5);

const raw = curve(logspace(20, 20000, 80), 2);
const target = curve(logspace(20, 20000, 80), 0);
const result = api.optimize(raw, target);
assert(result.bands.length <= 10);
for (const b of result.bands) {
  assert(Number.isFinite(b.freq) && b.freq >= 20 && b.freq <= 12000);
  assert(Number.isFinite(b.gain) && b.gain >= -12 && b.gain <= 3);
  assert(Number.isFinite(b.q) && b.q >= .3 && b.q <= 10);
}
assert.match(api.formatPEQ(result), /Filter/);
assert(!/PK 0 Hz/.test(api.formatPEQ(result)));

const hf = t.highFrequencyValidation(raw, result.bands);
assert(['PASS', 'FAIL', 'SKIP'].includes(hf.status));
assert(result.metrics.quantizationRescue === 'accepted' || result.metrics.quantizationRescue === 'rejected');
assert.equal(result.metrics.performanceMode, 'lm-irls');
assert(!result.metrics.solverFallback);
assert(result.metrics.solver && /Levenberg-Marquardt/.test(result.metrics.solver.name));
assert(result.metrics.stabilityPerturbation && ['PASS','WARN'].includes(result.metrics.stabilityPerturbation.status));
console.log('PEQ regression PASS', JSON.stringify({ bands: result.bands.length, rmse: result.metrics.rmseAfter, hf: result.metrics.hfValidation }));
