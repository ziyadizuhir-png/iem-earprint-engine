/* Runtime solver-constraint regression tests. Run with: node tests/test_peq_constraints_node.js */
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
api.CFG.resolution = { r0Points: 48, r1Points: 24, r2Points: 12, tonalOctaves: 1 / 6 };
api.CFG.solver = { ...api.CFG.solver, lmPoints: 36, maxIterations: 4, irlsIterations: 1, candidateLimit: 12 };
api.CFG.huberEnabled = false;

assert.deepEqual(Array.from(api.getAllowedGrowth(1)), [1]);
assert.deepEqual(Array.from(api.getAllowedGrowth(3)), [3]);
assert.deepEqual(Array.from(api.getAllowedGrowth(4)), [3, 4]);
assert.deepEqual(Array.from(api.getAllowedGrowth(6)), [3, 5, 6]);
assert.deepEqual(Array.from(api.getAllowedGrowth(7)), [3, 5, 7]);
assert.deepEqual(Array.from(api.getAllowedGrowth(10)), [3, 5, 7, 10]);

for (const invalid of [
  { maxBands: 0, minGain: -12, maxGain: 3 },
  { maxBands: 11, minGain: -12, maxGain: 3 },
  { maxBands: 4, minGain: 4, maxGain: 3 },
  { maxBands: 4, minGain: -13, maxGain: 3 },
  { maxBands: 4, minGain: -6, maxGain: 11 }
]) assert.throws(() => api.normalizeSolverConstraints(invalid));

const frequencies = Array.from({ length: 48 }, (_, i) => 20 * Math.pow(1000, i / 47));
const raw = frequencies.map(f => [f, 1.8 * Math.exp(-0.5 * Math.pow(Math.log2(f / 900), 2))]);
const target = frequencies.map(f => [f, 0.4 * Math.exp(-0.5 * Math.pow(Math.log2(f / 900), 2)) - 1.1 * Math.exp(-0.5 * Math.pow(Math.log2(f / 4200) / 0.18, 2))]);

for (const maxBands of [1, 3, 5, 10]) {
  const constraints = { maxBands, minGain: -12, maxGain: 3 };
  const result = api.solvePEQ({ curve: raw }, { curve: target }, constraints);
  assert(result.bands.length <= maxBands, `band ceiling exceeded for ${maxBands}`);
  assert.equal(JSON.stringify(result.metrics.solverConstraints), JSON.stringify(constraints));
  assert.equal(result.metrics.solverConfigurationHash, api.__test.solverConfigurationHash(constraints));
  for (const band of result.bands) assert(band.gain >= -12 && band.gain <= 3);
}

for (const constraints of [
  { maxBands: 5, minGain: -12, maxGain: 3 },
  { maxBands: 5, minGain: -6, maxGain: 2 },
  { maxBands: 5, minGain: -3, maxGain: 1 },
  { maxBands: 5, minGain: -12, maxGain: 10 },
  { maxBands: 5, minGain: 0, maxGain: 10 }
]) {
  const result = api.solvePEQ(raw, target, constraints);
  for (const band of result.bands) {
    assert(band.gain >= constraints.minGain, `${JSON.stringify(constraints)} minimum gain violated`);
    assert(band.gain <= constraints.maxGain, `${JSON.stringify(constraints)} maximum gain violated`);
  }
}

const first = api.solvePEQ(raw, target, { maxBands: 4, minGain: -6, maxGain: 2 });
const second = api.solvePEQ(raw, target, { maxBands: 4, minGain: -6, maxGain: 2 });
assert.equal(JSON.stringify(first.bands), JSON.stringify(second.bands), 'same input and constraints must be deterministic');
assert.notEqual(first.metrics.solverConfigurationHash, api.__test.solverConfigurationHash({ maxBands: 10, minGain: -12, maxGain: 3 }));
assert.equal(first.metrics.constraintValidation.bands, true);
assert.equal(first.metrics.constraintValidation.gains, true);
assert(first.metrics.diagnostics && first.metrics.diagnostics.worstError);

console.log('PEQ runtime constraints PASS', JSON.stringify({ hash: first.metrics.solverConfigurationHash, bands: first.bands.length }));
