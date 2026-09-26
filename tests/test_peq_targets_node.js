/* Full production-data PEQ regression. Run with: node tests/test_peq_targets_node.js */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { performance } = require('node:perf_hooks');

const source = fs.readFileSync('app/pudding_peq.js', 'utf8');
assert(!source.includes('__EARPRINT_PURE_CURVE__'), 'PEQ solver must not read Pure EarPrint as a hidden target');
assert(!source.includes('farCandidateAnchors'), 'Fixed far-anchor candidate bank must not be present');

const sandbox = {
  window: {},
  document: { readyState: 'loading', addEventListener() {} },
  location: { pathname: '/', hostname: 'localhost' },
  console, setTimeout, clearTimeout, URL, Blob, performance,
  fetch() { throw new Error('not used'); }
};
sandbox.window = sandbox;
vm.runInNewContext(source, sandbox, { filename: 'app/pudding_peq.js' });
const api = sandbox.MoondropPuddingPEQ;

function parseCurve(text) {
  const rows = [];
  for (const line of text.split(/\r?\n/)) {
    const m = line.trim().match(/^([-+]?\d+(?:\.\d+)?)\s+(?:Hz\s+)?([-+]?\d+(?:\.\d+)?)/i);
    if (m) rows.push([Number(m[1]), Number(m[2])]);
  }
  return rows;
}

const raw = parseCurve(fs.readFileSync('input/original_711/moondrop pudding fr.txt', 'utf8'));
const targets = fs.readdirSync('output').filter(name => name.endsWith('__robust_target.txt')).sort();
assert(targets.length >= 1, 'At least one Robust Target is required');

const summary = [];
for (const file of targets) {
  const target = parseCurve(fs.readFileSync('output/' + file, 'utf8'));
  const start = performance.now();
  const result = api.optimize(raw, target);
  const elapsedMs = performance.now() - start;

  assert.equal(result.metrics.performanceMode, 'lm-irls', `${file}: LM/IRLS production path not used`);
  assert(!result.metrics.solverFallback, `${file}: unexpected legacy solver fallback`);
  assert(result.bands.length <= api.CFG.bands, `${file}: band limit exceeded`);
  assert(result.metrics.hfValidation && result.metrics.hfValidation.status === 'PASS', `${file}: HF safety failed`);
  assert(result.metrics.stabilityPerturbation && result.metrics.stabilityPerturbation.status === 'PASS', `${file}: perturbation stability failed`);
  assert(result.metrics.exactExportSimulation === true, `${file}: exact export simulation missing`);

  for (const b of result.bands) {
    assert(Number.isFinite(b.freq) && b.freq >= 20 && b.freq <= 12000, `${file}: invalid Fc`);
    assert(Number.isFinite(b.gain) && b.gain >= -12 && b.gain <= 3, `${file}: invalid gain`);
    assert(Number.isFinite(b.q) && b.q >= 0.3 && b.q <= 10, `${file}: invalid Q`);
    assert(api.__test.biquadSafety(b).stable, `${file}: unstable biquad`);
  }

  summary.push({
    target: file,
    ms: Number(elapsedMs.toFixed(1)),
    bands: result.bands.length,
    rmse: Number(result.metrics.rmseAfter.toFixed(4)),
    p95: Number(result.metrics.p95After.toFixed(4)),
    max: Number(result.metrics.maxAfter.toFixed(4)),
    maxQ: Number(result.metrics.maxQ.toFixed(2))
  });
}
console.log('Full Robust Target PEQ regression PASS');
console.table(summary);
