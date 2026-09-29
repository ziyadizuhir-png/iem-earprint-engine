/* ISO226 weighting-layer regression. Run with: node tests/test_equal_loudness_node.js */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const sandbox = { window: {}, globalThis: {}, console };
vm.runInNewContext(fs.readFileSync('app/equal_loudness.js', 'utf8'), sandbox);
const elc = sandbox.window.IEMEqualLoudness;
assert.equal(elc.standard, 'ISO226');
for (const phon of [40, 60, 80]) {
  for (const f of [20, 1000, 12000]) {
    assert(Number.isFinite(elc.contourAt(f, phon)));
    assert(Number.isFinite(elc.optimizerWeight(f, phon, 0.5)));
  }
  assert(Math.abs(elc.relativeContourDb(1000, phon)) < 1e-12);
}
assert.equal(elc.optimizerWeight(1000, 60, 0.5), 1);
assert.notEqual(elc.optimizerWeight(100, 60, 0.5), 1);
console.log('ISO226 weighting regression PASS');
