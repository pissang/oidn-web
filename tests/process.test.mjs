import assert from 'node:assert/strict';
import test from 'node:test';
import {
  hdrTransferFuncCPU,
  hdrTransferFuncInverseCPU
} from '../lib/hdrTransfer.js';

const logXMax = Math.log(65505);
const logNormScale = 1 / logXMax;

test('Log HDR transfer matches the RTLightmap formula and round-trips', () => {
  const source = new Float32Array([
    0, 1, 5, 1,
    0.25, 20, 100, 1
  ]);
  const inputScale = 0.37;
  const encoded = hdrTransferFuncCPU({
    data: source,
    channels: 4,
    inputScale,
    transfer: 'log'
  });

  assert.equal(encoded[0], 0);
  assert.ok(Math.abs(encoded[1] - Math.log1p(inputScale) * logNormScale) < 1e-7);
  assert.ok(Math.abs(encoded[2] - Math.log1p(5 * inputScale) * logNormScale) < 1e-7);
  assert.equal(encoded[3], 1);

  const decoded = hdrTransferFuncInverseCPU({
    data: encoded,
    channels: 4,
    inputScale,
    transfer: 'log'
  });
  for (const index of [0, 1, 2, 4, 5, 6]) {
    assert.ok(Math.abs(decoded[index] - source[index]) < 2e-5,
      `channel ${index}: ${decoded[index]} vs ${source[index]}`);
  }
  assert.equal(decoded[3], 1);
  assert.equal(decoded[7], 1);
});

test('PU remains the default transfer', () => {
  const source = new Float32Array([0.5, 2, 10, 1]);
  const implicit = hdrTransferFuncCPU({ data: source, channels: 4, inputScale: 1 });
  const explicit = hdrTransferFuncCPU({ data: source, channels: 4, inputScale: 1, transfer: 'pu' });
  assert.deepEqual(implicit, explicit);
  assert.notEqual(implicit[1], hdrTransferFuncCPU({
    data: source,
    channels: 4,
    inputScale: 1,
    transfer: 'log'
  })[1]);
});

test('Log normalization uses the finite half-float HDR maximum', () => {
  const encoded = hdrTransferFuncCPU({
    data: new Float32Array([65504, 65504, 65504, 1]),
    channels: 4,
    inputScale: 1,
    transfer: 'log'
  });
  assert.ok(Math.abs(encoded[0] - 1) < 1e-7);
  const decoded = hdrTransferFuncInverseCPU({
    data: encoded,
    channels: 4,
    inputScale: 1,
    transfer: 'log'
  });
  assert.ok(Math.abs(decoded[0] - 65504) < 1e-3);
});
