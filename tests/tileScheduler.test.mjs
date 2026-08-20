import test from 'node:test';
import assert from 'node:assert/strict';

import {
  DynamicTileController,
  fitTileDimension,
  waitForSubmittedGPUWork
} from '../lib/tileScheduler.js';

test('starts conservatively and respects the hard maximum', () => {
  assert.equal(new DynamicTileController(512).tileSize, 384);
  assert.equal(new DynamicTileController(320).tileSize, 320);
  assert.equal(fitTileDimension(512, 384), 384);
  assert.equal(fitTileDimension(300, 384), 304);
});

test('reduces slow tiles and grows fast tiles within configured bounds', () => {
  const controller = new DynamicTileController(512);

  assert.equal(controller.observe([30, 32, 34]), true);
  assert.equal(controller.tileSize, 256);
  assert.equal(controller.observe([40]), false);
  assert.equal(controller.tileSize, 256);

  assert.equal(controller.observe([4, 6, 8]), true);
  assert.equal(controller.tileSize, 384);
  assert.equal(controller.observe([5]), true);
  assert.equal(controller.tileSize, 512);
  assert.equal(controller.observe([5]), false);
});

test('uses the median and ignores invalid timings', () => {
  const controller = new DynamicTileController(512);

  assert.equal(controller.observe([1, 30, Number.NaN]), false);
  assert.equal(controller.tileSize, 384);
  assert.equal(controller.observe([Number.NaN, Number.POSITIVE_INFINITY]), false);
});

test('can restore fixed-size behavior', () => {
  const controller = new DynamicTileController(500, false);

  assert.equal(controller.tileSize, 496);
  assert.equal(controller.observe([100]), false);
  assert.equal(controller.tileSize, 496);
});

test('supports custom adaptive limits and timing targets', () => {
  const controller = new DynamicTileController(768, {
    minTileSize: 128,
    initialTileSize: 512,
    targetTileTimeMs: 24,
    adjustmentStep: 64
  });

  assert.equal(controller.tileSize, 512);
  controller.observe([40]);
  assert.equal(controller.tileSize, 448);
  controller.observe([8]);
  assert.equal(controller.tileSize, 512);
});

test('waits for submitted GPU work before continuing', async () => {
  let release;
  let completed = false;
  const pendingGPUWork = new Promise((resolve) => {
    release = resolve;
  });
  const queue = {
    onSubmittedWorkDone: () => pendingGPUWork
  };

  const wait = waitForSubmittedGPUWork(queue).then(() => {
    completed = true;
  });
  await Promise.resolve();
  assert.equal(completed, false);

  release();
  await wait;
  assert.equal(completed, true);
});

test('does not strand scheduling when the GPU wait rejects', async () => {
  await assert.doesNotReject(() =>
    waitForSubmittedGPUWork({
      onSubmittedWorkDone: () => Promise.reject(new Error('device lost'))
    })
  );
});
