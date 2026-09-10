import test from 'node:test';
import assert from 'node:assert/strict';

import { planTileGrid } from '../lib/tileScheduler.js';

function assertExactCoverage(plan, width, height) {
  const coverage = new Uint8Array(width * height);
  for (const tile of plan.tiles) {
    const { input, output } = tile;
    assert.ok(input.x >= 0 && input.y >= 0);
    assert.ok(input.x + input.width <= width);
    assert.ok(input.y + input.height <= height);
    assert.ok(output.x >= input.x && output.y >= input.y);
    assert.ok(output.x + output.width <= input.x + input.width);
    assert.ok(output.y + output.height <= input.y + input.height);
    for (let y = output.y; y < output.y + output.height; y++) {
      for (let x = output.x; x < output.x + output.width; x++) {
        coverage[y * width + x]++;
      }
    }
  }
  assert.ok(coverage.every((count) => count === 1));
}

test('plans a balanced boundary-aware 3 by 2 grid for 720p', () => {
  const plan = planTileGrid(1280, 720, 432, 96);

  assert.equal(plan.columns, 3);
  assert.equal(plan.rows, 2);
  assert.equal(plan.tiles.length, 6);
  assert.equal(plan.overlap, 96);
  assert.equal(plan.inputShapeCount, 2);
  assert.equal(plan.inputPixelCount, 1_559_040);
  assert.deepEqual(
    plan.tiles.map((tile) => tile.output),
    [
      { x: 0, y: 0, width: 427, height: 360 },
      { x: 427, y: 0, width: 426, height: 360 },
      { x: 853, y: 0, width: 427, height: 360 },
      { x: 0, y: 360, width: 427, height: 360 },
      { x: 427, y: 360, width: 426, height: 360 },
      { x: 853, y: 360, width: 427, height: 360 }
    ]
  );
  assert.deepEqual(
    plan.tiles.map((tile) => tile.input),
    [
      { x: 0, y: 0, width: 528, height: 464 },
      { x: 328, y: 0, width: 624, height: 464 },
      { x: 752, y: 0, width: 528, height: 464 },
      { x: 0, y: 256, width: 528, height: 464 },
      { x: 328, y: 256, width: 624, height: 464 },
      { x: 752, y: 256, width: 528, height: 464 }
    ]
  );
  assertExactCoverage(plan, 1280, 720);
});

test('uses overlap only on sides shared with another tile', () => {
  const plan = planTileGrid(1280, 720, 432, 80);
  const [topLeft, topMiddle, topRight] = plan.tiles;

  assert.equal(topLeft.input.x, 0);
  assert.equal(topLeft.input.y, 0);
  assert.ok(topLeft.input.width >= topLeft.output.width + 80);
  assert.ok(topMiddle.output.x - topMiddle.input.x >= 80);
  assert.ok(
    topMiddle.input.x + topMiddle.input.width -
      (topMiddle.output.x + topMiddle.output.width) >= 80
  );
  assert.equal(topRight.input.x + topRight.input.width, 1280);
  assert.equal(plan.tiles.at(-1).input.y + plan.tiles.at(-1).input.height, 720);
  assertExactCoverage(plan, 1280, 720);
});

test('keeps tiled model input shapes aligned for uneven image dimensions', () => {
  const plan = planTileGrid(1300, 721, 432, 80);

  assert.ok(plan.tiles.length > 1);
  assert.ok(plan.inputShapeCount <= 2);
  for (const tile of plan.tiles) {
    assert.equal(tile.input.width % 16, 0);
    assert.equal(tile.input.height % 16, 0);
  }
  assertExactCoverage(plan, 1300, 721);
});

test('buckets common 1080p boundary shapes without exceeding the cache', () => {
  const plan = planTileGrid(1920, 1080, 432, 96);

  assert.equal(plan.columns, 5);
  assert.equal(plan.rows, 3);
  assert.equal(plan.inputShapeCount, 2);
  assert.equal(plan.inputPixelCount, 4_285_440);
  assert.deepEqual(
    [...new Set(plan.tiles.map(({ input }) => `${input.width}x${input.height}`))],
    ['576x464', '576x560']
  );
  assertExactCoverage(plan, 1920, 1080);
});

test('validates planner inputs', () => {
  assert.throws(() => planTileGrid(0, 720, 432, 96));
  assert.throws(() => planTileGrid(1280, -1, 432, 96));
  assert.throws(() => planTileGrid(1280, 720, 0, 96));
  assert.throws(() => planTileGrid(1280, 720, 432, -1));
});
