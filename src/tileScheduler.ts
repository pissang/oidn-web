export interface DynamicTileOptions {
  /** Smallest output tile edge. Defaults to 256. */
  minTileSize?: number;
  /** Initial output tile edge. Defaults to 384. */
  initialTileSize?: number;
  /** Desired upper bound for one tile's GPU latency. Defaults to 16 ms. */
  targetTileTimeMs?: number;
  /** Amount added or removed after each completed execution. Defaults to 128. */
  adjustmentStep?: number;
}

export type DynamicTileSetting = boolean | DynamicTileOptions;

const defaultMinTileSize = 256;
const defaultInitialTileSize = 384;
const defaultTargetTileTimeMs = 16;
const defaultAdjustmentStep = 128;
export const OIDN_TILE_ALIGNMENT = 16;

function alignUp(value: number, alignment: number) {
  return Math.ceil(value / alignment) * alignment;
}

function alignDown(value: number, alignment: number) {
  return Math.floor(value / alignment) * alignment;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Fits one output dimension without exceeding the configured tile cap. */
export function fitTileDimension(imageSize: number, maxTileSize: number) {
  return imageSize <= maxTileSize
    ? Math.min(alignUp(imageSize, OIDN_TILE_ALIGNMENT), maxTileSize)
    : maxTileSize;
}

/**
 * Chooses an output tile size from completed GPU timings. A size change is
 * applied to the next execution so one in-flight tiled pass keeps a stable
 * overlap, allocation plan, and pipeline set.
 */
export class DynamicTileController {
  readonly enabled: boolean;
  readonly maxTileSize: number;
  readonly minTileSize: number;
  readonly targetTileTimeMs: number;

  private _tileSize: number;
  private _adjustmentStep: number;

  constructor(maxTileSize: number, setting: DynamicTileSetting = true) {
    const options = typeof setting === 'object' ? setting : {};
    this.enabled = setting !== false;
    this.maxTileSize = Math.max(
      OIDN_TILE_ALIGNMENT,
      alignDown(maxTileSize, OIDN_TILE_ALIGNMENT)
    );
    this.minTileSize = clamp(
      alignUp(options.minTileSize ?? defaultMinTileSize, OIDN_TILE_ALIGNMENT),
      OIDN_TILE_ALIGNMENT,
      this.maxTileSize
    );
    this.targetTileTimeMs = Math.max(
      1,
      options.targetTileTimeMs ?? defaultTargetTileTimeMs
    );
    this._adjustmentStep = Math.max(
      OIDN_TILE_ALIGNMENT,
      alignUp(
        options.adjustmentStep ?? defaultAdjustmentStep,
        OIDN_TILE_ALIGNMENT
      )
    );
    this._tileSize = this.enabled
      ? clamp(
          alignUp(
            options.initialTileSize ?? defaultInitialTileSize,
            OIDN_TILE_ALIGNMENT
          ),
          this.minTileSize,
          this.maxTileSize
        )
      : this.maxTileSize;
  }

  get tileSize() {
    return this._tileSize;
  }

  /** Returns true when the next execution should use a different tile size. */
  observe(tileTimesMs: number[]) {
    if (!this.enabled || tileTimesMs.length === 0) return false;

    const validTimes = tileTimesMs.filter(
      (duration) => Number.isFinite(duration) && duration >= 0
    );
    if (validTimes.length === 0) return false;

    const observedTime = median(validTimes);
    let nextTileSize = this._tileSize;
    if (observedTime > this.targetTileTimeMs * 1.25) {
      nextTileSize -= this._adjustmentStep;
    } else if (observedTime < this.targetTileTimeMs * 0.65) {
      nextTileSize += this._adjustmentStep;
    }

    nextTileSize = clamp(
      alignUp(nextTileSize, OIDN_TILE_ALIGNMENT),
      this.minTileSize,
      this.maxTileSize
    );
    if (nextTileSize === this._tileSize) return false;
    this._tileSize = nextTileSize;
    return true;
  }
}

/** Waits for all work submitted before this call and tolerates device loss. */
export async function waitForSubmittedGPUWork(
  queue: Pick<GPUQueue, 'onSubmittedWorkDone'>
) {
  try {
    await queue.onSubmittedWorkDone();
  } catch {
    // tileExecute has no error callback. Preserve its completion/cancellation
    // behavior and let the owning GPUDevice report device loss separately.
  }
}
