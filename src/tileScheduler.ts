export interface DynamicTileOptions {
  /** Smallest output tile edge. Defaults to 256. */
  minTileSize?: number;
  /** Initial maximum output tile edge. Defaults to 432. */
  initialTileSize?: number;
  /** Desired upper bound for one tile's GPU latency. Defaults to 16 ms. */
  targetTileTimeMs?: number;
  /** Amount added or removed after each completed execution. Defaults to 16. */
  adjustmentStep?: number;
}

export type DynamicTileSetting = boolean | DynamicTileOptions;

const defaultMinTileSize = 256;
const defaultInitialTileSize = 432;
const defaultTargetTileTimeMs = 16;
const defaultAdjustmentStep = 16;
export const OIDN_TILE_ALIGNMENT = 16;

export interface TileRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PlannedTile {
  column: number;
  row: number;
  input: TileRect;
  output: TileRect;
}

export interface TilePlan {
  columns: number;
  rows: number;
  overlap: number;
  maxOutputWidth: number;
  maxOutputHeight: number;
  inputPixelCount: number;
  inputShapeCount: number;
  tiles: PlannedTile[];
}

function alignUp(value: number, alignment: number) {
  return Math.ceil(value / alignment) * alignment;
}

function alignDown(value: number, alignment: number) {
  return Math.floor(value / alignment) * alignment;
}

function partitionRange(length: number, parts: number, index: number) {
  const start = Math.round(index * length / parts);
  const end = Math.round((index + 1) * length / parts);
  return { start, end };
}

function alignRangeWithinImage(
  start: number,
  end: number,
  imageSize: number,
  minimumSize = 0
) {
  const currentSize = end - start;
  const alignedSize = Math.min(
    imageSize,
    Math.max(
      minimumSize,
      alignUp(currentSize, OIDN_TILE_ALIGNMENT)
    )
  );
  let alignedStart = start - Math.floor((alignedSize - currentSize) / 2);
  alignedStart = clamp(alignedStart, 0, imageSize - alignedSize);
  return { start: alignedStart, end: alignedStart + alignedSize };
}

/**
 * Splits an image into balanced rectangular output regions. Only sides that
 * touch another tile receive a halo; image edges rely on the model's normal
 * boundary behavior instead of doing redundant off-edge work.
 */
export function planTileGrid(
  width: number,
  height: number,
  maxTileSize: number,
  overlap: number
): TilePlan {
  if (!Number.isInteger(width) || width <= 0 ||
      !Number.isInteger(height) || height <= 0) {
    throw new Error('Tile grid dimensions must be positive integers');
  }
  if (!Number.isFinite(maxTileSize) || maxTileSize <= 0) {
    throw new Error('Maximum tile size must be positive');
  }
  if (!Number.isFinite(overlap) || overlap < 0) {
    throw new Error('Tile overlap must be non-negative');
  }

  const alignedMaximum = Math.max(
    OIDN_TILE_ALIGNMENT,
    alignDown(maxTileSize, OIDN_TILE_ALIGNMENT)
  );
  const alignedOverlap = alignUp(overlap, OIDN_TILE_ALIGNMENT);
  const columns = Math.max(1, Math.ceil(width / alignedMaximum));
  const rows = Math.max(1, Math.ceil(height / alignedMaximum));
  const tiles: PlannedTile[] = [];
  let maxOutputWidth = 0;
  let maxOutputHeight = 0;

  for (let row = 0; row < rows; row++) {
    const outputY = partitionRange(height, rows, row);
    for (let column = 0; column < columns; column++) {
      const outputX = partitionRange(width, columns, column);
      const desiredInputX = {
        start: column === 0
          ? outputX.start
          : Math.max(0, outputX.start - alignedOverlap),
        end: column === columns - 1
          ? outputX.end
          : Math.min(width, outputX.end + alignedOverlap)
      };
      const desiredInputY = {
        start: row === 0
          ? outputY.start
          : Math.max(0, outputY.start - alignedOverlap),
        end: row === rows - 1
          ? outputY.end
          : Math.min(height, outputY.end + alignedOverlap)
      };
      const inputX = alignRangeWithinImage(
        desiredInputX.start,
        desiredInputX.end,
        width
      );
      const inputY = alignRangeWithinImage(
        desiredInputY.start,
        desiredInputY.end,
        height
      );
      const output: TileRect = {
        x: outputX.start,
        y: outputY.start,
        width: outputX.end - outputX.start,
        height: outputY.end - outputY.start
      };
      const input: TileRect = {
        x: inputX.start,
        y: inputY.start,
        width: inputX.end - inputX.start,
        height: inputY.end - inputY.start
      };
      maxOutputWidth = Math.max(maxOutputWidth, output.width);
      maxOutputHeight = Math.max(maxOutputHeight, output.height);
      tiles.push({ column, row, input, output });
    }
  }

  const inputShapes = () => new Set(
    tiles.map(({ input }) => `${input.width}x${input.height}`)
  );
  if (inputShapes().size > 2) {
    const maximumInputWidth = Math.max(
      ...tiles.map(({ input }) => input.width)
    );
    const maximumInputHeight = Math.max(
      ...tiles.map(({ input }) => input.height)
    );
    const uniformWidthPixelCount = tiles.reduce(
      (sum, { input }) => sum + maximumInputWidth * input.height,
      0
    );
    const uniformHeightPixelCount = tiles.reduce(
      (sum, { input }) => sum + input.width * maximumInputHeight,
      0
    );
    if (uniformWidthPixelCount <= uniformHeightPixelCount) {
      for (const tile of tiles) {
        const range = alignRangeWithinImage(
          tile.input.x,
          tile.input.x + tile.input.width,
          width,
          maximumInputWidth
        );
        tile.input.x = range.start;
        tile.input.width = range.end - range.start;
      }
    } else {
      for (const tile of tiles) {
        const range = alignRangeWithinImage(
          tile.input.y,
          tile.input.y + tile.input.height,
          height,
          maximumInputHeight
        );
        tile.input.y = range.start;
        tile.input.height = range.end - range.start;
      }
    }
  }

  const finalInputShapes = inputShapes();
  const inputPixelCount = tiles.reduce(
    (sum, { input }) => sum + input.width * input.height,
    0
  );

  return {
    columns,
    rows,
    overlap: alignedOverlap,
    maxOutputWidth,
    maxOutputHeight,
    inputPixelCount,
    inputShapeCount: finalInputShapes.size,
    tiles
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function percentile(values: number[], fraction: number) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.ceil(sorted.length * fraction) - 1];
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
  private _smoothedTileTimeMs?: number;
  private _completeExecutionsSinceChange = 0;

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

    // The first tile may still pay one-time GPU residency costs even after
    // shape preparation. It should not make a stable layout look too slow.
    const representativeTimes = validTimes.length >= 3
      ? validTimes.slice(1)
      : validTimes;
    const observedTime = percentile(representativeTimes, 0.75);
    this._smoothedTileTimeMs = this._smoothedTileTimeMs === undefined
      ? observedTime
      : this._smoothedTileTimeMs * 0.65 + observedTime * 0.35;
    this._completeExecutionsSinceChange++;
    if (this._completeExecutionsSinceChange < 2) return false;

    let nextTileSize = this._tileSize;
    if (this._smoothedTileTimeMs > this.targetTileTimeMs * 1.25) {
      nextTileSize -= this._adjustmentStep;
    } else if (this._smoothedTileTimeMs < this.targetTileTimeMs * 0.65) {
      nextTileSize += this._adjustmentStep;
    }

    nextTileSize = clamp(
      alignUp(nextTileSize, OIDN_TILE_ALIGNMENT),
      this.minTileSize,
      this.maxTileSize
    );
    if (nextTileSize === this._tileSize) return false;
    this._tileSize = nextTileSize;
    this._smoothedTileTimeMs = undefined;
    this._completeExecutionsSinceChange = 0;
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
