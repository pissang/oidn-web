import { HostTensor } from './tza';
import {
  GPUDataProcess,
  Tile,
  avgLogLum,
  hdrTransferFuncCPU,
  hdrTransferFuncInverseCPU
} from './process';
import {
  DynamicTileController,
  type DynamicTileSetting,
  fitTileDimension,
  OIDN_TILE_ALIGNMENT,
  waitForSubmittedGPUWork
} from './tileScheduler';
import {
  detectUNetModelSpec,
  validateUNetModel,
  type UNetModelSpec
} from './modelSpec';
import {
  NativeUNetExecutor,
  type NativeUNetKernelSetting,
  type NativeUNetPrecisionSetting
} from './nativeUNet';
import { WebNNUNetExecutor } from './webnnUNet';

export type UNetEngineSetting = 'auto' | 'wgsl' | 'webnn';

interface HDRImageData {
  data: Float32Array;
  width: number;
  height: number;
}

interface GPUImageData {
  data: GPUBuffer | GPUTexture;
  width: number;
  height: number;
}

interface GPUImageDataOutput {
  data: GPUBuffer;
  width: number;
  height: number;
}

export interface UNetExecutionStats {
  width: number;
  height: number;
  tileWidth: number;
  tileHeight: number;
  tileCount: number;
  durationMs: number;
  tileTimeMs: {
    min: number;
    median: number;
    mean: number;
    max: number;
  };
}

function roundUp(a: number, b: number) {
  return Math.ceil(a / b) * b;
}

function isGPUImageData(
  data: ImageData | GPUImageData | HDRImageData
): data is GPUImageData {
  return data.data instanceof GPUBuffer || data.data instanceof GPUTexture;
}

class UNet {
  private _device: GPUDevice | undefined;

  // TODO calculate the tile size from memory size
  // https://github.com/RenderKit/oidn/blob/713ec7838ba650f99e0a896549c0dca5eeb3652d/core/unet_filter.cpp#L287
  private _tileWidth = 0;
  private _tileHeight = 0;

  private _tileOverlapX = 0;
  private _tileOverlapY = 0;

  private _aux;
  private _hdr;

  private _dataProcessGPU?: GPUDataProcess;
  private _nativeExecutor?: NativeUNetExecutor;
  private _webNNExecutor?: WebNNUNetExecutor;
  private _modelSpec: UNetModelSpec;
  private _inputChannels: number;
  private _engine: UNetEngineSetting;

  private _dynamicTileController: DynamicTileController;
  private _lastExecution?: UNetExecutionStats;

  constructor(
    hostTensors: Map<string, HostTensor>,
    backend: { device: GPUDevice; adapterInfo: GPUAdapterInfo },
    opts: {
      /**
       * If use auxiliary data.
       */
      aux?: boolean;
      /**
       * If input is HDR image.
       */
      hdr?: boolean;
      maxTileSize?: number;
      dynamicTile?: DynamicTileSetting;
      /** Reserved for explicit native WGSL selection. */
      engine?: UNetEngineSetting;
      /** Arithmetic/storage precision used by the native WGSL engine. */
      precision?: NativeUNetPrecisionSetting;
      /** Model-independent convolution kernel selection. */
      kernel?: NativeUNetKernelSetting;
      /** Explicit descriptor for a new OIDN topology not in the built-in registry. */
      modelSpec?: UNetModelSpec;
    } = {}
  ) {
    this._aux = opts.aux || false;
    this._hdr = opts.hdr || false;
    this._engine = opts.engine ?? 'auto';
    const modelSpec = opts.modelSpec ?? detectUNetModelSpec(hostTensors);
    const validatedModel = validateUNetModel(hostTensors, modelSpec);
    this._modelSpec = validatedModel.spec;
    this._inputChannels = validatedModel.inputChannels;

    const expectedInputChannels = this._aux ? 9 : 3;
    if (validatedModel.inputChannels !== expectedInputChannels) {
      throw new Error(
        `OIDN model expects ${validatedModel.inputChannels} input channels, ` +
          `but aux=${this._aux} provides ${expectedInputChannels}`
      );
    }

    this._dynamicTileController = new DynamicTileController(
      opts.maxTileSize ?? 512,
      opts.dynamicTile
    );

    this._device = backend.device;
    if (this._engine === 'webnn') {
      this._webNNExecutor = new WebNNUNetExecutor(
        this._device,
        validatedModel,
        { precision: opts.precision }
      );
    } else {
      this._nativeExecutor = new NativeUNetExecutor(
        this._device,
        validatedModel,
        { precision: opts.precision, kernel: opts.kernel }
      );
    }
  }

  getDevice() {
    return this._device;
  }

  /** Completes backend compilation before first interactive use. */
  async prepare() {
    if (this._webNNExecutor) {
      await this._webNNExecutor.prepare();
      const overlap = roundUp(
        this._modelSpec.receptiveField / 2,
        OIDN_TILE_ALIGNMENT
      );
      const outputTileEdges = [
        this._dynamicTileController.tileSize,
        this._dynamicTileController.minTileSize
      ];
      await this._webNNExecutor.prewarm(
        [...new Set(outputTileEdges)].map((edge) => ({
          width: edge + 2 * overlap,
          height: edge + 2 * overlap
        }))
      );
      return;
    }
    await this._nativeExecutor!.prepare();
  }

  getRuntimeInfo() {
    return {
      configuredEngine: this._engine,
      gpuEngine: this._webNNExecutor ? 'webnn' as const : 'wgsl' as const,
      precision: (this._webNNExecutor ?? this._nativeExecutor!).precision,
      kernel: this._nativeExecutor
        ? {
            configured: this._nativeExecutor.kernelSetting,
            maxSpatialInputBlocks: this._nativeExecutor.maxSpatialInputBlocks,
            subgroupsAvailable: this._nativeExecutor.subgroupsAvailable
          }
        : undefined,
      webnn: this._webNNExecutor?.support,
      resources: (
        this._webNNExecutor ?? this._nativeExecutor!
      ).getResourceInfo(),
      model: this._modelSpec.id,
      modelFamily: this._modelSpec.family,
      inputChannels: this._inputChannels,
      dynamicTile: {
        enabled: this._dynamicTileController.enabled,
        currentTileSize: this._dynamicTileController.tileSize,
        minTileSize: this._dynamicTileController.minTileSize,
        maxTileSize: this._dynamicTileController.maxTileSize,
        targetTileTimeMs: this._dynamicTileController.targetTileTimeMs
      },
      lastExecution: this._lastExecution
    };
  }

  /** Captures per-node GPU timestamps for the next native tile execution. */
  profileNextExecution() {
    return this._nativeExecutor?.profileNextExecution() ?? false;
  }

  getLastExecutionProfile() {
    return this._nativeExecutor?.getLastExecutionProfile();
  }

  private _updateModel(width: number, height: number) {
    const maxTileSize = this._dynamicTileController.tileSize;

    let tileWidth = fitTileDimension(width, maxTileSize);
    let tileHeight = fitTileDimension(height, maxTileSize);
    const defaultTileOverlap = roundUp(
      this._modelSpec.receptiveField / 2,
      OIDN_TILE_ALIGNMENT
    );
    let tileOverlapX = defaultTileOverlap;
    let tileOverlapY = defaultTileOverlap;

    if (width <= maxTileSize) {
      tileOverlapX = 0;
    }
    if (height <= maxTileSize) {
      tileOverlapY = 0;
    }

    // Force width and height has same size. reduce the cache in memory
    const tileSize = Math.max(tileWidth, tileHeight);
    const tileOverlap = Math.max(tileOverlapX, tileOverlapY);
    tileWidth = tileSize;
    tileHeight = tileSize;
    tileOverlapX = tileOverlap;
    tileOverlapY = tileOverlap;

    if (
      tileWidth !== this._tileWidth ||
      tileHeight !== this._tileHeight ||
      tileOverlapX !== this._tileOverlapX ||
      tileOverlapY !== this._tileOverlapY
    ) {
      // console.log(tileWidth, tileHeight, tileOverlapX, tileOverlapY);
      this._tileWidth = tileWidth;
      this._tileHeight = tileHeight;
      this._tileOverlapX = tileOverlapX;
      this._tileOverlapY = tileOverlapY;
    }
  }

  private _getTileSizeWithOverlap() {
    return {
      width: this._tileWidth + 2 * this._tileOverlapX,
      height: this._tileHeight + 2 * this._tileOverlapY
    };
  }

  private _processImageData(
    color: ImageData | HDRImageData,
    albedo: ImageData | undefined,
    normal: ImageData | undefined,
    isHDR: boolean
  ) {
    const rawData = color.data;
    const pixelsCount = rawData.length / 4;
    const channels = this._aux ? 9 : 3;
    const tensorData = new Float32Array(pixelsCount * channels);

    if ((albedo && !normal) || (normal && !albedo)) {
      throw new Error('Normal map and albedo map are both required');
    }
    if (albedo && normal) {
      if (
        albedo.width !== normal.width ||
        albedo.height !== normal.height ||
        color.width !== albedo.width ||
        color.height !== albedo.height
      ) {
        throw new Error('Image size mismatch');
      }
    }

    const albedoData = albedo?.data;
    const normalData = normal?.data;
    for (let i = 0; i < rawData.length; i += 4) {
      const i2 = (i / 4) * channels;

      for (let c = 0; c < 3; c++) {
        if (isHDR) {
          tensorData[i2 + c] = rawData[i + c];
        } else {
          tensorData[i2 + c] = rawData[i + c] / 255;
        }
        if (albedoData) {
          tensorData[i2 + c + 3] = albedoData[i + c] / 255;
        }
        if (normalData) {
          tensorData[i2 + c + 6] = normalData[i + c] / 255;
        }
      }
    }

    return tensorData;
  }

  private _readTile(
    data: Float32Array,
    channels: number,
    srcTile: Tile,
    width: number
  ) {
    const tileData = new Float32Array(
      srcTile.width * srcTile.height * channels
    );
    for (let y = 0; y < srcTile.height; y++) {
      for (let x = 0; x < srcTile.width; x++) {
        const i2 = ((y + srcTile.y) * width + (x + srcTile.x)) * channels;
        const i1 = (y * srcTile.width + x) * channels;

        for (let c = 0; c < channels; c++) {
          tileData[i1 + c] = data[i2 + c];
        }
      }
    }
    return tileData;
  }

  private _writeTile(
    imageData: ImageData | HDRImageData,
    srcTile: Tile,
    dstTile: Tile,
    srcTileData: Float32Array,
    srcWidth: number,
    isHDR: boolean
  ) {
    const { data: outImageData, width } = imageData;
    const dx = dstTile.x - srcTile.x;
    const dy = dstTile.y - srcTile.y;
    for (let y = 0; y < dstTile.height; y++) {
      for (let x = 0; x < dstTile.width; x++) {
        const i1 = ((y + dy) * srcWidth + x + dx) * 3;
        const i2 = ((y + dstTile.y) * width + (x + dstTile.x)) * 4;

        for (let c = 0; c < 3; c++) {
          if (isHDR) {
            outImageData[i2 + c] = srcTileData[i1 + c];
          } else {
            outImageData[i2 + c] = Math.min(
              Math.max(srcTileData[i1 + c] * 255, 0),
              255
            );
          }
        }
        imageData.data[i2 + 3] = isHDR ? 1 : 255;
      }
    }
  }

  private async _executeTile(
    inputData:
      | Float32Array
      | {
          color: GPUBuffer | GPUTexture;
          // TODO optional
          albedo?: GPUBuffer | GPUTexture;
          normal?: GPUBuffer | GPUTexture;
        },
    outputTileData: ImageData | HDRImageData | undefined,
    outputImageData: ImageData | HDRImageData | undefined,
    i: number,
    j: number,
    width: number,
    height: number,
    isHDR: boolean,
    denoiseAlpha?: boolean
  ) {
    const channels = this._aux ? 9 : 3;
    const tileOverlapX = this._tileOverlapX;
    const tileOverlapY = this._tileOverlapY;
    let srcTileSize = this._getTileSizeWithOverlap();
    let dstTileSize = { width: this._tileWidth, height: this._tileHeight };

    let srcX0 = i > 0 ? i * dstTileSize.width - tileOverlapX : 0;
    let srcX1 = Math.min(srcX0 + srcTileSize.width, width);
    srcX0 = Math.max(srcX1 - srcTileSize.width, 0);

    let srcY0 = j > 0 ? j * dstTileSize.height - tileOverlapY : 0;
    let srcY1 = Math.min(srcY0 + srcTileSize.height, height);
    srcY0 = Math.max(srcY1 - srcTileSize.height, 0);

    const srcTileWidth = srcTileSize.width;
    const srcTileHeight = srcTileSize.height;

    const srcTile = new Tile(srcX0, srcY0, srcTileWidth, srcTileHeight);

    let nativeOutputBuffer: GPUBuffer | undefined;
    let denoisedData: Float32Array | undefined;
    let inputScale = 1;
    const device = this._device!;
    let dataProcessGPU = this._dataProcessGPU;

    if (inputData instanceof Float32Array) {
      let tileData = this._readTile(inputData, channels, srcTile, width);
      if (isHDR) {
        inputScale = avgLogLum({
          data: tileData,
          channels
        });
        tileData = hdrTransferFuncCPU({
          data: tileData,
          channels,
          inputScale
        });
      }
      denoisedData = await (this._webNNExecutor ?? this._nativeExecutor!).executeCPU(
        tileData,
        srcTileWidth,
        srcTileHeight
      );
    } else {
      if (!dataProcessGPU) {
        dataProcessGPU = this._dataProcessGPU = new GPUDataProcess(
          device,
          isHDR
        );
      }
      dataProcessGPU.setImageSize(width, height);
      dataProcessGPU.setInputTile(srcTile);
      // Display the noisy input instead of prev denoised result
      if (i === 0 && j === 0) {
        dataProcessGPU.copyInputDataToOutput(inputData.color);
      }
      const { color, albedo, normal } = dataProcessGPU.forward(
        inputData.color,
        this._aux ? inputData.albedo : undefined,
        this._aux ? inputData.normal : undefined,
        denoiseAlpha
      );

      nativeOutputBuffer = await (this._webNNExecutor ?? this._nativeExecutor!).execute(
        this._aux ? [color, albedo!, normal!] : [color],
        srcTileWidth,
        srcTileHeight
      );
    }

    let outBuffer: GPUBuffer;

    const dstWidth = Math.min(dstTileSize.width, width);
    const dstHeight = Math.min(dstTileSize.height, height);
    const dstTile = new Tile(i * dstWidth, j * dstHeight, dstWidth, dstHeight);
    dstTile.width = Math.min(dstTile.width, width - dstTile.x);
    dstTile.height = Math.min(dstTile.height, height - dstTile.y);

    if (inputData instanceof Float32Array) {
      if (isHDR) {
        denoisedData = hdrTransferFuncInverseCPU({
          data: denoisedData!,
          channels: 3,
          inputScale
        });
      }

      this._writeTile(
        outputImageData!,
        srcTile,
        dstTile,
        denoisedData!,
        srcTileSize.width,
        isHDR
      );

      for (let y = 0; y < dstHeight; y++) {
        for (let x = 0; x < dstWidth; x++) {
          const i1 = (y * dstWidth + x) * 4;
          const i2 = ((y + dstTile.y) * width + (x + dstTile.x)) * 4;
          for (let c = 0; c < 4; c++) {
            outputTileData!.data[i1 + c] = outputImageData!.data[i2 + c];
          }
        }
      }
    } else {
      dataProcessGPU!.setOutputTile(dstTile, srcTile);
      outBuffer = dataProcessGPU!.inverse(
        nativeOutputBuffer!,
        inputData.color
      );
    }
    return outBuffer!;
  }

  tileExecute<T extends ImageData | HDRImageData | GPUImageData>({
    color,
    albedo,
    normal,
    done,
    progress,
    denoiseAlpha
  }: {
    color: T;
    albedo?: ImageData | GPUImageData;
    normal?: ImageData | GPUImageData;
    /**
     * If denoise alpha channel. Otherwise denoise RGB channels.
     */
    denoiseAlpha?: boolean;
    done: (outputData: T extends GPUImageData ? GPUImageDataOutput : T) => void;
    progress?: (
      outputData: T extends GPUImageData ? GPUImageDataOutput : T,
      tileData: (T extends GPUImageData ? GPUImageDataOutput : T) | undefined,
      tile: Tile,
      currentIdx: number,
      totalIdx: number
    ) => void;
  }): () => void {
    if (this._aux && (!albedo || !normal)) {
      throw new Error('Normal map and albedo map are both required');
    }

    if (!this._aux) {
      if (albedo || normal) {
        throw new Error('Normal map and albedo map are not required');
      }
    }

    const width = color.width;
    const height = color.height;
    const adaptiveTileSize = this._dynamicTileController.tileSize;
    const shouldAdaptTileSize =
      width > adaptiveTileSize || height > adaptiveTileSize;
    this._updateModel(width, height);

    // TODO should fixed to be hdr when UNet is created.
    // weights of hdr and ldr is different

    const hdr = this._hdr || false;
    let rawData: Float32Array;
    if (!isGPUImageData(color)) {
      rawData = this._processImageData(
        color,
        albedo as ImageData | undefined,
        normal as ImageData | undefined,
        hdr
      );
    }
    const tileWidth = this._tileWidth;
    const tileHeight = this._tileHeight;
    const tileCountH = Math.ceil(height / tileHeight);
    const tileCountW = Math.ceil(width / tileWidth);

    function makeImageData(width: number, height: number) {
      return hdr
        ? {
            data: new Float32Array(width * height * 4),
            width,
            height
          }
        : new ImageData(width, height);
    }

    const outputImageData = isGPUImageData(color)
      ? undefined
      : makeImageData(width, height);
    const outputTileData = isGPUImageData(color)
      ? undefined
      : makeImageData(Math.min(tileWidth, width), Math.min(tileHeight, height));

    let aborted = false;

    const now = () =>
      typeof performance === 'undefined' ? Date.now() : performance.now();
    const executionStartTime = now();
    const tileTimesMs: number[] = [];
    const scheduleNextTile = (callback: () => void) => {
      if (typeof requestAnimationFrame === 'undefined') {
        setTimeout(callback, 0);
      } else {
        requestAnimationFrame(callback);
      }
    };

    const executeTile = async (i: number, j: number) => {
      if (aborted) {
        return;
      }
      const tileStartTime = now();
      const resGPUBuffer = await this._executeTile(
        isGPUImageData(color)
          ? {
              color: color.data,
              albedo: (albedo as GPUImageData | undefined)?.data,
              normal: (normal as GPUImageData | undefined)?.data
            }
          : rawData,
        outputTileData,
        outputImageData,
        i,
        j,
        width,
        height,
        hdr,
        denoiseAlpha
      );
      if (aborted) return;
      const output = outputImageData || {
        data: resGPUBuffer,
        width,
        height
      };
      progress?.(
        output as any,
        // Is undefined if using webgpu buffer
        outputTileData as any,
        new Tile(i * tileWidth, j * tileHeight, tileWidth, tileHeight),
        i + j * tileCountW,
        tileCountW * tileCountH
      );

      const hasNextTile = i + 1 < tileCountW || j + 1 < tileCountH;
      const continueAfterGPUWork = () => {
        tileTimesMs.push(now() - tileStartTime);
        if (aborted) return;

        if (hasNextTile) {
          scheduleNextTile(() => {
            if (aborted) return;
            if (i + 1 < tileCountW) {
              executeTile(i + 1, j);
            } else if (j + 1 < tileCountH) {
              executeTile(0, j + 1);
            }
          });
        } else {
          const sortedTileTimes = [...tileTimesMs].sort((a, b) => a - b);
          const middle = Math.floor(sortedTileTimes.length / 2);
          const medianTileTime = sortedTileTimes.length % 2
            ? sortedTileTimes[middle]
            : (sortedTileTimes[middle - 1] + sortedTileTimes[middle]) / 2;
          this._lastExecution = {
            width,
            height,
            tileWidth,
            tileHeight,
            tileCount: tileCountW * tileCountH,
            durationMs: now() - executionStartTime,
            tileTimeMs: {
              min: sortedTileTimes[0],
              median: medianTileTime,
              mean:
                sortedTileTimes.reduce((sum, value) => sum + value, 0) /
                sortedTileTimes.length,
              max: sortedTileTimes[sortedTileTimes.length - 1]
            }
          };
          // Adapt only from complete executions. Cancelled work is commonly
          // contending with interactive rendering and is not representative.
          if (shouldAdaptTileSize) {
            this._dynamicTileController.observe(tileTimesMs);
          }
          // console.log(memory());
          done(output as any);
        }
      };

      // requestAnimationFrame only throttles JavaScript submission. Waiting
      // for the queue here keeps at most one OIDN tile in flight, so aborting
      // cannot leave a long tail of already-submitted GPU work.
      void waitForSubmittedGPUWork(this._device!.queue).then(
        continueAfterGPUWork
      );
    };

    executeTile(0, 0);

    return () => {
      aborted = true;
    };
  }

  dispose() {
    this._dataProcessGPU?.dispose();
    this._nativeExecutor?.dispose();
    this._webNNExecutor?.dispose();
  }
}

export default UNet;
