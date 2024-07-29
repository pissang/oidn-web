// import * as tfjs from '@tensorflow/tfjs-core';
import { Tensor, Tensor1D, Tensor4D } from '@tensorflow/tfjs-core';
import type { SymbolicTensor } from '@tensorflow/tfjs-layers';
import { tensor } from '@tensorflow/tfjs-core/dist/ops/tensor';
import { tensor1d } from '@tensorflow/tfjs-core/dist/ops/tensor1d';
import { mirrorPad } from '@tensorflow/tfjs-core/dist/ops/mirror_pad';
import { pad4d } from '@tensorflow/tfjs-core/dist/ops/pad4d';
import { slice4d } from '@tensorflow/tfjs-core/dist/ops/slice4d';
import { concat4d } from '@tensorflow/tfjs-core/dist/ops/concat_4d';
import { ENGINE } from '@tensorflow/tfjs-core/dist/engine';
import {
  Conv2D,
  UpSampling2D
} from '@tensorflow/tfjs-layers/dist/layers/convolutional';
import { MaxPooling2D } from '@tensorflow/tfjs-layers/dist/layers/pooling';
import { Concatenate } from '@tensorflow/tfjs-layers/dist/layers/merge';
import { LayersModel } from '@tensorflow/tfjs-layers/dist/engine/training';
import { Input as TFInput } from '@tensorflow/tfjs-layers/dist/engine/input_layer';
import { HostTensor } from './tza';
import { Float16Array } from '@petamoriken/float16';
import {
  GPUDataProcess,
  Tile,
  avgLogLum,
  hdrTransferFuncCPU,
  hdrTransferFuncInverseCPU
} from './process';
import type { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';

import { profileAndLogKernelCode, memory } from './helper';

function getTensorData(
  ubytes: Uint8Array,
  type: HostTensor['desc']['dataType']
) {
  const buffer = ubytes.buffer;
  if (type === 'Float32') {
    return new Float32Array(ubytes.buffer);
  }
  const float16Data = new Float16Array(buffer);
  const float32Data = new Float32Array(float16Data.length);
  for (let i = 0; i < float32Data.length; ++i) {
    float32Data[i] = float16Data[i];
  }
  return float32Data;
}

function changeWeightShapes(weightData: Float32Array, dims: number[]) {
  const [O, C, H, W] = dims;
  const reorderedWeightData = new Float32Array(weightData.length);
  for (let o = 0; o < O; ++o) {
    for (let c = 0; c < C; ++c) {
      for (let h = 0; h < H; ++h) {
        for (let w = 0; w < W; ++w) {
          // Change OCHW to HWCO
          const idx = o * C * H * W + c * H * W + h * W + w;
          const idx2 = h * W * C * O + w * C * O + c * O + o;
          reorderedWeightData[idx2] = weightData[idx];
        }
      }
    }
  }
  return reorderedWeightData;
}

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

function roundUp(a: number, b: number) {
  return Math.ceil(a / b) * b;
}
// Returns the smallest integer larger than or equal to a which has remainder c when divided by b
function roundUp2(a: number, b: number, c: number) {
  return Math.ceil((a - c) / b) * b + c;
}

function isGPUImageData(
  data: ImageData | GPUImageData | HDRImageData
): data is GPUImageData {
  return data.data instanceof GPUBuffer || data.data instanceof GPUTexture;
}

const receptiveField = 174; // receptive field in pixels
const receptiveFieldLarge = 202;
// TODO metal is 32?
const minTileAlignment = 1;

const tileAlignment = 16; // required spatial alignment in pixels (padding may be necessary)

const defaultTileOverlap = roundUp(receptiveField / 2, tileAlignment);
const defaultTileOverlapLarge = roundUp(receptiveFieldLarge / 2, tileAlignment);

class UNet {
  private _tfModel: LayersModel | undefined;
  private _device: GPUDevice | undefined;

  // TODO calculate the tile size from memory size
  // https://github.com/RenderKit/oidn/blob/713ec7838ba650f99e0a896549c0dca5eeb3652d/core/unet_filter.cpp#L287
  private _tileWidth = 0;
  private _tileHeight = 0;

  private _tileOverlapX = 0;
  private _tileOverlapY = 0;

  private _aux;
  private _hdr;
  private _denoiseAlpha;

  private _dataProcessGPU?: GPUDataProcess;

  private _maxTileSize;

  private _tensors = new Map<string, Tensor>();

  constructor(
    private _hostTensors: Map<string, HostTensor>,
    private _backend: WebGPUBackend,
    opts: {
      /**
       * If use auxiliary data.
       */
      aux?: boolean;
      /**
       * If input is HDR image.
       */
      hdr?: boolean;
      /**
       * If denoise alpha channel. Otherwise denoise RGB channels.
       */
      denoiseAlpha?: boolean;
      maxTileSize?: number;
    } = {}
  ) {
    this._aux = opts.aux || false;
    this._hdr = opts.hdr || false;
    this._denoiseAlpha = opts.denoiseAlpha || false;

    this._maxTileSize = opts.maxTileSize ?? 512;

    this._device = this._backend.device;
  }

  getDevice() {
    return this._device;
  }

  private _buildModel(isLarge: boolean) {
    const aux = this._aux;
    const channels = 3 + (aux ? 6 : 0);
    const tileSize = this._getTileSizeWithOverlap();

    // TODO input process transferFunc
    // TODO input shape
    const input = TFInput({
      shape: [tileSize.height, tileSize.width, channels],
      dtype: 'float32'
    });

    this._tfModel = new LayersModel({
      inputs: [input],
      // TODO output process transferFunc
      outputs: isLarge ? this._addNetLarge(input) : this._addNet(input)
    });
  }

  private _createConv(
    name: string,
    source: SymbolicTensor,
    activation?: 'relu'
  ) {
    const weightTensorName = name + '.weight';
    const biasTensorName = name + '.bias';
    const tensors = this._tensors;
    let weightTensor = tensors.get(weightTensorName);
    let biasTensor = tensors.get(biasTensorName);
    const unetWeightTensor = this._hostTensors.get(weightTensorName)!;

    if (!weightTensor) {
      const weightDims = unetWeightTensor.desc.dims;
      weightTensor = tensor(
        changeWeightShapes(
          getTensorData(unetWeightTensor.data, unetWeightTensor.desc.dataType),
          weightDims
        ),
        [weightDims[2], weightDims[3], weightDims[1], weightDims[0]],
        'float32'
      );

      tensors.set(weightTensorName, weightTensor);
    }
    if (!biasTensor) {
      const unetBiasTensor = this._hostTensors.get(name + '.bias')!;
      biasTensor = tensor1d(
        getTensorData(unetBiasTensor.data, unetBiasTensor.desc.dataType),
        'float32'
      );
      tensors.set(biasTensorName, biasTensor);
    }
    // TODO whats the purpose of padded dims ?
    const convLayer = new Conv2D({
      name,
      filters: unetWeightTensor.desc.dims[0],
      kernelSize: unetWeightTensor.desc.dims.slice(2, 4) as [number, number],
      useBias: true,
      activation,
      padding: 'same',
      weights: [weightTensor, biasTensor],
      trainable: false
    });

    return convLayer.apply(source) as SymbolicTensor;
  }

  private _createConcatConv(
    name: string,
    source1: SymbolicTensor,
    source2: SymbolicTensor
  ) {
    const concatLayer = new Concatenate({ trainable: false, axis: 3 });
    //https://github.com/RenderKit/oidn/blob/713ec7838ba650f99e0a896549c0dca5eeb3652d/training/model.py#L40
    return this._createConv(
      name,
      // Concat on the channel
      concatLayer.apply([
        // convLayer.apply(source2) as SymbolicTensor,
        source1,
        source2
      ]) as SymbolicTensor,
      'relu'
    ) as SymbolicTensor;
  }

  private _createPooling(source: SymbolicTensor) {
    const poolingLayer = new MaxPooling2D({
      name: source.name + '/pooling',
      poolSize: [2, 2],
      strides: [2, 2],
      padding: 'same',
      trainable: false
    });
    // https://github.com/RenderKit/oidn/blob/713ec7838ba650f99e0a896549c0dca5eeb3652d/training/model.py#L33
    return poolingLayer.apply(source) as SymbolicTensor;
  }

  private _addUpsamplingLayer(source: SymbolicTensor) {
    const upsamplingLayer = new UpSampling2D({
      name: source.name + '/upsampling',
      size: [2, 2],
      trainable: false
    });
    return upsamplingLayer.apply(source) as SymbolicTensor;
  }

  private _addNet(input: SymbolicTensor) {
    let x = this._createConv('enc_conv0', input, 'relu');
    const pool1 = (x = this._createPooling(
      this._createConv('enc_conv1', x, 'relu')
    ));
    const pool2 = (x = this._createPooling(
      this._createConv('enc_conv2', x, 'relu')
    ));
    const pool3 = (x = this._createPooling(
      this._createConv('enc_conv3', x, 'relu')
    ));
    const pool4 = (x = this._createPooling(
      this._createConv('enc_conv4', x, 'relu')
    ));
    x = this._createConv('enc_conv5a', pool4, 'relu');
    x = this._addUpsamplingLayer(this._createConv('enc_conv5b', x, 'relu'));

    x = this._createConcatConv('dec_conv4a', x, pool3);
    x = this._addUpsamplingLayer(this._createConv('dec_conv4b', x, 'relu'));

    x = this._createConcatConv('dec_conv3a', x, pool2);
    x = this._addUpsamplingLayer(this._createConv('dec_conv3b', x, 'relu'));

    x = this._createConcatConv('dec_conv2a', x, pool1);
    x = this._addUpsamplingLayer(this._createConv('dec_conv2b', x, 'relu'));

    x = this._createConcatConv('dec_conv1a', x, input);
    x = this._createConv('dec_conv1b', x, 'relu');
    x = this._createConv('dec_conv0', x, 'relu');

    return x;
  }

  private _addNetLarge(input: SymbolicTensor) {
    let x = this._createConv('enc_conv1a', input, 'relu');
    const pool1 = (x = this._createPooling(
      this._createConv('enc_conv1b', x, 'relu')
    ));
    x = this._createConv('enc_conv2a', x, 'relu');
    const pool2 = (x = this._createPooling(
      this._createConv('enc_conv2b', x, 'relu')
    ));
    x = this._createConv('enc_conv3a', x, 'relu');
    const pool3 = (x = this._createPooling(
      this._createConv('enc_conv3b', x, 'relu')
    ));
    x = this._createConv('enc_conv4a', x, 'relu');
    const pool4 = (x = this._createPooling(
      this._createConv('enc_conv4b', x, 'relu')
    ));

    x = this._createConv('enc_conv5a', pool4, 'relu');
    x = this._addUpsamplingLayer(this._createConv('enc_conv5b', x, 'relu'));

    x = this._createConcatConv('dec_conv4a', x, pool3);
    x = this._addUpsamplingLayer(this._createConv('dec_conv4b', x, 'relu'));

    x = this._createConcatConv('dec_conv3a', x, pool2);
    x = this._addUpsamplingLayer(this._createConv('dec_conv3b', x, 'relu'));

    x = this._createConcatConv('dec_conv2a', x, pool1);
    x = this._addUpsamplingLayer(this._createConv('dec_conv2b', x, 'relu'));

    x = this._createConcatConv('dec_conv1a', x, input);
    x = this._createConv('dec_conv1b', x, 'relu');
    x = this._createConv('dec_conv1c', x, 'relu');

    return x;
  }

  private _updateModel(width: number, height: number) {
    const isLarge = this._tensors.has('enc_conv1b.weight');
    const maxTileSize = this._maxTileSize;

    let tileWidth = maxTileSize;
    let tileHeight = maxTileSize;
    let tileOverlapX = isLarge ? defaultTileOverlapLarge : defaultTileOverlap;
    let tileOverlapY = isLarge ? defaultTileOverlapLarge : defaultTileOverlap;

    if (width < maxTileSize + defaultTileOverlap * 2) {
      tileWidth = roundUp(width, tileAlignment);
      tileOverlapX = 0;
    }
    if (height < maxTileSize + defaultTileOverlap * 2) {
      tileHeight = roundUp(height, tileAlignment);
      tileOverlapY = 0;
    }

    if (
      tileWidth !== this._tileWidth ||
      tileHeight !== this._tileHeight ||
      tileOverlapX !== this._tileOverlapX ||
      tileOverlapY !== this._tileOverlapY ||
      !this._tfModel
    ) {
      this._tileWidth = tileWidth;
      this._tileHeight = tileHeight;
      this._tileOverlapX = tileOverlapX;
      this._tileOverlapY = tileOverlapY;

      if (this._tfModel) {
        this._tfModel.dispose();
      }

      this._buildModel(isLarge);
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

  private _executeTile(
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
    isHDR: boolean
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

    const srcTileWidth = Math.min(srcTileSize.width, width);
    const srcTileHeight = Math.min(srcTileSize.height, height);
    const needsResize =
      width < dstTileSize.width || height < dstTileSize.height;
    const srcTile = new Tile(srcX0, srcY0, srcTileWidth, srcTileHeight);

    let tileTensor!: Tensor;
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
      tileTensor = tensor(
        tileData,
        [1, srcTileHeight, srcTileWidth, channels],
        'float32'
      ) as Tensor4D;
    } else {
      if (!dataProcessGPU) {
        dataProcessGPU = this._dataProcessGPU = new GPUDataProcess(
          device,
          isHDR,
          this._denoiseAlpha
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
        this._aux ? inputData.normal : undefined
      );

      const createTensor = (buffer: GPUBuffer) => {
        const tmp = tensor({ buffer, zeroCopy: true }, [
          1,
          srcTileHeight,
          srcTileWidth,
          4
        ]) as Tensor4D;
        const ret = slice4d(
          tmp,
          [0, 0, 0, 0],
          [1, srcTileHeight, srcTileWidth, 3]
        );
        return ret;
      };

      if (this._aux) {
        const tensors = [color, albedo, normal].map((buffer) =>
          createTensor(buffer!)
        );
        tileTensor = concat4d(tensors, 3);
      } else {
        tileTensor = createTensor(color);
      }
    }
    // We need resize if input size is smaller than tile size. And is rounded up.
    if (needsResize) {
      const rawTileTensor = tileTensor;
      tileTensor = mirrorPad(
        rawTileTensor,
        [
          [0, 0],
          [0, srcTileSize.height - height],
          [0, srcTileSize.width - width],
          [0, 0]
        ],
        'reflect'
      );
    }

    let outBuffer: GPUBuffer;
    const outputTensor = this._tfModel!.predict(tileTensor) as Tensor;

    const dstWidth = Math.min(dstTileSize.width, width);
    const dstHeight = Math.min(dstTileSize.height, height);
    const dstTile = new Tile(i * dstWidth, j * dstHeight, dstWidth, dstHeight);
    dstTile.width = Math.min(dstTile.width, width - dstTile.x);
    dstTile.height = Math.min(dstTile.height, height - dstTile.y);

    if (inputData instanceof Float32Array) {
      let denoisedData = outputTensor.dataSync();
      if (isHDR) {
        denoisedData = hdrTransferFuncInverseCPU({
          data: denoisedData as Float32Array,
          channels: 3,
          inputScale
        });
      }

      this._writeTile(
        outputImageData!,
        srcTile,
        dstTile,
        denoisedData as Float32Array,
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
      // IMPORTANT
      // storage buffer has alignment. that 3 channels still needs 16 bytes data.
      // So we need to pad it to 4 channels.
      const outputTensor4Channnels = pad4d(outputTensor as Tensor4D, [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 1]
      ]);
      outBuffer = dataProcessGPU!.inverse(
        outputTensor4Channnels.dataToGPU().buffer!,
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
    progress
  }: {
    color: T;
    albedo?: ImageData | GPUImageData;
    normal?: ImageData | GPUImageData;
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

    const executeTile = (i: number, j: number) => {
      if (aborted) {
        return;
      }
      let resGPUBuffer;
      // profileAndLogKernelCode(() => {
      ENGINE.startScope();
      resGPUBuffer = this._executeTile(
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
        hdr
      );
      ENGINE.endScope();
      // }, true);
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

      if (i + 1 < tileCountW || j + 1 < tileCountH) {
        requestAnimationFrame(() => {
          if (i + 1 < tileCountW) {
            executeTile(i + 1, j);
          } else if (j + 1 < tileCountH) {
            executeTile(0, j + 1);
          }
        });
      } else {
        console.log(memory());
        done(output as any);
      }
    };

    executeTile(0, 0);

    return () => {
      aborted = true;
    };
  }

  dispose() {
    this._tfModel?.dispose();
    this._dataProcessGPU?.dispose();
    this._tensors.forEach((tensor) => tensor.dispose());
  }
}

export default UNet;
