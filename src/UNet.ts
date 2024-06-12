import * as tfjs from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { HostTensor } from './tza';
import { Float16Array } from '@petamoriken/float16';

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

function roundUp(a: number, b: number) {
  return Math.ceil(a / b) * b;
}
// Returns the smallest integer larger than or equal to a which has remainder c when divided by b
function roundUp2(a: number, b: number, c: number) {
  return Math.ceil((a - c) / b) * b + c;
}

const receptiveField = 174; // receptive field in pixels
// TODO metal is 32?
const minTileAlignment = 1;

const tileAlignment = 16; // required spatial alignment in pixels (padding may be necessary)

class UNet {
  private _tfModel!: tfjs.LayersModel;

  // TODO calculate the tile size from memory size
  // https://github.com/RenderKit/oidn/blob/713ec7838ba650f99e0a896549c0dca5eeb3652d/core/unet_filter.cpp#L287
  private _tileSize = 512;

  // private _tileOverlap = roundUp(receptiveField / 2, tileAlignment);
  private _tileOverlap = 0;

  private _aux = false;

  constructor(private _tensors: Map<string, HostTensor>) {}

  private _createConv(
    name: string,
    source: tfjs.SymbolicTensor,
    activation?: 'relu'
  ) {
    const unetWeightTensor = this._tensors.get(name + '.weight')!;
    const unetBiasTensor = this._tensors.get(name + '.bias')!;
    const weightDims = unetWeightTensor.desc.dims;
    const weightTensor = tfjs.tensor(
      changeWeightShapes(
        getTensorData(unetWeightTensor.data, unetWeightTensor.desc.dataType),
        weightDims
      ),
      [weightDims[2], weightDims[3], weightDims[1], weightDims[0]],
      'float32'
    );
    const biasTensor = tfjs.tensor1d(
      getTensorData(unetBiasTensor.data, unetBiasTensor.desc.dataType),
      'float32'
    );
    // TODO whats the purpose of padded dims ?
    const convLayer = tfjs.layers.conv2d({
      name,
      filters: unetWeightTensor.desc.dims[0],
      kernelSize: unetWeightTensor.desc.dims.slice(2, 4) as [number, number],
      useBias: true,
      activation,
      padding: 'same',
      weights: [weightTensor, biasTensor],
      trainable: false
    });

    return convLayer.apply(source) as tfjs.SymbolicTensor;
  }

  private _createConcatConv(
    name: string,
    source1: tfjs.SymbolicTensor,
    source2: tfjs.SymbolicTensor
  ) {
    //https://github.com/RenderKit/oidn/blob/713ec7838ba650f99e0a896549c0dca5eeb3652d/training/model.py#L40
    return this._createConv(
      name,
      // Concat on the channel
      tfjs.layers.concatenate({ trainable: false, axis: 3 }).apply([
        // convLayer.apply(source2) as tfjs.SymbolicTensor,
        source1,
        source2
      ]) as tfjs.SymbolicTensor,
      'relu'
    ) as tfjs.SymbolicTensor;
  }

  private _createPooling(source: tfjs.SymbolicTensor) {
    // https://github.com/RenderKit/oidn/blob/713ec7838ba650f99e0a896549c0dca5eeb3652d/training/model.py#L33
    return tfjs.layers
      .maxPooling2d({
        name: source.name + '/pooling',
        poolSize: [2, 2],
        strides: [2, 2],
        padding: 'same',
        trainable: false
      })
      .apply(source) as tfjs.SymbolicTensor;
  }

  private _addUpsamplingLayer(source: tfjs.SymbolicTensor) {
    return tfjs.layers
      .upSampling2d({
        name: source.name + '/upsampling',
        size: [2, 2],
        trainable: false
      })
      .apply(source) as tfjs.SymbolicTensor;
  }

  buildModel(aux: boolean) {
    this._aux = aux;

    const channels = 3 + (aux ? 6 : 0);
    const tileSize = this._getTileSizeWithOverlap();

    // TODO input process transferFunc
    // TODO input shape
    const input = tfjs.input({
      shape: [tileSize, tileSize, channels],
      dtype: 'float32'
    });

    const encConv0 = this._createConv('enc_conv0', input, 'relu');
    const pool1 = this._createPooling(
      this._createConv('enc_conv1', encConv0, 'relu')
    );
    const pool2 = this._createPooling(
      this._createConv('enc_conv2', pool1, 'relu')
    );
    const pool3 = this._createPooling(
      this._createConv('enc_conv3', pool2, 'relu')
    );
    const pool4 = this._createPooling(
      this._createConv('enc_conv4', pool3, 'relu')
    );
    const encConv5a = this._createConv('enc_conv5a', pool4, 'relu');
    const upsample4 = this._addUpsamplingLayer(
      this._createConv('enc_conv5b', encConv5a, 'relu')
    );
    const decConv4a = this._createConcatConv('dec_conv4a', upsample4, pool3);
    const upsample3 = this._addUpsamplingLayer(
      this._createConv('dec_conv4b', decConv4a, 'relu')
    );
    const decConv3a = this._createConcatConv('dec_conv3a', upsample3, pool2);
    const upsample2 = this._addUpsamplingLayer(
      this._createConv('dec_conv3b', decConv3a, 'relu')
    );
    const decConv2a = this._createConcatConv('dec_conv2a', upsample2, pool1);
    const upsample1 = this._addUpsamplingLayer(
      this._createConv('dec_conv2b', decConv2a, 'relu')
    );
    const decConv1a = this._createConcatConv('dec_conv1a', upsample1, input);
    const decConv1b = this._createConv('dec_conv1b', decConv1a, 'relu');
    const decConv0 = this._createConv('dec_conv0', decConv1b, 'relu');

    this._tfModel = tfjs.model({
      inputs: [input],
      // TODO output process transferFunc
      outputs: decConv0
    });
  }

  setWebGPUBackend() {
    return tfjs.setBackend('webgpu');
  }

  private _getTileSizeWithOverlap() {
    return this._tileSize + 2 * this._tileOverlap;
  }

  private _processImageData(
    color: ImageData,
    albedo?: ImageData,
    normal?: ImageData
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

    for (let i = 0; i < rawData.length; i += 4) {
      const i2 = (i / 4) * channels;
      tensorData[i2] = rawData[i] / 255;
      tensorData[i2 + 1] = rawData[i + 1] / 255;
      tensorData[i2 + 2] = rawData[i + 2] / 255;

      if (albedo) {
        const albedoData = albedo.data;
        tensorData[i2 + 3] *= albedoData[i] / 255;
        tensorData[i2 + 4] *= albedoData[i + 1] / 255;
        tensorData[i2 + 5] *= albedoData[i + 2] / 255;
      }
      if (normal) {
        const normalData = normal.data;
        tensorData[i2 + 6] = normalData[i] / 255;
        tensorData[i2 + 7] = normalData[i + 1] / 255;
        tensorData[i2 + 8] = normalData[i + 2] / 255;
      }
    }

    return tensorData;
  }

  private _readTile(
    data: Float32Array,
    channels: number,
    tileX: number,
    tileY: number,
    tileSize: number
  ) {
    const tileData = new Float32Array(tileSize * tileSize * channels);
    for (let y = 0; y < tileSize; y++) {
      for (let x = 0; x < tileSize; x++) {
        const i2 = ((y + tileY) * tileSize + (x + tileX)) * channels;
        const i1 = (y * tileSize + x) * channels;

        for (let c = 0; c < channels; c++) {
          tileData[i1 + c] = data[i2 + c];
        }
      }
    }
    return tileData;
  }

  private _writeTile(
    imageData: ImageData,
    tileX: number,
    tileY: number,
    tileSize: number,
    tileData: Float32Array
  ) {
    const outImageData = imageData.data;
    for (let y = 0; y < tileSize; y++) {
      for (let x = 0; x < tileSize; x++) {
        const i2 = ((y + tileY) * tileSize + (x + tileX)) * 4;
        const i1 = (y * tileSize + x) * 3;

        for (let c = 0; c < 3; c++) {
          outImageData[i2 + c] = Math.min(
            Math.max(tileData[i1 + c] * 255, 0),
            255
          );
        }
        imageData.data[i2 + 3] = 255;
      }
    }
  }

  private _executeTile(
    inputData: Float32Array,
    outputImageData: ImageData,
    i: number,
    j: number,
    width: number,
    height: number
  ) {
    const channels = this._aux ? 9 : 3;
    const tileOverlap = this._tileOverlap;
    const tileSize = this._getTileSizeWithOverlap();

    let y0 = i > 0 ? i * (tileSize - tileOverlap) : 0;
    let y1 = Math.min(y0 + tileSize, height);
    y0 = y1 - tileSize;

    let x0 = j > 0 ? j * (tileSize - tileOverlap) : 0;
    let x1 = Math.min(x0 + tileSize, width);
    x0 = x1 - tileSize;

    const tileData = this._readTile(inputData, channels, x0, y0, tileSize);
    const input = tfjs.tensor(
      tileData,
      [1, tileSize, tileSize, channels],
      'float32'
    );
    const output = this._tfModel!.predict(input) as tfjs.Tensor;
    const outputData = output.dataSync();
    output.dispose();

    const ox0 = i > 0 ? x0 + tileOverlap : x0;
    const oy0 = j > 0 ? y0 + tileOverlap : y0;

    this._writeTile(
      outputImageData,
      ox0,
      oy0,
      tileSize - 2 * tileOverlap,
      outputData as Float32Array
    );
  }

  executeImageData(color: ImageData, albedo?: ImageData, normal?: ImageData) {
    if (this._aux && (!albedo || !normal)) {
      throw new Error('Normal map and albedo map are both required');
    }

    if (!this._aux) {
      if (albedo || normal) {
        throw new Error('Normal map and albedo map are not required');
      }
    }

    const rawData = this._processImageData(color, albedo, normal);
    const tileOverlap = this._tileOverlap;
    const tileSize = this._getTileSizeWithOverlap();
    const width = color.width;
    const height = color.height;
    const tileCountH = Math.ceil(height / (tileSize - tileOverlap * 2));
    const tileCountW = Math.ceil(width / (tileSize - tileOverlap * 2));

    const outputImageData = new ImageData(width, height);
    // TODO small width and height

    // TODO tile pad?
    for (let j = 0; j < tileCountH; j++) {
      for (let i = 0; i < tileCountW; i++) {
        this._executeTile(rawData, outputImageData, i, j, width, height);
      }
    }
    return outputImageData;
  }

  dispose() {
    this._tfModel.dispose();
  }
}

export default UNet;
