import * as tfjs from '@tensorflow/tfjs';
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

class UNet {
  private _tfModel: tfjs.LayersModel;

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
    // const { desc, data } = this._tensors.get(name + '.weight')!;

    // TODO what's the difference between weight1 and weight2
    // const weightTensor = tfjs.tensor(
    //   changeWeightShapes(getTensorData(data, desc.dataType), desc.dims),
    //   [desc.dims[2], desc.dims[3], desc.dims[1], desc.dims[0]],
    //   'float32'
    // );

    // // TODO addOp?

    // const convLayer = tfjs.layers.conv2d({
    //   name,
    //   filters: desc.dims[0],
    //   kernelSize: desc.dims.slice(2, 4) as [number, number],
    //   useBias: false,
    //   // Identity
    //   activation: 'linear',
    //   padding: 'same',
    //   weights: [weightTensor],
    //   trainable: false
    // });

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

  buildModel() {
    const channels = 3;

    // TODO input process transferFunc
    // TODO input shape
    const input = tfjs.input({ shape: [512, 512, channels], dtype: 'float32' });

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

  executeImageData(image: ImageData) {
    const tensorData = new Float32Array((image.data.length / 4) * 3);
    for (let i = 0; i < image.data.length; i += 4) {
      tensorData[i / 4] = image.data[i] / 255;
      tensorData[i / 4 + 1] = image.data[i + 1] / 255;
      tensorData[i / 4 + 2] = image.data[i + 2] / 255;
    }
    const input = tfjs.tensor(
      tensorData,
      [image.height, image.width, 3],
      'float32'
    );
    const output = this._tfModel.predict(input) as tfjs.Tensor;
    const outputData = output.dataSync();
    output.dispose();
    const outputImageData = new ImageData(image.width, image.height);
    for (let i = 0; i < outputData.length; i += 3) {
      outputImageData.data[i] = outputData[i] * 255;
      outputImageData.data[i + 1] = outputData[i + 1] * 255;
      outputImageData.data[i + 2] = outputData[i + 2] * 255;
      // Keep alpha channel
      outputImageData.data[i + 3] = image.data[i + 3];
    }
    return outputImageData;
  }
}

export default UNet;
