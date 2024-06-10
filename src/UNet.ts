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

class UNet {
  private _tfModel: tfjs.LayersModel;

  constructor(private _tensors: Map<string, HostTensor>) {}

  private _createConv(
    name: string,
    source: tfjs.SymbolicTensor,
    activation?: 'reLU'
  ) {
    const unetWeightTensor = this._tensors.get(name + '.weight')!;
    const unetBiasTensor = this._tensors.get(name + '.bias')!;
    const weightTensor = tfjs.tensor(
      getTensorData(unetWeightTensor.data, unetWeightTensor.desc.dataType),
      unetWeightTensor.desc.dims,
      'float32'
    );
    const biasTensor = tfjs.tensor(
      getTensorData(unetBiasTensor.data, unetBiasTensor.desc.dataType),
      unetBiasTensor.desc.dims,
      'float32'
    );
    // TODO whats the purpose of padded dims ?
    const convLayer = tfjs.layers.conv2d({
      name,
      filters: unetWeightTensor.desc.dims[0],
      kernelSize: unetWeightTensor.desc.dims.slice(2, 4) as [number, number],
      useBias: false,
      // NCHW
      dataFormat: 'channelsFirst',
      // Identity
      activation: 'linear',
      padding: 'same',
      weights: [weightTensor],
      trainable: false
    });
    // https://github.com/RenderKit/oidn/blob/713ec7838ba650f99e0a896549c0dca5eeb3652d/core/concat_conv_hwc.cpp#L26
    // dst = conv(src1, weight1) + bias
    const out = tfjs.layers.add().apply([
      // TODO
      convLayer.apply(source) as tfjs.SymbolicTensor,
      tfjs.input(biasTensor)
    ]);
    if (activation) {
      return tfjs.layers[activation]().apply(out) as tfjs.SymbolicTensor;
    }
    return out as tfjs.SymbolicTensor;
  }

  private _createConcatConv(
    name: string,
    source1: tfjs.SymbolicTensor,
    source2: tfjs.SymbolicTensor
  ) {
    // https://github.com/RenderKit/oidn/blob/713ec7838ba650f99e0a896549c0dca5eeb3652d/core/concat_conv_hwc.cpp#L29
    // dst = activation(conv(src2, weight2) + dst)

    const { desc, data } = this._tensors.get(name + '.weight')!;

    // TODO what's the difference between weight1 and weight2
    const weightTensor = tfjs.tensor(
      getTensorData(data, desc.dataType),
      desc.dims,
      'float32'
    );

    // TODO addOp?

    const convLayer = tfjs.layers.conv2d({
      name,
      filters: desc.dims[0],
      kernelSize: desc.dims.slice(2, 4) as [number, number],
      useBias: false,
      // NCHW
      dataFormat: 'channelsFirst',
      // Identity
      activation: 'linear',
      padding: 'same',
      weights: [weightTensor],
      trainable: false
    });

    return tfjs.layers
      .reLU()
      .apply(
        tfjs.layers
          .add()
          .apply([
            convLayer.apply(source2) as tfjs.SymbolicTensor,
            this._createConv(name, source1)
          ])
      ) as tfjs.SymbolicTensor;
  }

  private _createPooling(source: tfjs.SymbolicTensor) {
    // https://github.com/RenderKit/oidn/blob/713ec7838ba650f99e0a896549c0dca5eeb3652d/training/model.py#L33
    return tfjs.layers
      .maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2],
        // NCHW
        dataFormat: 'channelsFirst',
        padding: 'same'
      })
      .apply(source) as tfjs.SymbolicTensor;
  }

  private _addUpsamplingLayer(source: tfjs.SymbolicTensor) {
    return tfjs.layers
      .upSampling2d({
        size: [2, 2],
        // NCHW
        dataFormat: 'channelsFirst'
      })
      .apply(source) as tfjs.SymbolicTensor;
  }

  buildModel() {
    const channels = 3;

    // TODO input process transferFunc
    // TODO input shape
    const input = tfjs.input({ shape: [channels, 512, 512], dtype: 'float32' });

    const encConv0 = this._createConv('enc_conv0', input, 'reLU');
    const pool1 = this._createPooling(
      this._createConv('enc_conv1', encConv0, 'reLU')
    );
    const pool2 = this._createPooling(
      this._createConv('enc_conv2', pool1, 'reLU')
    );
    const pool3 = this._createPooling(
      this._createConv('enc_conv3', pool2, 'reLU')
    );
    const pool4 = this._createPooling(
      this._createConv('enc_conv4', pool3, 'reLU')
    );
    const encConv5a = this._createConv('enc_conv5a', pool4, 'reLU');
    const upsample4 = this._createConv('enc_conv5b', encConv5a, 'reLU');
    const decConv4a = this._createConcatConv('dec_conv4a', upsample4, pool3);
    const upsample3 = this._addUpsamplingLayer(
      this._createConv('dec_conv4b', decConv4a, 'reLU')
    );
    const decConv3a = this._createConcatConv('dec_conv3a', upsample3, pool2);
    const upsample2 = this._addUpsamplingLayer(
      this._createConv('dec_conv3b', decConv3a, 'reLU')
    );
    const decConv2a = this._createConcatConv('dec_conv2a', upsample2, pool1);
    const upsample1 = this._addUpsamplingLayer(
      this._createConv('dec_conv2b', decConv2a, 'reLU')
    );
    const decConv1a = this._createConcatConv('dec_conv1a', upsample1, input);
    const decConv1b = this._createConv('dec_conv1b', decConv1a, 'reLU');
    const decConv0 = this._createConv('dec_conv0', decConv1b, 'reLU');

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
      [3, image.width, image.height],
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
