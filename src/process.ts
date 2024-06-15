import { WGPUComputePass } from './WGPUComputePass';

const a = 1.41283765e3;
const b = 1.64593172;
const c = 4.31384981e-1;
const d = -2.94139609e-3;
const e = 1.92653254e-1;
const f = 6.26026094e-3;
const g = 9.98620152e-1;
const y0 = 1.5794576e-6;
const y1 = 3.22087631e-2;
const x0 = 2.23151711e-3;
const x1 = 3.70974749e-1;

function PUForward(y: number) {
  if (y <= y0) {
    y = a * y;
  } else if (y <= y1) {
    y = b * Math.pow(y, c) + d;
  } else {
    y = e * Math.log(y + f) + g;
  }
  return y;
}

function PUInverse(x: number) {
  if (x <= x0) {
    x = x / a;
  } else if (x <= x1) {
    x = Math.pow((x - d) / b, 1 / c);
  } else {
    x = Math.exp((x - g) / e) - f;
  }
  return x;
}
const yMax = 65504;
const xMax = PUForward(yMax);
const normScale = 1 / xMax;
const rcpNormScale = xMax;

export class Tile {
  constructor(
    public x: number,
    public y: number,
    public width: number,
    public height: number
  ) {}
}

// TODO put in gpu
export function avgLogLum({
  data,
  channels
}: {
  data: Float32Array;
  channels: number;
}) {
  let totalLuminance = 0;

  for (let i = 0; i < data.length; i += channels) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const lum = 0.212671 * r + 0.71516 * g + 0.072169 * b;

    // TODO performance
    // https://github.com/RenderKit/oidn/blob/713ec7838ba650f99e0a896549c0dca5eeb3652d/devices/cpu/cpu_autoexposure.cpp#L43
    totalLuminance += Math.log2(lum + 0.0001);
  }
  const numPixels = data.length / channels;
  const averageLuminance = totalLuminance / numPixels;
  const key = 0.18;

  // https://github.com/RenderKit/oidn/blob/713ec7838ba650f99e0a896549c0dca5eeb3652d/training/color.py#L173
  return key / Math.pow(2, averageLuminance);
}

export function hdrTransferFuncCPU({
  data,
  channels,
  inputScale
}: {
  data: Float32Array;
  channels: number;
  inputScale: number;
}) {
  const newData = new Float32Array(data.length);
  newData.set(data);
  // https://github.com/RenderKit/oidn/blob/713ec7838ba650f99e0a896549c0dca5eeb3652d/core/color.h#L71
  for (let i = 0; i < newData.length; i += channels) {
    // First three are color
    for (let c = 0; c < 3; c++) {
      let y = newData[i + c] * inputScale;
      // https://github.com/RenderKit/oidn/blob/713ec7838ba650f99e0a896549c0dca5eeb3652d/devices/cpu/color.ispc#L135
      newData[i + c] = PUForward(y) * normScale;
    }
  }

  return newData;
}

export function hdrTransferFuncInverseCPU({
  data,
  channels,
  inputScale
}: {
  data: Float32Array;
  channels: number;
  inputScale: number;
}) {
  const newData = new Float32Array(data.length);
  newData.set(data);

  const outputScale = 1 / inputScale;
  for (let i = 0; i < newData.length; i += channels) {
    for (let c = 0; c < 3; c++) {
      let x = newData[i + c] * rcpNormScale;
      newData[i + c] = PUInverse(x) * outputScale;
    }
  }

  return newData;
}

const constsCode = `
const a = ${a};
const b = ${b};
const c = ${c};
const d = ${d};
const e = ${e};
const f = ${f};
const g = ${g};
const y0 =${y0};
const y1 =${y1};
const x0 =${x0};
const x1 =${x1};

const normScale = ${normScale};
const rcpNormScale = ${rcpNormScale};
`;

export class GPUDataProcess {
  private _inputPass;
  private _outputPass;
  constructor(private _device: GPUDevice) {
    this._inputPass = new WGPUComputePass('inputPass', this._device, {
      inputs: ['color', 'albedo', 'normal'],
      outputs: ['color', 'albedo', 'normal'],
      uniforms: [
        {
          label: 'inputScale',
          type: 'f32',
          data: new Float32Array([1])
        },
        {
          label: 'inputSize',
          type: 'vec2<f32>',
          data: new Float32Array(2)
        },
        {
          label: 'outputSize',
          type: 'vec2<f32>',
          data: new Float32Array(2)
        },
        {
          label: 'inputOffset',
          type: 'vec2<f32>',
          data: new Float32Array(2)
        }
      ],
      csDefine: /* wgsl */ `
${constsCode}
fn PUForward(y: f32) -> f32 {
  if (y <= y0) {
    return a * y;
  } else if (y <= y1) {
    return b * pow(y, c) + d;
  } else {
    return e * log(y + f) + g;
  }
}
`,
      csMain: /* wgsl */ `
let x = f32(globalId.x);
let y = f32(globalId.y);
let inIdx = i32((y + inputOffset.y) * inputSize.x + (x + inputOffset.x));
let col = in_color[inIdx] * inputScale;
let alb = in_albedo[inIdx];
let nor = in_normal[inIdx];

let outIdx = i32(y * outputSize.x + x);
out_color[outIdx] = vec3f(PUForward(col.r), PUForward(col.g), PUForward(col.b)) * normScale;
out_normal[outIdx] = nor.rgb;
out_albedo[outIdx] = alb.rgb;
`
    });

    this._outputPass = new WGPUComputePass('outputPass', this._device, {
      inputs: ['color'],
      outputs: ['color'],
      uniforms: [
        {
          // TODO inputScale from avg log lum
          label: 'inputScale',
          type: 'f32',
          data: new Float32Array([1])
        },
        {
          label: 'inputSize',
          type: 'vec2<f32>',
          data: new Float32Array(2)
        },
        {
          label: 'outputSize',
          type: 'vec2<f32>',
          data: new Float32Array(2)
        },
        {
          label: 'imageSize',
          type: 'vec2<f32>',
          data: new Float32Array(2)
        },
        {
          label: 'inputOffset',
          type: 'vec2<f32>',
          data: new Float32Array(2)
        },
        {
          label: 'outputOffset',
          type: 'vec2<f32>',
          data: new Float32Array(2)
        }
      ],
      csDefine: /* wgsl */ `
${constsCode}
fn PUInverse(y: f32) -> f32 {
  if (y <= x0) {
    return y / a;
  } else if (y <= x1) {
    return pow((y - d) / b, 1 / c);
  } else {
    return exp((y - g) / e) - f;
  }
}
`,
      csMain: /* wgsl */ `
let x = f32(globalId.x);
let y = f32(globalId.y);
if (x >= outputSize.x || y >= outputSize.y) {
  return;
}
let inIdx = i32((y + inputOffset.y) * inputSize.x + (x + inputOffset.x));
let outIdx = i32((y + outputOffset.y) * imageSize.x + (x + outputOffset.x));
let col = in_color[inIdx] * rcpNormScale;
// TODO alpha
out_color[outIdx] = vec4f(vec3f(PUInverse(col.r), PUInverse(col.g), PUInverse(col.b)) / inputScale, 1.0);
`
    });
  }

  setImageSize(w: number, h: number) {
    this._inputPass.setUniform('inputSize', new Float32Array([w, h]));
    this._outputPass.setUniform('imageSize', new Float32Array([w, h]));
    this._outputPass.setSize(w, h);
  }

  setInputTile(tile: Tile) {
    const inputPass = this._inputPass;
    const size = new Float32Array([tile.width, tile.height]);
    inputPass.setUniform('inputOffset', new Float32Array([tile.x, tile.y]));
    inputPass.setUniform('outputSize', size);
    inputPass.setSize(size[0], size[1]);

    this._outputPass.setUniform('inputSize', size);
  }

  setOutputTile(tile: Tile) {
    const outputPass = this._outputPass;
    const size = new Float32Array([tile.width, tile.height]);
    outputPass.setUniform('outputSize', size);
    outputPass.setUniform('inputOffset', new Float32Array([tile.x, tile.y]));
    outputPass.setExecuteSize(size[0], size[1]);
  }

  forward(
    colorBuffer: GPUBuffer,
    // TODO optional albedo and normal.
    albedoBuffer: GPUBuffer,
    normalBuffer: GPUBuffer
  ) {
    const inputGPUPass = this._inputPass;
    // TODO input scale
    inputGPUPass.setOutputParams({
      color: { channels: 3 },
      albedo: { channels: 3 },
      normal: { channels: 3 }
    });

    const commandEncoder = this._device.createCommandEncoder();
    inputGPUPass.createPass(commandEncoder, {
      color: { buffer: colorBuffer, channels: 4 },
      albedo: { buffer: albedoBuffer, channels: 4 },
      normal: { buffer: normalBuffer, channels: 4 }
    });
    this._device.queue.submit([commandEncoder.finish()]);

    return {
      color: inputGPUPass.getOutputBuffer('color'),
      albedo: inputGPUPass.getOutputBuffer('albedo'),
      normal: inputGPUPass.getOutputBuffer('normal')
    };
  }

  inverse(buffer: GPUBuffer) {
    const device = this._device;

    const commandEncoder = device.createCommandEncoder();
    const outputGPUPass = this._outputPass;
    // TODO input scale
    outputGPUPass.setOutputParams({
      color: { channels: 4 }
    });

    outputGPUPass.createPass(commandEncoder, {
      color: { buffer: buffer, channels: 4 }
    });
    this._device.queue.submit([commandEncoder.finish()]);

    return outputGPUPass.getOutputBuffer('color');
  }
}
