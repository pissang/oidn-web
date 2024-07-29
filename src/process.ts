import { buffer } from '@tensorflow/tfjs';
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
  private _inputPassAux;
  private _inputPassColor;
  private _outputPass;

  private _isInputTexture = false;

  constructor(
    private _device: GPUDevice,
    private _isHDR: boolean,
    private _denoiseAlpha: boolean
  ) {
    const commonUniforms = [
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
    ];
    this._inputPassAux = new WGPUComputePass('inputPassAux', this._device, {
      inputs: ['color', 'albedo', 'normal'],
      outputs: ['color', 'albedo', 'normal'],
      uniforms: commonUniforms,
      csDefine: '',
      csMain: ``
    });
    this._inputPassColor = new WGPUComputePass('inputPassColor', this._device, {
      inputs: ['color'],
      outputs: ['color'],
      uniforms: commonUniforms,
      csDefine: '',
      csMain: ``
    });
    this._outputPass = new WGPUComputePass('outputPass', this._device, {
      inputs: ['color', 'raw'],
      outputs: ['color'],
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
      csDefine: '',
      csMain: ``
    });
    // TODO input scale
    this._inputPassAux.setOutputParams({
      color: { channels: 3 },
      albedo: { channels: 3 },
      normal: { channels: 3 }
    });
    this._inputPassColor.setOutputParams({
      color: { channels: 3 }
    });
    // TODO input scale
    this._outputPass.setOutputParams({
      color: { channels: 4 }
    });
  }

  private _updatePasses(isInputTexture: boolean) {
    if (this._isInputTexture === isInputTexture) {
      return;
    }

    this._isInputTexture = isInputTexture;
    const isHDR = this._isHDR;
    const denoiseAlpha = this._denoiseAlpha;
    const commonCSDefine = /* wgsl */ `
${constsCode}
fn PUForward(y: f32) -> f32 {
  if (y <= y0) {
    return a * y;
  } else if (y <= y1) {
    return b * pow(y, c) + d;
  } else {
    return e * log(y + f) + g;
  }
}`;
    function readInputCode(inputName: string) {
      return isInputTexture
        ? `textureLoad(in_${inputName}, globalId.xy + vec2u(inputOffset), 0)`
        : `in_${inputName}[inIdx]'`;
    }
    const commonCSMain = /* wgsl */ `
let x = f32(globalId.x);
let y = f32(globalId.y);
let inIdx = i32((y + inputOffset.y) * inputSize.x + (x + inputOffset.x));
let col = ${readInputCode('color')};

let outIdx = i32(y * outputSize.x + x);

if (${denoiseAlpha}) {
  out_color[outIdx] = vec3f(col.a);
}
else if (${isHDR}) {
  out_color[outIdx] = vec3f(PUForward(col.r * inputScale), PUForward(col.g * inputScale), PUForward(col.b * inputScale)) * normScale;
}
else {
  out_color[outIdx] = col.rgb;
}
`;
    this._inputPassAux.setCSCode({
      csDefine: commonCSDefine,
      csMain: /* wgsl */ `
${commonCSMain}
let alb = ${readInputCode('albedo')};
let nor = ${readInputCode('normal')};
out_normal[outIdx] = nor.rgb;
out_albedo[outIdx] = alb.rgb;
  `
    });

    this._inputPassColor.setCSCode({
      csDefine: commonCSDefine,
      csMain: /* wgsl */ `
${commonCSMain}
`
    });
    this._outputPass.setCSCode({
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
let inIdx = i32((y + inputOffset.y) * inputSize.x + x + inputOffset.x);
let outIdx = i32((y + outputOffset.y) * imageSize.x + x + outputOffset.x);
let col = in_color[inIdx];
let raw = ${
        isInputTexture
          ? 'textureLoad(in_raw, globalId.xy + vec2u(outputOffset), 0)'
          : 'in_raw[outIdx]'
      };

if (${denoiseAlpha}) {
  out_color[outIdx] = vec4f(raw.rgb, col.r);
}
else if (${isHDR}) {
  out_color[outIdx] = vec4f(
    vec3f(PUInverse(col.r * rcpNormScale), PUInverse(col.g * rcpNormScale), PUInverse(col.b * rcpNormScale)) / inputScale,
    // Pick the alpha
    raw.a
  );
}
else {
  out_color[outIdx] = vec4f(col.rgb, raw.a);
}
`
    });
  }

  setImageSize(w: number, h: number) {
    this._inputPassAux.setUniform('inputSize', new Float32Array([w, h]));
    this._inputPassColor.setUniform('inputSize', new Float32Array([w, h]));
    this._outputPass.setUniform('imageSize', new Float32Array([w, h]));
    this._outputPass.setSize(w, h);
  }

  setInputTile(tile: Tile) {
    const size = new Float32Array([tile.width, tile.height]);
    [this._inputPassAux, this._inputPassColor].forEach((inputPass) => {
      inputPass.setUniform('inputOffset', new Float32Array([tile.x, tile.y]));
      inputPass.setUniform('outputSize', size);
      inputPass.setSize(size[0], size[1]);
    });

    this._outputPass.setUniform('inputSize', size);
  }

  setOutputTile(dstTile: Tile, srcTile: Tile) {
    const outputPass = this._outputPass;
    const size = new Float32Array([dstTile.width, dstTile.height]);
    const dx = dstTile.x - srcTile.x;
    const dy = dstTile.y - srcTile.y;
    outputPass.setUniform('outputSize', size);
    outputPass.setUniform('inputOffset', new Float32Array([dx, dy]));
    outputPass.setUniform(
      'outputOffset',
      new Float32Array([dstTile.x, dstTile.y])
    );
    outputPass.setExecuteSize(size[0], size[1]);
  }

  forward(
    colorBuffer: GPUBuffer | GPUTexture,
    // TODO optional albedo and normal.
    albedoBuffer: GPUBuffer | GPUTexture | undefined,
    normalBuffer: GPUBuffer | GPUTexture | undefined
  ) {
    const isInputTexture = colorBuffer instanceof GPUTexture;
    this._updatePasses(isInputTexture);

    const inputPassAux = this._inputPassAux;
    const inputPassColor = this._inputPassColor;
    const commandEncoder = this._device.createCommandEncoder();

    function createInput(bufferOrTex: GPUBuffer | GPUTexture) {
      return bufferOrTex instanceof GPUTexture
        ? {
            texture: bufferOrTex,
            channels: 4
          }
        : {
            buffer: bufferOrTex,
            channels: 4
          };
    }
    if (albedoBuffer && normalBuffer) {
      inputPassAux.createPass(commandEncoder, {
        color: createInput(colorBuffer),
        albedo: createInput(albedoBuffer),
        normal: createInput(normalBuffer)
      });
    } else {
      inputPassColor.createPass(commandEncoder, {
        color: createInput(colorBuffer)
      });
    }
    this._device.queue.submit([commandEncoder.finish()]);

    return albedoBuffer && normalBuffer
      ? {
          color: inputPassAux.getOutput('color'),
          albedo: inputPassAux.getOutput('albedo'),
          normal: inputPassAux.getOutput('normal')
        }
      : {
          color: inputPassColor.getOutput('color')
        };
  }

  inverse(buffer: GPUBuffer, raw: GPUBuffer | GPUTexture) {
    const device = this._device;

    const commandEncoder = device.createCommandEncoder();
    const outputGPUPass = this._outputPass;

    outputGPUPass.createPass(commandEncoder, {
      color: { buffer: buffer, channels: 4 },
      raw:
        raw instanceof GPUBuffer
          ? { buffer: raw, channels: 4 }
          : { texture: raw, channels: 4 }
    });
    this._device.queue.submit([commandEncoder.finish()]);

    return outputGPUPass.getOutput('color');
  }

  copyInputDataToOutput(inputColorBuffer: GPUBuffer | GPUTexture) {
    if (inputColorBuffer instanceof GPUTexture) {
      // TODO
      return;
    }
    const encoder = this._device.createCommandEncoder();
    const outputGPUPass = this._outputPass;
    const colorBuffer = outputGPUPass.getOutput('color');
    encoder.copyBufferToBuffer(
      inputColorBuffer,
      0,
      colorBuffer,
      0,
      colorBuffer.size
    );

    this._device.queue.submit([encoder.finish()]);
  }

  dispose() {
    this._outputPass.dispose();
    this._inputPassAux.dispose();
  }
}
