function isStorageParamsEqual(
  params: WGPUComputePassOutput,
  other: WGPUComputePassOutput
) {
  return params.channels === other.channels;
}

export const WORKGROUP_SIZE = 8;

export interface Uniform {
  label: string;
  type: string;
  data: Float32Array | Int32Array | Uint32Array;
}
export interface WGPUComputePassInput {
  buffer?: GPUBuffer;
  texture?: GPUTexture;
  channels: number;
}

export interface WGPUComputePassOutput {
  channels: number;
}

export class WGPUComputePass<I extends string, O extends string> {
  private autoUpdateOutputBuffer = true;

  private _label;

  private _device;
  private _outputBuffers: Record<
    string,
    {
      buffer: GPUBuffer;
      params: WGPUComputePassOutput;
    }
  > = {};

  private _pipeline!: GPUComputePipeline;
  private _bindGroups: GPUBindGroup[] = [];
  private _needsUpdatePipeline = true;
  private _needsResizeBuffer = true;

  private _inputs: string[] = [];
  private _outputs: string[] = [];
  private _uniforms: Uniform[] = [];
  private _uniformBuffers: Record<string, GPUBuffer> = {};

  private _width = 10;
  private _height = 10;

  private _execWidth?: number;
  private _execHeight?: number;

  private _csCode = '';
  private _csMain;
  private _csDefine;

  private _groupOffsets = {
    inputs: 0,
    uniforms: 1,
    outputs: 2
  };

  constructor(
    label: string,
    device: GPUDevice,
    opts: {
      inputs: I[];
      outputs: O[];
      csMain: string;
      csDefine?: string;
      uniforms: Uniform[];
    }
  ) {
    this._label = label;
    this._device = device;
    this._csMain = opts.csMain;
    this._csDefine = opts.csDefine;
    this._inputs = opts.inputs;
    this._outputs = opts.outputs;
    this._uniforms = opts.uniforms;

    opts.uniforms.forEach((uniform) => {
      this._uniformBuffers[uniform.label] = device.createBuffer({
        label: this._label,
        size: uniform.data.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });
      this._device.queue.writeBuffer(
        this._uniformBuffers[uniform.label],
        0,
        uniform.data
      );
    });
  }

  setCSCode({ csDefine, csMain }: { csDefine: string; csMain: string }) {
    this._csDefine = csDefine;
    this._csMain = csMain;
    this._needsUpdatePipeline = true;
  }

  setSize(width: number, height: number) {
    width = Math.ceil(width);
    height = Math.ceil(height);
    const sizeChanged = width !== this._width || height !== this._height;
    this._width = width;
    this._height = height;
    if (sizeChanged) {
      this._needsResizeBuffer = true;
      this._needsUpdatePipeline = true;
    }
  }

  setExecuteSize(width: number, height: number) {
    width = Math.ceil(width);
    height = Math.ceil(height);
    this._execWidth = width;
    this._execHeight = height;
  }

  setOutputParams(outputParams: Record<O, WGPUComputePassOutput>) {
    if (this.autoUpdateOutputBuffer) {
      this._updateOutputBuffers(outputParams);
    }
    this._needsUpdatePipeline = true;
  }

  setOutputBuffers(outputBuffers: Record<O, GPUBuffer>) {
    this._outputBuffers = Object.keys(outputBuffers).reduce((obj, key) => {
      obj[key] = {
        buffer: outputBuffers[key as O],
        params: { channels: 4 }
      };
      return obj;
    }, {} as WGPUComputePass<I, O>['_outputBuffers']);
  }

  setUniform(label: string, data: Float32Array | Int32Array | Uint32Array) {
    const buffer = this._uniformBuffers[label];
    this._device.queue.writeBuffer(buffer, 0, data);
  }

  getOutput(name: O) {
    return this._outputBuffers[name].buffer;
  }

  dispose() {
    Object.keys(this._uniformBuffers).forEach((key) => {
      (this._uniformBuffers as any)[key].destroy();
    });
    Object.keys(this._outputBuffers).forEach((key) => {
      (this._outputBuffers as any)[key].buffer.destroy();
    });
  }

  private _createBuffer(params: WGPUComputePassOutput) {
    // const byteLength = this._width * this._height * params.channels * 4;
    // Buffer data needs to be aligned with 8, 16
    const byteLength = this._width * this._height * 4 * 4;
    return this._device.createBuffer({
      label: this._label,
      // webgpu needs buffer at least 80 bytes.
      size: Math.max(byteLength, 80),
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC
    });
  }

  private _resizeOutputBuffers() {
    const outputBuffers = this._outputBuffers;
    for (const key in outputBuffers) {
      const { buffer, params } = outputBuffers[key];
      buffer.destroy();
      outputBuffers[key].buffer = this._createBuffer(params);
    }
  }

  private _updateOutputBuffers(
    outputParams: Record<string, WGPUComputePassOutput>
  ) {
    const outputBuffers = this._outputBuffers;
    for (const key in outputParams) {
      const params = outputParams[key];
      if (
        !isStorageParamsEqual(
          params,
          outputBuffers[key]?.params || ({} as WGPUComputePassOutput)
        )
      ) {
        outputBuffers[key]?.buffer.destroy();
        const buffer = this._createBuffer(params);
        outputBuffers[key] = {
          buffer,
          params
        };
      }
    }
  }

  private _updatePipeline(
    inputParams: Record<string, WGPUComputePassInput>,
    inputTypes: Record<string, 'texture' | 'buffer'>
  ) {
    if (!this._needsUpdatePipeline) {
      return;
    }
    this._needsUpdatePipeline = false;
    const device = this._device;
    const csCode = this._getFullCs(inputParams, inputTypes);
    if (csCode === this._csCode) {
      return;
    }
    this._csCode = csCode;
    this._pipeline = device.createComputePipeline({
      label: this._label,
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          label: this._label,
          code: csCode
        }),
        entryPoint: 'main'
      }
    });
    this._updateBindGroups();
  }

  private _getFullCs(
    inputParams: Record<string, WGPUComputePassInput>,
    inputTypes: Record<string, 'texture' | 'buffer'>
  ) {
    const inputs = this._inputs;
    const uniforms = this._uniforms;
    let offset = 0;
    const groupOffsets = (this._groupOffsets = {
      inputs: 0,
      uniforms: 0,
      outputs: 0
    });
    if (inputs.length > 0) {
      offset++;
    }
    if (uniforms.length > 0) {
      groupOffsets.uniforms = offset;
      offset++;
    }
    groupOffsets.outputs = offset;
    const cs = `
${inputs
  .sort()
  .map((inputName, idx) => {
    const bindingPrefix = `@group(${groupOffsets.inputs}) @binding(${idx}) `;
    const varName = `in_${inputName}`;
    // TODO more channels option.
    return inputTypes[inputName] === 'texture'
      ? `${bindingPrefix} var ${varName}: texture_2d<f32>;`
      : `${bindingPrefix} var<storage, read> ${varName}: array<vec${inputParams[inputName].channels}f>;`;
  })
  .join('\n')}
${this._uniforms
  .map(
    (uniform, idx) =>
      `@group(${groupOffsets.uniforms}) @binding(${idx}) var<uniform> ${uniform.label}: ${uniform.type};`
  )
  .join('\n')}

${this._outputs
  .map(
    (name, idx) =>
      `@group(${groupOffsets.outputs}) @binding(${idx}) var<storage, read_write> out_${name}: array<vec${this._outputBuffers[name].params.channels}f>;`
  )
  .join('\n')}
${this._csDefine ?? ''}
@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE}, 1)
fn main(@builtin(global_invocation_id) globalId: vec3u) {
${this._csMain}
}
`;

    return cs;
  }

  private _updateBindGroups() {
    const bindGroups: GPUBindGroup[] = [];
    const device = this._device;
    const groupOffsets = this._groupOffsets;

    if (this._uniforms.length > 0) {
      bindGroups[groupOffsets.uniforms] = device.createBindGroup({
        label: this._label,
        layout: this._pipeline.getBindGroupLayout(groupOffsets.uniforms),
        entries: this._uniforms.map(
          (uniform, idx) =>
            ({
              binding: idx,
              resource: {
                buffer: this._uniformBuffers[uniform.label]
              }
            } as GPUBindGroupEntry)
        )
      });
    }

    this._bindGroups = bindGroups;
  }

  createPass(
    commandEncoder: GPUCommandEncoder,
    inputs: Record<I, WGPUComputePassInput>
  ) {
    if (this._needsResizeBuffer && this.autoUpdateOutputBuffer) {
      this._resizeOutputBuffers();
      this._needsResizeBuffer = false;
    }

    const inputTypes = this._inputs.reduce((obj, inputName) => {
      obj[inputName] = inputs[inputName as I].buffer ? 'buffer' : 'texture';
      return obj;
    }, {} as Record<string, 'texture' | 'buffer'>);

    this._updatePipeline(inputs, inputTypes);

    const groupOffsets = this._groupOffsets;
    // TODO createBindGroup every time?
    if (this._inputs.length > 0) {
      this._bindGroups[groupOffsets.inputs] = this._device.createBindGroup({
        label: this._label,
        layout: this._pipeline.getBindGroupLayout(groupOffsets.inputs),
        entries: this._inputs.map((inputName, idx) => ({
          binding: idx,
          // TODO
          resource: inputs[inputName as I].buffer
            ? {
                buffer: inputs[inputName as I].buffer!
              }
            : inputs[inputName as I].texture!.createView()
        }))
      });
    }

    // Outputs
    this._bindGroups[groupOffsets.outputs] = this._device.createBindGroup({
      label: this._label,
      layout: this._pipeline.getBindGroupLayout(groupOffsets.outputs),
      entries: this._outputs.map((bufferName, idx) => ({
        binding: idx,
        resource: {
          buffer: this._outputBuffers[bufferName].buffer
        }
      }))
    });

    // Begin the render pass
    const computePass = commandEncoder.beginComputePass();

    // Draw a full quad
    computePass.setPipeline(this._pipeline);
    // Bind groups
    this._bindGroups.forEach((bindGroup, idx) => {
      computePass.setBindGroup(idx, bindGroup);
    });
    computePass.dispatchWorkgroups(
      Math.ceil((this._execWidth ?? this._width) / WORKGROUP_SIZE),
      Math.ceil((this._execHeight ?? this._height) / WORKGROUP_SIZE),
      1
    );
    // End the render pass
    computePass.end();
  }
}
