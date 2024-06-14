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

export interface WGPUComputePassOutput {
  channels: number;
}

export class WGPUComputePass<I extends string, O extends string> {
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

  private _inputs: string[] = [];
  private _outputs: string[] = [];
  private _outputsParams: Record<string, WGPUComputePassOutput> = {};
  private _uniforms: Uniform[] = [];
  private _uniformBuffers: Record<string, GPUBuffer> = {};

  private _width = 10;
  private _height = 10;

  private _execWidth?: number;
  private _execHeight?: number;

  private _csCode = '';
  private _csMain;
  private _csDefine;

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

  setSize(width: number, height: number) {
    width = Math.ceil(width);
    height = Math.ceil(height);
    const sizeChanged = width !== this._width || height !== this._height;
    this._width = width;
    this._height = height;
    if (sizeChanged) {
      this._resizeOutputBuffers();
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
    this._outputsParams = outputParams;
    this._updateOutputBuffers(outputParams);
    this._needsUpdatePipeline = true;
  }

  setUniform(label: string, data: Float32Array | Int32Array | Uint32Array) {
    const buffer = this._uniformBuffers[label];
    this._device.queue.writeBuffer(buffer, 0, data);
  }

  getOutputBuffer(name: O) {
    return this._outputBuffers[name].buffer;
  }

  dispose() {
    Object.keys(this._uniformBuffers).forEach((key) => {
      (this._uniformBuffers as any)[key].destroy();
    });
    Object.keys(this._outputBuffers).forEach((key) => {
      (this._outputBuffers as any)[key].texture.destroy();
    });
  }

  private _createBuffer(params: WGPUComputePassOutput) {
    const byteLength = this._width * this._height * params.channels * 4;
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
    const outputTextures = this._outputBuffers;
    for (const key in outputTextures) {
      const { buffer, params } = outputTextures[key];
      buffer.destroy();
      outputTextures[key].buffer = this._createBuffer(params);
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

  private _updatePipeline() {
    if (!this._needsUpdatePipeline) {
      return;
    }
    this._needsUpdatePipeline = false;
    const device = this._device;
    const csCode = this._getFullCs();
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

  private _getFullCs() {
    const inputs = this._inputs;
    const hasInputs = inputs.length > 0;
    const cs = `
${inputs
  .sort()
  .map(
    (bufferName, idx) =>
      // TODO more channels option.
      `@group(0) @binding(${idx}) var<storage, read> in_${bufferName}: array<vec4f>;`
  )
  .join('\n')}
${this._uniforms
  .map(
    (uniform, idx) =>
      `@group(${hasInputs ? 1 : 0}) @binding(${idx}) var<uniform> ${
        uniform.label
      }: ${uniform.type};`
  )
  .join('\n')}

${this._outputs
  .map(
    (name, idx) =>
      `@group(${
        hasInputs ? 2 : 1
      }) @binding(${idx}) var<storage, read_write> out_${name}: array<vec${
        this._outputsParams[name].channels
      }f>;`
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

    //TODO
    const uniformBindGroupIndex = this._inputs.length > 0 ? 1 : 0;
    if (this._uniforms.length > 0) {
      bindGroups[uniformBindGroupIndex] = device.createBindGroup({
        label: this._label,
        layout: this._pipeline.getBindGroupLayout(uniformBindGroupIndex),
        entries: this._uniforms.map(
          (uniform, idx) =>
            ({
              binding: idx,
              resource: {
                buffer: this._uniformBuffers[uniform.label]
              }
            } satisfies GPUBindGroupEntry)
        )
      });
    }

    this._bindGroups = bindGroups;
  }

  createPass(
    commandEncoder: GPUCommandEncoder,
    inputBuffers: Record<I, GPUBuffer>
  ) {
    this._updatePipeline();

    // TODO createBindGroup every time?
    if (this._inputs.length > 0) {
      this._bindGroups[0] = this._device.createBindGroup({
        label: this._label,
        layout: this._pipeline.getBindGroupLayout(0),
        entries: this._inputs.map((bufferName, idx) => ({
          binding: idx,
          // TODO
          resource: {
            buffer: inputBuffers[bufferName as I]
          }
        }))
      });
    }
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
