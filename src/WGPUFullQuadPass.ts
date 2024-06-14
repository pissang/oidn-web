export const fullScreenQuadVertexShaderWGSL = /*wgsl */ `
@vertex
fn main(
  @builtin(vertex_index) vertexIndex: u32
) -> @builtin(position) vec4f {
  const pos = array(
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
    vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
  );

  return vec4<f32>(pos[vertexIndex], 0.0, 1.0);
}
`;

function isTextureParamsEqual(
  params: WGPUFullQuadPassOutput,
  other: WGPUFullQuadPassOutput
) {
  return params.format === other.format;
}

export interface Uniform {
  label: string;
  type: string;
  data: Float32Array | Int32Array | Uint32Array;
}

export interface WGPUFullQuadPassOutput {
  format: GPUTextureFormat;
}
export class WGPUFullQuadPass<I extends string, O extends string> {
  private _label;

  private _device;
  private _outputTextures: Record<
    string,
    {
      texture: GPUTexture;
      params: WGPUFullQuadPassOutput;
    }
  > = {};

  private _pipeline!: GPURenderPipeline;
  private _bindGroups: GPUBindGroup[] = [];
  private _needsUpdatePipeline = true;
  /**
   * When render to canvas
   */
  private _renderToScreen?: {
    screenTexture: GPUTexture;
    presentationFormat: GPUTextureFormat;
  };
  private _inputs: string[] = [];
  private _outputs: string[] = [];
  private _uniforms: Uniform[] = [];
  private _uniformBuffers: Record<string, GPUBuffer> = {};

  private _width = 10;
  private _height = 10;

  private _fsCode = '';
  private _fsMain;
  private _fsDefine;

  constructor(
    label: string,
    device: GPUDevice,
    opts: {
      inputs: I[];
      outputs: O[];
      fsMain: string;
      fsDefine?: string;
      uniforms: Uniform[];
    }
  ) {
    this._label = label;
    this._device = device;
    this._fsMain = opts.fsMain;
    this._fsDefine = opts.fsDefine;
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
      this._resizeOutputTextures();
      this._needsUpdatePipeline = true;
    }
  }

  setOutputParams(outputParams: Record<O, WGPUFullQuadPassOutput>) {
    this._renderToScreen = undefined;
    this._updateOutputTextures(outputParams);
    this._needsUpdatePipeline = true;
  }

  setRenderToScreen(
    screenTexture: GPUTexture,
    presentationFormat: GPUTextureFormat
  ) {
    this._renderToScreen = {
      screenTexture,
      presentationFormat
    };
  }

  setUniform(label: string, data: Float32Array | Int32Array | Uint32Array) {
    const buffer = this._uniformBuffers[label];
    this._device.queue.writeBuffer(buffer, 0, data);
  }

  getOutputTexture(name: O) {
    return this._outputTextures[name].texture;
  }

  dispose() {
    Object.keys(this._uniformBuffers).forEach((key) => {
      (this._uniformBuffers as any)[key].destroy();
    });
    Object.keys(this._outputTextures).forEach((key) => {
      (this._outputTextures as any)[key].texture.destroy();
    });
  }

  private _createTexture(params: WGPUFullQuadPassOutput) {
    return this._device.createTexture({
      label: this._label,
      size: {
        width: this._width,
        height: this._height,
        depthOrArrayLayers: 1
      },
      format: params.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    });
  }

  private _resizeOutputTextures() {
    const outputTextures = this._outputTextures;
    for (const key in outputTextures) {
      const { texture, params } = outputTextures[key];
      texture.destroy();
      outputTextures[key].texture = this._createTexture(params);
    }
  }

  private _updateOutputTextures(
    outputParams: Record<string, WGPUFullQuadPassOutput>
  ) {
    const outputTextures = this._outputTextures;
    for (const key in outputParams) {
      const params = outputParams[key];
      if (
        !isTextureParamsEqual(
          params,
          outputTextures[key]?.params || ({} as WGPUFullQuadPassOutput)
        )
      ) {
        outputTextures[key]?.texture.destroy();
        const texture = this._createTexture(params);
        outputTextures[key] = {
          texture,
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
    const fsCode = this._getFullFs();
    if (fsCode === this._fsCode) {
      return;
    }
    const { screenTexture, presentationFormat } = this._renderToScreen || {};
    this._fsCode = fsCode;
    this._pipeline = device.createRenderPipeline({
      label: this._label,
      layout: 'auto',
      vertex: {
        module: device.createShaderModule({
          label: this._label,
          code: fullScreenQuadVertexShaderWGSL
        }),
        entryPoint: 'main'
      },
      fragment: {
        module: device.createShaderModule({
          label: this._label,
          code: fsCode
        }),
        entryPoint: 'main',
        targets: screenTexture
          ? [
              {
                format: presentationFormat!
              }
            ]
          : this._outputs.map((key) => ({
              format: this._outputTextures[key].params.format
            }))
      },
      primitive: {
        topology: 'triangle-list'
      }
    });
    this._updateBindGroups();
  }

  private _getFullFs() {
    const inputs = this._inputs;
    const hasInputs = inputs.length > 0;
    const fs = `
${inputs
  .sort()
  .map(
    (textureName, idx) =>
      `@group(0) @binding(${idx}) var ${textureName}: texture_2d<f32>;`
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

struct FSOutput {
${this._outputs
  .map((name, idx) => `@location(${idx}) ${name}: vec4f,`)
  .join('\n')}
}
${this._fsDefine ?? ''}
@fragment
fn main(
  @builtin(position) coord: vec4f
) -> FSOutput {
  var uv = vec2i(floor(coord.xy));
  var output: FSOutput;
${this._fsMain}
  return output;
}
`;

    return fs;
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
    inputTextures: Record<I, GPUTexture>
  ) {
    this._updatePipeline();

    // TODO createBindGrou every time?
    if (this._inputs.length > 0) {
      this._bindGroups[0] = this._device.createBindGroup({
        label: this._label,
        layout: this._pipeline.getBindGroupLayout(0),
        entries: this._inputs.map((textureName, idx) => ({
          binding: idx,
          // TODO
          resource: inputTextures[textureName as I].createView()
        }))
      });
    }
    // Begin the render pass
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: this._renderToScreen
        ? [
            {
              view: this._renderToScreen.screenTexture.createView(),
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
              storeOp: 'store' as GPUStoreOp,
              loadOp: 'clear' as GPULoadOp
            }
          ]
        : this._outputs.map(
            (textureName) =>
              ({
                view: this._outputTextures[textureName].texture.createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                loadOp: 'clear' as GPULoadOp,
                storeOp: 'store' as GPUStoreOp
              } satisfies GPURenderPassColorAttachment)
          )
    });

    // Draw a full quad
    renderPass.setPipeline(this._pipeline);
    // Bind groups
    this._bindGroups.forEach((bindGroup, idx) => {
      renderPass.setBindGroup(idx, bindGroup);
    });
    renderPass.draw(6, 1, 0, 0);
    // End the render pass
    renderPass.end();
  }
}
