import { Float16Array } from '@petamoriken/float16';
import type {
  Conv2DNodeSpec,
  ValidatedUNetModel
} from './modelSpec';
import type { HostTensor } from './tza';
import type {
  NativeUNetPrecision,
  NativeUNetPrecisionSetting
} from './nativeUNet';
import {
  OIDNResourceTracker,
  type OIDNResourceSnapshot
} from './resourceTracker.js';

type MLDataType = 'float16' | 'float32';
type MLOperandLike = object;
type MLGraphLike = { destroy?: () => void; devices?: readonly string[] };
type MLTensorLike = { destroy: () => void };

interface MLContextLike {
  createExportableTensor(
    descriptor: Record<string, unknown>,
    device: GPUDevice
  ): Promise<MLTensorLike>;
  dispatch(
    graph: MLGraphLike,
    inputs: Record<string, MLTensorLike>,
    outputs: Record<string, MLTensorLike>
  ): void;
  exportToGPU(tensor: MLTensorLike): Promise<GPUBuffer>;
  opSupportLimits?: () => Record<string, any>;
  readTensor(tensor: MLTensorLike): Promise<ArrayBuffer>;
  writeTensor(tensor: MLTensorLike, data: ArrayBufferView): void;
  destroy?: () => void;
}

interface MLGraphBuilderLike {
  input(name: string, descriptor: Record<string, unknown>): MLOperandLike;
  constant(
    descriptor: Record<string, unknown>,
    data: ArrayBufferView
  ): MLOperandLike;
  conv2d(
    input: MLOperandLike,
    filter: MLOperandLike,
    options: Record<string, unknown>
  ): MLOperandLike;
  relu(input: MLOperandLike): MLOperandLike;
  maxPool2d(
    input: MLOperandLike,
    options: Record<string, unknown>
  ): MLOperandLike;
  resample2d(
    input: MLOperandLike,
    options: Record<string, unknown>
  ): MLOperandLike;
  concat(inputs: readonly MLOperandLike[], axis: number): MLOperandLike;
  build(outputs: Record<string, MLOperandLike>): Promise<MLGraphLike>;
}

interface WebNNShapeExecution {
  graph: MLGraphLike;
  inputTensor: MLTensorLike;
  outputTensor: MLTensorLike;
  outputBuffer: GPUBuffer;
  inputUniform: GPUBuffer;
  outputUniform: GPUBuffer;
  width: number;
  height: number;
  lastUsed: number;
}

interface WebNNInteropPipelines {
  input: GPUComputePipeline;
  output: GPUComputePipeline;
}

const interopPipelinesByDevice = new WeakMap<
  GPUDevice,
  Map<number, WebNNInteropPipelines>
>();

function interopPipelines(device: GPUDevice, sourceCount: number) {
  let bySourceCount = interopPipelinesByDevice.get(device);
  if (!bySourceCount) {
    bySourceCount = new Map();
    interopPipelinesByDevice.set(device, bySourceCount);
  }
  let pipelines = bySourceCount.get(sourceCount);
  if (!pipelines) {
    const inputModule = device.createShaderModule({
      label: `oidn/webnn/input-pack/${sourceCount}`,
      code: createInputPackShader(sourceCount)
    });
    const outputModule = device.createShaderModule({
      label: 'oidn/webnn/output-unpack',
      code: createOutputUnpackShader()
    });
    pipelines = {
      input: device.createComputePipeline({
        label: `oidn/webnn/input-pack/${sourceCount}`,
        layout: 'auto',
        compute: { module: inputModule, entryPoint: 'main' }
      }),
      output: device.createComputePipeline({
        label: 'oidn/webnn/output-unpack',
        layout: 'auto',
        compute: { module: outputModule, entryPoint: 'main' }
      })
    };
    bySourceCount.set(sourceCount, pipelines);
  }
  return pipelines;
}

export interface WebNNUNetOptions {
  precision?: NativeUNetPrecisionSetting;
  shapeCacheSize?: number;
}

export interface WebNNRuntimeSupport {
  available: boolean;
  reason?: string;
  fp16Conv: boolean;
  gpuInterop: boolean;
}

const WORKGROUP_SIZE = 8;

function roundUp(value: number, alignment: number) {
  return Math.ceil(value / alignment) * alignment;
}

function createMappedBuffer(
  device: GPUDevice,
  label: string,
  data: ArrayBufferView,
  usage: GPUBufferUsageFlags
) {
  const buffer = device.createBuffer({
    label,
    size: roundUp(data.byteLength, 4),
    usage,
    mappedAtCreation: true
  });
  new Uint8Array(buffer.getMappedRange()).set(
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
  );
  buffer.unmap();
  return buffer;
}

function uniformBuffer(
  device: GPUDevice,
  label: string,
  values: readonly number[]
) {
  const data = new Uint32Array(roundUp(values.length, 4));
  data.set(values);
  return createMappedBuffer(device, label, data, GPUBufferUsage.UNIFORM);
}

function hasDataType(
  limits: Record<string, any>,
  op: string,
  operand: string,
  dataType: MLDataType
) {
  return Boolean(
    limits?.[op]?.[operand]?.dataTypes?.includes?.(dataType)
  );
}

function tensorBytes(tensor: HostTensor, precision: NativeUNetPrecision) {
  if (
    precision === 'fp16' &&
    tensor.desc.dataType === 'Float16'
  ) {
    return new Uint8Array(
      tensor.data.buffer,
      tensor.data.byteOffset,
      tensor.data.byteLength
    );
  }
  if (
    precision === 'fp32' &&
    tensor.desc.dataType === 'Float32'
  ) {
    return new Uint8Array(
      tensor.data.buffer,
      tensor.data.byteOffset,
      tensor.data.byteLength
    );
  }

  const source = tensor.desc.dataType === 'Float32'
    ? new Float32Array(
        tensor.data.buffer,
        tensor.data.byteOffset,
        tensor.data.byteLength / 4
      )
    : new Float16Array(
        tensor.data.buffer,
        tensor.data.byteOffset,
        tensor.data.byteLength / 2
      );
  const converted = precision === 'fp16'
    ? new Float16Array(source)
    : new Float32Array(source);
  return new Uint8Array(
    converted.buffer,
    converted.byteOffset,
    converted.byteLength
  );
}

function createInputPackShader(sourceCount: number) {
  const sources = Array.from(
    { length: sourceCount },
    (_, index) =>
      `@group(0) @binding(${index}) var<storage, read> input${index}: array<vec4<f32>>;`
  ).join('\n');
  const branches = Array.from({ length: sourceCount }, (_, index) => {
    const firstChannel = index * 3;
    return `if (channel < ${firstChannel + 3}u) {
      return input${index}[pixel][channel - ${firstChannel}u];
    }`;
  }).join('\n  ');
  return /* wgsl */ `enable f16;
struct Params { width: u32, height: u32, channels: u32, padding: u32 }
${sources}
@group(0) @binding(${sourceCount}) var<storage, read_write> outputData: array<f16>;
@group(0) @binding(${sourceCount + 1}) var<uniform> params: Params;

fn readChannel(pixel: u32, channel: u32) -> f32 {
  ${branches}
  return 0.0;
}

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.channels) {
    return;
  }
  let pixel = gid.y * params.width + gid.x;
  let outputIndex = (gid.z * params.height + gid.y) * params.width + gid.x;
  outputData[outputIndex] = f16(readChannel(pixel, gid.z));
}
`;
}

function createOutputUnpackShader() {
  return /* wgsl */ `enable f16;
struct Params { width: u32, height: u32, padding0: u32, padding1: u32 }
@group(0) @binding(0) var<storage, read> inputData: array<f16>;
@group(0) @binding(1) var<storage, read_write> outputData: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let pixel = gid.y * params.width + gid.x;
  let plane = params.width * params.height;
  outputData[pixel] = vec4<f32>(
    f32(inputData[pixel]),
    f32(inputData[plane + pixel]),
    f32(inputData[plane * 2u + pixel]),
    0.0
  );
}
`;
}

function resolveWebNNPrecision(
  device: GPUDevice,
  requested: NativeUNetPrecisionSetting
): NativeUNetPrecision {
  if (requested === 'fp32') {
    throw new Error(
      'OIDN WebNN GPU interop currently requires FP16 exportable tensors'
    );
  }
  if (!device.features.has('shader-f16')) {
    throw new Error('OIDN WebNN requires shader-f16 on the shared GPUDevice');
  }
  return 'fp16';
}

function activation(
  builder: MLGraphBuilderLike,
  operand: MLOperandLike,
  kind: Conv2DNodeSpec['activation']
) {
  return kind === 'relu' ? builder.relu(operand) : operand;
}

/** Experimental model-driven WebNN executor with FP16 WebGPU interop. */
export class WebNNUNetExecutor {
  readonly precision: NativeUNetPrecision;
  readonly support: WebNNRuntimeSupport;

  private _context!: MLContextLike;
  private _builderConstructor!: new (
    context: MLContextLike
  ) => MLGraphBuilderLike;
  private _shapeCache = new Map<string, WebNNShapeExecution>();
  private _shapePromises = new Map<string, Promise<WebNNShapeExecution>>();
  private _retiredExecutions = new Set<WebNNShapeExecution>();
  private _pendingCreationCount = 0;
  private _shapeCacheSize: number;
  private _clock = 0;
  private _inputPipeline: GPUComputePipeline;
  private _outputPipeline: GPUComputePipeline;
  private _resources = new OIDNResourceTracker();
  private _disposed = false;

  constructor(
    private _device: GPUDevice,
    private _model: ValidatedUNetModel,
    options: WebNNUNetOptions = {}
  ) {
    this.precision = resolveWebNNPrecision(
      _device,
      options.precision ?? 'auto'
    );
    this._shapeCacheSize = Math.max(1, options.shapeCacheSize ?? 2);
    this.support = {
      available: false,
      fp16Conv: false,
      gpuInterop: false
    };

    const pipelines = interopPipelines(
      _device,
      _model.inputChannels / 3
    );
    this._inputPipeline = pipelines.input;
    this._outputPipeline = pipelines.output;
  }

  async prepare() {
    if (this._disposed) throw new Error('OIDN WebNN executor is disposed');
    const webNN = (globalThis.navigator as any)?.ml;
    const Builder = (globalThis as any).MLGraphBuilder;
    if (!webNN?.createContext || typeof Builder !== 'function') {
      this.support.reason = 'WebNN is not exposed by this browser';
      throw new Error(this.support.reason);
    }
    this._builderConstructor = Builder;
    try {
      try {
        // Chromium's experimental implementation only enables WebGPU tensor
        // interop for an explicitly GPU-backed context.
        this._context = await webNN.createContext({
          deviceType: 'gpu',
          powerPreference: 'high-performance'
        });
      } catch {
        this._context = await webNN.createContext({ deviceType: 'gpu' });
      }
      this._resources.track('ml-context', this._context);
      if (this._disposed) throw new Error('OIDN WebNN executor is disposed');

      if (
        typeof this._context.createExportableTensor !== 'function' ||
        typeof this._context.exportToGPU !== 'function'
      ) {
        this.support.reason = 'WebNN WebGPU tensor interop is unavailable';
        throw new Error(this.support.reason);
      }
      const limits = this._context.opSupportLimits?.() ?? {};
      this.support.fp16Conv =
        hasDataType(limits, 'conv2d', 'input', 'float16') &&
        hasDataType(limits, 'conv2d', 'filter', 'float16') &&
        hasDataType(limits, 'conv2d', 'output', 'float16');
      if (!this.support.fp16Conv) {
        this.support.reason = 'WebNN does not support FP16 conv2d';
        throw new Error(this.support.reason);
      }

      let probe: MLTensorLike | undefined;
      let probeBuffer: GPUBuffer | undefined;
      try {
        probe = this._resources.track(
          'ml-tensor',
          await this._context.createExportableTensor(
            { dataType: 'float16', shape: [4] },
            this._device
          )
        );
        probeBuffer = this._resources.track(
          'gpu-buffer',
          await this._context.exportToGPU(probe)
        );
        this.support.gpuInterop = true;
      } catch (error) {
        this.support.reason =
          `WebNN FP16 WebGPU interop failed: ${String(error)}`;
        throw new Error(this.support.reason);
      } finally {
        this._releaseBuffer(probeBuffer);
        this._releaseTensor(probe);
      }
      this.support.available = true;
    } catch (error) {
      this._releaseContext();
      throw error;
    }
  }

  private _constant(
    builder: MLGraphBuilderLike,
    tensor: HostTensor
  ) {
    return builder.constant(
      {
        dataType: 'float16',
        shape: [...tensor.desc.dims]
      },
      tensorBytes(tensor, this.precision)
    );
  }

  private async _createExecution(width: number, height: number) {
    if (this._disposed) throw new Error('OIDN WebNN executor is disposed');
    const builder = new this._builderConstructor(this._context);
    const values = new Map<string, MLOperandLike>();
    const shapes = new Map<string, [number, number, number]>();
    values.set(
      this._model.spec.input,
      builder.input('input', {
        dataType: 'float16',
        shape: [1, this._model.inputChannels, height, width]
      })
    );
    shapes.set(this._model.spec.input, [this._model.inputChannels, height, width]);

    for (const node of this._model.spec.nodes) {
      let result: MLOperandLike;
      let shape: [number, number, number];
      if (node.op === 'conv2d') {
        const inputShape = shapes.get(node.input)!;
        const tensors = this._model.convTensors.get(node.id)!;
        const convolution = builder.conv2d(
          values.get(node.input)!,
          this._constant(builder, tensors.weight),
          {
            bias: this._constant(builder, tensors.bias),
            padding: [1, 1, 1, 1],
            inputLayout: 'nchw',
            filterLayout: 'oihw'
          }
        );
        result = activation(builder, convolution, node.activation);
        shape = [tensors.outputChannels, inputShape[1], inputShape[2]];
      } else if (node.op === 'maxPool2d') {
        const inputShape = shapes.get(node.input)!;
        result = builder.maxPool2d(values.get(node.input)!, {
          windowDimensions: [2, 2],
          strides: [2, 2],
          padding: [0, inputShape[1] % 2, 0, inputShape[2] % 2],
          layout: 'nchw'
        });
        shape = [
          inputShape[0],
          Math.ceil(inputShape[1] / 2),
          Math.ceil(inputShape[2] / 2)
        ];
      } else if (node.op === 'upsample2d') {
        const inputShape = shapes.get(node.input)!;
        result = builder.resample2d(values.get(node.input)!, {
          mode: 'nearest-neighbor',
          axes: [2, 3],
          scales: [2, 2]
        });
        shape = [inputShape[0], inputShape[1] * 2, inputShape[2] * 2];
      } else {
        const inputShapes = node.inputs.map((input) => shapes.get(input)!);
        if (
          inputShapes.some(
            (candidate) =>
              candidate[1] !== inputShapes[0][1] ||
              candidate[2] !== inputShapes[0][2]
          )
        ) {
          throw new Error(
            `WebNN concat ${node.id} has mismatched spatial shapes`
          );
        }
        result = builder.concat(
          node.inputs.map((input) => values.get(input)!),
          1
        );
        shape = [
          inputShapes.reduce((sum, candidate) => sum + candidate[0], 0),
          inputShapes[0][1],
          inputShapes[0][2]
        ];
      }
      values.set(node.id, result);
      shapes.set(node.id, shape);
    }

    let graph: MLGraphLike | undefined;
    let inputTensor: MLTensorLike | undefined;
    let outputTensor: MLTensorLike | undefined;
    let outputBuffer: GPUBuffer | undefined;
    let inputUniform: GPUBuffer | undefined;
    let outputUniform: GPUBuffer | undefined;
    try {
      graph = this._resources.track(
        'ml-graph',
        await builder.build({
          output: values.get(this._model.spec.output)!
        })
      );
      inputTensor = this._resources.track(
        'ml-tensor',
        await this._context.createExportableTensor(
          {
            dataType: 'float16',
            shape: [1, this._model.inputChannels, height, width],
            writable: true
          },
          this._device
        )
      );
      outputTensor = this._resources.track(
        'ml-tensor',
        await this._context.createExportableTensor(
          {
            dataType: 'float16',
            shape: [1, this._model.outputChannels, height, width],
            readable: true
          },
          this._device
        )
      );
      outputBuffer = this._resources.track(
        'gpu-buffer',
        this._device.createBuffer({
          label: `oidn/webnn/output/${width}x${height}`,
          size: width * height * 4 * 4,
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST
        })
      );
      inputUniform = this._resources.track(
        'gpu-buffer',
        uniformBuffer(
          this._device,
          `oidn/webnn/input/${width}x${height}`,
          [width, height, this._model.inputChannels]
        )
      );
      outputUniform = this._resources.track(
        'gpu-buffer',
        uniformBuffer(
          this._device,
          `oidn/webnn/output/${width}x${height}`,
          [width, height]
        )
      );
      const execution: WebNNShapeExecution = {
        graph,
        inputTensor,
        outputTensor,
        outputBuffer,
        inputUniform,
        outputUniform,
        width,
        height,
        lastUsed: ++this._clock
      };
      return execution;
    } catch (error) {
      this._releaseBuffer(outputUniform);
      this._releaseBuffer(inputUniform);
      this._releaseBuffer(outputBuffer);
      this._releaseTensor(outputTensor);
      this._releaseTensor(inputTensor);
      this._releaseGraph(graph);
      throw error;
    }
  }

  private async _execution(width: number, height: number) {
    const key = `${width}x${height}`;
    let execution = this._shapeCache.get(key);
    if (!execution) {
      let pending = this._shapePromises.get(key);
      if (!pending) {
        pending = (async () => {
          this._pendingCreationCount++;
          try {
            return await this._createExecution(width, height);
          } finally {
            this._pendingCreationCount--;
          }
        })();
        this._shapePromises.set(key, pending);
      }
      try {
        execution = await pending;
        if (this._disposed) {
          this._destroyExecution(execution);
          throw new Error('OIDN WebNN executor is disposed');
        }
        this._shapeCache.set(key, execution);
      } finally {
        if (this._shapePromises.get(key) === pending) {
          this._shapePromises.delete(key);
        }
      }
      if (this._shapeCache.size > this._shapeCacheSize) {
        const oldest = [...this._shapeCache.entries()]
          .filter(([candidate]) => candidate !== key)
          .sort((left, right) => left[1].lastUsed - right[1].lastUsed)[0];
        if (oldest) {
          this._shapeCache.delete(oldest[0]);
          this._retireExecution(oldest[1]);
        }
      }
    }
    execution.lastUsed = ++this._clock;
    return execution;
  }

  /** Compiles common tile shapes while the host still reports model loading. */
  async prewarm(shapes: readonly { width: number; height: number }[]) {
    for (const shape of shapes) {
      await this._execution(shape.width, shape.height);
    }
  }

  async execute(
    inputBuffers: readonly GPUBuffer[],
    width: number,
    height: number
  ) {
    const sourceCount = this._model.inputChannels / 3;
    if (inputBuffers.length !== sourceCount) {
      throw new Error(
        `OIDN WebNN expected ${sourceCount} input buffers, got ${inputBuffers.length}`
      );
    }
    const execution = await this._execution(width, height);
    const inputGPUBuffer = this._resources.track(
      'gpu-buffer',
      await this._context.exportToGPU(execution.inputTensor)
    );
    try {
      const inputEntries: GPUBindGroupEntry[] = inputBuffers.map(
        (buffer, binding) => ({ binding, resource: { buffer } })
      );
      inputEntries.push({
        binding: sourceCount,
        resource: { buffer: inputGPUBuffer }
      });
      inputEntries.push({
        binding: sourceCount + 1,
        resource: { buffer: execution.inputUniform }
      });
      const inputBindGroup = this._device.createBindGroup({
        label: 'oidn/webnn/input-bindings',
        layout: this._inputPipeline.getBindGroupLayout(0),
        entries: inputEntries
      });
      const inputEncoder = this._device.createCommandEncoder({
        label: 'oidn/webnn/input-pack'
      });
      const inputPass = inputEncoder.beginComputePass();
      inputPass.setPipeline(this._inputPipeline);
      inputPass.setBindGroup(0, inputBindGroup);
      inputPass.dispatchWorkgroups(
        Math.ceil(width / WORKGROUP_SIZE),
        Math.ceil(height / WORKGROUP_SIZE),
        this._model.inputChannels
      );
      inputPass.end();
      this._device.queue.submit([inputEncoder.finish()]);
    } finally {
      this._releaseBuffer(inputGPUBuffer);
    }

    this._context.dispatch(
      execution.graph,
      { input: execution.inputTensor },
      { output: execution.outputTensor }
    );
    const outputGPUBuffer = this._resources.track(
      'gpu-buffer',
      await this._context.exportToGPU(execution.outputTensor)
    );
    try {
      const outputBindGroup = this._device.createBindGroup({
        label: 'oidn/webnn/output-bindings',
        layout: this._outputPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: outputGPUBuffer } },
          { binding: 1, resource: { buffer: execution.outputBuffer } },
          { binding: 2, resource: { buffer: execution.outputUniform } }
        ]
      });
      const outputEncoder = this._device.createCommandEncoder({
        label: 'oidn/webnn/output-unpack'
      });
      const outputPass = outputEncoder.beginComputePass();
      outputPass.setPipeline(this._outputPipeline);
      outputPass.setBindGroup(0, outputBindGroup);
      outputPass.dispatchWorkgroups(
        Math.ceil(width / WORKGROUP_SIZE),
        Math.ceil(height / WORKGROUP_SIZE)
      );
      outputPass.end();
      this._device.queue.submit([outputEncoder.finish()]);
    } finally {
      this._releaseBuffer(outputGPUBuffer);
    }
    return execution.outputBuffer;
  }

  async executeCPU(input: Float32Array, width: number, height: number) {
    const execution = await this._execution(width, height);
    const plane = width * height;
    const packed = new Float16Array(plane * this._model.inputChannels);
    for (let pixel = 0; pixel < plane; pixel++) {
      for (let channel = 0; channel < this._model.inputChannels; channel++) {
        packed[channel * plane + pixel] =
          input[pixel * this._model.inputChannels + channel];
      }
    }
    this._context.writeTensor(execution.inputTensor, packed);
    this._context.dispatch(
      execution.graph,
      { input: execution.inputTensor },
      { output: execution.outputTensor }
    );
    const result = new Float16Array(
      await this._context.readTensor(execution.outputTensor)
    );
    const unpacked = new Float32Array(plane * this._model.outputChannels);
    for (let pixel = 0; pixel < plane; pixel++) {
      for (let channel = 0; channel < this._model.outputChannels; channel++) {
        unpacked[pixel * this._model.outputChannels + channel] =
          result[channel * plane + pixel];
      }
    }
    return unpacked;
  }

  private _destroyExecution(execution: WebNNShapeExecution) {
    this._releaseGraph(execution.graph);
    this._releaseTensor(execution.inputTensor);
    this._releaseTensor(execution.outputTensor);
    this._releaseBuffer(execution.outputBuffer);
    this._releaseBuffer(execution.inputUniform);
    this._releaseBuffer(execution.outputUniform);
  }

  private _retireExecution(execution: WebNNShapeExecution) {
    this._retiredExecutions.add(execution);
    void this._device.queue.onSubmittedWorkDone().catch(() => undefined).then(() => {
      this._retiredExecutions.delete(execution);
      this._destroyExecution(execution);
    });
  }

  private _releaseBuffer(buffer: GPUBuffer | undefined) {
    this._resources.release('gpu-buffer', buffer, () => buffer!.destroy());
  }

  private _releaseTensor(tensor: MLTensorLike | undefined) {
    this._resources.release('ml-tensor', tensor, () => tensor!.destroy());
  }

  private _releaseGraph(graph: MLGraphLike | undefined) {
    this._resources.release('ml-graph', graph, () => graph!.destroy?.());
  }

  private _releaseContext() {
    this._resources.release(
      'ml-context',
      this._context,
      () => this._context.destroy?.()
    );
  }

  getResourceInfo(): OIDNResourceSnapshot {
    return this._resources.snapshot(
      this._pendingCreationCount + this._retiredExecutions.size
    );
  }

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    for (const execution of this._shapeCache.values()) {
      this._destroyExecution(execution);
    }
    this._shapeCache.clear();
    for (const execution of this._retiredExecutions) {
      this._destroyExecution(execution);
    }
    this._retiredExecutions.clear();
    this._shapePromises.clear();
    this._releaseContext();
  }
}
