import { Float16Array } from '@petamoriken/float16';
import {
  optimizeModelGraph,
  planModelExecution,
  type ExecutableModelNode,
  type ModelExecutionPlan,
  type ModelValueShape
} from './graphOptimizer';
import type {
  Conv2DNodeSpec,
  UNetModelGraph,
  ValidatedConvTensor,
  ValidatedUNetModel
} from './modelSpec';
import type { HostTensor } from './tza';

export type NativeUNetPrecision = 'fp32' | 'fp16';
export type NativeUNetPrecisionSetting = NativeUNetPrecision | 'auto';

export interface NativeUNetOptions {
  precision?: NativeUNetPrecisionSetting;
  /** Maximum number of shape-dependent activation plans retained. */
  shapeCacheSize?: number;
}

export interface NativeUNetLayerTiming {
  id: string;
  durationMs: number;
}

export interface NativeUNetExecutionProfile {
  totalMs: number;
  layers: NativeUNetLayerTiming[];
}

interface PackedConvBuffers {
  weights: GPUBuffer;
  bias: GPUBuffer;
}

interface NativePipelineSpec {
  key: string;
  code: string;
}

interface ActivationSlot {
  buffer: GPUBuffer;
  capacity: number;
  activeValue?: string;
}

interface CachedExecution {
  plan: ModelExecutionPlan;
  valueBuffers: Map<string, GPUBuffer>;
  slots: ActivationSlot[];
  nodeBindings: GPUBindGroup[];
  nodePipelines: GPUComputePipeline[];
  inputPipeline: GPUComputePipeline;
  inputUniform: GPUBuffer;
  ownedBuffers: GPUBuffer[];
  cpuInputBuffers?: GPUBuffer[];
  cpuReadbackBuffer?: GPUBuffer;
  lastUsed: number;
}

const WORKGROUP_SIZE = 8;
const TILED_CONV_WORKGROUP = 8;
const TILED_CONV_ROWS_PER_THREAD = 4;
const TILED_CONV_M =
  TILED_CONV_WORKGROUP * TILED_CONV_ROWS_PER_THREAD;
const TILED_CONV_N_BLOCKS = TILED_CONV_WORKGROUP;
const TILED_CONV_K_BLOCKS = 8;

function roundUp(value: number, alignment: number) {
  return Math.ceil(value / alignment) * alignment;
}

function blocksForChannels(channels: number) {
  return Math.ceil(channels / 4);
}

function activationByteSize(
  shape: ModelValueShape,
  bytesPerScalar: number
) {
  return (
    shape.width *
    shape.height *
    blocksForChannels(shape.channels) *
    4 *
    bytesPerScalar
  );
}

let float16ToFloat32Lookup: Float32Array | undefined;

function halfBitsToNumber(bits: number) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) {
    return sign * fraction * 2 ** -24;
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Infinity : NaN;
  }
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

function halfLookup() {
  if (!float16ToFloat32Lookup) {
    float16ToFloat32Lookup = new Float32Array(1 << 16);
    for (let bits = 0; bits < float16ToFloat32Lookup.length; bits++) {
      float16ToFloat32Lookup[bits] = halfBitsToNumber(bits);
    }
  }
  return float16ToFloat32Lookup;
}

function tensorFloat32Values(tensor: HostTensor): Float32Array {
  if (tensor.desc.dataType === 'Float32') {
    return new Float32Array(
      tensor.data.buffer,
      tensor.data.byteOffset,
      tensor.data.byteLength / 4
    );
  }
  const bits = new Uint16Array(
    tensor.data.buffer,
    tensor.data.byteOffset,
    tensor.data.byteLength / 2
  );
  const values = new Float32Array(bits.length);
  if (bits.length < 4096) {
    for (let index = 0; index < bits.length; index++) {
      values[index] = halfBitsToNumber(bits[index]);
    }
  } else {
    const lookup = halfLookup();
    for (let index = 0; index < bits.length; index++) {
      values[index] = lookup[bits[index]];
    }
  }
  return values;
}

function createMappedBuffer(
  device: GPUDevice,
  label: string,
  data: ArrayBufferView,
  usage: GPUBufferUsageFlags
) {
  const size = roundUp(data.byteLength, 4);
  const buffer = device.createBuffer({
    label,
    size,
    usage,
    mappedAtCreation: true
  });
  new Uint8Array(buffer.getMappedRange()).set(
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
  );
  buffer.unmap();
  return buffer;
}

function createUniformBuffer(
  device: GPUDevice,
  label: string,
  values: readonly number[]
) {
  const data = new Uint32Array(roundUp(values.length, 4));
  data.set(values);
  return createMappedBuffer(device, label, data, GPUBufferUsage.UNIFORM);
}

function packConvTensors(
  device: GPUDevice,
  id: string,
  tensors: ValidatedConvTensor,
  precision: NativeUNetPrecision
): PackedConvBuffers {
  const inputBlocks = blocksForChannels(tensors.inputChannels);
  const outputBlocks = blocksForChannels(tensors.outputChannels);
  const packedWeightCount =
    outputBlocks *
    tensors.kernelHeight *
    tensors.kernelWidth *
    inputBlocks *
    4 *
    4;
  const canCopyHalfBits =
    precision === 'fp16' && tensors.weight.desc.dataType === 'Float16';
  const packedWeights = canCopyHalfBits
    ? new Uint16Array(packedWeightCount)
    : precision === 'fp16'
      ? new Float16Array(packedWeightCount)
      : new Float32Array(packedWeightCount);
  const sourceWeights = canCopyHalfBits
    ? new Uint16Array(
        tensors.weight.data.buffer,
        tensors.weight.data.byteOffset,
        tensors.weight.data.byteLength / 2
      )
    : tensorFloat32Values(tensors.weight);

  for (let outputBlock = 0; outputBlock < outputBlocks; outputBlock++) {
    for (let y = 0; y < tensors.kernelHeight; y++) {
      for (let x = 0; x < tensors.kernelWidth; x++) {
        for (let inputBlock = 0; inputBlock < inputBlocks; inputBlock++) {
          for (let outputLane = 0; outputLane < 4; outputLane++) {
            const outputChannel = outputBlock * 4 + outputLane;
            for (let inputLane = 0; inputLane < 4; inputLane++) {
              const inputChannel = inputBlock * 4 + inputLane;
              const packedBlock =
                ((((outputBlock * tensors.kernelHeight + y) *
                  tensors.kernelWidth +
                  x) *
                  inputBlocks +
                  inputBlock) *
                  16);
              // One vec4 contains the four output lanes for an input lane.
              // Direct and tiled kernels share this layout, keeping model
              // descriptors independent from the selected kernel.
              const packedIndex =
                packedBlock + inputLane * 4 + outputLane;
              if (
                outputChannel < tensors.outputChannels &&
                inputChannel < tensors.inputChannels
              ) {
                const sourceIndex =
                  ((outputChannel * tensors.inputChannels + inputChannel) *
                    tensors.kernelHeight +
                    y) *
                    tensors.kernelWidth +
                  x;
                packedWeights[packedIndex] = sourceWeights[sourceIndex];
              }
            }
          }
        }
      }
    }
  }

  // Bias stays f32 even for half activations because convolution accumulates
  // in f32. Padded lanes are zero and never escape the final three channels.
  const packedBias = new Float32Array(outputBlocks * 4);
  packedBias.set(tensorFloat32Values(tensors.bias));

  return {
    weights: createMappedBuffer(
      device,
      `oidn/${id}/weights/${precision}`,
      packedWeights,
      GPUBufferUsage.STORAGE
    ),
    bias: createMappedBuffer(
      device,
      `oidn/${id}/bias`,
      packedBias,
      GPUBufferUsage.STORAGE
    )
  };
}

function storageVecType(precision: NativeUNetPrecision) {
  return precision === 'fp16' ? 'vec4<f16>' : 'vec4<f32>';
}

function shaderPreamble(precision: NativeUNetPrecision) {
  return precision === 'fp16' ? 'enable f16;\n' : '';
}

function storeExpression(
  expression: string,
  outputPrecision: NativeUNetPrecision
) {
  return outputPrecision === 'fp16'
    ? `vec4<f16>(${expression})`
    : expression;
}

function activationExpression(
  expression: string,
  activation: Conv2DNodeSpec['activation']
) {
  return activation === 'relu'
    ? `max(${expression}, vec4<f32>(0.0))`
    : expression;
}

function accumulationCode(
  inputExpression: string,
  weightBase: string,
  precision: NativeUNetPrecision
) {
  if (precision === 'fp32') {
    return /* wgsl */ `
let inputValue = vec4<f32>(${inputExpression});
let weightBase = ${weightBase};
acc = fma(vec4<f32>(weights[weightBase]), vec4<f32>(inputValue.x), acc);
acc = fma(vec4<f32>(weights[weightBase + 1u]), vec4<f32>(inputValue.y), acc);
acc = fma(vec4<f32>(weights[weightBase + 2u]), vec4<f32>(inputValue.z), acc);
acc = fma(vec4<f32>(weights[weightBase + 3u]), vec4<f32>(inputValue.w), acc);
`;
  }
  return /* wgsl */ `
let inputValue = vec4<f16>(${inputExpression});
let weightBase = ${weightBase};
var partial = vec4<f16>(0.0h);
partial = fma(weights[weightBase], vec4<f16>(inputValue.x), partial);
partial = fma(weights[weightBase + 1u], vec4<f16>(inputValue.y), partial);
partial = fma(weights[weightBase + 2u], vec4<f16>(inputValue.z), partial);
partial = fma(weights[weightBase + 3u], vec4<f16>(inputValue.w), partial);
acc += vec4<f32>(partial);
`;
}

function tiledAccumulationCode(precision: NativeUNetPrecision) {
  if (precision === 'fp16') {
    return /* wgsl */ `
    var partial: array<vec4<f16>, ${TILED_CONV_ROWS_PER_THREAD}>;
    for (var row = 0u; row < ${TILED_CONV_ROWS_PER_THREAD}u; row++) {
      partial[row] = vec4<f16>(0.0h);
    }
    for (var tileK = 0u; tileK < ${TILED_CONV_K_BLOCKS}u; tileK++) {
      let weightBase =
        (tileK * ${TILED_CONV_N_BLOCKS}u + localId.x) * 4u;
      for (var row = 0u; row < ${TILED_CONV_ROWS_PER_THREAD}u; row++) {
        let tileSpatial =
          localId.y * ${TILED_CONV_ROWS_PER_THREAD}u + row;
        let inputValue =
          inputTile[tileSpatial * ${TILED_CONV_K_BLOCKS}u + tileK];
        partial[row] = fma(
          weightTile[weightBase],
          vec4<f16>(inputValue.x),
          partial[row]
        );
        partial[row] = fma(
          weightTile[weightBase + 1u],
          vec4<f16>(inputValue.y),
          partial[row]
        );
        partial[row] = fma(
          weightTile[weightBase + 2u],
          vec4<f16>(inputValue.z),
          partial[row]
        );
        partial[row] = fma(
          weightTile[weightBase + 3u],
          vec4<f16>(inputValue.w),
          partial[row]
        );
      }
    }
    for (var row = 0u; row < ${TILED_CONV_ROWS_PER_THREAD}u; row++) {
      acc[row] += vec4<f32>(partial[row]);
    }
`;
  }
  return /* wgsl */ `
    for (var tileK = 0u; tileK < ${TILED_CONV_K_BLOCKS}u; tileK++) {
      let weightBase =
        (tileK * ${TILED_CONV_N_BLOCKS}u + localId.x) * 4u;
      for (var row = 0u; row < ${TILED_CONV_ROWS_PER_THREAD}u; row++) {
        let tileSpatial =
          localId.y * ${TILED_CONV_ROWS_PER_THREAD}u + row;
        let inputValue =
          inputTile[tileSpatial * ${TILED_CONV_K_BLOCKS}u + tileK];
        acc[row] = fma(
          weightTile[weightBase],
          vec4<f32>(inputValue.x),
          acc[row]
        );
        acc[row] = fma(
          weightTile[weightBase + 1u],
          vec4<f32>(inputValue.y),
          acc[row]
        );
        acc[row] = fma(
          weightTile[weightBase + 2u],
          vec4<f32>(inputValue.z),
          acc[row]
        );
        acc[row] = fma(
          weightTile[weightBase + 3u],
          vec4<f32>(inputValue.w),
          acc[row]
        );
      }
    }
`;
}

function createConvShader(
  precision: NativeUNetPrecision,
  outputPrecision: NativeUNetPrecision,
  activation: Conv2DNodeSpec['activation'],
  inputBlocks: number,
  outputBlocks: number
) {
  const inputType = storageVecType(precision);
  const weightType = storageVecType(precision);
  const outputType = storageVecType(outputPrecision);
  const stored = storeExpression(
    activationExpression('acc', activation),
    outputPrecision
  );

  return /* wgsl */ `${shaderPreamble(precision)}
struct Params {
  inputWidth: u32,
  inputHeight: u32,
  outputWidth: u32,
  outputHeight: u32,
  inputBlocks: u32,
  outputBlocks: u32,
}
@group(0) @binding(0) var<storage, read> inputData: array<${inputType}>;
@group(0) @binding(1) var<storage, read> weights: array<${weightType}>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outputData: array<${outputType}>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.outputWidth || gid.y >= params.outputHeight || gid.z >= ${outputBlocks}u) {
    return;
  }
  var acc = bias[gid.z];
  for (var ky = 0u; ky < 3u; ky++) {
    let inputY = i32(gid.y) + i32(ky) - 1;
    if (inputY < 0 || inputY >= i32(params.inputHeight)) { continue; }
    for (var kx = 0u; kx < 3u; kx++) {
      let inputX = i32(gid.x) + i32(kx) - 1;
      if (inputX < 0 || inputX >= i32(params.inputWidth)) { continue; }
      let pixelBase = (u32(inputY) * params.inputWidth + u32(inputX)) * ${inputBlocks}u;
      for (var inputBlock = 0u; inputBlock < ${inputBlocks}u; inputBlock++) {
        ${accumulationCode(
          'inputData[pixelBase + inputBlock]',
          `((((gid.z * 3u + ky) * 3u + kx) * ${inputBlocks}u + inputBlock) * 4u)`,
          precision
        )}
      }
    }
  }
  let outputIndex = (gid.y * params.outputWidth + gid.x) * ${outputBlocks}u + gid.z;
  outputData[outputIndex] = ${stored};
}
`;
}

/**
 * Implicit-GEMM convolution for FP32. This follows the proven packed WebGPU
 * shape: one 8x8 workgroup computes 32 spatial rows by 8 vec4 output blocks,
 * with four output rows per thread and an eight-vec4 K tile.
 */
function createTiledConvShader(
  precision: NativeUNetPrecision,
  outputPrecision: NativeUNetPrecision,
  activation: Conv2DNodeSpec['activation'],
  inputBlocks: number,
  outputBlocks: number
) {
  const inputType = storageVecType(precision);
  const outputType = storageVecType(outputPrecision);
  const stored = storeExpression(
    activationExpression('acc', activation),
    outputPrecision
  );
  const workgroupThreads =
    TILED_CONV_WORKGROUP * TILED_CONV_WORKGROUP;
  const inputTileValues = TILED_CONV_M * TILED_CONV_K_BLOCKS;
  const weightTileValues =
    TILED_CONV_K_BLOCKS * TILED_CONV_N_BLOCKS * 4;
  return /* wgsl */ `${shaderPreamble(precision)}
struct Params {
  inputWidth: u32,
  inputHeight: u32,
  outputWidth: u32,
  outputHeight: u32,
  inputBlocks: u32,
  outputBlocks: u32,
}
@group(0) @binding(0) var<storage, read> inputData: array<${inputType}>;
@group(0) @binding(1) var<storage, read> weights: array<${inputType}>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outputData: array<${outputType}>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> inputTile: array<${inputType}, ${inputTileValues}>;
var<workgroup> weightTile: array<${inputType}, ${weightTileValues}>;

@compute @workgroup_size(${TILED_CONV_WORKGROUP}, ${TILED_CONV_WORKGROUP}, 1)
fn main(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let spatialBase =
    workgroupId.x * ${TILED_CONV_M}u +
    localId.y * ${TILED_CONV_ROWS_PER_THREAD}u;
  let outputBlock =
    workgroupId.y * ${TILED_CONV_N_BLOCKS}u + localId.x;
  let spatialCount = params.outputWidth * params.outputHeight;
  var acc: array<vec4<f32>, ${TILED_CONV_ROWS_PER_THREAD}>;
  if (outputBlock < ${outputBlocks}u) {
    for (var row = 0u; row < ${TILED_CONV_ROWS_PER_THREAD}u; row++) {
      acc[row] = bias[outputBlock];
    }
  }

  let localLinear =
    localId.y * ${TILED_CONV_WORKGROUP}u + localId.x;
  let totalK = ${inputBlocks * 9}u;
  for (var kBase = 0u; kBase < totalK; kBase += ${TILED_CONV_K_BLOCKS}u) {
    for (
      var loadIndex = localLinear;
      loadIndex < ${inputTileValues}u;
      loadIndex += ${workgroupThreads}u
    ) {
      let tileSpatial = loadIndex / ${TILED_CONV_K_BLOCKS}u;
      let tileK = loadIndex % ${TILED_CONV_K_BLOCKS}u;
      let inputSpatialIndex =
        workgroupId.x * ${TILED_CONV_M}u + tileSpatial;
      let kIndex = kBase + tileK;
      var value = ${inputType}(0.0);
      if (inputSpatialIndex < spatialCount && kIndex < totalK) {
        let outputY = inputSpatialIndex / params.outputWidth;
        let outputX = inputSpatialIndex % params.outputWidth;
        let inputBlock = kIndex % ${inputBlocks}u;
        let kernelIndex = kIndex / ${inputBlocks}u;
        let kernelY = kernelIndex / 3u;
        let kernelX = kernelIndex % 3u;
        let inputY = i32(outputY) + i32(kernelY) - 1;
        let inputX = i32(outputX) + i32(kernelX) - 1;
        if (
          inputY >= 0 && inputY < i32(params.inputHeight) &&
          inputX >= 0 && inputX < i32(params.inputWidth)
        ) {
          let inputIndex =
            (u32(inputY) * params.inputWidth + u32(inputX)) *
            ${inputBlocks}u + inputBlock;
          value = inputData[inputIndex];
        }
      }
      inputTile[loadIndex] = value;
    }

    for (
      var loadIndex = localLinear;
      loadIndex < ${weightTileValues}u;
      loadIndex += ${workgroupThreads}u
    ) {
      let tileK = loadIndex / ${TILED_CONV_N_BLOCKS * 4}u;
      let outputRemainder = loadIndex % ${TILED_CONV_N_BLOCKS * 4}u;
      let tileOutputBlock = outputRemainder / 4u;
      let outputLane = outputRemainder % 4u;
      let loadedOutputBlock =
        workgroupId.y * ${TILED_CONV_N_BLOCKS}u + tileOutputBlock;
      let kIndex = kBase + tileK;
      var value = ${inputType}(0.0);
      if (loadedOutputBlock < ${outputBlocks}u && kIndex < totalK) {
        let inputBlock = kIndex % ${inputBlocks}u;
        let kernelIndex = kIndex / ${inputBlocks}u;
        let kernelY = kernelIndex / 3u;
        let kernelX = kernelIndex % 3u;
        let weightIndex =
          ((((loadedOutputBlock * 3u + kernelY) * 3u + kernelX) *
            ${inputBlocks}u + inputBlock) * 4u + outputLane);
        value = weights[weightIndex];
      }
      weightTile[loadIndex] = value;
    }

    workgroupBarrier();
    ${tiledAccumulationCode(precision)}
    workgroupBarrier();
  }

  if (outputBlock < ${outputBlocks}u) {
    for (var row = 0u; row < ${TILED_CONV_ROWS_PER_THREAD}u; row++) {
      let spatialIndex = spatialBase + row;
      if (spatialIndex < spatialCount) {
        let outputIndex = spatialIndex * ${outputBlocks}u + outputBlock;
        outputData[outputIndex] = ${stored.replaceAll('acc', 'acc[row]')};
      }
    }
  }
}
`;
}

function createFusedConvPoolShader(
  precision: NativeUNetPrecision,
  activation: Conv2DNodeSpec['activation'],
  inputBlocks: number,
  outputBlocks: number
) {
  const valueType = storageVecType(precision);
  const activated = activationExpression('acc', activation);
  return /* wgsl */ `${shaderPreamble(precision)}
struct Params {
  inputWidth: u32,
  inputHeight: u32,
  outputWidth: u32,
  outputHeight: u32,
  inputBlocks: u32,
  outputBlocks: u32,
}
@group(0) @binding(0) var<storage, read> inputData: array<${valueType}>;
@group(0) @binding(1) var<storage, read> weights: array<${valueType}>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outputData: array<${valueType}>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.outputWidth || gid.y >= params.outputHeight || gid.z >= ${outputBlocks}u) {
    return;
  }
  var pooled = vec4<f32>(-3.402823466e+38);
  for (var py = 0u; py < 2u; py++) {
    let centerY = gid.y * 2u + py;
    if (centerY >= params.inputHeight) { continue; }
    for (var px = 0u; px < 2u; px++) {
      let centerX = gid.x * 2u + px;
      if (centerX >= params.inputWidth) { continue; }
      var acc = bias[gid.z];
      for (var ky = 0u; ky < 3u; ky++) {
        let inputY = i32(centerY) + i32(ky) - 1;
        if (inputY < 0 || inputY >= i32(params.inputHeight)) { continue; }
        for (var kx = 0u; kx < 3u; kx++) {
          let inputX = i32(centerX) + i32(kx) - 1;
          if (inputX < 0 || inputX >= i32(params.inputWidth)) { continue; }
          let pixelBase = (u32(inputY) * params.inputWidth + u32(inputX)) * ${inputBlocks}u;
          for (var inputBlock = 0u; inputBlock < ${inputBlocks}u; inputBlock++) {
            ${accumulationCode(
              'inputData[pixelBase + inputBlock]',
              `((((gid.z * 3u + ky) * 3u + kx) * ${inputBlocks}u + inputBlock) * 4u)`,
              precision
            )}
          }
        }
      }
      pooled = max(pooled, ${activated});
    }
  }
  let outputIndex = (gid.y * params.outputWidth + gid.x) * ${outputBlocks}u + gid.z;
  outputData[outputIndex] = ${storeExpression('pooled', precision)};
}
`;
}

function createMaxPoolShader(
  precision: NativeUNetPrecision,
  outputBlocks: number
) {
  const valueType = storageVecType(precision);
  return /* wgsl */ `${shaderPreamble(precision)}
struct Params {
  inputWidth: u32,
  inputHeight: u32,
  outputWidth: u32,
  outputHeight: u32,
  outputBlocks: u32,
}
@group(0) @binding(0) var<storage, read> inputData: array<${valueType}>;
@group(0) @binding(1) var<storage, read_write> outputData: array<${valueType}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (
    gid.x >= params.outputWidth ||
    gid.y >= params.outputHeight ||
    gid.z >= ${outputBlocks}u
  ) {
    return;
  }
  var pooled = vec4<f32>(-3.402823466e+38);
  for (var py = 0u; py < 2u; py++) {
    let inputY = gid.y * 2u + py;
    if (inputY >= params.inputHeight) { continue; }
    for (var px = 0u; px < 2u; px++) {
      let inputX = gid.x * 2u + px;
      if (inputX >= params.inputWidth) { continue; }
      let inputIndex =
        (inputY * params.inputWidth + inputX) * ${outputBlocks}u + gid.z;
      pooled = max(pooled, vec4<f32>(inputData[inputIndex]));
    }
  }
  let outputIndex =
    (gid.y * params.outputWidth + gid.x) * ${outputBlocks}u + gid.z;
  outputData[outputIndex] = ${storeExpression('pooled', precision)};
}
`;
}

function createTiledDecoderShader(
  precision: NativeUNetPrecision,
  activation: Conv2DNodeSpec['activation'],
  sourceBlocks: readonly [number, number],
  upsampledSource: 0 | 1,
  outputBlocks: number
) {
  const valueType = storageVecType(precision);
  const inputBlocks = sourceBlocks[0] + sourceBlocks[1];
  const stored = storeExpression(
    activationExpression('acc[row]', activation),
    precision
  );
  const workgroupThreads =
    TILED_CONV_WORKGROUP * TILED_CONV_WORKGROUP;
  const inputTileValues = TILED_CONV_M * TILED_CONV_K_BLOCKS;
  const weightTileValues =
    TILED_CONV_K_BLOCKS * TILED_CONV_N_BLOCKS * 4;
  const sourceRead = (source: 0 | 1, blockExpression: string) => {
    const sourceX =
      source === upsampledSource
        ? 'u32(inputX) / 2u'
        : 'u32(inputX)';
    const sourceY =
      source === upsampledSource
        ? 'u32(inputY) / 2u'
        : 'u32(inputY)';
    return /* wgsl */ `
          {
            let sourceBlock = ${blockExpression};
            let sourceX = ${sourceX};
            let sourceY = ${sourceY};
            let sourceIndex =
              (sourceY * params.source${source}Width + sourceX) *
              ${sourceBlocks[source]}u + sourceBlock;
            value = input${source}[sourceIndex];
          }`;
  };
  return /* wgsl */ `${shaderPreamble(precision)}
struct Params {
  outputWidth: u32,
  outputHeight: u32,
  outputBlocks: u32,
  inputBlocks: u32,
  source0Width: u32,
  source0Height: u32,
  source1Width: u32,
  source1Height: u32,
}
@group(0) @binding(0) var<storage, read> input0: array<${valueType}>;
@group(0) @binding(1) var<storage, read> input1: array<${valueType}>;
@group(0) @binding(2) var<storage, read> weights: array<${valueType}>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> outputData: array<${valueType}>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> inputTile: array<${valueType}, ${inputTileValues}>;
var<workgroup> weightTile: array<${valueType}, ${weightTileValues}>;

@compute @workgroup_size(${TILED_CONV_WORKGROUP}, ${TILED_CONV_WORKGROUP}, 1)
fn main(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let spatialBase =
    workgroupId.x * ${TILED_CONV_M}u +
    localId.y * ${TILED_CONV_ROWS_PER_THREAD}u;
  let outputBlock =
    workgroupId.y * ${TILED_CONV_N_BLOCKS}u + localId.x;
  let spatialCount = params.outputWidth * params.outputHeight;
  var acc: array<vec4<f32>, ${TILED_CONV_ROWS_PER_THREAD}>;
  if (outputBlock < ${outputBlocks}u) {
    for (var row = 0u; row < ${TILED_CONV_ROWS_PER_THREAD}u; row++) {
      acc[row] = bias[outputBlock];
    }
  }

  let localLinear =
    localId.y * ${TILED_CONV_WORKGROUP}u + localId.x;
  let totalK = ${inputBlocks * 9}u;
  for (var kBase = 0u; kBase < totalK; kBase += ${TILED_CONV_K_BLOCKS}u) {
    for (
      var loadIndex = localLinear;
      loadIndex < ${inputTileValues}u;
      loadIndex += ${workgroupThreads}u
    ) {
      let tileSpatial = loadIndex / ${TILED_CONV_K_BLOCKS}u;
      let tileK = loadIndex % ${TILED_CONV_K_BLOCKS}u;
      let outputSpatialIndex =
        workgroupId.x * ${TILED_CONV_M}u + tileSpatial;
      let kIndex = kBase + tileK;
      var value = ${valueType}(0.0);
      if (outputSpatialIndex < spatialCount && kIndex < totalK) {
        let outputY = outputSpatialIndex / params.outputWidth;
        let outputX = outputSpatialIndex % params.outputWidth;
        let inputBlock = kIndex % ${inputBlocks}u;
        let kernelIndex = kIndex / ${inputBlocks}u;
        let kernelY = kernelIndex / 3u;
        let kernelX = kernelIndex % 3u;
        let inputY = i32(outputY) + i32(kernelY) - 1;
        let inputX = i32(outputX) + i32(kernelX) - 1;
        if (
          inputY >= 0 && inputY < i32(params.outputHeight) &&
          inputX >= 0 && inputX < i32(params.outputWidth)
        ) {
          if (inputBlock < ${sourceBlocks[0]}u) {
            ${sourceRead(0, 'inputBlock')}
          } else {
            ${sourceRead(1, `inputBlock - ${sourceBlocks[0]}u`)}
          }
        }
      }
      inputTile[loadIndex] = value;
    }

    for (
      var loadIndex = localLinear;
      loadIndex < ${weightTileValues}u;
      loadIndex += ${workgroupThreads}u
    ) {
      let tileK = loadIndex / ${TILED_CONV_N_BLOCKS * 4}u;
      let outputRemainder = loadIndex % ${TILED_CONV_N_BLOCKS * 4}u;
      let tileOutputBlock = outputRemainder / 4u;
      let outputLane = outputRemainder % 4u;
      let loadedOutputBlock =
        workgroupId.y * ${TILED_CONV_N_BLOCKS}u + tileOutputBlock;
      let kIndex = kBase + tileK;
      var value = ${valueType}(0.0);
      if (loadedOutputBlock < ${outputBlocks}u && kIndex < totalK) {
        let inputBlock = kIndex % ${inputBlocks}u;
        let kernelIndex = kIndex / ${inputBlocks}u;
        let kernelY = kernelIndex / 3u;
        let kernelX = kernelIndex % 3u;
        let weightIndex =
          ((((loadedOutputBlock * 3u + kernelY) * 3u + kernelX) *
            ${inputBlocks}u + inputBlock) * 4u + outputLane);
        value = weights[weightIndex];
      }
      weightTile[loadIndex] = value;
    }

    workgroupBarrier();
    ${tiledAccumulationCode(precision)}
    workgroupBarrier();
  }

  if (outputBlock < ${outputBlocks}u) {
    for (var row = 0u; row < ${TILED_CONV_ROWS_PER_THREAD}u; row++) {
      let spatialIndex = spatialBase + row;
      if (spatialIndex < spatialCount) {
        let outputIndex = spatialIndex * ${outputBlocks}u + outputBlock;
        outputData[outputIndex] = ${stored};
      }
    }
  }
}
`;
}

function createFusedDecoderShader(
  precision: NativeUNetPrecision,
  activation: Conv2DNodeSpec['activation'],
  sourceBlocks: readonly [number, number],
  upsampledSource: 0 | 1,
  outputBlocks: number
) {
  const valueType = storageVecType(precision);
  const inputBlocks = sourceBlocks[0] + sourceBlocks[1];
  const sourceCode = (source: 0 | 1, blockOffset: number) => {
    const isUpsampled = source === upsampledSource;
    return /* wgsl */ `
      {
        let sourceX = ${isUpsampled ? 'u32(inputX) / 2u' : 'u32(inputX)'};
        let sourceY = ${isUpsampled ? 'u32(inputY) / 2u' : 'u32(inputY)'};
        let sourcePixelBase = (sourceY * params.source${source}Width + sourceX) * ${sourceBlocks[source]}u;
        for (var sourceBlock = 0u; sourceBlock < ${sourceBlocks[source]}u; sourceBlock++) {
          let inputBlock = ${blockOffset}u + sourceBlock;
          ${accumulationCode(
            `input${source}[sourcePixelBase + sourceBlock]`,
            `((((gid.z * 3u + ky) * 3u + kx) * ${inputBlocks}u + inputBlock) * 4u)`,
            precision
          )}
        }
      }
`;
  };
  const stored = storeExpression(
    activationExpression('acc', activation),
    precision
  );
  return /* wgsl */ `${shaderPreamble(precision)}
struct Params {
  outputWidth: u32,
  outputHeight: u32,
  outputBlocks: u32,
  inputBlocks: u32,
  source0Width: u32,
  source0Height: u32,
  source1Width: u32,
  source1Height: u32,
}
@group(0) @binding(0) var<storage, read> input0: array<${valueType}>;
@group(0) @binding(1) var<storage, read> input1: array<${valueType}>;
@group(0) @binding(2) var<storage, read> weights: array<${valueType}>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> outputData: array<${valueType}>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.outputWidth || gid.y >= params.outputHeight || gid.z >= ${outputBlocks}u) {
    return;
  }
  var acc = bias[gid.z];
  for (var ky = 0u; ky < 3u; ky++) {
    let inputY = i32(gid.y) + i32(ky) - 1;
    if (inputY < 0 || inputY >= i32(params.outputHeight)) { continue; }
    for (var kx = 0u; kx < 3u; kx++) {
      let inputX = i32(gid.x) + i32(kx) - 1;
      if (inputX < 0 || inputX >= i32(params.outputWidth)) { continue; }
      ${sourceCode(0, 0)}
      ${sourceCode(1, sourceBlocks[0])}
    }
  }
  let outputIndex = (gid.y * params.outputWidth + gid.x) * ${outputBlocks}u + gid.z;
  outputData[outputIndex] = ${stored};
}
`;
}

function createInputPackShader(
  precision: NativeUNetPrecision,
  sourceCount: number
) {
  const outputType = storageVecType(precision);
  const inputBindings = Array.from(
    { length: sourceCount },
    (_, index) =>
      `@group(0) @binding(${index}) var<storage, read> input${index}: array<vec4<f32>>;`
  ).join('\n');
  const readBranches = Array.from({ length: sourceCount }, (_, index) => {
    const firstChannel = index * 3;
    return `if (channel < ${firstChannel + 3}u) { return input${index}[pixel][channel - ${firstChannel}u]; }`;
  }).join('\n  ');
  const outputBinding = sourceCount;
  const paramsBinding = sourceCount + 1;
  return /* wgsl */ `${shaderPreamble(precision)}
struct Params {
  width: u32,
  height: u32,
  outputBlocks: u32,
  inputChannels: u32,
}
${inputBindings}
@group(0) @binding(${outputBinding}) var<storage, read_write> outputData: array<${outputType}>;
@group(0) @binding(${paramsBinding}) var<uniform> params: Params;

fn readChannel(pixel: u32, channel: u32) -> f32 {
  ${readBranches}
  return 0.0;
}

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.outputBlocks) {
    return;
  }
  let pixel = gid.y * params.width + gid.x;
  let firstChannel = gid.z * 4u;
  let value = vec4<f32>(
    readChannel(pixel, firstChannel),
    readChannel(pixel, firstChannel + 1u),
    readChannel(pixel, firstChannel + 2u),
    readChannel(pixel, firstChannel + 3u)
  );
  outputData[pixel * params.outputBlocks + gid.z] = ${storeExpression('value', precision)};
}
`;
}

function executableInputs(node: ExecutableModelNode): readonly string[] {
  if (node.op === 'concat') return node.inputs;
  if (node.op === 'fusedUpsampleConcatConv2d') {
    return node.inputs.map((input) => input.value);
  }
  return [node.input];
}

export function resolveNativeUNetPrecision(
  device: GPUDevice,
  requested: NativeUNetPrecisionSetting = 'auto'
): NativeUNetPrecision {
  const hasShaderF16 = device.features.has('shader-f16');
  if (requested === 'fp16' && !hasShaderF16) {
    throw new Error(
      'OIDN FP16 was requested but the GPUDevice does not have shader-f16 enabled'
    );
  }
  if (requested === 'auto') return hasShaderF16 ? 'fp16' : 'fp32';
  return requested;
}

/** Native, model-driven OIDN U-Net executor. */
export class NativeUNetExecutor {
  readonly precision: NativeUNetPrecision;

  private _model: UNetModelGraph;
  private _packedConvs = new Map<string, PackedConvBuffers>();
  private _pipelineCache = new Map<string, GPUComputePipeline>();
  private _pipelinePromises = new Map<
    string,
    Promise<GPUComputePipeline>
  >();
  private _executionCache = new Map<string, CachedExecution>();
  private _clock = 0;
  private _shapeCacheSize: number;
  private _profileNextExecution = false;
  private _lastExecutionProfile?: Promise<NativeUNetExecutionProfile>;

  constructor(
    private _device: GPUDevice,
    model: ValidatedUNetModel,
    options: NativeUNetOptions = {}
  ) {
    this.precision = resolveNativeUNetPrecision(
      _device,
      options.precision ?? 'auto'
    );
    this._shapeCacheSize = Math.max(1, options.shapeCacheSize ?? 2);

    if (
      model.inputChannels % 3 !== 0 ||
      model.inputChannels < 3 ||
      model.inputChannels > 9
    ) {
      throw new Error(
        `Native OIDN expects 3, 6, or 9 input channels, got ${model.inputChannels}`
      );
    }
    this._model = {
      spec: model.spec,
      inputChannels: model.inputChannels,
      outputChannels: model.outputChannels,
      channelsByValue: new Map(model.channelsByValue),
      convChannels: new Map(model.convChannels)
    };
    for (const node of optimizeModelGraph(this._model, {
      fuseConvPool: false
    }).nodes) {
      if (
        node.op !== 'conv2d' &&
        node.op !== 'maxPool2d' &&
        node.op !== 'fusedConvReluMaxPool2d' &&
        node.op !== 'fusedUpsampleConcatConv2d'
      ) {
        throw new Error(
          `Native OIDN descriptor ${model.spec.id} leaves unsupported ` +
            `${node.op} node ${node.id} after graph optimization`
        );
      }
    }
    for (const [id, tensors] of model.convTensors) {
      this._packedConvs.set(
        id,
        packConvTensors(_device, id, tensors, this.precision)
      );
    }
  }

  private _pipeline(key: string, code: string) {
    let pipeline = this._pipelineCache.get(key);
    if (!pipeline) {
      pipeline = this._device.createComputePipeline({
        label: `oidn/${key}`,
        layout: 'auto',
        compute: {
          module: this._device.createShaderModule({
            label: `oidn/${key}`,
            code
          }),
          entryPoint: 'main'
        }
      });
      this._pipelineCache.set(key, pipeline);
    }
    return pipeline;
  }

  private _pipelineAsync(key: string, code: string) {
    const ready = this._pipelineCache.get(key);
    if (ready) return Promise.resolve(ready);
    const pending = this._pipelinePromises.get(key);
    if (pending) return pending;
    const promise = this._device.createComputePipelineAsync({
      label: `oidn/${key}`,
      layout: 'auto',
      compute: {
        module: this._device.createShaderModule({
          label: `oidn/${key}`,
          code
        }),
        entryPoint: 'main'
      }
    }).then((pipeline) => {
      this._pipelineCache.set(key, pipeline);
      this._pipelinePromises.delete(key);
      return pipeline;
    }, (error) => {
      this._pipelinePromises.delete(key);
      throw error;
    });
    this._pipelinePromises.set(key, promise);
    return promise;
  }

  private _nodePipelineSpec(
    node: ExecutableModelNode,
    isFinal: boolean
  ): NativePipelineSpec {
    if (node.op === 'conv2d') {
      const outputPrecision = isFinal ? 'fp32' : this.precision;
      const tiled = this.precision === 'fp32' && !isFinal;
      const inputBlocks = blocksForChannels(
        this._model.convChannels.get(node.id)!.inputChannels
      );
      const outputBlocks = blocksForChannels(
        this._model.convChannels.get(node.id)!.outputChannels
      );
      const key =
        `${tiled ? 'conv-tiled' : 'conv'}/${this.precision}/` +
        `${outputPrecision}/${node.activation}/` +
        `in${inputBlocks}/out${outputBlocks}`;
      return {
        key,
        code: tiled
          ? createTiledConvShader(
              this.precision,
              outputPrecision,
              node.activation,
              inputBlocks,
              outputBlocks
            )
          : createConvShader(
              this.precision,
              outputPrecision,
              node.activation,
              inputBlocks,
              outputBlocks
            )
      };
    }
    if (node.op === 'maxPool2d') {
      const outputBlocks = blocksForChannels(
        this._model.channelsByValue.get(node.id)!
      );
      const key = `max-pool/${this.precision}/out${outputBlocks}`;
      return {
        key,
        code: createMaxPoolShader(this.precision, outputBlocks)
      };
    }
    if (node.op === 'fusedConvReluMaxPool2d') {
      const inputBlocks = blocksForChannels(
        this._model.convChannels.get(node.conv.id)!.inputChannels
      );
      const outputBlocks = blocksForChannels(
        this._model.convChannels.get(node.conv.id)!.outputChannels
      );
      const key =
        `conv-pool/${this.precision}/${node.conv.activation}/` +
        `in${inputBlocks}/out${outputBlocks}`;
      return {
        key,
        code: createFusedConvPoolShader(
          this.precision,
          node.conv.activation,
          inputBlocks,
          outputBlocks
        )
      };
    }
    if (node.op === 'fusedUpsampleConcatConv2d') {
      if (node.inputs.length !== 2) {
        throw new Error(`Native fused decoder ${node.id} requires two inputs`);
      }
      const sourceBlocks = node.inputs.map((input) =>
        blocksForChannels(this._model.channelsByValue.get(input.value)!)
      ) as [number, number];
      const upsampledSource = node.inputs.findIndex((input) => input.upsample);
      if (upsampledSource !== 0 && upsampledSource !== 1) {
        throw new Error(`Native fused decoder ${node.id} has no upsample input`);
      }
      const tiled = this.precision === 'fp32';
      const outputBlocks = blocksForChannels(
        this._model.convChannels.get(node.conv.id)!.outputChannels
      );
      const key =
        `${tiled ? 'decoder-tiled' : 'decoder'}/${this.precision}/` +
        `${node.conv.activation}/` +
        `${sourceBlocks.join('+')}/out${outputBlocks}/up${upsampledSource}`;
      return {
        key,
        code: tiled
          ? createTiledDecoderShader(
              this.precision,
              node.conv.activation,
              sourceBlocks,
              upsampledSource,
              outputBlocks
            )
          : createFusedDecoderShader(
              this.precision,
              node.conv.activation,
              sourceBlocks,
              upsampledSource,
              outputBlocks
            )
      };
    }
    throw new Error(
      `Native OIDN does not implement unfused ${node.op} node ${node.id}`
    );
  }

  private _nodePipeline(node: ExecutableModelNode, isFinal: boolean) {
    const { key, code } = this._nodePipelineSpec(node, isFinal);
    return this._pipeline(key, code);
  }

  /** Compiles all shape-independent kernels before the model reports ready. */
  async prepare() {
    const graph = optimizeModelGraph(this._model, { fuseConvPool: false });
    const sourceCount = this._model.inputChannels / 3;
    const specs: NativePipelineSpec[] = [
      {
        key: `pack/${this.precision}/${sourceCount}`,
        code: createInputPackShader(this.precision, sourceCount)
      },
      ...graph.nodes.map((node) =>
        this._nodePipelineSpec(node, node.id === graph.spec.output)
      )
    ];
    await Promise.all(
      specs.map(({ key, code }) => this._pipelineAsync(key, code))
    );
  }

  private _createExecution(width: number, height: number): CachedExecution {
    const plan = planModelExecution(this._model, width, height, {
      fuseConvPool: false
    });
    const valueBuffers = new Map<string, GPUBuffer>();
    const slots: ActivationSlot[] = [];
    const lastUses = new Map<string, number>();
    plan.nodes.forEach((node, index) => {
      for (const input of executableInputs(node)) lastUses.set(input, index);
    });
    lastUses.set(plan.spec.output, plan.nodes.length);

    const allocate = (
      value: string,
      shape: ModelValueShape,
      bytesPerScalar: number,
      index: number
    ) => {
      for (const slot of slots) {
        if (
          slot.activeValue &&
          (lastUses.get(slot.activeValue) ?? -1) < index
        ) {
          slot.activeValue = undefined;
        }
      }
      const requiredSize = activationByteSize(shape, bytesPerScalar);
      let slot = slots
        .filter((candidate) => !candidate.activeValue && candidate.capacity >= requiredSize)
        .sort((a, b) => a.capacity - b.capacity)[0];
      if (!slot) {
        const buffer = this._device.createBuffer({
          label: `oidn/activation/${width}x${height}/${slots.length}`,
          size: roundUp(requiredSize, 4),
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST
        });
        slot = { buffer, capacity: requiredSize };
        slots.push(slot);
      }
      slot.activeValue = value;
      valueBuffers.set(value, slot.buffer);
    };

    allocate(
      plan.spec.input,
      plan.inputShape,
      this.precision === 'fp16' ? 2 : 4,
      -1
    );
    plan.plannedNodes.forEach(({ node, outputShape }, index) => {
      const isFinal = node.id === plan.spec.output;
      allocate(
        node.id,
        outputShape,
        isFinal || this.precision === 'fp32' ? 4 : 2,
        index
      );
    });

    const inputSourceCount = this._model.inputChannels / 3;
    const inputKey = `pack/${this.precision}/${inputSourceCount}`;
    const inputPipeline = this._pipeline(
      inputKey,
      createInputPackShader(this.precision, inputSourceCount)
    );
    const nodePipelines: GPUComputePipeline[] = [];
    const nodeBindings: GPUBindGroup[] = [];
    const ownedBuffers: GPUBuffer[] = [];

    plan.plannedNodes.forEach(({ node, outputShape }, index) => {
      const isFinal = node.id === plan.spec.output;
      const pipeline = this._nodePipeline(node, isFinal);
      nodePipelines.push(pipeline);
      const output = valueBuffers.get(node.id)!;
      let entries: GPUBindGroupEntry[];
      let uniformValues: number[];
      let convId: string;

      if (node.op === 'conv2d') {
        const inputShape = plan.valueShapes.get(node.input)!;
        convId = node.id;
        uniformValues = [
          inputShape.width,
          inputShape.height,
          outputShape.width,
          outputShape.height,
          blocksForChannels(inputShape.channels),
          blocksForChannels(outputShape.channels)
        ];
        const packed = this._packedConvs.get(convId)!;
        entries = [
          { binding: 0, resource: { buffer: valueBuffers.get(node.input)! } },
          { binding: 1, resource: { buffer: packed.weights } },
          { binding: 2, resource: { buffer: packed.bias } },
          { binding: 3, resource: { buffer: output } }
        ];
      } else if (node.op === 'maxPool2d') {
        const inputShape = plan.valueShapes.get(node.input)!;
        uniformValues = [
          inputShape.width,
          inputShape.height,
          outputShape.width,
          outputShape.height,
          blocksForChannels(outputShape.channels)
        ];
        entries = [
          { binding: 0, resource: { buffer: valueBuffers.get(node.input)! } },
          { binding: 1, resource: { buffer: output } }
        ];
      } else if (node.op === 'fusedConvReluMaxPool2d') {
        const inputShape = plan.valueShapes.get(node.input)!;
        convId = node.conv.id;
        uniformValues = [
          inputShape.width,
          inputShape.height,
          outputShape.width,
          outputShape.height,
          blocksForChannels(inputShape.channels),
          blocksForChannels(outputShape.channels)
        ];
        const packed = this._packedConvs.get(convId)!;
        entries = [
          { binding: 0, resource: { buffer: valueBuffers.get(node.input)! } },
          { binding: 1, resource: { buffer: packed.weights } },
          { binding: 2, resource: { buffer: packed.bias } },
          { binding: 3, resource: { buffer: output } }
        ];
      } else if (node.op === 'fusedUpsampleConcatConv2d') {
        convId = node.conv.id;
        const firstShape = plan.valueShapes.get(node.inputs[0].value)!;
        const secondShape = plan.valueShapes.get(node.inputs[1].value)!;
        uniformValues = [
          outputShape.width,
          outputShape.height,
          blocksForChannels(outputShape.channels),
          blocksForChannels(this._model.convChannels.get(convId)!.inputChannels),
          firstShape.width,
          firstShape.height,
          secondShape.width,
          secondShape.height
        ];
        const packed = this._packedConvs.get(convId)!;
        entries = [
          {
            binding: 0,
            resource: { buffer: valueBuffers.get(node.inputs[0].value)! }
          },
          {
            binding: 1,
            resource: { buffer: valueBuffers.get(node.inputs[1].value)! }
          },
          { binding: 2, resource: { buffer: packed.weights } },
          { binding: 3, resource: { buffer: packed.bias } },
          { binding: 4, resource: { buffer: output } }
        ];
      } else {
        throw new Error(`Unexpected native node ${node.op}`);
      }

      const uniform = createUniformBuffer(
        this._device,
        `oidn/${node.id}/params/${width}x${height}`,
        uniformValues
      );
      ownedBuffers.push(uniform);
      entries.push({ binding: entries.length, resource: { buffer: uniform } });
      nodeBindings.push(
        this._device.createBindGroup({
          label: `oidn/${node.id}/bindings`,
          layout: pipeline.getBindGroupLayout(0),
          entries
        })
      );
    });

    const inputUniform = createUniformBuffer(
      this._device,
      `oidn/input/params/${width}x${height}`,
      [
        width,
        height,
        blocksForChannels(this._model.inputChannels),
        this._model.inputChannels
      ]
    );
    ownedBuffers.push(inputUniform);

    return {
      plan,
      valueBuffers,
      slots,
      nodeBindings,
      nodePipelines,
      inputPipeline,
      inputUniform,
      ownedBuffers,
      lastUsed: ++this._clock
    };
  }

  private _execution(width: number, height: number) {
    const key = `${width}x${height}`;
    let execution = this._executionCache.get(key);
    if (!execution) {
      execution = this._createExecution(width, height);
      this._executionCache.set(key, execution);
      if (this._executionCache.size > this._shapeCacheSize) {
        const oldest = [...this._executionCache.entries()]
          .filter(([candidateKey]) => candidateKey !== key)
          .sort((a, b) => a[1].lastUsed - b[1].lastUsed)[0];
        if (oldest) {
          this._executionCache.delete(oldest[0]);
          // Commands using an evicted plan may still be submitted. Defer actual
          // destruction until all work currently on the shared queue completes.
          void this._device.queue.onSubmittedWorkDone().then(() => {
            oldest[1].slots.forEach((slot) => slot.buffer.destroy());
            oldest[1].ownedBuffers.forEach((buffer) => buffer.destroy());
          });
        }
      }
    }
    execution.lastUsed = ++this._clock;
    return execution;
  }

  /** Captures per-pass GPU timestamps for the next execute call when supported. */
  profileNextExecution() {
    if (!this._device.features.has('timestamp-query')) return false;
    this._profileNextExecution = true;
    return true;
  }

  getLastExecutionProfile() {
    return this._lastExecutionProfile;
  }

  execute(inputBuffers: readonly GPUBuffer[], width: number, height: number) {
    const sourceCount = this._model.inputChannels / 3;
    if (inputBuffers.length !== sourceCount) {
      throw new Error(
        `Native OIDN expected ${sourceCount} input buffers, got ${inputBuffers.length}`
      );
    }
    const execution = this._execution(width, height);
    const profileLabels = [
      'input-pack',
      ...execution.plan.nodes.map((node) => node.id)
    ];
    const shouldProfile =
      this._profileNextExecution &&
      this._device.features.has('timestamp-query');
    this._profileNextExecution = false;
    const queryCount = profileLabels.length * 2;
    const querySet = shouldProfile
      ? this._device.createQuerySet({ type: 'timestamp', count: queryCount })
      : undefined;
    const queryBufferSize = queryCount * 8;
    const queryResolveBuffer = shouldProfile
      ? this._device.createBuffer({
          label: `oidn/profile/resolve/${width}x${height}`,
          size: queryBufferSize,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
        })
      : undefined;
    const queryReadbackBuffer = shouldProfile
      ? this._device.createBuffer({
          label: `oidn/profile/readback/${width}x${height}`,
          size: queryBufferSize,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        })
      : undefined;
    const passDescriptor = (label: string, index: number) => ({
      label,
      ...(querySet
        ? {
            timestampWrites: {
              querySet,
              beginningOfPassWriteIndex: index * 2,
              endOfPassWriteIndex: index * 2 + 1
            }
          }
        : {})
    });
    const encoder = this._device.createCommandEncoder({
      label: `oidn/native/${width}x${height}`
    });

    const inputEntries: GPUBindGroupEntry[] = inputBuffers.map(
      (buffer, binding) => ({ binding, resource: { buffer } })
    );
    inputEntries.push({
      binding: sourceCount,
      resource: { buffer: execution.valueBuffers.get(execution.plan.spec.input)! }
    });
    inputEntries.push({
      binding: sourceCount + 1,
      resource: { buffer: execution.inputUniform }
    });
    const inputBindings = this._device.createBindGroup({
      label: 'oidn/input/bindings',
      layout: execution.inputPipeline.getBindGroupLayout(0),
      entries: inputEntries
    });
    {
      const pass = encoder.beginComputePass(
        passDescriptor('oidn/input-pack', 0)
      );
      pass.setPipeline(execution.inputPipeline);
      pass.setBindGroup(0, inputBindings);
      pass.dispatchWorkgroups(
        Math.ceil(width / WORKGROUP_SIZE),
        Math.ceil(height / WORKGROUP_SIZE),
        blocksForChannels(this._model.inputChannels)
      );
      pass.end();
    }

    execution.plan.plannedNodes.forEach(({ node, outputShape }, index) => {
      const pass = encoder.beginComputePass(
        passDescriptor(`oidn/${execution.plan.nodes[index].id}`, index + 1)
      );
      pass.setPipeline(execution.nodePipelines[index]);
      pass.setBindGroup(0, execution.nodeBindings[index]);
      if (
        this.precision === 'fp32' &&
        (node.op === 'fusedUpsampleConcatConv2d' ||
          (node.op === 'conv2d' &&
            node.id !== execution.plan.spec.output))
      ) {
        pass.dispatchWorkgroups(
          Math.ceil(
            (outputShape.width * outputShape.height) / TILED_CONV_M
          ),
          Math.ceil(
            blocksForChannels(outputShape.channels) / TILED_CONV_N_BLOCKS
          ),
          1
        );
      } else {
        pass.dispatchWorkgroups(
          Math.ceil(outputShape.width / WORKGROUP_SIZE),
          Math.ceil(outputShape.height / WORKGROUP_SIZE),
          blocksForChannels(outputShape.channels)
        );
      }
      pass.end();
    });

    if (querySet) {
      encoder.resolveQuerySet(
        querySet,
        0,
        queryCount,
        queryResolveBuffer!,
        0
      );
      encoder.copyBufferToBuffer(
        queryResolveBuffer!,
        0,
        queryReadbackBuffer!,
        0,
        queryBufferSize
      );
    }

    this._device.queue.submit([encoder.finish()]);
    if (querySet) {
      this._lastExecutionProfile = (async () => {
        try {
          await queryReadbackBuffer!.mapAsync(GPUMapMode.READ);
          const timestamps = new BigUint64Array(
            queryReadbackBuffer!.getMappedRange()
          );
          const layers = profileLabels.map((id, index) => ({
            id,
            durationMs:
              Number(timestamps[index * 2 + 1] - timestamps[index * 2]) /
              1_000_000
          }));
          return {
            totalMs: layers.reduce(
              (sum, layer) => sum + layer.durationMs,
              0
            ),
            layers
          };
        } finally {
          if (queryReadbackBuffer!.mapState === 'mapped') {
            queryReadbackBuffer!.unmap();
          }
          querySet.destroy();
          queryResolveBuffer!.destroy();
          queryReadbackBuffer!.destroy();
        }
      })();
    }
    return execution.valueBuffers.get(execution.plan.spec.output)!;
  }

  /** Compatibility path for ImageData/HDR arrays without TensorFlow.js. */
  async executeCPU(
    interleavedInput: Float32Array,
    width: number,
    height: number
  ): Promise<Float32Array> {
    const expectedLength = width * height * this._model.inputChannels;
    if (interleavedInput.length !== expectedLength) {
      throw new Error(
        `Native OIDN CPU input has ${interleavedInput.length} values, expected ${expectedLength}`
      );
    }
    const execution = this._execution(width, height);
    const sourceCount = this._model.inputChannels / 3;
    const pixelCount = width * height;
    if (!execution.cpuInputBuffers) {
      execution.cpuInputBuffers = Array.from({ length: sourceCount }, (_, index) => {
        const buffer = this._device.createBuffer({
          label: `oidn/cpu-input/${width}x${height}/${index}`,
          size: pixelCount * 16,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        execution.ownedBuffers.push(buffer);
        return buffer;
      });
      execution.cpuReadbackBuffer = this._device.createBuffer({
        label: `oidn/cpu-readback/${width}x${height}`,
        size: pixelCount * 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
      execution.ownedBuffers.push(execution.cpuReadbackBuffer);
    }

    for (let source = 0; source < sourceCount; source++) {
      const upload = new Float32Array(pixelCount * 4);
      for (let pixel = 0; pixel < pixelCount; pixel++) {
        const inputOffset =
          pixel * this._model.inputChannels + source * 3;
        const outputOffset = pixel * 4;
        upload[outputOffset] = interleavedInput[inputOffset];
        upload[outputOffset + 1] = interleavedInput[inputOffset + 1];
        upload[outputOffset + 2] = interleavedInput[inputOffset + 2];
      }
      this._device.queue.writeBuffer(
        execution.cpuInputBuffers[source],
        0,
        upload
      );
    }

    const output = this.execute(execution.cpuInputBuffers, width, height);
    const encoder = this._device.createCommandEncoder({
      label: `oidn/cpu-readback/${width}x${height}`
    });
    encoder.copyBufferToBuffer(
      output,
      0,
      execution.cpuReadbackBuffer!,
      0,
      pixelCount * 16
    );
    this._device.queue.submit([encoder.finish()]);
    await execution.cpuReadbackBuffer!.mapAsync(GPUMapMode.READ);
    const rgba = new Float32Array(
      execution.cpuReadbackBuffer!.getMappedRange()
    );
    const rgb = new Float32Array(pixelCount * 3);
    for (let pixel = 0; pixel < pixelCount; pixel++) {
      rgb[pixel * 3] = rgba[pixel * 4];
      rgb[pixel * 3 + 1] = rgba[pixel * 4 + 1];
      rgb[pixel * 3 + 2] = rgba[pixel * 4 + 2];
    }
    execution.cpuReadbackBuffer!.unmap();
    return rgb;
  }

  dispose() {
    for (const packed of this._packedConvs.values()) {
      packed.weights.destroy();
      packed.bias.destroy();
    }
    this._packedConvs.clear();
    for (const execution of this._executionCache.values()) {
      execution.slots.forEach((slot) => slot.buffer.destroy());
      execution.ownedBuffers.forEach((buffer) => buffer.destroy());
    }
    this._executionCache.clear();
    this._pipelineCache.clear();
  }
}
