export type FinalRgbPrecision = 'fp16' | 'fp32';
export type FinalRgbActivation = 'relu' | 'identity';

const WORKGROUP_SIZE = 8;
const PATCH_SIZE = WORKGROUP_SIZE + 2;
const KERNEL_ELEMENTS = 3 * 3;

/** Shared storage required by the 8x8 final-RGB workgroup. */
export function sharedMemoryBytes(
  precision: FinalRgbPrecision,
  inputBlocks: number,
  cacheWeights = false
) {
  const bytesPerScalar = precision === 'fp16' ? 2 : 4;
  const inputVec4Count = PATCH_SIZE * PATCH_SIZE * inputBlocks;
  const weightVec4Count = cacheWeights
    ? KERNEL_ELEMENTS * inputBlocks * 4
    : 0;
  const vec4Count = inputVec4Count + weightVec4Count;
  return vec4Count * 4 * bytesPerScalar;
}

function storageVecType(precision: FinalRgbPrecision) {
  return precision === 'fp16' ? 'vec4<f16>' : 'vec4<f32>';
}

function shaderPreamble(precision: FinalRgbPrecision) {
  return precision === 'fp16' ? 'enable f16;\n' : '';
}

function accumulationCode(
  precision: FinalRgbPrecision,
  inputExpression: string,
  weightExpression: string
) {
  const weight = (lane: number) =>
    `${weightExpression}[weightBase + ${lane}u]`;
  if (precision === 'fp16') {
    return /* wgsl */ `
        let inputValue = vec4<f16>(${inputExpression});
        var partial = vec4<f16>(0.0h);
        partial = fma(${weight(0)}, vec4<f16>(inputValue.x), partial);
        partial = fma(${weight(1)}, vec4<f16>(inputValue.y), partial);
        partial = fma(${weight(2)}, vec4<f16>(inputValue.z), partial);
        partial = fma(${weight(3)}, vec4<f16>(inputValue.w), partial);
        acc += vec4<f32>(partial);
`;
  }
  return /* wgsl */ `
        let inputValue = vec4<f32>(${inputExpression});
        acc = fma(vec4<f32>(${weight(0)}), vec4<f32>(inputValue.x), acc);
        acc = fma(vec4<f32>(${weight(1)}), vec4<f32>(inputValue.y), acc);
        acc = fma(vec4<f32>(${weight(2)}), vec4<f32>(inputValue.z), acc);
        acc = fma(vec4<f32>(${weight(3)}), vec4<f32>(inputValue.w), acc);
`;
}

/**
 * Builds the final three-channel same-padded convolution shader.
 *
 * The bind-group layout and Params block intentionally match createConvShader:
 * input, weights, bias, output, then uniform params. Weights retain the
 * output-major packed ABI, with four vec4 values per (kernel position, input
 * block), one for each input lane.
 */
export function createFinalRgbShader(
  precision: FinalRgbPrecision,
  activation: FinalRgbActivation,
  inputBlocks: number,
  cacheWeights = false
) {
  if (!Number.isInteger(inputBlocks) || inputBlocks < 1) {
    throw new Error(`Final RGB shader requires positive input blocks, got ${inputBlocks}`);
  }
  const inputType = storageVecType(precision);
  const outputType = 'vec4<f32>';
  const inputTileValues = PATCH_SIZE * PATCH_SIZE * inputBlocks;
  const weightTileValues = KERNEL_ELEMENTS * inputBlocks * 4;
  const stored = activation === 'relu'
    ? 'max(acc, vec4<f32>(0.0))'
    : 'acc';
  const accumulation = accumulationCode(
    precision,
    'inputTile[patchBase + inputBlock]',
    cacheWeights ? 'weightTile' : 'weights'
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
@group(0) @binding(1) var<storage, read> weights: array<${inputType}>;
@group(0) @binding(2) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outputData: array<${outputType}>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> inputTile: array<${inputType}, ${inputTileValues}>;
${cacheWeights
    ? `var<workgroup> weightTile: array<${inputType}, ${weightTileValues}>;`
    : ''}

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE}, 1)
fn main(
  @builtin(local_invocation_id) localId: vec3<u32>,
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let localLinear = localId.y * ${WORKGROUP_SIZE}u + localId.x;
  for (
    var loadIndex = localLinear;
    loadIndex < ${inputTileValues}u;
    loadIndex += ${WORKGROUP_SIZE * WORKGROUP_SIZE}u
  ) {
    let tilePixel = loadIndex / ${inputBlocks}u;
    let inputBlock = loadIndex % ${inputBlocks}u;
    let tileX = tilePixel % ${PATCH_SIZE}u;
    let tileY = tilePixel / ${PATCH_SIZE}u;
    let inputX = i32(workgroupId.x * ${WORKGROUP_SIZE}u + tileX) - 1;
    let inputY = i32(workgroupId.y * ${WORKGROUP_SIZE}u + tileY) - 1;
    var value = ${inputType}(0.0);
    if (
      inputX >= 0 && inputX < i32(params.inputWidth) &&
      inputY >= 0 && inputY < i32(params.inputHeight)
    ) {
      let inputIndex =
        (u32(inputY) * params.inputWidth + u32(inputX)) *
        ${inputBlocks}u + inputBlock;
      value = inputData[inputIndex];
    }
    inputTile[loadIndex] = value;
  }

${cacheWeights ? `  for (
    var loadIndex = localLinear;
    loadIndex < ${weightTileValues}u;
    loadIndex += ${WORKGROUP_SIZE * WORKGROUP_SIZE}u
  ) {
    weightTile[loadIndex] = weights[loadIndex];
  }

` : ''}  // Out-of-range invocations must reach this barrier before returning.
  workgroupBarrier();

  let outputInBounds =
    gid.x < params.outputWidth &&
    gid.y < params.outputHeight &&
    gid.z < params.outputBlocks;
  if (!outputInBounds) {
    return;
  }

  var acc = bias[gid.z];
  for (var ky = 0u; ky < 3u; ky++) {
    let inputY = i32(gid.y) + i32(ky) - 1;
    if (inputY < 0 || inputY >= i32(params.inputHeight)) {
      continue;
    }
    for (var kx = 0u; kx < 3u; kx++) {
      let inputX = i32(gid.x) + i32(kx) - 1;
      if (inputX < 0 || inputX >= i32(params.inputWidth)) {
        continue;
      }
      let patchBase =
        ((localId.y + ky) * ${PATCH_SIZE}u + localId.x + kx) *
        ${inputBlocks}u;
      for (var inputBlock = 0u; inputBlock < ${inputBlocks}u; inputBlock++) {
        let weightBase =
          ((ky * 3u + kx) * ${inputBlocks}u + inputBlock) * 4u;
        ${accumulation}
      }
    }
  }

  let outputIndex =
    (gid.y * params.outputWidth + gid.x) * ${1}u + gid.z;
  outputData[outputIndex] = ${stored};
}
`;
}
