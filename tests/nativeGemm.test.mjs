import test from 'node:test';
import assert from 'node:assert/strict';

import { Float16Array } from '@petamoriken/float16';
import { NativeUNetExecutor } from '../lib/nativeUNet.js';
import { validateUNetModel } from '../lib/modelSpec.js';
import { HostTensor, TensorDesc } from '../lib/tza.js';

const GPU_USAGE = {
  STORAGE: 1,
  COPY_SRC: 2,
  COPY_DST: 4,
  UNIFORM: 8,
  QUERY_RESOLVE: 16,
  MAP_READ: 32
};

const TEST_SPEC = {
  schemaVersion: 1,
  id: 'native-gemm-test-unet',
  family: 'native-gemm-test',
  input: 'input',
  output: 'output',
  receptiveField: 5,
  nodes: [
    {
      op: 'conv2d',
      id: 'hidden',
      input: 'input',
      weight: 'hidden.weight',
      bias: 'hidden.bias',
      activation: 'relu',
      padding: 'same'
    },
    {
      op: 'conv2d',
      id: 'output',
      input: 'hidden',
      weight: 'output.weight',
      bias: 'output.bias',
      activation: 'identity',
      padding: 'same'
    }
  ]
};

class FakeBuffer {
  constructor(size, mappedAtCreation) {
    this.data = new ArrayBuffer(size);
    this.destroyCalls = 0;
    this.mapState = mappedAtCreation ? 'mapped' : 'unmapped';
  }

  getMappedRange() {
    return this.data;
  }

  unmap() {
    this.mapState = 'unmapped';
  }

  destroy() {
    this.destroyCalls++;
  }
}

function fakeDevice({ maxComputeWorkgroupStorageSize = 32768 } = {}) {
  const shaderLabels = [];
  const shaderCodes = [];
  const pipelineLabels = [];
  const device = {
    features: new Set(['shader-f16']),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupStorageSize
    },
    buffers: [],
    shaderLabels,
    shaderCodes,
    pipelineLabels,
    queue: {
      submit() {},
      onSubmittedWorkDone: () => Promise.resolve()
    },
    createBuffer(descriptor) {
      const buffer = new FakeBuffer(descriptor.size, descriptor.mappedAtCreation);
      this.buffers.push(buffer);
      return buffer;
    },
    createShaderModule(descriptor) {
      shaderLabels.push(descriptor.label);
      shaderCodes.push(descriptor.code);
      return { code: descriptor.code };
    },
    createComputePipeline(descriptor) {
      pipelineLabels.push(descriptor.label);
      return { getBindGroupLayout: () => ({}) };
    },
    async createComputePipelineAsync(descriptor) {
      pipelineLabels.push(descriptor.label);
      return { getBindGroupLayout: () => ({}) };
    },
    createBindGroup() {
      return {};
    },
    createCommandEncoder() {
      return {
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {}
          };
        },
        finish() {
          return {};
        }
      };
    }
  };
  return device;
}

function fusedFinalModel(dataType = 'Float16') {
  const spec = {
    schemaVersion: 1,
    id: 'native-gemm-fused-final-test',
    family: 'native-gemm-fused-final-test',
    input: 'input',
    output: 'output',
    receptiveField: 7,
    nodes: [
      { op: 'conv2d', id: 'branch', input: 'input', weight: 'branch.weight', bias: 'branch.bias', activation: 'relu', padding: 'same' },
      { op: 'upsample2d', id: 'up', input: 'branch', scale: 2, mode: 'nearest' },
      { op: 'conv2d', id: 'skip', input: 'input', weight: 'skip.weight', bias: 'skip.bias', activation: 'relu', padding: 'same' },
      { op: 'concat', id: 'join', inputs: ['up', 'skip'], axis: 'channels' },
      { op: 'conv2d', id: 'output', input: 'join', weight: 'output.weight', bias: 'output.bias', activation: 'identity', padding: 'same' }
    ]
  };
  const tensors = new Map([
    ['branch.weight', tensor([4, 3, 3, 3], 'oihw', dataType, Array(4 * 3 * 3 * 3).fill(0))],
    ['branch.bias', tensor([4], 'x', dataType, [0, 0, 0, 0])],
    ['skip.weight', tensor([4, 3, 3, 3], 'oihw', dataType, Array(4 * 3 * 3 * 3).fill(0))],
    ['skip.bias', tensor([4], 'x', dataType, [0, 0, 0, 0])],
    ['output.weight', tensor([3, 8, 3, 3], 'oihw', dataType, Array(3 * 8 * 3 * 3).fill(0))],
    ['output.bias', tensor([3], 'x', dataType, [1, 2, 3])]
  ]);
  return validateUNetModel(tensors, spec);
}

function wideFinalModel(dataType = 'Float16') {
  const spec = {
    schemaVersion: 1,
    id: 'native-gemm-wide-final-test',
    family: 'native-gemm-wide-final-test',
    input: 'input',
    output: 'output',
    receptiveField: 3,
    nodes: [
      { op: 'conv2d', id: 'hidden', input: 'input', weight: 'hidden.weight', bias: 'hidden.bias', activation: 'relu', padding: 'same' },
      { op: 'conv2d', id: 'output', input: 'hidden', weight: 'output.weight', bias: 'output.bias', activation: 'identity', padding: 'same' }
    ]
  };
  const tensors = new Map([
    ['hidden.weight', tensor([32, 3, 3, 3], 'oihw', dataType, Array(32 * 3 * 3 * 3).fill(0))],
    ['hidden.bias', tensor([32], 'x', dataType, Array(32).fill(1))],
    ['output.weight', tensor([3, 32, 3, 3], 'oihw', dataType, Array(3 * 32 * 3 * 3).fill(0))],
    ['output.bias', tensor([3], 'x', dataType, [1, 2, 3])]
  ]);
  return validateUNetModel(tensors, spec);
}

function tensor(dims, layout, dataType, values) {
  const desc = new TensorDesc();
  desc.dims = [...dims];
  desc.paddedDims = [...dims];
  desc.layout = layout;
  desc.dataType = dataType;
  const typed = dataType === 'Float16'
    ? new Float16Array(values)
    : new Float32Array(values);
  return new HostTensor(
    desc,
    new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength)
  );
}

function testModel(dataType) {
  const hiddenValues = Array.from({ length: 5 * 3 * 3 * 3 }, (_, index) =>
    1 + index / 16
  );
  const outputValues = Array.from({ length: 3 * 5 * 3 * 3 }, (_, index) =>
    11 + index / 16
  );
  const tensors = new Map([
    ['hidden.weight', tensor([5, 3, 3, 3], 'oihw', dataType, hiddenValues)],
    ['hidden.bias', tensor([5], 'x', dataType, [1, 2, 3, 4, 5])],
    ['output.weight', tensor([3, 5, 3, 3], 'oihw', dataType, outputValues)],
    ['output.bias', tensor([3], 'x', dataType, [7, 8, 9])]
  ]);
  return {
    model: validateUNetModel(tensors, TEST_SPEC),
    hiddenValues
  };
}

function expectedPacked(hiddenValues, dataType, layout) {
  const inputChannels = 3;
  const outputChannels = 5;
  const inputBlocks = Math.ceil(inputChannels / 4);
  const outputBlocks = Math.ceil(outputChannels / 4);
  const values = new Array(outputBlocks * 3 * 3 * inputBlocks * 16).fill(0);
  for (let outputBlock = 0; outputBlock < outputBlocks; outputBlock++) {
    for (let y = 0; y < 3; y++) {
      for (let x = 0; x < 3; x++) {
        for (let inputBlock = 0; inputBlock < inputBlocks; inputBlock++) {
          const packedBlock = layout === 'k-major'
            ? ((((y * 3 + x) * inputBlocks + inputBlock) * outputBlocks + outputBlock) * 16)
            : ((((outputBlock * 3 + y) * 3 + x) * inputBlocks + inputBlock) * 16);
          for (let outputLane = 0; outputLane < 4; outputLane++) {
            const outputChannel = outputBlock * 4 + outputLane;
            for (let inputLane = 0; inputLane < 4; inputLane++) {
              const inputChannel = inputBlock * 4 + inputLane;
              const index = packedBlock + inputLane * 4 + outputLane;
              if (outputChannel < outputChannels && inputChannel < inputChannels) {
                values[index] = hiddenValues[
                  ((outputChannel * inputChannels + inputChannel) * 3 + y) * 3 + x
                ];
              }
            }
          }
        }
      }
    }
  }
  if (dataType === 'Float16') {
    const converted = new Float16Array(values);
    return Array.from(new Uint16Array(converted.buffer, converted.byteOffset, converted.length));
  }
  return values;
}

function unpackWeightBuffer(buffer, dataType) {
  return dataType === 'Float16'
    ? Array.from(new Uint16Array(buffer.data))
    : Array.from(new Float32Array(buffer.data));
}

function withGpuUsage(run) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'GPUBufferUsage');
  Object.defineProperty(globalThis, 'GPUBufferUsage', {
    configurable: true,
    value: GPU_USAGE
  });
  return Promise.resolve()
    .then(run)
    .finally(() => {
      if (previous) Object.defineProperty(globalThis, 'GPUBufferUsage', previous);
      else delete globalThis.GPUBufferUsage;
    });
}

function executorOptions(precision, weightLayout, overrides = {}) {
  return {
    precision,
    kernel: 'implicit-gemm',
    gemm: {
      addressMode: 'analytic',
      weightLayout,
      rowsPerThread: 4,
      workgroupSize: [8, 8],
      ...overrides
    }
  };
}

for (const [dataType, precision] of [
  ['Float32', 'fp32'],
  ['Float16', 'fp16']
]) {
  test(`packs ${dataType} output-major and k-major weights with padded lanes`, async () => {
    await withGpuUsage(() => {
      const outputMajorDevice = fakeDevice();
      const outputMajor = new NativeUNetExecutor(
        outputMajorDevice,
        testModel(dataType).model,
        executorOptions(precision, 'output-major')
      );
      const kMajorDevice = fakeDevice();
      const kMajor = new NativeUNetExecutor(
        kMajorDevice,
        testModel(dataType).model,
        executorOptions(precision, 'k-major')
      );
      const hiddenValues = testModel(dataType).hiddenValues;
      const expectedOutput = expectedPacked(hiddenValues, dataType, 'output-major');
      const expectedK = expectedPacked(hiddenValues, dataType, 'k-major');

      // Each executor packs hidden weights first, then hidden bias.
      assert.deepEqual(
        unpackWeightBuffer(outputMajorDevice.buffers[0], dataType),
        expectedOutput
      );
      assert.deepEqual(
        unpackWeightBuffer(kMajorDevice.buffers[0], dataType),
        expectedK
      );
      assert.notDeepEqual(expectedOutput, expectedK);
      assert.equal(outputMajorDevice.buffers[0].data.byteLength, kMajorDevice.buffers[0].data.byteLength);

      outputMajor.dispose();
      kMajor.dispose();
    });
  });
}

test('freezes normalized GEMM configuration without mutating caller input', async () => {
  await withGpuUsage(() => {
    const device = fakeDevice();
    const input = {
      addressMode: 'incremental',
      weightLayout: 'k-major',
      rowsPerThread: 2,
      workgroupSize: [4, 8]
    };
    const executor = new NativeUNetExecutor(
      device,
      testModel('Float32').model,
      { precision: 'fp32', kernel: 'implicit-gemm', gemm: input }
    );
    assert.deepEqual(executor.gemm, {
      finalLayer: 'shared-auto',
      loadMode: 'native',
      addressMode: 'incremental',
      weightLayout: 'k-major',
      rowsPerThread: 2,
      workgroupSize: [4, 8]
    });
    assert.ok(Object.isFrozen(executor.gemm));
    assert.ok(Object.isFrozen(executor.gemm.workgroupSize));
    assert.notStrictEqual(executor.gemm.workgroupSize, input.workgroupSize);
    assert.deepEqual(input, {
      addressMode: 'incremental',
      weightLayout: 'k-major',
      rowsPerThread: 2,
      workgroupSize: [4, 8]
    });
    assert.throws(() => {
      executor.gemm.addressMode = 'analytic';
    }, TypeError);
    executor.dispose();
  });
});

test('rejects unsupported GEMM address, layout, register, workgroup, and storage settings', async () => {
  await withGpuUsage(() => {
    const model = testModel('Float32').model;
    for (const gemm of [
      { addressMode: 'bad' },
      { loadMode: 'bad' },
      { finalLayer: 'bad' },
      { weightLayout: 'bad' },
      { rowsPerThread: 3 },
      { workgroupSize: [2, 8] },
      { workgroupSize: [8, 16] }
    ]) {
      assert.throws(
        () => new NativeUNetExecutor(
          fakeDevice(),
          model,
          { precision: 'fp32', kernel: 'implicit-gemm', gemm }
        ),
        /Unsupported GEMM/
      );
    }
    assert.throws(
      () => new NativeUNetExecutor(
        fakeDevice({ maxComputeWorkgroupStorageSize: 1 }),
        model,
        executorOptions('fp32', 'output-major')
      ),
      /Unsupported GEMM tile configuration/
    );
  });
});

test('pipeline cache ABI separates address, weight layout, and tile variants', async () => {
  await withGpuUsage(async () => {
    const device = fakeDevice();
    const model = testModel('Float32').model;
    const analytic = new NativeUNetExecutor(
      device,
      model,
      executorOptions('fp32', 'output-major')
    );
    await analytic.prepare();
    const firstLabels = [...device.pipelineLabels];
    const kMajor = new NativeUNetExecutor(
      device,
      model,
      executorOptions('fp32', 'k-major', {
        addressMode: 'incremental',
        loadMode: 'packed-all',
        rowsPerThread: 2,
        workgroupSize: [4, 8]
      })
    );
    await kMajor.prepare();
    const secondLabels = device.pipelineLabels.slice(firstLabels.length);
    assert.ok(firstLabels.some((label) => label.includes('address-analytic/weights-output-major-v1/tile-8x8-r4')));
    assert.ok(secondLabels.some((label) => label.includes('address-incremental/weights-k-major-v1/tile-4x8-r2/loads-packed-all')));
    assert.notDeepEqual(firstLabels, secondLabels);
    analytic.dispose();
    kMajor.dispose();
  });
});

test('selects final shared-load variants and falls back when the halo exceeds shared memory', async () => {
  await withGpuUsage(async () => {
    const model = testModel('Float16').model;
    const device = fakeDevice();
    const sharedInput = new NativeUNetExecutor(
      device,
      model,
      executorOptions('fp16', 'k-major', { finalLayer: 'shared-input' })
    );
    await sharedInput.prepare();
    const firstLabels = [...device.pipelineLabels];
    assert.ok(firstLabels.some((label) =>
      label.includes('conv-final-rgb/fp16/identity/in2/weights-storage')
    ));

    const sharedWeights = new NativeUNetExecutor(
      device,
      model,
      executorOptions('fp16', 'k-major', { finalLayer: 'shared-input-weights' })
    );
    await sharedWeights.prepare();
    const secondLabels = device.pipelineLabels.slice(firstLabels.length);
    assert.ok(secondLabels.some((label) =>
      label.includes('conv-final-rgb/fp16/identity/in2/weights-shared')
    ));
    assert.notDeepEqual(firstLabels, secondLabels);

    const fallback = new NativeUNetExecutor(
      fakeDevice({ maxComputeWorkgroupStorageSize: 4096 }),
      wideFinalModel(),
      executorOptions('fp16', 'k-major', { finalLayer: 'shared-input' })
    );
    await fallback.prepare();
    assert.ok(fallback._pipelineCache instanceof Map);
    assert.ok([...fallback._pipelineCache.keys()].some((key) =>
      key.includes('conv-direct/fp16/fp32/identity/in8/out1')
    ));
    assert.ok(![...fallback._pipelineCache.keys()].some((key) =>
      key.includes('conv-final-rgb/')
    ));
    sharedInput.dispose();
    sharedWeights.dispose();
    fallback.dispose();
  });
});

test('shared-auto chooses weight cache, input-only cache, then direct fallback by limit', async () => {
  await withGpuUsage(async () => {
    const cases = [
      [32768, 'weights-shared'],
      [8192, 'weights-storage']
    ];
    for (const [storageLimit, weightMode] of cases) {
      const device = fakeDevice({ maxComputeWorkgroupStorageSize: storageLimit });
      const executor = new NativeUNetExecutor(
        device,
        wideFinalModel(),
        executorOptions('fp16', 'k-major', { finalLayer: 'shared-auto' })
      );
      await executor.prepare();
      assert.ok(device.pipelineLabels.some((label) =>
        label.includes(`conv-final-rgb/fp16/identity/in8/${weightMode}`)
      ));
      executor.dispose();
    }

    const fallbackDevice = fakeDevice({ maxComputeWorkgroupStorageSize: 4096 });
    const fallback = new NativeUNetExecutor(
      fallbackDevice,
      wideFinalModel(),
      executorOptions('fp16', 'k-major', { finalLayer: 'shared-auto' })
    );
    await fallback.prepare();
    assert.ok(fallbackDevice.pipelineLabels.some((label) =>
      label.includes('conv-direct/fp16/fp32/identity/in8/out1')
    ));
    assert.ok(!fallbackDevice.pipelineLabels.some((label) =>
      label.includes('conv-final-rgb/')
    ));
    fallback.dispose();
  });
});

test('explicit direct kernel ignores final-layer and GEMM load tuning flags', async () => {
  await withGpuUsage(async () => {
    const device = fakeDevice();
    const executor = new NativeUNetExecutor(
      device,
      testModel('Float16').model,
      {
        precision: 'fp16',
        kernel: 'direct',
        gemm: {
          finalLayer: 'shared-input-weights',
          loadMode: 'packed-all',
          addressMode: 'base-offset',
          weightLayout: 'k-major',
          rowsPerThread: 2,
          workgroupSize: [4, 8]
        }
      }
    );
    await executor.prepare();
    assert.ok(device.pipelineLabels.some((label) =>
      label.includes('conv-direct/fp16/fp32/identity/in2/out1')
    ));
    assert.ok(!device.pipelineLabels.some((label) => label.includes('conv-final-rgb/')));
    assert.equal(executor._packedConvs.get('output').weightLayout, 'output-major');
    executor.dispose();
  });
});

test('final fused decoder uses direct output-major ABI and fp32 storage', async () => {
  await withGpuUsage(async () => {
    const device = fakeDevice();
    const executor = new NativeUNetExecutor(
      device,
      fusedFinalModel(),
      executorOptions('fp16', 'k-major')
    );
    await executor.prepare();
    assert.ok(device.pipelineLabels.some((label) =>
      label.includes('decoder-direct/fp16/fp32/identity/1+1/out1/up0')
    ));
    // The final direct convolution must retain the old output-major packing.
    assert.equal(executor._packedConvs.get('output').weightLayout, 'output-major');
    executor.dispose();
  });
});
