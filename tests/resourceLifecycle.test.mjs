import test from 'node:test';
import assert from 'node:assert/strict';

import { validateUNetModel } from '../lib/modelSpec.js';
import { OIDNResourceTracker } from '../lib/resourceTracker.js';
import { HostTensor, TensorDesc } from '../lib/tza.js';
import { NativeUNetExecutor } from '../lib/nativeUNet.js';
import { WebNNUNetExecutor } from '../lib/webnnUNet.js';

const TEST_SPEC = {
  schemaVersion: 1,
  id: 'resource-test-unet',
  family: 'resource-test',
  input: 'input',
  output: 'output',
  receptiveField: 3,
  nodes: [
    {
      op: 'conv2d',
      id: 'output',
      input: 'input',
      weight: 'output.weight',
      bias: 'output.bias',
      activation: 'identity',
      padding: 'same'
    }
  ]
};

function hostTensor(dims, layout = 'x') {
  const desc = new TensorDesc();
  desc.dims = [...dims];
  desc.paddedDims = [...dims];
  desc.layout = layout;
  desc.dataType = 'Float16';
  return new HostTensor(desc, new Uint8Array(desc.getByteSize()));
}

function testModel() {
  return validateUNetModel(
    new Map([
      ['output.weight', hostTensor([3, 3, 3, 3], 'oihw')],
      ['output.bias', hostTensor([3])]
    ]),
    TEST_SPEC
  );
}

class FakeResource {
  destroyCalls = 0;

  destroy() {
    this.destroyCalls++;
  }
}

class FakeBuffer extends FakeResource {
  mapState = 'unmapped';

  constructor(size = 16, mappedAtCreation = false) {
    super();
    this.data = new ArrayBuffer(size);
    this.mapState = mappedAtCreation ? 'mapped' : 'unmapped';
  }

  getMappedRange() {
    return this.data;
  }

  unmap() {
    this.mapState = 'unmapped';
  }
}

function fakeDevice() {
  const device = {
    features: new Set(['shader-f16']),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupStorageSize: 32768
    },
    failBindGroup: false,
    failBufferAt: undefined,
    createBufferCalls: 0,
    buffers: [],
    queue: {
      submit() {},
      onSubmittedWorkDone: () => Promise.resolve()
    },
    createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createComputePipelineAsync: async () => ({
      getBindGroupLayout: () => ({})
    }),
    createBuffer(descriptor) {
      this.createBufferCalls++;
      if (this.createBufferCalls === this.failBufferAt) {
        throw new Error('injected buffer allocation failure');
      }
      const buffer = new FakeBuffer(
        descriptor.size,
        descriptor.mappedAtCreation
      );
      this.buffers.push(buffer);
      return buffer;
    },
    createBindGroup() {
      if (this.failBindGroup) throw new Error('injected bind group failure');
      return {};
    },
    createCommandEncoder: () => ({
      beginComputePass: () => ({
        setPipeline() {},
        setBindGroup() {},
        dispatchWorkgroups() {},
        end() {}
      }),
      finish: () => ({})
    })
  };
  return device;
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

async function withFakeWebNN(run, { build } = {}) {
  const previous = {
    navigator: Object.getOwnPropertyDescriptor(globalThis, 'navigator'),
    builder: Object.getOwnPropertyDescriptor(globalThis, 'MLGraphBuilder'),
    usage: Object.getOwnPropertyDescriptor(globalThis, 'GPUBufferUsage')
  };
  const device = fakeDevice();
  const context = new FakeResource();
  Object.assign(context, {
    async createExportableTensor() {
      return new FakeResource();
    },
    async exportToGPU() {
      return device.createBuffer({ size: 16 });
    },
    dispatch() {},
    opSupportLimits() {
      const operand = { dataTypes: ['float16'] };
      return { conv2d: { input: operand, filter: operand, output: operand } };
    },
    async readTensor() {
      return new ArrayBuffer(0);
    },
    writeTensor() {}
  });

  class FakeBuilder {
    input() {
      return {};
    }
    constant() {
      return {};
    }
    conv2d() {
      return {};
    }
    relu(value) {
      return value;
    }
    maxPool2d() {
      return {};
    }
    resample2d() {
      return {};
    }
    concat() {
      return {};
    }
    async build(outputs) {
      return build ? build(outputs) : new FakeResource();
    }
  }

  Object.defineProperties(globalThis, {
    navigator: {
      configurable: true,
      value: { ml: { createContext: async () => context } }
    },
    MLGraphBuilder: { configurable: true, value: FakeBuilder },
    GPUBufferUsage: {
      configurable: true,
      value: {
        STORAGE: 1,
        COPY_SRC: 2,
        COPY_DST: 4,
        UNIFORM: 8,
        QUERY_RESOLVE: 16,
        MAP_READ: 32
      }
    }
  });

  try {
    await run({ device, context });
  } finally {
    for (const [key, descriptor] of Object.entries(previous)) {
      const name =
        key === 'builder'
          ? 'MLGraphBuilder'
          : key === 'usage'
          ? 'GPUBufferUsage'
          : key;
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete globalThis[name];
    }
  }
}

test('resource tracker releases each owned handle exactly once', () => {
  const tracker = new OIDNResourceTracker();
  const buffer = new FakeResource();
  tracker.track('gpu-buffer', buffer);
  tracker.track('gpu-buffer', buffer);

  assert.deepEqual(tracker.snapshot(), {
    live: 1,
    created: 1,
    destroyed: 0,
    peakLive: 1,
    pending: 0,
    byKind: {
      'gpu-buffer': { created: 1, destroyed: 0, live: 1, peakLive: 1 }
    }
  });
  tracker.release('gpu-buffer', buffer, () => buffer.destroy());
  tracker.release('gpu-buffer', buffer, () => buffer.destroy());
  assert.equal(buffer.destroyCalls, 1);
  assert.equal(tracker.snapshot().live, 0);
  assert.equal(tracker.snapshot().created, tracker.snapshot().destroyed);
});

test('native WGSL shape eviction and dispose release every owned GPU buffer', async () => {
  await withFakeWebNN(async ({ device }) => {
    const executor = new NativeUNetExecutor(device, testModel(), {
      precision: 'fp16',
      shapeCacheSize: 2,
      kernel: 'direct'
    });
    const input = new FakeBuffer();
    executor.execute([input], 16, 16);
    executor.execute([input], 32, 16);
    executor.execute([input], 48, 16);
    await new Promise((resolve) => setImmediate(resolve));

    const cached = executor.getResourceInfo();
    assert.equal(cached.pending, 0);
    assert.equal(cached.byKind['gpu-buffer'].live, 10);
    assert.equal(cached.byKind['gpu-buffer'].destroyed, 4);

    executor.dispose();
    executor.dispose();
    const disposed = executor.getResourceInfo();
    assert.equal(disposed.pending, 0);
    assert.equal(disposed.live, 0);
    assert.equal(disposed.created, disposed.destroyed);
    assert.ok(device.buffers.every((buffer) => buffer.destroyCalls === 1));
    assert.equal(input.destroyCalls, 0);
  });
});

test('native WGSL rolls back a partially allocated shape on failure', async () => {
  await withFakeWebNN(async ({ device }) => {
    const executor = new NativeUNetExecutor(device, testModel(), {
      precision: 'fp16',
      kernel: 'direct'
    });
    const baseline = executor.getResourceInfo().live;
    device.failBufferAt = device.createBufferCalls + 3;

    assert.throws(
      () => executor.execute([new FakeBuffer()], 16, 16),
      /injected buffer allocation failure/
    );
    assert.equal(executor.getResourceInfo().live, baseline);
    executor.dispose();
    assert.equal(executor.getResourceInfo().live, 0);
    assert.ok(device.buffers.every((buffer) => buffer.destroyCalls === 1));
  });
});

test('WebNN shape cache stays bounded and dispose returns to zero live resources', async () => {
  await withFakeWebNN(async ({ device, context }) => {
    const executor = new WebNNUNetExecutor(device, testModel(), {
      precision: 'fp16',
      shapeCacheSize: 2
    });
    await executor.prepare();
    await executor.prewarm([
      { width: 16, height: 16 },
      { width: 32, height: 16 },
      { width: 48, height: 16 }
    ]);
    await Promise.resolve();

    const cached = executor.getResourceInfo();
    assert.equal(cached.pending, 0);
    assert.equal(cached.byKind['ml-context'].live, 1);
    assert.equal(cached.byKind['ml-graph'].live, 2);
    assert.equal(cached.byKind['ml-tensor'].live, 4);
    assert.equal(cached.byKind['gpu-buffer'].live, 6);

    executor.dispose();
    executor.dispose();
    const disposed = executor.getResourceInfo();
    assert.equal(disposed.pending, 0);
    assert.equal(disposed.live, 0);
    assert.equal(disposed.created, disposed.destroyed);
    assert.equal(context.destroyCalls, 1);
    assert.ok(device.buffers.every((buffer) => buffer.destroyCalls === 1));
  });
});

test('WebNN cleans a graph that finishes after dispose', async () => {
  const build = deferred();
  await withFakeWebNN(
    async ({ device }) => {
      const executor = new WebNNUNetExecutor(device, testModel(), {
        precision: 'fp16'
      });
      await executor.prepare();
      const prewarm = executor.prewarm([{ width: 16, height: 16 }]);
      await Promise.resolve();
      assert.equal(executor.getResourceInfo().pending, 1);

      executor.dispose();
      build.resolve(new FakeResource());
      await assert.rejects(prewarm, /disposed/);

      const resources = executor.getResourceInfo();
      assert.equal(resources.pending, 0);
      assert.equal(resources.live, 0);
      assert.equal(resources.created, resources.destroyed);
    },
    { build: () => build.promise }
  );
});

test('WebNN releases exported GPU buffers when command setup throws', async () => {
  await withFakeWebNN(async ({ device }) => {
    const executor = new WebNNUNetExecutor(device, testModel(), {
      precision: 'fp16'
    });
    await executor.prepare();
    await executor.prewarm([{ width: 16, height: 16 }]);
    const baseline = executor.getResourceInfo().live;
    device.failBindGroup = true;

    await assert.rejects(
      executor.execute([new FakeBuffer()], 16, 16),
      /injected bind group failure/
    );
    assert.equal(executor.getResourceInfo().live, baseline);
    executor.dispose();
    assert.equal(executor.getResourceInfo().live, 0);
  });
});

test('WebNN prepare failures destroy the partially initialized context', async () => {
  await withFakeWebNN(async ({ device, context }) => {
    context.opSupportLimits = () => ({});
    const executor = new WebNNUNetExecutor(device, testModel(), {
      precision: 'fp16'
    });
    await assert.rejects(executor.prepare(), /does not support FP16/);
    assert.equal(context.destroyCalls, 1);
    assert.equal(executor.getResourceInfo().live, 0);
  });
});
