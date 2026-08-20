import test from 'node:test';
import assert from 'node:assert/strict';

import {
  detectUNetModelSpec,
  OIDN_UNET_LARGE_SPEC,
  OIDN_UNET_SMALL_SPEC,
  validateUNetModel
} from '../lib/modelSpec.js';
import { optimizeModelGraph, planModelExecution } from '../lib/graphOptimizer.js';
import { HostTensor, TensorDesc } from '../lib/tza.js';

function hostTensor(dims, layout = 'x', dataType = 'Float16') {
  const desc = new TensorDesc();
  desc.dims = [...dims];
  desc.paddedDims = [...dims];
  desc.layout = layout;
  desc.dataType = dataType;
  return new HostTensor(desc, new Uint8Array(desc.getByteSize()));
}

function makeModelTensors(spec, inputChannels = 3) {
  const tensors = new Map();
  const channels = new Map([[spec.input, inputChannels]]);

  for (const node of spec.nodes) {
    if (node.op === 'conv2d') {
      const inChannels = channels.get(node.input);
      assert.notEqual(inChannels, undefined);
      const outChannels = node.id === spec.output ? 3 : 4;
      tensors.set(
        node.weight,
        hostTensor([outChannels, inChannels, 3, 3], 'oihw')
      );
      tensors.set(node.bias, hostTensor([outChannels]));
      channels.set(node.id, outChannels);
    } else if (node.op === 'concat') {
      channels.set(
        node.id,
        node.inputs.reduce((sum, input) => sum + channels.get(input), 0)
      );
    } else {
      channels.set(node.id, channels.get(node.input));
    }
  }
  return tensors;
}

test('detects and validates built-in small and large descriptors', () => {
  for (const spec of [OIDN_UNET_SMALL_SPEC, OIDN_UNET_LARGE_SPEC]) {
    const tensors = makeModelTensors(spec, 9);
    assert.equal(detectUNetModelSpec(tensors), spec);
    const validated = validateUNetModel(tensors);
    assert.equal(validated.spec, spec);
    assert.equal(validated.inputChannels, 9);
    assert.equal(validated.outputChannels, 3);
    assert.equal(validated.tensorDataType, 'Float16');
  }
});

test('rejects an unknown topology instead of silently using a stale graph', () => {
  const tensors = makeModelTensors(OIDN_UNET_SMALL_SPEC);
  tensors.set('future_block.weight', hostTensor([4, 4, 3, 3], 'oihw'));
  assert.throws(
    () => detectUNetModelSpec(tensors),
    /Unsupported OIDN model topology/
  );
});

test('validates tensor byte length and graph channel flow', () => {
  const tensors = makeModelTensors(OIDN_UNET_SMALL_SPEC);
  tensors.get('enc_conv0.weight').data = new Uint8Array(2);
  assert.throws(
    () => validateUNetModel(tensors, OIDN_UNET_SMALL_SPEC),
    /bytes, expected/
  );

  const wrongChannels = makeModelTensors(OIDN_UNET_SMALL_SPEC);
  wrongChannels.set('enc_conv1.weight', hostTensor([4, 7, 3, 3], 'oihw'));
  assert.throws(
    () => validateUNetModel(wrongChannels, OIDN_UNET_SMALL_SPEC),
    /expects 7 input channels/
  );
});

test('fuses topology patterns without hard-coding a model family', () => {
  const validated = validateUNetModel(
    makeModelTensors(OIDN_UNET_SMALL_SPEC),
    OIDN_UNET_SMALL_SPEC
  );
  const graph = optimizeModelGraph(validated);

  assert.deepEqual(graph.fusions, {
    convPool: 4,
    upsampleConcatConv: 4
  });
  assert.equal(
    graph.nodes.filter((node) => node.op === 'fusedConvReluMaxPool2d').length,
    4
  );
  assert.equal(
    graph.nodes.filter((node) => node.op === 'fusedUpsampleConcatConv2d')
      .length,
    4
  );
  assert.equal(graph.nodes.length, 16);
});

test('plans fused graph shapes and lifetimes', () => {
  const validated = validateUNetModel(
    makeModelTensors(OIDN_UNET_LARGE_SPEC, 9),
    OIDN_UNET_LARGE_SPEC
  );
  const plan = planModelExecution(validated, 384, 256);

  assert.deepEqual(plan.inputShape, { width: 384, height: 256, channels: 9 });
  assert.deepEqual(plan.valueShapes.get('pool4'), {
    width: 24,
    height: 16,
    channels: 4
  });
  assert.deepEqual(plan.valueShapes.get(OIDN_UNET_LARGE_SPEC.output), {
    width: 384,
    height: 256,
    channels: 3
  });
  assert.equal(plan.plannedNodes.at(-1).lastUse, plan.nodes.length);
});
