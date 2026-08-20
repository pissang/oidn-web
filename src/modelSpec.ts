import type { HostTensor } from './tza';

/**
 * Versioned, runtime-independent description of an OIDN network.
 *
 * TZA contains named tensors but no executable graph. Keeping the graph in a
 * small declarative descriptor makes model upgrades independent from the WGSL
 * kernels: a new OIDN topology only needs a new descriptor and validation
 * fixture unless it introduces a genuinely new operation.
 */
export interface UNetModelSpec {
  schemaVersion: 1;
  id: string;
  family: 'oidn-unet-small' | 'oidn-unet-large' | (string & {});
  input: string;
  output: string;
  receptiveField: number;
  nodes: readonly ModelNodeSpec[];
  /** Reject unrecognised tensors so an upstream topology change is explicit. */
  allowAdditionalTensors?: boolean;
}

export type ModelActivation = 'identity' | 'relu';

export interface Conv2DNodeSpec {
  op: 'conv2d';
  id: string;
  input: string;
  weight: string;
  bias: string;
  activation: ModelActivation;
  padding: 'same';
}

export interface MaxPool2DNodeSpec {
  op: 'maxPool2d';
  id: string;
  input: string;
  size: 2;
  stride: 2;
  padding: 'same';
}

export interface Upsample2DNodeSpec {
  op: 'upsample2d';
  id: string;
  input: string;
  scale: 2;
  mode: 'nearest';
}

export interface ConcatNodeSpec {
  op: 'concat';
  id: string;
  inputs: readonly string[];
  axis: 'channels';
}

export type ModelNodeSpec =
  | Conv2DNodeSpec
  | MaxPool2DNodeSpec
  | Upsample2DNodeSpec
  | ConcatNodeSpec;

export interface ValidatedConvTensor {
  weight: HostTensor;
  bias: HostTensor;
  inputChannels: number;
  outputChannels: number;
  kernelHeight: number;
  kernelWidth: number;
}

export interface ModelConvChannels {
  inputChannels: number;
  outputChannels: number;
}

export interface UNetModelGraph {
  spec: UNetModelSpec;
  inputChannels: number;
  outputChannels: number;
  channelsByValue: ReadonlyMap<string, number>;
  convChannels: ReadonlyMap<string, ModelConvChannels>;
}

export interface ValidatedUNetModel extends UNetModelGraph {
  tensorDataType: HostTensor['desc']['dataType'];
  convTensors: ReadonlyMap<string, ValidatedConvTensor>;
}

function conv(id: string, input: string): Conv2DNodeSpec {
  return {
    op: 'conv2d',
    id,
    input,
    weight: `${id}.weight`,
    bias: `${id}.bias`,
    activation: 'relu',
    padding: 'same'
  };
}

function pool(id: string, input: string): MaxPool2DNodeSpec {
  return {
    op: 'maxPool2d',
    id,
    input,
    size: 2,
    stride: 2,
    padding: 'same'
  };
}

function upsample(id: string, input: string): Upsample2DNodeSpec {
  return {
    op: 'upsample2d',
    id,
    input,
    scale: 2,
    mode: 'nearest'
  };
}

function concat(
  id: string,
  first: string,
  second: string
): ConcatNodeSpec {
  return {
    op: 'concat',
    id,
    inputs: [first, second],
    axis: 'channels'
  };
}

export const OIDN_UNET_SMALL_SPEC: UNetModelSpec = {
  schemaVersion: 1,
  id: 'oidn-unet-small-v1',
  family: 'oidn-unet-small',
  input: 'input',
  output: 'dec_conv0',
  receptiveField: 174,
  nodes: [
    conv('enc_conv0', 'input'),
    conv('enc_conv1', 'enc_conv0'),
    pool('pool1', 'enc_conv1'),
    conv('enc_conv2', 'pool1'),
    pool('pool2', 'enc_conv2'),
    conv('enc_conv3', 'pool2'),
    pool('pool3', 'enc_conv3'),
    conv('enc_conv4', 'pool3'),
    pool('pool4', 'enc_conv4'),
    conv('enc_conv5a', 'pool4'),
    conv('enc_conv5b', 'enc_conv5a'),
    upsample('up4', 'enc_conv5b'),
    concat('concat4', 'up4', 'pool3'),
    conv('dec_conv4a', 'concat4'),
    conv('dec_conv4b', 'dec_conv4a'),
    upsample('up3', 'dec_conv4b'),
    concat('concat3', 'up3', 'pool2'),
    conv('dec_conv3a', 'concat3'),
    conv('dec_conv3b', 'dec_conv3a'),
    upsample('up2', 'dec_conv3b'),
    concat('concat2', 'up2', 'pool1'),
    conv('dec_conv2a', 'concat2'),
    conv('dec_conv2b', 'dec_conv2a'),
    upsample('up1', 'dec_conv2b'),
    concat('concat1', 'up1', 'input'),
    conv('dec_conv1a', 'concat1'),
    conv('dec_conv1b', 'dec_conv1a'),
    conv('dec_conv0', 'dec_conv1b')
  ]
};

export const OIDN_UNET_LARGE_SPEC: UNetModelSpec = {
  schemaVersion: 1,
  id: 'oidn-unet-large-v1',
  family: 'oidn-unet-large',
  input: 'input',
  output: 'dec_conv1c',
  receptiveField: 202,
  nodes: [
    conv('enc_conv1a', 'input'),
    conv('enc_conv1b', 'enc_conv1a'),
    pool('pool1', 'enc_conv1b'),
    conv('enc_conv2a', 'pool1'),
    conv('enc_conv2b', 'enc_conv2a'),
    pool('pool2', 'enc_conv2b'),
    conv('enc_conv3a', 'pool2'),
    conv('enc_conv3b', 'enc_conv3a'),
    pool('pool3', 'enc_conv3b'),
    conv('enc_conv4a', 'pool3'),
    conv('enc_conv4b', 'enc_conv4a'),
    pool('pool4', 'enc_conv4b'),
    conv('enc_conv5a', 'pool4'),
    conv('enc_conv5b', 'enc_conv5a'),
    upsample('up4', 'enc_conv5b'),
    concat('concat4', 'up4', 'pool3'),
    conv('dec_conv4a', 'concat4'),
    conv('dec_conv4b', 'dec_conv4a'),
    upsample('up3', 'dec_conv4b'),
    concat('concat3', 'up3', 'pool2'),
    conv('dec_conv3a', 'concat3'),
    conv('dec_conv3b', 'dec_conv3a'),
    upsample('up2', 'dec_conv3b'),
    concat('concat2', 'up2', 'pool1'),
    conv('dec_conv2a', 'concat2'),
    conv('dec_conv2b', 'dec_conv2a'),
    upsample('up1', 'dec_conv2b'),
    concat('concat1', 'up1', 'input'),
    conv('dec_conv1a', 'concat1'),
    conv('dec_conv1b', 'dec_conv1a'),
    conv('dec_conv1c', 'dec_conv1b')
  ]
};

const BUILTIN_MODEL_SPECS = [
  OIDN_UNET_SMALL_SPEC,
  OIDN_UNET_LARGE_SPEC
] as const;

function expectedTensorNames(spec: UNetModelSpec): Set<string> {
  const names = new Set<string>();
  for (const node of spec.nodes) {
    if (node.op === 'conv2d') {
      names.add(node.weight);
      names.add(node.bias);
    }
  }
  return names;
}

function tensorByteSize(tensor: HostTensor): number {
  return tensor.desc.getByteSize();
}

function describeNames(names: Iterable<string>): string {
  return [...names].sort().join(', ');
}

export function detectUNetModelSpec(
  tensors: ReadonlyMap<string, HostTensor>,
  specs: readonly UNetModelSpec[] = BUILTIN_MODEL_SPECS
): UNetModelSpec {
  const matches = specs.filter((spec) => {
    const expected = expectedTensorNames(spec);
    if ([...expected].some((name) => !tensors.has(name))) return false;
    return (
      spec.allowAdditionalTensors === true ||
      [...tensors.keys()].every((name) => expected.has(name))
    );
  });

  if (matches.length === 1) return matches[0];
  if (matches.length > 1) {
    throw new Error(
      `Ambiguous OIDN model topology: ${matches.map((spec) => spec.id).join(', ')}`
    );
  }

  throw new Error(
    `Unsupported OIDN model topology. TZA tensors: ${describeNames(tensors.keys())}`
  );
}

function requireTensor(
  tensors: ReadonlyMap<string, HostTensor>,
  name: string,
  modelId: string
): HostTensor {
  const tensor = tensors.get(name);
  if (!tensor) {
    throw new Error(`Model ${modelId} is missing tensor ${name}`);
  }
  if (tensor.data.byteLength !== tensorByteSize(tensor)) {
    throw new Error(
      `Tensor ${name} has ${tensor.data.byteLength} bytes, expected ${tensorByteSize(tensor)}`
    );
  }
  return tensor;
}

/** Validate tensor layout, shapes, dtypes, graph order, and channel flow. */
export function validateUNetModel(
  tensors: ReadonlyMap<string, HostTensor>,
  spec: UNetModelSpec = detectUNetModelSpec(tensors)
): ValidatedUNetModel {
  if (spec.schemaVersion !== 1) {
    throw new Error(`Unsupported model descriptor schema ${spec.schemaVersion}`);
  }

  const expected = expectedTensorNames(spec);
  if (!spec.allowAdditionalTensors) {
    const additional = [...tensors.keys()].filter((name) => !expected.has(name));
    if (additional.length > 0) {
      throw new Error(
        `Model ${spec.id} has unexpected tensors: ${describeNames(additional)}`
      );
    }
  }

  const channelsByValue = new Map<string, number>();
  const convTensors = new Map<string, ValidatedConvTensor>();
  const convChannels = new Map<string, ModelConvChannels>();
  const producedValues = new Set<string>([spec.input]);
  let inputChannels: number | undefined;
  let commonDataType: HostTensor['desc']['dataType'] | undefined;

  const getChannels = (value: string, nodeId: string) => {
    const channels = channelsByValue.get(value);
    if (channels === undefined) {
      throw new Error(
        `Model ${spec.id} node ${nodeId} reads unknown or forward value ${value}`
      );
    }
    return channels;
  };

  for (const node of spec.nodes) {
    if (producedValues.has(node.id)) {
      throw new Error(`Model ${spec.id} produces duplicate value ${node.id}`);
    }

    if (node.op === 'conv2d') {
      const weight = requireTensor(tensors, node.weight, spec.id);
      const bias = requireTensor(tensors, node.bias, spec.id);
      const dims = weight.desc.dims;

      if (weight.desc.layout !== 'oihw' || dims.length !== 4) {
        throw new Error(`Tensor ${node.weight} must use OIHW layout`);
      }
      if (dims[2] !== 3 || dims[3] !== 3) {
        throw new Error(`Tensor ${node.weight} must use a 3x3 kernel`);
      }
      if (bias.desc.layout !== 'x' || bias.desc.dims.length !== 1) {
        throw new Error(`Tensor ${node.bias} must be a one-dimensional bias`);
      }
      if (bias.desc.dims[0] !== dims[0]) {
        throw new Error(
          `Tensor ${node.bias} has ${bias.desc.dims[0]} channels, expected ${dims[0]}`
        );
      }
      if (weight.desc.dataType !== bias.desc.dataType) {
        throw new Error(`Weight and bias dtype differ for ${node.id}`);
      }
      if (commonDataType && commonDataType !== weight.desc.dataType) {
        throw new Error(`Mixed tensor dtypes are not supported by model ${spec.id}`);
      }
      commonDataType = weight.desc.dataType;

      if (node.input === spec.input && inputChannels === undefined) {
        inputChannels = dims[1];
        channelsByValue.set(spec.input, inputChannels);
      }
      const actualInputChannels = getChannels(node.input, node.id);
      if (actualInputChannels !== dims[1]) {
        throw new Error(
          `Tensor ${node.weight} expects ${dims[1]} input channels, ` +
            `but ${node.input} provides ${actualInputChannels}`
        );
      }

      channelsByValue.set(node.id, dims[0]);
      convTensors.set(node.id, {
        weight,
        bias,
        inputChannels: dims[1],
        outputChannels: dims[0],
        kernelHeight: dims[2],
        kernelWidth: dims[3]
      });
      convChannels.set(node.id, {
        inputChannels: dims[1],
        outputChannels: dims[0]
      });
    } else if (node.op === 'concat') {
      if (node.inputs.length < 2) {
        throw new Error(`Concat ${node.id} requires at least two inputs`);
      }
      const channels = node.inputs.reduce(
        (sum, value) => sum + getChannels(value, node.id),
        0
      );
      channelsByValue.set(node.id, channels);
    } else {
      channelsByValue.set(node.id, getChannels(node.input, node.id));
    }

    producedValues.add(node.id);
  }

  if (inputChannels === undefined || commonDataType === undefined) {
    throw new Error(`Model ${spec.id} has no convolution reading its input`);
  }
  const outputChannels = channelsByValue.get(spec.output);
  if (outputChannels === undefined) {
    throw new Error(`Model ${spec.id} output ${spec.output} is not produced`);
  }
  if (outputChannels !== 3) {
    throw new Error(`Model ${spec.id} must produce 3 channels, got ${outputChannels}`);
  }

  return {
    spec,
    inputChannels,
    outputChannels,
    tensorDataType: commonDataType,
    channelsByValue,
    convChannels,
    convTensors
  };
}
