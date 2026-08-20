import type {
  Conv2DNodeSpec,
  MaxPool2DNodeSpec,
  ModelNodeSpec,
  UNetModelSpec,
  Upsample2DNodeSpec,
  UNetModelGraph
} from './modelSpec';

export interface FusedConvPoolNodeSpec {
  op: 'fusedConvReluMaxPool2d';
  id: string;
  input: string;
  conv: Conv2DNodeSpec;
  pool: MaxPool2DNodeSpec;
}

export interface FusedUpsampleConcatConvNodeSpec {
  op: 'fusedUpsampleConcatConv2d';
  id: string;
  /** Inputs stay in concat order because that order selects weight channels. */
  inputs: readonly {
    value: string;
    upsample?: Upsample2DNodeSpec;
  }[];
  conv: Conv2DNodeSpec;
}

export type ExecutableModelNode =
  | ModelNodeSpec
  | FusedConvPoolNodeSpec
  | FusedUpsampleConcatConvNodeSpec;

export interface GraphOptimizationOptions {
  fuseConvPool?: boolean;
  fuseUpsampleConcatConv?: boolean;
}

export interface OptimizedModelGraph {
  spec: UNetModelSpec;
  nodes: readonly ExecutableModelNode[];
  fusions: {
    convPool: number;
    upsampleConcatConv: number;
  };
}

function nodeInputs(node: ModelNodeSpec): readonly string[] {
  return node.op === 'concat' ? node.inputs : [node.input];
}

function buildConsumers(spec: UNetModelSpec) {
  const consumers = new Map<string, ModelNodeSpec[]>();
  for (const node of spec.nodes) {
    for (const input of nodeInputs(node)) {
      const list = consumers.get(input) ?? [];
      list.push(node);
      consumers.set(input, list);
    }
  }
  return consumers;
}

function onlyConsumer<T extends ModelNodeSpec['op']>(
  consumers: ReadonlyMap<string, ModelNodeSpec[]>,
  value: string,
  op: T
): Extract<ModelNodeSpec, { op: T }> | undefined {
  const list = consumers.get(value);
  if (list?.length !== 1 || list[0].op !== op) return undefined;
  return list[0] as Extract<ModelNodeSpec, { op: T }>;
}

/**
 * Applies topology-only fusions. It never depends on a particular OIDN model
 * name, so new descriptors automatically benefit from known graph patterns.
 */
export function optimizeModelGraph(
  validated: UNetModelGraph,
  options: GraphOptimizationOptions = {}
): OptimizedModelGraph {
  const spec = validated.spec;
  const consumers = buildConsumers(spec);
  const nodesById = new Map(spec.nodes.map((node) => [node.id, node]));
  const eliminated = new Set<string>();
  const fusedAt = new Map<string, ExecutableModelNode>();
  let convPool = 0;
  let upsampleConcatConv = 0;

  if (options.fuseConvPool !== false) {
    for (const node of spec.nodes) {
      if (node.op !== 'conv2d' || node.activation !== 'relu') continue;
      const pool = onlyConsumer(consumers, node.id, 'maxPool2d');
      if (!pool || pool.size !== 2 || pool.stride !== 2) continue;

      eliminated.add(node.id);
      fusedAt.set(pool.id, {
        op: 'fusedConvReluMaxPool2d',
        id: pool.id,
        input: node.input,
        conv: node,
        pool
      });
      convPool++;
    }
  }

  if (options.fuseUpsampleConcatConv !== false) {
    for (const node of spec.nodes) {
      if (node.op !== 'conv2d') continue;
      const concat = nodesById.get(node.input);
      if (concat?.op !== 'concat' || concat.inputs.length !== 2) continue;
      if (onlyConsumer(consumers, concat.id, 'conv2d') !== node) continue;
      // The native blocked layout can remove concat only when the source
      // boundary is also a vec4 boundary. Other graphs keep the generic ops
      // and can use the compatibility engine until a scalar-tail kernel exists.
      const firstInputChannels = validated.channelsByValue.get(concat.inputs[0]);
      if (firstInputChannels === undefined || firstInputChannels % 4 !== 0) {
        continue;
      }

      const inputs = concat.inputs.map((value) => {
        const candidate = nodesById.get(value);
        if (
          candidate?.op === 'upsample2d' &&
          candidate.scale === 2 &&
          candidate.mode === 'nearest' &&
          onlyConsumer(consumers, candidate.id, 'concat') === concat
        ) {
          return { value: candidate.input, upsample: candidate };
        }
        return { value };
      });
      const upsampleCount = inputs.filter((input) => input.upsample).length;
      if (upsampleCount !== 1) continue;

      eliminated.add(concat.id);
      for (const value of concat.inputs) {
        const candidate = nodesById.get(value);
        if (candidate?.op === 'upsample2d') eliminated.add(candidate.id);
      }
      fusedAt.set(node.id, {
        op: 'fusedUpsampleConcatConv2d',
        id: node.id,
        inputs,
        conv: node
      });
      upsampleConcatConv++;
    }
  }

  const nodes: ExecutableModelNode[] = [];
  for (const node of spec.nodes) {
    const fused = fusedAt.get(node.id);
    if (fused) {
      nodes.push(fused);
    } else if (!eliminated.has(node.id)) {
      nodes.push(node);
    }
  }

  return {
    spec,
    nodes,
    fusions: { convPool, upsampleConcatConv }
  };
}

export interface ModelValueShape {
  width: number;
  height: number;
  channels: number;
}

export interface PlannedModelNode {
  node: ExecutableModelNode;
  outputShape: ModelValueShape;
  /** Last planned node that reads the output; output itself uses nodes.length. */
  lastUse: number;
}

export interface ModelExecutionPlan extends OptimizedModelGraph {
  inputShape: ModelValueShape;
  valueShapes: ReadonlyMap<string, ModelValueShape>;
  plannedNodes: readonly PlannedModelNode[];
}

function executableInputs(node: ExecutableModelNode): readonly string[] {
  if (node.op === 'concat') return node.inputs;
  if (node.op === 'fusedUpsampleConcatConv2d') {
    return node.inputs.map((input) => input.value);
  }
  return [node.input];
}

function sameSpatialShape(
  left: ModelValueShape,
  right: ModelValueShape
): boolean {
  return left.width === right.width && left.height === right.height;
}

/** Resolve all runtime shapes and value lifetimes before allocating GPU data. */
export function planModelExecution(
  validated: UNetModelGraph,
  width: number,
  height: number,
  options?: GraphOptimizationOptions
): ModelExecutionPlan {
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new Error(`Invalid model input size ${width}x${height}`);
  }

  const graph = optimizeModelGraph(validated, options);
  const inputShape = { width, height, channels: validated.inputChannels };
  const valueShapes = new Map<string, ModelValueShape>([
    [validated.spec.input, inputShape]
  ]);
  const outputShapes: ModelValueShape[] = [];

  const shapeOf = (value: string, nodeId: string) => {
    const shape = valueShapes.get(value);
    if (!shape) throw new Error(`Planned node ${nodeId} reads missing value ${value}`);
    return shape;
  };

  for (const node of graph.nodes) {
    let outputShape: ModelValueShape;
    if (node.op === 'conv2d') {
      const input = shapeOf(node.input, node.id);
      outputShape = {
        width: input.width,
        height: input.height,
        channels: validated.convChannels.get(node.id)!.outputChannels
      };
    } else if (node.op === 'maxPool2d') {
      const input = shapeOf(node.input, node.id);
      outputShape = {
        width: Math.ceil(input.width / 2),
        height: Math.ceil(input.height / 2),
        channels: input.channels
      };
    } else if (node.op === 'upsample2d') {
      const input = shapeOf(node.input, node.id);
      outputShape = {
        width: input.width * 2,
        height: input.height * 2,
        channels: input.channels
      };
    } else if (node.op === 'concat') {
      const inputs = node.inputs.map((value) => shapeOf(value, node.id));
      if (inputs.some((shape) => !sameSpatialShape(shape, inputs[0]))) {
        throw new Error(`Concat ${node.id} has mismatched spatial shapes`);
      }
      outputShape = {
        width: inputs[0].width,
        height: inputs[0].height,
        channels: inputs.reduce((sum, shape) => sum + shape.channels, 0)
      };
    } else if (node.op === 'fusedConvReluMaxPool2d') {
      const input = shapeOf(node.input, node.id);
      outputShape = {
        width: Math.ceil(input.width / 2),
        height: Math.ceil(input.height / 2),
        channels: validated.convChannels.get(node.conv.id)!.outputChannels
      };
    } else {
      const inputs = node.inputs.map((input) => {
        const source = shapeOf(input.value, node.id);
        return input.upsample
          ? { ...source, width: source.width * 2, height: source.height * 2 }
          : source;
      });
      if (inputs.some((shape) => !sameSpatialShape(shape, inputs[0]))) {
        throw new Error(`Fused decoder ${node.id} has mismatched spatial shapes`);
      }
      outputShape = {
        width: inputs[0].width,
        height: inputs[0].height,
        channels: validated.convChannels.get(node.conv.id)!.outputChannels
      };
    }

    valueShapes.set(node.id, outputShape);
    outputShapes.push(outputShape);
  }

  const lastUses = new Map<string, number>();
  graph.nodes.forEach((node, index) => {
    for (const input of executableInputs(node)) lastUses.set(input, index);
  });
  lastUses.set(validated.spec.output, graph.nodes.length);

  const plannedNodes = graph.nodes.map((node, index) => ({
    node,
    outputShape: outputShapes[index],
    lastUse: lastUses.get(node.id) ?? index
  }));

  return { ...graph, inputShape, valueShapes, plannedNodes };
}
