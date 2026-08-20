import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

import { parseTZA } from '../lib/tza.js';
import {
  detectUNetModelSpec,
  validateUNetModel
} from '../lib/modelSpec.js';

const modelPath = process.argv[2];
if (!modelPath) {
  console.error('Usage: npm run model:inspect -- path/to/model.tza');
  process.exit(2);
}

const absolutePath = resolve(modelPath);
const file = await readFile(absolutePath);
const arrayBuffer = file.buffer.slice(
  file.byteOffset,
  file.byteOffset + file.byteLength
);
const tensors = parseTZA(arrayBuffer);
const tensorSignature = [...tensors]
  .map(([name, tensor]) => ({
    name,
    dims: tensor.desc.dims,
    layout: tensor.desc.layout,
    dataType: tensor.desc.dataType,
    bytes: tensor.data.byteLength
  }))
  .sort((left, right) => left.name.localeCompare(right.name));

let model;
let compatibilityError;
try {
  const spec = detectUNetModelSpec(tensors);
  const validated = validateUNetModel(tensors, spec);
  model = {
    descriptor: spec.id,
    family: spec.family,
    inputChannels: validated.inputChannels,
    outputChannels: validated.outputChannels,
    dataType: validated.tensorDataType,
    receptiveField: spec.receptiveField
  };
} catch (error) {
  compatibilityError = error instanceof Error ? error.message : String(error);
  process.exitCode = 1;
}

console.log(
  JSON.stringify(
    {
      path: absolutePath,
      sha256: createHash('sha256').update(file).digest('hex'),
      model,
      compatibilityError,
      tensors: tensorSignature
    },
    null,
    2
  )
);
