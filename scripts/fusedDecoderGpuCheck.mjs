import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { chromium } from 'playwright-core';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const mime = new Map([
  ['.js', 'text/javascript'],
  ['.mjs', 'text/javascript'],
  ['.json', 'application/json'],
  ['.css', 'text/css'],
  ['.html', 'text/html']
]);

const server = http.createServer((request, response) => {
  const requestPath = decodeURIComponent((request.url ?? '/').split('?')[0]);
  if (requestPath === '/') {
    response.writeHead(200, { 'Content-Type': 'text/html', 'Cache-Control': 'no-store' });
    response.end(`<!doctype html><script type="importmap">${JSON.stringify({
      imports: { '@petamoriken/float16': '/node_modules/@petamoriken/float16/src/index.mjs' }
    })}</script><title>OIDN fused decoder GPU check</title>`);
    return;
  }
  const file = path.resolve(root, `.${requestPath}`);
  if (!file.startsWith(root + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
    response.writeHead(404);
    response.end();
    return;
  }
  response.writeHead(200, {
    'Content-Type': mime.get(path.extname(file)) ?? 'application/octet-stream',
    'Cache-Control': 'no-store'
  });
  fs.createReadStream(file).pipe(response);
});

await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const { port } = server.address();
const browser = await chromium.launch({
  headless: true,
  executablePath: process.env.CHROME_PATH ?? 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  args: [
    '--enable-unsafe-webgpu',
    '--no-sandbox',
    '--disable-gpu-sandbox',
    '--disable-features=RendererCodeIntegrity',
    '--disable-direct-composition',
    '--do-not-de-elevate',
    '--force_high_performance_gpu'
  ]
});

try {
  const page = await browser.newPage();
  const result = await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load' });
  if (!result?.ok()) throw new Error(`HTTP server failed: ${result?.status()}`);
  const check = await page.evaluate(async () => {
    const [{ NativeUNetExecutor }, { validateUNetModel }, { HostTensor, TensorDesc }] =
      await Promise.all([
        import('/lib/nativeUNet.js'),
        import('/lib/modelSpec.js'),
        import('/lib/tza.js')
      ]);
    if (!navigator.gpu) throw new Error('navigator.gpu unavailable');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter');
    const device = await adapter.requestDevice({ requiredFeatures: ['shader-f16'] });
    const tensor = (dims, layout, values) => {
      const desc = new TensorDesc();
      desc.dims = [...dims];
      desc.paddedDims = [...dims];
      desc.layout = layout;
      desc.dataType = 'Float32';
      const typed = new Float32Array(values);
      return new HostTensor(
        desc,
        new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength)
      );
    };
    const centerWeights = (outputChannels, inputChannels, values) => {
      const weights = new Float32Array(outputChannels * inputChannels * 9);
      for (const [output, input, value] of values) {
        weights[((output * inputChannels + input) * 3 + 1) * 3 + 1] = value;
      }
      return weights;
    };
    const spec = {
      schemaVersion: 1,
      id: 'fused-final-gpu-check',
      family: 'fused-final-gpu-check',
      input: 'input',
      output: 'output',
      receptiveField: 7,
      nodes: [
        { op: 'conv2d', id: 'branch', input: 'input', weight: 'branch.weight', bias: 'branch.bias', activation: 'relu', padding: 'same' },
        { op: 'maxPool2d', id: 'pool', input: 'branch', size: 2, stride: 2, padding: 'same' },
        { op: 'upsample2d', id: 'up', input: 'pool', scale: 2, mode: 'nearest' },
        { op: 'conv2d', id: 'skip', input: 'input', weight: 'skip.weight', bias: 'skip.bias', activation: 'relu', padding: 'same' },
        { op: 'concat', id: 'join', inputs: ['up', 'skip'], axis: 'channels' },
        { op: 'conv2d', id: 'output', input: 'join', weight: 'output.weight', bias: 'output.bias', activation: 'identity', padding: 'same' }
      ]
    };
    const tensors = new Map([
      ['branch.weight', tensor([4, 3, 3, 3], 'oihw', centerWeights(4, 3, []))],
      ['branch.bias', tensor([4], 'x', [1, 2, 3, 4])],
      ['skip.weight', tensor([4, 3, 3, 3], 'oihw', centerWeights(4, 3, []))],
      ['skip.bias', tensor([4], 'x', [5, 6, 7, 8])],
      ['output.weight', tensor([3, 8, 3, 3], 'oihw', centerWeights(3, 8, [[0, 0, 1], [0, 4, 2], [1, 1, 1], [1, 5, 2], [2, 2, 1], [2, 6, 2]]))],
      ['output.bias', tensor([3], 'x', [0.5, 1.5, 2.5])]
    ]);
    const model = validateUNetModel(tensors, spec);
    const executor = new NativeUNetExecutor(device, model, {
      precision: 'fp16',
      kernel: 'implicit-gemm',
      gemm: { addressMode: 'incremental', weightLayout: 'k-major', rowsPerThread: 8, workgroupSize: [8, 8] }
    });
    await executor.prepare();
    const input = new Float32Array(4 * 4 * 3).fill(0.25);
    const output = await executor.executeCPU(input, 4, 4);
    const expected = [11.5, 15.5, 19.5];
    let maxError = 0;
    if (!output.every(Number.isFinite)) throw new Error('Nonfinite output');
    for (let pixel = 0; pixel < 16; pixel++) {
      for (let channel = 0; channel < 3; channel++) {
        maxError = Math.max(maxError, Math.abs(output[pixel * 3 + channel] - expected[channel]));
      }
    }
    executor.dispose();
    device.destroy?.();
    return { maxError, expected, output: Array.from(output.slice(0, 3)) };
  });
  if (check.maxError > 1e-3) {
    throw new Error(`Fused final decoder mismatch: max error ${check.maxError}; first output ${check.output}`);
  } else {
    console.log(`PASS: fused final decoder max error ${check.maxError}`);
  }
} finally {
  await browser.close();
  await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
}
