#!/usr/bin/env node
import { createServer } from 'node:http';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function options(argv) {
  const result = { cycles: 6, engine: 'wgsl' };
  for (let index = 0; index < argv.length; index += 2) {
    const name = argv[index];
    const value = argv[index + 1];
    if (name === '--cycles') result.cycles = Number(value);
    else if (name === '--engine') result.engine = value;
    else if (name === '--chrome') result.chrome = value;
    else throw new Error(`Unknown option ${name}`);
  }
  if (!Number.isInteger(result.cycles) || result.cycles < 2) {
    throw new Error('--cycles must be an integer of at least 2');
  }
  if (!['wgsl', 'webnn'].includes(result.engine)) {
    throw new Error('--engine must be wgsl or webnn');
  }
  return result;
}

function defaultChrome() {
  return [
    process.env.CHROME_PATH,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium'
  ].find((candidate) => candidate && existsSync(candidate));
}

async function server() {
  const files = new Map([
    ['/oidn.js', path.join(root, 'dist/oidn.js')],
    ['/model.tza', path.join(root, 'weights/rt_hdr.tza')]
  ]);
  const instance = createServer(async (request, response) => {
    const pathname = new URL(request.url, 'http://127.0.0.1').pathname;
    if (pathname === '/') {
      response.writeHead(200, { 'content-type': 'text/html' });
      response.end(
        '<!doctype html><meta charset="utf-8"><title>OIDN resource stress</title>'
      );
      return;
    }
    if (pathname === '/favicon.ico') {
      response.writeHead(204).end();
      return;
    }
    const file = files.get(pathname);
    if (!file) {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, {
      'content-type': pathname.endsWith('.js')
        ? 'text/javascript'
        : 'application/octet-stream',
      'cache-control': 'no-store'
    });
    response.end(await readFile(file));
  });
  await new Promise((resolve, reject) => {
    instance.once('error', reject);
    instance.listen(0, '127.0.0.1', resolve);
  });
  return {
    origin: `http://127.0.0.1:${instance.address().port}`,
    close: () =>
      new Promise((resolve, reject) =>
        instance.close((error) => (error ? reject(error) : resolve()))
      )
  };
}

async function main() {
  const config = options(process.argv.slice(2));
  const chrome = config.chrome || defaultChrome();
  if (!chrome) throw new Error('Chrome not found; pass --chrome <path>');

  const { spawn } = await import('node:child_process');
  await new Promise((resolve, reject) => {
    const child = spawn('npm', ['run', 'build'], {
      cwd: root,
      stdio: 'inherit'
    });
    child.once('error', reject);
    child.once('exit', (code) =>
      code === 0
        ? resolve()
        : reject(new Error(`OIDN build exited with ${code}`))
    );
  });

  const host = await server();
  let browser;
  try {
    browser = await chromium.launch({
      executablePath: chrome,
      headless: true,
      args: [
        '--enable-unsafe-webgpu',
        '--enable-precise-memory-info',
        '--js-flags=--expose-gc',
        '--enable-features=Vulkan,UseSkiaRenderer,WebMachineLearningNeuralNetwork'
      ]
    });
    const page = await browser.newPage();
    const browserErrors = [];
    page.on('pageerror', (error) => browserErrors.push(error.message));
    page.on('console', (message) => {
      if (message.type() === 'error') browserErrors.push(message.text());
    });
    await page.goto(host.origin);
    const result = await page.evaluate(
      async ({ cycles, engine, origin }) => {
        if (!navigator.gpu) throw new Error('WebGPU is unavailable');
        if (engine === 'webnn' && !navigator.ml) {
          throw new Error('WebNN is unavailable in this Chrome configuration');
        }
        const module = await import(`${origin}/oidn.js`);
        const model = await fetch(`${origin}/model.tza`).then((response) =>
          response.arrayBuffer()
        );
        const adapter = await navigator.gpu.requestAdapter({
          powerPreference: 'high-performance'
        });
        if (!adapter) throw new Error('No WebGPU adapter');
        const requiredFeatures = adapter.features.has('shader-f16')
          ? ['shader-f16']
          : [];
        const device = await adapter.requestDevice({ requiredFeatures });
        const adapterInfo = adapter.info ?? {};
        const shapes = [
          { width: 32, height: 32 },
          { width: 48, height: 48 },
          { width: 80, height: 64 }
        ];
        const heap = [];
        const resources = [];

        for (let cycle = 0; cycle < cycles; cycle++) {
          device.pushErrorScope('validation');
          const unet = await module.initUNetFromBuffer(
            model.slice(0),
            { device },
            {
              aux: false,
              hdr: true,
              engine,
              precision: requiredFeatures.length ? 'fp16' : 'fp32',
              maxTileSize: 128,
              dynamicTile: false
            }
          );
          for (const shape of shapes) {
            const values = new Float32Array(shape.width * shape.height * 4);
            values.fill(0.25 + cycle * 0.001);
            const input = device.createBuffer({
              size: values.byteLength,
              usage:
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_SRC |
                GPUBufferUsage.COPY_DST,
              mappedAtCreation: true
            });
            new Float32Array(input.getMappedRange()).set(values);
            input.unmap();
            await new Promise((resolve, reject) => {
              try {
                unet.tileExecute({
                  color: { data: input, ...shape },
                  done: resolve
                });
              } catch (error) {
                reject(error);
              }
            });
            await device.queue.onSubmittedWorkDone();
            input.destroy();
          }
          const beforeDispose = unet.getRuntimeInfo().resources;
          unet.dispose();
          await device.queue.onSubmittedWorkDone();
          const afterDispose = unet.getRuntimeInfo().resources;
          const validationError = await device.popErrorScope();
          if (validationError) throw new Error(validationError.message);
          if (
            afterDispose.live !== 0 ||
            afterDispose.pending !== 0 ||
            afterDispose.created !== afterDispose.destroyed
          ) {
            throw new Error(
              `cycle ${cycle}: resources did not return to zero: ` +
                JSON.stringify(afterDispose)
            );
          }
          resources.push({ beforeDispose, afterDispose });
          globalThis.gc?.();
          heap.push(performance.memory?.usedJSHeapSize ?? null);
        }
        device.destroy();
        return { heap, resources, adapter: adapterInfo };
      },
      { ...config, origin: host.origin }
    );

    if (browserErrors.length) {
      throw new Error(`Browser errors:\n${browserErrors.join('\n')}`);
    }
    const measuredHeap = result.heap.filter(Number.isFinite).slice(2);
    const retainedHeapBytes =
      measuredHeap.length > 1
        ? measuredHeap.at(-1) - Math.min(...measuredHeap)
        : null;
    if (retainedHeapBytes !== null && retainedHeapBytes > 32 * 1024 * 1024) {
      throw new Error(
        `JS heap retained ${(retainedHeapBytes / 1024 / 1024).toFixed(1)} MiB`
      );
    }
    console.log(
      JSON.stringify(
        {
          status: 'passed',
          engine: config.engine,
          cycles: config.cycles,
          retainedHeapBytes,
          peakOwnedResources: Math.max(
            ...result.resources.map((entry) => entry.beforeDispose.peakLive)
          ),
          adapter: result.adapter
        },
        null,
        2
      )
    );
  } finally {
    await browser?.close();
    await host.close();
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack : error);
  process.exitCode = 1;
});
