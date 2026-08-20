#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const benchmarkDirectory = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(benchmarkDirectory, '..');

function printHelp() {
  console.log(`Usage: npm run benchmark -- [options]

Options:
  --width <n>          Input width (default: 512)
  --height <n>         Input height (default: 512)
  --tile-size <n>      Fixed output tile edge (default: 512)
  --warmup <n>         Warmup executions per runtime (default: 1)
  --runs <n>           Measured executions per runtime (default: 5)
  --baseline <commit>  TFJS commit (default: nearest TFJS ancestor)
  --chrome <path>      Chrome/Chromium executable
  --output <path>      JSON output (default: benchmarks/results/latest.json)
  --help               Show this message
`);
}

function parseArgs(argv) {
  const options = {
    width: 512,
    height: 512,
    tileSize: 512,
    warmup: 1,
    runs: 5,
    output: path.join(benchmarkDirectory, 'results/latest.json')
  };
  const names = {
    width: 'width',
    height: 'height',
    'tile-size': 'tileSize',
    warmup: 'warmup',
    runs: 'runs',
    baseline: 'baseline',
    chrome: 'chrome',
    output: 'output'
  };
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index];
    if (argument === '--help' || argument === '-h') {
      options.help = true;
      continue;
    }
    if (!argument.startsWith('--') || !names[argument.slice(2)]) {
      throw new Error(`Unknown option ${argument}`);
    }
    const key = argument.slice(2);
    const value = argv[++index];
    if (value === undefined || value.startsWith('--')) {
      throw new Error(`Missing value for ${argument}`);
    }
    options[names[key]] = ['width', 'height', 'tile-size', 'warmup', 'runs'].includes(key)
      ? Number(value)
      : value;
  }
  for (const key of ['width', 'height', 'tileSize', 'runs']) {
    if (!Number.isInteger(options[key]) || options[key] <= 0) {
      throw new Error(`${key} must be a positive integer`);
    }
  }
  if (!Number.isInteger(options.warmup) || options.warmup < 0) {
    throw new Error('warmup must be a non-negative integer');
  }
  options.output = path.resolve(projectRoot, options.output);
  return options;
}

function defaultChrome() {
  return [
    process.env.CHROME_PATH,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser'
  ].find((candidate) => candidate && existsSync(candidate));
}

function run(command, args, { cwd = projectRoot, quiet = false } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      stdio: quiet ? ['ignore', 'pipe', 'pipe'] : 'inherit'
    });
    let stdout = '';
    let stderr = '';
    if (quiet) {
      child.stdout.on('data', (data) => { stdout += data; });
      child.stderr.on('data', (data) => { stderr += data; });
    }
    child.on('error', reject);
    child.on('exit', (code, signal) => {
      if (code === 0) resolve({ stdout, stderr });
      else reject(new Error(
        `${command} exited with ${code ?? signal}${stderr ? `\n${stderr.slice(-4000)}` : ''}`
      ));
    });
  });
}

async function gitText(args) {
  return (await run('git', args, { quiet: true })).stdout.trim();
}

async function findTFJSBaseline(explicitCommit) {
  if (explicitCommit) {
    await gitText(['rev-parse', '--verify', `${explicitCommit}^{commit}`]);
    return gitText(['rev-parse', explicitCommit]);
  }
  const commits = (await gitText(['rev-list', 'HEAD', '--', 'package.json']))
    .split('\n')
    .filter(Boolean);
  for (const commit of commits) {
    const packageJson = JSON.parse(await gitText(['show', `${commit}:package.json`]));
    const dependencies = {
      ...packageJson.dependencies,
      ...packageJson.devDependencies
    };
    if (Object.keys(dependencies).some((name) => name.startsWith('@tensorflow/tfjs'))) {
      return commit;
    }
  }
  throw new Error('Could not find a TensorFlow.js ancestor; pass --baseline <commit>');
}

function contentType(filePath) {
  if (filePath.endsWith('.js')) return 'text/javascript; charset=utf-8';
  if (filePath.endsWith('.tza')) return 'application/octet-stream';
  return 'text/html; charset=utf-8';
}

async function startServer(files) {
  const server = createServer(async (request, response) => {
    const pathname = new URL(request.url, 'http://127.0.0.1').pathname;
    if (pathname === '/') {
      response.writeHead(200, { 'content-type': contentType('.html') });
      response.end('<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,"><title>OIDN benchmark</title>');
      return;
    }
    if (pathname === '/favicon.ico') {
      response.writeHead(204);
      response.end();
      return;
    }
    const filePath = files.get(pathname);
    if (!filePath) {
      response.writeHead(404);
      response.end('Not found');
      return;
    }
    try {
      response.writeHead(200, {
        'content-type': contentType(filePath),
        'cache-control': 'no-store'
      });
      response.end(await readFile(filePath));
    } catch (error) {
      response.writeHead(500);
      response.end(String(error));
    }
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  return {
    origin: `http://127.0.0.1:${server.address().port}`,
    close: () => new Promise((resolve, reject) =>
      server.close((error) => error ? reject(error) : resolve())
    )
  };
}

function summarize(times) {
  const sorted = [...times].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return {
    minMs: sorted[0],
    medianMs: sorted.length % 2
      ? sorted[middle]
      : (sorted[middle - 1] + sorted[middle]) / 2,
    meanMs: sorted.reduce((sum, value) => sum + value, 0) / sorted.length,
    p95Ms: sorted[Math.ceil(sorted.length * 0.95) - 1],
    maxMs: sorted[sorted.length - 1]
  };
}

function compareSamples(reference, candidate) {
  if (!reference || !candidate || reference.length !== candidate.length) {
    return {
      passed: false,
      reason: 'output sample count mismatch'
    };
  }
  let absoluteError = 0;
  let squaredError = 0;
  let maxAbsoluteError = 0;
  for (let index = 0; index < reference.length; index++) {
    const error = Math.abs(reference[index] - candidate[index]);
    if (!Number.isFinite(error)) {
      return { passed: false, reason: `non-finite output at sample ${index}` };
    }
    absoluteError += error;
    squaredError += error * error;
    maxAbsoluteError = Math.max(maxAbsoluteError, error);
  }
  return {
    sampleCount: reference.length,
    meanAbsoluteError: absoluteError / reference.length,
    rootMeanSquaredError: Math.sqrt(squaredError / reference.length),
    maxAbsoluteError
  };
}

async function benchmarkVariant(browser, origin, options, variant) {
  const page = await browser.newPage();
  const messages = [];
  page.on('console', (message) => {
    if (message.type() === 'warning' || message.type() === 'error') {
      messages.push(`${message.type()}: ${message.text()}`);
    }
  });
  page.on('pageerror', (error) => messages.push(`pageerror: ${error.message}`));
  await page.goto(origin, { waitUntil: 'load' });
  try {
    const result = await page.evaluate(async (config) => {
      if (!navigator.gpu) throw new Error('WebGPU is unavailable');
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error('No WebGPU adapter is available');
      const supportsFP16 = adapter.features.has('shader-f16');
      if (config.precision === 'fp16' && !supportsFP16) {
        return { skipped: 'shader-f16 is unavailable' };
      }
      const requiredFeatures = config.precision === 'fp16' ? ['shader-f16'] : [];
      if (adapter.features.has('timestamp-query')) requiredFeatures.push('timestamp-query');
      const device = await adapter.requestDevice({
        requiredFeatures,
        requiredLimits: {
          maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
          maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
          maxBufferSize: adapter.limits.maxBufferSize,
          maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
          maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup
        }
      });
      const adapterInfo = adapter.info ?? await adapter.requestAdapterInfo?.() ?? {};
      const oidn = await import(config.moduleUrl);
      const initStartedAt = performance.now();
      const runtimeOptions = {
        aux: true,
        hdr: true,
        maxTileSize: config.tileSize
      };
      if (!config.baseline) {
        runtimeOptions.engine = 'wgsl';
        runtimeOptions.precision = config.precision;
        runtimeOptions.dynamicTile = false;
      }
      const unet = await oidn.initUNetFromURL(
        config.modelUrl,
        { device, adapterInfo },
        runtimeOptions
      );
      const initializationMs = performance.now() - initStartedAt;

      const pixelCount = config.width * config.height;
      const makePixels = (kind) => {
        const values = new Float32Array(pixelCount * 4);
        for (let index = 0; index < pixelCount; index++) {
          const x = index % config.width;
          const y = Math.floor(index / config.width);
          const offset = index * 4;
          if (kind === 'color') {
            values[offset] = 0.1 + 4 * x / Math.max(1, config.width - 1);
            values[offset + 1] = 0.05 + 2 * y / Math.max(1, config.height - 1);
            values[offset + 2] = 0.2 + ((x * 17 + y * 13) % 97) / 97;
          } else if (kind === 'albedo') {
            values[offset] = 0.2 + 0.7 * x / Math.max(1, config.width - 1);
            values[offset + 1] = 0.3 + 0.6 * y / Math.max(1, config.height - 1);
            values[offset + 2] = 0.55;
          } else {
            values[offset] = 0.5;
            values[offset + 1] = 0.5;
            values[offset + 2] = 1;
          }
          values[offset + 3] = 1;
        }
        return values;
      };
      const createInputBuffer = (kind) => {
        const values = makePixels(kind);
        const buffer = device.createBuffer({
          size: values.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
          mappedAtCreation: true
        });
        new Float32Array(buffer.getMappedRange()).set(values);
        buffer.unmap();
        return buffer;
      };
      const buffers = {
        color: createInputBuffer('color'),
        albedo: createInputBuffer('albedo'),
        normal: createInputBuffer('normal')
      };
      const image = (data) => ({ data, width: config.width, height: config.height });
      let lastOutput;
      const execute = async () => {
        const startedAt = performance.now();
        await new Promise((resolve, reject) => {
          try {
            unet.tileExecute({
              color: image(buffers.color),
              albedo: image(buffers.albedo),
              normal: image(buffers.normal),
              done: (output) => {
                lastOutput = output;
                resolve();
              }
            });
          } catch (error) {
            reject(error);
          }
        });
        await device.queue.onSubmittedWorkDone();
        return performance.now() - startedAt;
      };
      for (let index = 0; index < config.warmup; index++) await execute();
      const timesMs = [];
      for (let index = 0; index < config.runs; index++) timesMs.push(await execute());
      let executionProfile;
      if (unet.profileNextExecution?.()) {
        await execute();
        executionProfile = await unet.getLastExecutionProfile?.();
      }
      if (!lastOutput?.data) throw new Error('OIDN did not return an output buffer');
      const outputReadback = device.createBuffer({
        size: pixelCount * 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
      const outputEncoder = device.createCommandEncoder();
      outputEncoder.copyBufferToBuffer(
        lastOutput.data,
        0,
        outputReadback,
        0,
        pixelCount * 16
      );
      device.queue.submit([outputEncoder.finish()]);
      await outputReadback.mapAsync(GPUMapMode.READ);
      const outputValues = new Float32Array(outputReadback.getMappedRange());
      const outputSamples = [];
      const sampleStride = Math.max(1, Math.floor(pixelCount / 2048));
      for (let pixel = 0; pixel < pixelCount; pixel += sampleStride) {
        const offset = pixel * 4;
        outputSamples.push(
          outputValues[offset],
          outputValues[offset + 1],
          outputValues[offset + 2]
        );
      }
      outputReadback.unmap();
      outputReadback.destroy();
      const runtimeInfo = unet.getRuntimeInfo?.();
      unet.dispose?.();
      Object.values(buffers).forEach((buffer) => buffer.destroy());
      device.destroy();
      return {
        initializationMs,
        timesMs,
        executionProfile,
        outputSamples,
        runtimeInfo,
        supportsFP16,
        adapter: {
          vendor: adapterInfo.vendor ?? '',
          architecture: adapterInfo.architecture ?? '',
          device: adapterInfo.device ?? '',
          description: adapterInfo.description ?? ''
        }
      };
    }, {
      moduleUrl: `${origin}/${variant.bundle}/oidn.js`,
      modelUrl: `${origin}/weights/rt_hdr_calb_cnrm_large.tza`,
      baseline: variant.baseline,
      precision: variant.precision,
      width: options.width,
      height: options.height,
      tileSize: options.tileSize,
      warmup: options.warmup,
      runs: options.runs
    });
    if (result.skipped) return { ...variant, skipped: result.skipped, messages };
    return {
      ...variant,
      ...result,
      summary: summarize(result.timesMs),
      messages
    };
  } finally {
    await page.close();
  }
}

function formatNumber(value) {
  return value == null ? '-' : value.toFixed(2);
}

function formatExponential(value) {
  return Number.isFinite(value) ? value.toExponential(2) : '-';
}

function markdownReport(report) {
  const baselineMedian = report.results.find((result) => result.baseline)?.summary?.medianMs;
  const rows = report.results.map((result) => {
    if (result.skipped) return `| ${result.label} | skipped | - | - | - | ${result.skipped} |`;
    const speedup = baselineMedian / result.summary.medianMs;
    return `| ${result.label} | ${formatNumber(result.initializationMs)} | ${formatNumber(result.summary.medianMs)} | ${formatNumber(result.summary.p95Ms)} | ${speedup.toFixed(2)}x | ${result.runtimeInfo?.precision ?? 'fp32'} |`;
  });
  const validationRows = report.results
    .filter((result) => !result.baseline && !result.skipped)
    .map((result) => {
      const validation = result.validation;
      const status = validation.passed ? 'pass' : 'FAIL';
      return `| ${result.label} | ${status} | ${formatExponential(validation.meanAbsoluteError)} | ${formatExponential(validation.rootMeanSquaredError)} | ${formatExponential(validation.maxAbsoluteError)} | ${validation.sampleCount ?? '-'} |`;
    });
  const profileSections = report.results
    .filter((result) => result.executionProfile)
    .map((result) => {
      const hotLayers = [...result.executionProfile.layers]
        .sort((left, right) => right.durationMs - left.durationMs)
        .slice(0, 5)
        .map((layer) => `| ${layer.id} | ${formatNumber(layer.durationMs)} |`)
        .join('\n');
      return `### ${result.label}\n\n` +
        `Profiled GPU total: ${formatNumber(result.executionProfile.totalMs)} ms\n\n` +
        `| Node | GPU ms |\n| --- | ---: |\n${hotLayers}`;
    });
  return `# oidn-web benchmark\n\n` +
    `- Current: \`${report.currentCommit}\`\n` +
    `- TFJS baseline: \`${report.baselineCommit}\`\n` +
    `- Input: ${report.settings.width}x${report.settings.height}, fixed tile ${report.settings.tileSize}, ${report.settings.runs} runs after ${report.settings.warmup} warmup(s)\n` +
    `- Adapter: ${report.adapter.description || report.adapter.device || report.adapter.vendor || 'unknown'}\n\n` +
    `| Runtime | Init ms | Median ms | P95 ms | Speedup | Precision |\n` +
    `| --- | ---: | ---: | ---: | ---: | --- |\n${rows.join('\n')}\n\n` +
    `## Output validation\n\n` +
    `Compared against sampled TFJS FP32 output.\n\n` +
    `| Runtime | Status | MAE | RMSE | Max error | Samples |\n` +
    `| --- | --- | ---: | ---: | ---: | ---: |\n${validationRows.join('\n')}\n\n` +
    `## GPU hot layers\n\n${profileSections.join('\n\n') || 'Timestamp queries unavailable.'}\n`;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }
  const chrome = options.chrome || defaultChrome();
  if (!chrome) throw new Error('Chrome/Chromium not found; pass --chrome or CHROME_PATH');
  const baselineCommit = await findTFJSBaseline(options.baseline);
  const currentCommit = await gitText(['rev-parse', 'HEAD']);
  console.log(`TFJS baseline: ${baselineCommit.slice(0, 12)}`);
  console.log(`Current runtime: ${currentCommit.slice(0, 12)} + working tree`);

  await run('npm', ['run', 'build']);
  const temporaryRoot = await mkdtemp(path.join(os.tmpdir(), 'oidn-benchmark-'));
  const baselineRoot = path.join(temporaryRoot, 'baseline');
  let worktreeAdded = false;
  let server;
  let browser;
  try {
    // The repository stores model files through Git LFS, but the benchmark
    // serves the current checkout's weights. Disable LFS filters for this
    // source-only historical worktree so git-lfs is not a prerequisite.
    await run('git', [
      '-c', 'filter.lfs.smudge=',
      '-c', 'filter.lfs.process=',
      '-c', 'filter.lfs.required=false',
      '-c', 'core.hooksPath=/dev/null',
      'worktree', 'add', '--detach', baselineRoot, baselineCommit
    ]);
    worktreeAdded = true;
    await run('npm', ['ci', '--ignore-scripts'], { cwd: baselineRoot });
    await run('npm', ['run', 'build'], { cwd: baselineRoot });

    server = await startServer(new Map([
      ['/current/oidn.js', path.join(projectRoot, 'dist/oidn.js')],
      ['/baseline/oidn.js', path.join(baselineRoot, 'dist/oidn.js')],
      ['/weights/rt_hdr_calb_cnrm_large.tza', path.join(projectRoot, 'weights/rt_hdr_calb_cnrm_large.tza')]
    ]));
    browser = await chromium.launch({
      executablePath: chrome,
      headless: true,
      args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,UseSkiaRenderer']
    });
    const variants = [
      { label: `TFJS (${baselineCommit.slice(0, 7)})`, bundle: 'baseline', baseline: true },
      { label: 'WGSL FP32', bundle: 'current', baseline: false, precision: 'fp32' },
      { label: 'WGSL FP16', bundle: 'current', baseline: false, precision: 'fp16' }
    ];
    const results = [];
    for (const variant of variants) {
      console.log(`Benchmarking ${variant.label}...`);
      results.push(await benchmarkVariant(browser, server.origin, options, variant));
    }
    const referenceSamples = results.find((result) => result.baseline)?.outputSamples;
    for (const result of results) {
      if (!result.baseline && !result.skipped) {
        const comparison = compareSamples(referenceSamples, result.outputSamples);
        const maxMeanError = result.precision === 'fp16' ? 5e-3 : 1e-4;
        const maxAbsoluteError = result.precision === 'fp16' ? 5e-2 : 1e-3;
        const hasComparableOutput =
          Number.isFinite(comparison.meanAbsoluteError) &&
          Number.isFinite(comparison.maxAbsoluteError);
        result.validation = {
          ...comparison,
          passed:
            hasComparableOutput &&
            comparison.meanAbsoluteError <= maxMeanError &&
            comparison.maxAbsoluteError <= maxAbsoluteError,
          thresholds: { maxMeanError, maxAbsoluteError }
        };
      }
      delete result.outputSamples;
    }
    const firstCompleted = results.find((result) => !result.skipped);
    const report = {
      generatedAt: new Date().toISOString(),
      currentCommit,
      baselineCommit,
      settings: {
        width: options.width,
        height: options.height,
        tileSize: options.tileSize,
        warmup: options.warmup,
        runs: options.runs,
        model: 'rt_hdr_calb_cnrm_large.tza',
        gpuQueueCompletionIncluded: true
      },
      browser: chrome,
      adapter: firstCompleted?.adapter ?? {},
      results
    };
    const markdown = markdownReport(report);
    await mkdir(path.dirname(options.output), { recursive: true });
    await writeFile(options.output, JSON.stringify(report, null, 2));
    const markdownPath = options.output.endsWith('.json')
      ? options.output.slice(0, -5) + '.md'
      : options.output + '.md';
    await writeFile(markdownPath, markdown);
    console.log(`\n${markdown}`);
    console.log(`JSON: ${options.output}`);
    console.log(`Markdown: ${markdownPath}`);
    const failedValidation = results.find(
      (result) => result.validation && !result.validation.passed
    );
    if (failedValidation) {
      throw new Error(`${failedValidation.label} output validation failed`);
    }
  } finally {
    await browser?.close();
    await server?.close();
    if (worktreeAdded) {
      await run('git', ['worktree', 'remove', '--force', baselineRoot], { quiet: true });
    }
    await rm(temporaryRoot, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack : error);
  process.exitCode = 1;
});
