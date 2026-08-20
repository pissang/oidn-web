# Open Image Denoise on the Web

This library brings the state-of-the-art AI-based denoising library [Open Image Denoise](https://github.com/RenderKit/oidn) to the web.
Currently it's only available on the browsers support WebGPU.

It's used in the [Vector to 3D](https://www.figma.com/community/plugin/1264600219316901594/) Figma plugin for high quality rendering and denoising.

|                                           2000 Samples                                           |                                     3 Samples                                      |                                   3 Samples + Denoised                                   |
| :----------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: |
| ![](https://github.com/pissang/oidn-web/blob/main/examples/test/ground-truth.png 'Ground Truth') | ![](https://github.com/pissang/oidn-web/blob/main/examples/test/noisy.png 'Noisy') | ![](https://github.com/pissang/oidn-web/blob/main/examples/test/denoised.png 'Denoised') |

## How it works

The OIDN U-Net runs directly on WebGPU with model-driven WGSL compute
pipelines. TensorFlow.js is not used. Convolution activations use a blocked
four-channel layout, encoder `conv + ReLU + max-pool` and decoder
`upsample + concat + conv` patterns are fused, and all network dispatches for a
tile are submitted in one command buffer.

TZA half-float weights stay half-float when the device enables `shader-f16`.
Convolution accumulates in FP32 and the final output is FP32. Devices without
`shader-f16` automatically use the native FP32 path.

## How to Use

[Basic Example](https://oidn-web-example.vercel.app/) ([Code](https://github.com/pissang/oidn-web-example/blob/main/src/main.js))

[Use with three-gpu-pathtracer](https://oidn-web-example.vercel.app/three-gpu-pathtracer.html) ([Code](https://github.com/pissang/oidn-web-example/blob/main/src/three-gpu-pathtracer.js))

### Install

```shell
npm i oidn-web
```

The TZA weights files are not included in the package. You can find them in this repo or [oidn-weights](https://github.com/RenderKit/oidn-weights).

### Denoise a noisy LDR image

```ts
import { UNet, initUNetFromURL } from 'oidn-web';
initUNetFromURL('./weights/rt_ldr.tza').then((unet) => {
  // Read the image data.
  const noisyImageData = noisyCanvas
    .getContext('2d')
    .getImageData(0, 0, width, height);

  // Tile execute the denoising.
  // If the resolution is high. It will split the input into tiles and execute one tile per frame.
  const abortDenoising = unet.tileExecute({
    // The color input for LDR image is 4 channels.
    // In the format of Uint8ClampedArray or Uint8Array.
    color: noisyImageData,
    done(denoised) {
      console.log('Finished');
    },
    progress(denoised, tileData, tile) {
      // Put the denoised tile on the output canvas
      outputCtx.putImageData(tileData, tile.x, tile.y);
    }
  });
});
```

### Denoise a noisy HDR image

```ts
import { UNet, initUNetFromURL } from 'oidn-web';
initUNetFromURL('./weights/rt_hdr.tza', undefined, {
  // It's hdr input.
  hdr: true
}).then((unet) => {
  const abortDenoising = unet.tileExecute({
    // The color input for HDR image is 4 channels.
    // In the format of Float32Array.
    color: { data: noisyColor, width, height },
    done(denoised) {
      console.log('Finished');
    },
    progress(denoised, tileData, tile) {
      // The denoised data and tileData has same format with the input.
    }
  });
});
```

### Use auxiliary images

```ts
import { UNet, initUNetFromURL } from 'oidn-web';
initUNetFromURL('./weights/rt_hdr_alb_nrm.tza', undefined, {
  aux: true,
  hdr: true
}).then((unet) => {
  const abortDenoising = unet.tileExecute({
    // Same as examples before. noisyColor of HDR image is Float32Array. LDR image is Uint8ClampedArray.
    color: { data: noisyColor, width, height },
    // Normal and albedo are both 4 channels in Uint8ClampedArray.
    normal: { data: normalData, width, height },
    albedo: { data: albedoData, width, height },

    done(denoised) {
      console.log('Finished');
    },
    progress(denoised, tileData, tile) {
      ///...
    }
  });
});
```

### Integrate into your WebGPU Pipeline

If you already have a WebGPU path tracer. You can integrate the oidn-web into your pipeline. It supports input/output gpu buffers to avoid the cost of syncing between CPU and GPU.

`hdr` and `aux` are required in the WebGPU pipeline.

```ts
initUNetFromURL(
  './weights/rt_hdr_alb_nrm.tza',
  {
    // Share GPUDevice and GPUAdapterInfo with the native WGSL runtime.
    device,
    adapterInfo
  },
  {
    aux: true,
    hdr: true
  }
).then((unet) => {
  const abortDenoising = unet.tileExecute({
    // Inputs are all GPUBuffer
    color: { data: colorBuffer, width, height },
    normal: { data: normalBuffer, width, height },
    albedo: { data: albedoBuffer, width, height },

    done(denoised) {
      console.log('Finished');
    },
    progress(denoised) {
      // Denoised data is also a GPUBuffer.
      // tileData is undefined if using GPUBuffer as input/output
    }
  });
});
```

### Use smaller and larger weights.

OIDN also provides a large weights file, which provides a better quality, and a small weights file, which provides a better performance.

```ts
// Change the weights file to large and nothing else needs to do.
initUNetFromURL('./weights/rt_hdr_calb_cnrm_large.tza', ...);
```

```ts
// Change the weights file to small and nothing else needs to do.
initUNetFromURL('./weights/rt_hdr_alb_nrm_small.tza', ...);
```

Other combinations can be found in the [oidn-weights](https://github.com/RenderKit/oidn-weights)

### FP16 and runtime information

Standalone initialization requests `shader-f16` when the adapter supports it.
When sharing a device, optional features must be requested when that device is
created; WebGPU features cannot be enabled afterward.

```ts
const requiredFeatures = adapter.features.has('shader-f16')
  ? ['shader-f16']
  : [];
const device = await adapter.requestDevice({ requiredFeatures });

const unet = await initUNetFromURL(modelUrl, { device, adapterInfo }, {
  aux: true,
  hdr: true,
  precision: 'auto' // 'fp16' enforces support; 'fp32' is deterministic fallback
});

console.log(unet.getRuntimeInfo());
// { gpuEngine: 'wgsl', precision: 'fp16', model: 'oidn-unet-large-v1', ... }
```

### Updating to a new OIDN model

TZA stores tensors but not the executable graph. The runtime therefore keeps
the graph in a versioned `UNetModelSpec`, separate from shader and precision
code. Built-in descriptors cover the current OIDN small and large RT U-Nets.
At load time the descriptor is detected from the complete tensor-name set, and
tensor layout, dtype, byte length, kernel shape, bias shape, and graph channel
flow are validated before GPU resources are created.

If an OIDN update keeps one of these topologies and tensor names, changed
channel widths are handled automatically. If it adds or renames nodes, add a
new descriptor (or pass `modelSpec`) and its validation fixture. Existing graph
fusion rules apply to the new descriptor without changes to WGSL kernels.

Use the inspection command to get a stable SHA-256, full tensor signature, and
descriptor compatibility result for an upstream weight file:

```shell
npm run model:inspect -- weights/rt_hdr_alb_nrm.tza
```

```ts
const unet = await initUNetFromURL(newModelUrl, backend, {
  aux: true,
  hdr: true,
  modelSpec: newOidnModelSpec
});
```

### GPU backpressure and dynamic tiles

`tileExecute` waits for the submitted GPU work of a tile before scheduling the
next tile. This keeps at most one OIDN tile in flight, which makes cancellation
responsive instead of leaving queued denoising work ahead of interactive
rendering.

Tile sizing is adaptive by default. `maxTileSize` is a hard upper bound; the
completed GPU time of a tiled execution adjusts the tile size used by the next
execution. Single-tile images do not affect the estimate. The default range
starts at 384 pixels, does not go below 256, and targets about 16 ms of GPU work
per tile.

```ts
initUNetFromURL('./weights/rt_hdr_alb_nrm.tza', backend, {
  aux: true,
  hdr: true,
  maxTileSize: 512,
  dynamicTile: {
    minTileSize: 256,
    initialTileSize: 384,
    targetTileTimeMs: 16
  }
});

// Restore fixed-size behavior when deterministic tiling is preferred.
initUNetFromURL('./weights/rt_hdr_alb_nrm.tza', backend, {
  aux: true,
  hdr: true,
  maxTileSize: 512,
  dynamicTile: false
});
```

### Benchmark the native runtime against TFJS

The browser benchmark automatically finds the nearest ancestor whose package
still depends on TensorFlow.js, builds that commit in a temporary worktree, and
compares it with the current WGSL FP32 and FP16 runtimes. Each measured run
waits for the WebGPU queue to finish, so the result includes execution rather
than only JavaScript command submission.

```shell
npm run benchmark -- --width 512 --height 512 --tile-size 512 --runs 5
```

Results are printed as a table and written to
`benchmarks/results/latest.{json,md}`. Use `--baseline <commit>` to pin an
explicit historical version or `--chrome <path>` to select a browser.

## Credits

Huge thanks to Max Liani for his series: https://maxliani.wordpress.com/2023/03/17/dnnd-1-a-deep-neural-network-dive/. My work is mostly inspired by it.
