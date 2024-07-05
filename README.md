# Open Image Denoise on the Web

This library brings the state-of-the-art AI-based denoising library [Open Image Denoise](https://github.com/RenderKit/oidn) to the web.
Currently it's only available on the browsers support WebGPU.

It's used in the [Vector to 3D](https://www.figma.com/community/plugin/1264600219316901594/) Figma plugin for high quality rendering and denoising.

|                                           2000 Samples                                           |                                     3 Samples                                      |                                   3 Samples + Denoised                                   |
| :----------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: |
| ![](https://github.com/pissang/oidn-web/blob/main/examples/test/ground-truth.png 'Ground Truth') | ![](https://github.com/pissang/oidn-web/blob/main/examples/test/noisy.png 'Noisy') | ![](https://github.com/pissang/oidn-web/blob/main/examples/test/denoised.png 'Denoised') |

## How it Works.

It uses [tfjs](https://github.com/tensorflow/tfjs) to build the UNet model used by the OIDN. Then use the model to do prediction with a WebGPU backend from the image data.

## How to Use

[Basic Example](https://oidn-web-example.vercel.app/) [Code](https://github.com/pissang/oidn-web-example/blob/main/src/main.js)

[Use with three-gpu-pathtracer](https://oidn-web-example.vercel.app/three-gpu-pathtracer.html) [code](https://github.com/pissang/oidn-web-example/blob/main/src/three-gpu-pathtracer.js)

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
    color: { data: noisyImageData, width, height },
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
    // Share GPUDevice and GPUAdapterInfo to the TFJS WebGPU backend
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

## Credits

Huge thanks to Max Liani for his series: https://maxliani.wordpress.com/2023/03/17/dnnd-1-a-deep-neural-network-dive/. My work is mostly inspired by it.
