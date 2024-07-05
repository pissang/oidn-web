# Open Image Denoise on the Web

This library brings the state-of-the-art AI-based denoising library [Open Image Denoise](https://github.com/RenderKit/oidn) to the web.

Many thanks to Max Liani for his series https://maxliani.wordpress.com/2023/03/17/dnnd-1-a-deep-neural-network-dive/. My work is mostly inspired by it.

## How it Works.

It uses [tfjs](https://github.com/tensorflow/tfjs) to build the UNet model used by the OIDN. Then use the model to do prediction with a WebGPU backend from the image data.

Currently it's only available on the browser with WebGPU enabled.

## How to Use

### Denoise a noisy LDR image.

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

### Denoise a noisy HDR image.

```ts
import { UNet, initUNetFromURL } from 'oidn-web';
initUNetFromURL('./weights/rt_hdr.tza', undefined, {
  // Need to let the library know it's hdr input.
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

### Using AUX

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
