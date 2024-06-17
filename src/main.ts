import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu/dist/base';
import { parseTZA } from './tza';
import UNet from './UNet';
import { initWebGPUBackend, initWebGPUBackendWithDevice } from './backend';

export { parseTZA, UNet };

export async function initUNetFromBuffer(
  tzaBuffer: ArrayBuffer,
  backendParams?: { device: GPUDevice; adapterInfo: GPUAdapterInfo },
  opts?: {
    aux?: boolean;
    hdr?: boolean;
    maxTileSize?: number;
  }
) {
  const backend = await (backendParams
    ? initWebGPUBackendWithDevice(
        backendParams.device,
        backendParams.adapterInfo
      )
    : initWebGPUBackend());
  const tensors = parseTZA(tzaBuffer);
  const unet = new UNet(tensors, backend!, opts);
  return unet;
}

export async function initUNetFromURL(
  modelPath: string,
  backendParams?: { device: GPUDevice; adapterInfo: GPUAdapterInfo },
  opts?: {
    aux?: boolean;
    hdr?: boolean;
    maxTileSize?: number;
  }
) {
  return fetch(modelPath)
    .then((res) => res.arrayBuffer())
    .then((ab) => {
      return initUNetFromBuffer(ab, backendParams, opts);
    });
}
