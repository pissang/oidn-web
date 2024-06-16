import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu/dist/base';
import { parseTZA } from './tza';
import UNet from './UNet';
import { initWebGPUBackend, initWebGPUBackendWithDevice } from './backend';

export { parseTZA, UNet };

export async function initUNEtFromModelBuffer(
  tzaBuffer: ArrayBuffer,
  backendParams?: { device: GPUDevice; adapterInfo: GPUAdapterInfo },
  opts?: {
    aux?: boolean;
    hdr?: boolean;
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

export async function initUNetFromModelPath(
  modelPath: string,
  backendParams?: { device: GPUDevice; adapterInfo: GPUAdapterInfo },
  opts?: {
    aux?: boolean;
    hdr?: boolean;
  }
) {
  return fetch(modelPath)
    .then((res) => res.arrayBuffer())
    .then((ab) => {
      return initUNEtFromModelBuffer(ab, backendParams, opts);
    });
}
