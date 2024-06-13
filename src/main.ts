import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu/dist/base';
import { parseTZA } from './tza';
import UNet from './UNet';
import { initWebGPUBackend } from './backend';

export { parseTZA, UNet };

export async function initUNEtFromModelBuffer(
  tzaBuffer: ArrayBuffer,
  backend?: WebGPUBackend,
  opts?: {
    aux?: boolean;
    hdr?: boolean;
  }
) {
  if (!backend) {
    backend = await initWebGPUBackend();
  }
  const tensors = parseTZA(tzaBuffer);
  const unet = new UNet(tensors, backend, opts);
  return unet;
}

export async function initUNetFromModelPath(
  modelPath: string,
  backend?: WebGPUBackend,
  opts?: {
    aux?: boolean;
    hdr?: boolean;
  }
) {
  if (!backend) {
    backend = await initWebGPUBackend();
  }
  return fetch(modelPath)
    .then((res) => res.arrayBuffer())
    .then((ab) => {
      return initUNEtFromModelBuffer(ab, backend, opts);
    });
}
