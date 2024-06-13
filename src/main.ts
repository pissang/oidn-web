import { parseTZA } from './tza';
import UNet from './UNet';

export { parseTZA, UNet };

export function initUNEtFromModelBuffer(
  buffer: ArrayBuffer,
  opts: {
    aux?: boolean;
    hdr?: boolean;
  }
) {
  const tensors = parseTZA(buffer);
  const unet = new UNet(tensors, {
    aux: opts.aux,
    hdr: opts.hdr
  });
  return unet.setWebGPUBackend().then(() => {
    return unet;
  });
}

export function initUNetFromModelPath(
  modelPath: string,
  opts: {
    aux?: boolean;
    hdr?: boolean;
  }
) {
  return fetch(modelPath)
    .then((res) => res.arrayBuffer())
    .then((ab) => {
      return initUNEtFromModelBuffer(ab, opts);
    });
}
