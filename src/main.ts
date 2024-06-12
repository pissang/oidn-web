import { parseTZA } from './tza';
import UNet from './UNet';

export { parseTZA, UNet };

export function initUNetWithModelPath(
  modelPath: string,
  opts: {
    aux?: boolean;
  }
) {
  return fetch(modelPath)
    .then((res) => res.arrayBuffer())
    .then((ab) => {
      const tensors = parseTZA(ab);
      const unet = new UNet(tensors);
      return unet.setWebGPUBackend().then(() => {
        unet.buildModel(opts.aux ?? false);
        return unet;
      });
    });
}
