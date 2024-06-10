import { parseTZA } from './tza';
import UNet from './UNet';

export { parseTZA, UNet };

export function initUNetWithModelPath(modelPath: string) {
  return fetch(modelPath)
    .then((res) => res.arrayBuffer())
    .then((ab) => {
      const tensors = parseTZA(ab);
      return new UNet(tensors);
    });
}
