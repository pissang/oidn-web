import { initUNetWithModelPath } from '../src/main';
import testImage from './test/test_1spp.png';

const rawCtx = (document.getElementById('raw') as HTMLCanvasElement).getContext(
  '2d'
)!;
const denoisedCtx = (
  document.getElementById('denoised') as HTMLCanvasElement
).getContext('2d')!;

initUNetWithModelPath('../weights/rt_ldr.tza', {
  aux: false
}).then((unet) => {
  const rawImage = new Image();
  rawImage.src = testImage;
  rawImage.onload = () => {
    rawCtx.canvas.width = rawImage.width;
    rawCtx.canvas.height = rawImage.height;
    denoisedCtx.canvas.width = rawImage.width;
    denoisedCtx.canvas.height = rawImage.height;

    rawCtx.drawImage(rawImage, 0, 0);

    const rawData = rawCtx.getImageData(0, 0, rawImage.width, rawImage.height);
    console.time('denoising');
    const denoisedData = unet.executeImageData(rawData);
    console.timeEnd('denoising');
    denoisedCtx.putImageData(denoisedData, 0, 0);
  };
});
