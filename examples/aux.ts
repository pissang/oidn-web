import { initUNetFromModelPath } from '../src/main';
import testColor from './test/test4_color.png';
import testAlbedo from './test/test4_albedo.png';
import testNorm from './test/test4_norm.png';

const rawCtx = (document.getElementById('raw') as HTMLCanvasElement).getContext(
  '2d'
)!;
const denoisedCtx = (
  document.getElementById('denoised') as HTMLCanvasElement
).getContext('2d')!;

function loadImage(url: string) {
  return new Promise<HTMLImageElement>((resolve) => {
    const image = new Image();
    image.src = url;
    image.onload = () => {
      resolve(image);
    };
  });
}

initUNetFromModelPath('../weights/rt_ldr_alb_nrm.tza', {
  aux: true
}).then((unet) => {
  Promise.all([
    loadImage(testColor),
    loadImage(testAlbedo),
    loadImage(testNorm)
  ]).then(([colorImage, albedoImage, normImage]) => {
    const w = colorImage.width;
    const h = colorImage.height;
    rawCtx.canvas.width = w;
    rawCtx.canvas.height = h;
    denoisedCtx.canvas.width = w;
    denoisedCtx.canvas.height = h;

    rawCtx.clearRect(0, 0, w, h);
    rawCtx.drawImage(albedoImage, 0, 0, w, h);
    const albedoData = rawCtx.getImageData(0, 0, w, h);
    rawCtx.clearRect(0, 0, w, h);
    rawCtx.drawImage(normImage, 0, 0, w, h);
    const normData = rawCtx.getImageData(0, 0, w, h);

    rawCtx.clearRect(0, 0, w, h);
    rawCtx.drawImage(colorImage, 0, 0, w, h);
    const colorData = rawCtx.getImageData(0, 0, w, h);
    console.time('denoising');
    unet
      .executeImageData(
        colorData,
        albedoData,
        normData,
        (tileData, _, tile) => {
          denoisedCtx.putImageData(tileData, tile.x, tile.y);
        }
      )
      .then((denoisedData) => {
        // denoisedCtx.putImageData(denoisedData, 0, 0);
        console.timeEnd('denoising');
      });
  });
});
