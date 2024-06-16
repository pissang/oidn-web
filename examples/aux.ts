import { initUNetFromModelPath } from '../src/main';

const rawCtx = (document.getElementById('raw') as HTMLCanvasElement).getContext(
  '2d'
)!;
const denoisedCtx = (
  document.getElementById('denoised') as HTMLCanvasElement
).getContext('2d')!;

let abortDenoising;

function loadImage(url: string) {
  return new Promise<HTMLImageElement>((resolve) => {
    const image = new Image();
    image.src = url;
    image.onload = () => {
      resolve(image);
    };
  });
}

initUNetFromModelPath('../weights/rt_ldr_alb_nrm.tza', undefined, {
  aux: true
}).then((unet) => {
  Promise.all([
    loadImage('./test/test4_color.png'),
    loadImage('./test/test4_albedo.png'),
    loadImage('./test/test4_norm.png')
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

    abortDenoising = unet.progressiveExecute({
      color: colorData,
      albedo: albedoData,
      normal: normData,
      done() {
        console.timeEnd('denoising');
      },
      progress(_, tileData, tile) {
        denoisedCtx.putImageData(tileData, tile.x, tile.y);
      }
    });
  });
});
