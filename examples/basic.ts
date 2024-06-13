import { UNet, initUNetFromModelPath } from '../src/main';

const rawCtx = (document.getElementById('raw') as HTMLCanvasElement).getContext(
  '2d'
)!;
const denoisedCtx = (
  document.getElementById('denoised') as HTMLCanvasElement
).getContext('2d')!;

let abortDenoising;

function denoise(rawImage: HTMLImageElement, unet: UNet) {
  if (abortDenoising) {
    abortDenoising();
  }

  const width = +(document.getElementById('width') as HTMLInputElement).value;
  const height = +(document.getElementById('height') as HTMLInputElement).value;

  rawCtx.canvas.width = width;
  rawCtx.canvas.height = height;
  denoisedCtx.canvas.width = width;
  denoisedCtx.canvas.height = height;

  const pattern = rawCtx.createPattern(rawImage, 'repeat');
  rawCtx.fillStyle = pattern;
  rawCtx.fillRect(0, 0, width, height);

  const rawData = rawCtx.getImageData(0, 0, width, height);
  console.time('denoising');
  abortDenoising = unet.progressiveExecute({
    color: rawData,
    done(denoised) {
      console.timeEnd('denoising');
      // denoisedCtx.putImageData(denoised, 0, 0);
    },
    progress(tileData, _, tile) {
      denoisedCtx.putImageData(tileData, tile.x, tile.y);
    }
  });
}

initUNetFromModelPath('../weights/rt_ldr.tza', {
  aux: false
}).then((unet) => {
  const rawImage = new Image();
  rawImage.src = './test/test_1spp.png';
  rawImage.onload = () => {
    denoise(rawImage, unet);
  };

  document.getElementById('resize')!.addEventListener('click', () => {
    denoise(rawImage, unet);
  });
});
