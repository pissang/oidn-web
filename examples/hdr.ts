import { initUNetWithModelPath } from '../src/main';
import testColor from './test/test.hdr';
import testAlbedo from './test/test4_albedo.png';
import testNorm from './test/test4_norm.png';
import { readHDR } from '../src/hdr';

const rawCtx = (document.getElementById('raw') as HTMLCanvasElement).getContext(
  '2d'
)!;
const denoisedCtx = (
  document.getElementById('denoised') as HTMLCanvasElement
).getContext('2d')!;

function loadHDR(url: string) {
  return fetch(url)
    .then((res) => res.arrayBuffer())
    .then((ab) => {
      const res = readHDR(new Uint8Array(ab));
      if (typeof res === 'string') {
        throw new Error(res);
      }
      const data = new Float32Array((res.rgbFloat.length / 3) * 4);
      for (let i = 0; i < res.rgbFloat.length / 3; i++) {
        data[i * 4 + 0] = res.rgbFloat[i * 3 + 0];
        data[i * 4 + 1] = res.rgbFloat[i * 3 + 1];
        data[i * 4 + 2] = res.rgbFloat[i * 3 + 2];
        data[i * 4 + 3] = 1;
      }
      return {
        data,
        width: res.width,
        height: res.height
      };
    });
}
function loadImage(url: string) {
  return new Promise<HTMLImageElement>((resolve) => {
    const image = new Image();
    image.src = url;
    image.onload = () => {
      resolve(image);
    };
  });
}

initUNetWithModelPath('../weights/rt_ldr_alb_nrm.tza', {
  aux: true
}).then((unet) => {
  Promise.all([
    loadHDR(testColor),
    loadImage(testAlbedo),
    loadImage(testNorm)
  ]).then(([colorData, albedoImage, normImage]) => {
    const w = colorData.width;
    const h = colorData.height;
    rawCtx.canvas.width = w;
    rawCtx.canvas.height = h;
    denoisedCtx.canvas.width = w;
    denoisedCtx.canvas.height = h;

    rawCtx.clearRect(0, 0, w, h);
    rawCtx.drawImage(albedoImage, 0, 0);
    const albedoData = rawCtx.getImageData(0, 0, w, h);
    rawCtx.clearRect(0, 0, w, h);
    rawCtx.drawImage(normImage, 0, 0);
    const normData = rawCtx.getImageData(0, 0, w, h);

    console.time('denoising');
    const denoisedData = unet.executeImageData(colorData, albedoData, normData);
    console.timeEnd('denoising');
    denoisedCtx.putImageData(denoisedData, 0, 0);
  });
});
