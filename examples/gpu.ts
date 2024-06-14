import { initUNetFromModelPath } from '../src/main';
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

let abortDenoising;

function clamp(v: number) {
  return Math.min(1.0, Math.max(0.0, v));
}
function aces(v: number) {
  return clamp((v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14));
}

function convertHDRDataToImageData(hdrData: {
  data: Float32Array;
  width: number;
  height: number;
}) {
  const { data, width, height } = hdrData;
  const newData = new Uint8ClampedArray(data.length);
  for (let i = 0; i < data.length; i += 4) {
    const r = Math.sqrt(aces(data[i])) * 255;
    const g = Math.sqrt(aces(data[i + 1])) * 255;
    const b = Math.sqrt(aces(data[i + 2])) * 255;
    newData[i + 0] = r;
    newData[i + 1] = g;
    newData[i + 2] = b;
    newData[i + 3] = 255;
  }
  return new ImageData(newData, width, height);
}

initUNetFromModelPath('../weights/rt_hdr_alb_nrm.tza', undefined, {
  aux: true
}).then((unet) => {
  Promise.all([
    loadHDR('./test/test4_color.hdr'),
    loadImage('./test/test4_albedo.png'),
    loadImage('./test/test4_norm.png')
  ]).then(([colorData, albedoImage, normImage]) => {
    const w = colorData.width;
    const h = colorData.height;
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

    rawCtx.putImageData(convertHDRDataToImageData(colorData), 0, 0);

    console.time('denoising');

    const device = unet.getDevice();
    const colorBuffer = device.createBuffer({
      size: colorData.data.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_WRITE
    });
    const albedoBuffer = device.createBuffer({
      size: albedoData.data.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_WRITE
    });
    const normalBuffer = device.createBuffer({
      size: normData.data.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_WRITE
    });
    device.queue.writeBuffer(colorBuffer, 0, colorData.data);
    device.queue.writeBuffer(
      albedoBuffer,
      0,
      new Float32Array(albedoData.data)
    );
    device.queue.writeBuffer(normalBuffer, 0, new Float32Array(normData.data));
    abortDenoising = unet.progressiveExecute({
      color: { data: colorBuffer, width: w, height: h },
      albedo: { data: albedoBuffer, width: w, height: h },
      normal: { data: normalBuffer, width: w, height: h },
      hdr: true,
      done(finalBuffer) {
        console.timeEnd('denoising');

        finalBuffer.data.mapAsync(GPUMapMode.READ).then((mapped) => {
          denoisedCtx.putImageData(
            convertHDRDataToImageData({
              data: new Float32Array(finalBuffer.data.getMappedRange()),
              width: w,
              height: h
            }),
            0,
            0
          );
          finalBuffer.data.unmap();
        });
      },
      progress(tileData, _, tile) {}
    });
  });
});
