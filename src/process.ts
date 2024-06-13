// TODO put in gpu
export function avgLogLum({
  data,
  channels
}: {
  data: Float32Array;
  channels: number;
}) {
  let totalLuminance = 0;

  for (let i = 0; i < data.length; i += channels) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const lum = 0.212671 * r + 0.71516 * g + 0.072169 * b;

    // TODO performance
    // https://github.com/RenderKit/oidn/blob/713ec7838ba650f99e0a896549c0dca5eeb3652d/devices/cpu/cpu_autoexposure.cpp#L43
    totalLuminance += Math.log10(lum + 0.0001) / Math.log10(2);
  }
  const numPixels = data.length / channels;
  const averageLuminance = totalLuminance / numPixels;
  const key = 0.18;

  // https://github.com/RenderKit/oidn/blob/713ec7838ba650f99e0a896549c0dca5eeb3652d/training/color.py#L173
  return key / Math.pow(2, averageLuminance);
}

const a = 1.41283765e3;
const b = 1.64593172;
const c = 4.31384981e-1;
const d = -2.94139609e-3;
const e = 1.92653254e-1;
const f = 6.26026094e-3;
const g = 9.98620152e-1;
const y0 = 1.5794576e-6;
const y1 = 3.22087631e-2;
const x0 = 2.23151711e-3;
const x1 = 3.70974749e-1;

export function hdrTransferFunc({
  data,
  channels,
  inputScale
}: {
  data: Float32Array;
  channels: number;
  inputScale: number;
}) {
  const newData = new Float32Array(data.length);
  newData.set(data);
  // https://github.com/RenderKit/oidn/blob/713ec7838ba650f99e0a896549c0dca5eeb3652d/core/color.h#L71
  for (let i = 0; i < newData.length; i += channels) {
    // First three are color
    for (let c = 0; c < 3; c++) {
      let y = newData[i + c] * inputScale;
      if (y <= y0) {
        y = a * y;
      } else if (y <= y1) {
        y = b * Math.pow(y, c) + d;
      } else {
        y = e * Math.log(y + f) + g;
      }
      newData[i + c] = y;
    }
  }

  return newData;
}

export function hdrTransferFuncInverse({
  data,
  channels,
  inputScale
}: {
  data: Float32Array;
  channels: number;
  inputScale: number;
}) {
  const newData = new Float32Array(data.length);
  newData.set(data);

  const outputScale = 1 / inputScale;
  for (let i = 0; i < newData.length; i += channels) {
    for (let c = 0; c < 3; c++) {
      let x = newData[i + c] * outputScale;

      if (x <= x0) {
        x = x / a;
      } else if (x <= x1) {
        x = Math.pow((x - d) / b, 1 / c);
      } else {
        x = Math.exp((x - g) / e) - f;
      }
      newData[i + c] = x;
    }
  }

  return newData;
}
