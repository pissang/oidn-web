/** HDR transfer functions used by the OIDN input and output processing passes. */
export type HDRTransfer = 'pu' | 'log';

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
const yMax = 65504;
const puXMax = puForward(yMax);
const puNormScale = 1 / puXMax;
const puRcpNormScale = puXMax;
const logXMax = Math.log(yMax + 1);
const logNormScale = 1 / logXMax;

function puForward(y: number) {
  if (y <= y0) return a * y;
  if (y <= y1) return b * Math.pow(y, c) + d;
  return e * Math.log(y + f) + g;
}

function puInverse(x: number) {
  if (x <= x0) return x / a;
  if (x <= x1) return Math.pow((x - d) / b, 1 / c);
  return Math.exp((x - g) / e) - f;
}

function forward(y: number, transfer: HDRTransfer) {
  return transfer === 'log'
    ? Math.log(y + 1) * logNormScale
    : puForward(y) * puNormScale;
}

function inverse(x: number, transfer: HDRTransfer) {
  return transfer === 'log'
    ? Math.exp(x * logXMax) - 1
    : puInverse(x * puRcpNormScale);
}

export function hdrTransferFuncCPU({
  data,
  channels,
  inputScale,
  transfer = 'pu'
}: {
  data: Float32Array;
  channels: number;
  inputScale: number;
  transfer?: HDRTransfer;
}) {
  const newData = new Float32Array(data);
  for (let i = 0; i < newData.length; i += channels) {
    for (let channel = 0; channel < 3; channel++) {
      newData[i + channel] = forward(
        newData[i + channel] * inputScale,
        transfer
      );
    }
  }
  return newData;
}

export function hdrTransferFuncInverseCPU({
  data,
  channels,
  inputScale,
  transfer = 'pu'
}: {
  data: Float32Array;
  channels: number;
  inputScale: number;
  transfer?: HDRTransfer;
}) {
  const newData = new Float32Array(data);
  const outputScale = 1 / inputScale;
  for (let i = 0; i < newData.length; i += channels) {
    for (let channel = 0; channel < 3; channel++) {
      newData[i + channel] = inverse(newData[i + channel], transfer) * outputScale;
    }
  }
  return newData;
}
