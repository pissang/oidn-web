import {
  KernelConfig,
  registerKernel
} from '@tensorflow/tfjs-core/dist/kernel_registry';

import { mirrorPadConfig } from '@tensorflow/tfjs-backend-webgpu/dist/kernels/MirrorPad';
import { sliceConfig } from '@tensorflow/tfjs-backend-webgpu/dist/kernels/Slice';
import { fusedConv2DConfig } from '@tensorflow/tfjs-backend-webgpu/dist/kernels/FusedConv2D';
import { maxPoolConfig } from '@tensorflow/tfjs-backend-webgpu/dist/kernels/MaxPool';
import { resizeNearestNeighborConfig } from '@tensorflow/tfjs-backend-webgpu/dist/kernels/ResizeNearestNeighbor';
import { concatConfig } from '@tensorflow/tfjs-backend-webgpu/dist/kernels/Concat';
import { identityConfig } from '@tensorflow/tfjs-backend-webgpu/dist/kernels/Identity';

const kernelConfigs: KernelConfig[] = [
  mirrorPadConfig,
  sliceConfig,
  fusedConv2DConfig,
  maxPoolConfig,
  resizeNearestNeighborConfig,
  concatConfig,
  identityConfig
];

for (const kernelConfig of kernelConfigs) {
  registerKernel({
    ...kernelConfig,
    backendName: 'webgpu-oidn'
  });
}
