import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu/dist/base';
import { ENGINE } from '@tensorflow/tfjs-core/dist/engine';

import './kernels';

export async function initWebGPUBackend() {
  try {
    const gpuDescriptor: GPURequestAdapterOptions = {
      powerPreference: 'high-performance'
    };

    const adapter = (await navigator.gpu.requestAdapter(gpuDescriptor))!;
    const deviceDescriptor: GPUDeviceDescriptor = {};

    const requiredFeatures = [];
    if (adapter.features.has('timestamp-query')) {
      requiredFeatures.push('timestamp-query');
    }
    if (adapter.features.has('bgra8unorm-storage')) {
      requiredFeatures.push(['bgra8unorm-storage']);
    }
    deviceDescriptor.requiredFeatures =
      requiredFeatures as Iterable<GPUFeatureName>;

    const adapterLimits = adapter.limits;
    deviceDescriptor.requiredLimits = {
      maxComputeWorkgroupStorageSize:
        adapterLimits.maxComputeWorkgroupStorageSize,
      maxComputeWorkgroupsPerDimension:
        adapterLimits.maxComputeWorkgroupsPerDimension,
      maxStorageBufferBindingSize: adapterLimits.maxStorageBufferBindingSize,
      maxBufferSize: adapterLimits.maxBufferSize,
      maxComputeWorkgroupSizeX: adapterLimits.maxComputeWorkgroupSizeX,
      maxComputeInvocationsPerWorkgroup:
        adapterLimits.maxComputeInvocationsPerWorkgroup
    };
    const device = await adapter.requestDevice(deviceDescriptor);
    const adapterInfo = await adapter.requestAdapterInfo();

    return initWebGPUBackendWithDevice(device, adapterInfo);
  } catch (e) {}
}

export async function initWebGPUBackendWithDevice(
  device: GPUDevice,
  adapter: GPUAdapterInfo
) {
  // TODO multiple device and adapter in one backend
  let backend = ENGINE.findBackend('webgpu-oidn');
  if (backend != null) {
    return backend as WebGPUBackend;
  }

  backend = new WebGPUBackend(device, adapter);
  ENGINE.registerBackend('webgpu-oidn', () => backend);
  await ENGINE.setBackend('webgpu-oidn');

  return backend as WebGPUBackend;
}
