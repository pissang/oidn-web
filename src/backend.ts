export async function initWebGPUBackend() {
  if (!navigator.gpu) throw new Error('WebGPU is not available');
  const gpuDescriptor: GPURequestAdapterOptions = {
    powerPreference: 'high-performance'
  };

  const adapter = await navigator.gpu.requestAdapter(gpuDescriptor);
  if (!adapter) throw new Error('No WebGPU adapter is available');
  const deviceDescriptor: GPUDeviceDescriptor = {};

  const requiredFeatures: GPUFeatureName[] = [];
  if (adapter.features.has('timestamp-query')) {
    requiredFeatures.push('timestamp-query');
  }
  if (adapter.features.has('bgra8unorm-storage')) {
    requiredFeatures.push('bgra8unorm-storage');
  }
  if (adapter.features.has('shader-f16')) {
    requiredFeatures.push('shader-f16');
  }
  deviceDescriptor.requiredFeatures = requiredFeatures;

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
  return adapter.requestDevice(deviceDescriptor);
}
