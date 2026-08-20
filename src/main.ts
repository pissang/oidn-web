import { parseTZA } from './tza';
import UNet from './UNet';
import type { UNetEngineSetting, UNetExecutionStats } from './UNet';
import { initWebGPUBackend, initWebGPUBackendWithDevice } from './backend';
import type { DynamicTileSetting } from './tileScheduler';
import type { UNetModelSpec } from './modelSpec';
import type {
  NativeUNetKernelSetting,
  NativeUNetPrecisionSetting
} from './nativeUNet';

export { parseTZA, UNet };
export type { DynamicTileOptions, DynamicTileSetting } from './tileScheduler';
export {
  detectUNetModelSpec,
  OIDN_UNET_LARGE_SPEC,
  OIDN_UNET_SMALL_SPEC,
  validateUNetModel
} from './modelSpec';
export type {
  ModelNodeSpec,
  UNetModelGraph,
  UNetModelSpec,
  ValidatedUNetModel
} from './modelSpec';
export { optimizeModelGraph, planModelExecution } from './graphOptimizer';
export type {
  ExecutableModelNode,
  GraphOptimizationOptions,
  ModelExecutionPlan,
  ModelValueShape,
  OptimizedModelGraph
} from './graphOptimizer';
export {
  NativeUNetExecutor,
  resolveNativeUNetPrecision
} from './nativeUNet';
export type {
  NativeUNetExecutionProfile,
  NativeUNetKernel,
  NativeUNetKernelSetting,
  NativeUNetLayerTiming,
  NativeUNetOptions,
  NativeUNetPrecision,
  NativeUNetPrecisionSetting
} from './nativeUNet';

export interface UNetOptions {
  aux?: boolean;
  hdr?: boolean;
  /** Hard upper bound for an output tile edge. Defaults to 512. */
  maxTileSize?: number;
  /** Adaptive GPU-time-based tile sizing. Enabled by default. */
  dynamicTile?: DynamicTileSetting;
  /** `auto` uses stable WGSL; `webnn` opts into the experimental WebNN backend. */
  engine?: UNetEngineSetting;
  /** `auto` selects FP16 when shader-f16 was enabled on the GPUDevice. */
  precision?: NativeUNetPrecisionSetting;
  /** `auto` selects kernels from precision, operation shape, and GPU limits. */
  kernel?: NativeUNetKernelSetting;
  /** Versioned topology descriptor for future/custom OIDN TZA models. */
  modelSpec?: UNetModelSpec;
}

export type { UNetEngineSetting, UNetExecutionStats } from './UNet';
export { WebNNUNetExecutor } from './webnnUNet';
export type { WebNNRuntimeSupport, WebNNUNetOptions } from './webnnUNet';

export async function initUNetFromBuffer(
  tzaBuffer: ArrayBuffer,
  backendParams?: { device: GPUDevice; adapterInfo: GPUAdapterInfo },
  opts?: UNetOptions
) {
  const backend = await (backendParams
    ? initWebGPUBackendWithDevice(
        backendParams.device,
        backendParams.adapterInfo
      )
    : initWebGPUBackend());
  const tensors = parseTZA(tzaBuffer);
  const unet = new UNet(tensors, backend, opts);
  await unet.prepare();
  return unet;
}

export async function initUNetFromURL(
  modelPath: string,
  backendParams?: { device: GPUDevice; adapterInfo: GPUAdapterInfo },
  opts?: UNetOptions
) {
  return fetch(modelPath)
    .then((res) => res.arrayBuffer())
    .then((ab) => {
      return initUNetFromBuffer(ab, backendParams, opts);
    });
}
