# Changelog

All notable changes to this project are documented in this file.

## [0.4.0] - 2026-08-20

oidn-web 0.4.0 replaces the TensorFlow.js inference stack with a purpose-built, model-driven WebGPU runtime. Existing `initUNetFromURL` and `initUNetFromBuffer` integrations remain supported while gaining native FP16, adaptive scheduling, runtime diagnostics, and stronger model validation.

### Highlights

- Removed TensorFlow.js and its WebGPU backend from the runtime dependencies.
- Added a custom WGSL U-Net executor with automatic FP16 selection and a deterministic FP32 fallback.
- Added built-in topology descriptors for the current OIDN small and large RT models, including clean auxiliary variants.
- Reduced the reference Vite ESM bundle from approximately 168.8 KB to 31.1 KB gzip compared with 0.3.5, a reduction of about 82%.
- Reached a 27.6 ms median for a 512 x 512 Clean Aux Large inference on the tested Apple GPU, versus 35.0 ms for the TensorFlow.js baseline (1.27x faster). Performance depends on the browser, GPU, model, and tile size.

### Added

- Model-driven graph planning and lifetime-aware activation-buffer reuse.
- Versioned `UNetModelSpec` descriptors, topology detection, and validation of tensor names, layouts, data types, byte lengths, kernel shapes, bias shapes, and graph channel flow.
- `modelSpec` initialization option for future or custom OIDN model topologies.
- `npm run model:inspect` for model hashes, tensor signatures, and descriptor compatibility checks.
- Dynamic tile sizing with configurable minimum, initial, and maximum tile sizes and a target GPU time.
- GPU queue backpressure between tiles to keep cancellation and rendering interaction responsive.
- Runtime diagnostics through `getRuntimeInfo()`, including the selected engine, precision, model, kernel capabilities, tile state, and resource statistics.
- Per-layer GPU profiling through `profileNextExecution()` and `getLastExecutionProfile()` when `timestamp-query` is enabled.
- Cross-version browser benchmarks with queue-complete timings, sampled output validation, TFJS baseline comparison, and per-layer GPU profiles.
- Explicit, idempotent resource disposal and resource lifecycle accounting.
- Automated model, graph, scheduler, resource leak, allocation rollback, and late-async-cleanup tests.

### Changed

- U-Net inference now runs directly in WGSL. The stable `auto` engine selects the native WGSL backend and no longer initializes TensorFlow.js.
- FP16-capable devices retain TZA half-float weights and use short FP16 FMA accumulation groups folded into FP32 accumulators. The final output remains FP32.
- Devices without `shader-f16` automatically use native FP32 inference.
- Convolution tensors and activations use a blocked four-channel layout with channel-specialized shaders.
- Decoder `upsample + concat + conv` patterns are fused, and all network passes for a tile are encoded into one command buffer.
- Shape-independent pipelines are compiled asynchronously before initialization resolves to avoid first-denoise shader compilation stalls.
- `maxTileSize` is now a hard upper bound for the adaptive tile controller.
- A shared `GPUDevice` uses FP16 only when `shader-f16` was requested during device creation; WebGPU features cannot be enabled afterward.

### Migration notes

- No changes are required for basic `initUNetFromURL` or `initUNetFromBuffer` usage.
- Call `dispose()` when a U-Net instance is no longer needed.
- When supplying an existing `GPUDevice`, request `shader-f16` before creating the device if FP16 inference is desired.
- Set `dynamicTile: false` to restore fixed-size tiling.

[0.4.0]: https://github.com/pissang/oidn-web/compare/v0.3.5...v0.4.0
