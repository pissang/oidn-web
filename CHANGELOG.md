# Changelog

All notable changes to this project are documented in this file.

## [0.5.0] - 2026-10-09

oidn-web 0.5.0 simplifies device setup, makes tiled execution always settle, and switches FP16 convolutions to implicit GEMM. It contains breaking changes for TypeScript callers that pass `adapterInfo`, for code that constructs `UNet` directly, and for code that relies on square tiles or per-frame tile pacing.

### Performance

Measured with `npm run benchmark -- --direct-gemm-only` in headless Chrome, using the Clean Aux Large model with FP16. Times are medians. Performance depends on the browser, GPU, model, and tile size. Devices without `shader-f16` use FP32, which already used implicit GEMM in 0.4.0, so their gains are expected to be smaller.

| GPU | Input | 0.4.0 | 0.5.0 | Speedup |
| --- | --- | ---: | ---: | ---: |
| Apple GPU (macOS) | 512 x 512, one tile | 27.2 ms | 19.1 ms | 1.42x |
| Apple GPU (macOS) | 1920 x 1080, 512 px tiles (12 tiles) | 742 ms | 369 ms | 2.01x |
| NVIDIA RTX 4060 Laptop (Windows 11, Chrome 153) | 512 x 512, one tile | 339 ms | 39.7 ms | 8.55x |
| NVIDIA RTX 4060 Laptop (Windows 11, Chrome 153) | 1920 x 1080, 512 px tiles (12 tiles) | 8844 ms | 555 ms | 15.94x |

- Implicit GEMM is now the default for FP16. The 0.4.0 direct FP16 kernel was especially slow on the tested NVIDIA GPU. Network GPU time for a 512 x 512 tile drops from 26.8 ms to 18.5 ms on the Apple GPU, and from 360 ms to 38.5 ms on the RTX 4060.
- Overlap only on shared tile edges reduces the input each tile computes. With the direct kernel alone, the 1920 x 1080 case is about 1.5x faster than 0.4.0 on both GPUs.

### Breaking changes

- `initUNetFromURL` and `initUNetFromBuffer` take `{ device }` as the second argument. `adapterInfo` was only needed by the removed TensorFlow.js backend and is no longer accepted by the type. Extra properties are ignored at runtime, but TypeScript rejects an object literal that still contains `adapterInfo`.
- `new UNet(tensors, device, options)` takes a `GPUDevice` instead of a `{ device, adapterInfo }` object.
- `tileExecute` now continues on the event loop between tiles by default instead of waiting for `requestAnimationFrame`. Pass `scheduling: 'animation-frame'` to keep pacing tiles to display frames.
- Tiles are balanced rectangles with overlap only on edges shared with another tile, instead of fixed-size squares. Tile count, position, and size reported to `progress` differ from 0.4.0. The default `dynamicTile.initialTileSize` is now 432.
- `UNetExecutionStats` no longer has `tileWidth` and `tileHeight`. Use `tileColumns`, `tileRows`, `tileOverlap`, `inputPixelCount`, and `inputShapeCount` instead.
- FP16 convolutions use implicit GEMM by default. Output differs slightly from 0.4.0 because accumulation order changed. Set `kernel: 'direct'` to reproduce the 0.4.0 FP16 path.

### Added

- `hdrTransfer: 'log'` for RTLightmap HDR models, alongside the default PU transfer.
- `error` callback on `tileExecute` for asynchronous failures, WebGPU device loss, and exceptions thrown by `progress` or `done`. `progress` and `done` may return promises.
- `scheduling: 'event-loop' | 'animation-frame'`, `tileOverlap`, and `wholeImage` options on `tileExecute`.
- `prepareForImage(width, height)` to create per-shape GPU resources before the first denoise.
- `planTileGrid` and its `TilePlan`, `PlannedTile`, and `TileRect` types.
- Experimental `gemm` tuning options. These are intended for benchmarking and are not covered by semver.
- `hdrTransfer` and `activeExecutionCount` in `getRuntimeInfo()`, and the active GEMM configuration under `kernel.gemm`.

### Changed

- Implicit GEMM tiles are selected by output alignment and GPU limits, with optimized addressing, weight layout, register tiles, shared-memory layout, and pooling access.
- The final RGB convolution uses an adaptive shared-memory cache.
- Adaptive tile sizing uses the smoothed P75 tile GPU time, excludes the cold first tile, ignores cancelled and single-tile work, and buckets input shapes to at most two sizes.
- `animation-frame` scheduling falls back to a 100 ms timer so execution still completes in hidden tabs.

### Fixed

- Tiled execution always settles with `done` or `error`, or stops silently after abort, including when `requestAnimationFrame` never fires or the device is lost.
- Completed executions release their device-loss listeners.
- Edge tiles whose size is not a multiple of 16 replicate edge pixels into the padded model input.
- The npm package no longer includes benchmark results, tests, and scripts.
- Renamed `examples/aux.*` to `examples/auxiliary.*` so the repository can be checked out on Windows, where `AUX` is a reserved file name.

### Migration notes

- Replace `{ device, adapterInfo }` with `{ device }`.
- Replace `new UNet(tensors, { device, adapterInfo }, options)` with `new UNet(tensors, device, options)`.
- Interactive renderers that share the GPU with OIDN should pass `scheduling: 'animation-frame'`.
- Pass an `error` callback to handle failures. Without one, failures are logged to the console.

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

[0.5.0]: https://github.com/pissang/oidn-web/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/pissang/oidn-web/compare/v0.3.5...v0.4.0
