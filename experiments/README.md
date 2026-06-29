# BitNet Hardware Feasibility Scaffold

This workspace starts from Microsoft's official `bitnet.cpp` implementation in
`../BitNet`. The purpose of this scaffold is to keep the model proof and the
hardware proof tied to the same artifact:

1. Run the published BitNet model on CPU.
2. Record short-context throughput and memory behavior.
3. Estimate the cost of a hardwired ternary ASIC using the same model shape.
4. Replace estimates with measured kernel data as we build simulator blocks.

## Baseline Model

The default baseline is Microsoft's `BitNet-b1.58-2B-4T`, a 2.4B parameter
native 1.58-bit model with ternary weights. The relevant public implementation
is in `../BitNet`, which provides:

- CPU inference through `run_inference.py`
- end-to-end benchmark script at `utils/e2e_benchmark.py`
- dummy-model generation for layout-only benchmarking
- GGUF conversion utilities
- optimized ternary kernels

The public model config says:

- hidden size: 2560
- intermediate size: 6912
- layers: 30
- attention heads: 20
- key/value heads: 5
- head dim: 128
- max context: 4096 tokens
- vocabulary: 128256

## First Questions

For the target market, we care less about frontier quality and more about
short-context cost:

- context sizes: 256, 512, 1024, 2048 tokens
- output sizes: 32, 64, 128, 256 tokens
- many simultaneous users
- KV cache precision: fp16, int8, int4
- hardware metric: dollars and joules per million generated tokens

## Local Toolchain Status

At the time this scaffold was created and exercised:

- Python 3.9 is present.
- A local `.venv` was created with `cmake`, `huggingface_hub`, and BitNet's
  Python requirements.
- Apple clang 17 is present.
- The official setup script needed two local adjustments on this Mac:
  - put the venv `cmake` on `PATH`
  - configure CMake with `CMAKE_OSX_SYSROOT` and the SDK libc++ include path
- A Metal-enabled build segfaulted during smoke inference. A CPU-only build with
  `GGML_METAL=OFF` ran successfully.

The working CMake configuration was:

```bash
.venv/bin/cmake -B BitNet/build \
  -DBITNET_ARM_TL1=OFF \
  -DGGML_METAL=OFF \
  -DCMAKE_C_COMPILER=/Library/Developer/CommandLineTools/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/Library/Developer/CommandLineTools/usr/bin/clang++ \
  -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
  '-DCMAKE_CXX_FLAGS=-isystem /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1'
```

## Next Milestone

Once the CPU runner is built and the GGUF model is downloaded, run:

```bash
python BitNet/utils/e2e_benchmark.py \
  -m BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p 512 \
  -n 128 \
  -t 4
```

Then feed the measured tokens/sec into:

```bash
python experiments/bitnet_cost_model.py --users 1000 --context 512 --generated 128
```

## CPU Baseline

The official GGUF model was downloaded to:

```text
BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
```

Smoke inference succeeded with the CPU-only build. The model metadata reported:

- model type: 2B
- params: 2.41B in loader metadata, 2.74B in `llama-bench`
- model file type: `I2_S - 2 bpw ternary`
- loaded CPU tensor buffer: about 1.10 GiB
- `llama-bench` model size: about 1.71 GiB
- fp16 KV at 512 context: 37.50 MiB per sequence

Four-thread CPU benchmark results are saved in `experiments/results/`.

| Prompt | Generated | Prompt t/s | Decode t/s |
|---:|---:|---:|---:|
| 256 | 32 | 35.37 | 33.60 |
| 256 | 128 | 33.88 | 24.11 |
| 256 | 256 | 22.08 | 20.84 |
| 512 | 32 | 18.95 | 21.43 |
| 512 | 128 | 20.90 | 21.03 |
| 512 | 256 | 17.73 | 17.72 |
| 1024 | 32 | 15.71 | 21.93 |
| 1024 | 128 | 20.26 | 21.32 |
| 1024 | 256 | 18.00 | 17.58 |

For 1,000 simultaneous users at 512-token context, the current cost model gives:

- fp16 KV: 37.50 MiB/user, 36.62 GiB active KV
- int8 KV: 18.75 MiB/user, 18.31 GiB active KV
- int4 KV: 9.38 MiB/user, 9.16 GiB active KV

## Experiment Ladder

1. Establish the published baseline.
   - download the official GGUF model
   - build `bitnet.cpp`
   - run 256/512/1024-token prompt benchmarks
   - save tokens/sec, wall time, CPU type, thread count, and power if available

2. Build a hardware-faithful accounting model.
   - KV bytes per user
   - KV read bandwidth per generated token
   - activation bytes per layer
   - ternary weight payload and layout assumptions
   - target chip bandwidth and watts

3. Create golden vectors.
   - capture one prompt through the official implementation
   - export token IDs, logits, selected tokens, and KV-cache dimensions
   - later compare the hardware simulator against these outputs

4. Replace estimates with block measurements.
   - ternary matrix-vector tile
   - activation quant/dequant
   - attention/KV tile
   - scheduler for many short requests
