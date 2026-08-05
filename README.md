# llama.cpp + TurboQuant+

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3A0cc4m%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [compile times](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-compile-times.md) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) integrating the [TurboQuant+](https://github.com/TheTom/llama-cpp-turboquant) implementation of TheTom's TurboQuant+ codec. This fork adds KV-cache and weight quantization types, cross-backend kernel support, and model-family-specific fixes.

## What this fork adds

### KV-cache quantization (runtime, `--cache-type-k` / `--cache-type-v`)

| Type | Bits | Use case |
|---|---|---|
| `turbo4` | ~4.5 | Lightest compression; safe starting point |
| `turbo3` | ~3.5 | Recommended: ~4.6× V compression at <1.5% PPL loss |
| `turbo2` | ~2.0 | Aggressive; pair with Boundary V protection |

**Key insight**: [Asymmetric K/V compression](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md) — **V tolerates aggressive compression, K does not**. Always keep K at higher precision (`f16` or `q8_0`) and compress V.

Recommended configs:
- **Safe start**: `--cache-type-k f16 --cache-type-v turbo4`
- **Default**: `--cache-type-k q8_0 --cache-type-v turbo3`
- **Long context**: `--cache-type-k q8_0 --cache-type-v turbo2`

### Weight quantization (offline, via `llama-quantize`)

| Type | Bits | Notes |
|---|---|---|
| `TQ4_1S` | ~4.5 | Recommended: V2.1 fused Metal kernels, CUDA `dp4a` 3.5× faster |
| `TQ3_1S` | ~3.5 | Smaller VRAM than `q8_0`; accept ~1-2 PPL bump |

```bash
llama-quantize model.f16.gguf model.tq4_1s.gguf TQ4_1S
```

### Advanced features

- **Auto-asymmetric K/V** — complementary codec selection when both sides are turbo/TQ types
- **Boundary V (layer-aware)** — auto-enabled for `turbo2-V`; protects sensitive layers
- **Sparse V dequantization** — Metal-only; skips dequant for low-attention positions
- **Flash Attention** — auto-enabled with backend-specific kernels

### Backend coverage

| Backend | Kernels | Flash Attn | Notes |
|---|---|---|---|
| **Metal** (Apple Silicon) | V2.1 fused, TurboFlash | Yes (`dk=512` for Gemma 4) | Sparse V across family |
| **CUDA** (NVIDIA) | `dp4a` for TQ4_1S, warp-cooperative dequant | Yes (turbo VEC FA +9%) | Multi-GPU support |
| **HIP/ROCm** (AMD) | Portable `ggml_cuda_dp4a`, scalar half fallback | Yes (VEC FA forced) | RDNA3/4, CDNA3/4 |
| **Vulkan** | `TQ4_1S` weights, `SET_ROWS` for turbo K/V | coopmat flash attn | Compute-shader path |
| **SYCL** (Intel) | `SET_ROWS`, WHT rotation | VEC FA for all combos | Intel Arc A380/B70 |

### Model-family support

- **Gemma 4** — MoE token routing, op-concurrency handling
- **Large MoE** — up to 256-expert routing kernels
- **Hybrid architectures** (Mamba/GDN) — speculative decoding integration
- All existing llama.cpp models remain fully supported

---

## Quick start

### Build from source

```bash
# Clone this fork (not upstream!)
git clone https://github.com/eyalezer/llama.cpp.git

cd llama.cpp

# Build with your preferred backend
cmake -B build -DGGML_METAL=ON && cmake --build build -j
# or: cmake -B build -DGGML_CUDA=ON && cmake --build build -j
# or: cmake -B build -DGGML_VULKAN=ON && cmake --build build -j

# Run a model with turbo KV cache
llama-cli -m model.gguf --cache-type-k q8_0 --cache-type-v turbo3 -p "Hello!"
```

---

## Why the `+`?

TurboQuant+ extends Google's original TurboQuant (ICLR 2026) paper with:

- Asymmetric K/V policy ([paper](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md))
- Layer-aware Boundary V protection ([paper](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/layer-aware-v-compression.md))
- Attention-gated sparse V dequantization ([paper](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/sparse-v-dequant.md))
- `TQ3_1S` / `TQ4_1S` weight quantization ([paper](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/weight-compression-tq4.md))
- Cross-backend kernel coverage (CUDA `dp4a`, HIP RDNA/CDNA, Vulkan coopmat, Metal V2.1)

The trailing `+` denotes ongoing extension work; the original TurboQuant codec remains the foundation.

---

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
