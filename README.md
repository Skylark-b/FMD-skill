# deploy-frontier-model

> End-to-end local deployment of frontier AI models — provide a URL, a model name, or even just a model family, and get a running, validated instance in a single session.

---

## Problem Statement

Deploying a frontier model locally is a multi-stage engineering task that spans several domains:

1. **Requirement analysis** — parse model cards and config files to extract hardware prerequisites (VRAM, RAM, disk), Python version constraints, and dependency trees.
2. **Platform qualification** — cross-reference requirements against the host machine's GPU compute capability, available memory, and free storage.
3. **Runtime provisioning** — detect or install Python toolchains (CPython, Conda/Miniconda), configure environment variables, and establish shell integration.
4. **Environment construction** — create an isolated Conda environment with the correct Python version, install PyTorch compiled against the host's CUDA driver, and resolve the full dependency DAG.
5. **Artifact acquisition** — download model weights (often tens of GB) with resumable transfers, verify checksums, and handle gated-repo authentication.
6. **Integration testing** — author a minimal inference harness, execute a smoke test, and triage failures (OOM, missing tokenizer, CUDA kernel mismatch, dtype incompatibility).

A failure at any stage can cascade, and debugging from a cold start frequently costs hours. This skill automates the entire pipeline end-to-end.

---

## Trigger Conditions

The skill activates when the user's message contains any of the following intents:

| Language | Trigger phrases |
|----------|----------------|
| English | `deploy a frontier model`, `download a model`, `run a model locally`, `deploy a large language model`, `set up a model`, `install a model` |
| Chinese | `部署前沿模型`, `下载模型`, `本地运行模型`, `安装模型`, `配置模型环境` |

It also activates when a user directly provides a HuggingFace URL or mentions a model family/code name with deployment intent.

**Invocation examples:**

```
# By URL
Deploy https://huggingface.co/Qwen/Qwen2.5-0.5B to D:\models

# By precise model name
帮我部署 Qwen3-8B

# By vague family name — skill will interactively narrow down
Download and set up a small Qwen model on my machine
```

---

## Pipeline Architecture

```
User Input (URL / name / family)
      │
      ▼
┌──────────────────────────────────────────────────┐
│ Phase 1    Model Resolution & Analysis            │
│                                                │
│  1.1 Input dispatch:                            │
│      • URL provided        → direct use         │
│      • Precise name        → models.jsonl lookup│
│      • Vague / family name → interactive select │
│        (size tier → specific model)             │
│                                                │
│  1.2 Gated-model auth (Llama, Phi-4, etc.)     │
│                                                │
│  1.3 WebFetch model page, extract params,       │
│      hardware requirements, Python version,     │
│      dependency manifest, loading method        │
└──────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│ Phase 2    Hardware Qualification                │
│            nvidia-smi / free / df → compare      │
│            against requirements → pass/warn/block│
└─────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│ Phase 3    Runtime Provisioning                  │
│            Detect Python + Conda. If absent:     │
│            install Miniconda, configure PATH,    │
│            verify shell integration              │
└─────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│ Phase 4    Environment Construction              │
│            conda create → pip install PyTorch    │
│            (CUDA-matched) → install dependency   │
│            tree → error → diagnose → retry loop  │
└─────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│ Phase 5    Artifact Acquisition                  │
│            huggingface_hub download with resume, │
│            verify file count & total size,       │
│            handle gated-model auth flow          │
└─────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│ Phase 6    Integration Test                      │
│            Minimal inference script → execute →  │
│            validate coherent output → triage     │
│            failures with diagnostic table        │
└─────────────────────────────────────────────────┘
      │
      ▼
   Deployment Report
```

---

## Model Registry & Interactive Selection

The skill ships with a curated registry of 260+ frontier models across 30+ families (`references/models.jsonl`). When the user provides a URL, the registry is bypassed. Otherwise:

### Input → resolution path

| User says | Dispatches to | UX |
|-----------|--------------|-----|
| `https://huggingface.co/Qwen/Qwen2.5-0.5B` | Case A — URL | Direct use, no interaction |
| `Qwen3-8B`, `Llama-3.2-3B` | Case B — Precise name | `AskUserQuestion` if 2-4 near-matches; direct if exact |
| `千问`, `Qwen`, `一个小参数量的Llama` | Case C — Vague/family | Two-step: size tier → specific model |

### Case C in detail

When the user's input is vague (e.g., "deploy a Qwen model"), the skill narrows the selection in two interactive steps:

**Step 1 — Size tier.** The user picks from four tiers, each annotated with hardware guidance:

| Tier | Disk required | Param range | Typical GPU |
|------|-------------|-------------|-------------|
| Lightweight | < 10 GB | 0.5B–4B | 4–8 GB VRAM (RTX 3050/3060) |
| Mid-range | 10–30 GB | 7B–14B | 8–16 GB VRAM (RTX 3080/4070) |
| Heavy | 30–80 GB | 32B–72B | 24 GB+ VRAM (RTX 4090) |
| Flagship | 80+ GB | 100B+ / MoE | Multi-GPU / A100 |

If Phase 2 (hardware check) has already run, the skill highlights the tier that fits the user's GPU.

**Step 2 — Specific model.** Up to 4 models from the chosen tier are presented via `AskUserQuestion`, each showing parameter count and estimated disk footprint. The 4 displayed are the most relevant (latest generation, most downloaded, or best-reviewed).

### Covered model families

`models.jsonl` includes: **Qwen** (2/2.5/3/3.5, Coder, VL, Math, Audio, Omni, QwQ), **Llama** (2/3/3.1/3.2/3.3/4, CodeLlama, Guard), **DeepSeek** (V2/V2.5/V3/V3.2/V4, R1, Coder, Math, VL, Prover), **Mistral** (7B/Mixtral/Nemo/Small/Large, Codestral, Mathstral, Pixtral), **Gemma** (1/2/3, CodeGemma, RecurrentGemma, PaliGemma), **Yi** (1.5, Coder, VL), **InternLM** (2/2.5/3, InternVL 2/2.5/3), **MiniCPM** (3, V), **GLM** (ChatGLM3, GLM-4/5/5.1, Z1, CogVLM2, CogView4), **Baichuan** (2, M1), **Command R** (R/R+/A, Aya 23/Expanse), **Phi** (1.5/2/3/3.5/4), **Falcon** (7B/40B/180B, Mamba), **DBRX**, **Kimi** (K2.5/K2.6), **OLMo** (1/2, OLMoE, Tulu 3), **Granite** (3.0/3.1/3.2, Code), **GPT-OSS**, **SmolLM2/3**, **OpenELM**, **MiniMax**, **Hunyuan**, **Skywork** (13B, Sky-T1), **Orion**, **TeleChat**, **BlueLM**, **XVERSE**, **Aquila**, **Yuan**, **StarCoder2**, **WizardCoder**, **Magicoder**, **CodeGeeX**, **CodeQwen**, **LLaVA** (1.6, NeXT), **Fuyu**, **Idefics**, **Bunny**, **BLIP2**, **InstructBLIP**, **TinyLlama**, **MobileLLM**, **StableLM**, **BLOOM/BLOOMZ**, **Flan-T5**, **Zephyr**, **Nous-Hermes**, **Dolphin**, **OpenHermes**, **BioMistral**, **Meditron**, **BioGPT**, **FinGPT**, **SaulLM**, **Jamba**, **Mamba**, **RWKV**, **Amber**, **CrystalCoder**, **Pythia**, **GPT-NeoX**, **MPT**, **RedPajama**, **Solar**, **Jais**, **SambaLingo**, **SEA-LION**, **EagleX**, **StripedHyena**, **Persimmon**, **NuminaMath**, **Whisper**, **Nemotron**, **Cosmos**, **Arctic**, **Camel**, and more.

---

---

## Hardware Qualification

When requirements are not explicitly stated on the model page, the skill applies conservative heuristics:

| Precision | Disk per 1B params | VRAM per 1B params |
|-----------|-------------------|---------------------|
| FP32      | ~4 GB             | ~4.8 GB             |
| FP16      | ~2 GB             | ~2.4 GB             |
| INT8      | ~1 GB             | ~1.2 GB             |
| INT4      | ~0.5 GB           | ~0.6 GB             |

System RAM is estimated at 1.5× VRAM for KV cache and activation memory overhead.

If the host fails qualification, the skill presents a **blocking warning** with concrete mitigations (quantization, CPU offloading via `device_map="auto"`, or cloud fallback). Execution does not proceed past this gate without explicit user confirmation.

---

## Dependency Resolution Strategy

PyTorch is installed **first** and pinned to the host's CUDA driver version (detected via `nvidia-smi`):

```
CUDA 12.x → pip install torch --index-url .../cu121
CUDA 11.x → pip install torch --index-url .../cu118
No GPU    → pip install torch --index-url .../cpu
```

Subsequent dependencies are installed per the model's manifest. On failure, the skill applies a **diagnose-fix-retry loop**:

| Failure mode | Root cause | Resolution |
|---|---|---|
| `error: Microsoft Visual C++ 14.0 is required` | Missing build tools | Install Visual Studio Build Tools (Windows) or `build-essential` (Linux) |
| `No matching distribution found for torch==x.x.x` | CUDA version mismatch | Downgrade PyTorch index to `cu118` |
| `ERROR: Cannot uninstall X` | Dependency conflict | Destroy and recreate Conda environment |
| `fatal error: xxx.h: No such file` | Missing system headers | `apt install python3-dev` / `brew install python@3.10` |
| `gcc: internal compiler error: Killed` | OOM during compilation | `export MAX_JOBS=1` to serialize build |
| `ImportError: libcublas.so.11` | CUDA toolkit not on LD_LIBRARY_PATH | Verify `$CONDA_PREFIX/lib` in linker path |

The loop repeats until **all** packages report successful installation.

---

## Output Artifacts

Upon successful deployment, the user receives:

```
=== Model deployment successful! ===

  Model:       <model-name>
  Environment: conda env "<env-name>" (Python <version>)
  Location:    <download-path>  (<disk-usage>)
  GPU:         <gpu-name> (cuda:<id>)

  Test prompt:  <input>
  Test output:  <output>

To use this model later:
  conda activate <env-name>
  python your_script.py
```

Additionally, an inference test script is placed alongside the model weights for future validation.

---

## Platform Support

| Platform | GPU backend | Notes |
|----------|------------|-------|
| Windows (x64) | CUDA (NVIDIA) | Visual Studio Build Tools may be required for packages built from source |
| Linux (x64) | CUDA (NVIDIA) | `build-essential`, `python3-dev` prerequisites for compiled wheels |
| macOS (Intel) | CPU only | No CUDA support |
| macOS (Apple Silicon) | MPS (Metal) | `torch.backends.mps.is_available()`, limited op coverage |

---

## Constraints & Caveats

- **Network:** Model downloads range from ~0.5 GB to 400+ GB. A stable connection is required; downloads are resumable.
- **Disk:** The target drive must have free space ≥ 1.5× the estimated model size (download + extraction overhead).
- **Authentication:** Gated models (e.g., Llama, Mistral) require a HuggingFace access token with read permissions.
- **Python version:** The skill defaults to Python 3.10 when the model page does not specify a version, as it provides the widest ML ecosystem compatibility.
- **Conda installation:** If Miniconda is not detected, the skill will install it (~500 MB) after user confirmation.
