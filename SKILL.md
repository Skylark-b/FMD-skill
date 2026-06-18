---
name: deploy-frontier-model
description: 'This skill should be used when the user asks to "deploy a frontier model", "download a model", "run a model locally", "deploy a large language model", "set up a model", "install a model", or mentions model deployment. It handles the full workflow: fetching model requirements from a URL, checking local hardware/environment, setting up conda, downloading the model, building a dedicated conda environment, installing all dependencies, and running a smoke test to confirm success.'
version: 1.0.0
---

# Deploy Frontier Model

This skill handles end-to-end deployment of frontier AI models (LLMs, vision models, etc.) — from a model page URL to a locally running, tested instance.

## Overview

Deploying frontier models requires coordinated steps: understanding model requirements, verifying hardware/software prerequisites, building an isolated Python environment, downloading large model weights, and validating the deployment. This skill walks through every step and fixes issues as they arise.

---

## Phase 1 — Gather model information

### 1.1 Determine the model URL

The user may provide either a **URL**, a **precise model name**, or a **vague model name / family name**.

#### Case A — URL provided

Use the URL directly and proceed to Phase 1.2.

#### Case B — Precise model name (exact or near-exact match)

If the user says something like `Qwen3-8B`, `Llama-3.2-3B`, `DeepSeek-R1-Distill-Qwen-7B`:

1. Read `references/models.jsonl` from this skill's directory (`.claude/skills/deploy-frontier-model/references/models.jsonl`).
2. Search case-insensitively. If exactly one match → use it directly.
3. If a few near-matches (2-4) → use `AskUserQuestion` to let the user pick. Present each option with its parameter count as context:

   ```
   "Which model do you want?" → options:
   - Qwen3-4B (4B params, ~8 GB disk)
   - Qwen3-8B (8B params, ~16 GB disk)
   - Qwen3-32B (32B params, ~64 GB disk)
   ```

#### Case C — Vague / family name (many matches)

If the user says `千问`, `Qwen`, `Llama`, `部署一个千问模型`, `一个小参数量的Qwen`:

**Step 1 — Narrow by size tier.** Present the user with size tiers (use `AskUserQuestion`):

| Tier | Disk needed | Example models | Suitable GPU |
|------|------------|----------------|-------------|
| Lightweight | < 10 GB | 0.5B–4B params | 4-8 GB VRAM (RTX 3050/3060) |
| Mid-range | 10–30 GB | 7B–14B params | 8-16 GB VRAM (RTX 3080/4070) |
| Heavy | 30–80 GB | 32B–72B params | 24 GB+ VRAM (RTX 4090) |
| Flagship | 80+ GB | 100B+ params / MoE | Multi-GPU / A100 |

Help the user choose by noting their hardware (already detected in Phase 2 — if Phase 2 runs first, use it; otherwise mention that lightweight models will run on most machines).

**Step 2 — List matching models in that tier.** After the user picks a tier, present the specific models from `models.jsonl` that fall into that size range (use `AskUserQuestion`, max 4 options). Example for "Lightweight" tier matching "Qwen":

```
Which Qwen model? (Lightweight tier)
- Qwen2.5-0.5B (0.5B, ~1 GB)
- Qwen2.5-1.5B (1.5B, ~3 GB)
- Qwen3-0.6B (0.6B, ~1.2 GB)
- Qwen3-4B (4B, ~8 GB)
```

If more than 4 models match the tier, pick the 4 most relevant (prefer latest generation, most downloaded, or best-reviewed). Add the model size estimate to each option to help the user decide.

**Step 3 — Use the selected model's URL** and proceed to Phase 1.2.

#### Fallback

If the model name is not found in `models.jsonl` at all, inform the user and ask for a direct URL.

### 1.2 Resolve gated-model authentication

Check if the model requires a HuggingFace access token (gated models include: all `meta-llama/*`, `microsoft/Phi-4*`, `mistralai/Mistral-Large*`, `moonshotai/*`). If so, guide the user:
1. Go to https://huggingface.co/settings/tokens
2. Create a token with read permissions
3. Run `huggingface-cli login` in the conda environment and paste the token

### 1.3 Fetch and parse the model page

Use `WebFetch` to retrieve the model page. Extract the following:

| What to extract | Where to look |
|---|---|
| Model name | Page title, repo name, or model card header |
| Model size (parameters) | Model card, README, config.json |
| Disk space needed | Model card ("weights", "disk", "storage") — if unspecified, estimate ~2 GB per 1B params for FP16 |
| GPU VRAM needed | Model card ("GPU memory", "VRAM") — note quantization options |
| System RAM needed | Model card ("RAM", "memory") — if not listed, estimate 1.5x VRAM |
| Python version required | `requirements.txt`, `setup.py`, `pyproject.toml`, or documentation |
| Key dependencies | `requirements.txt`, `environment.yml`, or install instructions in the README |
| Model loading method | Transformers, llama.cpp, vLLM, Ollama, etc. |
| Additional notes | Auth tokens, build steps, CUDA version requirements, etc. |

If hardware requirements are not explicitly listed, estimate:
- **Disk:** 2 GB per 1B params (FP16), 1 GB per 1B (INT8), 0.5 GB per 1B (INT4)
- **GPU VRAM:** disk estimate + 20% inference overhead
- **System RAM:** 32 GB minimum for models >7B; 64 GB+ for models >30B

---

## Phase 2 — Hardware check

### 2.1 Inspect local hardware

**Windows (PowerShell / bash):**
```bash
nvidia-smi 2>/dev/null || echo "NO_NVIDIA_GPU"
wmic ComputerSystem get TotalPhysicalMemory 2>/dev/null || free -h 2>/dev/null || system_profiler SPHardwareDataType 2>/dev/null
wmic LogicalDisk where "DeviceID='C:'" get FreeSpace 2>/dev/null || df -h / 2>/dev/null
```

**Linux:**
```bash
nvidia-smi 2>/dev/null || echo "NO_NVIDIA_GPU"
free -h
df -h /home
```

**macOS:**
```bash
system_profiler SPDisplaysDataType 2>/dev/null | grep -i "chip\|vram\|metal"
sysctl hw.memsize
df -h /Users
```

### 2.2 Report findings

Present a comparison table:

```
| Requirement | Minimum Needed | Your Machine | Status |
|------------|---------------|--------------|--------|
| GPU VRAM   | 16 GB         | 24 GB        | OK     |
| System RAM | 32 GB         | 64 GB        | OK     |
| Disk Space | 50 GB         | 120 GB free  | OK     |
| GPU Compute| CUDA 7.5+     | CUDA 8.0     | OK     |
```

If any requirement is not met, **warn the user clearly** and suggest mitigations (quantization, CPU-only mode, cloud options). Do not proceed without user confirmation when hardware is insufficient.

---

## Phase 3 — Environment check and setup

### 3.1 Check for Python

```bash
python --version 2>/dev/null || python3 --version 2>/dev/null || echo "NO_PYTHON"
```

### 3.2 Check for Conda

```bash
conda --version 2>/dev/null || echo "NO_CONDA"
```

Also check for Miniconda/Anaconda installations:

```bash
# Windows
where conda 2>/dev/null || dir "%USERPROFILE%\Miniconda3" 2>/dev/null || dir "%USERPROFILE%\Anaconda3" 2>/dev/null

# Linux/macOS
which conda 2>/dev/null || ls ~/miniconda3/bin/conda 2>/dev/null || ls ~/anaconda3/bin/conda 2>/dev/null
```

### 3.3 If Python is missing — install it

**Windows:**
1. Download the Python installer from `https://www.python.org/downloads/` (prefer the version identified in Phase 1, otherwise default to Python 3.10)
2. Run the installer with "Add Python to PATH" checked
3. Verify: `python --version`

**Linux:**
```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv
```

**macOS:**
```bash
brew install python@3.10
```

### 3.4 If Conda is missing — install Miniconda

**Windows:**
```bash
curl -o "%TEMP%\miniconda.exe" https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
"%TEMP%\miniconda.exe" /S /D=%USERPROFILE%\Miniconda3
```

**Linux:**
```bash
curl -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
```

**macOS (Intel):**
```bash
curl -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init zsh
```

**macOS (Apple Silicon):**
```bash
curl -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init zsh
```

After installation, refresh the shell and verify:
```bash
conda --version
```

---

## Phase 4 — Create the model environment

### 4.1 Determine the Python version

Use the Python version from Phase 1. If the model page does not specify one, use:
- **Python 3.10** — most stable, widely compatible version for ML workloads

### 4.2 Create a conda environment

Use the model name (lowercase, hyphens replacing spaces) as the environment name:

```bash
conda create -n <model-name> python=<version> -y
conda activate <model-name>
```

If `conda activate` does not work in the current shell:

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate <model-name>
```

On Windows:
```bash
call %USERPROFILE%\Miniconda3\Scripts\activate.bat <model-name>
```

### 4.3 Install PyTorch first

PyTorch must be installed before other dependencies to ensure CUDA compatibility:

```bash
# CUDA 12.1 (most common)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 (compatibility)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU-only (fallback)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Determine the CUDA version with `nvidia-smi` (top-right corner). Choose the highest compatible PyTorch index.

### 4.4 Install model dependencies

Based on the model page findings from Phase 1. Common patterns:

**HuggingFace Transformers:**
```bash
pip install transformers accelerate sentencepiece protobuf
```

**llama.cpp / GGUF:**
```bash
pip install llama-cpp-python
```

**vLLM:**
```bash
pip install vllm
```

**General ML stack:**
```bash
pip install numpy scipy einops safetensors peft bitsandbytes
```

If the model page references a `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4.5 Dependency error handling

If any `pip install` command fails:

1. **Read the error carefully** — identify which package failed and why
2. **Common fixes:**
   - Build errors → install system tools: `sudo apt install build-essential python3-dev` (Linux) or Visual Studio Build Tools (Windows)
   - CUDA mismatch → try `pip install <package> --index-url https://download.pytorch.org/whl/cu118`
   - Dependency conflicts → create a fresh conda environment and retry
   - Missing system libraries → install via apt/brew before retrying pip
   - Out-of-memory during build → `export MAX_JOBS=1` to limit parallelism
3. **Apply the fix and retry the failed command**
4. **Repeat until ALL dependencies install without error**

Do not proceed to Phase 5 until every dependency is confirmed installed.

---

## Phase 5 — Download the model

### 5.1 Determine the download method

| Model format | Download method |
|---|---|
| HuggingFace (safetensors/pytorch) | `huggingface_hub.snapshot_download()` or `git lfs` |
| GGUF files | `huggingface_hub.hf_hub_download()` for single files, or direct URL |
| Pickle/torch save | `torch.load()` or direct download |

### 5.2 Download model weights

Create a models directory and download:

```bash
mkdir -p ~/models/<model-name>
```

```python
from huggingface_hub import snapshot_download, hf_hub_download

# Full model repo
snapshot_download(
    repo_id="<org>/<model-name>",
    local_dir=os.path.expanduser("~/models/<model-name>"),
    ignore_patterns=["*.md", "*.txt"],
)

# Single GGUF file
hf_hub_download(
    repo_id="<org>/<model-name>",
    filename="<filename>.gguf",
    local_dir=os.path.expanduser("~/models/<model-name>"),
)
```

If the model requires authentication (gated model), instruct the user:
1. Go to https://huggingface.co/settings/tokens
2. Create a token with read permissions
3. Run `huggingface-cli login` and paste the token

### 5.3 Verify the download

```bash
ls -lh ~/models/<model-name>/
du -sh ~/models/<model-name>/
```

---

## Phase 6 — Smoke test

### 6.1 Find the simplest inference example

Search the model page for "Quick Start", "Usage", or "Inference". Write a minimal test script.

**Transformers text generation:**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "~/models/<model-name>"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**GGUF / llama-cpp:**
```python
from llama_cpp import Llama

llm = Llama(model_path="~/models/<model-name>/<file>.gguf", n_ctx=2048)
output = llm("Hello, my name is", max_tokens=20)
print(output["choices"][0]["text"])
```

### 6.2 Run the test

```bash
python test_inference.py
```

### 6.3 Troubleshoot failures

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` | Model exceeds GPU VRAM | Use quantization (`load_in_8bit=True`, `load_in_4bit=True`) or CPU offloading |
| `ModuleNotFoundError` | Missing dependency | Return to Phase 4, install the missing package |
| `config.json not found` | Incomplete download or wrong path | Check the directory, re-download if needed |
| `tokenizer not found` | Tokenizer files missing | Ensure `tokenizer.*` files are in the model directory |
| `CUDA error: no kernel image` | CUDA compute capability mismatch | Install PyTorch matching your GPU's compute capability |
| Slow inference | Running on CPU | Verify `torch.cuda.is_available()`, check `device_map` |

### 6.4 Confirm success

Once the test runs without errors and produces coherent output:

```
=== Model deployment successful! ===

  Model:       <model-name>
  Environment: conda env "<env-name>" (Python <version>)
  Location:    ~/models/<model-name>/
  Disk used:   <size>
  Test output: <snippet>

To use this model later:
  conda activate <env-name>
  python your_script.py
```

---

## Critical rules

1. **Never proceed without user confirmation if hardware is insufficient** — warn clearly and offer mitigations
2. **Always install PyTorch first**, before any other ML packages
3. **Fix every dependency error before moving on** — a broken env causes cascading failures
4. **Always use conda environments for isolation** — never `pip install` model dependencies globally
5. **Verify the download** before declaring success — check file sizes on disk
6. **The test must produce coherent output** — loading without errors but producing garbage is not success
7. **Own the full process** — do not ask the user to fix problems themselves

---

## Platform-specific notes

### Windows
- Use `%USERPROFILE%` instead of `~` in paths
- CUDA packages may require Visual Studio Build Tools (e.g., for `bitsandbytes`)
- Long paths: use `\\?\` prefix or enable long path support in the registry
- `conda activate` may need `call conda.bat activate` in some shells

### macOS
- No CUDA — PyTorch MPS (Metal) acceleration is available for some operations
- Apple Silicon: check `torch.backends.mps.is_available()` for GPU access
- Some models may need `device_map="mps"` or fall back to CPU

### Linux
- May need `sudo apt install build-essential python3-dev` for source-compiled packages
- Docker-based deployment requires the NVIDIA Container Toolkit
- Check `nvidia-smi` for the CUDA version before choosing the PyTorch index URL
