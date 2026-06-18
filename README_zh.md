# deploy-frontier-model

> 前沿 AI 模型端到端本地部署——提供一个 URL、一个模型名甚至一个模型家族，全程自动化交付可运行、已验证的实例。

---

## 解决了什么问题

前沿模型的本地部署横跨多个工程领域，任一环节出错都会引发级联故障：

1. **需求解析** —— 从模型卡与配置文件中提取硬件先决条件（显存、内存、磁盘）、Python 版本约束及依赖树。
2. **平台鉴定** —— 将需求与本机 GPU 算力、可用内存、剩余磁盘空间逐项比对。
3. **运行时供应** —— 检测或安装 Python 工具链（CPython、Conda/Miniconda），配置环境变量，完成 Shell 集成。
4. **环境构建** —— 创建隔离的 Conda 环境，安装匹配本机 CUDA 驱动的 PyTorch，解析并安装完整依赖 DAG。
5. **制品获取** —— 断点续传下载模型权重（常达数十 GB），校验完整性，处理受限仓库的身份认证。
6. **集成测试** —— 编写最小推理脚本，执行冒烟测试，排查各类故障（OOM、tokenizer 缺失、CUDA kernel 不匹配、dtype 不兼容）。

从零开始排查往往耗时数小时。此 Skill 将上述流程全自动化。

---

## 触发方式

当用户消息包含以下任一意图时，Skill 自动激活：

| 语言 | 触发短语 |
|------|---------|
| 英文 | `deploy a frontier model`, `download a model`, `run a model locally`, `deploy a large language model`, `set up a model`, `install a model` |
| 中文 | `部署前沿模型`, `下载模型`, `本地运行模型`, `安装模型`, `配置模型环境` |

直接发送 HuggingFace URL、模型代码名或模型家族名并附带部署意图时也会触发。

**调用示例：**

```
# 通过 URL 部署
帮我把 https://huggingface.co/Qwen/Qwen2.5-0.5B 部署到 D:\models

# 通过精确模型名部署
帮我部署 Qwen3-8B

# 通过模糊家族名部署 —— skill 会交互式缩小选择范围
帮我下载一个轻量级的千问模型
```

---

## 流水线架构

```
用户输入 (URL / 名称 / 家族)
      │
      ▼
┌──────────────────────────────────────────────────┐
│ 阶段一    模型解析与分析                           │
│                                                │
│  1.1 输入分流：                                 │
│      • 提供 URL            → 直接使用            │
│      • 精确模型名           → models.jsonl 查找   │
│      • 模糊 / 家族名        → 交互式选择          │
│        (规模层级 → 具体模型)                     │
│                                                │
│  1.2 受限模型鉴权（Llama、Phi-4 等）            │
│                                                │
│  1.3 WebFetch 抓取模型页面，提取参数量、          │
│      硬件需求、Python 版本、依赖清单、           │
│      模型加载方式                                │
└──────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│ 阶段二    硬件鉴定                                │
│           nvidia-smi / free / df → 需求对比      │
│           → 通过 / 警告 / 阻断                   │
└─────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│ 阶段三    运行时供应                              │
│           检测 Python + Conda。若缺失：           │
│           安装 Miniconda，配置 PATH，             │
│           验证 Shell 集成                         │
└─────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│ 阶段四    环境构建                                │
│           conda create → pip install PyTorch     │
│           (CUDA 适配) → 安装依赖树               │
│           → 报错 → 诊断 → 重试循环               │
└─────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│ 阶段五    制品获取                                │
│           huggingface_hub 断点续传下载，          │
│           校验文件数量与总体积，                  │
│           处理受限仓库的 Token 认证流程           │
└─────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│ 阶段六    集成测试                                │
│           最小推理脚本 → 执行 → 验证输出通畅      │
│           → 按诊断表排查故障                      │
└─────────────────────────────────────────────────┘
      │
      ▼
   部署报告
```

---

## 模型注册表与交互式选择

Skill 内置了一个覆盖 30+ 个家族、260+ 个前沿模型的注册表（`references/models.jsonl`）。当用户提供 URL 时直接使用；提供名称时则按以下路径解析：

### 输入 → 解析路径

| 用户输入 | 分发路径 | 交互方式 |
|---------|---------|---------|
| `https://huggingface.co/Qwen/Qwen2.5-0.5B` | 情况 A — URL | 直接使用，无需交互 |
| `Qwen3-8B`、`Llama-3.2-3B` | 情况 B — 精确名称 | 2-4 个近似匹配时 `AskUserQuestion` 弹选择框；精确命中则直接使用 |
| `千问`、`Qwen`、`一个小参数量的Llama` | 情况 C — 模糊/家族名 | 两步选择：规模层级 → 具体模型 |

### 情况 C 详解

用户输入模糊（如"部署一个千问模型"），Skill 分两步交互缩小范围：

**第一步 —— 规模层级。** 用户从四个层级中选择，每个层级标注硬件参考：

| 层级 | 磁盘需求 | 参数范围 | 适用显卡 |
|------|---------|---------|---------|
| 轻量级 | < 10 GB | 0.5B–4B | 4–8 GB 显存 (RTX 3050/3060) |
| 中量级 | 10–30 GB | 7B–14B | 8–16 GB 显存 (RTX 3080/4070) |
| 重量级 | 30–80 GB | 32B–72B | 24 GB+ 显存 (RTX 4090) |
| 旗舰级 | 80+ GB | 100B+ / MoE | 多卡 / A100 |

如果阶段二（硬件检查）已完成，Skill 会高亮适合用户 GPU 的层级。

**第二步 —— 具体模型。** 从选定层级中通过 `AskUserQuestion` 展示最多 4 个模型，每个标注参数量与磁盘估算。优先展示最新一代、下载量最高或口碑最佳的模型。

### 覆盖的模型家族

`models.jsonl` 涵盖：**Qwen**（2.5/3/3.5、Coder、VL、QwQ）、**Llama**（3.1–4）、**DeepSeek**（V2.5/V3/V3.2/V4、R1、Coder）、**Mistral/Mixtral/Ministral/Pixtral**、**Gemma 2/3**、**Yi**、**InternLM/InternVL**、**ChatGLM3/GLM-4/GLM-5/GLM-Z1**、**Baichuan**、**Command R**、**Phi**（3/3.5/4）、**Falcon**、**DBRX**、**Kimi K2.5/K2.6**、**OLMo/OLMoE**、**SmolLM**、**Hunyuan**、**GPT-OSS** 等。

---

---

## 硬件鉴定策略

当模型页面未明确标注硬件需求时，Skill 采用以下保守估算：

| 精度   | 磁盘占用 / 1B 参数 | 显存占用 / 1B 参数 |
|--------|-------------------|-------------------|
| FP32   | ~4 GB             | ~4.8 GB           |
| FP16   | ~2 GB             | ~2.4 GB           |
| INT8   | ~1 GB             | ~1.2 GB           |
| INT4   | ~0.5 GB           | ~0.6 GB           |

系统内存按显存的 1.5 倍估算，预留 KV Cache 与激活值开销。

若本机不满足需求，Skill 会给出**阻断性警告**并提供具体缓解方案（量化、`device_map="auto"` CPU 卸载、云端兜底）。未获用户明确确认前不会继续执行。

---

## 依赖解决策略

PyTorch **优先安装**，且严格绑定本机 CUDA 驱动版本（通过 `nvidia-smi` 检测）：

```
CUDA 12.x → pip install torch --index-url .../cu121
CUDA 11.x → pip install torch --index-url .../cu118
无 GPU    → pip install torch --index-url .../cpu
```

后续依赖按模型清单安装。失败时进入**诊断-修复-重试**循环：

| 报错 | 根因 | 解决方案 |
|---|---|---|
| `error: Microsoft Visual C++ 14.0 is required` | 缺少构建工具 | 安装 Visual Studio Build Tools (Windows) 或 `build-essential` (Linux) |
| `No matching distribution found for torch==x.x.x` | CUDA 版本不匹配 | 降级 PyTorch 索引至 `cu118` |
| `ERROR: Cannot uninstall X` | 依赖冲突 | 销毁并重建 Conda 环境 |
| `fatal error: xxx.h: No such file` | 缺少系统头文件 | `apt install python3-dev` / `brew install python@3.10` |
| `gcc: internal compiler error: Killed` | 编译过程内存不足 | `export MAX_JOBS=1` 串行编译 |
| `ImportError: libcublas.so.11` | CUDA 工具链未在链接路径 | 确认 `$CONDA_PREFIX/lib` 在 `LD_LIBRARY_PATH` 中 |

循环直至**全部**包报告安装成功。

---

## 部署产出

部署成功后，用户将获得：

```
=== 模型部署成功！ ===

  模型:        <模型名称>
  虚拟环境:    conda env "<环境名>" (Python <版本>)
  本地路径:    <下载路径>  (<磁盘占用>)
  GPU:         <GPU 名称> (cuda:<id>)

  测试输入:    <prompt>
  测试输出:    <模型回复>

后续使用方式:
  conda activate <环境名>
  python your_script.py
```

同时，一份推理测试脚本会置于模型权重同级目录，方便后续验证。

---

## 平台支持

| 平台 | GPU 后端 | 备注 |
|------|---------|------|
| Windows (x64) | CUDA (NVIDIA) | 源码编译的包可能需要 Visual Studio Build Tools |
| Linux (x64) | CUDA (NVIDIA) | 编译依赖: `build-essential`, `python3-dev` |
| macOS (Intel) | 仅 CPU | 无 CUDA 支持 |
| macOS (Apple Silicon) | MPS (Metal) | `torch.backends.mps.is_available()`，部分算子不支持 |

---

## 约束与注意事项

- **网络：** 模型下载量从 ~0.5 GB 到 400+ GB 不等，需稳定网络连接；支持断点续传。
- **磁盘：** 目标盘符可用空间需 ≥ 模型预估体积的 1.5 倍（含下载与解压开销）。
- **认证：** 受限模型（如 Llama、Mistral）需提供具有读取权限的 HuggingFace Access Token。
- **Python 版本：** 模型页面未指定时默认使用 Python 3.10，该版本拥有最广泛的 ML 生态兼容性。
- **Conda 安装：** 若未检测到 Miniconda，Skill 会在征得用户确认后安装（约 500 MB）。
