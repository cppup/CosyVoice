# CosyVoice3 多语言 TTS 系统化训练指南

> 本指南面向使用 CosyVoice3（Fun-CosyVoice3-0.5B）进行多语言 TTS 训练的工程师与研究人员，系统覆盖：多机多卡训练、训练加速、推理加速、端点续训、训练最佳实践、各 Example 场景解析及相关论文整合。

---

## 目录

1. [系统架构速览](#1-系统架构速览)
2. [各 Example 特性与选型指南](#2-各-example-特性与选型指南)
3. [数据准备流程](#3-数据准备流程)
4. [多机多卡训练](#4-多机多卡训练)
5. [训练加速技巧](#5-训练加速技巧)
6. [端点续训（Checkpoint Resume）](#6-端点续训checkpoint-resume)
7. [推理加速](#7-推理加速)
8. [多语言训练最佳实践](#8-多语言训练最佳实践)
9. [强化学习后处理（DPO / GRPO）](#9-强化学习后处理dpo--grpo)
10. [训练监控与调试](#10-训练监控与调试)
11. [高频操作速查表](#11-高频操作速查表)
12. [相关论文整合](#12-相关论文整合)

---

## 1. 系统架构速览

CosyVoice 系列 TTS 系统由三个串联模块组成：

```
文本 → [LLM (Qwen2)] → 离散语音 Token → [Flow (CausalMaskedDiffWithDiT)] → Mel → [HiFiGAN/HiFT] → 波形
```

| 模块 | CosyVoice 1.0 | CosyVoice 2.0 | CosyVoice 3.0 |
|------|---------------|---------------|---------------|
| LLM | TransformerLM（自研） | CosyVoice2LM（Qwen2） | CosyVoice3LM（Qwen2 + Instruct） |
| Flow | MaskedDiffWithXvec | CausalMaskedDiffWithDiT | CausalMaskedDiffWithDiT（改进） |
| Vocoder | HiFi-GAN | CausalHiFTGenerator | CausalHiFTGenerator |
| 采样率 | 22050 Hz | 24000 Hz | 24000 Hz |
| Token 帧率 | 50 Hz | 25 Hz | 25 Hz |
| 多语言 | 单语（中文为主） | 支持 | 原生多语言，9 语言 + 18+ 方言 |
| 流式推理 | ❌ | ✅ | ✅（≤150ms 首包） |
| 语言控制 | ❌ | ❌ | ✅（Instruct 机制） |

**训练入口**：`cosyvoice/bin/train.py`
**训练工具**：`cosyvoice/utils/train_utils.py`、`cosyvoice/utils/executor.py`

---

## 2. 各 Example 特性与选型指南

本库提供以下五套 Recipe，针对不同目标场景：

### 2.1 `examples/magicdata-read/cosyvoice`

| 属性 | 详情 |
|------|------|
| 模型版本 | CosyVoice 1.0（`CosyVoice-300M`） |
| 数据集 | MagicData-READ（中文阅读语音，约 755 小时） |
| 语言 | 中文 |
| 适用场景 | **中文单语 SFT**；理解 CosyVoice 1.0 架构；基线实验 |
| Tokenizer | speech_tokenizer_v1（50 Hz） |
| 特点 | 结构最简单，适合入门；不支持 Instruct 控制 |

### 2.2 `examples/libritts/cosyvoice`

| 属性 | 详情 |
|------|------|
| 模型版本 | CosyVoice 1.0（`CosyVoice-300M`） |
| 数据集 | LibriTTS（英文，约 585 小时） |
| 语言 | 英文 |
| 适用场景 | **英文单语 SFT**；验证 CosyVoice 1.0 英文能力 |
| Tokenizer | speech_tokenizer_v1（50 Hz） |
| 特点 | 数据量较大（100+360+500 小时），适合资源充足场景 |

### 2.3 `examples/libritts/cosyvoice2`（**推荐：多语 SFT 基础**）

| 属性 | 详情 |
|------|------|
| 模型版本 | CosyVoice 2.0（`CosyVoice2-0.5B`） |
| 数据集 | LibriTTS（英文） |
| 语言 | 英文（可扩展至多语） |
| 适用场景 | **英文/多语 SFT**；DPO 偏好优化（`run_dpo.sh`）；推理加速实验 |
| Tokenizer | speech_tokenizer_v3（25 Hz） |
| 特点 | 支持 DPO（`run_dpo.sh`）；4 卡 DDP 为默认配置；可切换 DeepSpeed |

### 2.4 `examples/libritts/cosyvoice3`（**推荐：多语言训练首选**）

| 属性 | 详情 |
|------|------|
| 模型版本 | CosyVoice 3.0（`Fun-CosyVoice3-0.5B`） |
| 数据集 | LibriTTS（英文，可替换为多语混合数据） |
| 语言 | 多语言（原生支持 9 语言 + 18+ 方言） |
| 适用场景 | **多语言 TTS 训练**；Instruct 控制（语言/方言/情感/风格） |
| Tokenizer | speech_tokenizer_v3（25 Hz） |
| 特点 | 在 `prepare_data.py` 中通过 `--instruct` 注入语言控制 Prompt；`mix_ratio: [5, 15]` 控制文本/语音比例 |

**关键差异**：CosyVoice3 的 `prepare_data.py` 接受 `--instruct` 参数，将控制指令（如语言、情感）嵌入序列前缀：

```bash
python local/prepare_data.py \
  --src_dir $data_dir/LibriTTS/$x \
  --des_dir data/$x \
  --instruct "You are a helpful assistant.<|endofprompt|>"
```

### 2.5 `examples/grpo/cosyvoice2`（**推荐：质量提升后训练**）

| 属性 | 详情 |
|------|------|
| 模型版本 | CosyVoice 2.0（LLM 部分，HuggingFace 格式） |
| 框架 | veRL（GRPO/PPO） + vLLM + Triton ASR |
| 适用场景 | **强化学习后训练**，以 WER/CER 为奖励信号提升内容一致性 |
| 效果 | CosyVoice3 zero_shot_zh CER: 4.08% → 3.36%（GRPO） |
| 特点 | 需要独立 Docker 环境；奖励函数为 ASR 转写误差率 |

### 选型决策树

```
目标是多语言TTS训练？
├── 是 → 使用 cosyvoice3 recipe（examples/libritts/cosyvoice3）
│         训练完成后可选做 GRPO（examples/grpo/cosyvoice2）
└── 否 → 单语（中文）→ magicdata-read/cosyvoice
         单语（英文）→ libritts/cosyvoice2（含 DPO 选项）
         探索 v1 架构 → libritts/cosyvoice 或 magicdata-read/cosyvoice
```

---

## 3. 数据准备流程

### 3.1 标准流程（以 cosyvoice3 为例）

```bash
# Stage 0: 原始数据整理，生成 wav.scp / text / utt2spk / spk2utt
python local/prepare_data.py \
  --src_dir $data_dir \
  --des_dir data/$x \
  --instruct "You are a helpful assistant.<|endofprompt|>"  # CosyVoice3 专属

# Stage 1: 提取 CAMPPlus 说话人向量
tools/extract_embedding.py \
  --dir data/$x \
  --onnx_path $pretrained_model_dir/campplus.onnx

# Stage 2: 提取离散语音 Token（可跳过，使用在线提取；但离线提取可大幅提升训练速度）
tools/extract_speech_token.py \
  --dir data/$x \
  --onnx_path $pretrained_model_dir/speech_tokenizer_v3.onnx

# Stage 3: 打包为 Parquet 格式（分布式训练的高效数据格式）
tools/make_parquet_list.py \
  --num_utts_per_parquet 1000 \
  --num_processes 10 \
  --src_dir data/$x \
  --des_dir data/$x/parquet
```

> **注意**：注释中写明 "embedding/token extraction is not necessary now as we support online feature extraction, **but training speed will be influenced**"。在大规模训练中，**强烈建议离线预提取**以避免 I/O 瓶颈。

### 3.2 多语言数据混合

对于多语训练，将不同语言的 `data.list` 拼接即可：

```bash
# 中英混合
cat data/zh/parquet/data.list \
    data/en/parquet/data.list \
    data/ja/parquet/data.list > data/train.data.list
```

CosyVoice3 通过 Instruct 前缀区分语言，**无需**额外的语言 ID 嵌入层。

### 3.3 数据格式说明

Parquet 文件的每行包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `utt` | str | 语音唯一 ID |
| `audio_data` | bytes | 原始音频二进制 |
| `wav` | str | 音频路径 |
| `text` | str | 原始文本 |
| `spk` | str | 说话人 ID |
| `utt_embedding` | bytes | 说话人向量（pt） |
| `spk_embedding` | bytes | 说话人平均向量（pt） |
| `speech_token` | bytes | 离散语音 Token 序列（pt） |
| `instruct` | str | CosyVoice3 控制指令（可选） |
| `reject_speech_token` | bytes | DPO 负样本 Token（仅 DPO） |

---

## 4. 多机多卡训练

### 4.1 单机多卡（标准配置）

使用 PyTorch `torchrun`（DDP）：

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986

torchrun \
  --nnodes=1 \
  --nproc_per_node=$num_gpus \
  --rdzv_id=$job_id \
  --rdzv_backend="c10d" \
  --rdzv_endpoint="localhost:1234" \
  cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  [其他参数...]
```

### 4.2 多机多卡（Multi-node DDP）

```bash
# 主节点（MASTER_ADDR）
MASTER_NODE_IP="10.0.0.1"
NUM_NODES=4
NUM_GPUS_PER_NODE=8
JOB_ID=2024

torchrun \
  --nnodes=$NUM_NODES \
  --nproc_per_node=$NUM_GPUS_PER_NODE \
  --rdzv_id=$JOB_ID \
  --rdzv_backend="c10d" \
  --rdzv_endpoint="${MASTER_NODE_IP}:29500" \
  cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --ddp.dist_backend nccl \
  [其他参数...]
```

所有节点执行相同命令；`--rdzv_endpoint` 指向主节点 IP。

> **Tip**: 多机训练时，确保所有节点的数据路径（`--train_data`、`--cv_data`）一致可访问（如 NFS 挂载或对象存储）。

### 4.3 多机多卡（DeepSpeed）

DeepSpeed 适合大模型（GPU 内存不足时）：

```bash
torchrun \
  --nnodes=$NUM_NODES \
  --nproc_per_node=$NUM_GPUS_PER_NODE \
  --rdzv_id=$JOB_ID \
  --rdzv_backend="c10d" \
  --rdzv_endpoint="${MASTER_NODE_IP}:29500" \
  cosyvoice/bin/train.py \
  --train_engine deepspeed \
  --deepspeed_config conf/ds_stage2.json \
  --deepspeed.save_states model+optimizer \
  [其他参数...]
```

`conf/ds_stage2.json` 中的关键字段：

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 5,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "none" },
    "overlap_comm": true,
    "reduce_scatter": true,
    "contiguous_gradients": true
  }
}
```

**ZeRO Stage 选择建议**：

| Stage | 适用场景 | 内存节省 | 通信开销 |
|-------|----------|----------|----------|
| Stage 1 | 参数量 < 1B，GPU 内存充足 | 低 | 低 |
| Stage 2（默认） | 参数量 ~0.5B，当前推荐 | 中 | 中 |
| Stage 3 | 参数量 > 2B，极端内存不足 | 高 | 高 |

### 4.4 DDP vs DeepSpeed 对比

| 维度 | torch_ddp | deepspeed (ZeRO-2) |
|------|-----------|---------------------|
| 实现复杂度 | 低 | 中 |
| 内存效率 | 一般 | 高（优化器状态分片） |
| 训练速度 | 快（低通信开销） | 快（支持 overlap_comm） |
| 断点续训 | 简单（加载 .pt） | 自动（save_checkpoint） |
| GAN 训练（HiFiGAN） | ✅ | ❌（当前不支持） |
| 推荐场景 | LLM/Flow SFT，GPU 内存充足 | LLM SFT，GPU 内存受限 |

---

## 5. 训练加速技巧

### 5.1 混合精度训练（AMP）

启用 `--use_amp` 参数，使用 BF16（torch_ddp 模式）或 FP16/BF16（DeepSpeed 模式）：

```bash
torchrun ... cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --use_amp \           # 启用 torch.cuda.amp.autocast，dtype=bf16
  [其他参数...]
```

在 `train_utils.py` 中，BF16/FP16 分支：

```python
# train_utils.py - batch_forward()
if info_dict['train_engine'] == 'torch_ddp':
    autocast = torch.cuda.amp.autocast(enabled=scaler is not None, dtype=dtype)
else:
    autocast = torch.cuda.amp.autocast(enabled=True, dtype=dtype, cache_enabled=False)
```

**建议**：优先使用 BF16（较 FP16 数值更稳定，特别是 Transformer 训练）。在 `ds_stage2.json` 中设置 `"bf16": {"enabled": true}`。

### 5.2 梯度累积

通过 `accum_grad` 模拟更大批次：

```yaml
# conf/cosyvoice3.yaml
train_conf:
  accum_grad: 2      # 等效于 2x batch size
  grad_clip: 5
```

> **注意**：GAN 训练（HiFiGAN）中 `accum_grad` 必须为 1。

### 5.3 数据加载加速

```bash
torchrun ... cosyvoice/bin/train.py \
  --num_workers 4 \       # DataLoader 子进程数，通常设为 2-8
  --prefetch 100 \        # 预取批次数
  --pin_memory \          # 锁页内存，减少 CPU→GPU 传输延迟
  [其他参数...]
```

> **注意**：代码注释中提示 "do not use persistent_workers=True, as whisper tokenizer opens tiktoken file each time"，因此不使用 `persistent_workers`。

### 5.4 离线特征预提取（最重要的加速手段）

在线提取 Embedding 和 Speech Token 会严重拖慢训练。**在数据准备阶段执行 Stage 1 和 Stage 2**，将特征写入 Parquet：

```bash
# 一次性离线提取，之后训练无需重复计算
tools/extract_embedding.py --dir data/$x --onnx_path $pretrained_model_dir/campplus.onnx
tools/extract_speech_token.py --dir data/$x --onnx_path $pretrained_model_dir/speech_tokenizer_v3.onnx
```

### 5.5 动态批处理（Dynamic Batching）

CosyVoice 使用动态批处理策略，以帧数为上限而非样本数：

```yaml
# conf/cosyvoice3.yaml
batch: !name:cosyvoice.dataset.processor.batch
    batch_type: 'dynamic'
    max_frames_in_batch: 2000   # 根据 GPU 内存调整

sort: !name:cosyvoice.dataset.processor.sort
    sort_size: 500    # 在 shuffle_size 内排序以减少 padding

shuffle: !name:cosyvoice.dataset.processor.shuffle
    shuffle_size: 1000
```

**内存调优**：`max_frames_in_batch` 越大，GPU 利用率越高，但内存也越大。从 `1000` 开始，逐步增大直至 OOM 前降一档。

### 5.6 Step 级保存（避免长时间不保存）

```yaml
train_conf:
  save_per_step: 1000   # 每 1000 步保存一次，-1 表示仅 epoch 末保存
```

对于大数据集，建议启用 Step 级保存，防止训练中断导致大量计算浪费。

### 5.7 DeepSpeed overlap_comm 优化

在 `ds_stage2.json` 中启用通信与计算重叠：

```json
"zero_optimization": {
  "stage": 2,
  "overlap_comm": true,          // 梯度通信与反向传播重叠
  "reduce_scatter": true,
  "contiguous_gradients": true   // 连续内存减少碎片
}
```

---

## 6. 端点续训（Checkpoint Resume）

### 6.1 torch_ddp 模式续训

训练自动在 `model_dir` 保存 checkpoint，格式为：

```
exp/cosyvoice3/llm/torch_ddp/
├── init.pt              # 初始化 checkpoint
├── epoch_0_whole.pt     # Epoch 0 结束时保存
├── epoch_0_whole.yaml   # 对应的 loss/step 信息
├── epoch_1_step_1000.pt # Step 级保存（若启用）
└── ...
```

续训只需指定 `--checkpoint` 为最新的 checkpoint：

```bash
torchrun ... cosyvoice/bin/train.py \
  --checkpoint exp/cosyvoice3/llm/torch_ddp/epoch_5_whole.pt \
  --model_dir exp/cosyvoice3/llm/torch_ddp \
  [其他参数...]
```

`train.py` 中续训逻辑：

```python
state_dict = torch.load(args.checkpoint, map_location='cpu')
model.load_state_dict(state_dict, strict=False)
if 'step' in state_dict:
    start_step = state_dict['step']    # 恢复 step 计数
if 'epoch' in state_dict:
    start_epoch = state_dict['epoch']  # 恢复 epoch 计数
scheduler.set_step(start_step)         # 学习率调度器也恢复到正确位置
```

> **关键**：`strict=False` 允许从预训练模型加载部分参数（如新增模块不在 checkpoint 中），也支持跨模型版本迁移。

### 6.2 DeepSpeed 模式续训

DeepSpeed 将优化器状态也保存在 checkpoint 中（当 `--deepspeed.save_states model+optimizer` 时）：

```
exp/cosyvoice3/llm/deepspeed/
├── llm/                    # DeepSpeed checkpoint 目录
│   ├── mp_rank_00_model_states.pt
│   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
│   └── ...
└── llm.yaml
```

DeepSpeed 续训通过 `model.load_checkpoint(load_dir)` 自动恢复，**但当前 `train.py` 并未显式调用此接口**。实践中，推荐：

1. 使用 `--checkpoint` 加载模型权重（优化器状态不续）；
2. 或修改 `init_optimizer_and_scheduler` 调用 `deepspeed.initialize` 时传入 `load_dir`。

**推荐续训配置**：

```bash
# 仅续训模型权重（最常用）
torchrun ... cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --checkpoint exp/cosyvoice3/llm/torch_ddp/epoch_5_whole.pt \
  --model_dir exp/cosyvoice3/llm/torch_ddp_resume \
  [其他参数...]
```

### 6.3 模型平均（Model Averaging）提升最终性能

```bash
python cosyvoice/bin/average_model.py \
  --dst_model exp/cosyvoice3/llm/torch_ddp/llm.pt \
  --src_path exp/cosyvoice3/llm/torch_ddp \
  --num 5 \
  --val_best   # 自动选取 CV loss 最低的 5 个 checkpoint 平均
```

平均策略：读取所有 `.yaml` 文件中的 `loss_dict.loss`，取最小 N 个对应的 checkpoint 做等权平均，显著提升泛化性能。

---

## 7. 推理加速

### 7.1 JIT 导出（适合中小规模部署）

```bash
python cosyvoice/bin/export_jit.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B
```

导出文件：

- `flow.encoder.fp32.zip` / `flow.encoder.fp16.zip`
- `llm.text_encoder.fp32.zip` / `llm.text_encoder.fp16.zip`（仅 CosyVoice 1.0）

使用 `torch.jit.freeze` + `torch.jit.optimize_for_inference` 优化：

```python
script = torch.jit.script(model)
script = torch.jit.freeze(script)
script = torch.jit.optimize_for_inference(script)
```

加载推理：

```python
cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', load_jit=True)
```

### 7.2 ONNX 导出

```bash
python cosyvoice/bin/export_onnx.py --model_dir pretrained_models/Fun-CosyVoice3-0.5B
```

适合跨平台部署（TensorRT、ONNX Runtime）。

### 7.3 vLLM 加速（LLM 推理加速，推荐）

CosyVoice2/3 支持 vLLM 0.9.0 和 0.11.x+（V1 引擎）：

```bash
# 安装 vLLM 环境
conda create -n cosyvoice_vllm --clone cosyvoice
conda activate cosyvoice_vllm
pip install vllm==v0.11.0 transformers==4.57.1 numpy==1.26.4

# 使用 vLLM 推理
python vllm_example.py
```

在代码中启用：

```python
cosyvoice = AutoModel(
    model_dir='pretrained_models/Fun-CosyVoice3-0.5B',
    load_trt=True,
    load_vllm=True,
    fp16=False   # CosyVoice3 使用 fp16=False
)
```

**vLLM 加速效果**：在同等硬件下，LLM 推理吞吐量可提升 2-5x（通过 PagedAttention 和连续批处理）。

### 7.4 TensorRT-LLM + Triton 推理服务器（生产环境推荐）

适合高并发生产部署，由 NVIDIA Yuekai Zhang 贡献（`runtime/triton_trtllm/`）：

```bash
cd runtime/triton_trtllm
# 一键启动（Docker Compose）
docker compose up

# 或分步执行
bash run.sh 0 3   # 下载模型、转换 TRT、配置 Triton、启动服务
bash run.sh 4 4   # 单请求测试
bash run.sh 5 5   # 吞吐量 Benchmark
```

**性能参考**（单 L20 GPU，~170s 音频，26 条样本）：

| 模式 | 并发 | 平均延迟 | RTF |
|------|------|----------|-----|
| 流式，无 speaker cache | 1 | 220ms | 0.124 |
| 流式，有 speaker cache | 1 | 190ms | 0.116 |
| 非流式 | 4 | ~1100ms | 0.09 |

**支持模式**：
- `Decoupled=True`：流式 TTS（首包延迟 ≤ 150ms）
- `Decoupled=False`：非流式 TTS

### 7.5 FlashAttention 加速

在 DiT 模型中启用 Flash Attention，需要安装 `flash-attn` 包：

```bash
pip install flash-attn --no-build-isolation
```

CosyVoice3 的 `CausalMaskedDiffWithDiT` 在支持 Flash Attention 时自动使用（通过 `torch.nn.functional.scaled_dot_product_attention`）。

### 7.6 推理加速方案对比

| 方案 | 适用场景 | 加速幅度 | 部署复杂度 |
|------|----------|----------|------------|
| 原始推理 | 研究/实验 | 基准 | 低 |
| JIT（load_jit=True） | 单机部署 | 1.2-1.5x | 低 |
| ONNX | 跨平台部署 | 1.3-2x | 中 |
| vLLM | LLM 部分加速 | 2-5x（LLM） | 中 |
| TRT-LLM + Triton | 高并发生产 | 3-8x | 高 |

---

## 8. 多语言训练最佳实践

### 8.1 使用 `examples/libritts/cosyvoice3` 作为起点

CosyVoice3 是多语言训练的首选版本，其关键设计：

1. **Instruct 前缀**：通过 `<|endofprompt|>` 分隔控制指令与文本内容
2. **mix_ratio: [5, 15]**：LLM 中文本 token 与语音 token 混合采样比例
3. **原生多语言支持**：9 种语言 + 18+ 中文方言

### 8.2 多语言数据配比策略

**温度采样法**（推荐）：按语言数据量的 T 次方根比例采样，平衡高资源/低资源语言：

```python
# 示例：中(1000h)/英(500h)/日(100h)/韩(50h) 数据混合
# T=0.7 时的采样权重
weights = {
    'zh': 1000**0.7,   # ≈ 251
    'en': 500**0.7,    # ≈ 146
    'ja': 100**0.7,    # ≈ 43
    'ko': 50**0.7,     # ≈ 25
}
```

实践中，直接将各语言 `data.list` 按行数比例拼接，或使用数据集权重采样：

```bash
# 简单拼接（按数据量自然比例）
cat data/zh/parquet/data.list data/en/parquet/data.list \
    data/ja/parquet/data.list data/ko/parquet/data.list > data/train.data.list

# 或通过重复低资源语言数据平衡
awk 'NR>=1{for(i=1;i<=3;i++) print}' data/ko/parquet/data.list >> data/train.data.list
```

### 8.3 Instruct 控制信号设计

CosyVoice3 通过 Instruct 前缀实现多语言/方言/风格控制。在准备多语言数据时，为每种语言设计不同的 Instruct 前缀：

```bash
# 中文数据准备
python local/prepare_data.py \
  --src_dir $zh_data \
  --des_dir data/zh \
  --instruct "You are a helpful assistant.<|endofprompt|>"

# 日文数据准备
python local/prepare_data.py \
  --src_dir $ja_data \
  --des_dir data/ja \
  --instruct "You are a helpful assistant.<|endofprompt|>"

# 带情感控制（示例）
python local/prepare_data.py \
  --src_dir $data \
  --des_dir data/happy \
  --instruct "Speak in a happy tone.<|endofprompt|>"
```

### 8.4 说话人 Embedding 的多语言处理

CAMPPlus 说话人 embedding 是语言无关的，可跨语言迁移。训练时：

```yaml
# conf/cosyvoice3.yaml
padding: !name:cosyvoice.dataset.processor.padding
    use_spk_embedding: False  # 预训练阶段不使用（False）
    # 改为 True 进行 SFT 以保持说话人相似度
```

**多语言 SFT 建议**：
- 预训练阶段（大数据量）：`use_spk_embedding: False`
- SFT 阶段（少量目标语言数据）：`use_spk_embedding: True`

### 8.5 学习率与调度器配置

```yaml
train_conf:
  optim: adam
  optim_conf:
    lr: 1e-5          # SFT 使用较小学习率；预训练可用 1e-4
  scheduler: constantlr   # SFT 用 constantlr；预训练可用 warmuplr
  scheduler_conf:
    warmup_steps: 2500    # 仅 warmuplr 生效
  max_epoch: 200
  grad_clip: 5
  accum_grad: 2
```

**学习率策略对照**：

| 场景 | scheduler | lr | warmup_steps |
|------|-----------|-----|--------------|
| 从零预训练 | `warmuplr` | 1e-4 | 5000 |
| 在预训练模型 SFT | `constantlr` | 1e-5 | N/A |
| DPO 偏好优化 | `constantlr` | 1e-5 | N/A |

### 8.6 三阶段训练流程

```
阶段 1: LLM SFT（文本→语音 Token 对齐）
  ↓ 收敛后
阶段 2: Flow SFT（语音 Token→Mel 谱精化）
  ↓ 收敛后
阶段 3: HiFiGAN SFT（Mel→波形，GAN 训练）
  ↓ 可选
阶段 4: DPO/GRPO（强化学习质量提升）
```

在 `run.sh` 的 Stage 5 中，三个模型分别独立训练：

```bash
for model in llm flow hifigan; do
  torchrun ... --model $model --checkpoint $pretrained_model_dir/$model.pt
done
```

**建议顺序**：先完成 LLM 训练（对内容一致性影响最大），再训练 Flow 和 HiFiGAN。

---

## 9. 强化学习后处理（DPO / GRPO）

### 9.1 DPO 偏好优化

使用 `examples/libritts/cosyvoice2/run_dpo.sh`，需要准备正负样本对：

```bash
# 生成负样本（使用当前模型生成，质量低于参考音频的作为 reject）
python local/prepare_reject_sample.py \
  --src_dir data/$x \
  --des_dir data/${x}_reject \
  --ref_model $pretrained_model_dir

# 打包含 DPO 字段的 Parquet
tools/make_parquet_list.py --dpo [其他参数...]

# DPO 训练
torchrun ... cosyvoice/bin/train.py \
  --model llm \
  --dpo \                              # 启用 DPO
  --ref_model $pretrained_model_dir/llm.pt  # 参考模型路径
  [其他参数...]
```

DPO 损失实现（`cosyvoice/utils/losses.py`）：

```python
# train_utils.py - batch_forward()
preference_loss, chosen_reward, reject_reward = dpo_loss(
    chosen_logps, rejected_logps, reference_chosen_logps, reference_rejected_logps
)
total_loss = preference_loss + sft_loss  # DPO + SFT 联合训练
```

### 9.2 GRPO 强化学习

使用 `examples/grpo/cosyvoice2`，奖励信号为 ASR 转写误差率：

```bash
# 启动 ASR 奖励服务
python3 token2wav_asr_server.py --number-of-devices 8

# GRPO 训练（8 卡）
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  custom_reward_function.path=reward_tts.py \
  trainer.resume_mode='auto' \        # 支持自动续训
  trainer.total_epochs=1
```

**GRPO 效果（论文/实验数据）**：

| 模型 | CosyVoice3 zero_shot_zh CER |
|------|------------------------------|
| CosyVoice2 LLM（官方） | 4.08% |
| + GRPO | **3.36%（-18%）** |
| Fun-CosyVoice3-0.5B-2512 | 1.21%（test-zh） |
| Fun-CosyVoice3-0.5B-2512_RL | **0.81%（-33%）** |

---

## 10. 训练监控与调试

### 10.1 TensorBoard 监控

```bash
tensorboard --logdir tensorboard/cosyvoice3/llm/torch_ddp --port 6006
```

监控指标：

| 指标 | 说明 |
|------|------|
| `TRAIN/loss` | 训练损失 |
| `CV/loss` | 验证集损失（主要参考指标） |
| `TRAIN/lr` | 当前学习率 |
| `TRAIN/grad_norm` | 梯度范数（>5 时触发裁剪） |
| `TRAIN/dpo_loss` | DPO 偏好损失（DPO 模式） |
| `TRAIN/dpo_acc` | DPO 正样本奖励高于负样本的比例 |

### 10.2 日志级别

`train.py` 中设置为 `logging.DEBUG`，训练过程中每 `log_interval` 步输出：

```
TRAIN Batch 1/1000 loss 2.123456 lr 0.00001000 grad_norm 1.234567 rank 0
```

### 10.3 常见问题排查

**问题 1：`get infinite grad_norm`**

```python
# train_utils.py 中已处理
if torch.isfinite(grad_norm):
    optimizer.step()
else:
    logging.warning('get infinite grad_norm, check your code/data if it appears frequently')
```

处理：偶发可忽略；频繁出现需检查数据（是否有 NaN）或降低学习率。

**问题 2：不同 rank 数据量不均（Uneven workload）**

`cosyvoice_join` 函数处理此问题，检测到不均时当前 rank 主动退出当前 epoch：

```python
def cosyvoice_join(group_join, info_dict):
    try:
        dist.monitored_barrier(group=group_join, timeout=...)
        return False
    except RuntimeError:
        logging.info("Detected uneven workload distribution...")
        return True  # Break current epoch
```

**问题 3：DeepSpeed 内存估算**

```python
# train_utils.py - wrap_cuda_model()
estimate_zero2_model_states_mem_needs_all_live(
    model,
    num_gpus_per_node=local_world_size,
    num_nodes=world_size // local_world_size)
```

训练开始时会输出 ZeRO-2 内存需求估算，作为 GPU 内存规划参考。

---

## 11. 高频操作速查表

### 完整 SFT 流程（cosyvoice3，单机 8 卡）

```bash
cd examples/libritts/cosyvoice3

# 1. 数据准备（Stage 0-3）
bash run.sh -1 3

# 2. 启动 LLM 训练（Stage 5）
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
num_gpus=8
job_id=1986
train_engine=torch_ddp
pretrained_model_dir=../../../pretrained_models/Fun-CosyVoice3-0.5B

cat data/{train-clean-100,train-clean-360,train-other-500}/parquet/data.list > data/train.data.list
cat data/{dev-clean,dev-other}/parquet/data.list > data/dev.data.list

torchrun --nnodes=1 --nproc_per_node=$num_gpus \
    --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
  ../../../cosyvoice/bin/train.py \
  --train_engine $train_engine \
  --config conf/cosyvoice3.yaml \
  --train_data data/train.data.list \
  --cv_data data/dev.data.list \
  --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
  --onnx_path $pretrained_model_dir \
  --model llm \
  --checkpoint $pretrained_model_dir/llm.pt \
  --model_dir $(pwd)/exp/cosyvoice3/llm/$train_engine \
  --tensorboard_dir $(pwd)/tensorboard/cosyvoice3/llm/$train_engine \
  --ddp.dist_backend nccl \
  --num_workers 4 \
  --prefetch 100 \
  --pin_memory \
  --use_amp \
  --deepspeed_config ./conf/ds_stage2.json \
  --deepspeed.save_states model+optimizer

# 3. 模型平均
python cosyvoice/bin/average_model.py \
  --dst_model $(pwd)/exp/cosyvoice3/llm/$train_engine/llm.pt \
  --src_path $(pwd)/exp/cosyvoice3/llm/$train_engine \
  --num 5 --val_best

# 4. 导出推理模型
python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
```

### 从 Checkpoint 续训

```bash
torchrun [分布式参数] cosyvoice/bin/train.py \
  --checkpoint exp/cosyvoice3/llm/torch_ddp/epoch_10_whole.pt \
  --model_dir exp/cosyvoice3/llm/torch_ddp \
  [其他参数同上]
```

### 切换到 DeepSpeed 并启用 BF16

1. 修改 `conf/ds_stage2.json`：`"bf16": {"enabled": true}`
2. 训练命令改为 `--train_engine deepspeed`

### 快速验证推理效果

```python
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import soundfile as sf

cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B',
                      load_jit=True, load_trt=True, load_vllm=True)

# 多语言零样本克隆
for i, (audio, sample_rate) in enumerate(cosyvoice.inference_zero_shot(
    '收到好友的生日礼物，笑容如花儿般绽放。',
    'You are a helpful assistant.<|endofprompt|>希望你以后能做的比我还好。',
    'asset/zero_shot_prompt.wav'
)):
    sf.write(f'output_{i}.wav', audio, sample_rate)
```

---

## 12. 相关论文整合

### 12.1 核心论文

#### CosyVoice 1.0
- **标题**：CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens
- **链接**：[https://funaudiollm.github.io/pdf/CosyVoice_v1.pdf](https://funaudiollm.github.io/pdf/CosyVoice_v1.pdf)
- **关键贡献**：
  - 监督式离散语音 Token（speech tokenizer）作为 LLM 建模目标
  - CAMPPlus 说话人 embedding 用于零样本克隆
  - Conditional Flow Matching（CFM）用于 Mel 谱生成
  - Repetition Aware Sampling（RAS）解决 LLM 重复问题

#### CosyVoice 2.0
- **标题**：CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models
- **链接**：[https://arxiv.org/pdf/2412.10117](https://arxiv.org/pdf/2412.10117)
- **关键贡献**：
  - Qwen2 作为 LLM 主干，取代自研 TransformerLM
  - 流式推理：LLM KV Cache + 因果 DiT（CausalMaskedDiffWithDiT）
  - 25 Hz Token 帧率（相比 v1 的 50 Hz）
  - 统一的 SFT + 流式推理框架

#### Fun-CosyVoice 3.0
- **标题**：Fun-CosyVoice3: Scalable Zero-Shot Multilingual TTS Synthesis with Instruction Following
- **链接**：[https://arxiv.org/pdf/2505.17589](https://arxiv.org/pdf/2505.17589)
- **关键贡献**：
  - Instruct 机制：通过 `<|endofprompt|>` 控制语言/方言/情感/风格
  - 原生支持 9 语言 + 18+ 中文方言，跨语言零样本克隆
  - 音调填充（Pronunciation Inpainting）：中文拼音和英文 CMU 音素级控制
  - 无需传统文本归一化前端（内置数字/符号读法）
  - RL 训练（GRPO）：CER 从 1.21% → 0.81%

### 12.2 关键技术论文

#### RAS（Repetition Aware Sampling）
- **出处**：CosyVoice 1.0 论文 § 3.3
- **问题**：自回归 LLM 在生成语音 Token 时容易陷入重复循环
- **方案**：在 top-p/top-k 采样基础上，检测近期 `win_size` 个 Token 的重复模式并惩罚重复 Token：
  ```python
  # cosyvoice/utils/common.py
  def ras_sampling(logits, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
      ...
  ```

#### Conditional Flow Matching（CFM）
- **相关论文**：Flow Matching for Generative Modeling（Lipman et al., 2022）
- **在 CosyVoice 中的应用**：`cosyvoice/flow/flow_matching.py` 中的 `CausalConditionalCFM`
- **参数**：`sigma_min=1e-6`，`t_scheduler='cosine'`，`training_cfg_rate=0.2`

#### DiT（Diffusion Transformer）
- **相关论文**：Scalable Diffusion Models with Transformers（Peebles & Xie, 2023）
- **在 CosyVoice 中的应用**：`cosyvoice/flow/DiT/dit.py`，用于替代 U-Net 作为 Flow 的去噪估计器
- **CosyVoice3 配置**：`depth=22, heads=16, dim=1024`

#### DPO（Direct Preference Optimization）
- **相关论文**：Direct Preference Optimization: Your Language Model is Secretly a Reward Model（Rafailov et al., 2023）
- **在 CosyVoice 中的应用**：`examples/libritts/cosyvoice2/run_dpo.sh`，以人工或模型评估的语音质量排序为偏好信号
- **实现**：`cosyvoice/utils/losses.py`，`DPOLoss`（beta=0.01）

#### GRPO（Group Relative Policy Optimization）
- **相关论文**：DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models（Shao et al., 2024）
- **在 CosyVoice 中的应用**：`examples/grpo/cosyvoice2`，用 ASR CER 作为奖励信号
- **框架**：veRL（volcengine/verl）

### 12.3 基础设施论文

| 技术 | 论文/来源 | 在仓库中的体现 |
|------|-----------|----------------|
| Qwen2 | Qwen2 Technical Report（Alibaba, 2024） | `cosyvoice.llm.llm.Qwen2Encoder` |
| CAMPPlus | CAM++: A Fast and Accurate Speaker Verification... | `campplus.onnx` |
| HiFi-GAN | HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis（Kong et al., 2020） | `cosyvoice/hifigan/` |
| DeepSpeed ZeRO | ZeRO: Memory Optimizations Toward Training Trillion Parameter Models（Rajbhandari et al., 2020） | `conf/ds_stage2.json` |
| vLLM | Efficient Memory Management for Large Language Model Serving with PagedAttention（Kwon et al., 2023） | `cosyvoice/vllm/` |
| SenseVoice | SenseVoice: Scalable Multilingual Audio Understanding Model（FunAudioLLM） | GRPO 奖励函数中的 ASR |

---

## 附录：关键配置参数速查

### cosyvoice3.yaml 重要参数

```yaml
# 模型参数
llm:
  speech_token_size: 6561     # 语音词表大小
  mix_ratio: [5, 15]          # 文本:语音 token 混合比（控制生成速度）
  sampling:
    top_p: 0.8                # RAS 采样参数
    top_k: 25
    win_size: 10              # 重复检测窗口
    tau_r: 0.1                # 重复惩罚系数

flow:
  pre_lookahead_len: 3        # 因果 DiT 前瞻长度
  chunk_size: 25              # 流式推理 chunk 大小
  num_decoding_left_chunks: -1 # -1 表示使用所有左侧 chunk

# 数据处理
filter:
  max_length: 6000            # 最大帧数
  min_length: 100
  token_max_length: 200       # 最大文本 token 数
  token_min_length: 1

batch:
  batch_type: 'dynamic'
  max_frames_in_batch: 2000   # 关键调优参数

# 训练参数
train_conf:
  lr: 1e-5                    # SFT 学习率
  scheduler: constantlr
  max_epoch: 200
  grad_clip: 5
  accum_grad: 2
  save_per_step: -1           # -1 表示仅 epoch 末保存
```

### 环境变量速查

| 变量 | 说明 | 示例 |
|------|------|------|
| `CUDA_VISIBLE_DEVICES` | 可用 GPU | `"0,1,2,3"` |
| `WORLD_SIZE` | 总 GPU 数（torchrun 自动设置） | `8` |
| `RANK` | 全局 rank（torchrun 自动设置） | `0` |
| `LOCAL_RANK` | 本节点 rank（torchrun 自动设置） | `0` |
| `PYTHONPATH` | 需包含 Matcha-TTS | `third_party/Matcha-TTS` |
| `onnx_path` | ONNX 模型路径（train.py 设置） | 自动从 `--onnx_path` 注入 |
