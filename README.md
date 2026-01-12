# LLM-Memory-Research

面向长程记忆与检索的实验框架，包含滑动窗口、标准 RAG 与 NSCMA 三种方案，并提供合成数据的知识更新与多跳推理评测。

- Baselines: SlidingWindow, StandardRAG
- Ours: NSCMA（Neural-Search + Curated-Graph + Memory-Aware），结合向量检索与语义图谱，支持策展更新

## 目录结构
- config.py: 全局配置（模型、硬件与实验参数）
- models.py: ModelEngine，集成 Transformers 与 Sentence-Transformers，4-bit 量化推理
- data_loader.py: 合成数据生成（知识更新、多跳推理）
- memory_systems.py: 三种记忆系统实现（SlidingWindow/StandardRAG/NSCMA）
- main.py: 实验入口（可运行三类实验）
- requirements.txt: 依赖列表

## 环境与安装
- Python 3.10+
- 可选 GPU（建议 12GB+，4-bit 量化可在更小显存上运行）
- CUDA/驱动需与 PyTorch 匹配

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

若 bitsandbytes 在 Windows 上异常，可尝试 WSL/Ubuntu 或改用 CPU（见下文“配置”）。

## 配置
编辑 config.py：
- LLM_MODEL_ID: 默认 Qwen/Qwen2.5-7B-Instruct
- EMBEDDING_MODEL_ID: 默认 BAAI/bge-m3
- DEVICE: "cuda" 或 "cpu"
- LOAD_IN_4BIT: 是否 4-bit 量化加载
- MAX_NEW_TOKENS: 生成长度
- TEST_SAMPLE_LIMIT: 每个实验的数据条数
- RAG_TOP_K: RAG 检索条数
- SURPRISE_THRESHOLD: NSCMA 的“惊奇度”阈值

示例（CPU 跑通优先）：
- DEVICE="cpu"
- LOAD_IN_4BIT=False

## 运行
默认运行消融实验（Experiment 3）：
```bash
python main.py
```

其他实验（在 main.py 中按需启用/注释）：
- Experiment 1: 变长干扰下的知识更新鲁棒性
- Experiment 2: 多跳推理（含长距离噪音）

输出
- 控制台打印各系统准确率与运行耗时
- Experiment 1 额外保存 exp1_results.json

## 方法说明
- SlidingWindow: 维护固定窗口文本上下文，简单高效但易被长噪声“冲走”信息
- StandardRAG: Chroma 向量库检索，Sentence-Transformers 编码，Top-K 拼接回答
- NSCMA:
  - Neural Buffer: 依据“惊奇度”筛选增量信息
  - Vector Store: Chroma 向量检索兜底
  - Semantic Graph: 使用 NetworkX 构建实体关系，节点标准化防断连
  - Curator: 同一主体下相似关系的新事实覆盖旧事实
  - Query: 图谱路径增强 + 向量检索回退，动态 Prompt 选择

## 数据生成
- generate_synthetic_update: 植入“旧事实→噪音→新事实→噪音”，考察记忆更新与干扰鲁棒性
- generate_synthetic_multihop: 逆序+噪音的三跳链路（Object→Container→Room→Building）

## 性能与内存
- 生成阶段强制截断输入至 1024 tokens，减小显存峰值
- 使用 torch.cuda.empty_cache() 回收显存碎片
- 4-bit 量化建议搭配半精度计算（nf4 + float16）

## 故障排除
- bitsandbytes 报错：使用 WSL/Ubuntu 或切换 CPU（LOAD_IN_4BIT=False, DEVICE="cpu"）
- 显存不足：降低 MAX_NEW_TOKENS、RAG_TOP_K；确保输入截断；缩小 TEST_SAMPLE_LIMIT
- 模型下载失败：检查网络/HF 镜像；提前手动下载模型权重

## 免责声明
本仓库仅用于研究用途。请遵循相应模型与依赖的许可证与使用条款。
