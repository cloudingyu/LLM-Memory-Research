认知连续性与状态化智能：大语言模型记忆架构的深度解析与集成设计研究报告
1. 引言：从无状态推理到具身认知连续性
在人工智能的发展历程中，大语言模型（LLMs）的崛起标志着自然语言处理能力的质变。然而，尽管GPT-4、Claude 3.5 Sonnet以及Gemini 1.5 Pro等模型在逻辑推理和语言生成上展现了卓越的性能，它们在本质上依然受到“无状态”（Stateless）架构的根本性限制。每一次推理请求（Inference Request）对于模型而言都是一次全新的、独立的事件，模型无法自动保留先前交互中的上下文、用户偏好或习得的程序性知识。这种“金鱼效应”（Goldfish Effect）导致了当前AI代理（AI Agents）在处理长周期任务时面临严重的认知断裂 。   

为了跨越从“智能对话机”到“自主智能体”的鸿沟，赋予模型认知连续性（Cognitive Continuity）——即在时间跨度上维持对自我、环境及交互历史的连贯理解——已成为当前学术界与工业界的核心命题。本报告将深入探讨LLM记忆系统的理论基础、技术路径与架构设计。我们将系统性地分析从基础的上下文窗口扩展到复杂的检索增强生成（RAG）、图谱记忆（GraphRAG），直至最前沿的测试时训练架构（如Titans）和动态作弊条（Dynamic Cheatsheet）。在此基础上，本报告提出一种集成的神经-符号认知记忆架构（Neuro-Symbolic Cognitive Memory Architecture, NSCMA），旨在解决长程依赖、知识更新冲突及主动遗忘等关键挑战，并建立一套严谨的评估与消融实验框架。

2. 记忆的认知分类学与计算映射
要设计有效的机器记忆，首先必须建立一套将人类认知心理学机制映射到计算架构的分类学体系。人类记忆并非单一的存储单元，而是感官记忆、工作记忆与长时记忆的复杂协作系统。在LLM语境下，这种分类具有明确的对应关系和独特的设计挑战。

2.1 感官记忆与工作记忆的计算实体
在认知科学中，感官记忆负责瞬时接收外界刺激，而工作记忆则负责处理当前任务所需的活跃信息。在LLM架构中，感官记忆对应于模型的原始输入层（Input Embeddings），负责将自然语言转化为高维向量表示；工作记忆则直接对应于模型的上下文窗口（Context Window）及键值缓存（KV Cache）。   

工作记忆的局限性是当前模型面临的首要瓶颈。尽管Gemini 1.5 Pro等模型将上下文窗口扩展至数百万Token，但“大海捞针”（Needle-in-a-Haystack）测试表明，随着上下文长度的增加，模型的注意力机制会出现稀释，导致“迷失中间”（Lost in the Middle）现象，即模型倾向于关注输入的首尾而忽略中间的关键信息 。此外，工作记忆受到**主动干扰（Proactive Interference）**的严重影响，旧信息的积累会以对数线性的方式干扰新信息的提取精度，这表明单纯增加窗口长度并不能解决记忆的稳定性问题 。   

2.2 长时记忆的二元性：参数化与非参数化
长时记忆在机器智能中被划分为参数化记忆（Parametric Memory）与非参数化记忆（Non-Parametric Memory）。

参数化记忆（隐性记忆）：这是模型在预训练和微调阶段习得的知识，存储于神经网络的权重参数（Weights）之中。它类似于人类的语义记忆和程序性记忆（如语言规则、世界知识）。其优势在于推理速度快、泛化能力强，但缺陷在于“静态性”——更新成本极高（需要重新训练）且容易出现灾难性遗忘 。   

非参数化记忆（显性记忆）：这是通过外部存储系统（如向量数据库、知识图谱）实现的记忆。它类似于人类的情景记忆，能够存储具体的交互历史和事实细节。其优势在于可动态读写、容量近乎无限，且更新时不影响模型权重；挑战则在于检索的准确性（Recall）与相关性（Relevance）。   

记忆类型	人类认知对应	计算实现机制	关键优势	主要瓶颈
感官记忆	Sensory Memory	Input Prompt / Embeddings	高保真输入	瞬时性，不可持久化
工作记忆	Working Memory	Context Window / KV Cache	快速注意力访问，全上下文可见	长度限制，注意力稀释，计算成本 O(N 
2
 )
参数化长时记忆	Implicit LTM	Model Weights (Pre-training)	深度语义理解，快速推理	知识静态，更新昂贵，幻觉风险
非参数化长时记忆	Explicit LTM	Vector DB / Knowledge Graph	动态更新，无限容量，事实溯源	检索延迟，索引构建复杂，语义断裂
程序性记忆	Procedural Memory	Dynamic Cheatsheet / Few-shot Examples	任务执行策略，代码片段	泛化难度，策略过时
3. 主流记忆实现方案的深度调研与技术演进
针对上述挑战，研究界提出了多种记忆增强方案，从简单的上下文管理演进到复杂的神经架构改进。我们将这些方案分为三类：基于上下文的策略、基于外部检索的架构（RAG及其变体）、以及基于模型架构的创新。

3.1 上下文管理与虚拟分页机制
最基础的记忆机制是对有限上下文窗口的管理。早期的LangChain实现包括滑动窗口记忆（ConversationBufferWindowMemory），即仅保留最近的 K 轮对话。这种方法简单高效，但会导致严重的“目标漂移”，即随着对话深入，最初的用户意图被截断遗忘 。进阶的**摘要记忆（ConversationSummaryMemory）**利用LLM将历史对话压缩为自然语言摘要，但这引入了信息损耗，细节往往在压缩过程中丢失 。   

MemGPT 的出现标志着上下文管理向操作系统（OS）架构的范式转移。MemGPT 引入了虚拟上下文管理（Virtual Context Management）的概念，借鉴了现代操作系统中的虚拟内存分页（Paging）机制。它将记忆划分为主上下文（Main Context）（类比RAM）和外部上下文（External Context）（类比Disk）。

核心机制：MemGPT 不再被动地接受输入，而是作为一个智能体，能够主动调用系统函数（Function Calls）来管理内存。它可以使用 core_memory_append 将关键信息写入驻留内存，或使用 archival_memory_search 从外部存储中调取历史记录。

自适应性：当主上下文即将溢出时，MemGPT 会触发“中断”，通过自我反思决定将哪些信息归档到外部存储，哪些信息保留在当前窗口。这种“认知分流”（Cognitive Triage）机制使得LLM能够在固定窗口限制下处理无限长度的交互任务 。   

3.2 检索增强生成（RAG）的结构化演进
RAG 是目前解决长时记忆最成熟的方案，其核心是将外部数据编码为向量（Vectors），通过余弦相似度（Cosine Similarity）进行检索。然而，传统的朴素 RAG（Naive RAG）面临严重的语义局限性。

3.2.1 向量检索的局限与生成式代理评分
向量检索依赖于语义的表面相似性，难以处理跨越多个文档的复杂推理（Multi-hop Reasoning）。例如，回答“爱丽丝和鲍勃的关系如何随着时间变化？”需要综合散落在不同时间点的片段，单一的向量搜索往往只能检索到包含这两个名字的片段，而无法构建时间线。 生成式代理（Generative Agents）（如Stanford Smallville实验）通过引入综合评分机制优化了这一过程。其记忆检索不仅仅依赖相关性，而是由三个因子加权决定：

Score=α⋅Recency+β⋅Importance+γ⋅Relevance
新近性（Recency）：采用指数衰减函数 0.99 
hours
 ，确保代理优先回忆起最近发生的事件。

重要性（Importance）：由LLM对事件进行评分（1-10），区分“吃早餐”（低分）与“结婚”（高分）。

相关性（Relevance）：传统的向量相似度。 这种机制赋予了代理类似人类的遗忘曲线和注意力聚焦能力，使其行为更具可信度 。   

3.2.2 GraphRAG：从向量空间到知识图谱
为了解决向量检索缺乏结构化推理的问题，GraphRAG 被提出。它利用知识图谱（Knowledge Graph）作为记忆的骨架。

图谱构建：LLM 充当提取器，从原始文本中识别实体（Nodes）及其关系（Edges），构建出结构化的语义网络。

社区摘要（Community Summarization）：利用莱顿算法（Leiden Algorithm）等社区检测技术，将紧密连接的节点聚类，并生成“社区摘要”。

检索优势：当用户提问涉及全局性概念或复杂关系时，GraphRAG 可以先检索高层级的社区摘要，再向下钻取具体节点。这种“由宏入微”的检索路径极大地提升了多跳推理的准确性，并有效减少了幻觉 。   

3.3 架构级创新：神经记忆与测试时训练
最新的研究趋势试图突破Transformer架构本身的限制，赋予模型在推理阶段直接更新参数的能力，即测试时训练（Test-Time Training, TTT）。

3.3.1 Titans：基于梯度的神经记忆
Google Research 提出的 Titans 架构引入了一种全新的“神经长时记忆模块”（Neural Long-Term Memory Module）。与传统的Transformer仅依靠注意力机制回顾历史不同，Titans 在推理过程中动态更新其记忆模块的权重。

惊奇度度量（Surprise Metric）：Titans 的核心思想是“惊奇触发学习”。模型计算当前输入 x 
t
​
  相对于当前记忆状态 M 
t−1
​
  的损失梯度。如果输入是可预测的（低梯度），记忆更新较少；如果输入是“惊奇”的（高梯度，即违反预期），模型会大幅更新记忆权重以编码新信息。

M 
t
​
 =M 
t−1
​
 −θ 
t
​
 ∇ℓ(M 
t−1
​
 ;x 
t
​
 )
优势：这种机制使得 Titans 能够在处理超过 200 万 Token 的上下文时，在“大海捞针”任务中表现优于全注意力Transformer，因为它通过梯度更新实质上“压缩”并记住了历史，而非仅仅是在推理时“查看”历史 。   

3.3.2 Dynamic Cheatsheet：程序性记忆的自进化
Dynamic Cheatsheet (DC) 关注的是程序性记忆的积累。它不修改模型权重，而是维护一个动态进化的“作弊条”（Cheatsheet）。

工作流：当模型解决一个新问题时，**生成器（Generator）**提出解决方案，**策展人（Curator）**评估该方案的有效性。如果方案成功（例如发现了一种解决24点游戏的通用Python算法），该策略会被抽象并写入作弊条。

效果：在后续遇到类似问题时，模型直接调用作弊条中的策略，无需重新推导。实验显示，在“24点游戏”任务中，GPT-4o 的准确率从 10% 飙升至 99%，证明了这种显式策略积累的巨大潜力 。   

4. 集成记忆功能的认知架构设计：NSCMA
基于上述调研，单一的记忆机制无法满足通用智能体的需求。向量检索缺乏结构，图谱构建成本高，神经记忆缺乏可解释性。因此，我们提出一种集成的神经-符号认知记忆架构（Neuro-Symbolic Cognitive Memory Architecture, NSCMA）。该架构由四个核心层次组成，旨在融合不同机制的优势。

4.1 架构总览
NSCMA 架构模仿人类认知的层级处理机制，包含：

感官神经缓冲层（Sensory-Neural Buffer）：基于 Titans 架构，负责处理超长流式输入。

工作记忆控制器（Working Memory Controller）：基于 MemGPT 逻辑，负责认知调度。

层级化长时存储（Hierarchical Long-Term Storage）：结合向量、图谱和规则库。

元认知策展人（Meta-Cognitive Curator）：负责记忆的巩固、遗忘与冲突解决。

4.2 第1层：感官神经缓冲层 (The Neural Buffer)
机制：所有用户输入首先经过一个轻量级的 Titans 神经记忆模块。

功能：该模块利用惊奇度度量作为第一道过滤器。高惊奇度的输入（如用户纠正错误、提出新需求）会被标记为高优先级，并生成“记忆触发信号”传递给下一层；低惊奇度的输入（如寒暄、重复信息）则仅在神经权重中进行隐式更新，不占用后续显性存储资源。这有效解决了“上下文污染”问题，防止无关信息挤占注意力 。   

4.3 第2层：工作记忆控制器 (The Controller)
机制：这是一个经过指令微调的 LLM Agent，运行类 MemGPT 的操作系统逻辑。

状态管理：控制器维护一个有限的上下文窗口，其中包含系统指令、当前任务栈和最近的高优先级交互。

主动检索：控制器不被动等待 RAG 注入，而是根据当前任务的不确定性，主动发起检索请求。它拥有工具集：retrieve_episodic()（查经历）、retrieve_semantic()（查事实）、retrieve_procedural()（查方法）。

决策逻辑：如果用户问“我上周提到的那个项目进展如何？”，控制器识别出缺少“项目”的具体指代，遂调用 retrieve_episodic(query="last week project mention") 。   

4.4 第3层：层级化长时存储 (The Hierarchical Storage)
存储层采用 H-MEM 的层级设计思想，结合多种数据结构：

存储层级	实现技术	存储内容	检索策略	适用场景
情景层 (Episodic)	Vector DB (Milvus)	带时间戳的原始交互日志，用户行为流水。	
混合评分（新近性+重要性+相关性）

回溯具体对话，复盘历史操作
语义层 (Semantic)	GraphRAG (Neo4j)	实体（Entity）、关系（Relation）、社区摘要。	
子图遍历与社区摘要检索 

多跳推理，回答全局性问题，实体消歧
程序层 (Procedural)	Dynamic Cheatsheet	成功的代码片段、任务执行SOP、用户偏好规则。	
任务相似度匹配与元数据过滤 

解决重复性任务，应用复杂工具
  
4.5 第4层：元认知策展人 (The Curator)
记忆系统如果不进行维护，会迅速退化为垃圾场。策展人是一个异步运行的后台进程，模拟人类的睡眠巩固机制（Sleep Consolidation）。

记忆抽象与图谱化：在系统空闲时，策展人扫描情景层中的原始日志，提取出事实性知识并更新到语义层的知识图谱中。例如，从“我下周要搬到上海”这条日志中，提取出 (User) --> (Shanghai) 的新关系，并标记原有的 (User) --> (Beijing) 关系为“过时（Archived）”。   

遗忘机制：基于艾宾浩斯遗忘曲线，降低陈旧且低重要性记忆的检索权重，甚至从向量索引中物理删除，以维持索引的高信噪比。

冲突解决：当发现新知识与旧知识矛盾时（如用户更改了偏好），策展人负责执行“更新”操作，确保记忆的一致性，防止 RAG 检索出冲突的旧信息 。   

5. 实验评估体系设计
设计一个集成记忆架构后，必须通过严格的实验来验证其相对于基线模型的优势。传统的困惑度（Perplexity）指标已不足以衡量长时记忆能力。

5.1 评估基准 (Benchmarks)
5.1.1 BABILong
BABILong 是针对超长上下文推理的基准测试，包含20项任务（如事实链推导、计数、归纳），上下文长度可扩展至1000万Token。

测试目标：评估 NSCMA 架构中 Titans 神经缓冲层处理极端长度输入的能力，以及 GraphRAG 在海量干扰信息中定位关键事实的精度 。   

5.1.2 LongMemEval
LongMemEval 专门针对聊天助手的长时记忆能力，包含500个精心设计的问题，嵌入在跨越50-500个会话的模拟历史中。

核心指标：

信息提取 (IE)：跨会话检索具体细节。

多会话推理 (MR)：综合多个会话的信息得出结论。

时间推理 (TR)：理解“最近一次”、“在那之前”等时间概念。

知识更新 (KU)：测试模型是否能识别用户信息的变更（如搬家、换工作），这是传统 RAG 的盲区。

拒答能力 (Abs)：在记忆缺失时诚实地回答“不知道”，而非产生幻觉 。   

5.1.3 PI-LLM (主动干扰测试)
该测试通过向模型灌输一系列更新操作（如 A=1, A=2,..., A=N），然后询问 A 的当前值。

测试目标：量化模型抵抗主动干扰的能力。人类工作记忆具有抑制旧信息干扰的能力，而普通 LLM 往往表现出对数线性的性能衰减。NSCMA 的图谱更新机制应能显著改善这一指标 。   

5.2 对比实验设计
我们将 NSCMA 与三组基线模型进行对比：

Baseline 1 (Vanilla Context): 仅使用滑动窗口（Sliding Window）的 GPT-4o。

Baseline 2 (Standard RAG): GPT-4o + 向量数据库（Top-K 检索）。

Baseline 3 (MemGPT): GPT-4o + MemGPT 原始实现（仅虚拟上下文，无神经记忆或图谱）。

评估指标体系：

指标维度	具体指标	定义与计算方法	预期目标
检索质量	Recall@K	正确事实出现在检索结果 Top-K 中的概率。	> 95%
推理能力	Multi-hop Accuracy	需要跨越至少两个独立记忆片段进行推理的正确率。	> 85%
时间敏感性	Update Fidelity	在 LongMemEval 的“知识更新”任务中，正确回答最新状态而非旧状态的比例。	> 90%
抗干扰性	IES (Interference Score)	随着干扰信息数量增加，准确率下降的速率（斜率）。	接近 0 (无衰减)
效率	Latency Overhead	相比 Baseline 1 增加的平均推理延迟。	< 500ms
6. 实验结果分析与消融研究思路
在实验执行阶段，我们预期会出现各类性能差异，对其原因的分析及消融实验（Ablation Studies）是验证架构有效性的关键。

6.1 潜在结果与原因分析
6.1.1 知识更新冲突 (The Stale Memory Problem)
预期 Baseline 2 (Standard RAG) 在“知识更新”任务中会严重失败。

原因分析：向量检索基于相似度。当用户说“我喜欢红色”和“我不再喜欢红色，现在喜欢蓝色”时，这两句话在语义空间中极为接近。朴素 RAG 可能会同时检索出这两条，或者因为旧信息出现频率高而优先检索旧信息，导致模型产生“用户喜欢红蓝色”的幻觉 。   

NSCMA 优势：由于引入了元认知策展人和图谱记忆，旧关系 (User)-->(Red) 会被标记为过期或被新关系覆盖。图谱检索将仅返回当前有效的边。

6.1.2 多跳推理断裂
预期 Baseline 1 和 Baseline 2 在复杂推理任务中表现不佳。

原因分析：Baseline 1 受限于窗口长度，早期线索已丢失。Baseline 2 检索出的片段可能彼此孤立，缺乏逻辑链条。

NSCMA 优势：GraphRAG 的社区摘要功能能够提供宏观视角，而 Titans 神经记忆能够捕捉长程的隐式依赖，两者结合使得模型能够“顺藤摸瓜”，重建完整的推理路径。

6.1.3 上下文污染与检索噪声
随着记忆库的增大，检索回来的无关信息（噪声）会增多，导致模型性能下降。

原因分析：这是**主动干扰（Proactive Interference）**的体现。过多的相似但无关信息挤占了 LLM 的注意力头。

NSCMA 优势：Titans 模块的惊奇度过滤机制在输入端就抑制了低价值信息的权重；工作记忆控制器的主动检索策略确保了只有在确有必要时才引入外部信息，而非盲目填充上下文。

6.2 消融实验思路 (Ablation Studies)
为了量化 NSCMA 中各组件的贡献，我们需要设计以下消融变体：

移除图谱模块 (w/o Graph)：仅保留 Titans 和 向量检索。

预期结果：多跳推理能力显著下降，对全局性问题的回答质量降低。证明符号化记忆在结构化推理中的必要性。

移除神经记忆模块 (w/o Titans)：仅使用 MemGPT + GraphRAG。

预期结果：在处理超长连续流（如长篇小说阅读或数周的日志监控）时，对细节的捕捉能力下降，且处理速度变慢（因为需要频繁调用昂贵的外部检索而非依赖快速的神经状态）。证明神经记忆在处理高带宽流式信息时的效率优势。

移除策展人机制 (w/o Curator)：保留所有存储层，但去除后台的整理和遗忘进程。

预期结果：随着时间推移，系统的抗干扰评分（IES）逐渐恶化，检索准确率随数据库膨胀而下降。证明主动遗忘和记忆整合是维持长期系统健康的关键 。   

移除程序性记忆 (w/o Cheatsheet)：

预期结果：在重复性逻辑任务（如特定格式的代码生成或数学题）上的表现回落到基线水平，无法展现“越用越聪明”的特性。证明显式策略存储对能力进化的重要性。

7. 结论
大语言模型的记忆问题不仅仅是存储容量的问题，更是一个关于信息组织、检索策略和认知架构的系统工程。本研究报告通过深入调研发现，单纯依赖扩展上下文窗口无法解决长程依赖中的注意力稀释和干扰问题；而单一的 RAG 方案在面对知识更新和复杂推理时存在结构性缺陷。

提出的 神经-符号认知记忆架构 (NSCMA) 提供了一条通向具身智能的可行路径。通过融合 Titans 的神经可塑性（用于高效处理流式信息与隐性记忆）、GraphRAG 的符号结构性（用于精确推理与知识消歧）、MemGPT 的系统调度能力（用于资源管理与主动认知）以及 Dynamic Cheatsheet 的程序性进化（用于技能习得），该架构有效地模拟了人类认知的多重记忆系统。

未来的研究方向应进一步聚焦于睡眠巩固算法的优化，即如何更高效地将非结构化的情景日志转化为结构化的语义图谱；以及硬件加速技术，以降低测试时训练（Titans）带来的额外计算开销。随着这些技术的成熟，我们有望见证新一代“不知疲倦、过目不忘且持续进化”的智能体的诞生，它们将彻底改变人机交互的深度与广度。


arxiv.org
Cognitive Memory in Large Language Models - arXiv
在新窗口中打开

medium.com
The Three Memory Types Every LLM Developer Must Know | by Sahil Nanga - Medium
在新窗口中打开

adasci.org
What Role Does Memory Play in the Performance of LLMs? | ADaSci Blog
在新窗口中打开

datacamp.com
How Does LLM Memory Work? Building Context-Aware AI Applications - DataCamp
在新窗口中打开

openreview.net
BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack
在新窗口中打开

arxiv.org
Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length - arXiv
在新窗口中打开

lawrence-emenike.medium.com
A Straightforward explanation of Parametric vs. Non-Parametric Memory in LLMs
在新窗口中打开

arunbaby.com
Memory Architectures - Arun Baby
在新窗口中打开

aurelio.ai
Conversational Memory in LangChain | Aurelio AI
在新窗口中打开

projectpro.io
Types of LangChain Memory and How to Use Them - ProjectPro
在新窗口中打开

langchain-doc.readthedocs.io
ConversationSummaryBufferMe
在新窗口中打开

informationmatters.org
MemGPT: Engineering Semantic Memory through Adaptive Retention and Context Summarization - Information Matters
在新窗口中打开

getfocal.co
MemGPT: A Deep Dive - Focal
在新窗口中打开

readwise-assets.s3.amazonaws.com
MemGPT: Towards LLMs as Operating Systems - AWS
在新窗口中打开

hioscar.ai
10: Memory & Retrieval for LLMs — OscarAI - Oscar Health
在新窗口中打开

pmc.ncbi.nlm.nih.gov
Integrating large language model-based agents into a virtual patient chatbot for clinical anamnesis training - PMC - NIH
在新窗口中打开

docs.cloud.google.com
GraphRAG infrastructure for generative AI using Vertex AI and Spanner Graph | Cloud Architecture Center
在新窗口中打开

ibm.com
What is GraphRAG? - IBM
在新窗口中打开

pureinsights.com
GraphRAG: When Your RAG Needs a Memory Palace - Pureinsights
在新窗口中打开

arxiv.org
Titans: Learning to Memorize at Test Time - arXiv
在新窗口中打开

openreview.net
Titans: Learning to Memorize at Test Time - OpenReview
在新窗口中打开

blog.trukhin.com
BYTEBURST #5 “The Rise of Neural Memory Systems and Test-Time Learning”
在新窗口中打开

arxiv.org
Titans: Learning to Memorize at Test Time - arXiv
在新窗口中打开

arxiv.org
Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory - arXiv
在新窗口中打开

arxiv.org
Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory - arXiv
在新窗口中打开

shaped.ai
Titans: Learning to Memorize at Test Time - A Breakthrough in Neural Memory Systems
在新窗口中打开

pub.towardsai.net
Inside MemGPT: An LLM Framework for Autonomous Agents Inspired by Operating Systems Architectures | by Jesus Rodriguez | Towards AI
在新窗口中打开

arxiv.org
A-Mem: Agentic Memory for LLM Agents - arXiv
在新窗口中打开

ai.plainenglish.io
Forgetting in AI Agent Memory Systems | by Volodymyr Pavlyshyn
在新窗口中打开

okoone.com
Fixing the way LLMs handle memory - Okoone
在新窗口中打开

github.com
BABILong: a long-context needle-in-a-haystack benchmark for LLMs - GitHub
在新窗口中打开

openreview.net
LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory
在新窗口中打开

emergentmind.com
LongMemEval: LLM Long-Term Memory Benchmark - Emergent Mind
在新窗口中打开

xiaowu0162.github.io
LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory - Di Wu
在新窗口中打开
