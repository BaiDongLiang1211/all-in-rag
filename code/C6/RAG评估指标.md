# RAG 评估指标详解：计算方法、答案正确性与指标选择

> 适用对象：正在学习或搭建“检索 → 重排序 → 上下文压缩 → 生成答案”式 RAG 系统的开发者。  
> 核心原则：**检索到了吗？回答有依据吗？回答正确且完整吗？** 三个问题应分开评估。

## 1. RAG 评估到底评什么？

### 1.1 RAG 三元组

| 维度 | 比较什么 | 典型问题 | 常用评估方式 |
|---|---|---|---|
| 上下文相关性（Context Relevance） | 查询 vs 检索上下文 | 找到的材料是否相关？ | Precision@K、Recall@K、MRR、MAP、NDCG；或 LLM 评价 |
| 忠实度（Faithfulness / Groundedness） | 生成答案 vs 检索上下文 | 答案中的声明是否能得到材料支持？ | 逐事实声明核验、LLM 评价 |
| 答案相关性（Answer Relevance） | 生成答案 vs 用户问题 | 是否切题、有没有答非所问？ | LLM 评价、问题与答案语义评价 |

**三元组不是全部。** 还应独立评估：**答案正确性与完整性**（通常需要参考答案或事实真值）、不可回答问题的拒答能力，以及延迟、成本、鲁棒性。

### 1.2 最容易混淆的几种“正确”

- **检索相关**：某个文档是否与查询相关。
- **有依据（Faithfulness）**：回答是否被*提供给模型的上下文*支持。
- **事实正确（Factual Correctness）**：回答是否与*真实事实或参考答案*一致。
- **完整（Completeness）**：参考答案要求覆盖的关键事实是否都回答了。
- **切题（Answer Relevance）**：是否回应了用户的问题。

例如：检索上下文错误地声称“某公司成立于 2010 年”，模型照抄该年份。此时回答可能**忠实度高**，但对真实事实的**正确性低**。二者不能相互代替。

---

## 2. Accuracy、Precision、Recall、F1：不要混为一谈

### 2.1 分类意义上的 Accuracy（准确率）

如果把文档是否相关看成二分类任务：

| | 实际相关 | 实际不相关 |
|---|---:|---:|
| 判为相关 | TP | FP |
| 判为不相关 | FN | TN |

公式：

$$
\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

$$
\mathrm{Precision}=\frac{TP}{TP+FP},\qquad
\mathrm{Recall}=\frac{TP}{TP+FN}
$$

$$
F_1=\frac{2PR}{P+R}=\frac{2TP}{2TP+FP+FN}
$$

**例子：** 100 篇文档里只有 10 篇相关。系统正确找出 8 篇，误把 5 篇不相关文档判为相关。

- TP = 8，FN = 2，FP = 5，TN = 85。
- Accuracy = (8 + 85) / 100 = **0.93**。
- Precision = 8 / (8 + 5) ≈ **0.615**。
- Recall = 8 / (8 + 2) = **0.80**。

**为什么检索评估不优先看普通 Accuracy？** 大型知识库中不相关文档非常多，TN 往往很大：即使检索结果夹杂较多无关文档，Accuracy 也可能很高。实际 RAG 检索通常关注前 K 个结果，因此排序指标更有针对性。

> 注意：RAG 问答领域有人把“Accuracy”用来指**答案正确率**或 Exact Match（严格匹配率）。它与上述“文档相关性二分类 Accuracy”不是同一个评估定义，报告时应写清楚。

### 2.2 Precision@K 与 Recall@K（检索指标）

$$
\mathrm{Precision}@K=\frac{\text{Top-K 中相关文档数}}{K}
$$

$$
\mathrm{Recall}@K=\frac{\text{Top-K 中相关文档数}}{\text{该问题的全部真实相关文档数}}
$$

例如，知识库有 4 篇相关文档，前 5 个检索结果中有 3 篇相关：

- Precision@5 = 3/5 = **0.60**。
- Recall@5 = 3/4 = **0.75**。
- F1@5 = 2×0.60×0.75/(0.60+0.75) ≈ **0.667**。

它们能评估“找得准、找得全”，但**不区分这 3 篇相关文档排在第 1、2、3 位还是第 3、4、5 位**。

### 2.3 MRR、AP、MAP：加入排序信息

**MRR（平均倒数排名）**只关注**第一篇相关文档**的位置：

$$
\mathrm{MRR}=\frac{1}{|Q|}\sum_{q\in Q}\frac{1}{\operatorname{rank}_q}
$$

如果某个查询的第一篇相关文档位于第 4 名，则该查询的 Reciprocal Rank = 1/4 = 0.25；如果没有找到相关文档，通常记为 0。

**AP（Average Precision，单个查询的平均精确率）**关注所有相关文档的排名。二元相关性下常见定义：

$$
\mathrm{AP}(q)=\frac{1}{R_q}\sum_{k=1}^{N}\mathrm{Precision}@k\cdot\mathrm{rel}_q(k)
$$

其中：
- $R_q$：该查询在评估语料中的全部真实相关文档数；
- $N$：参与评价的结果列表长度；
- $\mathrm{rel}_q(k)$：第 $k$ 位相关为 1，否则为 0。

**例子：** 某问题总共 3 篇相关文档，检索顺序是 `[相关, 不相关, 相关, 相关, 不相关]`。

| 排名 | 相关？ | Precision@k | 对 AP 的贡献 |
|---:|:---:|---:|---:|
| 1 | 1 | 1/1 = 1.000 | 1.000 |
| 2 | 0 | 1/2 = 0.500 | 0 |
| 3 | 1 | 2/3 ≈ 0.667 | 0.667 |
| 4 | 1 | 3/4 = 0.750 | 0.750 |
| 5 | 0 | 3/5 = 0.600 | 0 |

$$
\mathrm{AP}=\frac{1+2/3+3/4}{3}\approx\boxed{0.806}
$$

**MAP（Mean Average Precision）**是多个查询 AP 的平均值：

$$
\mathrm{MAP}=\frac{1}{|Q|}\sum_{q\in Q}\mathrm{AP}(q)
$$

例如三个问题的 AP 分别为 0.80、0.90、0.70，则 MAP = **0.80**。

> 注意：AP@K 存在不同实现约定。有的实现用全部相关文档数 $R_q$ 做分母，有的截断版本使用 $\min(R_q,K)$。对比系统时必须固定公式、K 值和相关性标注。

**NDCG@K（归一化折损累计增益）**适用于“不相关、部分相关、高度相关”等多级相关性标签：

$$
\mathrm{DCG}@K=\sum_{k=1}^{K}\frac{2^{g_k}-1}{\log_2(k+1)},\qquad
\mathrm{NDCG}@K=\frac{\mathrm{DCG}@K}{\mathrm{IDCG}@K}
$$

其中 $g_k$ 是第 $k$ 位的相关性等级，IDCG 是理想排序的 DCG。它既考虑相关程度，又给予靠前文档更高权重。

---

## 3. 答案正确性与完整性（重点）：到底怎么算？

### 3.1 没有唯一的“Answer Correctness”公式

不同项目与评估框架可能采用：

1. **Exact Match（完全匹配）**：用于单一、明确的短答案；
2. **事实点匹配**：将参考答案与生成答案拆成事实声明，再计算 TP、FP、FN；
3. **语义相似度**：衡量两种表达是否语义接近；
4. **LLM-as-a-Judge**：由评估模型按评分规则判定事实正确、覆盖完整程度；
5. **组合得分**：例如事实匹配得分与语义相似度加权。

对于知识问答 RAG，**事实点匹配 + 完整性评价**通常比简单的字符串完全匹配更容易解释。

### 3.2 手算实例：AI 有哪些缺陷？

**用户问题：** AI 目前有哪些主要缺陷？

**参考答案包含四个事实点：**

1. 容易产生幻觉；
2. 可能存在算法偏见；
3. 存在数据隐私风险；
4. 模型往往消耗大量算力。

**模型生成的回答：**

> AI 容易产生幻觉，也可能存在算法偏见，但是 AI 模型几乎不消耗算力。

逐事实检查：

| 对应事实 | 评价 | 统计 |
|---|---|---|
| 正确提到“容易产生幻觉” | 命中参考事实 | TP +1 |
| 正确提到“算法偏见” | 命中参考事实 | TP +1 |
| 没提到“数据隐私风险” | 遗漏参考事实 | FN +1 |
| 未正确陈述“消耗大量算力” | 遗漏正确事实 | FN +1 |
| 错误声称“几乎不消耗算力” | 错误陈述 | FP +1 |

于是 $TP=2,\ FP=1,\ FN=2$。

**（1）事实精确率（Factual Precision）：** 回答中提出的事实有多少正确？

$$
P_{\mathrm{fact}}=\frac{TP}{TP+FP}=\frac{2}{3}\approx\boxed{0.667}
$$

**（2）事实召回率（Factual Recall）：** 参考答案应覆盖的事实有多少被正确说出？

$$
R_{\mathrm{fact}}=\frac{TP}{TP+FN}=\frac24=\boxed{0.500}
$$

这里的 **Factual Recall 可作为“事实完整性”的一种操作化指标**，不同于检索的 Recall@K。

**（3）事实 F1：** 同时考虑说对多少和有没有说全。

$$
F_{1,\mathrm{fact}}=\frac{2TP}{2TP+FP+FN}
=\frac{4}{4+1+2}=\boxed{0.571}
$$

**这个例子如何理解：** 模型说出来的三个事实中两个正确，故事实 Precision 较高；但四个应覆盖事实只覆盖两个，故完整性（事实 Recall）只有 0.5。

> 事实拆分与匹配存在判断规则：同一句错误陈述可以同时导致一个 FP（说错）和一个 FN（漏掉正确事实），如上面的“算力”例子。应统一标注规范，不应把“未被参考答案逐字写出”一律当作错误，需允许事实等价与合理释义。

### 3.3 为什么有时还要结合语义相似度？

相同事实可以用完全不同的措辞表达。例如“算法存在偏差”和“模型可能有偏见”可能语义接近，纯字符串匹配无法正确识别。

因此，一些 Answer Correctness 实现把事实匹配得分 $F$ 与答案-参考答案语义相似度 $S$ 组合：

$$
AC=w_fF+w_sS,\qquad w_f+w_s=1
$$

**演示计算（非通用默认参数）：** 假设 $F=0.571$、$S=0.800$，人为取 $w_f=0.75$、$w_s=0.25$：

$$
AC=0.75\times0.571+0.25\times0.800\approx\boxed{0.628}
$$

- $0.800$ 是**假设的语义分数**，不是通过本例实际测出来的。
- 权重 **0.75/0.25 仅用于演示**，不同版本/框架的默认值与计算方式可能不同。
- 语义相似不代表事实必然正确；如果评价任务强调事实核查，应给予事实匹配更多关注。

### 3.4 在 Ragas 中怎样做？

Ragas 的相关指标名称会随版本和 API 风格变化。一个常见区分是：

- **FactualCorrectness**：根据参考答案和生成答案进行事实声明匹配，支持事实 Precision、Recall、F1 等模式。
- **AnswerCorrectness**：在部分版本/实现中，将事实匹配与语义相似性等信号组合。
- **Faithfulness**：不是和参考答案对比，而是检查生成答案是否由检索上下文支持。

下面是**使用支持 `ragas.metrics.collections.FactualCorrectness` 的版本**的一种写法（实际使用时请核对已安装版本的 API）：

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import FactualCorrectness

client = AsyncOpenAI()   # 需要配置 API Key
judge_llm = llm_factory("gpt-4o-mini", client=client)

metric = FactualCorrectness(
    llm=judge_llm,
    mode="f1",
    atomicity="high",
    coverage="high",
)

result = await metric.ascore(
    response="AI容易产生幻觉，也可能存在算法偏见，但是AI模型几乎不消耗算力。",
    reference="AI容易产生幻觉、存在算法偏见、存在数据隐私风险，并且消耗大量算力。",
)
print(result.value)
```

这个例子是演示接口，不保证不同版本得到相同的事实划分或分数；`await` 需要在异步函数或支持顶层 `await` 的环境中运行。若使用 DashScope 兼容接口，需按该环境实际支持的客户端和评估模型做适配。

### 3.5 Faithfulness 与 Answer Correctness 是两张不同的“考卷”

| 指标 | 对比依据 | 低分意味着什么 |
|---|---|---|
| Faithfulness | 生成答案 vs **检索上下文** | 答案出现上下文未支持的陈述 |
| Factual Precision | 生成答案 vs **参考事实** | 回答中有错误事实 |
| Factual Recall / Completeness | 生成答案 vs **参考事实** | 漏掉应回答的事实 |
| Factual F1 | 上述事实 Precision、Recall | 正确性与完整性没有兼顾 |
| Answer Relevance | 生成答案 vs **原始问题** | 答非所问或没有直接回应 |

需要特别注意：**Faithfulness 高，不自动保证 Factual Correctness 高；Answer Relevance 高，也不保证两者高。**

---

## 4. n-gram Precision：BLEU、ROUGE 为什么这么算？

### 4.1 什么是 n-gram？

n-gram 是连续的 $n$ 个 Token（词或字，依分词方案确定）。

例如已分词句子：`我 / 喜欢 / 吃 / 苹果`

- 1-gram：`我`、`喜欢`、`吃`、`苹果`，共 4 个；
- 2-gram：`我 喜欢`、`喜欢 吃`、`吃 苹果`，共 3 个；
- 3-gram：`我 喜欢 吃`、`喜欢 吃 苹果`，共 2 个。

若句子长度为 $L$ 个 Token，且 $L\ge n$，那么 n-gram 总数为 $L-n+1$。

### 4.2 手算 1-gram、2-gram Precision

**参考答案：** `我 / 喜欢 / 吃 / 新鲜 / 苹果`（5 个 Token）  
**生成答案：** `我 / 喜欢 / 吃 / 苹果`（4 个 Token）

**1-gram：** 生成答案的 4 个 Token 均在参考答案中出现。

$$
p_1=\frac44=\boxed{1.000}
$$

**2-gram：**

- 参考答案：`我 喜欢`、`喜欢 吃`、`吃 新鲜`、`新鲜 苹果`；
- 生成答案：`我 喜欢`（✓）、`喜欢 吃`（✓）、`吃 苹果`（✗）。

虽然“吃”和“苹果”都在参考答案里出现，但**它们在参考答案中不相邻**，因此“吃 苹果”不是匹配的 2-gram。

$$
p_2=\frac23\approx\boxed{0.667}
$$

BLEU 使用的是**修正后的 n-gram Precision（Modified Precision）**：匹配次数要受到参考答案中同一 n-gram 出现次数的上限约束，避免生成文本通过重复同一个词刷分。

### 4.3 BLEU 为什么还要长度惩罚？

以 BLEU-2（对 1-gram、2-gram 赋相同权重）为例：

$$
\mathrm{BLEU}_2=BP\times\sqrt{p_1p_2}
$$

其中 $BP$（Brevity Penalty，长度惩罚）为：

$$
BP=\begin{cases}
1,&c\ge r\\
e^{1-r/c},&c<r
\end{cases}
$$

$c$ 是生成答案长度，$r$ 是参考答案长度。此例 $c=4$、$r=5$：

$$
BP=e^{1-5/4}=e^{-0.25}\approx0.779
$$

$$
\mathrm{BLEU}_2\approx0.779\times\sqrt{1\times2/3}\approx\boxed{0.636}
$$

> 这是示范 BLEU-2 的手算，未使用平滑。完整 BLEU 通常涉及 1～4-gram、多参考答案匹配、语料级累计等实现细节；分数并不能直接被解释为“事实正确的概率”。

### 4.4 ROUGE-N 与 BLEU 的关注点不同

ROUGE-N 的召回率形式：

$$
\mathrm{ROUGE}\text{-}N_{\mathrm{recall}}=\frac{\text{匹配的 n-gram 数量}}{\text{参考答案中的 n-gram 总数}}
$$

同样的例子：

| 指标 | 1-gram | 2-gram |
|---|---:|---:|
| 生成侧 Precision | 4/4 = 1.000 | 2/3 ≈ 0.667 |
| 参考侧 Recall | 4/5 = 0.800 | 2/4 = 0.500 |

**理解方式：** BLEU 更偏重生成文本的 n-gram 匹配精度，并通过 BP 惩罚过短输出；ROUGE-N 召回率形式更强调参考答案中的内容覆盖。不同 ROUGE 实现也可能报告 Precision 和 F1。

**特别注意中文分词：** 字级与词级分词会得到不同 n-gram 与不同分数。必须在不同实验间保持分词、大小写、标点、归一化规则一致。

**关于原教程示例的笔误：** `狗 在 床 上面` 按空格分词只有 **4** 个 Token，不是 5 个；`上面` 与 `上` 也不是完全相同的 Token，因此不能直接按“4/5”或“全部匹配”推算。

---

## 5. RAG 到底该优先评估哪些指标？

**没有对所有 RAG 应用都适用的唯一“最重要指标”。** 对于以事实问答为主、重视答案质量的 RAG 系统，我建议：

| 优先级 | 指标 | 关键问题 |
|---|---|---|
| 核心 | **关键证据 Recall@K / 证据覆盖率** | 正确回答所需的证据是否找到了？ |
| 核心 | **Faithfulness** | 回答中的事实是否得到检索材料支持？ |
| 核心 | **答案事实正确性与完整性**（Factual Precision、Recall、F1；或经过校准的 Answer Correctness） | 有没有说错？有没有漏答？ |
| 重要 | Precision@K、上下文噪声比例 | 是否混入太多不相关内容？ |
| 重要 | MRR / MAP / NDCG@K | 更相关的材料是否排在前面？ |
| 重要 | Answer Relevance | 是否紧扣原始问题？ |
| 辅助 | ROUGE、BLEU、METEOR | 与参考答案在词语、短语及部分语义上是否一致？ |
| 必需的工程指标 | 延迟、Token 成本、故障率、拒答表现 | 系统是否能真正稳定上线？ |

**为什么不把 BLEU/ROUGE 排第一？** 同一事实可能有多种正确表达，词汇重叠低不一定意味着答案错误；反过来，文本和参考答案很像，也不代表没有幻觉或遗漏。

### 5.1 根据业务类型调整优先级

| 场景 | 更应优先关注 |
|---|---|
| 企业知识库问答 | 证据召回、事实正确性、Faithfulness |
| 法律/医疗等高风险问答 | 事实与来源核验、覆盖度、拒答能力、风险控制 |
| FAQ/短答案问答 | 答案正确率或 Exact Match、Hit@K、延迟 |
| 多文档综合问答 | 多证据覆盖、答案完整性、Faithfulness |
| 搜索/文档推荐 | NDCG@K、MAP、MRR、Precision@K |
| 摘要任务 | 事实一致性、关键内容覆盖、ROUGE（辅助） |

如果一定要为**最终用户体验**选一个主指标，可选择经过人工校准的**答案事实正确性与完整性综合评分**；但调试系统时，一定要同时看**检索证据召回**与**忠实度**，否则无法诊断问题出在哪一步。

---

## 6. 你的 FAISS + ColBERT + LLM 压缩实验，应怎样评估？

假设比较三套系统：

- **方案 A**：FAISS 基础检索；
- **方案 B**：FAISS + ColBERT 重排序；
- **方案 C**：FAISS + ColBERT + LLMChainExtractor 压缩。

建议固定：测试问题、知识库版本、嵌入模型、生成模型、Prompt、温度及评估器版本。每个问题记录标准答案、相关证据 ID、召回文档、最终提供给生成器的上下文和最终答案。

| 指标 | 方案 A | 方案 B | 方案 C | 要检查什么 |
|---|---|---|---|---|
| Recall@5 / 证据覆盖 | 待测 | 待测 | 待测 | 重排/压缩是否丢失必需证据？ |
| Precision@5 | 待测 | 待测 | 待测 | 前五条材料有多少真正相关？ |
| MAP@5 / NDCG@5 | 待测 | 待测 | 待测 | 排名质量是否改善？ |
| Faithfulness | 待测 | 待测 | 待测 | 回答是否始终有证据支持？ |
| Factual Precision | 待测 | 待测 | 待测 | 有没有新增错误事实？ |
| Factual Recall | 待测 | 待测 | 待测 | 必要事实回答得全不全？ |
| Factual F1 | 待测 | 待测 | 待测 | 正确性与完整性的综合变化？ |
| Token 消耗 / 时延 | 待测 | 待测 | 待测 | 质量提升是否值得额外成本？ |

注意：LLM 压缩会改变 `page_content`，有时还会改变文本块边界。比较检索召回时，最好保留原始来源和证据 ID，或定义统一的事实证据单元，避免由于切分与压缩方式不同造成无法公平比较。

### 6.1 一个可直接运行的 Python 计算示例

下面展示如何在**人工完成事实匹配、得到 TP/FP/FN** 后计算几个核心指标；代码并不会自动替你判断事实对错。

```python
# 事实正确性与完整性
TP, FP, FN = 2, 1, 2

factual_precision = TP / (TP + FP) if TP + FP else 0.0
factual_recall = TP / (TP + FN) if TP + FN else 0.0
factual_f1 = (2 * TP / (2 * TP + FP + FN)
              if 2 * TP + FP + FN else 0.0)

print("Factual Precision:", round(factual_precision, 3))  # 0.667
print("Factual Recall:", round(factual_recall, 3))        # 0.5
print("Factual F1:", round(factual_f1, 3))                # 0.571

# 检索 AP（3 篇相关文档都在返回列表中）
relevance = [1, 0, 1, 1, 0]
total_relevant = 3
hits = 0
precision_sum = 0.0
for rank, is_relevant in enumerate(relevance, start=1):
    if is_relevant:
        hits += 1
        precision_sum += hits / rank
ap = precision_sum / total_relevant
print("AP:", round(ap, 3))                               # 0.806
```

---

## 7. 一页总结：指标回答的到底是什么问题？

| 你真正想问的事 | 指标 |
|---|---|
| 所有相关证据找回了多少？ | **Recall@K** |
| 前 K 条检索结果有多少相关？ | **Precision@K** |
| 第一个正确结果来得够早吗？ | **MRR** |
| 所有相关文档整体排得好吗？ | **AP / MAP / NDCG** |
| 回答所说的事实有多少正确？ | **Factual Precision** |
| 标准答案要求的事实有多少回答到了？ | **Factual Recall / Completeness** |
| 正确与完整如何平衡？ | **Factual F1** |
| 回答有没有检索材料依据？ | **Faithfulness** |
| 回答是否切题？ | **Answer Relevance** |
| 与标准答案的 n-gram 有多像？ | **BLEU、ROUGE-N** |
| 系统运行是否划算、可靠？ | **延迟、成本、失败率、拒答表现** |

**最终结论：** 典型知识问答 RAG 的首要评估组合应包含**证据召回 + 忠实度 + 答案事实正确性与完整性**。其他指标用于补充排序质量、文本形式、运行性能，而不是取代这三个核心问题。

### 参考阅读

- [Ragas：Factual Correctness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/factual_correctness/)
- [Ragas：Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
- [Ragas 官方文档](https://docs.ragas.io/)
- [TruLens：RAG Triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/)

> 文中计算例子与数值均为教学示例；具体项目的评估公式、数据标注规范和阈值应在实验前固定并记录。
