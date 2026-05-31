# RecSys Challenge 2026 - Music Conversational Recommendation System

## 比赛任务

Music-CRS (Music Conversational Recommendation System) 挑战赛聚焦于**对话式音乐推荐**场景。传统的静态推荐列表正在被动态的对话交互所取代，用户通过自然语言与 AI 交互，系统需要：

1. **自然语言理解 (NLU)** - 理解用户模糊、细粒度的音乐偏好
2. **多轮对话探索** - 通过对话深入挖掘用户的音乐品味
3. **上下文感知推荐** - 结合对话历史、用户画像、歌曲元数据进行推荐

### 数据集

| 数据集 | 规模 | 描述 |
|--------|------|------|
| TalkPlayData-Challenge-Dataset | 1k sessions | 多轮音乐对话，包含用户画像、对话目标、目标进度评估。每轮约8个对话轮次 |
| TalkPlayData-Challenge-Track-Metadata | 50.4k tracks | 歌曲元数据：名称、艺术家、专辑、标签、流行度、发行日期 |
| TalkPlayData-Challenge-User-Metadata | 9.09k users | 用户人口统计学：年龄、性别、国家 |
| TalkPlayData-Challenge-Track-Embeddings | 50.4k tracks | 预计算的歌曲嵌入向量 |
| TalkPlayData-Challenge-User-Embeddings | 9.09k users | 预计算的用户嵌入向量 |

---

## 比赛规则

### 关键约束

1. **必须使用全部曲目** - 推理阶段必须从整个曲库中检索候选曲目，不得使用 `track_split_types` 过滤子集
2. **提交格式** - 严格遵循 JSON 格式
3. **每个 turn 独立预测** - 需要为每个 session × turn 提供预测
4. **无重复 track_id** - 每个预测列表中的 track_id 必须唯一
5. **排序重要性** - track_id 按相关性排序（最相关的在前）

### 提交格式

```json
[
  {
    "session_id": "69137__2020-02-08",
    "user_id": "69137",
    "turn_number": 1,
    "predicted_track_ids": [
      "715f8aff-7c99-46b8-8f9d-6d1aa1ae0372",
      "73562c63-02e3-4278-baf3-aeb3252f8b33"
    ],
    "predicted_response": "Here are some songs you might enjoy."
  }
]
```

### 评估指标

#### 检索指标 (Retrieval Metrics)
- **nDCG@{1, 10, 20}** - 在所有 8 个对话轮次上宏平均

#### 多样性指标 (Diversity Metrics)
- **Catalog Diversity** - 推荐的不同曲目数 / 总曲目数，越高表示覆盖越广
- **Lexical Diversity (Distinct-2)** - 独特bigram数 / 总bigram数，越高表示词汇越丰富

### Baseline 结果 (Devset)

| Method | nDCG@1 | nDCG@10 | nDCG@20 | Catalog Diversity | Lexical Diversity |
|--------|-------:|--------:|--------:|------------------:|------------------:|
| Random | 0.0000 | 0.0001 | 0.0001 | 0.9652 | 0.0000 |
| Popularity | 0.0005 | 0.0018 | 0.0024 | 0.0004 | 0.0000 |
| LLaMA-1B + BM25 | 0.0098 | 0.0627 | 0.0815 | 0.3795 | 0.2558 |

---

## Baseline 架构

Baseline 采用**两阶段管道 (Two-Stage Pipeline)**：

```
用户输入 → [Stage 1: Retrieval] → Top-20 候选曲目 → [Stage 2: LLM] → 自然语言响应
```

### 核心组件

| 组件 | 实现 | 代码位置 |
|------|------|----------|
| LLM | Llama-3.2-1B-Instruct | `mcrs/lm_modules/llama.py` |
| Retrieval (BM25) | BM25 稀疏检索 | `mcrs/retrieval_modules/bm25.py` |
| Retrieval (BERT) | BERT 密集向量检索 | `mcrs/retrieval_modules/bert.py` |
| Item DB | 歌曲元数据存储 | `mcrs/db_item/music_catalog.py` |
| User DB | 用户画像存储 | `mcrs/db_user/user_profile.py` |

### Stage 1: Retrieval

**BM25 模式**：
- 将歌曲元数据（track_name, artist_name, album_name, release_date）拼接成文本
- 使用 BM25 索引构建全量曲库的倒排索引
- 查询时返回相关性最高的 Top-20 曲目

**BERT 模式**：
- 使用 BERT encoder 对歌曲元数据编码为向量
- 查询时计算余弦相似度，返回 Top-20 最相似曲目

### Stage 2: Response Generation

1. 拼接系统提示（Role + Response Generation + Personalization）
2. 组装对话历史
3. 将 Top-1 推荐歌曲信息注入
4. 使用 Llama-3.2-1B-Instruct 生成自然语言响应

### 关键代码流程

```python
# mcrs/crs_baseline.py
def chat(self, user_query, user_id=None):
    # 1. 保存对话历史
    self.session_memory.append({"role": "user", "content": user_query})

    # 2. 构建系统提示（含用户画像）
    system_prompt = self._get_system_prompt(user_id)

    # 3. 检索阶段 - 拼接所有对话历史作为查询
    retrieval_input = "\n".join([f"{c['role']}: {c['content']}" for c in self.session_memory])
    retrieval_items = self.retrieval.text_to_item_retrieval(retrieval_input, topk=20)

    # 4. 获取 Top-1 推荐歌曲元数据
    recommend_item = self.item_db.id_to_metadata(retrieval_items[0])

    # 5. LLM 生成响应
    response = self.lm.response_generation(system_prompt, self.session_memory, recommend_item)

    return {"retrieval_items": retrieval_items, "recommend_item": recommend_item, "response": response}
```

---

## 优化思路

### 1. 改进 Item Representation

**问题**：当前只使用 track_name, artist_name, album_name

**方向 A：扩展文本字段**
- 添加 genre tags、mood labels、release year、popularity scores
- 修改 `corpus_types` 配置加入 `tag_list`

**方向 B：融合音频特征**
- 使用 CLAP (Contrastive Language-Audio Pretraining) 提取音频特征
- 找到"听起来相似"而不仅是"描述相似"的歌曲

**方向 C：更好的嵌入模型**
- Qwen2.5-Embedding（多语言支持）
- E5 / BGE（当前最佳文本嵌入）
- Contriever（无监督对比学习）
- ColBERT（更精细的词级别匹配）

### 2. 添加 Reranker 模块

在初始检索后增加第二阶段排序：

**Embedding-based Reranking**：
- 利用预计算的用户嵌入和歌曲嵌入
- 计算 user-item 相似度进行重排
- 结合多模态信号：文本相关性 + 音频相似度 + 用户偏好

**LLM-based Reranking**：
- 对 Top-K 候选让 LLM 判断相关性
- 提示词："Rank these tracks by relevance to: {user_query}"
- 可用模型：Llama-3-8B, Qwen-7B

```python
# 扩展后的管道
retrieval_items = self.retrieval.text_to_item_retrieval(query, topk=100)
if self.reranker:
    retrieval_items = self.reranker.rerank(query, candidates=retrieval_items[:50], topk=20)
```

### 3. 生成式检索 (Generative Retrieval)

**概念**：用端到端生成替代 retrieve-then-generate

**Semantic IDs 方法**：
- 为每首歌曲分配层级语义 ID（如 `jazz/smooth/piano/0042`）
- Fine-tune LLM 直接根据用户 query 生成 track IDs
- 单模型同时完成检索和生成

**优势**：
- 统一架构，简化流程
- 可建模复杂用户意图
- 充分利用 LLM 的推理能力

### 4. 对话历史建模优化

当前方案简单拼接所有历史作为检索 query，可探索：

- **注意力机制**：对历史对话进行加权
- **上下文压缩**：总结或提取关键偏好信息
- **用户意图追踪**：识别用户偏好的演变

### 5. 用户画像增强

当前用户画像仅包含年龄、性别、国家，可补充：

- 基于历史交互构建用户音乐品味向量
- 利用 TalkPlayData-Challenge-User-Embeddings 预计算嵌入
- 融合协同过滤信号

---

## 重要提醒

1. **必须使用 `all_tracks`** - 配置中 `track_split_types` 必须为 `["all_tracks"]`
2. **Blind Set 评估** - 需要提交到 [CodaBench](https://www.codabench.org/) 进行盲测
3. **时间线** - 2026年6月30日截止提交，9月 ACM RecSys 2026 公布结果

---

## 快速开始

```bash
# 安装
uv venv .venv --python=3.10
source .venv/bin/activate
uv pip install -e .
uv pip install flash-attn --no-build-isolation

# BM25 baseline 推理
python run_inference_devset.py --tid llama1b_bm25_devset --batch_size 16

# BERT baseline 推理
python run_inference_devset.py --tid llama1b_bert_devset --batch_size 16

# 评估
python -m music_crs_evaluator.evaluate_devset --eval_dataset devset --tid <tid>
```
