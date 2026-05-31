# RecSys Challenge 2026 - 优化方案

## 概述

当前 Baseline (LLaMA-1B + BM25) 在 Devset 上的表现为：
- nDCG@10: **0.0627**
- nDCG@20: **0.0815**
- Catalog Diversity: **0.3795**
- Lexical Diversity: **0.2558**

优化目标：**显著提升 nDCG 的同时保持或提升 Diversity 指标**。

---

## 一、Item Representation 优化

### 1.1 扩展文本特征

**现状**：仅使用 `track_name`, `artist_name`, `album_name`, `release_date`

**优化方案**：
```python
# mcrs/db_item/music_catalog.py - 修改 _stringify_metadata 方法

def _stringify_metadata(self, metadata: Dict) -> str:
    fields = [
        f"track_name: {metadata['track_name']}",
        f"artist_name: {metadata['artist_name']}",
        f"album_name: {metadata['album_name']}",
        f"release_date: {metadata['release_date']}",
        f"genre_tags: {', '.join(metadata.get('tag_list', []))}",  # 新增
        f"popularity: {metadata.get('popularity', 0)}",  # 新增
    ]
    return "\n".join(fields)
```

**预期收益**：nDCG +5%~10%（更多语义信息有助于检索匹配）

### 1.2 融合音频特征

使用 **CLAP** (Contrastive Language-Audio Pretraining) 提取音频 embedding：

```python
# mcrs/db_item/audio_features.py

import torch
from transformers import ClapProcessor, ClapModel

class AudioFeatureExtractor:
    def __init__(self, model_name="laion/clap-htsat-fused"):
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def extract_embedding(self, audio_path: str) -> torch.Tensor:
        inputs = self.processor(audio_type="audio", audios=[audio_path], return_tensors="pt")
        embeddings = self.model.get_audio_features(**inputs)
        return embeddings

# 在检索时结合文本和音频相似度
def hybrid_retrieval(text_score, audio_score, alpha=0.7):
    return alpha * text_score + (1 - alpha) * audio_score
```

**预期收益**：nDCG +10%~15%（音频特征能捕捉"听起来相似"而非仅"描述相似"）

### 1.3 使用更强的 Embedding 模型

替换 BERT 为更先进的文本嵌入模型：

| 模型 | 特点 | 预期收益 |
|------|------|----------|
| **E5-mistral-7b** | 当前 SOTA 文本嵌入 | nDCG +15%~20% |
| **BGE-large-zh** | 中英双语支持 | nDCG +10%~15% |
| **Qwen2.5-Embedding** | 多语言、阿里开源 | nDCG +10%~15% |
| **ColBERT** | 词级别 late interaction | nDCG +8%~12% |

```python
# mcrs/retrieval_modules/e5_retriever.py

from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class E5Retriever:
    def __init__(self, model_name="intfloat/e5-mistral-7b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def encode(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        # E5 需要添加 "query: " 或 "passage: " 前缀
        prefixed_texts = [f"passage: {t}" for t in texts]
        all_embeddings = []
        for i in range(0, len(prefixed_texts), batch_size):
            batch = prefixed_texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)

    def text_to_item_retrieval(self, query: str, topk: int = 20):
        query_emb = self.encode([f"query: {query}"])
        scores = F.cosine_similarity(query_emb, self.item_embeddings)
        top_indices = scores.topk(topk).indices
        return [self.track_ids[i] for i in top_indices]
```

---

## 二、Retrieval 模块优化

### 2.1 混合检索策略

**问题**：单一检索方式难以覆盖所有 query 类型

**解决方案**：融合 BM25 + Dense Retrieval

```python
# mcrs/retrieval_modules/hybrid_retriever.py

class HybridRetriever:
    def __init__(self, bm25_model, dense_model, weight_bm25=0.4, weight_dense=0.6):
        self.bm25_model = bm25_model
        self.dense_model = dense_model
        self.weight_bm25 = weight_bm25
        self.weight_dense = weight_dense

    def text_to_item_retrieval(self, query: str, topk: int = 100):
        # BM25 召回
        bm25_scores = self.bm25_model.get_scores(query)  # 需要添加此方法
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())

        # Dense 召回
        dense_scores = self.dense_model.get_scores(query)

        # 加权融合
        hybrid_scores = self.weight_bm25 * bm25_scores + self.weight_dense * dense_scores
        top_indices = hybrid_scores.topk(topk).indices

        return [self.track_ids[i] for i in top_indices]
```

### 2.2 Query 改写增强

利用 LLM 改写用户 query，提升检索召回：

```python
# mcrs/retrieval_modules/query_expansion.py

class QueryExpander:
    def __init__(self, lm_model, tokenizer):
        self.lm = lm_model
        self.tokenizer = tokenizer

    def expand_query(self, user_query: str, chat_history: list) -> list[str]:
        prompt = f"""Given the conversation history and current query, generate 3 diverse search queries:
Conversation History:
{self._format_history(chat_history)}

Current Query: {user_query}

Generate 3 alternative search queries that capture different aspects of the user's intent.
Output format: One query per line"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.lm.generate(**inputs, max_new_tokens=100)
        expanded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        queries = [user_query] + [q.strip() for q in expanded.split("\n") if q.strip()]
        return queries[:4]  # 最多4个查询

    def multi_query_retrieval(self, query: str, retriever, topk_per_query: int = 10):
        expanded_queries = self.expand_query(query, chat_history)
        all_items = []
        for q in expanded_queries:
            items = retriever.text_to_item_retrieval(q, topk=topk_per_query)
            all_items.extend(items)

        # MMR (Maximal Marginal Relevance) 去重
        return self._mmr_dedup(all_items, topk=20)

    def _mmr_dedup(self, items: list, topk: int) -> list:
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
                if len(result) >= topk:
                    break
        return result
```

**预期收益**：nDCG +8%~12%（Query 改写能捕获更多相关结果）

### 2.3 对话上下文感知检索

**问题**：当前简单拼接所有对话历史，未考虑上下文重要性

**解决方案**：
```python
# mcrs/retrieval_modules/context_aware_retriever.py

class ContextAwareRetriever:
    def __init__(self, base_retriever, importance_scorer):
        self.base_retriever = base_retriever
        self.importance_scorer = importance_scorer

    def _compute_turn_weights(self, chat_history: list) -> list[float]:
        weights = []
        for turn in chat_history:
            role = turn.get("role", "")
            content = turn.get("content", "")

            # 角色权重：用户 query 通常更重要
            role_weight = 1.5 if role == "user" else 1.0

            # 内容权重：包含偏好关键词的 turn 权重更高
            pref_keywords = ["like", "love", "prefer", "enjoy", "want", "need", "song", "music", "artist"]
            content_weight = 1.0 + 0.5 * sum(1 for kw in pref_keywords if kw in content.lower())

            weights.append(role_weight * content_weight)

        # 近期权重衰减（越近的 turn 权重越高）
        decay_weights = [w * (1 + 0.1 * i) for i, w in enumerate(reversed(weights))]

        # 归一化
        total = sum(decay_weights)
        return [w / total for w in decay_weights]

    def _build_weighted_query(self, chat_history: list, weights: list[float]) -> str:
        query_parts = []
        for turn, weight in zip(chat_history, weights):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            # 重复重要内容以增加权重
            repeats = max(1, int(weight * 3))
            query_parts.append(f"{role}: {content} " * repeats)
        return "\n".join(query_parts)

    def text_to_item_retrieval(self, user_query: str, chat_history: list, topk: int = 20):
        # 计算权重并构建加权查询
        weights = self._compute_turn_weights(chat_history)
        weighted_query = self._build_weighted_query(chat_history, weights)
        combined_query = f"{weighted_query}\nuser: {user_query}"

        return self.base_retriever.text_to_item_retrieval(combined_query, topk)
```

---

## 三、Reranker 模块

### 3.1 Embedding-based Reranker

利用预计算的用户和歌曲嵌入进行重排：

```python
# mcrs/reranker/embedding_reranker.py

from datasets import load_dataset
import torch
import torch.nn.functional as F

class EmbeddingReranker:
    def __init__(self, item_embeddings_path, user_embeddings_path, device="cuda"):
        self.device = device

        # 加载预计算嵌入
        self.item_embeddings = torch.load(item_embeddings_path)
        self.user_embeddings = torch.load(user_embeddings_path)
        self.track_ids = self.item_embeddings["track_ids"]
        self.item_embs = self.item_embeddings["embeddings"].to(device)
        self.user_embs = self.user_embeddings["embeddings"].to(device)

        self.track_id_to_idx = {tid: idx for idx, tid in enumerate(self.track_ids)}

    def get_user_preference_vector(self, user_id: str, listening_history: list[str]) -> torch.Tensor:
        """构建用户偏好向量"""
        if listening_history:
            history_embs = []
            for track_id in listening_history:
                if track_id in self.track_id_to_idx:
                    history_embs.append(self.item_embs[self.track_id_to_idx[track_id]])
            if history_embs:
                return torch.stack(history_embs).mean(dim=0)

        # Fallback: 使用用户预计算嵌入
        if user_id in self.user_embs:
            return self.user_embs[user_id]
        return torch.zeros(self.item_embs.shape[1]).to(self.device)

    def rerank(self, user_id: str, query: str, candidates: list[str],
               listening_history: list[str], topk: int = 20) -> list[str]:
        user_pref = self.get_user_preference_vector(user_id, listening_history)

        # 计算每个候选的得分
        scores = []
        for track_id in candidates:
            if track_id in self.track_id_to_idx:
                item_emb = self.item_embs[self.track_id_to_idx[track_id]]
                # 用户偏好相似度
                pref_score = F.cosine_similarity(user_pref.unsqueeze(0), item_emb.unsqueeze(0)).item()
                scores.append((track_id, pref_score))
            else:
                scores.append((track_id, 0.0))

        # 按得分排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return [track_id for track_id, _ in scores[:topk]]
```

### 3.2 LLM-based Reranker

使用 LLM 对候选曲目进行重排：

```python
# mcrs/reranker/llm_reranker.py

class LLMReranker:
    def __init__(self, lm_model, tokenizer, item_db):
        self.lm = lm_model
        self.tokenizer = tokenizer
        self.item_db = item_db

    def rerank(self, query: str, candidates: list[str], topk: int = 10) -> list[str]:
        # 获取候选曲目元数据
        candidate_info = []
        for track_id in candidates[:20]:  # 最多 rerank 20 个
            meta = self.item_db.id_to_metadata(track_id)
            candidate_info.append(f"- {meta['track_name']} by {meta['artist_name']}")

        prompt = f"""Given the user query, rank these music tracks by relevance (most relevant first).
Only output the track names in ranked order, one per line.

User Query: {query}

Candidate Tracks:
{chr(10).join(candidate_info)}

Ranked List:"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.lm.generate(**inputs, max_new_tokens=200)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 解析 LLM 输出，映射回 track_ids
        ranked_tracks = self._parse_reranked_response(response, candidates)

        # 如果解析失败，使用原顺序
        return ranked_tracks if ranked_tracks else candidates[:topk]

    def _parse_reranked_response(self, response: str, candidates: list[str]) -> list[str]:
        # 简单解析：提取响应中的歌曲名，匹配回 candidates
        # 这里需要更 robust 的解析逻辑
        ranked = []
        for line in response.split("\n"):
            line = line.strip()
            if line and not line.startswith("-") and not line.startswith("*"):
                # 尝试匹配
                for track_id in candidates:
                    meta = self.item_db.id_to_metadata(track_id)
                    if meta['track_name'].lower() in line.lower():
                        if track_id not in ranked:
                            ranked.append(track_id)
        return ranked[:10]
```

**预期收益**：nDCG@1 +5%~10%（更精准的 Top-1 排序）

---

## 四、Response Generation 优化

### 4.1 改进 System Prompt

```python
# mcrs/system_prompts/response_generation_v2.txt

Based on the user query and the recommended track from tool calling results, provide a brief response that:
1. MUST base your response on the previously recommended track from the tool calling results
2. If the recommended track doesn't match the user's query, apologize and acknowledge the mismatch
3. If it's a good match, acknowledge that you've found music that matches their request with enthusiasm
4. Share key details: title, artist, genre tags, mood/style, release year
5. Briefly explain WHY this track matches their specific request
6. Add a PERSONALIZED touch based on user's listening history and demographics
7. Invite further interaction

Response Format:
- Keep it concise (2-3 sentences)
- Use natural, conversational tone
- Highlight 1-2 unique qualities of the song
- End with a question to engage the user
```

### 4.2 融合多候选信息

让 LLM 参考多个候选曲目生成响应：

```python
# mcrs/lm_modules/improved_llama.py

class ImprovedLLAMAModule:
    def response_generation(self, sys_prompt, chat_history, recommend_items, max_new_tokens=128):
        # recommend_items 现在是 list
        items_info = []
        for i, item in enumerate(recommend_items[:5], 1):
            items_info.append(
                f"{i}. {item['track_name']} by {item['artist_name']} "
                f"(Genre: {', '.join(item.get('tag_list', [])[:3])})"
            )

        # 在 system prompt 中注入候选信息
        enhanced_prompt = sys_prompt + f"\n\nTop Candidate Tracks:\n{chr(10).join(items_info)}"

        # 调用 LLM 生成
        return self._call_llm(enhanced_prompt, chat_history, max_new_tokens)
```

### 4.3 控制响应多样性

为提升 Lexical Diversity，在生成时引入随机性：

```python
# mcrs/lm_modules/diverse_response.py

def generate_diverse_response(lm, sys_prompt, chat_history, item, temperature=0.8):
    # 高 temperature 促进词汇多样性
    outputs = lm.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=128,
        temperature=temperature,
        top_p=0.95,
        do_sample=True  # 启用采样
    )
    return tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
```

---

## 五、完整优化管道

```python
# mcrs/improved_crs.py

class ImprovedCRS:
    """
    优化后的 CRS 管道
    Stage 1: Query Expansion + Hybrid Retrieval → Top-100
    Stage 2: Embedding-based Reranking → Top-20
    Stage 3: LLM Reranking (optional) → Top-10
    Stage 4: Diverse Response Generation
    """

    def __init__(self, config):
        # 初始化各模块
        self.query_expander = QueryExpander(config.lm)
        self.bm25_retriever = BM25_MODEL(...)
        self.dense_retriever = E5Retriever(...)
        self.embedding_reranker = EmbeddingReranker(...)
        self.llm_reranker = LLMReranker(...)
        self.lm_module = ImprovedLLAMAModule(...)
        self.item_db = MusicCatalogDB(...)
        self.user_db = UserProfileDB(...)

    def chat(self, user_query, user_id, chat_history):
        # 1. Query Expansion
        expanded_queries = self.query_expander.expand_query(user_query, chat_history)

        # 2. Hybrid Retrieval
        candidates = self.hybrid_retrieval(expanded_queries, topk=100)

        # 3. Embedding-based Reranking
        listening_history = self._get_listening_history(user_id)
        reranked = self.embedding_reranker.rerank(
            user_id, user_query, candidates, listening_history, topk=20
        )

        # 4. LLM-based Reranking (针对 Top-5)
        if len(reranked) >= 5:
            final_reranked = self.llm_reranker.rerank(user_query, reranked[:5], topk=5)
            reranked = final_reranked + reranked[5:]

        # 5. Diverse Response Generation
        top_items = [self.item_db.id_to_metadata(tid) for tid in reranked[:5]]
        response = self.lm_module.response_generation(
            self._get_system_prompt(user_id),
            chat_history,
            top_items,
            temperature=0.8
        )

        return {
            "retrieval_items": reranked,
            "recommend_item": top_items[0],
            "response": response
        }
```

---

## 六、实施路线图

### Phase 1: 快速迭代 (Week 1-2)
1. ✅ 扩展 Item Representation（添加 tag_list, popularity）
2. ✅ 切换到 E5/BGE Embedding 模型
3. ✅ 实现 Hybrid Retrieval (BM25 + Dense)

**目标**：nDCG@10 从 0.063 → **0.09+**

### Phase 2: 精细优化 (Week 3-4)
1. 实现 Embedding-based Reranker
2. 添加 Query Expansion 模块
3. Context-aware Retrieval 优化

**目标**：nDCG@10 从 0.09 → **0.12+**

### Phase 3: 高级特性 (Week 5-6)
1. LLM-based Reranker
2. Audio Feature Fusion (CLAP)
3. Diverse Response Generation

**目标**：nDCG@10 从 0.12 → **0.15+**, 同时保持 Diversity

### Phase 4: 调优 & 提交 (Week 7-8)
1. 超参数调优（融合权重、topk 选择）
2. Blind Set 推理优化
3. 性能优化（批处理、缓存）

**目标**：最终提交达到 **Top-10** 排名

---

## 七、预期收益汇总

| 优化项 | 预期 nDCG@10 提升 | 实现难度 |
|--------|-------------------|----------|
| 扩展文本特征 | +5%~10% | ⭐ 简单 |
| E5 Embedding | +15%~20% | ⭐⭐ 中等 |
| Hybrid Retrieval | +5%~8% | ⭐⭐ 中等 |
| Query Expansion | +8%~12% | ⭐⭐⭐ 较难 |
| Embedding Reranker | +5%~10% | ⭐⭐ 中等 |
| LLM Reranker | +5%~10% | ⭐⭐⭐ 较难 |
| Audio Features | +10%~15% | ⭐⭐⭐ 较难 |

**综合预期**：通过以上优化，nDCG@10 可从 **0.063 提升至 0.15+**，提升幅度约 **140%**。

---

## 八、资源链接

- 数据集: https://huggingface.co/collections/talkpl-ai/talkplay-data-challenge
- Track Embeddings: https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Track-Embeddings
- User Embeddings: https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-User-Embeddings
- E5 Model: https://huggingface.co/intfloat/e5-mistral-7b
- CLAP: https://huggingface.co/laion/clap-htsat-fused
