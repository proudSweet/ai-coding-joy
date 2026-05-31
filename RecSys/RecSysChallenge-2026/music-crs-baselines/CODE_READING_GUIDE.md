# Music-CRS Baselines 代码阅读指南

## 目录结构

```
music-crs-baselines/
├── config/                          # 配置文件目录
│   ├── llama1b_bm25_devset.yaml     # BM25 配置 (开发集)
│   ├── llama1b_bm25_blindset_A.yaml # BM25 配置 (盲测集A)
│   ├── llama1b_bert_devset.yaml     # BERT 配置 (开发集)
│   └── llama1b_bert_blindset_A.yaml  # BERT 配置 (盲测集A)
├── lowerbound/                      # 下界基线 (非学习式)
│   ├── popularity.py                 # 基于流行度的推荐
│   └── random_sample.py              # 随机推荐
├── mcrs/                             # 核心系统代码
│   ├── __init__.py                   # 入口点, load_crs_baseline()
│   ├── crs_baseline.py               # CRS 主类 (两阶段管道)
│   ├── db_item/                     # 歌曲数据库
│   │   ├── __init__.py
│   │   └── music_catalog.py          # MusicCatalogDB 类
│   ├── db_user/                     # 用户数据库
│   │   ├── __init__.py
│   │   └── user_profile.py           # UserProfileDB 类
│   ├── lm_modules/                  # LLM 模块
│   │   ├── __init__.py
│   │   └── llama.py                  # LLAMA_MODEL 类
│   ├── retrieval_modules/           # 检索模块
│   │   ├── __init__.py
│   │   ├── bm25.py                   # BM25_MODEL 类
│   │   └── bert.py                   # BERT_MODEL 类
│   └── system_prompts/              # 系统提示词
│       ├── roleplay.txt              # 角色设定
│       ├── personalization.txt        # 个性化提示
│       └── response_generation.txt   # 响应生成提示
├── tips/                            # 优化提示
├── run_inference_devset.py          # 开发集推理脚本
├── run_inference_blindset.py        # 盲测集推理脚本
└── pyproject.toml                   # 项目依赖
```

---

## 依赖关系图

```
run_inference_devset.py / run_inference_blindset.py
         │
         ▼
load_crs_baseline()  [mcrs/__init__.py]
         │
         ▼
CRS_BASELINE  [crs_baseline.py]
    │
    ├──┬──────────────────────────────┐
    │  │                              │
    ▼  ▼                              ▼
load_lm_module()              load_retrieval_module()
    │                              │
    ▼                              ▼
LLAMA_MODEL                    BM25_MODEL / BERT_MODEL
[lm_modules/llama.py]          [retrieval_modules/*.py]
                                    │
                                    ▼
                               MusicCatalogDB
                               [db_item/music_catalog.py]
                                    │
                               UserProfileDB
                               [db_user/user_profile.py]
```

---

## 核心类详解

### 1. 入口点: `mcrs/__init__.py`

```python
from .crs_baseline import CRS_BASELINE

def load_crs_baseline(
    lm_type="meta-llama/Llama-3.2-1B-Instruct",
    retrieval_type="bm25",
    item_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
    user_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
    track_split_types: list[str] = ["all_tracks"],
    user_split_types: list[str] = ["all_users"],
    corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
    cache_dir="./cache",
    device="cuda",
    attn_implementation="eager",
    dtype=torch.bfloat16
):
    return CRS_BASELINE(...)
```

**作用**: 工厂函数,根据配置创建 CRS_BASELINE 实例,统一管理子模块初始化。

---

### 2. 主类: `mcrs/crs_baseline.py` - CRS_BASELINE

**核心两阶段管道**:

```python
class CRS_BASELINE:
    def chat(self, user_query: str, user_id: Optional[str] = None) -> dict:
        # Stage 0: 保存用户 query 到对话历史
        self.session_memory.append({"role": "user", "content": user_query})

        # Stage 1: 检索
        system_prompt = self._get_system_prompt(user_id)
        retrieval_input = self._build_retrieval_input()  # 拼接对话历史
        retrieval_items = self.retrieval.text_to_item_retrieval(retrieval_input, topk=20)
        recommend_item = self.item_db.id_to_metadata(retrieval_items[0])

        # Stage 2: LLM 生成响应
        response = self.lm.response_generation(system_prompt, self.session_memory, recommend_item)

        return {
            "user_id": user_id,
            "user_query": user_query,
            "retrieval_items": retrieval_items,    # Top-20 候选曲目
            "recommend_item": recommend_item,     # Top-1 推荐 (字符串格式)
            "response": response                   # LLM 生成的文本响应
        }
```

**关键方法**:

| 方法 | 作用 |
|------|------|
| `_reset_session_memory()` | 清空当前会话的对话历史 |
| `_upload_session_memory(chat_history)` | 加载外部对话历史到 session |
| `_get_system_prompt(user_id)` | 拼接系统提示 (Role + Response + Personalization) |
| `chat()` | 单轮对话处理 |
| `batch_chat()` | 批量对话处理 |

**数据流**:
```
用户输入 → session_memory → 拼接为检索 query → BM25/BERT 检索 Top-20
                                                        ↓
                                     Top-1 元数据 → LLM 生成响应 ← 系统提示
```

---

### 3. 检索模块: `mcrs/retrieval_modules/`

#### 3.1 BM25 检索: `bm25.py` - BM25_MODEL

```python
class BM25_MODEL:
    def __init__(self, dataset_name, split_types, corpus_types, cache_dir):
        # 1. 加载歌曲元数据
        self.metadata_dict = self._load_corpus()

        # 2. 构建或加载 BM25 索引
        if os.path.exists(f"{cache_dir}/bm25/{corpus_name}"):
            self.bm25_model, self.track_ids = self._load_bm25()
        else:
            self.build_index()  # 创建索引并缓存

    def _stringify_metadata(self, metadata: dict) -> str:
        # 将歌曲元数据拼接为文本
        # corpus_types: ["track_name", "artist_name", "album_name"]
        metadata_str = ""
        for corpus_type in self.corpus_types:
            entity = metadata[corpus_type]
            if isinstance(entity, list):
                entity = ", ".join(entity)
            metadata_str += f"{corpus_type}: {entity}\n"
        return metadata_str

    def build_index(self):
        # 1. 遍历所有歌曲,拼接元数据为字符串
        corpus = [self._stringify_metadata(self.metadata_dict[tid]) for tid in track_ids]

        # 2. 分词
        corpus_tokens = bm25s.tokenize(corpus)

        # 3. 构建 BM25 索引
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        # 4. 缓存到磁盘
        retriever.save(f"{cache_dir}/bm25/{corpus_name}", corpus=corpus)

    def text_to_item_retrieval(self, query: str, topk: int) -> list[str]:
        # 1. 对查询分词
        query_tokens = bm25s.tokenize(query)

        # 2. 检索 Top-k
        results, scores = self.bm25_model.retrieve(query_tokens, k=topk)

        # 3. 返回 track_ids (按相关性排序)
        return [self.track_ids[i] for i in results[0]]
```

**缓存机制**:
- 索引缓存在 `cache/bm25/{corpus_name}/` 目录
- 首次运行后自动复用,避免重复构建

#### 3.2 BERT 密集检索: `bert.py` - BERT_MODEL

```python
class BERT_MODEL:
    def __init__(self, dataset_name, split_types, corpus_types, cache_dir,
                 model_name="bert-base-uncased", device=None, batch_size=32, max_length=128):

        # 1. 加载 BERT 模型和 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # 2. 加载或构建 embedding 索引
        if os.path.exists("embeddings.pt"):
            self.embeddings, self.track_ids = self._load_index()
        else:
            self.build_index()

    def build_index(self):
        # 1. 遍历所有歌曲,生成文本表示
        corpus = [self._stringify_metadata(metadata) for metadata in self.metadata_dict.values()]

        # 2. 批量编码为向量
        all_embeddings = []
        for i in range(0, len(corpus), self.batch_size):
            batch = corpus[i:i+self.batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            outputs = self.model(**inputs)
            # Mean pooling: 平均所有 token 的 embedding
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings)

        # 3. 合并并缓存
        self.embeddings = torch.cat(all_embeddings, dim=0)
        torch.save(self.embeddings, "embeddings.pt")

    def text_to_item_retrieval(self, query: str, topk: int) -> list[str]:
        # 1. 编码查询向量
        inputs = self.tokenizer(query, return_tensors="pt")
        outputs = self.model(**inputs)
        query_emb = outputs.last_hidden_state.mean(dim=1)

        # 2. 计算余弦相似度
        scores = F.cosine_similarity(query_emb, self.embeddings)

        # 3. 取 Top-k
        top_indices = scores.topk(topk).indices
        return [self.track_ids[i] for i in top_indices]
```

**两种检索方式对比**:

| 特性 | BM25 | BERT |
|------|------|------|
| 类型 | 稀疏检索 (词匹配) | 密集检索 (向量语义) |
| 语义理解 | ❌ 精确词匹配 | ✅ 语义相似 |
| 计算速度 | 快 | 较慢 (需 GPU) |
| 依赖 | bm25s 库 | transformers + torch |
| 缓存大小 | 小 | 大 (50k × 768维) |

---

### 4. LLM 模块: `mcrs/lm_modules/llama.py` - LLAMA_MODEL

```python
class LLAMA_MODEL:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct",
                 device="cuda", attn_implementation="eager", dtype=torch.bfloat16):
        # 加载 tokenizer 和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation=attn_implementation,
            dtype=dtype
        )
        self.device = device
        self.lm.eval()

    def _format_chat_history(self, sys_prompt, chat_history: list, recommend_item: str):
        # 构建对话格式
        chat_data = [
            {"role": "system", "content": sys_prompt},  # 系统提示
            *chat_history,                               # 对话历史
            {"role": "assistant", "content": recommend_item}  # 注入推荐歌曲
        ]
        # 使用 chat template 格式化
        return self.tokenizer.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True)

    def response_generation(self, sys_prompt, chat_history, recommend_item, max_new_tokens=512):
        # 1. 格式化对话
        formatted_chat = self._format_chat_history(sys_prompt, chat_history, recommend_item)

        # 2. Tokenize
        inputs = self.tokenizer(formatted_chat, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # 3. 生成
        with torch.no_grad():
            outputs = self.lm.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens
            )

        # 4. 解码 (跳过输入部分,只取新生成的内容)
        generated_text = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0]

        return generated_text
```

**Chat Template 格式示例**:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert music recommendation assistant...

<|start_header_id|>user<|end_header_id|>
I want some jazz music

<|start_header_id|>assistant<|end_header_id|>
track_id: xxx, track_name: Take Five, artist_name: Dave Brubeck...

<|start_header_id|>assistant<|end_header_id|>
Here are some songs you might enjoy...
```

---

### 5. 数据库模块

#### 5.1 歌曲数据库: `mcrs/db_item/music_catalog.py` - MusicCatalogDB

```python
class MusicCatalogDB:
    def __init__(self, dataset_name, split_types, corpus_types):
        # 加载 HuggingFace 数据集
        metadata_dataset = load_dataset(dataset_name)
        metadata_concat_dataset = concatenate_datasets([
            metadata_dataset[split_type] for split_type in split_types
        ])

        # 构建 track_id → metadata 的映射
        self.metadata_dict = {item["track_id"]: item for item in metadata_concat_dataset}
        self.corpus_types = corpus_types

    def id_to_metadata(self, track_id: str) -> str:
        # 将歌曲元数据转换为字符串,供 LLM 使用
        metadata = self.metadata_dict[track_id]
        entity_str = f"track_id: {track_id}"

        for corpus_type in self.corpus_types:
            # 处理列表类型 (如 tag_list)
            corpus_type_value = ", ".join(metadata[corpus_type]).lower()
            entity_str += f", {corpus_type}: {corpus_type_value}"

        return entity_str
        # 输出示例: "track_id: xxx, track_name: take five, artist_name: dave brubeck, album_name: time out"
```

#### 5.2 用户数据库: `mcrs/db_user/user_profile.py` - UserProfileDB

```python
class UserProfileDB:
    def __init__(self, dataset_name, split_types):
        metadata_dataset = load_dataset(dataset_name)
        metadata_concat_dataset = concatenate_datasets([
            metadata_dataset[split_type] for split_type in split_types
        ])

        # 默认列: user_id, age_group, gender, country_name
        self.default_columns = ['user_id', 'age_group', 'gender', 'country_name']
        self.user_profiles = {item["user_id"]: item for item in metadata_concat_dataset}

    def id_to_profile_str(self, user_id: str) -> str:
        # 格式化为字符串,注入系统提示
        user_profile = self.user_profiles[user_id]
        profile_str = [f"{key}: {user_profile[key]}" for key in self.default_columns]
        return "\n".join(profile_str)
        # 输出示例:
        # user_id: 12345
        # age_group: 25-34
        # gender: male
        # country_name: United States
```

---

### 6. 系统提示词: `mcrs/system_prompts/`

#### 6.1 roleplay.txt (角色设定)
```
You are an expert music recommendation assistant. Your task is to understand user preferences and provide personalized music recommendations.
```

#### 6.2 personalization.txt (个性化)
```
Consider the following user profile when generating your response. Take into account the user's age group, country, and gender to personalize your recommendations and communication style appropriately.
```
(后面会拼接用户画像字符串)

#### 6.3 response_generation.txt (响应生成)
```
Based on the user query and the recommended track from tool calling results, provide a brief response that:
1. MUST base your response on the previously recommended track from the tool calling results
2. If the recommended track doesn't match the user's query, apologize and acknowledge the mismatch...
3. If it's a good match, acknowledge that you've found music that matches their request with enthusiasm and confidence
4. Share key details including title, artist, and relevant musical information (genre, mood, style, or notable characteristics)
5. Briefly explain why this track is a good match for their specific request or preferences (or apologize if it's not a good match)
6. Invite further interaction by asking if they'd like to explore similar tracks, need recommendations for different moods, or have any other music preferences to discuss
```

---

### 7. 推理脚本: `run_inference_devset.py`

```python
def chat_history_parser(conversations, music_crs, target_turn_number):
    """
    解析对话历史,用于多轮推荐

    对话数据结构:
    - turn_number: 1-8
    - role: "user" | "assistant" | "music"
    - content: 文本内容 或 track_id (当 role="music" 时)

    处理逻辑:
    1. 取出 target_turn_number 之前的所有对话作为历史
    2. role="music" 的内容转换为歌曲元数据字符串
    3. 返回 chat_history (列表) 和当前 turn 的 user_query
    """
    ...

def main(args):
    # 1. 加载配置和模型
    config = OmegaConf.load(f"config/{args.tid}.yaml")
    music_crs = load_crs_baseline(**config)

    # 2. 加载测试数据集
    db = load_dataset(config.test_dataset_name, split="test")

    # 3. 构造 batch 数据 (所有 session × 所有 turn)
    batch_data = []
    for item in db:
        for target_turn_number in range(1, 9):
            chat_history, user_query = chat_history_parser(item['conversations'], music_crs, target_turn_number)
            batch_data.append({
                'user_query': user_query,
                'user_id': item['user_id'],
                'session_memory': chat_history
            })

    # 4. 批量推理
    for i in tqdm(range(0, len(batch_data), args.batch_size)):
        batch = batch_data[i:i+args.batch_size]
        results = music_crs.batch_chat(batch)
        ...

    # 5. 保存结果
    # 格式: session_id, user_id, turn_number, predicted_track_ids, predicted_response
```

---

## 配置说明: `config/*.yaml`

```yaml
lm_type: "meta-llama/Llama-3.2-1B-Instruct"    # LLM 模型
retrieval_type: "bm25"                          # 检索方式: bm25 或 bert
test_dataset_name: "talkpl-ai/TalkPlayData-Challenge-Dataset"
item_db_name: "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
user_db_name: "talkpl-ai/TalkPlayData-Challenge-User-Metadata"
track_split_types:
  - "all_tracks"                                # ⚠️ 必须使用 all_tracks
user_split_types:
  - "all_users"
corpus_types:                                   # 用于检索的歌曲字段
  - "track_name"
  - "artist_name"
  - "album_name"
  - "release_date"
cache_dir: "./cache"
device: "cuda"
attn_implementation: "flash_attention_2"        # 加速 LLM 推理
```

---

## 下界基线: `lowerbound/`

### random_sample.py
随机从全量曲库抽取 20 首歌曲作为推荐,用于评估随机基线。

### popularity.py
统计训练集中出现频率最高的 20 首歌曲,作为流行度基线。

---

## 关键数据流总结

```
输入数据 (HuggingFace Dataset)
    │
    ├─→ MusicCatalogDB      ──→ track_id → metadata 字符串
    ├─→ UserProfileDB       ──→ user_id → profile 字符串
    │
    └─→ 对话历史 ──→ BM25/BERT ──→ Top-20 track_ids
                               │
                               ├─→ Top-1 ──→ LLM 生成响应
                               │
                               └─→ Top-20 ──→ predicted_track_ids (提交格式)

输出 JSON:
{
  "session_id": "69137__2020-02-08",
  "user_id": "69137",
  "turn_number": 1,
  "predicted_track_ids": ["...", "...", ...],  // Top-20
  "predicted_response": "Here are some songs..."
}
```

---

## 扩展指南

### 添加新的检索模型

1. 在 `mcrs/retrieval_modules/` 创建新文件, 如 `e5.py`
2. 实现 `text_to_item_retrieval(self, query: str, topk: int) -> list[str]` 方法
3. 在 `mcrs/retrieval_modules/__init__.py` 的 `load_retrieval_module()` 中添加分支
4. 在 config 中设置 `retrieval_type: "e5"`

### 添加新的 LLM

1. 在 `mcrs/lm_modules/` 创建新文件, 如 `qwen.py`
2. 实现 `response_generation()` 和 `batch_response_generation()` 方法
3. 在 `mcrs/lm_modules/__init__.py` 的 `load_lm_module()` 中添加分支
4. 在 config 中设置 `lm_type: "Qwen/Qwen2.5-7B"`

### 添加新的评估指标

1. 在 `music-crs-evaluator/metrics/` 中添加新文件
2. 参考 `metrics_recsys.py` 实现 `get_xxx()` 函数
3. 在 `evaluate_devset.py` 中调用并保存结果
