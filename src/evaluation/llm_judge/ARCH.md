# LLM Judge 评测系统

## 模块职责

| 模块 | 角色 | 说明 |
|------|------|------|
| `run.py` | 编排器 | 总入口，多进程编排 rollout + judge |
| `rollout.py` | 推理 | 独立运行，支持 HF / API 模型，写 JSONL |
| `judge.py` | 评分 | 两种使用方式：库（run.py 内部调用）+ 脚本（手动打分）|

### judge 的双入口

```
作为库:
    run.py ──import judge──→ judge.score(record) → score
                                   ↑
                            run.py 管理 buffer、轮询、控制流程

作为脚本:
    python judge.py --input X --output Y
    → 自己管理 buffer、轮询、控制流程
```

两种方式共享同一组评分函数，不冲突。

---

## 数据流

```
data/eval/multiturn.jsonl
         │
         ▼
    run.py
         │
         ├── 1. 创建 eval_output/{rollouts,reports}/
         │
         ├── 2. 遍历数据，同一遍做两件事:
         │       │
         │       ├── 提取 reference → rollouts/ref.jsonl
         │       │
         │       └── 逐 turn 调用 rollout.rollout() / rollout subprocess
         │               生成答案 → rollouts/ls.jsonl
         │
         └── 3. 读完所有行后，调用 judge.score() 逐行打分
                     → reports/ls_score.jsonl
```

---

## 目录结构

```
eval_output/
├── run.json                     ← 配置快照
├── rollouts/
│   ├── ref.jsonl                ← {id, turn, messages, reference}
│   └── ls.jsonl                 ← {id, turn, messages, predicted}
└── reports/
    ├── ls_score.jsonl           ← {id, turn, messages, predicted, score, reasoning}
    └── ls_vs_qwen.jsonl         ← {id, turn, messages_a, messages_b, winner, reasoning}
```

---

## 运行模式

### 串行（单进程）

rollout 和 judge 在同一个进程里顺序执行，流程简单：

```
1. rollout(model, data) → ls.jsonl
2. judge.score(ls.jsonl, ref.jsonl) → report.jsonl
```

### 并行（多进程）

rollout 作为子进程运行，编排器同时消费：

```
run.py
 ├── spawn rollout subprocess (逐 turn 写 ls.jsonl + flush)
 ├── 轮训 rollout 输出:
 │     新行 → judge.score(record) → report.jsonl
 ├── rollout 退出 → 消费剩余行 → 终止
```

---

## 多模型对比

需要对比两个模型时（如 LatentSeeker vs 原 Qwen）：

```bash
# 跑两遍 rollout，各自写自己的 jsonl
python -m src.evaluation.llm_judge.rollout \
    --model_path outputs/multitask_sft \
    --data_path data/eval/multiturn.jsonl \
    --output eval_output/rollouts/ls.jsonl

python -m src.evaluation.llm_judge.rollout \
    --api-base http://localhost:8000/v1 \
    --model Qwen3Coder \
    --data_path data/eval/multiturn.jsonl \
    --output eval_output/rollouts/qwen.jsonl

# run.py 读取两份 jsonl，按 (id, turn) 配对，调用 judge.compare()
python -m src.evaluation.llm_judge.run \
    --compare ls,qwen \
    --output_dir eval_output
```

---

## 评分方式

### judge.score()

```python
def score(api: JudgeAPI, messages: list) -> dict:
    """单评: 完整对话上下文 → {score: 1-5, reasoning: str}"""
```

### judge.compare()

```python
def compare(api: JudgeAPI, messages_a: list, messages_b: list) -> dict:
    """对比: 两个完整对话 → {winner: "a"|"b"|"tie", reasoning: str}"""
```

两者都不维护状态，不轮询，不读写文件。全部由编排器控制。
