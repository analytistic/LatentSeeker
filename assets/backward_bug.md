# Backward 错误根因：in-place `index_add_` 导致的双路梯度冲突

## 错误现象

```
RuntimeError: The size of tensor a (2560) must match tensor b (279040)
  at non-singleton dimension 1
```

发生在 `loss.backward()`，具体在 `LongertextMerger.forward()` 的 `MulBackward0`：

```python
c_norm_sq = (pooled * pooled).sum(dim=-1, keepdim=True)  # line 249
```

## 计算图结构

每个 `LongertextMerger` 的 `pooled` 张量同时参与两条梯度路径：

```
                                   ┌── SVD loss (自有的重建损失)
pooled (= summed) ────────────────┤
                                   └── LM loss (通过 deepstack 注入 → decoder → LM head)
```

### 路径1: SVD loss

```python
pooled  →  c_norm_sq = (pooled * pooled).sum(dim=-1)  →  alpha  →  recon  →  svd_loss
pooled  →  c_expanded = pooled[pool_indices]          →  alpha  →  recon  →  svd_loss
```

### 路径2: LM loss

```python
pooled  →  deepstack_features → deepstack_longtext_embeds[layer]
        →  _longtext_deepstack_process(embed=pooled) → hidden_states
        →  decoder layer → ... → LM head → lm_loss
```

## 根因

问题出在 `summed` 的构造方式：

```python
# 旧代码（有问题的）
summed = torch.zeros(total_output, D)
summed.index_add_(0, pool_indices, weighted)  # ← in-place 修改
pooled = summed  # 别名，和被 in-place 修改的是同一个张量
```

`index_add_` 是**就地操作**（in-place），它会：

1. 在 forward 时直接修改 `summed` 的存储，递增 version counter
2. Autograd 记录下这个 in-place 操作节点
3. `pooled` 只是 `summed` 的 Python 别名——指向同一个张量对象

当 `pooled` 被两条梯度路径共享时：

- **SVD loss backward**：经过 `c_norm_sq`, `c_expanded` 等路径到达 `pooled`
- **LM loss backward**：经过 deepstack injection → `_longtext_deepstack_process` → `embed(=pooled)` 到达同一张量

Autograd 在反向传播时，两条路径都尝试对这个**被就地修改过的张量**做梯度累积。由于 in-place 操作的特殊版本跟踪机制，`MulBackward0` 保存的张量元数据与实际梯度形状不匹配，导致 `MulBackward0` 看到 `pooled` 被展平为 1D `[279040]`（即 `109 × 2560` 的展平），而 grad_output 仍保持 2D `[109, 2560]`。

**为什么是 index_add_ 而不是普通操作？**

`zeros → index_add_` 这个模式中，`index_add_` 就地改写了 zeros 的输出张量。PyTorch 对 in-place 操作的 backward 实现需要额外的版本检查。当双路梯度汇聚时，版本检查的中间状态可能导致 torch 对 saved tensor 做出错误的形状解读。

## 修复

用 out-of-place 的 `torch.index_add` 替代：

```python
# 新代码
summed = torch.zeros(total_output, D)
summed = torch.index_add(summed, 0, pool_indices, weighted)  # ← 返回新张量，不修改原值
pooled = summed
```

`torch.index_add` 会创建**全新的张量**作为输出，`summed` 被重新赋值为这个新张量。这个新张量不存在 in-place 版本冲突问题，两条梯度路径可以安全地汇聚。

同理修复 `weight_sum`：

```python
weight_sum = torch.zeros(total_output)
weight_sum = torch.index_add(weight_sum, 0, pool_indices, gate_scores_exp)
```

## 验证方法

在 debug 配置下重新运行训练，确认 backward 不再报错，且 svd_loss 能在 TensorBoard 上正常记录：
