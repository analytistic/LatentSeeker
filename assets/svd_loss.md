# SVD 辅助 Loss 推导

## 问题

一个 bin 里有 r 个 token，每个是 D 维向量：$X = [x_0, x_1, ..., x_{r-1}] \in \mathbb{R}^{r \times D}$

现在要压缩成 1 个 token $c \in \mathbb{R}^D$：
- **mean pooling**：$c = \frac{1}{r}\sum x_i$，每个 token 等权
- **learned gate**：$c = \sum w_i x_i$，学出来的权重 $w_i$
- **SVD**：$c = v_1$，第一右奇异向量，**最优 rank-1 近似**

## SVD 的数学意义

对 X 做 SVD：

$$X = U S V^T$$

- $U \in \mathbb{R}^{r \times r}$：左奇异向量
- $S \in \mathbb{R}^{r}$：奇异值（降序）
- $V \in \mathbb{R}^{r \times D}$：右奇异向量

**第一右奇异向量 $v_1$** 满足：

$$v_1 = \arg\max_{||v||=1} ||X v||^2$$

即：在所有单位向量中，$v_1$ 让 $X$ 投影过去的**方差最大**。

同时 $v_1$ 也是最优 rank-1 近似的解：

$$\min_{c} \sum_{i} ||x_i - \alpha_i c||^2$$

其中 $\alpha_i = \frac{x_i \cdot c}{||c||^2}$ 是最优投影系数。

## 推导：最小化重建误差 $\iff$ SVD

### Step 1: 写出重建误差

对于一个给定向量 c，每个 token $x_i$ 在 c 上的投影为 $\alpha_i c$，其中 $\alpha_i = \frac{x_i \cdot c}{||c||^2}$。

重建误差：
$$\mathcal{L}(c) = \sum_{i=0}^{r-1} ||x_i - \alpha_i c||^2$$

### Step 2: 展开

$$||x_i - \alpha_i c||^2 = ||x_i||^2 - 2\alpha_i(x_i\cdot c) + \alpha_i^2||c||^2$$

将 $\alpha_i = \frac{x_i \cdot c}{||c||^2}$ 代入：

$$ = ||x_i||^2 - 2\frac{(x_i\cdot c)^2}{||c||^2} + \frac{(x_i\cdot c)^2}{||c||^2}$$

$$ = ||x_i||^2 - \frac{(x_i\cdot c)^2}{||c||^2}$$

对所有 token 求和：

$$\mathcal{L}(c) = \sum_i ||x_i||^2 - \frac{1}{||c||^2} \sum_i (x_i\cdot c)^2$$

第一项 $\sum_i ||x_i||^2$ 与 c 无关，所以最小化 $\mathcal{L}(c)$ $\iff$ **最大化** $\frac{1}{||c||^2} \sum_i (x_i\cdot c)^2$。

### Step 3: 写成矩阵形式

$$\sum_i (x_i\cdot c)^2 = ||Xc||^2 = c^T X^T X c$$

所以最大化 $\frac{c^T X^T X c}{c^T c}$ —— 这就是 **Rayleigh quotient**。

### Step 4: Rayleigh quotient 的解

$$\max_{c} \frac{c^T X^T X c}{c^T c}$$

$X^T X$ 是实对称半正定矩阵，它的最大特征值对应的特征向量就是使 Rayleigh quotient 最大的向量。

而 $X = USV^T$，所以 $X^T X = V S^2 V^T$，特征向量就是 $V$ 的列（右奇异向量），最大特征值 $s_1^2$ 对应的就是 $v_1$（第一右奇异向量）。

**结论：最小化重建误差 $\mathcal{L}(c)$ 的最优解 $c^* = v_1$，即 SVD 的第一主成分。**

## 实现为辅助 loss

不需要真的做 SVD，直接用重建误差作为 loss：

```python
# c:         [M, D]   压缩后的 token（每个 bin 一个）
# x:         [N, D]   原始 token（每个 bin 里 r 个）
# pool_idx:  [N]      每个 x 对应的 bin 编号

# 按 pool_idx 展开 c
c_expanded = c[pool_idx]          # [N, D]

# 投影系数 αᵢ = (xᵢ·c) / ||c||²
dot = (x * c_expanded).sum(dim=-1, keepdim=True)    # [N, 1]
c_norm_sq = (c * c).sum(dim=-1, keepdim=True)        # [M, 1]
alpha = dot / c_norm_sq[pool_idx].clamp(min=1e-8)    # [N, 1]

# 重建
recon = alpha * c_expanded                            # [N, D]

# SVD loss
svd_loss = F.mse_loss(recon, x)                       # 标量

# 总 loss
total_loss = task_loss + lambda * svd_loss
```

## λ 的作用

| λ | 效果 | 适用场景 |
|---|------|---------|
| λ=0 | gate 学任何权重，不管信息损失 | 极端压缩 |
| λ 小 | gate 主要听下游信号，稍微约束信息保留 | 大部分情况 |
| λ 大 | 近似 SVD，gate 行为接近第一主成分 | 需要高保真重建 |
| λ→∞ | 完全等价 SVD，gate 权重由数据方差决定 | 纯特征降维 |

对于压缩到 ratio 2+ 的场景，λ 设一个中间值（比如 0.1），让 gate 既保留下游有用的特征，又不丢失 bin 内的信息。
