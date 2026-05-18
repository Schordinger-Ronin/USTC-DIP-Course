# Implementation of Digital Image Process(Yudong Guo) Assignment 4 - 3DGS

**Name:** Zijian Zhang(张子健)

**Student ID:** SA25001083

该仓库包含中国科学技术大学数学科学学院数字图像处理（MATH6420P.01）课程的作业4（3DGS）的实现过程以及实验结果。主要实现了使用 PyTorch 实现完整的 3DGS 全过程，并使用了多个数据集运行 官方 3DGS 代码，从渲染质量、训练速度、显存占用三方面对两个数据集运行的结果进行对比。

由于算力及操作系统缘故，本次实验在本地修改完代码之后上传到 Google Drive上，并使用Google Colab进行数据集训练：

<img src="picture_results/2.png" alt="Google Colab展示" width="800">

## Task 1 Structure-from-Motion with COLMAP
第一步直接运行`mvs_with_colmap.py`文件即可（某些部分可能需要一定的微调），需要注意的是在`Feature matching`这一步的时候可能需要较多时间，因为是 100 张图片两两匹配，需要进行 4950 次比对，即需要 4950 对高维矩阵的乘法。如果不能使用GPU进行加速的话会非常慢。

第二步直接运行`debug_mvs_by_projecting_pts.py`文件即可，这个文件的主要作用就是对第一步生成的基础点云进行验证，是否符合要求，运行完成后会生成如下的`projections`文件夹（共包含100张图片）：

<img src="picture_results/3.png" alt="projections文件夹" width="400">

其中部分图片如下（对chair和lego模型各展示一个点云图与原图）：

<img src="picture_results/4.png" alt="chair" width="400">
<img src="picture_results/5.png" alt="lego" width="400">

如果`projections`文件夹中出现了100张图片，并且点云分布与原图差异不大的话基本可以说明 Task 1 的运行结果是正确的。

## Task 2 Simplified 3D Gaussian Splatting

观察 Task 1 的输出可以发现，COLMAP 恢复的 3D 点对于稠密渲染来说过于稀疏。下面将每个点扩展为一个 3D 高斯，使其覆盖周围空间。
### 1. 3D Gaussian Initialization
在`gaussian_model.py`中补全由四元数和缩放参数构造 3D 协方差矩阵。
在 3DGS 中，3D 高斯分布的概率密度函数由均值 $\mu$（位置）和协方差矩阵 $\Sigma$（形状）决定。为了确保在反向传播和梯度下降优化过程中，协方差矩阵 $\Sigma$ 始终保持为物理上有效的**半正定矩阵（Positive Semi-Definite）**，算法不直接优化 $\Sigma$ 的元素，而是将其分解为独立的旋转矩阵 $R$ 和缩放矩阵 $S$ 的组合。

根据椭球体的几何变换原理，一个标准球体经过缩放和旋转后，其协方差矩阵可表示为：
$$\Sigma = R S S^T R^T$$

为了在代码中高效计算，我们可以定义一个复合变换矩阵 $M$，代表先缩放后旋转的操作：
$$M = R S$$

此时，协方差矩阵可以简化计算为：
$$\Sigma = M M^T$$
### 2. Project 3D Gaussians to 2D
在 3DGS 的前向渲染过程中，主要包含三个核心阶段：将 3D 高斯椭球投影到 2D 图像平面、计算像素级别的 2D 高斯概率密度值、以及基于按深度排序的高斯体进行 Alpha-blending 体素渲染。

3D 高斯在空间中的形状由 3D 协方差矩阵 $\Sigma$ 描述。当我们用针孔相机模型将其投影到 2D 图像平面时，由于透视投影是非线性变换，我们需要在相机坐标系下对其进行局部线性化（泰勒展开一阶近似）。根据论文公式，投影后的 2D 协方差矩阵 $\Sigma^{\prime}$ 计算如下：
$$\Sigma^{\prime} = J W \Sigma W^T J^T$$
其中，$W$ 是世界坐标系到相机坐标系的旋转变换矩阵（即代码中的 $R$），$J$ 是透视投影变换的雅可比矩阵。

在 `compute_projection` 函数中，首先将 3D 点变换到相机坐标系。针对雅可比矩阵 $J$ 的构造，由于透视投影公式为 $u = f_x \frac{x_c}{z_c}, v = f_y \frac{y_c}{z_c}$，对其求偏导可得：

$$
J = \begin{bmatrix} 
\frac{f_x}{z_c} & 0 & -\frac{f_x x_c}{z_c^2} \\ 
0 & \frac{f_y}{z_c} & -\frac{f_y y_c}{z_c^2} 
\end{bmatrix}
$$

代码中利用 `depths` ($z_c$) 并配合 `clamp` 防止除零错误，解析计算了 $J$ 的各个元素。随后利用 `torch.bmm` 实现了批量的高效矩阵连乘，完成了协方差从 3D 到 2D 的降维投影。

### 3. Compute 2D Gaussian Values
在得到 2D 图像平面上的高斯中心 $\mu_i$ 和协方差 $\Sigma_i$ 后，需要计算每个高斯体在特定像素 $\mathbf{x}$ 处的概率密度函数值。公式如下：
$$f(\mathbf{x} ; \mu_i, \Sigma_i) = \frac{1}{2 \pi \sqrt{|\Sigma_i|}} \exp \left( -\frac{1}{2}(\mathbf{x}-\mu_i)^T \Sigma_i^{-1}(\mathbf{x}-\mu_i) \right)$$

在 `compute_gaussian_values` 函数中，直接进行 $(N, H, W, 2)$ 维度的高维矩阵乘法会导致极大的显存开销。因此采用代数展开的优化策略：
1. **解析求逆与提取：** 使用 `torch.inverse` 求出 $\Sigma^{-1}$ 后，将其 4 个元素单独提取为 `inv_c00, inv_c01, inv_c10, inv_c11`。
2. **显式二次型计算：** 将矩阵乘法 $dx^T \Sigma^{-1} dx$ 显式展开为标量乘法求和：`dx0 * inv_c00 * dx0 + dx0 * inv_c01 * dx1 + ...`。
同时引入微小扰动 `eps * torch.eye(2)` 保证了协方差矩阵的非奇异性。

### 4. Volume Rendering via α-blending
3DGS 借鉴了 NeRF 的体素渲染思想。对于像素 $\mathbf{x}$ 所在的光线，将穿过的 $N$ 个高斯体按深度由近到远排序。每个高斯体在像素处的实际不透明度为 $\alpha_{(x, i)} = o_i \cdot f(\mathbf{x})$。
透射率 $T_{(x, i)}$ 表示光线穿过前 $i-1$ 个高斯体后剩余的能量，定义为累积乘积：
$$T_{(x, i)} = \prod_{j<i} (1 - \alpha_{(x, j)})$$
最终像素颜色是各高斯颜色按权重 $W_i = T_i \alpha_i$ 的累加。

在 `forward` 函数的最后部分，计算透射率 $T_i$ 是一大难点。由于公式要求的是严格的“前 $i-1$ 项连乘”，而 PyTorch 的 `torch.cumprod` 是包含当前项的。采用了如下的张量进行操作：
```python
transmittance = 1.0 - alphas
T = torch.cat([torch.ones((1, self.H, self.W), ...), transmittance[:-1]], dim=0)
T = torch.cumprod(T, dim=0)
```

完成上述代码补全后，直接使用如下代码开始训练即可（我使用的是Colab中的高RAM和A100 GPU进行训练）：
```python
python train.py --colmap_dir data/chair --checkpoint_dir data/chair/checkpoints
```

<img src="picture_results/6.png" alt="训练过程" width="800">

训练完成后如下（训练200个epoch一共花费28分钟，但是实际看下来其实训练50个eopch就可以了，从50到200个epoch的loss值下降不多）：

<img src="picture_results/7.png" alt="训练结果" width="400">
<img src="picture_results/8.png" alt="训练时间" width="400">

训练完成后，可用 render_3dgs_mv.py 沿一个绕场景中心的水平圆轨迹渲染一段连续视角视频，便于直观检查重建质量：
```python
%cd "/content/drive/MyDrive/Assignment 4 - 3DGS"

!python3.10 render_3dgs_mv.py \
    --colmap_dir data/chair \
    --checkpoint data/chair/checkpoints/checkpoint_000180.pt \
    --num_frames 240 \
    --fps 30
```

对chair和lego数据集重复以上步骤，此外我自己还选择了一个playroom数据集进行训练但是会出现报错（首个 Epoch 损失值正常，但随后的 Epoch 中 Loss 突变为 `NaN`）：

<img src="picture_results/9.png" alt="报错" width="800">

具体原因大致有以下两点：
#### 1. 正规化常数导致的 Alpha 越界与透射率异常

在标准的多元高斯分布中，概率密度函数包含一个正规化常数以保证其在全空间的积分为 1：
$$f(\mathbf{x}) = \frac{1}{2 \pi \sqrt{|\Sigma|}} \exp \left( -\frac{1}{2}(\mathbf{x}-\mu)^T \Sigma^{-1}(\mathbf{x}-\mu) \right)$$

然而，在 3DGS 的优化过程中，若某个高斯椭球在不断迭代中体积急剧缩小，其协方差矩阵的行列式 $|\Sigma|$ 将趋近于 0。此时，正规化常数 $\frac{1}{2 \pi \sqrt{|\Sigma|}}$ 会呈现非线性地急剧放大，甚至数值溢出。

在 Alpha-blending（体积渲染）的物理模型中，高斯体对当前像素的实际不透明度（遮挡概率）定义为：
$$\alpha_{(x, i)} = o_i \cdot f(\mathbf{x})$$
其中 $o_i \in [0, 1]$。当 $f(\mathbf{x})$ 因上述原因暴增时，将导致计算出的 $\alpha_{(x, i)} > 1$，违背了光学渲染中“遮挡率不得大于 100%”的物理约束。

更为致命的是，在计算透射率（Transmittance）时：
$$T_{(x, i)} = \prod_{j<i} (1 - \alpha_{(x, j)})$$
一旦 $\alpha > 1$，连乘项 $(1 - \alpha)$ 将变为负数。在深度学习的反向传播机制中，对包含负数的连乘算子求导会引发极度的不稳定性，导致梯度瞬间爆炸，进而污染网络中所有的可学习参数，使其在下一轮迭代中全部退化为 `NaN`。

*(注：为规避此缺陷，在 3DGS 的官方 CUDA 实现中，出于数值稳定性的考量，通常会直接舍弃该正规化常数项，而将权重控制完全交由可学习的透明度 $o_i$ 和指数项来决定。)*

#### 2. 原生矩阵求逆算子 (`torch.inverse`) 的数值脆弱性

在计算 2D 高斯取值时，需要求解投影后的 2D 协方差矩阵的逆矩阵 $\Sigma^{\prime -1}$。在早期的代码实现中，常直接调用 PyTorch 的原生求逆算子 `torch.inverse()`。

然而，在面对包含数万乃至数百万个微小 $2 \times 2$ 矩阵的批量运算时，通用求逆算法表现出极大的数值脆弱性。一旦经过投影后的 2D 协方差矩阵退化为奇异矩阵（Singular Matrix）或近似奇异矩阵（即行列式接近机器零 epsilon），`torch.inverse()` 会直接输出 `Inf`（无穷大）或 `NaN`。

在自动微分计算图中，即便这些产生了 `NaN` 的点在后续操作中被掩码（Mask）乘以 0 进行过滤，`NaN` 依然会沿着梯度反向传播路径污染整个计算图，最终导致训练崩溃。

## Task 3 Compare with the Official 3DGS Implementation
本作业为纯 PyTorch 实现，训练速度与显存效率远不如官方实现，且未实现 tile-based rasterizer 和adaptive Gaussian densification 等关键模块。我使用了lego、chair和playroom数据集运行 官方 3DGS：

<img src="picture_results/10.png" alt="官方3DGS" width="400">
<img src="picture_results/11.png" alt="官方3DGS" width="400">
<img src="picture_results/12.png" alt="官方3DGS" width="400">
<img src="picture_results/13.png" alt="官方3DGS" width="400">

因为官方3DGS中可以生成一个.ply文件，所以最后写了一个`export_ply.py`文件来把之前方法生成的.pt文件转化为.ply文件，便于分析和比较结果。

## 结果分析
### 3DGS by pure PyTorch 结果分析
对于chair文件夹的结果如下：

<img src="picture_results/14.png" alt="3DGS结果" width="400">

<img src="picture_results/render_mv_chair.gif" width="70%">

对于lego文件夹的结果如下：

<img src="picture_results/render_mv_lego.gif" width="70%">

转化成.ply文件后的效果如下：

<img src="picture_results/15.png" alt="chair3DGS结果" width="400">
<img src="picture_results/16.png" alt="chair3DGS结果" width="400">

### Offical 3DGS结果分析
使用lego、chair和playroom数据集，运行官方3DGS代码后的.ply结果截图分别如下：

<img src="picture_results/20.png" alt="offical3DGS结果" width="400">
<img src="picture_results/21.png" alt="offical3DGS结果" width="400">
<img src="picture_results/22.png" alt="offical3DGS结果" width="400">

<img src="picture_results/17.png" alt="offical3DGS结果" width="400">
<img src="picture_results/18.png" alt="offical3DGS结果" width="400">
<img src="picture_results/19.png" alt="offical3DGS结果" width="400">

动图/视频渲染后的结果如下：

<img src="picture_results/point_cloud_iteration_30000_chair.gif" width="70%">
<img src="picture_results/point_cloud_iteration_30000_lego.gif" width="70%">

需要的显存较大，有些场景甚至需要24GB的内存，如果电脑/服务器提供不了较大内存只能缩小照片的大小。
训练时间较长，单个场景的构建使用T4 GPU需要40-60分钟不等，使用A100 GPU需要10-20分钟不等。
使用其他数据集，4个场景，包含drjohnson、playroom、train、truck4个文件夹，每个文件夹中分别包含307、234、301、251张图片，渲染结果如下：

<img src="picture_results/23.png" alt="offical3DGS结果" width="400">
<img src="picture_results/24.png" alt="offical3DGS结果" width="400">
<img src="picture_results/25.png" alt="offical3DGS结果" width="400">
<img src="picture_results/26.png" alt="offical3DGS结果" width="400">

### 官方代码的优化策略 1 - 自适应密度控制
在论文中提到：每迭代 100 次执行一次，重点关注视图空间中“位置梯度过大”的区域 。

克隆 (Clone) —— 解决“欠重建”
场景： 针对细节缺失区域的小体积高斯球 。
操作： 复制相同大小的高斯球，并沿位置梯度方向移动，填补几何空白 。

分裂 (Split) —— 解决“过度重建”
场景： 针对覆盖面积过大且方差高的大体积高斯球 。
操作： 强制将一个大球分裂为两个小球，并将缩放比例缩小为原来的 1/1.6 。

透明度剔除： 定期清理 α 值低于设定阈值的无效透明高斯点 。

体积限制： 强制移除在世界空间中体积过大，或在屏幕空间投影足迹过大的“巨型”高斯体 。

每隔 3000 次迭代，将所有高斯的 α 值强行重置为接近 0 。

效果： 有效的几何体会再次被优化拉高透明度，而遮挡相机的错误漂浮物则会因无法恢复而被系统自动剔除 。

### 官方代码的优化策略 2 - 极速光栅化管线
将整个屏幕切分为 16×16 像素的小瓦片 (Tiles) 。执行严格的视锥体剔除 (Frustum Culling)，丢弃不可见的高斯体 。

为每个留在视野内的高斯实例分配包含“瓦片 ID + 深度”的 Key 值。使用极速的 GPU 基数排序 (Radix Sort) 瞬间完成所有高斯体的空间与深度排序 。

为每个瓦片分配一个线程块，利用共享内存并行加载高斯数据 。从前向后逐像素混合颜色与透明度 α 。一旦某像素透明度 α 达到饱和（即光线无法穿透），计算立即停止，节省算力 。

无限制反向传播：不限制接收梯度更新的高斯体数量，完美适应任何复杂深度的场景 。

仅需保存前向传播结束时的“最终累积不透明度” 。

在反向遍历时，通过做除法即可倒推恢复出所有中间步骤的透明度，从而极快地完成梯度计算并更新参数 。

### 多方面对比
| 评估维度 | 纯 PyTorch 实现 | 官方 C++/CUDA 实现 |
| :--- | :--- | :--- |
| **渲染质量** | 低 | 高 |
| **训练速度** | 慢 | 快 |
| **显存占用** | 少 | 多 |

造成上述差别的主要有以下几点原因：

#### 1.像素级全局开销和基于瓦片的光栅化器 (Tile-based Rasterizer)

* **纯 PyTorch 版的局限 ($O(N \times H \times W)$)：**
    在手写的 PyTorch 实现中，为了计算每个高斯点在全图所有像素的影响力，代码构建一个形状为 `(N, H, W)` 的高斯概率密度张量。
    在数学形式上，它对空间中 $N$ 个高斯点与 $H \times W$ 个像素点进行了全组合的隐式密集计算。当分辨率提升（如 $1024 \times 1024$）或高斯点数 $N$ 随迭代增加时，这个三维张量在显存中会呈现几何级数增长。例如：
    $$\text{Memory Size} = N \times H \times W \times 4 \,\text{bytes}$$
    若 $N=100,000, H=W=1024$，仅这一个中间激活层就需要占用 **400 GB** 的显存，所以必须限制点数并极限降低分辨率，且计算效率因极低的高效并行度而极其低下。
* **官方 Tile-based Rasterizer 的方案 ($O(M \log M) + O(\text{Tiles} \times N_{\text{tile}})$)：**
    官方实现放弃了全局像素矩阵的计算。它首先将屏幕划分为 $16 \times 16$ 像素的 **Tiles（瓦片）**。
    1.  **视锥裁剪与筛查：** 仅保留与当前 Tile 轴对齐包围盒（AABB）相交的 2D 高斯体，剔除全图 90% 以上的不相关高斯点。
    2.  **基数排序：** 利用 GPU 硬件极度优化的前缀和算子，将每个 Tile 内的高斯点按照深度（Depth）进行快速排序。
    3.  **单线程块处理：** 每个 Thread Block 负责一个 Tile。Block 内的线程并发读取当前 Tile 对应的高斯队列，利用共享内存按序进行 Alpha 混合渲染。当累积透射率 $T_i$ 降至机器零（如 `< 0.0001`）时，**提前终止光线**，直接跳过后续成千上万个高斯点的无用计算。

#### 2.静态点云初始化和自适应高斯致密化 (Adaptive Gaussian Densification)

* **纯 PyTorch 版的局限：**
    纯 PyTorch 版代码中的高斯点数在训练开始后是**完全固定**的。模型只能在现有的稀疏初始点（通常由 SfM 或随机采样生成）上优化位置和缩放。
    * 在重建不足的区域（如大面积结构复杂的空缺表面），由于缺乏高斯体覆盖，画面会出现严重的空洞和色块模糊。
    * 在重建过度的区域（如稀疏点云在背景中形成长条状异常结构），高斯体无法自我销毁，导致画面充满片状的“浮游伪影”。
* **官方的自适应致密化机制：**
    官方实现每隔固定的迭代步数，会统计每个高斯点在位置 $\mu$ 上的**平均视图空间位置梯度** $\nabla_{\mu} \mathcal{L}$。若梯度模长大于设定阈值 $\tau_{\text{pos}}$，说明该区域尚未完全收敛，使用如下自适应致密化步骤进行优化：
    1.  **克隆 (Clone)：** 若该点的缩放因子的最大值 $S_{\max}$ 小于阈值，判定为结构缺失，原位复制一个相同大小的高斯体，并沿梯度方向微调。
    2.  **分裂 (Split)：** 若该点的缩放因子 $S_{\max}$ 较大，判定为空间跨度过大导致的分辨率不足，将其分裂为两个体积缩小为原本 $\frac{1}{1.6}$ 并在位置上进行高斯采样的子高斯体。
    3.  **大范围裁剪 (Pruning)：** 定期将不透明度 $o_i$ 低于极小值（如 $<0.05$）的高斯点直接从显存中抹除，或者将体积过大（覆盖太多像素）的高斯体强制销毁。
    这种动态调整点云拓扑的能力，使官方版能用最少的点表达平坦区域，集中数百万个极小的高斯球去死磕乐高积木拼缝、锐利边缘等高频细节。

#### 3.漫反射颜色表达和高阶球谐函数 (Spherical Harmonics)

* **纯 PyTorch 版：**
    仅将颜色参数存储为对数空间下的普通三维向量 `colors: (N, 3)`，经 Sigmoid 激活后得到标准的局域 RGB 颜色。这种设计隐式假设场景是完美的**朗伯体（漫反射表面）**。无论相机从哪一个角度看这块乐高，其反射的颜色都是完全恒定的。
* **官方版：**
    为每个高斯点引入了 **球谐系数（Spherical Harmonics, SH）**。颜色不再是一个定值，而是视角的函数。公式表达为：
    $$C(\theta, \phi) = \sum_{l=0}^{l_{\max}} \sum_{m=-l}^{l} k_l^m Y_l^m(\theta, \phi)$$
    官方 30000 步模型最高采用 3 阶球谐函数，每个高斯点需要存 $3 \times (3+1)^2 = 48$ 个通道的系数（$f\_{\text{dc}}$ 与 $f_{\text{rest}}$）。
    * 当相机轨道旋转时，渲染器根据当前视线矢量，动态求解球谐基函数 $Y_l^m$，实时算出当前视角下的高光和环境色。官方版生成的 `.ply` 模型在拖入外部查看器（[SuperSplat](https://superspl.at/editor)）时，能完美呈现出乐高塑料积木表面流动的镜面高光反射和深浅不一的阴影层次。

## 代码上传及补充说明
本次作业均在Colab上运行完成，所有文件都是在本地写完之后上传到 `Google Drive `后使用 `Assignment4_3DGS.ipynb `运行。

本项目运行后文件夹大小如下图（总大小1.4 GB ）：

<img src="picture_results/1.png" alt="3DGS文件夹" width="200">

由于部分 `.ply `文件大小太大，只保留了必要的文件：

<img src="picture_results/27.png" alt="ply文件" width="400">

且为了上传的方便，只保留了基本的代码和最终结果，去掉了部分中间结果。
