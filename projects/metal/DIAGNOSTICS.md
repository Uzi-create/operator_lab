> 2026-09-26 目录已整理。本文保留原算法说明与历史测量；旧命令/导入方式请按[当前目录指南](../../README.md)使用，历史清单不代表现行布局。

# 新增缺陷诊断算子与统一验证

本轮修复已有接口边界问题，新增两个独立算子，不改变 v1/v2 检测门限。
代码：defect_ops.py；演示：demo_defect_ops.py。

## 一键运行

在 `.` 中：

```bash
# 四张实拍：复核分组 + 局部对比度热图
bash operator_lab/metal/run-metal.command --diagnostics

# 构建动态库、运行两套 Python 测试和全部 C++ 内核的 sanitizer 检查
bash operator_lab/metal/run-metal.command --verify --sanitizers

# 仍可运行旧入口和 v1/v2 对照
bash operator_lab/metal/run-metal.command
bash operator_lab/metal/run-metal.command --compare
```

测试命令任一阶段失败都会返回非零状态并停止，不会跳过失败后宣称通过。
验证时要有支持 AddressSanitizer/UndefinedBehaviorSanitizer 的 C++ 编译器。
脚本的解释器选择规则仍同 README；依赖 NumPy/OpenCV。

## 1. 掩膜内环形背景对比度

```python
from defect_ops import local_defect_contrast

result = local_defect_contrast(gray_float32, metal_mask,
    inner_radius=3, outer_radius=12, noise_floor=0.02)
score = result['score']
valid = result['valid']
```

围绕每个中心像素取外方框减内方框的环形邻域：

```text
mu = 有效环形邻域灰度均值
variance = 有效环形邻域灰度方差
residual = 中心灰度 - mu
score = abs(residual) / sqrt(variance + noise_floor²)
```

内框排除了缺陷附近像素，减少缺陷自身污染背景统计。
利用 OpenCV boxFilter 的局部求和计算数目、灰度和、平方和；临时图使用 double。
只计算 mask 内的样本，不把 ROI 外黑色补零当成真实背景。
默认至少 8 个有效邻居，且占裁剪后几何环形面积的 50%；否则 valid=False、数值置零。
因此 score=0 既可能表示无差异，也可能表示不可评估，必须一起看 valid。
灰度要求有限 float32/float64 [0,1]；掩膜形状一致；半径使用工作图像素。

这不是缺陷概率：反光、纹理和阴影同样可能产生高分。
使用窄环可能比较到缺陷本身，使用宽环可能跨越曲面明暗变化。
该算子用于解释和筛选候选，尚未设生产 OK/NG 阈值。

## 2. 方向约束的断裂片段分组

```python
from defect_ops import group_defect_fragments

groups = group_defect_fragments(scratch_mask,
    max_gap=12, max_angle_deg=25, max_lateral=2)
```

先求 8 连通域，按 PCA 提取长轴及两端点。只有同时满足以下条件才建立关联：

- 两个片段均足够细长，且主方向相近；
- 最近端点距离不超过 max_gap；
- 两端点的连线也与两个长轴对齐；
- 连线相对长轴的横向偏移不超过 max_lateral。

之后用并查集形成复核组。端点距离使用像素中心间距离，不能直接理解为缺失像素数。
不会填补任何像素，不修改输入 mask；输出保留 member labels、关联 links、面积与包围框。
面积是原片段面积之和，不是分组框的面积。传递关联可连接多段，因此总长度可能超过 max_gap。
组件编号只在当前输入有效，不是跨帧跟踪 ID。

避免把平行相邻划痕合并成一条，但已经相交且在输入中连通的结构无法在此处拆开。
PCA 端点是近似几何描述，弯曲或分叉划伤可能不合并；这里只归组，不宣称是真实缺陷数。
复杂度 O(K²)，默认最多 512 个合格连通域，超过时明确报错，避免静默长时间运算。

## 修复项

- 原图 bbox 改为左上向下取整、右下向上取整并裁剪，确保框覆盖响应且不超出原图。
- 小数 ROI 顶点四舍五入后裁剪，防止接近图像边界时生成越界坐标。
- 动态库在临时目录完成编译，再原子替换；失败保留旧库，并清理临时文件。
- ridge 的 auto 模式在动态库存在但架构错误等无法加载时回退 NumPy；native 模式明确报错。
- 环形对比度设置噪声下限的最小值，避免过小数值平方下溢导致无效分数。
- 新增统一 verify.py 和持久化 C++ 检查源码 tests/native_checks.cpp，检查可重复执行。

动态库重建后，已有 Python 进程可能仍持有旧库，应重启调用进程；不支持进程内热替换。

## 四图结果

本机 macOS ARM64，工作图 720×960，每项预热后测量五次，包含新增两个算子的计算和分配：

| 图片 | 输入片段 | 复核组 | 新增耗时 |
|---|---:|---:|---:|
| 1 | 3 | 3 | 12.21 ms |
| 2 | 12 | 11 | 12.32 ms |
| 3 | 3 | 3 | 12.35 ms |
| 4 | 14 | 14 | 12.52 ms |

不含先前 v2 流程、读图和保存，不能把这一列当成完整检测耗时。
结果在 `../output/metal_diagnostics/index.html`。
左图显示分组框，青框代表多片段组；右图显示 score/6 截断后的热图，黑色为不可评估区域。
JSON 坐标是 **720×960 工作图坐标**，与 v1/v2 的原图坐标不同，文件中已记录尺寸与单位。
`*_contrast.npz` 保留未做显示截断的 score、residual、background、valid 和邻域计数。

## 验证结果与限制

本轮统一验证通过：基础/比赛/构建 34 项 + 金属模块 25 项 = **59 项 Python 测试**。
新增测试包含逐像素暴力环形参考、掩膜背景隔离、空支持、旋转片段、平行/垂直不误合并、
阈值边界、组件数量限制、库加载失败、编译失败保留旧文件及坐标回映射。
全部 C++ 内核还通过本次小尺寸多参数 AddressSanitizer / UndefinedBehaviorSanitizer 检查。

测试通过只覆盖所测条件，不保证没有错误。实拍仍存在误报/漏检，
未用标注数据验证识别准确率，也不能仅凭色斑确认氧化。
