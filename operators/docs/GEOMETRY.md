> 2026-09-26 目录已整理。本文保留原算法说明与历史测量；旧命令/导入方式请按[当前目录指南](../../README.md)使用，历史清单不代表现行布局。

# 边缘、距离场与羽化算子

新增四个 Python 接口：canny、distance_transform、signed_distance、feather。
C++ 核心位于 geometry.hpp，仍只需要 Python 标准库和 C++17 编译器。

## 运行

```bash
cd "operator_lab"
python3 build.py
python3 -m unittest discover -v
python3 demo_geometry.py
```

效果页：output/geometry/index.html。可以指定 --width、--height、--repeats。

## 调用示例

输入为按行排列的 array('f')，长度 width×height，像素有限且处于 [0,1]。

```python
from array import array
from operators import canny, distance_transform, signed_distance, feather

w, h = 32, 24
image = array('f', (float(8 <= x < 24 and 6 <= y < 18)
                   for y in range(h) for x in range(w)))
edges = canny(image, w, h, low=0.08, high=0.20)
inside_distance = distance_transform(image, w, h)
sdf = signed_distance(image, w, h)
alpha = feather(image, w, h, feather_width=4)
# distance_transform 求到零背景的距离，因此要反转边缘图。
distance_to_edges = distance_transform(array('f', (1-v for v in edges)), w, h)
```

距离输出不处于 [0,1]，不能直接作为原有灰度算子的输入；显示前需要明确选择映射范围。

## Canny：多阶段边缘检测

1. 可分离五点二项式平滑，权重 [1,4,6,4,1]/16，近似 Gaussian，边界复制。
2. 3×3 Sobel 求横纵梯度，除以 4，再计算 L2 梯度幅值。
3. 按四个方向做非极大值抑制，平顶采用一侧严格大于、另一侧大于等于的规则。
4. 以大于等于 high 的强边缘为种子，迭代遍历 8 邻域，连接大于等于 low 的弱边缘。

参数要求 0 < low <= high；阈值采用上述浮点梯度单位，不是 OpenCV uint8 的阈值单位。
输出为 float32 的 0/1。最外一圈像素固定为 0；宽或高不足 3 时没有边缘。
不采用递归，因此长边缘链不会造成递归栈溢出。总时间 O(HW)，额外空间 O(HW)。
当前实现保留 6 张 double 中间图，约 48 字节/像素，另加追踪栈、输入输出。
方向量化、固定平滑核和抑制规则可能让弱轮廓在拐角或强度突变处断开；
迟滞连接不会跨越断点。不是完整复刻 OpenCV，也不宣称达到其性能。

原理参考：[OpenCV Canny 教程](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)。

## 精确欧氏距离变换

distance_transform 把 <=0.5 的像素中心视作背景点，计算每个像素到最近背景点的欧氏距离。
前景条件为 >0.5，距离单位为像素，背景本身距离为零。

没有使用反复扫描近似距离。先沿行、再沿列求抛物线下包络，
把二维平方距离的最小化分解为两次线性一维变换，最后开平方。
算法时间 O(HW)，空间 O(HW)，每行/列的候选点只入栈出栈一次。
“精确”指计算离散像素中心间的欧氏距离，而非 chamfer 近似；仍存在浮点舍入。
中间使用 double，输出 float32。

- 不把图像外部隐式当背景。
- 全黑图距离全为 0。
- 全白图没有背景点，距离全为 +inf。
- 一维变换会跳过无穷候选，避免 inf-inf 产生 NaN。

signed_distance 在内部返回到背景的正距离，在外部返回到前景的负距离。
全白为 +inf，全黑为 -inf。它计算到相反类别的像素中心的距离，
不是到连续几何轮廓的亚像素距离，边界两侧通常是 +1 和 -1。

参考：[Felzenszwalb 与 Huttenlocher 的距离变换论文与项目页](https://cs.brown.edu/people/pfelzens/dt/)。
核心代码自行实现，未复制项目页的实现代码。

## 距离驱动的羽化

feather 先计算有符号距离 d，再执行：

```text
t = clamp(0.5 + d/(2*feather_width), 0, 1)
alpha = t*t*(3-2*t)
```

feather_width 是像素单位的过渡半宽，必须大于零。输入先以 >0.5 二值化；
不保留输入已有的软 alpha。全白/全黑掩膜分别保持 1/0。
用途是生成可以用于图像混合的平滑 alpha；本算子本身不执行 RGB 合成。
它依据几何距离构造过渡，不会像大窗口均值那样把整个窄区域平均一遍；
但窄于羽化宽度的结构仍可能没有完全不透明的内部区域。

## 数据与显示

演示包括：带噪图 → Canny → 到边缘的距离，以及掩膜 → 内部/有符号距离 → 羽化。

PNG 距离图经过显示映射：到边缘距离在 24 px 截断，越近越亮；
内部距离在 80 px 显示为白；有符号距离采用 clamp(0.5+d/80)。
同目录 distance.f32、edge_distance.f32、signed_distance.f32 保留原始 float32 数据。
这些文件无头部，按行排列，使用本机字节序，尺寸和字节序见 benchmark.json。
在同字节序机器上读取：

```python
from array import array
from pathlib import Path
values = array('f')
values.frombytes(Path('output/geometry/distance.f32').read_bytes())
```

## 本轮验证与性能

完整 21 项 unittest 通过，包含旧接口回归、距离变换与暴力最近点对照、
3-4-5 距离、全空/全满图、单行单列、羽化对称性、Canny 独立二维卷积参考、
单像素阶跃轮廓、强弱边缘连接及非法输入。
AddressSanitizer 与 UndefinedBehaviorSanitizer 的小尺寸多模式及长弱链检查通过。

macOS ARM64 / Python 3.9.6 / C++17 -O3 / 640×360，预热一次后测量五次中位数：

| 算子 | 中位数 |
|---|---:|
| Canny，low=.08、high=.20 | 2.971 ms |
| 到 Canny 边缘的距离 | 1.452 ms |
| 掩膜内部欧氏距离 | 1.572 ms |
| 有符号距离 | 2.748 ms |
| 12 px 半宽羽化（含距离计算） | 3.013 ms |

包含 Python 调用、输入校验、输出及临时分配；不含生成数据、文件读写与首次动态库加载。
结果依赖当前输入和机器，不代表所有图像，也没有进行 OpenCV 性能对比。
