> 2026-09-26 目录已整理。本文保留原算法说明与历史测量；旧命令/导入方式请按[当前目录指南](../../README.md)使用，历史清单不代表现行布局。

# 复杂算子：局部模型、直方图与滑动极值

本轮增加 6 个可直接调用的模式：guided、clahe、erode、dilate、open、close。
实现位于 `advanced.hpp`，经原有 C 接口调用。无需额外 Python 依赖。

## 一键运行

```bash
cd "operator_lab"
python3 build.py
python3 -m unittest discover -v
python3 demo_advanced.py
```

打开 `output/advanced/index.html` 看输入输出对比及基准数据。
默认 640×360，可用 --width、--height、--repeats 修改尺寸和测量次数。

## 自引导滤波 guided

用途：保边平滑，或作为细节分离的基础。
本版使用输入自身作为引导图，暂不接受独立引导图。

```python
from operators import guided
out = guided(pixels, width, height, radius=7, scale=0.12)
```

算法分两轮局部统计：

```text
m = box_mean(I)
v = max(box_mean(I²) - m², 0)
a = v / (v + scale²)
b = (1-a) × m
q = box_mean(a) × I + box_mean(b)
```

与之前的 adaptive 不同，这里会对局部线性模型的系数 a、b 再求均值。
scale 为归一化灰度尺度，epsilon=scale²；越大通常越平滑。
使用 double 积分图，时间 O(HW)，内存 O(HW)。当前实现偏向可读性，
保留了多张中间图，峰值临时 double 缓冲区约 64 字节/像素，另加输入输出。
所有 box_mean 都裁剪越界窗口，并使用有效像素计数。

参考：[He、Sun、Tang 的 Guided Image Filtering 项目页](https://people.csail.mit.edu/kaiming/eccv10/index.html)。
这是已有算法的自写实现，不是新算法声明。

## CLAHE 局部对比度增强

用途：增强局部灰度差异，让低对比度纹理更明显。

```python
from operators import clahe
out = clahe(pixels, width, height, tile_size=64, clip_limit=3.0)
```

步骤：分块 → 256-bin 直方图 → 裁剪及重分配 → 累积分布映射 → 分块中心双线性插值。

tile_size 是每块的边长（像素），不是分块数量，必须 >=2。
clip_limit >=1；裁剪阈值为 max(1, clip_limit × tile_area / 256)。
裁掉的频数以浮点数均匀分配到全部 256 个 bin；分配后的频数允许再次超过初始阈值。
不足完整大小的边缘块使用真实尺寸，插值以真实中心为准，图像外沿固定使用最近中心。
输入灰度通过 floor(value×255+0.5) 映射到 bin，输出为累计概率，不减最小非零 CDF。

时间 O(HW + 256×分块数)，映射表额外空间约 1024×分块数 字节。
这是一种明确边界与重分配规则的 CLAHE 实现，不保证与 OpenCV 逐像素相同。
常量图不保证保持原亮度；噪声可能被放大；256-bin 量化可能在平滑渐变中出现色阶。

参考：[OpenCV CLAHE 原理说明](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)。

## 形态学：腐蚀、膨胀、开闭运算

用途：清理二值掩膜、删除亮点、填充暗孔，也支持归一化灰度图。

```python
from operators import run
small = run(pixels, width, height, 'erode', radius=2)
large = run(pixels, width, height, 'dilate', radius=2)
opened = run(pixels, width, height, 'open', radius=2)
closed = run(pixels, width, height, 'close', radius=2)
cleaned = run(opened, width, height, 'close', radius=2)
```

- erode：方形窗口最小值。
- dilate：方形窗口最大值。
- open：先腐蚀，再膨胀，去除比窗口小的亮结构。
- close：先膨胀，再腐蚀，填补比窗口小的暗结构。

半径 r 对应 (2r+1)×(2r+1) 方形结构元素，支持灰度，不只是二值图。
利用方形窗口的可分离性，先处理行、再处理列；每条线使用单调队列。
每个索引最多入队/出队一次，因此每个基础操作 O(HW)，不随窗口面积成比例增长。
越界部分忽略，等价于腐蚀填 +∞、膨胀填 -∞；radius=0 为恒等。
基础操作额外空间约 4HW 字节加索引队列；开闭运算多一张 float32 中间图。
parameter 对形态学没有作用。运算会改变细线、边缘和孔洞，不能当成无损修复。

## 实测与验证

本机 macOS ARM64，Python 3.9.6，C++17 -O3，640×360。
每个算子预热一次后运行 5 次，取中位数；包括 Python 调用、校验、输出及临时分配，
不含图片生成/写入和首次动态库加载。

| 算子 | 参数 | 中位耗时 |
|---|---|---:|
| guided | radius=7, scale=0.12 | 1.503 ms |
| clahe | tile_size=64, clip_limit=3 | 0.910 ms |
| erode | radius=2 | 0.858 ms |
| dilate | radius=2 | 0.935 ms |
| open | radius=2 | 1.617 ms |
| close | radius=2 | 1.769 ms |

数字来自此次演示输入，不代表其他尺寸/数据分布，也没有与 OpenCV 进行速度对比。
演示最后一行是 open 后接 close 的组合，表格分别计时单次 open 和 close。

13 项 unittest 全部通过，包含旧算子回归和新增：
独立直接窗口引导滤波参考、形态学逐窗口极值参考、开闭运算幂等性与大小关系、
CLAHE 标量 CDF 参考和已知解析值，以及单行/单列、超大半径、局部块、非法参数。
此外以 AddressSanitizer/UndefinedBehaviorSanitizer 运行了 C++ 小尺寸多模式检查，未报告错误。
Linux 构建路径尚未实测；仍仅支持 [0,1] 灰度 float32 连续输入。
