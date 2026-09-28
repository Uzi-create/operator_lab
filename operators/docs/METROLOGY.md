# 批量卡尺几何测量

`operators.metrology_ops` 提供 `measure_line` 和 `measure_rectangle`。它们在已有粗定位附近精测边缘，不负责全图识别，不把像素自动换算成毫米。

## 多卡尺直线

```python
from operators.metrology_ops import measure_line

result = measure_line(gray, start=(50, 100), end=(550, 140),
                      num_calipers=48, search_half_length=10,
                      width=5, polarity='both', threshold=.03)
if result['success']:
    print(result['point'], result['direction'], result['rms'])
```

在参考线段上均匀布置卡尺，沿其左法向 `(-dy, dx)` 搜索边缘，在参考线方向平均 `width` 个采样。`bright/dark` 以这个扫描方向定义，反转起终点也会反转极性。最近峰或最强峰经抛物线亚像素插值，再用正交 RANSAC 拟合。

`edge_points` 与 `inliers` 对应；`caliper_indices` 对应原卡尺序号；`supported_calipers` 指示整个采样区域是否在图内；`coverage` 表示内点在参考线段上的跨度，不证明中间每段都有边缘。返回的 `segment_xy` 覆盖被测内点，而不是无条件延长至整张图。支持不足、覆盖不足、方向偏差过大时明确失败。

## 矩形边界

```python
from operators.metrology_ops import measure_rectangle

result = measure_rectangle(gray, center=(320, 240), size=(230, 150), angle_deg=23,
                           search_half_length=6, num_calipers=24)
if result['success']:
    print(result['corners_xy'], result['width_px'], result['height_px'])
```

`angle_deg` 从图像 +X 转向 +Y；宽度沿该方向，高度沿其左法向。每边只测中间 80% 以避开角点，四边独立拟合后求交，不强行把有问题的形状修成正交矩形。检验相邻垂直性、对边平行性和角点外推范围；透视下的梯形应先校正或分别测直线。角点依照先验局部坐标的左上、右上、右下、左下返回。

像其他测量接口一样，失败结果可能保留候选诊断，使用数值前必须检查 `success`。阈值单位是归一化灰度每像素；输入为 float32/float64 `[0,1]` 灰度。

## 首轮验证与性能

7 项测试覆盖已知直线与旋转矩形、扫描反转、噪声、遮挡、缺失边、视野外卡尺、参数错误，以及批量采样与独立旧卡尺的对照。

本机 640×480，48 个卡尺，31 次交替计时中位数：独立调用旧卡尺并拟合约 **14.70 ms**，批量卡尺完整接口约 **0.94 ms**；四边各 24 个卡尺的矩形测量约 **3.15 ms**。批量处理一次校验图像，一次执行采样和滤波，减少 Python 分派及重复整图检查。独立参考与批量拟合的法向位置差约 `0.00019 px`；其浮点采样网格舍入略有不同，并未宣称任意输入逐位相等。

同时检查 4 组不同旋转、随机种子和噪声的已知真值。单个演示样例的直线位置误差约 `0.002 px`，矩形尺寸误差约 `-0.0055 / +0.0034 px`。这是合成样例误差，不能外推为实际相机精度。

运行 `python projects/vision_robot/demo_metrology.py` 查看示例；运行 `python projects/benchmarks/benchmark_metrology.py` 重测并保存原始样本、P95 和源码哈希。输出分别位于 `projects/output/metrology` 与 `projects/output/metrology_benchmark`。
