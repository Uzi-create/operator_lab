> 2026-09-26 目录已整理。本文保留原算法说明与历史测量；旧命令/导入方式请按[当前目录指南](../../README.md)使用，历史清单不代表现行布局。

# 通用视觉与机器人算子（2026-09-25）

新增六类功能，沿用原有浮点图像、像素坐标和显式相机参数约定。实现位于 `measurement_ops.py`、`feature_ops.py`、`perception_ops.py`，点云的融合内核位于 `perception_raster.hpp`。这是独立的离线视觉工具集，不宣称 HALCON 接口兼容。

## 新算子

| 功能与入口 | 输入 / 输出 | 实现及适用范围 |
|---|---|---|
| 平移配准 `estimate_translation` | 同尺寸灰度图 → 位移、响应、对齐矩阵 | OpenCV FFT 相位相关；亚像素平移，不处理旋转或尺度变化 |
| 平面模板定位 `create_orb_template` + `locate_planar_template` | 缓存模板 + 场景 → 单应矩阵、四角、匹配点、内点、RMS | OpenCV ORB、Hamming、RANSAC；支持旋转、尺度及适度透视，要求足够纹理 |
| 条纹宽度 `measure_stripes` | 测量线段 → 完整亮条/暗槽的边缘、中心及宽度 | 横向平均、梯度峰值与亚像素插值；只配对相邻反向边缘，不跨过其他边缘拼接 |
| 圆边测量 `measure_circle` | 近似圆心/半径 → 圆心、半径、边缘点、内点及 RMS | 一次批量径向采样，再稳健拟合；用于已有定位的圆孔/圆边，不是全图找圆 |
| 点云深度图 `pointcloud_to_depth` | 相机坐标系点云 + 内参 → 深度、有效掩码及源索引 | C++ 融合投影/z-buffer；保留最近正 Z，同深度取最早源点 |
| 高度网格 `elevation_grid` | +Z 向上点云 + XY 范围/格宽 → 数量、最低/最高/平均高度及有效掩码 | C++ 两遍融合聚合；支持密度门槛、负坐标和末尾不完整网格 |

图像必须为非空 `float32/float64` 二维灰度，值域 `[0,1]`，坐标为像素中心 `(x,y)`。ORB 内部量化到 uint8；单应矩阵从模板坐标映射至场景坐标，四角顺序为左上、右上、右下、左下。特征模型缓存模板描述子，不缓存场景特征；模型使用不可写数组，可由不同调用共享。

配准返回 `shift_xy` 表示参考图到移动图的位移，对齐移动图的 `alignment_matrix` 使用反向位移。相位相关具有周期歧义，重复纹理或超过半图的位移可能定位错误；`response` 不是概率。低纹理、低响应或超过 `max_shift` 返回失败。

条纹宽度沿测量线方向，测量线不垂直于条纹时需要另做几何校正。圆边测量要求初始位置落在搜索带内；亮/暗极性描述从内向外的变化。支持点数、内点比例及角覆盖门槛共同筛选；拟合失败会给出原因。`success=False` 时不要使用拟合结果控制后续动作，诊断字段可能保留被拒绝的候选。亚像素数值不等于真实光学精度。

## 点云坐标与缺失值

两个点云算子接受有限 `N×3` 数组，空数组需为 `(0,3)`。非有限坐标直接报错；外参转换应先显式调用 `robot_ops.transform_points`，不会仅凭数组形状猜测坐标系。

- 深度使用相机光学坐标：X 向右、Y 向下、Z 向前，输出是 Z 而非欧氏距离。输入应已去畸变。像素取 `floor(u+0.5)`，有效足迹为 `[-0.5,width-0.5) × [-0.5,height-0.5)`；这与旧 `project_points` 的像素中心边界不同。缺失深度为 0，源索引为 -1，不能误作近处障碍物。
- 高度网格需要调用方提供 +Z 向上的坐标系；行随 Y 增大、列随 X 增大，`bounds_xy=(xmin,ymin,xmax,ymax)` 右/上边界不包含。空格高度为 NaN；点数低于 `min_points` 的格子保留统计，但 `valid=False`。最高点不是自动分类的障碍物，最低点也不是已确认地面。
- 深度图最多 16,777,216 像素，高度网格最多 4,000,000 格。单位沿用输入；输入以米为单位时，深度、高度和分辨率都以米解释。

`backend='auto'` 优先加载本地 C++ 动态库，不可用时回退 NumPy；可用 `native` 强制原生，或 `numpy` 指定参考实现。输入布局必要时转换为对齐 C 连续 float64；输出每次独立分配。C++ 在写出之前完成有限值、参数、大小、对齐及内存重叠检查，不使用 fast-math；离散像素投影禁用 FMA 合并，避免临界坐标落入不同像素。

## 运行示例

在本目录、安装 NumPy/OpenCV 的 Python 环境中运行：

```powershell
python -B build.py --native
python -B demo_vision_robot.py
python -B benchmark_vision_robot.py --repeats 31
python -B verify.py --native --demos --output output/verified_vision_robot
```

`--native` 构建针对当前 CPU；分发至其他电脑应采用默认构建或在目标电脑重新编译。依赖版本：本次使用 Python 3.11.4、NumPy 2.2.6、OpenCV 4.12.0、MinGW-w64 GCC。

```python
import numpy as np
from demo_vision_robot import make_inputs
from measurement_ops import estimate_translation, measure_stripes, measure_circle
from feature_ops import create_orb_template, locate_planar_template
from perception_ops import pointcloud_to_depth, elevation_grid

data = make_inputs()  # 可运行的合成输入，实际使用时替换为你的数据。
shift = estimate_translation(data['reference'], data['moving'], max_shift=(30, 30))
model = create_orb_template(data['template'])  # 同一模板只构建一次。
match = locate_planar_template(data['scene'], model)
widths = measure_stripes(data['measurement'], (350, 240), (600, 240))
circle = measure_circle(data['measurement'], (184, 242), 72, polarity='dark')
depth = pointcloud_to_depth(data['camera_points'], data['camera'])
grid = elevation_grid(data['world_points'], (-2, -1.5, 2, 1.5), .02, min_points=3)

if match['success']:
    print(match['corners_xy'], match['reprojection_rms'])
print([item['width_px'] for item in widths['stripes']])
print(grid['max_z'][grid['valid']])
```

示例页面：[output/vision_robot_new/index.html](../../projects/output/vision_robot_new/index.html)。`geometry.npz` 保存深度及网格数值，PNG 仅用于显示；`report.json` 保存与已知几何的误差检查。相机点云和高度网格点云分别生成，未暗中将两个坐标系混用。

## 性能与验证

以下是本批首次交付的历史测量。后续 ORB 优化、奇数尺寸配准修复及更严格的精度对照见 [准确性优先优化](../../projects/docs/history/ACCURACY_FIRST.md)；旧基准 SHA256 对应旧源码，不应当作当前版本清单。

本机 i5-13400F、Windows、OpenCV 默认 16 线程，最终测量的接口耗时中位数：

| 算子 / 规模 | 耗时 ms | 同次测量的 NumPy 参考 ms |
|---|---:|---:|
| 平移配准，640×480 | 7.720 | — |
| 条纹宽度，250 px 测量线，9 px 横向平均 | 0.559 | — |
| 圆边测量，128 条射线，12 px 搜索带 | 0.921 | — |
| ORB 模板构建，320×260，仅需一次 | 5.087 | — |
| ORB 缓存定位，640×480 场景 | 21.299 | — |
| 点云深度图，345,600 点 → 640×480 | 4.219 | 39.563 |
| 高度网格，200,000 点 → 200×150 | 2.575 | 17.919 |

点云两项分别约为同次 NumPy 参考的 9.4×、7.0× 速度。两者包括有限值检查与输出分配，没有改成较低分辨率或减少输入点来获取加速。数值对照中深度/索引/计数/极值一致，均值使用 `1e-14` 的绝对及相对容差。

补充 [OpenCV 单线程测量](../../projects/output/vision_robot_benchmark_threads1/RESULTS.md)：本例 ORB 定位约 55.51 ms，慢于默认 16 线程的 21.30 ms，因此没有在库内部修改全局线程数。不同轮次的频率、调度及缓存状态会影响结果，不能把所有差异归因于线程数。测试自定义线程数可运行 `python benchmark_vision_robot.py --opencv-threads 1 --output output/my_threads1`。

完整计时表、原始样本、P95、环境和源码/动态库 SHA256 见 [本机性能结果](../../projects/output/vision_robot_benchmark/RESULTS.md) 与 [benchmark.json](../../projects/output/vision_robot_benchmark/benchmark.json)。基准包含图像/点云检查与输出分配，排除输入生成、图像 IO；模板构建和缓存后的定位分别计时。预热 3 次，31 次交替顺序测量。

测试包括已知平移/透视/旋转与遮挡、条纹及圆形亚像素位置、随机纹理拒绝、无纹理/退化几何、缓存不变性、点云的独立标量参考、像素边界邻近浮点数、最近深度及源索引、空输入、非法参数、非对齐输入、极大有限高度的均值，以及 C++/NumPy 对照。默认通用构建及本机 `--native` 构建均验证点云接口。

最终通过 **123 项 Python 测试**（根目录 89 + metal 34，本批新增 32）、C++ 原生断言，以及 **10 组演示的 147 张 PNG** 解码检查。完整日志：[verification.log](../../projects/output/vision_robot_benchmark/verification.log)；本次演示：[已验证示例](../../projects/output/verified_vision_robot/vision_robot/index.html)。

合成示例中，模板最大角点误差 1.18 px，圆心误差 0.0083 px、半径误差约 -0.0032 px；两条已知宽度 32.4/25.4 px，测量为 32.436/25.429 px。深度反投影再光栅化完全一致，网格总点数守恒。误差值是单个已知合成场景的检查结果，不能外推为真实测量精度。

这些是离线合成数据与 Windows 本机性能，不是实际相机精度、实车安全或跨平台运行保证。未运行 Windows sanitizer；C++ 原生断言与接口测试会在验证入口执行。旧版算子仍保留，已有金属检测算法和阈值本轮未更改。

## 已有基础算子

这批接口可直接和先前模块组合：`vision_ops.py` 已有局部/环形统计、动态阈值、亮度校正、滞后阈值、孔洞填充、区域筛选及特征、平移模板匹配、线卡尺、稳健直线/圆拟合；`robot_ops.py` 已有深度反投影、投影、刚体变换、体素降采样、平面拟合、高度距离、已知对应点配准和深度法线。
