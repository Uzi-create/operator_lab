# 实验 1：同一算法，为什么速度差很多？

目标：用 iPhone 的同一帧调用两种 C++ 均值滤波实现，检查结果，再比较耗时。实验不需要标注。

## 先运行

```bash
cd /Users/cathypark/Documents/创建/operator_lab
../operator_lab_env/bin/python projects/camera/experiment_mean.py --camera 3
```

当前 iPhone 索引为 3，重新插拔后可能变化。退出其他相机预览后运行。N/P 切换半径，Q 退出。
四个区域依次为原始灰度图、优化结果、朴素结果、绝对差值乘 1000。显示中心原始像素裁片。
半径 15 对应 31×31 窗口，朴素实现会使实时预览明显变慢，属于本实验的观察结果。

固定同一张图采样并保存报告：

```bash
../operator_lab_env/bin/python projects/camera/experiment_mean.py --camera 3 --benchmark
```

输出位于 `projects/output/camera/experiment_mean/`，包括原图、结果图、对照图、RESULTS.md 和 report.json。
每个半径各预热一次，然后交替执行两种实现，各采样 7 次，报告中位数。整幅图逐点检查误差，包括边界；超过 1e-6 会报错。

## 按这条链读代码

1. `experiment_mean.py` 的 `main()`：`cap.read()` 取得 BGR 图像。
2. `prepare()`：转灰度、归一化到 [0,1]，装进 `array('f')` 连续浮点缓冲区。
3. `compare()`：调用下面两个接口，输入和半径完全相同。

```python
run(pixels, width, height, mode='mean', radius=radius)
run(pixels, width, height, mode='mean_naive', radius=radius)
```

4. `operators/core.py` 的 `run()`：检查参数，分配结果缓冲区，将名称转换为 1 或 3，再通过 ctypes 调用动态库的 `run_operator()`。
5. `operators/operators.cpp`：`mode == 3` 进入朴素邻域循环；`mode == 1` 进入积分图路径。
6. `compare()`：将返回缓冲区视为 NumPy 图像，比较全部像素；`panel()` 显示结果。

这两个都是 C++ 算子。区别不是 Python 与 C++，而是实现的计算量。

朴素版本在每个像素附近重新累加整个方框，工作量约为 `W × H × (2r+1)²`。
积分图版本先预计算累计和，每个方框用四个表项求和，工作量约为 `W × H`，半径变大时耗时增长通常较少。
两者在边界处都裁剪方框，并按实际像素数量求平均。

计时包括接口检查、分配与 C++ 计算，不包括采集、灰度转换、误差检查与显示；不能直接当作视频 FPS。
真实画面验证只说明该帧的数值结果一致，不能推导所有输入都逐位一致。

下一步学习：找到 C++ 的 `double sum = 0`，看朴素版本如何求和；再看 `rectangle()` 的四项相减，理解积分图为什么能减少重复计算。
