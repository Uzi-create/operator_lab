# 双目图像文件运行入口

`projects/vision_robot/run_stereo_pair.py` 读取左右 PNG/JPEG 与标定 JSON，依次执行缓存极线校正、双向 SGBM、米制深度和 XYZ 反投影。运行示例：

```powershell
python -B projects/vision_robot/run_stereo_pair.py --left 左图.png --right 右图.png --calibration stereo.json --output projects/output/my_stereo_pair --num-disparities 64 --min-depth 0.05 --max-depth 50
```

标定文件的字段如下；`T_right_from_left` 的平移单位决定深度和点云单位，示例为米。`distortion_left/right` 可省略或为 OpenCV 标准 4、5、8、12、14 系数向量；鱼眼模型不适用。

```json
{
  "left_camera": {"width": 320, "height": 240, "fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
  "right_camera": {"width": 320, "height": 240, "fx": 500.0, "fy": 500.0, "cx": 160.0, "cy": 120.0},
  "distortion_left": [0, 0, 0, 0, 0],
  "distortion_right": [0, 0, 0, 0, 0],
  "T_right_from_left": [[1, 0, 0, -0.12], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
}
```

输出包括 `rectified_left.png`、`rectified_right.png`、`depth.npy`、`disparity_px.npy`、`point_cloud_xyz.npy`、仅供观察的 `depth_preview.png` 和 `report.json`。无效深度/视差是 NaN，XYZ 只保存有效点；原图的像素位置与点云行号不同，应在需要映射时直接调用底层 `depth_to_points` 获取像素索引。深度预览按本张图的分位数着色，不能比较不同图片颜色对应的绝对距离。

这一入口不会自动标定相机、同步左右曝光、测量极线残差或判断纹理是否可靠。`success` 仅表示至少有一个有效深度点，不是现场精度认证。`demo_stereo_file_workflow.py` 生成可重复的合成左右图和 JSON，验证完整读写及 5 m 尺度；真实设备仍需用独立量块或测距结果验证。
