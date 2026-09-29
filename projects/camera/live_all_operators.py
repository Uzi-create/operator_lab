"""USB 摄像头 + 38 项算子实时浏览。

建议阅读顺序（不用从画框函数开始）：
1. 文件末尾 main()：程序入口，打开相机并循环读取一帧。
2. process()：把一帧转成灰度，找到当前算子的调用函数并执行。
3. verify_camera_operators.py 的 cases()：登记名字与实际调用的对应关系。
4. operators/core.py 的 run()：基础算子通过 ctypes 调用 C++ 动态库。
5. 回到 render_result() / panel()：将返回值画出来，交给 cv2.imshow 显示。

本文件的 main、process、panel 等都是项目自定义函数。
cv2.VideoCapture / cvtColor / imshow 等是 OpenCV 提供的函数或类。
"""
# 直接用 python 文件路径启动时，补上项目根目录，让 import operators 等能找到包。
# parents[2] 在这里就是 operator_lab；这不会安装库，也不会修改系统环境。
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter, time_ns

import cv2
import numpy as np
from projects.camera import verify_camera_operators as check

# check 是另一个 Python 模块的别名；check.cases 表示调用那个模块里的 cases 函数。
# NAMES 只决定左侧菜单的顺序，不包含算法。前 10 项来自 core 的 mode 名称表。
NAMES = tuple(check.core._MODES) + (
    'canny', 'distance_transform', 'signed_distance', 'feather', 'ridge_response',
    'box_stats', 'ring_stats', 'dynamic_threshold', 'illumination_correct',
    'local_defect_contrast', 'hysteresis_threshold', 'fill_holes', 'select_regions',
    'region_features', 'group_defect_fragments', 'measure_edges', 'measure_stripes',
    'measure_circle', 'measure_line', 'measure_rectangle', 'fit_line', 'fit_circle',
    'estimate_translation', 'match_template', 'create_shape_template', 'match_shape',
    'create_orb_template', 'locate_planar_template')
# 一个统计算子返回多张图，V 键依次切换这些字段。
FIELDS = ('score', 'background', 'variance', 'residual', 'valid', 'support_count')


def xy(point):
    # 自定义坐标小工具：浮点坐标四舍五入成 OpenCV 绘图需要的整数 (x, y)。
    return tuple(np.rint(np.clip(point, -32760, 32760)).astype(int))


def draw_points(image, points, color=(0, 220, 255)):
    # 自定义绘图函数：给算子返回的每个点画一个小圆，不做图像检测。
    for point in np.asarray(points).reshape(-1, 2):
        cv2.circle(image, xy(point), 2, color, -1)


def draw_geometry(image, result):
    """Rejected measurements show their observed points, never an accepted fit."""
    if not isinstance(result, dict): return
    if result.get('edge_points') is not None: draw_points(image, result['edge_points'])
    # 未通过测量检查时，只画观测点；不要把诊断用的拟合画成绿色成功结果。
    if result.get('success') is False: return
    if result.get('corners_xy') is not None:
        cv2.polylines(image, [np.array([xy(p) for p in result['corners_xy']], np.int32)], True, (0, 255, 0), 2)
    if result.get('segment_xy') is not None:
        a, b = result['segment_xy']
        cv2.line(image, xy(a), xy(b), (0, 255, 0), 2)
    elif result.get('point') is not None and result.get('direction') is not None:
        center, direction = result['point'], result['direction']
        cv2.line(image, xy(center-direction*max(image.shape)), xy(center+direction*max(image.shape)), (0, 255, 0), 2)
    if result.get('center') is not None and result.get('radius') is not None:
        radius = float(result['radius'])
        if 0 < radius < 2*max(image.shape):
            cv2.circle(image, xy(result['center']), round(radius), (0, 255, 0), 2)
    for match in result.get('matches', []): draw_geometry(image, match)


def render_result(frame, name, value, context, field=0):
    # value 是算子返回值：可能是像素数组、字典或区域列表，并不一定是一张图。
    # context 是 cases 准备的辅助信息（扫描起止点、模板位置等），供显示使用。
    info = []
    chosen = value
    if name in ('box_stats', 'ring_stats', 'local_defect_contrast'):
        available = [key for key in FIELDS if key in value]
        key = available[field % len(available)]
        chosen = value[key]
        info.append('Map: '+key+' (press V for next map)')
    # preview 是项目自定义函数：将可显示的数组转换为 uint8 灰度图。
    # 距离/分数等超出 [0,1] 的数值会归一化显示，原始 value 不被修改。
    image = check.preview(chosen, frame.shape[:2])
    if image is not None:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if name in ('distance_transform', 'signed_distance') or info:
            info.append('Display normalized when outside [0,1]; numeric result uses original units')
        return result, info
    # 返回值不是整张图时，在原图副本上画框、点、线；保留原始 frame 不变。
    result = frame.copy()
    h, w = frame.shape[:2]
    if name.startswith('measure_'):
        if name in ('measure_edges', 'measure_stripes', 'measure_line'):
            cv2.line(result, xy(context['start']), xy(context['end']), (255, 180, 0), 1)
        elif name == 'measure_circle':
            cv2.circle(result, (w//2, h//2), min(w, h)//4, (255, 180, 0), 1)
        elif name == 'measure_rectangle':
            cv2.rectangle(result, (w//4, h//4), (3*w//4, 3*h//4), (255, 180, 0), 1)
        info.append('Blue: approximate search geometry; yellow: observed points; green: accepted fit')
        for edge in value.get('edges', []): draw_points(result, [edge['xy']])
        for stripe in value.get('stripes', []):
            cv2.line(result, xy(stripe['start_xy']), xy(stripe['end_xy']), (0, 255, 0), 3)
        info.append(f'Edges: {len(value.get("edges", value.get("edge_points", [])))}; stripes: {len(value.get("stripes", []))}')
    if name in ('fit_line', 'fit_circle'):
        draw_points(result, context['points'])
        info.append('Fit of sampled scene edges; this does not identify a particular object')
    if name in ('region_features', 'group_defect_fragments', 'match_template'):
        for item in value:
            x, y, bw, bh = item['bbox_xywh']
            cv2.rectangle(result, xy((x, y)), xy((x+bw, y+bh)), (0, 255, 0), 2)
        info.append(f'Results: {len(value)}')
    if name == 'create_shape_template':
        x, y, bw, bh = context['template_box']
        cv2.rectangle(result, (x, y), (x+bw, y+bh), (255, 180, 0), 1)
        draw_points(result, context['models']['shape'].points_xy + (x, y))
        info.append('Shape template edges; angle=0, scale=1')
    if name == 'create_orb_template':
        points = context['models']['orb'].points_xy
        draw_points(result, points)
        info.append(f'ORB template keypoints: {len(points)}')
    if name in ('match_template', 'match_shape', 'locate_planar_template'):
        info.append('SAME-FRAME template demonstration; not a tracking accuracy test')
    if isinstance(value, dict):
        draw_geometry(result, value)
        if 'success' in value:
            info.append('DETECTED' if value['success'] else 'NO DETECTION: '+str(value.get('reason', '')))
        for key in ('rms', 'radius', 'width_px', 'height_px', 'response'):
            if value.get(key) is not None: info.append(f'{key}: {float(value[key]):.3f}')
        if name == 'estimate_translation' and value.get('success'):
            shift = value['shift_xy']
            cv2.arrowedLine(result, (w//2, h//2), xy(np.array((w/2, h/2))+shift), (0, 255, 0), 2)
            info.append(f'Previous -> current shift: dx={shift[0]:.2f}, dy={shift[1]:.2f} px')
    return result, info


def process(frame, previous, name, field=0):
    # 【接算子的入口】frame 是 OpenCV 相机帧：uint8，形状 (高, 宽, 3)，BGR 顺序。
    # cvtColor：OpenCV 转灰度；astype：NumPy 转 float32；/255：亮度缩放到 [0,1]。
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    context = {}
    # cases 返回 [(算子名, 实现说明, 待调用函数), ...]。
    # 这里把列表改成字典。例如 jobs['adaptive'] = ('custom C++', 一个 lambda 函数)。
    # 创建这些待调用函数不会把 38 项都执行一遍；但 cases 会先计算共享掩码、边缘等输入。
    jobs = {n: (backend, op) for n, backend, op in check.cases(gray, previous, context)}
    # 按菜单选中的 name 找到函数。operation 是可调用对象，不是计算结果。
    backend, operation = jobs[name]
    # 两个匹配算子要先建立模板。这两行末尾 () 才会执行相应创建函数。
    if name == 'match_shape': jobs['create_shape_template'][1]()
    if name == 'locate_planar_template': jobs['create_orb_template'][1]()
    # perf_counter 是 Python 标准库计时函数。
    # 时间只包住当前 operation，未包含上面的准备步骤及下面的绘图。
    begin = perf_counter()
    # 【真正调用当前算子】选 adaptive 时，这句会进入 cases 登记的 lambda，
    # 再进入 core.run(..., mode='adaptive', radius=3, parameter=.08)。
    value = operation()
    milliseconds = (perf_counter()-begin)*1000
    # summarize 是项目自定义结果检查/摘要函数，NaN 等异常会抛出错误。
    check.summarize(value)
    result, info = render_result(frame, name, value, context, field)
    # 返回：用于显示的 BGR 图、说明文字、API 耗时、实现说明。
    return result, info, milliseconds, backend


def image_view(image, size, zoom=1., center=(.5, .5)):
    """高清视口：100% 时直接复制原始像素，不把整幅图缩小后再放大。

    size 是可见画布区域 (宽, 高)。显示不下的部分裁掉，通过 center 拖动查看。
    zoom=0 是适应窗口；zoom=1 是一个图像像素对应一个画布像素。
    """
    width, height = size
    h, w = image.shape[:2]
    scale = min(width/w, height/h) if zoom == 0 else zoom
    crop_w, crop_h = min(w, int(np.ceil(width/scale))), min(h, int(np.ceil(height/scale)))
    x = int(np.clip(round(center[0]*w-crop_w/2), 0, w-crop_w))
    y = int(np.clip(round(center[1]*h-crop_h/2), 0, h-crop_h))
    crop = image[y:y+crop_h, x:x+crop_w]
    if scale != 1:
        crop = cv2.resize(crop, (max(1, round(crop_w*scale)), max(1, round(crop_h*scale))),
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    view = np.full((height, width, 3), 16, np.uint8)
    ch, cw = min(height, crop.shape[0]), min(width, crop.shape[1])
    top, left = (height-ch)//2, (width-cw)//2
    view[top:top+ch, left:left+cw] = crop[:ch, :cw]
    return view


def view_layout(size):
    width, height = size
    pane_width = (width-300)//2
    pane_height = height-360
    return ((280, 90, pane_width, pane_height),
            (290+pane_width, 90, pane_width, pane_height))


def panel(frame, result, index, info, milliseconds, backend, cycling,
          zoom=1., center=(.5, .5), size=(1260, 760)):
    # 只负责界面排版：np.full 创建画布，OpenCV 负责菜单、文字和图像缩放。
    # 这里不调用检测算法。frame 是原图，result 是 process 返回的显示图。
    canvas = np.full((size[1], size[0], 3), 24, np.uint8)
    panes = view_layout(size)
    for i, name in enumerate(NAMES):
        y = 58+i*18
        if i == index: cv2.rectangle(canvas, (0, y), (269, y+18), (90, 70, 20), -1)
        cv2.putText(canvas, f'{i+1:02d} {name}', (8, y+14), cv2.FONT_HERSHEY_SIMPLEX, .43, (240, 240, 240), 1)
    cv2.putText(canvas, 'Click an operator', (8, 27), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 255), 1)
    cv2.putText(canvas, f'{index+1}/38  {NAMES[index]}  |  API {milliseconds:.2f} ms', (282, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)
    cv2.putText(canvas, 'Original', (282, 73), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
    cv2.putText(canvas, 'Result', (panes[1][0]+2, 73), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
    # 两边共享缩放与中心位置。左侧始终使用完整采集原图，不使用缩小后的处理输入。
    for (x, y, w, h), image in zip(panes, (frame, result)):
        canvas[y:y+h, x:x+w] = image_view(image, (w, h), zoom, center)
    label = 'FIT whole image' if zoom == 0 else f'{zoom*100:.0f}% detail view (drag to see other areas)'
    lines = [backend, label, '1: original pixels / F: fit whole image / +/- or wheel: zoom / drag: pan',
             'Click / N next / P previous / Space cycle / V map / R original / S save / Q quit',
             'Auto-cycle: '+('ON (2 seconds each)' if cycling else 'OFF'),
             'API timing excludes input preparation, prerequisite template creation and display.'] + info
    for i, text in enumerate(lines[:12]):
        cv2.putText(canvas, text[:150], (282, panes[0][1]+panes[0][3]+30+i*20), cv2.FONT_HERSHEY_SIMPLEX, .44, (220, 220, 220), 1)
    return canvas


def main():
    # argparse 是 Python 标准库：读取命令行 --camera 1、--operator adaptive 等。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=1)
    parser.add_argument('--operator', choices=NAMES, default='adaptive')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--max-width', type=int, default=0,
                        help='Processing width limit; 0 keeps captured resolution. Original preview/save always uses the captured frame.')
    parser.add_argument('--save-dir', type=Path, default=Path(__file__).resolve().parents[1]/'output/camera/originals')
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--cycle', action='store_true')
    args = parser.parse_args()
    if args.camera < 0 or args.frames < 0 or min(args.width, args.height) < 96 or (args.max_width != 0 and args.max_width < 96):
        parser.error('camera/frames >=0, dimensions >=96, max-width=0 or >=96 required')
    # 【接摄像头】VideoCapture 是 OpenCV 的采集类，cap 是创建出来的采集对象。
    # args.camera 是相机索引；darwin 表示 macOS，使用 AVFoundation 视频后端。
    # 视频后端负责取图，与下方显示的算子实现 backend 不是同一回事。
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY)
    # state：当前菜单序号、统计图字段序号、是否轮播、最后切换时刻。
    state = {'index': NAMES.index(args.operator), 'field': 0, 'cycling': args.cycle, 'changed': perf_counter(),
             'zoom': 1., 'center': (.5, .5), 'drag': None, 'size': (1260, 760),
             'shapes': [(1080, 1920), (1080, 1920)]}
    # previous：上一帧灰度图，给 estimate_translation 使用；count：已处理帧数。
    title, previous, count, errors = 'Operator Lab | All 38', None, 0, 0
    original_title, show_original = 'Operator Lab | Captured Original', False
    def click(event, x, y, flags, parameter):
        # 自定义鼠标回调：OpenCV 收到点击后调用它，根据菜单行号修改 state。
        if event == cv2.EVENT_LBUTTONDOWN and x < 270 and y >= 58:
            index = (y-58)//18
            if index < len(NAMES): state.update(index=index, field=0, changed=perf_counter(), cycling=False)
        # 鼠标拖动图像时，改变左右视口的共同中心位置，不修改图像数据。
        panes = view_layout(state['size'])
        pane_index = next((i for i, (px, py, pw, ph) in enumerate(panes)
                           if px <= x < px+pw and py <= y < py+ph), None)
        if event == cv2.EVENT_LBUTTONDOWN and pane_index is not None:
            state['drag'] = (x, y, state['center'], pane_index)
        if event == cv2.EVENT_LBUTTONUP: state['drag'] = None
        if event == cv2.EVENT_MOUSEMOVE and state['drag'] is not None:
            sx, sy, center, i = state['drag']
            h, w = state['shapes'][i]
            _, _, pw, ph = panes[i]
            scale = state['zoom'] or min(pw/w, ph/h)
            state['center'] = tuple(np.clip((center[0]-(x-sx)/(scale*w),
                                            center[1]-(y-sy)/(scale*h)), 0, 1))
        if event == cv2.EVENT_MOUSEWHEEL and pane_index is not None:
            delta = (flags >> 16) & 0xffff
            if delta >= 32768: delta -= 65536
            if delta:
                state['zoom'] = float(np.clip((state['zoom'] or 1.)*(1.25 if delta > 0 else .8), .125, 16))
    try:
        if not cap.isOpened(): raise RuntimeError('Camera cannot open: check index, permission and other apps')
        # 向驱动申请采集尺寸，不保证设备一定接受；实际尺寸由读到的 frame.shape 决定。
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not args.headless:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 1260, 760)
            # 把自定义 click 函数交给 OpenCV，后续由窗口事件触发。
            cv2.setMouseCallback(title, click)
        # --frames 0 时一直循环；指定正数则处理对应帧数后退出。
        while not args.frames or count < args.frames:
            # 【每次取一帧】read 是 cap 的方法。ok 是是否读到；frame 是 NumPy 图像数组。
            ok, frame = cap.read()
            if not ok or frame is None or min(frame.shape[:2]) < 96: raise RuntimeError('No usable camera frame')
            # 保留采集原图：R 窗口和 S 保存使用它，不受算子处理缩小的影响。
            captured = frame
            if args.max_width and frame.shape[1] > args.max_width:
                frame = cv2.resize(frame, (args.max_width, round(frame.shape[0]*args.max_width/frame.shape[1])))
                if min(frame.shape[:2]) < 96: raise ValueError('Processing image too small; increase --max-width or use 0')
            # 这份 gray 用于记录上一帧；process 内也会转换当前帧，供各算子使用。
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
            if previous is None or previous.shape != gray.shape: previous = gray
            if state['cycling'] and (perf_counter()-state['changed'] >= 2 or args.headless):
                state.update(index=(state['index']+1)%len(NAMES), field=0, changed=perf_counter())
            # 菜单序号 -> 算子字符串，例如 name = 'adaptive'。
            name = NAMES[state['index']]
            begin = perf_counter()
            try:
                # 【相机图送进算子】process 是本文件自定义函数，不是 OpenCV 原生函数。
                result, info, milliseconds, backend = process(frame, previous, name, state['field'])
                status = 'completed'
            except (ValueError, LookupError) as error:
                result, info, milliseconds, backend = frame.copy(), ['INPUT/SCENE REJECTED: '+str(error)], (perf_counter()-begin)*1000, 'No accepted result'
                status = 'scene_rejected'
            except Exception as error:
                result, info, milliseconds, backend = frame.copy(), ['EXECUTION ERROR: '+str(error)], (perf_counter()-begin)*1000, 'No accepted result'
                status = 'execution_error'
                errors += 1
            # 当前帧处理完成后，才更新 previous，避免平移估计拿当前帧和自身比较。
            # 第一帧没有历史帧，前面会用自身初始化。
            previous, count = gray, count+1
            info = [f'Captured: {captured.shape[1]}x{captured.shape[0]}; processing: {frame.shape[1]}x{frame.shape[0]}'] + info
            if count == 1 or args.headless:
                print(json.dumps({'frame': count, 'operator': name, 'status': status, 'api_ms': milliseconds, 'details': info}), flush=True)
            if not args.headless:
                # 【显示】先调用自定义 panel 拼好界面，再由 OpenCV imshow 送到窗口。
                # 根据窗口当前大小重新创建视口，不将旧的低分辨率画布拉大。
                try:
                    _, _, ww, hh = cv2.getWindowImageRect(title)
                    if ww >= 900 and hh >= 760: state['size'] = (min(ww, 3840), min(hh, 2160))
                except cv2.error:
                    pass  # 某些后端不提供窗口尺寸，仍可使用固定画布的缩放/拖动。
                state['shapes'] = [captured.shape[:2], result.shape[:2]]
                cv2.imshow(title, panel(captured, result, state['index'], info, milliseconds, backend, state['cycling'],
                                       state['zoom'], state['center'], state['size']))
                if show_original:
                    if cv2.getWindowProperty(original_title, cv2.WND_PROP_VISIBLE) < 1:
                        show_original = False
                    else:
                        cv2.imshow(original_title, captured)
                # waitKey 不仅读按键，也让 OpenCV 处理窗口事件；1 表示请求等待约 1 ms。
                key = cv2.waitKey(1) & 255
                if count == 1: print('PREVIEW: visible='+str(cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)), flush=True)
                if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1: break
                if key in (ord('n'), ord('N'), ord(']'), ord('p'), ord('P'), ord('[')):
                    step = -1 if key in (ord('p'), ord('P'), ord('[')) else 1
                    state.update(index=(state['index']+step)%len(NAMES), field=0, changed=perf_counter(), cycling=False)
                if key == 32: state.update(cycling=not state['cycling'], changed=perf_counter())
                if key in (ord('v'), ord('V')): state['field'] += 1
                if key == ord('1'): state.update(zoom=1., center=(.5, .5))
                if key in (ord('f'), ord('F')): state.update(zoom=0., center=(.5, .5))
                if key in (ord('+'), ord('='), ord('-')):
                    state['zoom'] = float(np.clip((state['zoom'] or 1.)*(.8 if key == ord('-') else 1.25), .125, 16))
                if key in (ord('r'), ord('R')):
                    show_original = not show_original
                    if show_original:
                        cv2.namedWindow(original_title, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(original_title, min(1280, captured.shape[1]),
                                         round(min(1280, captured.shape[1])*captured.shape[0]/captured.shape[1]))
                        cv2.imshow(original_title, captured)
                    else:
                        cv2.destroyWindow(original_title)
                if key in (ord('s'), ord('S')):
                    # PNG 保存未经算子处理、未经缩小的采集帧；不是传感器 Bayer RAW 数据。
                    path = save_original(captured, args.save_dir)
                    print(f'SAVED ORIGINAL: {path} ({captured.shape[1]}x{captured.shape[0]})', flush=True)
    finally:
        # 无论正常退出还是发生异常，都释放相机；有窗口时再关闭窗口。
        cap.release()
        if not args.headless: cv2.destroyAllWindows()
    return 1 if errors else 0


def save_original(frame, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'camera_original_{time_ns()}.png'
    if not cv2.imwrite(str(path), frame): raise IOError('Cannot save original camera frame')
    return path


# 直接运行本文件时执行 main；被 import 时不会自动打开相机。
# SystemExit 将 main 的返回值作为进程退出码，0 表示没有记录到执行错误。
if __name__ == '__main__': raise SystemExit(main())
