import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from PIL import Image, ImageTk
import tkinter as tk
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import vgg19

# ====== 在此处直接设置参数 ======
frames_dir = "D:/_mmlab_swjtu/data/night/bergen_night/bergen01_night"#目标帧的绝对路径
insert_img_path = "D:/_mmlab_swjtu/data/object/stone1.png"#异常物体的绝对路径
start_seq = 1500  # 起始序号，如0000则填0
# 自动生成输出目录
parent_dir = os.path.dirname(frames_dir)
folder_name = os.path.basename(frames_dir)
output_dir = os.path.join(parent_dir, folder_name + "insert")
# =================================

def get_sorted_frames(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    files.sort()
    return files

def get_frame_indices(start_idx, step, count):
    return [start_idx + i * step for i in range(count)]

def draw_reference_line(image_path, last_line=None, last_length=None):
    root = tk.Tk()
    root.title("请画一条参考线（左键画线，右键重画，回车确认，自动水平，必须比上一次短）")
    img = Image.open(image_path)
    tk_img = ImageTk.PhotoImage(img)
    canvas = tk.Canvas(root, width=img.width, height=img.height)
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
    line = []
    warning_text = tk.StringVar()
    warning_label = tk.Label(root, textvariable=warning_text, fg="red")
    warning_label.pack()

    # 如果有上一次的线，先画出来（绿色）
    if last_line is not None:
        x0, y0, x1, y1 = last_line
        canvas.create_line(x0, y0, x1, y1, fill='green', width=2, dash=(4, 2))

    def on_click(event):
        if event.num == 1:
            if len(line) < 2:
                line.append((event.x, event.y))
                if len(line) == 2:
                    # 保持水平：第二点y坐标强制等于第一点y
                    x0, y0 = line[0]
                    x1, _ = line[1]
                    y1 = y0
                    line[1] = (x1, y1)
                    length = abs(x1 - x0)
                    # 检查长度
                    if last_length is not None and length >= last_length:
                        warning_text.set("必须比上一次短，请重画！")
                        line.clear()
                        canvas.delete("all")
                        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
                        if last_line is not None:
                            canvas.create_line(last_line[0], last_line[1], last_line[2], last_line[3], fill='green', width=2, dash=(4, 2))
                    else:
                        warning_text.set("")
                        canvas.create_line(x0, y0, x1, y1, fill='red', width=2)
        elif event.num == 3:
            line.clear()
            warning_text.set("")
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
            if last_line is not None:
                x0, y0, x1, y1 = last_line
                canvas.create_line(x0, y0, x1, y1, fill='green', width=2, dash=(4, 2))

    def on_return(event=None):
        root.quit()

    def on_esc(event=None):
        root.esc_pressed = True
        root.quit()

    root.esc_pressed = False
    canvas.bind("<Button-1>", on_click)
    canvas.bind("<Button-3>", on_click)
    root.bind("<Return>", on_return)
    root.bind("<Escape>", on_esc)
    root.protocol("WM_DELETE_WINDOW", on_return)

    root.mainloop()
    try:
        root.destroy()
    except tk.TclError:
        pass
    if getattr(root, "esc_pressed", False):
        return "ESC", None
    if len(line) == 2:
        x0, y0 = line[0]
        x1, y1 = line[1]
        y1 = y0
        length = abs(x1 - x0)
        return length, (x0, y0, x1, y1)
    elif last_line is not None:
        x0, y0, x1, y1 = last_line
        length = abs(x1 - x0)
        return length, (x0, y0, x1, y1)
    else:
        return None, None

def adain(content_feat, style_feat, eps=1e-5):
    size = content_feat.size()
    content_mean, content_std = content_feat.view(size[0], size[1], -1).mean(2), content_feat.view(size[0], size[1], -1).std(2) + eps
    style_mean, style_std = style_feat.view(size[0], size[1], -1).mean(2), style_feat.view(size[0], size[1], -1).std(2) + eps
    normalized = (content_feat - content_mean[:,:,None,None]) / content_std[:,:,None,None]
    return normalized * style_std[:,:,None,None] + style_mean[:,:,None,None]

def extract_vgg_features(img_tensor, vgg, layers=[21]):
    features = []
    x = img_tensor
    for i, layer in enumerate(vgg.features):
        x = layer(x)
        if i in layers:
            features.append(x)
    return features[0]

def style_transfer(content_img, style_img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = vgg19(pretrained=True).to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((content_img.size[1], content_img.size[0])),
        T.Lambda(lambda x: x[:3, :, :]),  # 只取RGB
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    inv_transform = T.Compose([
        T.Normalize(mean=[-2.118, -2.036, -1.804], std=[4.367, 4.464, 4.444]),
        T.Lambda(lambda x: torch.clamp(x, 0, 1)),
        T.ToPILImage()
    ])
    content = transform(content_img).unsqueeze(0).to(device)
    style = transform(style_img).unsqueeze(0).to(device)
    content_feat = extract_vgg_features(content, vgg)
    style_feat = extract_vgg_features(style, vgg)
    t = adain(content_feat, style_feat)
    # 直接反归一化回图像
    t_img = t[0].cpu()
    t_img = inv_transform(t_img)
    # 保留alpha通道
    if content_img.mode == "RGBA":
        t_img = t_img.convert("RGBA")
        t_img.putalpha(content_img.split()[-1])
    return t_img

def insert_object_to_frame(frame_path, insert_img_path, object_width, save_path, line_coords=None, mask_save_path=None, center=None):
    frame = Image.open(frame_path).convert("RGBA")
    insert_obj = Image.open(insert_img_path).convert("RGBA")
    w_percent = object_width / insert_obj.width
    new_height = int(insert_obj.height * w_percent)
    # 兼容不同Pillow版本的LANCZOS写法
    try:
        resample_lanczos = Image.LANCZOS
    except AttributeError:
        resample_lanczos = Image.ANTIALIAS
    insert_obj = insert_obj.resize((int(object_width), new_height), resample_lanczos)
    # 让物体方向与参考线一致
    if line_coords:
        x0, y0, x1, y1 = line_coords
        import math
        angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
        insert_obj = insert_obj.rotate(-angle, expand=True, resample=Image.BICUBIC)
        cx = int((x0 + x1) / 2 - insert_obj.width / 2)
        cy = int((y0 + y1) / 2 - insert_obj.height / 2)
    elif center is not None:
        # 用插值中心点，确保物体中心在插值点
        cx = int(center[0] - insert_obj.width / 2)
        cy = int(center[1] - insert_obj.height / 2)
    else:
        cx, cy = 0, 0

    # === 风格迁移 ===
    # 取背景区域作为风格图
    # bg_crop = frame.crop((max(cx,0), max(cy,0), max(cx,0)+insert_obj.width, max(cy,0)+insert_obj.height)).convert("RGB")
    # insert_rgb = insert_obj.convert("RGB")
    # try:
    #     stylized_obj = style_transfer(insert_rgb, bg_crop)
    #     # 保留alpha通道
    #     if insert_obj.mode == "RGBA":
    #         stylized_obj = stylized_obj.convert("RGBA")
    #         stylized_obj.putalpha(insert_obj.split()[-1])
    #     insert_obj = stylized_obj
    # except Exception as e:
    #     print("风格迁移失败，使用原图:", e)

    # === 蒙版羽化 ===
    from PIL import ImageEnhance, ImageFilter
    mask = insert_obj.split()[-1].filter(ImageFilter.GaussianBlur(radius=2))
    insert_obj.putalpha(mask)

    # --- 修正ROI越界和泊松融合尺寸问题 ---
    frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGRA)
    insert_cv = cv2.cvtColor(np.array(insert_obj), cv2.COLOR_RGBA2BGRA)
    h_frame, w_frame = frame_cv.shape[:2]
    h_obj, w_obj = insert_cv.shape[:2]
    # 计算插入区域，确保不越界
    x1 = max(cx, 0)
    y1 = max(cy, 0)
    x2 = min(cx + w_obj, w_frame)
    y2 = min(cy + h_obj, h_frame)
    obj_x1 = max(0, -cx)
    obj_y1 = max(0, -cy)
    obj_x2 = obj_x1 + (x2 - x1)
    obj_y2 = obj_y1 + (y2 - y1)
    if x2 <= x1 or y2 <= y1 or obj_x2 <= obj_x1 or obj_y2 <= obj_y1:
        frame.convert("RGB").save(save_path)
        if mask_save_path:
            Image.new("L", frame.size, 0).save(mask_save_path)
        return

    # 只对有效区域做融合
    temp_bg = frame_cv.copy()
    temp_insert = np.zeros_like(frame_cv)
    temp_insert[y1:y2, x1:x2] = insert_cv[obj_y1:obj_y2, obj_x1:obj_x2]
    mask = insert_cv[obj_y1:obj_y2, obj_x1:obj_x2, 3]
    mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.merge([mask, mask, mask])
    temp_mask = np.zeros_like(frame_cv[:, :, 0])
    temp_mask[y1:y2, x1:x2] = mask[:, :, 0]
    temp_insert_rgb = temp_insert[:, :, :3]
    temp_bg_rgb = temp_bg[:, :, :3]
    temp_mask_gray = temp_mask

    # 检查mask尺寸与插入区域一致，否则跳过泊松融合
    roi_h, roi_w = y2 - y1, x2 - x1
    if (roi_h > 0 and roi_w > 0 and
        temp_insert_rgb.shape[0] == temp_bg_rgb.shape[0] and
        temp_insert_rgb.shape[1] == temp_bg_rgb.shape[1] and
        temp_mask_gray.shape == temp_bg_rgb.shape[:2]):
        center = (x1 + roi_w // 2, y1 + roi_h // 2)
        try:
            blended = cv2.seamlessClone(
                temp_insert_rgb, temp_bg_rgb, temp_mask_gray, center, cv2.NORMAL_CLONE
            )
            result_img = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
            result_img.save(save_path)
        except cv2.error as e:
            print("泊松融合失败，保存原图:", e)
            frame.convert("RGB").save(save_path)
    else:
        print("泊松融合尺寸不符，保存原图")
        frame.convert("RGB").save(save_path)

    # === 生成掩码图 ===
    if mask_save_path:
        # 将mask保存到output_dir/masks/下，文件名与原图一致
        mask_dir = os.path.join(os.path.dirname(save_path), "masks")
        os.makedirs(mask_dir, exist_ok=True)
        mask_filename = os.path.basename(mask_save_path)
        mask_save_path = os.path.join(mask_dir, mask_filename)
        mask_img = Image.new("L", frame.size, 0)
        obj_mask = insert_obj.split()[-1].point(lambda p: 255 if p > 10 else 0)
        mask_img.paste(obj_mask, (cx, cy), obj_mask)
        mask_img.save(mask_save_path)

def select_start_frame(frames, start_idx):
    idx = start_idx
    selected = [False]

    def show_frame():
        img = Image.open(os.path.join(frames_dir, frames[idx]))
        tk_img = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        canvas.image = tk_img
        label_var.set(f"当前帧: {frames[idx]} (索引: {idx})\n←/→调整，回车确认")

    def on_left(event):
        nonlocal idx
        if idx - 5 >= 0:
            idx -= 5
            show_frame()

    def on_right(event):
        nonlocal idx
        if idx + 5 < len(frames):
            idx += 5
            show_frame()

    def on_return(event):
        selected[0] = True
        root.quit()

    root = tk.Tk()
    root.title("选择起始帧")
    img = Image.open(os.path.join(frames_dir, frames[idx]))
    tk_img = ImageTk.PhotoImage(img)
    canvas = tk.Canvas(root, width=img.width, height=img.height)
    canvas.pack()
    label_var = tk.StringVar()
    label = tk.Label(root, textvariable=label_var)
    label.pack()
    show_frame()
    root.bind("<Left>", on_left)
    root.bind("<Right>", on_right)
    root.bind("<Return>", on_return)
    root.mainloop()
    root.destroy()
    return idx

def save_video_from_images(image_dir, output_video_path, fps=5):
    # 获取所有jpg图片并排序
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') and not f.endswith('_mask.jpg')]
    images.sort()
    if not images:
        print("没有找到合成图片，无法生成视频")
        return
    # 读取第一张图片确定尺寸
    first_img = cv2.imread(os.path.join(image_dir, images[0]))
    height, width = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            video_writer.write(img)
    video_writer.release()
    print(f"视频已保存到: {output_video_path}")

def save_mask_video_from_images(mask_dir, output_video_path, fps=5):
    # 获取所有掩码图片并排序
    images = [f for f in os.listdir(mask_dir) if f.endswith('_mask.jpg')]
    images.sort()
    if not images:
        print(f"没有找到掩码图片，无法生成掩码视频（{mask_dir}）")
        return
    # 读取第一张图片确定尺寸
    first_img = cv2.imread(os.path.join(mask_dir, images[0]), cv2.IMREAD_GRAYSCALE)
    height, width = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)
    for img_name in images:
        img_path = os.path.join(mask_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            video_writer.write(img)
    video_writer.release()
    print(f"掩码视频已保存到: {output_video_path}")

def main():
    os.makedirs(output_dir, exist_ok=True)
    frames = get_sorted_frames(frames_dir)
    # 找到末尾帧的索引
    end_idx = None
    for idx, fname in enumerate(frames):
        if fname.split('_')[-1].split('.')[0] == f"{start_seq:04d}":
            end_idx = idx
            break
    if end_idx is None:
        print("未找到末尾帧")
        sys.exit(1)

    # 允许用户通过方向键选择末尾帧
    end_idx = select_start_frame(frames, end_idx)

    # 1. 标注阶段，记录每个标注帧的中心点和长度
    idx = end_idx
    saved_frames = []
    centers = []
    lengths = []
    mark_indices = []
    last_line = None
    last_length = None
    mark_step = 20  # 当前是每隔20帧标注一次（即两帧编号差为20）
    while idx >= 0:
        frame_path = os.path.join(frames_dir, frames[idx])
        print(f"处理帧: {frames[idx]}")
        length, line_coords = draw_reference_line(frame_path, last_line=last_line, last_length=last_length)
        if length == "ESC":
            print("按ESC，停止标注并开始合成视频")
            break
        if length is None:
            print("窗口关闭，继续下一张图片")
            idx -= mark_step
            continue
        save_path = os.path.join(output_dir, frames[idx])
        mask_save_path = os.path.splitext(save_path)[0] + "_mask.jpg"
        if line_coords:
            x0, y0, x1, y1 = line_coords
            cx = int((x0 + x1) / 2)
            cy = int((y0 + y1) / 2)
        else:
            cx, cy = 0, 0
        centers.append((cx, cy))
        lengths.append(length)
        saved_frames.append((idx, frames[idx], line_coords, length))
        mark_indices.append(idx)
        last_line = line_coords
        last_length = length
        idx -= mark_step
    # 2. 反转顺序，保证时间顺序
    saved_frames = saved_frames[::-1]
    centers = centers[::-1]
    lengths = lengths[::-1]
    mark_indices = mark_indices[::-1]

    # 3. 对每一帧进行平滑插值，生成所有帧的中心点和长度（拟合补帧，关键帧不变）
    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter

    # 只用你标注的帧编号和对应的中心点/长度做线性插值，保证关键帧严格通过
    all_indices = []
    for i in range(len(saved_frames) - 1):
        idx0, _, _, _ = saved_frames[i]
        idx1, _, _, _ = saved_frames[i + 1]
        all_indices.extend(list(range(idx0, idx1 + 1)))  # 包含右端点
    all_indices = sorted(set(all_indices))

    interp_x = np.array(mark_indices)
    interp_frames = np.array(all_indices)
    centers_np = np.array(centers)
    lengths_np = np.array(lengths)

    # 匀速运动插值：以累计距离为参数做线性插值，保证物体匀速
    key_points = np.stack(centers)
    dists = np.sqrt(np.sum(np.diff(key_points, axis=0)**2, axis=1))
    cum_dist = np.concatenate([[0], np.cumsum(dists)])
    total_dist = cum_dist[-1]

    # 修正：严格按照帧索引均匀采样（即每一帧对应一个均匀的t参数），保证速度均匀
    num_frames = len(all_indices)
    t_uniform = np.linspace(0, 1, num_frames)
    t_key = (np.array(mark_indices) - all_indices[0]) / (all_indices[-1] - all_indices[0])

    interp_func_x = interp1d(t_key, key_points[:, 0], kind='linear')
    interp_func_y = interp1d(t_key, key_points[:, 1], kind='linear')
    interp_func_l = interp1d(t_key, lengths_np, kind='linear')

    cx_interp = interp_func_x(t_uniform)
    cy_interp = interp_func_y(t_uniform)
    l_interp = interp_func_l(t_uniform)

    # --- 抖动平滑处理 ---
    def sg_smooth(arr, key_idx, window=17, poly=3):
        arr = np.asarray(arr)
        n = len(arr)
        # 关键帧索引
        key_idx = np.array(key_idx)
        # 生成mask，关键帧不动
        mask = np.zeros(n, dtype=bool)
        # 找到关键帧在all_indices中的位置
        for i, idx in enumerate(np.linspace(all_indices[0], all_indices[-1], num_frames, dtype=int)):
            if idx in key_idx:
                mask[i] = True
        arr_smooth = arr.copy()
        if n > window:
            from scipy.signal import savgol_filter
            arr_sg = savgol_filter(arr, window_length=window, polyorder=poly, mode='interp')
            for i in range(n):
                if not mask[i]:
                    arr_smooth[i] = arr_sg[i]
        return arr_smooth

    cx_smooth = sg_smooth(cx_interp, mark_indices, window=17, poly=3)
    cy_smooth = sg_smooth(cy_interp, mark_indices, window=17, poly=3)
    l_smooth = sg_smooth(l_interp, mark_indices, window=17, poly=3)

    interp_centers = list(zip(cx_smooth.astype(int), cy_smooth.astype(int)))
    interp_lengths = list(l_smooth)

    # 4. 重新插入物体，所有帧都用拟合/插值结果（保证关键帧拟合点与人工标注重合，不直接用line_coords）
    mark_idx_set = set(mark_indices)
    import shutil
    manual_dir = os.path.join(output_dir, "manual_keyframes")
    os.makedirs(manual_dir, exist_ok=True)
    for idx, frame_idx in enumerate(all_indices):
        frame_name = frames[frame_idx]
        frame_path = os.path.join(frames_dir, frame_name)
        save_path = os.path.join(output_dir, frame_name)
        mask_save_path = os.path.splitext(save_path)[0] + "_mask.jpg"
        cx, cy = interp_centers[idx]
        length = interp_lengths[idx]
        insert_object_to_frame(
            frame_path,
            insert_img_path,
            length,
            save_path,
            line_coords=None,
            mask_save_path=mask_save_path,
            center=(cx, cy)
        )
        # 复制关键帧到manual_keyframes
        if frame_idx in mark_idx_set:
            shutil.copy(save_path, os.path.join(manual_dir, frame_name))
        print(f"已保存: {save_path} 及 {mask_save_path}")

    # 标注结束后生成视频（只用本次合成的图片，按标注顺序）
    video_frames = [os.path.join(output_dir, frames[i]) for i in all_indices]
    temp_dir = os.path.join(output_dir, "_video_temp")
    os.makedirs(temp_dir, exist_ok=True)
    for i, img_path in enumerate(video_frames):
        img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(temp_dir, f"{i:04d}.jpg"), img)
    save_video_from_images(temp_dir, os.path.join(output_dir, "result.mp4"), fps=30)
    # 新增：在masks文件夹里生成掩码视频
    mask_dir = os.path.join(output_dir, "masks")
    mask_video_path = os.path.join(mask_dir, "mask_result.mp4")
    save_mask_video_from_images(mask_dir, mask_video_path, fps=30)
    # 清理临时图片
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)

if __name__ == "__main__":
    main()

