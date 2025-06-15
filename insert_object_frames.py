import os
import sys
from PIL import Image, ImageTk
import tkinter as tk
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import vgg19
import subprocess

# ====== 在此处直接设置参数 ======
frames_dir = "/media/huang/NVMe/_mmlab_swjtu/data/night/bergen_night/bergen01_night"
insert_img_path = "/media/huang/NVMe/_mmlab_swjtu/data/object/stone1.png"
start_seq = 1200  # 起始序号，如0000则填0
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

def draw_reference_line(image_path):
    root = tk.Tk()
    root.title("请画一条参考线（左键画线，右键重画，回车确认）")
    img = Image.open(image_path)
    tk_img = ImageTk.PhotoImage(img)
    canvas = tk.Canvas(root, width=img.width, height=img.height)
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
    line = []

    def on_click(event):
        # 左键：画线
        if event.num == 1:
            if len(line) < 2:
                line.append((event.x, event.y))
                if len(line) == 2:
                    canvas.create_line(line[0][0], line[0][1], line[1][0], line[1][1], fill='red', width=2)
        # 右键：重画
        elif event.num == 3:
            line.clear()
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

    def on_return(event):
        root.quit()

    canvas.bind("<Button-1>", on_click)
    canvas.bind("<Button-3>", on_click)
    root.bind("<Return>", on_return)
    root.mainloop()
    try:
        root.destroy()
    except tk.TclError:
        pass
    if len(line) == 2:
        x0, y0 = line[0]
        x1, y1 = line[1]
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
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
    # 检查尺寸
    if content.shape[2] < 8 or content.shape[3] < 8 or style.shape[2] < 8 or style.shape[3] < 8:
        raise ValueError("内容图或风格图尺寸过小，跳过风格迁移")
    content_feat = extract_vgg_features(content, vgg)
    style_feat = extract_vgg_features(style, vgg)
    if content_feat.shape[1] != style_feat.shape[1]:
        raise ValueError("风格迁移特征通道不一致，跳过风格迁移")
    t = adain(content_feat, style_feat)
    # 直接反归一化回图像
    t_img = t[0].cpu()
    t_img = inv_transform(t_img)
    # 保留alpha通道
    if content_img.mode == "RGBA":
        t_img = t_img.convert("RGBA")
        t_img.putalpha(content_img.split()[-1])
    return t_img

def insert_object_to_frame(frame_path, insert_img_path, object_width, save_path, line_coords=None, mask_save_path=None):
    frame = Image.open(frame_path).convert("RGBA")
    insert_obj = Image.open(insert_img_path).convert("RGBA")
    w_percent = object_width / insert_obj.width
    new_height = int(insert_obj.height * w_percent)
    # 兼容旧PIL
    insert_obj = insert_obj.resize((int(object_width), new_height), Image.LANCZOS)
    # 让物体方向与参考线一致
    if line_coords:
        x0, y0, x1, y1 = line_coords
        import math
        angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
        insert_obj = insert_obj.rotate(-angle, expand=True, resample=Image.BICUBIC)
        cx = int((x0 + x1) / 2 - insert_obj.width / 2)
        cy = int((y0 + y1) / 2 - insert_obj.height / 2)
    else:
        cx, cy = 0, 0

    # === 风格迁移 ===
    # 取背景区域作为风格图
    bg_crop = frame.crop((max(cx,0), max(cy,0), max(cx,0)+insert_obj.width, max(cy,0)+insert_obj.height)).convert("RGB")
    insert_rgb = insert_obj.convert("RGB")
    try:
        # 仅当插入物体和背景区域都大于8x8时才做风格迁移
        if insert_rgb.width >= 8 and insert_rgb.height >= 8 and bg_crop.width >= 8 and bg_crop.height >= 8:
            stylized_obj = style_transfer(insert_rgb, bg_crop)
            # 保留alpha通道
            if insert_obj.mode == "RGBA":
                stylized_obj = stylized_obj.convert("RGBA")
                stylized_obj.putalpha(insert_obj.split()[-1])
            insert_obj = stylized_obj
        else:
            raise ValueError("风格迁移区域过小，跳过风格迁移")
    except Exception as e:
        print("风格迁移失败，使用原图:", e)

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

def images_to_video(image_folder, output_video, fps=1):
    # 假设图片命名为 *.jpg
    # 先排序，重命名为 frame_%04d.jpg 的临时软链接，便于ffmpeg顺序读取
    import glob
    import shutil
    import tempfile
    images = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    # 排除mask图
    images = [img for img in images if not img.endswith("_mask.jpg")]
    if not images:
        print("未找到可用于合成视频的图片")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, img in enumerate(images):
            link_name = os.path.join(tmpdir, f"frame_{i:04d}.jpg")
            try:
                os.symlink(os.path.abspath(img), link_name)
            except AttributeError:
                shutil.copy(img, link_name)
        cmd = [
            'ffmpeg',
            '-y',
            '-framerate', str(fps),
            '-i', os.path.join(tmpdir, 'frame_%04d.jpg'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_video
        ]
        print("正在合成视频:", output_video)
        subprocess.run(cmd, check=True)
        print("视频已保存:", output_video)

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

    # 从末尾帧向前，每隔30帧，直到用户关闭窗口
    idx = end_idx
    while idx >= 0:
        frame_path = os.path.join(frames_dir, frames[idx])
        print(f"处理帧: {frames[idx]}")
        length, line_coords = draw_reference_line(frame_path)
        if length is None:
            print("窗口关闭，停止标注")
            break
        save_path = os.path.join(output_dir, frames[idx])
        mask_save_path = os.path.splitext(save_path)[0] + "_mask.jpg"
        insert_object_to_frame(frame_path, insert_img_path, length, save_path, line_coords, mask_save_path)
        print(f"已保存: {save_path} 及 {mask_save_path}")
        idx -= 30

    # === 合成视频 ===
    output_video = os.path.join(output_dir, "output_1fps.mp4")
    images_to_video(output_dir, output_video, fps=1)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

