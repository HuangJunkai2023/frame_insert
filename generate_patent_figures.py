# -*- coding: utf-8 -*-
"""
生成专利说明书附图（黑白、适配 Word 插图）
严格依据 insert_object_frames.py 的真实流程绘制，避免元素重叠。

图1 方法流程图（纵向，逐步过程，含真实回退路径）
图2 系统结构图（模块化，标记可选/注释功能）
图3 关键帧标注界面示意图（水平约束、长度递减、操作提示）
图4 插值与平滑处理示意图（线性插值 + SciPy Savitzky-Golay，关键帧保持）
图5 批量插入与泊松融合示意图（中心对齐、ROI裁剪、失败保存原图）
图6 掩码生成与文件组织示意图（阈值10、masks目录、黑掩码补全、移动逻辑）

输出：./patent_figures_bw/fig1_*.png ... fig6_*.png（300 DPI，白底黑线，无彩色）

依赖：matplotlib, numpy
可选字体：Microsoft YaHei / SimHei / Arial Unicode MS / DejaVu Sans（自动回退）
"""
from __future__ import annotations
import os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches

# 全局渲染为黑白风格，适配 Word（白底、黑线、灰填充）
matplotlib.rcParams.update({
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.facecolor': 'white',
    'text.color': '0.0',
    'axes.labelcolor': '0.0',
    'axes.edgecolor': '0.0',
    'grid.color': '0.5',
    'lines.color': '0.0',
    'patch.edgecolor': '0.0',
    'patch.facecolor': '0.9',  # 灰填充
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'font.size': 10,
    'pdf.fonttype': 42,   # 嵌入向量字体
    'ps.fonttype': 42,
})
# 字体回退列表（优先中文字体，避免乱码）
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

# 通用尺寸（英寸），导出 PNG 300DPI 可适配 A4/Word
FIG_W, FIG_H = 8.5, 11.0  # 纵向A4近似比例，利于避免重叠
DPI = 300


def ensure_out(out_dir: str) -> str:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    return out_dir


def vstack_boxes(ax, labels, x=0.08, top=0.93, bottom=0.07, box_width=0.84, min_gap=0.02, max_height=0.12):
    """
    在 [bottom, top] 范围内垂直均匀放置若干个同宽盒子，返回（boxes, positions）。
    自动计算合适的高度与间距，避免重叠。
    """
    n = len(labels)
    avail = top - bottom
    # 先给定预期gap，再求能放下的高度
    gap = max(min_gap, avail / (n * 6))  # 至少 min_gap，随数量自适应
    H = min(max_height, (avail - gap * (n - 1)) / n)
    # 如果仍放不下，进一步缩小高度
    if H <= 0:
        H = max(0.05, avail / (n + (n - 1)))
        gap = H
    ys = []
    y = top
    boxes = []
    for i, text in enumerate(labels):
        y_box = y - H
        b = draw_box(ax, (x, y_box), box_width, H, text)
        boxes.append(b)
        ys.append(y_box)
        y = y_box - gap
        if y < bottom:
            # 超界，提前终止
            break
    return boxes, ys, H, gap


def draw_box(ax, xy, w, h, text, fc=0.97, ec='0.0', lw=1.2, rounded=True, align_center=True):
    x, y = xy
    box = patches.FancyBboxPatch((x, y), w, h,
                                 boxstyle=patches.BoxStyle("Round", pad=0.02) if rounded else 'square',
                                 linewidth=lw, edgecolor=ec, facecolor=str(fc))
    ax.add_patch(box)
    if align_center:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', wrap=True)
    else:
        ax.text(x + 0.04*w, y + 0.6*h, text, ha='left', va='center', wrap=True)
    return box


def draw_arrow(ax, start, end, text=None, style='-|>', lw=1.2, mutation_scale=12, ls='-'):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, lw=lw, color='0.0', shrinkA=0, shrinkB=0, linestyle=ls))
    if text:
        mx, my = (start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0
        ax.text(mx, my + 0.02, text, ha='center', va='bottom')


# 图1 方法流程示意图

def fig1_method_flow(out_path: str):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 纵向流程，自动布局，文本手动换行
    x = 0.08
    W = 0.84
    labels = [
        '输入帧序列（.jpg 排序）',
        '起始帧选择界面：\n方向键±5移动，回车确认',
        '关键帧标注：\n水平约束、长度递减；\n左键绘制/右键重画/回车确认/ESC终止',
        '线性插值 + Savitzky-Golay 平滑：\n窗口17、阶3；关键帧保持不变',
        '批量插入：\n按插值中心对齐、基于宽度缩放；\n关键帧可基于线段旋转（批量不旋转）',
        'ROI边界检查与裁剪：\ncv2.seamlessClone(NORMAL_CLONE)；\n失败→保存原图',
        '掩码：\nalpha>10 阈值，保存至 output_dir/masks',
        '视频编码：\nresult.mp4 / mask_result.mp4（30fps）；\n临时帧按顺序输出',
        '备份/替换：\n备份原始帧→仅替换被修改帧；\n掩码文件夹移动/覆盖并补全黑掩码',
    ]

    boxes, ys, H, gap = vstack_boxes(ax, labels, x=x, top=0.93, bottom=0.07, box_width=W, min_gap=0.02, max_height=0.11)
    # 箭头：从上一个盒子底部 -> 下一个盒子顶部
    for i in range(len(boxes)-1):
        x_mid = x + W/2
        y_bottom_prev = ys[i]
        y_top_next = ys[i+1] + H
        draw_arrow(ax, (x_mid, y_bottom_prev), (x_mid, y_top_next))

    ax.set_title('图1 方法流程示意图（依据 insert_object_frames.py）', pad=10)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# 图2 系统结构示意图

def fig2_system_arch(out_path: str):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 纵向主干模块（自动布局，明确换行）
    x = 0.15
    W = 0.70
    names = [
        '输入/浏览模块：\n帧排序；起始帧选择UI',
        '关键帧标注模块：\n水平约束、长度递减、交互提示',
        '插值与平滑模块：\n线性插值 + savgol_filter；关键帧保持',
        '插入与融合模块：\n中心对齐缩放；ROI裁剪；seamlessClone；失败保存原图',
        '掩码生成模块：\nalpha阈值10→二值化；保存至 masks',
        '视频编码模块：\nresult.mp4 / mask_result.mp4（30fps）',
        '备份与替换模块：\n备份原始帧；只替换被修改帧；掩码移动/覆盖与补全',
    ]
    boxes, ys, H, gap = vstack_boxes(ax, names, x=x, top=0.90, bottom=0.10, box_width=W, min_gap=0.02, max_height=0.10)
    for i in range(len(boxes)-1):
        draw_arrow(ax, (x+W/2, ys[i]), (x+W/2, ys[i+1]+H))

    # 可选：风格迁移（默认注释）放在最上方
    opt_box = draw_box(ax, (x, 0.94), W, 0.05, '可选：风格迁移（VGG19 + AdaIN，代码默认注释）', fc=0.98)
    draw_arrow(ax, (x+W/2, 0.94), (x+W/2, ys[0]+H))

    ax.set_title('图2 系统结构示意图（依据 insert_object_frames.py）', pad=10)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# 图3 关键帧标注界面示意图

def fig3_annotation_ui(out_path: str):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 画一个“视频帧”画布
    frame = patches.Rectangle((0.08, 0.35), 0.84, 0.45, linewidth=1.2, edgecolor='0.0', facecolor='1.0')
    ax.add_patch(frame)
    # 将“视频帧预览”文字放入框内，减少上方留白
    ax.text(0.5, 0.78, '视频帧预览', ha='center', va='center')

    # 历史参考线（虚线）
    for y in [0.58, 0.55]:
        ax.plot([0.15, 0.6], [y, y], linestyle='--', color='0.0', lw=1.2)

    # 当前参考线（实线、较粗）
    ax.plot([0.2, 0.75], [0.48, 0.48], linestyle='-', color='0.0', lw=2.2)
    ax.scatter([0.2, 0.75], [0.48, 0.48], color='0.0', s=20)
    # 将“水平参考线”说明移动到横线下方并居中，避免与线条同一水平位置
    ax.text((0.2+0.75)/2, 0.465, '水平参考线（第二点y = 第一点y）', va='top', ha='center', wrap=True)

    # 长度递减约束警示框
    tri = patches.RegularPolygon((0.12, 0.73), numVertices=3, radius=0.015, orientation=math.pi,
                                 edgecolor='0.0', facecolor='0.9')
    ax.add_patch(tri)
    ax.text(0.15, 0.73, '约束：长度需比上次更短；违规会清空并提示“必须比上一次短，请重画！”', va='center', ha='left', wrap=True)

    # 操作提示栏
    tip = (
        '操作：左键绘制，右键重画，回车确认，ESC终止并进入合成；方向键用于选择起始帧。\n'
        '视觉反馈：当前线为红色实线，历史线为绿色虚线，违规显示红色警告文本。' 
    )
    # 将操作提示注释上移，靠近画布以减少留白
    ax.text(0.08, 0.29, tip, ha='left', va='center', wrap=True)

    ax.set_title('图3 关键帧标注界面示意图（依据 insert_object_frames.py）', pad=5)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# 图4 插值与平滑处理示意图

def fig4_interpolation_smoothing(out_path: str, n_frames: int = 120):
    # 构造关键帧与插值/平滑示例
    np.random.seed(0)
    k_idx = np.array([0, 20, 50, 80, 119])
    xk = np.array([100, 180, 260, 300, 360], dtype=float)
    yk = np.array([200, 210, 205, 198, 196], dtype=float)
    lk = np.array([120, 100, 80, 60, 50], dtype=float)

    t = np.arange(n_frames)
    # 线性插值
    x_lin = np.interp(t, k_idx, xk)
    y_lin = np.interp(t, k_idx, yk)
    l_lin = np.interp(t, k_idx, lk)

    # Savitzky-Golay 简易近似（使用卷积核近似，避免依赖scipy）
    # 这里给出一个长度17的平滑核（近似），并固定关键帧位置（示意与代码一致的意图）
    W = 17
    base = np.hanning(W)  # 黑白下足够示意
    base = base / base.sum()
    def smooth_keep_keys(arr, keys):
        s = np.convolve(arr, base, mode='same')
        s[keys] = arr[keys]
        return s

    x_s = smooth_keep_keys(x_lin, k_idx)
    y_s = smooth_keep_keys(y_lin, k_idx)
    l_s = smooth_keep_keys(l_lin, k_idx)

    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, FIG_H), dpi=DPI, sharex=True, constrained_layout=True)
    for ax, v_lin, v_s, vk, name in [
        (axes[0], x_lin, x_s, xk, 'x 位置'),
        (axes[1], y_lin, y_s, yk, 'y 位置'),
        (axes[2], l_lin, l_s, lk, '长度/像素宽度'),
    ]:
        ax.plot(t, v_lin, linestyle='--', color='0.0', lw=1.2, label='线性插值')
        ax.plot(t, v_s, linestyle='-', color='0.0', lw=2.0, label='Savitzky-Golay平滑（窗口17、阶3）')
        ax.plot(k_idx, vk, linestyle='None', marker='s', color='0.0', label='关键帧')
        ax.set_ylabel(name)
        ax.grid(True, linestyle=':', color='0.7')
        ax.legend(loc='best', frameon=False)

    axes[-1].set_xlabel('帧索引')
    fig.suptitle('图4 插值与平滑处理示意图（依据 insert_object_frames.py）', y=0.99)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# 图5 目标插入与泊松融合示意图

def fig5_poisson_fusion(out_path: str):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 左：背景帧
    bg = patches.Rectangle((0.06, 0.25), 0.28, 0.5, linewidth=1.2, edgecolor='0.0', facecolor='1.0')
    ax.add_patch(bg)
    ax.text(0.2, 0.78, '背景帧 ROI', ha='center', va='center')

    # 中：前景对象（含alpha边缘/羽化）
    fg = patches.Rectangle((0.39, 0.40), 0.22, 0.2, linewidth=1.2, edgecolor='0.0', facecolor='0.95')
    ax.add_patch(fg)
    # 画出羽化边框（多重灰度边）
    for i, g in enumerate([0.85, 0.9, 0.95]):
        r = patches.Rectangle((0.39 - 0.01*i, 0.40 - 0.01*i), 0.22 + 0.02*i, 0.2 + 0.02*i,
                              linewidth=0.8, edgecolor=str(0.2+0.2*i), facecolor='none', linestyle=':')
        ax.add_patch(r)
    ax.text(0.5, 0.64, '前景对象（按插值宽度缩放；关键帧可旋转；批量中心对齐）', ha='center', va='center', wrap=True)

    # 右：融合结果
    out = patches.Rectangle((0.72, 0.25), 0.28, 0.5, linewidth=1.2, edgecolor='0.0', facecolor='1.0')
    ax.add_patch(out)
    ax.text(0.86, 0.78, '融合结果', ha='center', va='center')

    # 矩形内放置对象轮廓，示意已融合
    fused = patches.Rectangle((0.78, 0.40), 0.16, 0.2, linewidth=1.2, edgecolor='0.0', facecolor='0.92')
    ax.add_patch(fused)

    # 箭头与说明
    draw_arrow(ax, (0.34, 0.50), (0.39, 0.50))
    draw_arrow(ax, (0.61, 0.50), (0.72, 0.50))

    ax.text(0.5, 0.23, '处理顺序：ROI裁剪 → seamlessClone(NORMAL_CLONE)；尺寸不符或异常 → 保存原始帧',
            ha='center', va='center', wrap=True)

    ax.set_title('图5 插入与泊松融合示意图（依据 insert_object_frames.py）', pad=10)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# 图6 掩码生成流程示意图

def fig6_mask_generation(out_path: str):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 三联图：Alpha通道 -> 阈值化 -> 二值掩码
    W, H = 0.25, 0.44
    x0 = 0.08

    # Alpha 渐变块
    grad = np.tile(np.linspace(0.2, 0.95, 200), (120, 1))
    ax.imshow(grad, extent=(x0, x0+W, 0.30, 0.74), cmap='gray', vmin=0, vmax=1, origin='lower')
    ax.add_patch(patches.Rectangle((x0, 0.30), W, H, fill=False, lw=1.2, edgecolor='0.0'))
    ax.text(x0 + W/2, 0.77, 'Alpha 通道', ha='center', va='center')

    # 阈值化标记
    thr_x = x0 + W/2
    ax.plot([thr_x, thr_x], [0.30, 0.74], linestyle='--', color='0.0', lw=1.2)
    ax.text(thr_x, 0.27, '阈值=10（alpha>10 → 255，否则0）', ha='center', va='top')

    # 箭头
    draw_arrow(ax, (x0+W, 0.52), (x0+W+0.06, 0.52))

    # 阈值结果（灰→两色分区）
    # 左半深灰，右半浅灰，表示被分割
    x1 = x0 + W + 0.08
    ax.add_patch(patches.Rectangle((x1, 0.30), W/2, H, facecolor='0.8', edgecolor='0.0', lw=1.2))
    ax.add_patch(patches.Rectangle((x1+W/2, 0.30), W/2, H, facecolor='0.95', edgecolor='0.0', lw=1.2))
    ax.text(x1 + W/2, 0.77, '阈值化', ha='center', va='center')

    draw_arrow(ax, (x1+W, 0.52), (x1+W+0.06, 0.52))

    # 二值掩码：黑白
    x2 = x1 + W + 0.08
    ax.add_patch(patches.Rectangle((x2, 0.30), W/2, H, facecolor='0.0', edgecolor='0.0', lw=1.2))
    ax.add_patch(patches.Rectangle((x2+W/2, 0.30), W/2, H, facecolor='1.0', edgecolor='0.0', lw=1.2))
    ax.text(x2 + W/2, 0.77, '二值掩码', ha='center', va='center')

    # 下方：文件组织与移动逻辑
    yb = 0.15
    draw_box(ax, (0.08, yb), 0.84, 0.06,
             '保存：output_dir/masks/<帧名>_mask.jpg；生成 mask_result.mp4\n若同级掩码文件夹已存在：仅覆盖本轮修改帧的掩码；否则移动并补全未修改帧为黑掩码', fc=0.98)

    ax.set_title('图6 掩码生成与文件组织示意图（依据 insert_object_frames.py）', pad=10)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def main(out_dir: str = 'patent_figures_bw'):
    out = ensure_out(out_dir)
    fig1_method_flow(os.path.join(out, 'fig1_method_flow.png'))
    fig2_system_arch(os.path.join(out, 'fig2_system_arch.png'))
    fig3_annotation_ui(os.path.join(out, 'fig3_annotation_ui.png'))
    fig4_interpolation_smoothing(os.path.join(out, 'fig4_interpolation_smoothing.png'))
    fig5_poisson_fusion(os.path.join(out, 'fig5_poisson_fusion.png'))
    fig6_mask_generation(os.path.join(out, 'fig6_mask_generation.png'))
    print(f'专利附图已生成：{os.path.abspath(out)}')


if __name__ == '__main__':
    main()
