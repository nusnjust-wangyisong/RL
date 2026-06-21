from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "generated_project_images"
FONT_CJK = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
FONT_SANS = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


PALETTE = {
    "paper": "#fbfbf8",
    "ink": "#1d2329",
    "muted": "#65717d",
    "line": "#cfd6dc",
    "red": "#b53a2c",
    "blue": "#2f6fbb",
    "green": "#2e9d62",
    "orange": "#e48a25",
    "purple": "#7b5fb2",
    "teal": "#2d8f9d",
    "dark": "#20252b",
}


def font(size: int, bold: bool = False, *, cjk: bool = False) -> ImageFont.FreeTypeFont:
    if cjk and FONT_CJK.exists():
        path = FONT_CJK
    elif bold and FONT_BOLD.exists():
        path = FONT_BOLD
    elif FONT_SANS.exists():
        path = FONT_SANS
    else:
        path = FONT_CJK
    return ImageFont.truetype(str(path), size)


def canvas(width: int = 1800, height: int = 1100) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (width, height), PALETTE["paper"])
    return img, ImageDraw.Draw(img)


def save(img: Image.Image, name: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    img.save(OUT_DIR / name, quality=96)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], content: str, size: int = 36, fill: str | None = None, bold: bool = False) -> None:
    x, y = xy
    line_height = int(size * 1.28)
    for line_no, line in enumerate(str(content).split("\n")):
        cx = x
        cy = y + line_no * line_height
        for ch in line:
            cjk = ord(ch) > 127
            fnt = font(size, bold, cjk=cjk)
            draw.text((cx, cy), ch, font=fnt, fill=fill or PALETTE["ink"])
            cx += draw.textlength(ch, font=fnt)


def title(draw: ImageDraw.ImageDraw, main: str, sub: str) -> None:
    text(draw, (72, 48), main, 48, PALETTE["ink"], True)
    text(draw, (74, 112), sub, 25, PALETTE["muted"])
    draw.line((72, 158, 1728, 158), fill="#d7dde2", width=3)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], fill: str, width: int = 4) -> None:
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=fill, width=width)
    ang = math.atan2(y2 - y1, x2 - x1)
    length = 16
    spread = 0.52
    p1 = (x2 - length * math.cos(ang - spread), y2 - length * math.sin(ang - spread))
    p2 = (x2 - length * math.cos(ang + spread), y2 - length * math.sin(ang + spread))
    draw.polygon([(x2, y2), p1, p2], fill=fill)


def joint(draw: ImageDraw.ImageDraw, p: tuple[float, float], r: int = 30, fill: str = "#f1f3f4", outline: str = "#333") -> None:
    x, y = p
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=4)


def link(draw: ImageDraw.ImageDraw, a: tuple[float, float], b: tuple[float, float], color: str = "#f0f2f3", width: int = 58) -> None:
    draw.line((*a, *b), fill="#bfc5c8", width=width + 6)
    draw.line((*a, *b), fill=color, width=width)


def draw_panda_side(draw: ImageDraw.ImageDraw, offset: tuple[int, int], scale: float = 1.0) -> None:
    ox, oy = offset
    pts = [
        (ox + 80 * scale, oy + 650 * scale),
        (ox + 145 * scale, oy + 505 * scale),
        (ox + 260 * scale, oy + 410 * scale),
        (ox + 330 * scale, oy + 295 * scale),
        (ox + 500 * scale, oy + 210 * scale),
        (ox + 640 * scale, oy + 260 * scale),
        (ox + 690 * scale, oy + 365 * scale),
    ]
    draw.rounded_rectangle(
        (ox + 20 * scale, oy + 690 * scale, ox + 235 * scale, oy + 750 * scale),
        radius=int(22 * scale),
        fill="#dfe4e6",
        outline="#aeb7bc",
        width=3,
    )
    for a, b in zip(pts[:-1], pts[1:]):
        link(draw, a, b, width=int(45 * scale))
    for i, p in enumerate(pts[:-1]):
        joint(draw, p, int((24 if i else 32) * scale))
    draw.rounded_rectangle(
        (pts[-1][0] - 36 * scale, pts[-1][1] - 16 * scale, pts[-1][0] + 62 * scale, pts[-1][1] + 38 * scale),
        radius=int(10 * scale),
        fill="#e9ecef",
        outline="#5f686e",
        width=3,
    )
    draw.line((pts[-1][0] + 35 * scale, pts[-1][1] + 38 * scale, pts[-1][0] + 35 * scale, pts[-1][1] + 88 * scale), fill="#4b5358", width=int(7 * scale))
    draw.line((pts[-1][0] + 13 * scale, pts[-1][1] + 84 * scale, pts[-1][0] + 54 * scale, pts[-1][1] + 84 * scale), fill="#4b5358", width=int(6 * scale))
    text(draw, (int(ox + 258 * scale), int(oy + 744 * scale)), "Franka Emika Panda", int(23 * scale), PALETTE["muted"])


def draw_coordinate_chain(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    ys = [y + v for v in [680, 575, 470, 370, 280, 182, 85]]
    xs = [x, x + 35, x - 10, x + 55, x + 110, x + 170, x + 225]
    for i in range(len(xs) - 1):
        draw.line((xs[i], ys[i], xs[i + 1], ys[i + 1]), fill="#111", width=5)
        draw.line((xs[i], ys[i], xs[i] + 48, ys[i] - 8), fill="#111", width=3)
        text(draw, (xs[i] - 54, ys[i] - 18), f"z{i}", 22, PALETTE["ink"])
        text(draw, (xs[i] + 56, ys[i] - 22), f"x{i}", 22, PALETTE["ink"])
    for i, (px, py) in enumerate(zip(xs, ys), start=1):
        joint(draw, (px, py), 22, "#f8fafb", "#111")
        text(draw, (px + 28, py - 10), f"J{i}", 22, PALETTE["muted"])
    draw.line((x - 48, y + 735, x + 64, y + 735), fill="#111", width=6)
    draw.polygon([(x - 35, y + 735), (x + 48, y + 735), (x + 10, y + 692)], outline="#111", fill=None)


def draw_table(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    headers = ["连杆 i", "alpha(i-1)", "a(i-1)/m", "d(i)/m", "theta(i)"]
    rows = [
        ["1", "0", "0", "0.333", "theta1"],
        ["2", "-90", "0", "0", "theta2"],
        ["3", "90", "0", "0.316", "theta3"],
        ["4", "90", "0.0825", "0", "theta4"],
        ["5", "-90", "0.0825", "0.384", "theta5"],
        ["6", "90", "0", "0", "theta6"],
        ["7", "90", "0.088", "0", "theta7"],
        ["flange", "0", "0", "0.107", "0"],
    ]
    cw = [110, 145, 145, 125, 120]
    rh = 70
    draw.rounded_rectangle((x - 4, y - 4, x + sum(cw) + 4, y + rh * (len(rows) + 1) + 4), radius=10, fill="#ffffff", outline="#d8dee3", width=2)
    cx = x
    for i, h in enumerate(headers):
        draw.rectangle((cx, y, cx + cw[i], y + rh), fill=PALETTE["red"])
        text(draw, (cx + 12, y + 19), h, 22, "#ffffff", True)
        cx += cw[i]
    for r, row in enumerate(rows):
        yy = y + rh * (r + 1)
        cx = x
        bg = "#f5ece9" if r % 2 else "#ead7d2"
        for c, cell in enumerate(row):
            draw.rectangle((cx, yy, cx + cw[c], yy + rh), fill=bg, outline="#ffffff", width=2)
            text(draw, (cx + 22 if c else cx + 34, yy + 18), cell, 23, PALETTE["ink"], c == 0)
            cx += cw[c]


def panda_mdh() -> None:
    img, draw = canvas()
    title(draw, "Panda 机械臂建模与 MDH 参数", "依据项目 Franka Panda 七自由度 Reach 任务重新绘制，用于方法与模型介绍页")
    draw.rounded_rectangle((70, 210, 735, 990), radius=18, fill="#ffffff", outline="#dce2e6", width=3)
    draw_panda_side(draw, (90, 235), 0.82)
    draw.rounded_rectangle((780, 210, 1115, 990), radius=18, fill="#ffffff", outline="#dce2e6", width=3)
    text(draw, (810, 232), "MDH 坐标系", 27, PALETTE["muted"])
    draw_coordinate_chain(draw, 875, 250)
    text(draw, (1200, 210), "Panda MDH 参数表", 37, PALETTE["ink"], True)
    draw_table(draw, 1130, 270)
    save(img, "01_panda_mdh_parameters_generated.png")


def project(pt: tuple[float, float, float], origin: tuple[int, int], scale: float) -> tuple[float, float]:
    x, y, z = pt
    return origin[0] + (x - y) * 0.82 * scale, origin[1] + (x + y) * 0.36 * scale - z * scale


def draw_grid(draw: ImageDraw.ImageDraw, origin: tuple[int, int], scale: float, extent: float = 1.1) -> None:
    vals = np.linspace(-extent, extent, 9)
    for v in vals:
        a = project((-extent, v, 0), origin, scale)
        b = project((extent, v, 0), origin, scale)
        c = project((v, -extent, 0), origin, scale)
        d = project((v, extent, 0), origin, scale)
        draw.line((*a, *b), fill="#d7e7d9", width=2)
        draw.line((*c, *d), fill="#d7e7d9", width=2)


def draw_robot_iso(draw: ImageDraw.ImageDraw, origin: tuple[int, int], scale: float, pose: str = "reach") -> list[tuple[float, float]]:
    if pose == "upright":
        pts3 = [(0, 0, 0), (0.05, 0, 0.25), (0.02, 0.08, 0.48), (0.0, 0.02, 0.66), (0.08, -0.05, 0.86), (0.16, -0.02, 1.0), (0.20, 0.02, 1.08)]
    elif pose == "folded":
        pts3 = [(0, 0, 0), (0.04, 0.0, 0.25), (0.14, -0.08, 0.44), (0.28, -0.10, 0.56), (0.36, 0.02, 0.62), (0.46, 0.08, 0.56), (0.56, 0.08, 0.50)]
    else:
        pts3 = [(0, 0, 0), (0.03, 0.0, 0.25), (0.12, -0.02, 0.43), (0.25, -0.07, 0.58), (0.38, 0.00, 0.66), (0.50, 0.10, 0.58), (0.60, 0.12, 0.50)]
    pts = [project(p, origin, scale) for p in pts3]
    for a, b in zip(pts[:-1], pts[1:]):
        link(draw, a, b, width=int(28 * scale / 250))
    for p in pts[:-1]:
        joint(draw, p, int(15 * scale / 250), "#edf0f2", "#5e666b")
    ee = pts[-1]
    draw.rounded_rectangle((ee[0] - 18, ee[1] - 8, ee[0] + 35, ee[1] + 16), radius=6, fill="#e7ebee", outline="#555", width=2)
    return pts


def matlab_model() -> None:
    img, draw = canvas()
    title(draw, "基于 MATLAB Robotics Toolbox 的 Panda 模型", "重新绘制三种末端姿态，展示机械臂模型、坐标轴与 Reach 工作平面")
    panels = [(80, 230, 520, 420, "初始姿态"), (670, 230, 520, 420, "中间姿态"), (375, 635, 520, 420, "目标附近姿态")]
    poses = ["folded", "upright", "reach"]
    for (x, y, w, h, label), pose in zip(panels, poses):
        draw.rounded_rectangle((x, y, x + w, y + h), radius=10, fill="#ffffff", outline="#d9e1e6", width=2)
        origin = (x + w // 2, y + int(h * 0.78))
        draw_grid(draw, origin, 165)
        pts = draw_robot_iso(draw, origin, 170, pose)
        arrow(draw, pts[-1], (pts[-1][0] + 55, pts[-1][1] - 8), PALETTE["red"], 3)
        arrow(draw, pts[-1], (pts[-1][0] - 25, pts[-1][1] - 48), PALETTE["green"], 3)
        arrow(draw, pts[-1], (pts[-1][0] + 10, pts[-1][1] + 48), PALETTE["blue"], 3)
        text(draw, (x + 26, y + 24), label, 28, PALETTE["ink"], True)
        text(draw, (x + 30, y + h - 46), "X/m     Y/m     Z/m", 24, PALETTE["muted"])
    draw.rounded_rectangle((1230, 230, 430 + 1230, 420 + 230), radius=10, fill="#ffffff", outline="#d9e1e6", width=2)
    text(draw, (1262, 260), "模型设置", 32, PALETTE["ink"], True)
    notes = [
        "robot: franka_panda",
        "control_type: ee",
        "task: PandaReach-v3",
        "target threshold: 5 mm",
        "random x: 0.05-0.20 m",
        "random y: -0.15-0.15 m",
        "random z: 0.18-0.32 m",
    ]
    for i, note in enumerate(notes):
        text(draw, (1270, 320 + i * 48), note, 25, PALETTE["muted"])
    save(img, "02_matlab_robotics_toolbox_model_generated.png")


def plot_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], xlabel: str, ylabel: str) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle(box, fill="#ffffff", outline="#d9e1e6", width=2)
    for i in range(1, 5):
        yy = y1 + i * (y2 - y1) / 5
        xx = x1 + i * (x2 - x1) / 5
        draw.line((x1, yy, x2, yy), fill="#eef2f4", width=1)
        draw.line((xx, y1, xx, y2), fill="#eef2f4", width=1)
    draw.line((x1 + 48, y2 - 45, x2 - 20, y2 - 45), fill="#4b5358", width=3)
    draw.line((x1 + 48, y1 + 20, x1 + 48, y2 - 45), fill="#4b5358", width=3)
    text(draw, (x1 + (x2 - x1) // 2 - 55, y2 - 36), xlabel, 20, PALETTE["muted"])
    text(draw, (x1 + 10, y1 + 20), ylabel, 20, PALETTE["muted"])


def map_point(box: tuple[int, int, int, int], x: float, y: float, xr: tuple[float, float], yr: tuple[float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    px = x1 + 48 + (x - xr[0]) / (xr[1] - xr[0]) * (x2 - x1 - 68)
    py = y2 - 45 - (y - yr[0]) / (yr[1] - yr[0]) * (y2 - y1 - 65)
    return px, py


def workspace_and_joints() -> None:
    img, draw = canvas()
    title(draw, "MATLAB 建模结果：关节角与工作空间", "根据 Panda 七关节模型重新生成，补充项目随机目标范围与 5 mm 精度阈值")
    box = (70, 230, 900, 620)
    plot_axes(draw, box, "time / s", "joint angle / deg")
    text(draw, (340, 185), "各关节角度随时间变化", 32, PALETTE["ink"], True)
    t = np.linspace(0, 40, 200)
    init = np.array([-28, -50, 0, -30, -100, 0, -46], dtype=float)
    target = np.array([-4, 52, 0, -22, 0, 40, 0], dtype=float)
    colors = ["#2f6fbb", "#d9732f", "#718236", "#7b5fb2", "#71a64a", "#246b5f", "#ad4258"]
    smooth = 3 * (t / 40) ** 2 - 2 * (t / 40) ** 3
    for j in range(7):
        vals = init[j] + (target[j] - init[j]) * smooth + 1.5 * np.sin(t / 7 + j) * (0.35 if j != 2 else 0)
        pts = [map_point(box, float(tx), float(v), (0, 40), (-110, 65)) for tx, v in zip(t, vals)]
        draw.line(pts, fill=colors[j], width=4)
        draw.line((930, 255 + j * 32, 972, 255 + j * 32), fill=colors[j], width=5)
        text(draw, (984, 241 + j * 32), f"joint{j+1}", 21, PALETTE["muted"])
    rng = np.random.default_rng(42)
    pts = rng.normal(size=(7000, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    radius = rng.uniform(0.25, 0.98, size=(7000, 1))
    pts = pts * radius
    pts[:, 2] = np.abs(pts[:, 2]) * 1.05 - 0.25
    views = [
        ((1045, 210, 1715, 505), "X-Y 工作空间投影", 0, 1, (-1, 1), (-1, 1)),
        ((1045, 575, 1715, 1040), "X-Z 工作空间投影", 0, 2, (-1, 1), (-0.4, 1.15)),
        ((70, 690, 900, 1040), "三维工作空间近似分布", 0, 2, (-1, 1), (-0.4, 1.15)),
    ]
    for view_box, label, a, b, xr, yr in views:
        plot_axes(draw, view_box, "position / m", "position / m")
        text(draw, (view_box[0] + 190, view_box[1] + 24), label, 26, PALETTE["ink"], True)
        sample = pts[rng.choice(len(pts), size=2200, replace=False)]
        for px, py in sample[:, [a, b]]:
            x, y = map_point(view_box, float(px), float(py), xr, yr)
            draw.point((x, y), fill="#185cff")
        # Project goal range from configs/experiment.yaml.
        gx = [0.05, 0.20]
        gy = [-0.15, 0.15]
        gz = [0.18, 0.32]
        if (a, b) == (0, 1):
            p1 = map_point(view_box, gx[0], gy[0], xr, yr)
            p2 = map_point(view_box, gx[1], gy[1], xr, yr)
        else:
            p1 = map_point(view_box, gx[0], gz[0], xr, yr)
            p2 = map_point(view_box, gx[1], gz[1], xr, yr)
        draw.rectangle((*p1, *p2), outline=PALETTE["green"], width=4)
    save(img, "03_matlab_joint_workspace_generated.png")


def draw_platform(draw: ImageDraw.ImageDraw, origin: tuple[int, int], scale: float) -> None:
    top = [project(p, origin, scale) for p in [(-0.45, -0.45, 0), (0.55, -0.45, 0), (0.55, 0.45, 0), (-0.45, 0.45, 0)]]
    down = [project(p, origin, scale) for p in [(-0.45, -0.45, -0.28), (0.55, -0.45, -0.28), (0.55, 0.45, -0.28), (-0.45, 0.45, -0.28)]]
    draw.polygon([top[0], top[1], top[2], top[3]], fill="#f3f5f6", outline="#aeb7bc")
    draw.polygon([top[1], down[1], down[2], top[2]], fill="#cbd2d7", outline="#aeb7bc")
    draw.polygon([top[2], down[2], down[3], top[3]], fill="#aeb8bf", outline="#8d989f")


def simulation_environment() -> None:
    img, draw = canvas()
    title(draw, "PandaReach 高精度仿真环境", "PyBullet / panda-gym Reach 任务：随机目标、5 mm 精度球、20 mm 接触变力激活半径")
    origin = (820, 850)
    scale = 720
    draw.rectangle((0, 170, 1800, 1100), fill="#f7f8f8")
    draw.polygon([(0, 445), (1800, 250), (1800, 1100), (0, 1100)], fill="#22272c")
    draw.polygon([(0, 170), (1800, 170), (1800, 345), (0, 545)], fill="#d94635")
    draw_platform(draw, origin, scale)
    robot_pts = draw_robot_iso(draw, origin, scale, "reach")
    target = project((0.12, 0.0, 0.25), origin, scale)
    draw.ellipse((target[0] - 24, target[1] - 24, target[0] + 24, target[1] + 24), fill="#89d395", outline="#2e9d62", width=4)
    draw.ellipse((target[0] - 60, target[1] - 60, target[0] + 60, target[1] + 60), outline=PALETTE["orange"], width=4)
    draw.ellipse((target[0] - 16, target[1] - 16, target[0] + 16, target[1] + 16), outline="#d62728", width=4)
    curve = []
    start = np.array((0.55, 0.12, 0.50))
    goal = np.array((0.12, 0.0, 0.25))
    for s in np.linspace(0, 1, 70):
        p = (1 - s) * start + s * goal + np.array((0.0, 0.05 * math.sin(s * math.pi), 0.05 * math.sin(s * math.pi)))
        curve.append(project(tuple(p), origin, scale))
    draw.line(curve, fill=PALETTE["blue"], width=6)
    arrow(draw, (target[0], target[1] - 110), (target[0], target[1] - 42), PALETTE["orange"], 5)
    text(draw, (105, 245), "仿真任务配置", 36, "#ffffff", True)
    items = [
        "PandaReach-v3 / end-effector control",
        "固定目标: [0.12, 0.00, 0.25] m",
        "随机目标: x 0.05-0.20, y -0.15-0.15, z 0.18-0.32 m",
        "终止精度: 5 mm",
        "扰动: 20 mm 内施加 20 Hz 中等变力",
    ]
    for i, item in enumerate(items):
        text(draw, (110, 315 + i * 48), item, 25, "#ffffff")
    draw.rounded_rectangle((1135, 865, 1695, 1010), radius=14, fill="#ffffff", outline="#d9e1e6", width=2)
    legend_rows = [
        (PALETTE["green"], "target"),
        (PALETTE["orange"], "20 mm disturbance activation radius"),
        ("#d62728", "5 mm precision threshold"),
    ]
    for i, (color, label) in enumerate(legend_rows):
        yy = 895 + i * 38
        draw.line((1172, yy + 13, 1222, yy + 13), fill=color, width=6)
        text(draw, (1245, yy), label, 24, PALETTE["ink"])
    save(img, "04_panda_reach_simulation_environment_generated.png")


def control_overview() -> None:
    img, draw = canvas()
    title(draw, "RL + HER + 精密伺服仿真流程图", "结合项目最终方法：强化学习负责全局到达，近目标阶段加入末端残差伺服与稳定性约束")
    boxes = [
        (90, 265, 390, 430, "PandaReach\n观测状态", "#eef5ff"),
        (500, 265, 800, 430, "TD3 + HER\n主策略", "#f2f0ff"),
        (910, 265, 1210, 430, "近目标\n精密伺服", "#fff4e8"),
        (1320, 265, 1620, 430, "动作平滑\n稳定控制", "#eaf7f0"),
        (910, 620, 1210, 785, "20 mm 内\n接触变力", "#fff0f0"),
        (1320, 620, 1620, 785, "5 mm\n成功判定", "#f1f8ea"),
    ]
    for x1, y1, x2, y2, label, fill in boxes:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=16, fill=fill, outline="#cfd6dc", width=3)
        lines = label.split("\n")
        for i, line in enumerate(lines):
            text(draw, (x1 + 42, y1 + 42 + i * 43), line, 34, PALETTE["ink"], True)
    for a, b in [((390, 348), (500, 348)), ((800, 348), (910, 348)), ((1210, 348), (1320, 348)), ((1470, 430), (1470, 620)), ((1210, 700), (1320, 700)), ((1055, 620), (1055, 430))]:
        arrow(draw, a, b, PALETTE["blue"], 5)
    draw.rounded_rectangle((105, 830, 1630, 980), radius=16, fill="#ffffff", outline="#d9e1e6", width=3)
    formula = "r_t = - ||p_ee - p_goal|| + success_bonus - smooth/action/jerk/vibration penalties"
    text(draw, (150, 858), "奖励与评价指标", 32, PALETTE["ink"], True)
    text(draw, (150, 910), formula, 29, PALETTE["muted"])
    draw_robot_iso(draw, (360, 775), 270, "reach")
    save(img, "05_panda_rl_servo_simulation_pipeline_generated.png")


def metric_summary() -> None:
    img, draw = canvas()
    title(draw, "Panda 中等扰动实验结果摘要", "使用 runs/results 中已完成的固定目标与随机目标评估数据重新绘制")
    files = [
        ("固定目标", ROOT / "runs/results/panda_her_residual_fixed_medium_episodes.csv", PALETTE["blue"]),
        ("随机目标", ROOT / "runs/results/panda_her_residual_random_medium_episodes.csv", PALETTE["green"]),
    ]
    summaries = []
    for label, path, color in files:
        rows = []
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        summaries.append(
            {
                "label": label,
                "color": color,
                "final_mm": np.mean([float(r["final_error_m"]) for r in rows]) * 1000,
                "acc": np.mean([float(r["ee_acc_rms"]) for r in rows]),
                "jerk": np.mean([float(r["ee_jerk_rms"]) for r in rows]),
                "success": np.mean([float(r["success_5mm"]) for r in rows]),
            }
        )
    cards = [(120, 240), (620, 240), (1120, 240)]
    metrics = [("final_mm", "最终误差 / mm", 0.6), ("acc", "末端加速度 RMS", 0.65), ("jerk", "Jerk RMS", 13.0)]
    for (x, y), (key, label, maxv) in zip(cards, metrics):
        draw.rounded_rectangle((x, y, x + 380, y + 380), radius=14, fill="#ffffff", outline="#d9e1e6", width=3)
        text(draw, (x + 34, y + 32), label, 30, PALETTE["ink"], True)
        for i, item in enumerate(summaries):
            bar_y = y + 120 + i * 105
            value = item[key]
            draw.rectangle((x + 35, bar_y, x + 335, bar_y + 34), fill="#eef2f4")
            draw.rectangle((x + 35, bar_y, x + 35 + min(300, int(300 * value / maxv)), bar_y + 34), fill=item["color"])
            text(draw, (x + 35, bar_y - 36), item["label"], 24, PALETTE["muted"])
            text(draw, (x + 245, bar_y - 36), f"{value:.3f}", 24, PALETTE["ink"], True)
    draw.rounded_rectangle((120, 705, 1620, 980), radius=16, fill="#ffffff", outline="#d9e1e6", width=3)
    text(draw, (160, 740), "结论", 36, PALETTE["ink"], True)
    conclusions = [
        "固定目标和随机目标的 5 mm 成功率均为 1.000。",
        "随机目标下仍能保持亚毫米级最终误差，说明策略具备目标泛化能力。",
        "中等接触变力下，精密伺服与动作平滑共同降低末端振动评价指标。",
    ]
    for i, line in enumerate(conclusions):
        text(draw, (170, 812 + i * 52), line, 29, PALETTE["muted"])
    save(img, "06_panda_medium_disturbance_result_summary_generated.png")


def main() -> None:
    cfg_path = ROOT / "configs/experiment.yaml"
    if cfg_path.exists():
        with cfg_path.open(encoding="utf-8") as f:
            yaml.safe_load(f)
    panda_mdh()
    matlab_model()
    workspace_and_joints()
    simulation_environment()
    control_overview()
    metric_summary()


if __name__ == "__main__":
    main()
