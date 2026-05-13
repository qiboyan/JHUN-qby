import os
import time
import cv2
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle
import warnings

warnings.filterwarnings("ignore")

# ===================== 全局配置 =====================
CONFIG = {
    "IMG_SIZE": 256,
    "DATA_ROOT": r"D:\unet\Satellite Images\img256_val_new",
    "SAVE_OUTPUT_DIR": r"D:\unet\system_output",
    "RADIO_PARAMS": {
        "max_radius_pixel": 180,
        "decay_power": 1.2,
        "building_attenuation_per_pixel": 0.012,
        "min_signal": 0.05,
        "shadow_fading_std": 1.5,
    },
    "VISUAL_PARAMS": {
        "station_radius_inner": 8,
        "station_radius_outer": 15,
        "station_color_inner": "white",
        "station_color_outer": "black"
    },
    "DEBUG": True
}

os.makedirs(CONFIG["SAVE_OUTPUT_DIR"], exist_ok=True)

# 热力图配色（固定不变）
radio_color_list = [
    "#00008B", "#1E90FF", "#00FFFF", "#00FF00",
    "#FFFF00", "#FFA500", "#FF4500", "#FF0000"
]
radio_cmap = LinearSegmentedColormap.from_list("radio_gradient", radio_color_list, N=256)
# 固定归一化范围，色条永远0-1
radio_norm = Normalize(vmin=0, vmax=1)

# 全局交互变量（布局固定，仅更新内容）
GLOBAL_STATE = {
    "ori_image": None,
    "building_mask": None,
    "radio_heat_map": None,
    "base_station_pos": None,
    "img_file_list": [],
    "current_img_idx": 0,
    "save_count": 0,
    "demo_seed": 42,
    # 固定布局的绘图对象
    "fig": None,
    "ax1": None,
    "ax2": None,
    "ax3": None,
    "ax4": None,
    "cbar_ax": None,  # 独立色条坐标轴，固定位置
    "cbar": None,  # 预创建的色条，只更新数据不重建
}


# ===================== 图像加载模块 =====================
def load_remote_sensing_image(file_path):
    if CONFIG["DEBUG"]:
        print(f"\n📥 尝试加载：{os.path.basename(file_path)}")

    try:
        with rasterio.open(file_path) as src:
            img_data = src.read()
        img_data = np.transpose(img_data, (1, 2, 0))
        if img_data.shape[-1] > 3:
            img_data = img_data[..., :3]

        img_data = img_data.astype(np.float32)
        for c in range(img_data.shape[-1]):
            channel_data = img_data[..., c]
            channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=0.0, neginf=0.0)
            p2, p98 = np.percentile(channel_data, (2, 98))
            if p98 - p2 > 1e-6:
                channel_data = np.clip(channel_data, p2, p98)
                channel_data = (channel_data - p2) / (p98 - p2 + 1e-8)
            else:
                channel_min, channel_max = channel_data.min(), channel_data.max()
                if channel_max - channel_min > 1e-6:
                    channel_data = (channel_data - channel_min) / (channel_max - channel_min + 1e-8)
                else:
                    channel_data = np.zeros_like(channel_data)
            img_data[..., c] = channel_data

        img_data = cv2.resize(img_data, (CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]))
        if CONFIG["DEBUG"]:
            print(f"✅ rasterio加载完成")
        return img_data.astype(np.float32)

    except Exception as rasterio_err:
        if CONFIG["DEBUG"]:
            print(f"⚠️  rasterio加载失败，尝试OpenCV")

    try:
        img_data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img_data is None:
            raise ValueError("OpenCV读取返回空")

        if len(img_data.shape) == 3:
            if img_data.shape[-1] >= 3:
                img_data = cv2.cvtColor(img_data[..., :3], cv2.COLOR_BGR2RGB)
            else:
                img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
        else:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)

        img_data = img_data.astype(np.float32) / 255.0
        img_data = cv2.resize(img_data, (CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]))
        if CONFIG["DEBUG"]:
            print(f"✅ OpenCV加载完成")
        return img_data

    except Exception as opencv_err:
        if CONFIG["DEBUG"]:
            print(f"⚠️  OpenCV加载失败，生成模拟图")

    return generate_demo_satellite_image()


def generate_demo_satellite_image(seed=None):
    if seed is not None:
        np.random.seed(seed)
    img = np.zeros((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"], 3), dtype=np.float32)
    img[..., 1] = 0.2 + np.random.rand() * 0.2
    img[..., 2] = 0.1 + np.random.rand() * 0.2
    num_buildings = 10 + np.random.randint(0, 10)
    for _ in range(num_buildings):
        x1 = np.random.randint(20, CONFIG["IMG_SIZE"] - 50)
        y1 = np.random.randint(20, CONFIG["IMG_SIZE"] - 50)
        x2 = x1 + np.random.randint(15, 50)
        y2 = y1 + np.random.randint(15, 50)
        img[y1:y2, x1:x2, :] = [0.4 + np.random.rand() * 0.2, 0.4 + np.random.rand() * 0.2,
                                0.4 + np.random.rand() * 0.2]
    if np.random.rand() > 0.5:
        road_y = np.random.randint(30, CONFIG["IMG_SIZE"] - 30)
        img[road_y:road_y + 5, :, :] = [0.2, 0.2, 0.2]
    if np.random.rand() > 0.5:
        road_x = np.random.randint(30, CONFIG["IMG_SIZE"] - 30)
        img[:, road_x:road_x + 5, :] = [0.2, 0.2, 0.2]
    if CONFIG["DEBUG"]:
        print(f"✅ 生成模拟测试图（种子: {seed}）")
    return img


def extract_building_mask(image):
    gray_img = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    gray_img = (gray_img * 255).astype(np.uint8)
    gray_img = cv2.equalizeHist(gray_img)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 1.2)
    _, binary_mask = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    building_mask = binary_mask / 255.0
    if CONFIG["DEBUG"]:
        print(f"🏢 建筑掩码生成，建筑占比：{np.sum(building_mask > 0.5) / (CONFIG['IMG_SIZE'] ** 2):.2%}")
    return building_mask


# ===================== 无线电传播模型 =====================
def calculate_path_building_count(build_mask, tx_pos, rx_pos):
    h, w = build_mask.shape
    x0, y0 = tx_pos
    x1, y1 = rx_pos
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    building_pixel_count = 0

    while True:
        if 0 <= x < w and 0 <= y < h:
            if build_mask[y, x] > 0.5:
                building_pixel_count += 1
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return building_pixel_count


def generate_radio_map(build_mask, station_pos):
    h, w = build_mask.shape
    bx, by = station_pos
    params = CONFIG["RADIO_PARAMS"]

    if not (0 <= bx < w and 0 <= by < h):
        return np.full((h, w), params["min_signal"], dtype=np.float32)

    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    distance = np.sqrt((x_coords - bx) ** 2 + (y_coords - by) ** 2)

    max_radius = params["max_radius_pixel"]
    signal_strength = 1.0 - (distance / max_radius) ** params["decay_power"]
    signal_strength = np.clip(signal_strength, params["min_signal"], 1.0)

    for y in range(h):
        for x in range(w):
            building_count = calculate_path_building_count(build_mask, (bx, by), (x, y))
            path_attenuation = np.exp(-building_count * params["building_attenuation_per_pixel"])
            signal_strength[y, x] *= path_attenuation

    if params["shadow_fading_std"] > 0:
        np.random.seed(int(bx * 1000 + by))
        shadow = np.random.normal(0, params["shadow_fading_std"] / 100, (h, w))
        shadow = cv2.GaussianBlur(shadow, (5, 5), 1.0)
        signal_strength = np.clip(signal_strength + shadow, params["min_signal"], 1.0)

    return signal_strength


# ===================== 交互事件 =====================
def mouse_click_event(event):
    if event.inaxes != GLOBAL_STATE["ax3"]:
        return
    if event.xdata is None or event.ydata is None:
        return

    click_x = int(np.clip(round(event.xdata), 0, CONFIG["IMG_SIZE"] - 1))
    click_y = int(np.clip(round(event.ydata), 0, CONFIG["IMG_SIZE"] - 1))
    GLOBAL_STATE["base_station_pos"] = (click_x, click_y)

    print(f"\n🖱️  检测到鼠标点击：({click_x}, {click_y})")

    infer_start = time.time()
    GLOBAL_STATE["radio_heat_map"] = generate_radio_map(
        GLOBAL_STATE["building_mask"],
        GLOBAL_STATE["base_station_pos"]
    )
    infer_cost = time.time() - infer_start

    heat_map = GLOBAL_STATE["radio_heat_map"]
    print("-" * 80)
    print(f"📍 基站坐标：X={click_x}, Y={click_y}")
    print(f"⚡ 推理耗时：{infer_cost:.4f} s")
    print(
        f"📶 信号范围：[{np.min(heat_map):.3f}, {np.max(heat_map):.3f}] | 最大半径：{CONFIG['RADIO_PARAMS']['max_radius_pixel']}像素")
    print("-" * 80)

    refresh_canvas()


def key_press_event(event):
    print(f"\n⌨️  检测到按键：'{event.key}'")

    if event.key.lower() == "n":
        if len(GLOBAL_STATE["img_file_list"]) > 0:
            GLOBAL_STATE["current_img_idx"] = (GLOBAL_STATE["current_img_idx"] + 1) % len(GLOBAL_STATE["img_file_list"])
            print(f"📷 切换到下一张：{GLOBAL_STATE['img_file_list'][GLOBAL_STATE['current_img_idx']]}")
        else:
            GLOBAL_STATE["demo_seed"] += 1
            print(f"📷 切换到新的模拟图（种子: {GLOBAL_STATE['demo_seed']}）")
        reset_station_and_heatmap()

    elif event.key.lower() == "p":
        if len(GLOBAL_STATE["img_file_list"]) > 0:
            GLOBAL_STATE["current_img_idx"] = (GLOBAL_STATE["current_img_idx"] - 1) % len(GLOBAL_STATE["img_file_list"])
            print(f"📷 切换到上一张：{GLOBAL_STATE['img_file_list'][GLOBAL_STATE['current_img_idx']]}")
        else:
            GLOBAL_STATE["demo_seed"] -= 1
            print(f"📷 切换到新的模拟图（种子: {GLOBAL_STATE['demo_seed']}）")
        reset_station_and_heatmap()

    elif event.key == "]":
        CONFIG["RADIO_PARAMS"]["max_radius_pixel"] = min(300, CONFIG["RADIO_PARAMS"]["max_radius_pixel"] + 10)
        print(f"🔧 最大传播半径调整为：{CONFIG['RADIO_PARAMS']['max_radius_pixel']}像素")
        if GLOBAL_STATE["base_station_pos"] is not None:
            GLOBAL_STATE["radio_heat_map"] = generate_radio_map(
                GLOBAL_STATE["building_mask"],
                GLOBAL_STATE["base_station_pos"]
            )
            refresh_canvas()

    elif event.key == "[":
        CONFIG["RADIO_PARAMS"]["max_radius_pixel"] = max(20, CONFIG["RADIO_PARAMS"]["max_radius_pixel"] - 10)
        print(f"🔧 最大传播半径调整为：{CONFIG['RADIO_PARAMS']['max_radius_pixel']}像素")
        if GLOBAL_STATE["base_station_pos"] is not None:
            GLOBAL_STATE["radio_heat_map"] = generate_radio_map(
                GLOBAL_STATE["building_mask"],
                GLOBAL_STATE["base_station_pos"]
            )
            refresh_canvas()

    elif event.key.lower() == "s":
        save_current_results()

    elif event.key.lower() == "q":
        print("\n👋 程序正常退出")
        plt.close()

    else:
        print(f"   未绑定该按键，可用按键：N/P/[]/S/Q")
        return

    if event.key.lower() in ["n", "p", "[", "]"]:
        refresh_canvas()


def reset_station_and_heatmap():
    if len(GLOBAL_STATE["img_file_list"]) > 0:
        img_path = os.path.join(CONFIG["DATA_ROOT"], GLOBAL_STATE["img_file_list"][GLOBAL_STATE["current_img_idx"]])
        GLOBAL_STATE["ori_image"] = load_remote_sensing_image(img_path)
    else:
        GLOBAL_STATE["ori_image"] = generate_demo_satellite_image(GLOBAL_STATE["demo_seed"])

    GLOBAL_STATE["building_mask"] = extract_building_mask(GLOBAL_STATE["ori_image"])
    GLOBAL_STATE["base_station_pos"] = None
    GLOBAL_STATE["radio_heat_map"] = None


def load_current_image():
    if len(GLOBAL_STATE["img_file_list"]) == 0:
        print("⚠️  无有效图像文件，使用模拟测试图")
        GLOBAL_STATE["ori_image"] = generate_demo_satellite_image(GLOBAL_STATE["demo_seed"])
        GLOBAL_STATE["building_mask"] = extract_building_mask(GLOBAL_STATE["ori_image"])
        return

    img_path = os.path.join(CONFIG["DATA_ROOT"], GLOBAL_STATE["img_file_list"][GLOBAL_STATE["current_img_idx"]])
    GLOBAL_STATE["ori_image"] = load_remote_sensing_image(img_path)
    GLOBAL_STATE["building_mask"] = extract_building_mask(GLOBAL_STATE["ori_image"])


# ===================== 【核心修复】画布刷新：彻底清空旧内容，布局永久固定 =====================
def refresh_canvas():
    ax1, ax2, ax3, ax4 = GLOBAL_STATE["ax1"], GLOBAL_STATE["ax2"], GLOBAL_STATE["ax3"], GLOBAL_STATE["ax4"]

    # 1. 清空所有子图内容（不改变子图位置/大小，布局永久固定）
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    # 2. 重绘(1)卫星遥感原图
    ax1.imshow(GLOBAL_STATE["ori_image"])
    ax1.set_title("(1) 卫星遥感原图", fontsize=12, pad=8, fontweight="bold")
    ax1.axis("off")

    # 3. 重绘(2)建筑掩码
    ax2.imshow(GLOBAL_STATE["building_mask"], cmap="gray")
    ax2.set_title("(2) 提取建筑掩码", fontsize=12, pad=8, fontweight="bold")
    ax2.axis("off")

    # 4. 重绘(3)基站放置区
    ax3.imshow(np.zeros((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]), dtype=np.float32), cmap="gray")
    if GLOBAL_STATE["base_station_pos"] is not None:
        bx, by = GLOBAL_STATE["base_station_pos"]
        circle_inner = Circle((bx, by), CONFIG["VISUAL_PARAMS"]["station_radius_inner"],
                              color="red", alpha=0.9)
        circle_outer = Circle((bx, by), CONFIG["VISUAL_PARAMS"]["station_radius_outer"],
                              color="yellow", alpha=0.6, fill=False, linewidth=2)
        ax3.add_patch(circle_inner)
        ax3.add_patch(circle_outer)
        ax3.text(bx + 15, by, f"({bx},{by})", fontsize=10, color="white",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
    ax3.set_title("(3) 基站放置区（点击此处放置）", fontsize=10, pad=8, fontweight="bold")
    ax3.set_xlim(0, CONFIG["IMG_SIZE"])
    ax3.set_ylim(CONFIG["IMG_SIZE"], 0)
    ax3.axis("off")

    # 5. 【核心修复】重绘(4)热力图：先clear彻底清空，再重绘所有内容，100%无残留
    if GLOBAL_STATE["radio_heat_map"] is not None:
        heat_map = GLOBAL_STATE["radio_heat_map"]
        # 重绘热力图
        im4 = ax4.imshow(heat_map, cmap=radio_cmap, norm=radio_norm)
        # 叠加建筑轮廓
        ax4.contour(GLOBAL_STATE["building_mask"], levels=[0.5], colors="white", linewidths=1.5, linestyles="-")
        # 叠加当前基站的传播半径虚线
        max_radius = CONFIG["RADIO_PARAMS"]["max_radius_pixel"]
        circle_radius = Circle(GLOBAL_STATE["base_station_pos"], max_radius,
                               color="white", alpha=0.5, fill=False, linewidth=1, linestyle="--")
        ax4.add_patch(circle_radius)
        # 叠加当前基站标记
        bx, by = GLOBAL_STATE["base_station_pos"]
        circle_hm_inner = Circle((bx, by), CONFIG["VISUAL_PARAMS"]["station_radius_inner"],
                                 color=CONFIG["VISUAL_PARAMS"]["station_color_inner"], alpha=1.0, zorder=5)
        circle_hm_outer = Circle((bx, by), CONFIG["VISUAL_PARAMS"]["station_radius_outer"],
                                 color=CONFIG["VISUAL_PARAMS"]["station_color_outer"], alpha=1.0, fill=False,
                                 linewidth=2, zorder=5)
        ax4.add_patch(circle_hm_inner)
        ax4.add_patch(circle_hm_outer)
        # 更新固定色条的数据
        GLOBAL_STATE["cbar"].update_normal(im4)
    else:
        # 无热力图时，显示空图和提示
        empty_map = np.zeros((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]), dtype=np.float32)
        im4 = ax4.imshow(empty_map, cmap=radio_cmap, norm=radio_norm)
        ax4.text(128, 128, "请在(3)区点击放置基站", ha="center", va="center",
                 fontsize=14, color="white", bbox=dict(facecolor="black", alpha=0.7))
        # 更新色条
        GLOBAL_STATE["cbar"].update_normal(im4)

    # 固定第4张图的标题和坐标轴
    ax4.set_title("(4) 无线电覆盖热力图（线性衰减+建筑遮挡）", fontsize=11, pad=8, fontweight="bold")
    ax4.axis("off")
    ax4.set_xlim(0, CONFIG["IMG_SIZE"])
    ax4.set_ylim(CONFIG["IMG_SIZE"], 0)

    # 【绝对禁止】不调用任何布局重计算函数，布局永久固定
    GLOBAL_STATE["fig"].canvas.draw_idle()


# ===================== 结果保存 =====================
def save_current_results():
    if GLOBAL_STATE["ori_image"] is None:
        print("⚠️ 无有效图像可保存")
        return

    GLOBAL_STATE["save_count"] += 1
    save_prefix = f"radio_heatmap_{GLOBAL_STATE['current_img_idx']}_{GLOBAL_STATE['save_count']}"
    save_path = os.path.join(CONFIG["SAVE_OUTPUT_DIR"], save_prefix)

    img1 = (GLOBAL_STATE["ori_image"] * 255).astype(np.uint8)
    img2 = (GLOBAL_STATE["building_mask"] * 255).astype(np.uint8)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    img3 = np.zeros((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"], 3), dtype=np.uint8)
    if GLOBAL_STATE["base_station_pos"] is not None:
        bx, by = GLOBAL_STATE["base_station_pos"]
        cv2.circle(img3, (bx, by), 8, (0, 0, 255), -1)
        cv2.circle(img3, (bx, by), 15, (0, 255, 255), 2)

    img4 = np.zeros((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"], 3), dtype=np.uint8)
    if GLOBAL_STATE["radio_heat_map"] is not None:
        heat_norm = np.clip(GLOBAL_STATE["radio_heat_map"], 0, 1)
        img4 = (radio_cmap(heat_norm)[:, :, :3] * 255).astype(np.uint8)

    top_row = np.hstack((img1, img2))
    bottom_row = np.hstack((img3, img4))
    final_img = np.vstack((top_row, bottom_row))

    cv2.imwrite(f"{save_path}.png", cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    if GLOBAL_STATE["radio_heat_map"] is not None:
        np.save(f"{save_path}_data.npy", GLOBAL_STATE["radio_heat_map"])

    print(f"\n💾 结果已保存：{save_path}.png")


# ===================== 系统初始化 =====================
def init_system():
    print("=" * 80)
    print(" 无线电地图预测演示系统（残留问题修复版）")
    print("=" * 80)

    if not os.path.exists(CONFIG["DATA_ROOT"]):
        print(f"❌ 错误：图像路径不存在！已自动启用模拟测试图模式")
        GLOBAL_STATE["img_file_list"] = []
    else:
        print(f"📂 正在扫描路径：{CONFIG['DATA_ROOT']}")
        support_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
        GLOBAL_STATE["img_file_list"] = sorted([
            f for f in os.listdir(CONFIG["DATA_ROOT"])
            if f.lower().endswith(support_exts)
        ])
        if len(GLOBAL_STATE["img_file_list"]) > 0:
            print(f"✅ 成功扫描到 {len(GLOBAL_STATE['img_file_list'])} 个图像文件")
        else:
            print(f"⚠️  路径下未找到支持的图像文件，已启用模拟测试图模式")

    load_current_image()

    print("\n💡 核心修复：")
    print("   1. 每次刷新彻底清空第4张图，彻底解决旧基站/范围虚线残留问题")
    print("   2. 布局永久固定，热力图不会缩小、色条不会左移")
    print("\n💡 操作说明：")
    print("   🖱️  点击(3)区放置基站 | ⌨️  N/P=切换 | []=调半径 | S=保存 | Q=退出")
    print("=" * 80 + "\n")


# ===================== 主程序入口（固定布局） =====================
if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 100

    init_system()

    # 创建固定大小的画布
    fig = plt.figure(figsize=(14, 12))
    GLOBAL_STATE["fig"] = fig

    # 用GridSpec永久固定2x2子图布局，位置、间距、比例永不改变
    gs = fig.add_gridspec(2, 2, wspace=0.15, hspace=0.2, left=0.05, right=0.9)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # 保存子图到全局变量
    GLOBAL_STATE["ax1"] = ax1
    GLOBAL_STATE["ax2"] = ax2
    GLOBAL_STATE["ax3"] = ax3
    GLOBAL_STATE["ax4"] = ax4

    # 单独创建固定位置的色条坐标轴，永远在最右侧，不挤压子图
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    GLOBAL_STATE["cbar_ax"] = cbar_ax

    # 预创建色条，绑定固定坐标轴，位置永不改变
    empty_map = np.zeros((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]), dtype=np.float32)
    init_im = ax4.imshow(empty_map, cmap=radio_cmap, norm=radio_norm)
    cbar = fig.colorbar(init_im, cax=cbar_ax)
    cbar.set_label("归一化信号强度 (1=最强红色, 0=最弱蓝色)", fontsize=10)
    GLOBAL_STATE["cbar"] = cbar

    # 全局标题固定位置
    fig.suptitle("面向卫星遥感的无线电地图预测演示系统", fontsize=16, y=0.98, fontweight="bold")

    # 绑定交互事件
    fig.canvas.mpl_connect("button_press_event", mouse_click_event)
    fig.canvas.mpl_connect("key_press_event", key_press_event)

    # 初始刷新
    refresh_canvas()

    # 显示窗口
    plt.show()