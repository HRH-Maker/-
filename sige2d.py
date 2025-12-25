import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import hsv_to_rgb

def skyrmion_spin_field(x, y, radius=5, center=(0, 0), polarity=1, chirality=1):
    """
    生成斯格明子自旋场
    
    参数:
    x, y: 坐标网格
    radius: 斯格明子半径
    center: 斯格明子中心坐标
    polarity: 极性 (+1 或 -1)
    chirality: 手性 (+1 或 -1)
    
    返回:
    sx, sy, sz: 自旋场的三个分量
    """
    # 计算相对中心点的极坐标
    xc = x - center[0]
    yc = y - center[1]
    r = np.sqrt(xc**2 + yc**2)
    phi = np.arctan2(yc, xc)
    
    # 斯格明子自旋场公式
    # 径向函数 f(r)，描述自旋从中心到边缘的变化
    f_r = np.pi * np.exp(-r / radius)
    
    # 自旋场分量
    sx = chirality * np.sin(f_r) * np.cos(phi + np.pi/2)
    sy = chirality * np.sin(f_r) * np.sin(phi + np.pi/2)
    sz = polarity * np.cos(f_r)
    
    # 在远离中心的地方，自旋指向z方向
    sz[r > 3*radius] = polarity
    
    return sx, sy, sz

def create_multi_skyrmion_grid(size=100, num_skyrmions=3):
    """
    创建包含多个斯格明子的自旋场
    
    参数:
    size: 网格大小
    num_skyrmions: 斯格明子数量
    
    返回:
    sx_total, sy_total, sz_total: 总自旋场分量
    """
    # 创建坐标网格
    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    X, Y = np.meshgrid(x, y)
    
    # 初始化自旋场
    sx_total = np.zeros_like(X)
    sy_total = np.zeros_like(X)
    sz_total = np.zeros_like(X)
    
    # 随机生成斯格明子参数
    np.random.seed(42)  # 设置随机种子以确保可重复性
    
    for i in range(num_skyrmions):
        # 随机生成斯格明子中心
        center_x = np.random.uniform(-6, 6)
        center_y = np.random.uniform(-6, 6)
        
        # 随机生成半径 (2到4之间)
        radius = np.random.uniform(2, 4)
        
        # 随机选择极性 (+1 或 -1)
        polarity = np.random.choice([1, -1])
        
        # 随机选择手性 (+1 或 -1)
        chirality = np.random.choice([1, -1])
        
        # 生成单个斯格明子自旋场
        sx, sy, sz = skyrmion_spin_field(X, Y, radius, (center_x, center_y), polarity, chirality)
        
        # 叠加到总场中
        # 使用权重函数，避免重叠区域产生冲突
        weight = np.exp(-((X-center_x)**2 + (Y-center_y)**2) / (2*radius**2))
        sx_total = sx_total * (1-weight) + sx * weight
        sy_total = sy_total * (1-weight) + sy * weight
        sz_total = sz_total * (1-weight) + sz * weight
    
    return X, Y, sx_total, sy_total, sz_total

def plot_skyrmion_field(X, Y, sx, sy, sz, arrow_scale=15):
    """
    绘制斯格明子自旋场
    
    参数:
    X, Y: 坐标网格
    sx, sy, sz: 自旋场分量
    arrow_scale: 箭头缩放因子
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 使用箭头表示自旋方向，颜色表示自旋的方位角
    ax1 = axes[0, 0]
    
    # 计算自旋方位角 (在xy平面上的方向)
    phi = np.arctan2(sy, sx)
    
    # 计算颜色: 使用HSV色彩空间，色相表示方位角
    h = (phi + np.pi) / (2 * np.pi)  # 将角度映射到 [0, 1]
    s = np.ones_like(h)  # 饱和度设为1
    v = np.ones_like(h)  # 亮度设为1
    
    # 将HSV转换为RGB
    hsv = np.dstack((h, s, v))
    rgb = hsv_to_rgb(hsv)
    
    # 绘制箭头，使用颜色表示方向
    # 为了清晰，我们每隔一定间隔绘制箭头
    step = max(1, X.shape[0] // 20)
    
    ax1.quiver(X[::step, ::step], Y[::step, ::step], 
               sx[::step, ::step], sy[::step, ::step],
               color=rgb[::step, ::step, :].reshape(-1, 3),
               scale=arrow_scale, width=0.003, headwidth=3)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('斯格明子自旋方向 (箭头颜色表示方向)')
    ax1.set_aspect('equal')
    
    # 2. 用颜色表示z分量
    ax2 = axes[0, 1]
    im2 = ax2.imshow(sz, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                     cmap='RdBu_r', origin='lower')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('自旋z分量 (红色: 向上, 蓝色: 向下)')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='$S_z$')
    
    # 3. 用颜色表示方位角
    ax3 = axes[1, 0]
    im3 = ax3.imshow(phi, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                     cmap='hsv', origin='lower')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('自旋方位角 $\phi$ (颜色表示方向)')
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3, label='$\phi$ (rad)')
    
    # 4. 自旋场的三维表示
    ax4 = axes[1, 1]
    
    # 计算自旋的仰角
    theta = np.arccos(sz)
    
    # 使用方位角和仰角计算颜色
    # 色相表示方位角，亮度表示仰角 (z分量)
    h = (phi + np.pi) / (2 * np.pi)
    v = (sz + 1) / 2  # 将z分量从[-1,1]映射到[0,1]
    s = np.ones_like(h)
    
    hsv = np.dstack((h, s, v))
    rgb_3d = hsv_to_rgb(hsv)
    
    # 在网格点上绘制彩色点
    step_3d = max(1, X.shape[0] // 40)
    X_sub = X[::step_3d, ::step_3d].flatten()
    Y_sub = Y[::step_3d, ::step_3d].flatten()
    sx_sub = sx[::step_3d, ::step_3d].flatten()
    sy_sub = sy[::step_3d, ::step_3d].flatten()
    sz_sub = sz[::step_3d, ::step_3d].flatten()
    colors_sub = rgb_3d[::step_3d, ::step_3d, :].reshape(-1, 3)
    
    # 绘制箭头
    ax4.quiver(X_sub, Y_sub, sx_sub, sy_sub, color=colors_sub,
               scale=arrow_scale*0.8, width=0.004, headwidth=4)
    
    # 在背景上显示z分量
    bg = ax4.imshow(sz, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                    cmap='gray', alpha=0.2, origin='lower')
    
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_title('自旋场综合表示')
    ax4.set_aspect('equal')
    
    plt.suptitle('斯格明子二维自旋场模拟', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_interactive_skyrmion():
    """创建交互式斯格明子模拟"""
    from matplotlib.widgets import Slider, Button
    
    # 创建初始网格
    size = 80
    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    X, Y = np.meshgrid(x, y)
    
    # 初始参数
    init_radius = 4.0
    init_polarity = 1
    init_chirality = 1
    
    # 初始自旋场
    sx, sy, sz = skyrmion_spin_field(X, Y, init_radius, (0, 0), init_polarity, init_chirality)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.1, bottom=0.35, top=0.95)
    
    # 计算颜色
    phi = np.arctan2(sy, sx)
    h = (phi + np.pi) / (2 * np.pi)
    s = np.ones_like(h)
    v = np.ones_like(h)
    hsv = np.dstack((h, s, v))
    rgb = hsv_to_rgb(hsv)
    
    # 绘制箭头
    step = max(1, size // 15)
    quiver = ax.quiver(X[::step, ::step], Y[::step, ::step], 
                       sx[::step, ::step], sy[::step, ::step],
                       color=rgb[::step, ::step, :].reshape(-1, 3),
                       scale=20, width=0.004, headwidth=4)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('交互式斯格明子模拟')
    ax.set_aspect('equal')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    
    # 创建滑块
    ax_radius = plt.axes([0.1, 0.2, 0.65, 0.03])
    ax_polarity = plt.axes([0.1, 0.15, 0.65, 0.03])
    ax_chirality = plt.axes([0.1, 0.1, 0.65, 0.03])
    
    slider_radius = Slider(ax_radius, '半径', 1.0, 8.0, valinit=init_radius)
    slider_polarity = Slider(ax_polarity, '极性', -1.0, 1.0, valinit=init_polarity, valstep=2.0)
    slider_chirality = Slider(ax_chirality, '手性', -1.0, 1.0, valinit=init_chirality, valstep=2.0)
    
    def update(val):
        # 获取当前滑块值
        radius = slider_radius.val
        polarity = int(slider_polarity.val)
        chirality = int(slider_chirality.val)
        
        # 更新自旋场
        sx, sy, sz = skyrmion_spin_field(X, Y, radius, (0, 0), polarity, chirality)
        
        # 更新颜色
        phi = np.arctan2(sy, sx)
        h = (phi + np.pi) / (2 * np.pi)
        s = np.ones_like(h)
        v = np.ones_like(h)
        hsv = np.dstack((h, s, v))
        rgb = hsv_to_rgb(hsv)
        
        # 更新箭头
        quiver.set_UVC(sx[::step, ::step], sy[::step, ::step])
        quiver.set_color(rgb[::step, ::step, :].reshape(-1, 3))
        
        fig.canvas.draw_idle()
    
    # 将更新函数连接到滑块
    slider_radius.on_changed(update)
    slider_polarity.on_changed(update)
    slider_chirality.on_changed(update)
    
    # 添加重置按钮
    reset_ax = plt.axes([0.8, 0.1, 0.1, 0.04])
    reset_button = Button(reset_ax, '重置', hovercolor='0.975')
    
    def reset(event):
        slider_radius.reset()
        slider_polarity.reset()
        slider_chirality.reset()
    
    reset_button.on_clicked(reset)
    
    plt.show()

# 主程序
if __name__ == "__main__":
    print("斯格明子二维流场模拟")
    print("=" * 50)
    
    # 选择模拟模式
    print("请选择模拟模式:")
    print("1. 单斯格明子模拟")
    print("2. 多斯格明子模拟")
    print("3. 交互式斯格明子模拟")
    
    try:
        mode = int(input("请输入模式编号 (1/2/3): "))
    except:
        mode = 1
    
    if mode == 1:
        # 单斯格明子模拟
        print("\n生成单斯格明子模拟...")
        size = 100
        
        # 创建坐标网格
        x = np.linspace(-10, 10, size)
        y = np.linspace(-10, 10, size)
        X, Y = np.meshgrid(x, y)
        
        # 生成自旋场
        sx, sy, sz = skyrmion_spin_field(X, Y, radius=4.0, center=(0, 0), polarity=1, chirality=1)
        
        # 绘制结果
        plot_skyrmion_field(X, Y, sx, sy, sz)
        
    elif mode == 2:
        # 多斯格明子模拟
        print("\n生成多斯格明子模拟...")
        X, Y, sx, sy, sz = create_multi_skyrmion_grid(size=120, num_skyrmions=4)
        
        # 绘制结果
        plot_skyrmion_field(X, Y, sx, sy, sz)
        
    elif mode == 3:
        # 交互式模拟
        print("\n启动交互式斯格明子模拟...")
        plot_interactive_skyrmion()
    
    else:
        print("无效的选择，使用默认单斯格明子模拟")
        size = 100
        x = np.linspace(-10, 10, size)
        y = np.linspace(-10, 10, size)
        X, Y = np.meshgrid(x, y)
        sx, sy, sz = skyrmion_spin_field(X, Y, radius=4.0, center=(0, 0), polarity=1, chirality=1)
        plot_skyrmion_field(X, Y, sx, sy, sz)