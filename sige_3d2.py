import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D

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
    f_r = np.pi * np.exp(-r / radius)
    
    # 自旋场分量
    sx = chirality * np.sin(f_r) * np.cos(phi + np.pi/2)
    sy = chirality * np.sin(f_r) * np.sin(phi + np.pi/2)
    sz = polarity * np.cos(f_r)
    
    # 在远离中心的地方，自旋指向z方向
    sz[r > 3*radius] = polarity
    
    return sx, sy, sz

def plot_3d_skyrmion_layer():
    """
    在三维空间中展示单层斯格明子自旋箭头
    """
    # 创建坐标网格
    size = 30  # 减小网格密度以便清晰显示
    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    X, Y = np.meshgrid(x, y)
    
    # 生成斯格明子自旋场
    sx, sy, sz = skyrmion_spin_field(X, Y, radius=4.0, center=(0, 0), polarity=1, chirality=1)
    
    # 创建3D图形
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 3D箭头视图 - 从不同角度观察
    angles = [(30, 45), (30, 135), (60, 45), (90, 0)]
    titles = ['视角1 (30°, 45°)', '视角2 (30°, 135°)', '视角3 (60°, 45°)', '俯视图 (90°, 0°)']
    
    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        # 计算自旋方位角用于颜色编码
        phi = np.arctan2(sy, sx)
        h = (phi + np.pi) / (2 * np.pi)  # 将方位角映射到 [0, 1] 作为色相
        
        # 创建颜色数组 - 使用方位角作为色相，z分量作为亮度
        s = np.ones_like(h)  # 饱和度设为1
        v = (sz + 1) / 2  # 将z分量从[-1,1]映射到[0,1]作为亮度
        hsv = np.dstack((h, s, v))
        colors = hsv_to_rgb(hsv)
        
        # 展平数组用于绘制
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = np.zeros_like(X_flat)  # 所有箭头在同一平面
        
        sx_flat = sx.flatten()
        sy_flat = sy.flatten()
        sz_flat = sz.flatten()
        colors_flat = colors.reshape(-1, 3)
        
        # 在3D空间中绘制箭头
        # 为了清晰，只绘制部分箭头
        step = 1  # 可以调整为2来减少箭头密度
        
        # 绘制箭头
        ax.quiver(X_flat[::step], Y_flat[::step], Z_flat[::step],
                  sx_flat[::step], sy_flat[::step], sz_flat[::step],
                  color=colors_flat[::step], 
                  length=1.5,  # 箭头长度
                  normalize=True,  # 标准化箭头长度
                  arrow_length_ratio=0.3,  # 箭头头部比例
                  linewidth=1.5,  # 线条宽度
                  alpha=0.8)  # 透明度
        
        # 设置坐标轴
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(titles[i])
        
        # 设置视角
        ax.view_init(elev=elev, azim=azim)
        
        # 设置坐标轴范围
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-1.5, 1.5])
        
        # 添加网格
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('斯格明子三维自旋场 - 单层箭头展示', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 2. 详细3D视图 - 一个大的3D图
    fig2 = plt.figure(figsize=(14, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    # 计算颜色
    phi = np.arctan2(sy, sx)
    h = (phi + np.pi) / (2 * np.pi)
    s = np.ones_like(h)
    v = (sz + 1) / 2
    hsv = np.dstack((h, s, v))
    colors = hsv_to_rgb(hsv)
    
    # 展平数组
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = np.zeros_like(X_flat)
    
    sx_flat = sx.flatten()
    sy_flat = sy.flatten()
    sz_flat = sz.flatten()
    colors_flat = colors.reshape(-1, 3)
    
    # 绘制所有箭头
    ax2.quiver(X_flat, Y_flat, Z_flat,
               sx_flat, sy_flat, sz_flat,
               color=colors_flat,
               length=1.5,
               normalize=True,
               arrow_length_ratio=0.3,
               linewidth=1.2,
               alpha=0.7)
    
    # 设置坐标轴
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_zlabel('Z', fontsize=12)
    ax2.set_title('斯格明子三维自旋场详细视图', fontsize=14, fontweight='bold')
    
    # 设置视角
    ax2.view_init(elev=25, azim=45)
    
    # 设置坐标轴范围
    ax2.set_xlim([-10, 10])
    ax2.set_ylim([-10, 10])
    ax2.set_zlim([-1.5, 1.5])
    
    # 添加网格
    ax2.grid(True, alpha=0.3)
    
    # 添加颜色说明
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches
    
    # 创建颜色图例
    legend_elements = [
        mpatches.Patch(color='red', label='+X方向 (右)'),
        mpatches.Patch(color='yellow', label='+Y方向 (上)'),
        mpatches.Patch(color='cyan', label='-X方向 (左)'),
        mpatches.Patch(color='purple', label='-Y方向 (下)'),
        mpatches.Patch(color='white', label='向上自旋'),
        mpatches.Patch(color='black', label='向下自旋'),
    ]
    
    ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    plt.tight_layout()
    plt.show()
    
    # 3. 交互式3D视图 - 可以旋转
    fig3 = plt.figure(figsize=(12, 10))
    ax3 = fig3.add_subplot(111, projection='3d')
    
    # 使用更密集的网格
    size_dense = 20
    x_dense = np.linspace(-10, 10, size_dense)
    y_dense = np.linspace(-10, 10, size_dense)
    X_dense, Y_dense = np.meshgrid(x_dense, y_dense)
    
    # 生成自旋场
    sx_dense, sy_dense, sz_dense = skyrmion_spin_field(X_dense, Y_dense, radius=4.0, center=(0, 0), polarity=1, chirality=1)
    
    # 计算颜色
    phi_dense = np.arctan2(sy_dense, sx_dense)
    h_dense = (phi_dense + np.pi) / (2 * np.pi)
    s_dense = np.ones_like(h_dense)
    v_dense = (sz_dense + 1) / 2
    hsv_dense = np.dstack((h_dense, s_dense, v_dense))
    colors_dense = hsv_to_rgb(hsv_dense)
    
    # 展平数组
    X_dense_flat = X_dense.flatten()
    Y_dense_flat = Y_dense.flatten()
    Z_dense_flat = np.zeros_like(X_dense_flat)
    
    sx_dense_flat = sx_dense.flatten()
    sy_dense_flat = sy_dense.flatten()
    sz_dense_flat = sz_dense.flatten()
    colors_dense_flat = colors_dense.reshape(-1, 3)
    
    # 绘制箭头
    ax3.quiver(X_dense_flat, Y_dense_flat, Z_dense_flat,
               sx_dense_flat, sy_dense_flat, sz_dense_flat,
               color=colors_dense_flat,
               length=1.2,
               normalize=True,
               arrow_length_ratio=0.25,
               linewidth=1.5,
               alpha=0.8)
    
    # 添加背景平面
    xx, yy = np.meshgrid(np.linspace(-10, 10, 2), np.linspace(-10, 10, 2))
    zz = np.zeros_like(xx)
    ax3.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
    
    # 设置坐标轴
    ax3.set_xlabel('X', fontsize=12)
    ax3.set_ylabel('Y', fontsize=12)
    ax3.set_zlabel('Z', fontsize=12)
    ax3.set_title('交互式斯格明子3D视图 - 可用鼠标旋转', fontsize=14, fontweight='bold')
    
    # 设置坐标轴范围
    ax3.set_xlim([-10, 10])
    ax3.set_ylim([-10, 10])
    ax3.set_zlim([-1.5, 1.5])
    
    # 添加网格
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_3d_multiple_skyrmions():
    """
    在三维空间中展示多个斯格明子
    """
    # 创建坐标网格
    size = 25
    x = np.linspace(-15, 15, size)
    y = np.linspace(-15, 15, size)
    X, Y = np.meshgrid(x, y)
    
    # 创建多个斯格明子
    sx_total = np.zeros_like(X)
    sy_total = np.zeros_like(X)
    sz_total = np.zeros_like(X)
    
    # 定义斯格明子参数
    skyrmions = [
        {'center': (-5, -5), 'radius': 3, 'polarity': 1, 'chirality': 1},
        {'center': (5, 5), 'radius': 3, 'polarity': 1, 'chirality': 1},
        {'center': (-5, 5), 'radius': 3, 'polarity': -1, 'chirality': 1},
        {'center': (5, -5), 'radius': 3, 'polarity': -1, 'chirality': -1},
    ]
    
    for skyrmion in skyrmions:
        sx, sy, sz = skyrmion_spin_field(X, Y, 
                                         radius=skyrmion['radius'], 
                                         center=skyrmion['center'], 
                                         polarity=skyrmion['polarity'], 
                                         chirality=skyrmion['chirality'])
        
        # 使用权重函数叠加
        center_x, center_y = skyrmion['center']
        radius = skyrmion['radius']
        weight = np.exp(-((X-center_x)**2 + (Y-center_y)**2) / (2*radius**2))
        
        sx_total = sx_total * (1-weight) + sx * weight
        sy_total = sy_total * (1-weight) + sy * weight
        sz_total = sz_total * (1-weight) + sz * weight
    
    # 创建3D图形
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 计算颜色
    phi = np.arctan2(sy_total, sx_total)
    h = (phi + np.pi) / (2 * np.pi)
    s = np.ones_like(h)
    v = (sz_total + 1) / 2
    hsv = np.dstack((h, s, v))
    colors = hsv_to_rgb(hsv)
    
    # 展平数组
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = np.zeros_like(X_flat)
    
    sx_flat = sx_total.flatten()
    sy_flat = sy_total.flatten()
    sz_flat = sz_total.flatten()
    colors_flat = colors.reshape(-1, 3)
    
    # 绘制箭头
    ax.quiver(X_flat, Y_flat, Z_flat,
              sx_flat, sy_flat, sz_flat,
              color=colors_flat,
              length=1.5,
              normalize=True,
              arrow_length_ratio=0.3,
              linewidth=1.2,
              alpha=0.7)
    
    # 添加背景平面
    xx, yy = np.meshgrid(np.linspace(-15, 15, 2), np.linspace(-15, 15, 2))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
    
    # 标记斯格明子中心
    for skyrmion in skyrmions:
        center_x, center_y = skyrmion['center']
        polarity = skyrmion['polarity']
        color = 'red' if polarity > 0 else 'blue'
        ax.scatter([center_x], [center_y], [0], color=color, s=100, alpha=0.7)
    
    # 设置坐标轴
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('多个斯格明子三维自旋场', fontsize=14, fontweight='bold')
    
    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    # 设置坐标轴范围
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_zlim([-1.5, 1.5])
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color='red', label='正极性斯格明子中心'),
        mpatches.Patch(color='blue', label='负极性斯格明子中心'),
        mpatches.Patch(color='green', alpha=0.5, label='自旋箭头 (颜色表示方向)'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    print("三维斯格明子自旋场模拟")
    print("=" * 50)
    
    # 选择模拟模式
    print("请选择三维模拟模式:")
    print("1. 单斯格明子三维视图")
    print("2. 多斯格明子三维视图")
    
    try:
        mode = int(input("请输入模式编号 (1/2): "))
    except:
        mode = 1
    
    if mode == 1:
        print("\n生成单斯格明子三维视图...")
        plot_3d_skyrmion_layer()
    elif mode == 2:
        print("\n生成多斯格明子三维视图...")
        plot_3d_multiple_skyrmions()
    else:
        print("无效的选择，使用默认单斯格明子三维视图")
        plot_3d_skyrmion_layer()