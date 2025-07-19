import io
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 通用的绘图到 tensor 函数
def fig_to_tensor(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    image = Image.open(buf)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    tensor_image = transform(image.convert("RGB"))

    plt.clf()
    plt.close(fig)
    del buf
    return tensor_image

# 修改后的 double_ellipse_draw 函数
def double_ellipse_draw(args):
    A, B, a, Px, Py, phi = args
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_xlim(-450, 450)
    ax.set_ylim(-450, 450)

    ax.set_facecolor('white')

    rect = patches.Rectangle((-Px / 2, -Py / 2), Px, Py, linewidth=1, edgecolor='yellow', facecolor='yellow')
    ax.add_patch(rect)

    ellipse_1 = patches.Ellipse((Px/4, 0), A, B, angle=-a, linewidth=1, edgecolor='red', facecolor='red')
    ellipse_2 = patches.Ellipse((-Px/4, 0), A, B, angle=a, linewidth=1, edgecolor='red', facecolor='red')

    trans = patches.Affine2D().rotate_deg(phi) + ax.transData
    ellipse_1.set_transform(trans)
    ellipse_2.set_transform(trans)

    ax.add_patch(ellipse_1)
    ax.add_patch(ellipse_2)

    return fig_to_tensor(fig)

# 修改后的 simple_circle_draw 函数
def circle_draw(args):
    A, B, Px, Py, phi = args
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_xlim(-450, 450)
    ax.set_ylim(-450, 450)

    ax.set_facecolor('white')

    rect = patches.Rectangle((-Px / 2, -Py / 2), Px, Py, linewidth=1, edgecolor='yellow', facecolor='yellow')
    ax.add_patch(rect)

    ellipse = patches.Ellipse((0, 0), A, B, angle=0, linewidth=1, edgecolor='red', facecolor='red')

    trans = patches.Affine2D().rotate_deg(phi) + ax.transData
    ellipse.set_transform(trans)

    ax.add_patch(ellipse)

    return fig_to_tensor(fig)

# 修改后的 rec_draw 函数
def lack_rec_draw(args):
    L, W, alpha, beta, gama, Px, Py, phi = args
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_xlim(-450, 450)
    ax.set_ylim(-450, 450)

    ax.set_facecolor('white')

    rect = patches.Rectangle((-Px / 2, -Py / 2), Px, Py, linewidth=1, edgecolor='yellow', facecolor='yellow')
    ax.add_patch(rect)

    rect_1 = patches.Rectangle((-W / 2, -L / 2), W, L * (1 - beta), angle=0, linewidth=1, edgecolor='red', facecolor='red')
    rect_2 = patches.Rectangle((-W / 2, -L / 2 + L * (1 - beta)), W * gama, L * beta, angle=0, linewidth=1, edgecolor='red', facecolor='red')
    rect_3 = patches.Rectangle((W / 2 - W * (1 - alpha - gama), -L / 2 + L * (1 - beta)), W * (1 - alpha - gama), L * beta, angle=0, linewidth=1, edgecolor='red', facecolor='red')

    trans = patches.Affine2D().rotate_deg(phi) + ax.transData
    rect_1.set_transform(trans)
    rect_2.set_transform(trans)
    rect_3.set_transform(trans)

    ax.add_patch(rect_1)
    ax.add_patch(rect_2)
    ax.add_patch(rect_3)

    return fig_to_tensor(fig)

# 修改后的 simple_rec_draw 函数
def rec_draw(args):
    L, W, Px, Py, phi = args
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_xlim(-450, 450)
    ax.set_ylim(-450, 450)

    ax.set_facecolor('white')

    rect = patches.Rectangle((-Px / 2, -Py / 2), Px, Py, linewidth=1, edgecolor='yellow', facecolor='yellow')
    ax.add_patch(rect)

    rect_1 = patches.Rectangle((-L / 2, -W / 2), L, W, angle=0, linewidth=1, edgecolor='red', facecolor='red')

    trans = patches.Affine2D().rotate_deg(phi) + ax.transData
    rect_1.set_transform(trans)

    ax.add_patch(rect_1)

    return fig_to_tensor(fig)

# 修改后的 draw_annulus 函数
def ring_draw(args):
    R, r, theta, phi, Px, Py = args
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_xlim(-450, 450)
    ax.set_ylim(-450, 450)

    ax.set_facecolor('white')

    rect = patches.Rectangle((-Px / 2, -Py / 2), Px, Py, linewidth=1, edgecolor='yellow', facecolor='yellow')
    ax.add_patch(rect)

    wedge_inner = patches.Wedge(center=(0, 0), r=r, theta1=0, theta2=360, width=0, edgecolor='yellow', facecolor='white')
    wedge_outer = patches.Wedge(center=(0, 0), r=R, theta1=-theta / 2, theta2=theta / 2, width=R - r, edgecolor='red', facecolor='red')

    trans = patches.Affine2D().rotate_deg(phi) + ax.transData
    wedge_inner.set_transform(trans)
    wedge_outer.set_transform(trans)

    ax.add_patch(wedge_outer)
    ax.add_patch(wedge_inner)

    return fig_to_tensor(fig)


def double_rectangle_draw(args):
    W1, L1, W2, L2, Px, Py, phi = args
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_xlim(-450, 450)
    ax.set_ylim(-450, 450)
    ax.set_facecolor('white')

    rect_center = (0, 0)
    rect_width = Px
    rect_height = Py
    rect = patches.Rectangle((rect_center[0] - rect_width / 2, rect_center[1] - rect_height / 2),
                             rect_width, rect_height, linewidth=1, edgecolor='yellow', facecolor='yellow')
    ax.add_patch(rect)

    rect_center = (Px / 4, 0)
    rect_width = W2
    rect_height = L2
    angle = 0
    rect_1 = patches.Rectangle((rect_center[0] - rect_width / 2, rect_center[1] - rect_height / 2),
                               rect_width, rect_height, angle=angle, linewidth=1, edgecolor='red', facecolor='red')

    rect_center = (-Px / 4, 0)
    rect_width = W1
    rect_height = L1
    rect_2 = patches.Rectangle((rect_center[0] - rect_width / 2, rect_center[1] - rect_height / 2),
                               rect_width, rect_height, angle=angle, linewidth=1, edgecolor='red', facecolor='red')

    trans = patches.Affine2D().rotate_deg(phi) + ax.transData
    rect_1.set_transform(trans)
    rect_2.set_transform(trans)

    ax.add_patch(rect_1)
    ax.add_patch(rect_2)

    return fig_to_tensor(fig)

def cross_draw(args):  # 修改1：将参数合并为一个元组
    W1, L1, W2, L2, offset, Px, Py, phi = args
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_xlim(-450, 450)
    ax.set_ylim(-450, 450)

    # 绘制白色背景
    ax.set_facecolor('white')

    # 绘制黄色矩形
    rect_center = (0, 0)
    rect_width = Px
    rect_height = Py
    rect = patches.Rectangle((rect_center[0] - rect_width / 2, rect_center[1] - rect_height / 2),
                             rect_width, rect_height, linewidth=1, edgecolor='yellow', facecolor='yellow')
    ax.add_patch(rect)

    # 绘制红色矩形1
    rect_center = (0, 0)
    rect_width = W2
    rect_height = L2
    angle = 0
    rect_1 = patches.Rectangle((rect_center[0] - rect_width / 2, rect_center[1] - rect_height / 2),
                               rect_width, rect_height, angle=angle, linewidth=1, edgecolor='red', facecolor='red')

    # 绘制红色矩形2
    rect_center = (0, offset)
    rect_width = L1
    rect_height = W1
    angle = 0
    rect_2 = patches.Rectangle((rect_center[0] - rect_width / 2, rect_center[1] - rect_height / 2),
                               rect_width, rect_height, angle=angle, linewidth=1, edgecolor='red', facecolor='red')

    # 将两个红色矩形作为一个整体，围绕画布中心进行旋转
    trans = patches.Affine2D().rotate_deg(phi) + ax.transData
    rect_1.set_transform(trans)
    rect_2.set_transform(trans)

    ax.add_patch(rect_1)
    ax.add_patch(rect_2)

    return fig_to_tensor(fig)