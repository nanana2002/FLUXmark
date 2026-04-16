from PIL import Image, ImageDraw

def draw_black_box(image_path, output_path, x, y, width, height):
    """
    在图片指定位置画黑色方块

    参数:
        image_path: 输入图片路径
        output_path: 输出图片路径
        x, y: 方块左上角的坐标
        width, height: 方块的宽度和高度
    """
    # 打开图片
    img = Image.open(image_path)

    # 创建绘图对象
    draw = ImageDraw.Draw(img)

    # 计算右下角坐标
    x2 = x + width
    y2 = y + height

    # 画黑色方块 (fill=0 表示黑色)
    draw.rectangle([x, y, x2, y2], fill=0)

    # 保存图片
    img.save(output_path)
    print(f"已保存到: {output_path}")

# ========== 使用示例 ==========

# 示例1: 在图片中心画一个 100x100 的黑色方块
draw_black_box(
    image_path='/data/daiyina/project_flux/pic/watermarked_cat.png',
    output_path='/data/daiyina/project_flux/pic/cat_with_box.png',
    x=200,      # 左上角 x 坐标
    y=200,      # 左上角 y 坐标
    width=100,  # 方块宽度
    height=100  # 方块高度
)

# 示例2: 在左上角画一个 50x50 的小黑块
draw_black_box(
    image_path='/data/daiyina/project_flux/pic/watermarked_cat.png',
    output_path='/data/daiyina/project_flux/pic/cat_with_box2.png',
    x=50,
    y=50,
    width=50,
    height=50
)

draw_black_box(
    image_path='/data/daiyina/project_flux/pic/watermarked_cat.png',
    output_path='/data/daiyina/project_flux/pic/cat_with_box3.png',
    x=200,
    y=100,
    width=50,
    height=50
)

draw_black_box(
    image_path='/data/daiyina/project_flux/pic/watermarked_cat.png',
    output_path='/data/daiyina/project_flux/pic/cat_with_box4.png',
    x=200,
    y=100,
    width=100,
    height=50
)

draw_black_box(
    image_path='/data/daiyina/project_flux/pic/watermarked_cat.png',
    output_path='/data/daiyina/project_flux/pic/cat_with_box5.png',
    x=200,
    y=100,
    width=100,
    height=100
)