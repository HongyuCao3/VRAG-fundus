import os
import tempfile
from PIL import Image

def split_image(img_path, m, n):
    # 打开图片
    img = Image.open(img_path)
    width, height = img.size

    # 计算每个子图的尺寸
    step_x = width // n
    step_y = height // m

    # 创建临时文件夹
    temp_dir = tempfile.mkdtemp()

    # 分割图片并保存到临时文件夹
    sub_images = []
    for i in range(m):
        for j in range(n):
            box = (j * step_x, i * step_y, (j + 1) * step_x, (i + 1) * step_y)
            sub_img = img.crop(box)
            sub_img_path = os.path.join(temp_dir, f"sub_img_{i}_{j}.png")
            sub_img.save(sub_img_path)
            sub_images.append(sub_img_path)

    return sub_images