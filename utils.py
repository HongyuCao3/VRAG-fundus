import os
import tempfile
from PIL import Image

def split_image(img_path, tmp_path, m, n):
    # 打开图片
    img = Image.open(img_path)
    width, height = img.size
    img_name = img_path.split("/")[-1]

    # 计算每个子图的尺寸
    step_x = width // n
    step_y = height // m


    # 分割图片并保存到临时文件夹
    sub_images = []
    for i in range(m):
        for j in range(n):
            box = (j * step_x, i * step_y, (j + 1) * step_x, (i + 1) * step_y)
            sub_img = img.crop(box)
            sub_img_path = os.path.join(tmp_path, f"{img_name}_sub_img_{i}_{j}.png")
            sub_img.save(sub_img_path)
            sub_images.append(sub_img_path)

    return sub_images

if __name__ == "__main__":
    img_path = "./data/DR/multidr/BRSET_img00149.jpg"
    tmp_path = "./data/tmp/"
    sub_imgs = split_image(img_path, tmp_path, 2, 2)
    print(sub_imgs)