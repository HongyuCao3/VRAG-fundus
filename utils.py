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


def delete_images(tmp_path):
    # 检查目录是否存在
    if not os.path.exists(tmp_path):
        print("错误：指定的目录不存在")
        return

    # 获取目录下的所有文件和文件夹
    items = os.listdir(tmp_path)

    # 检查是否只有图片文件
    only_images = all(item.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')) for item in items)

    if not only_images:
        print("错误：目录中包含非图片文件或子目录")
        return

    # 删除所有图片文件
    for item in items:
        file_path = os.path.join(tmp_path, item)
        os.remove(file_path)

    print("成功删除所有图片文件")



if __name__ == "__main__":
    img_path = "./data/DR/multidr/BRSET_img00149.jpg"
    tmp_path = "./data/tmp/"
    sub_imgs = split_image(img_path, tmp_path, 2, 2)
    print(sub_imgs)
    delete_images(tmp_path)