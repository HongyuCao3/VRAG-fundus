import shutil
import tempfile
import os, json
import tempfile
import PIL
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
    
    

# Function to reduce image size and save to temporary directory
def reduce_image_size(input_path, output_folder, size=(100, 100)):
    img = Image.open(input_path)
    img = img.resize(size, PIL.Image.Resampling.LANCZOS)
    # Create the output path
    output_path = os.path.join(output_folder, os.path.basename(input_path))
    img.save(output_path)
    return output_path

# Function to clean up the relative path by removing './'
def clean_path(path):
    return path.replace("./", "/")

# Function to generate Markdown with reduced image size
def json_to_markdown_with_low_res_images(data, temp_dir):
    markdown_output = f"# Accuracy: {data['accuracy']}\n\n"
    
    for result in data['results']:
        # Process main image
        low_res_main_img = clean_path(reduce_image_size(result['img_name'], temp_dir))
        markdown_output += f"## Image: {low_res_main_img}\n"
        markdown_output += f"- **Ground Truth:** {result['ground truth']}\n"
        markdown_output += f"- **LLM Response:** {result['llm respond']}\n"
        markdown_output += "- **Record Data:**\n"
        
        for txt, score in zip(result['record_data']['txt'], result['record_data']['score']):
            markdown_output += f"  - {txt}: {score}\n"

        # Process and display record_data['img'] images side by side
        markdown_output += "- **Images (Side by Side):**\n"
        markdown_output += "<div style='display: flex;'>\n"
        for img in result['record_data']['img']:
            low_res_img = clean_path(reduce_image_size(img, temp_dir))
            markdown_output += f"  <img src='{low_res_img}' width='100' height='100' style='margin-right: 10px;'>\n"
        markdown_output += "</div>\n"
        markdown_output += "\n"

        # Process original image
        low_res_org_img = clean_path(reduce_image_size(result['record_data']['org'], temp_dir))
        markdown_output += f"- **Original Image:** ![]({low_res_org_img})\n"
        markdown_output += f"- **Outputs:** {result['record_data']['outputs']}\n"
        markdown_output += f"- **Correct:** {result['correct']}\n\n"
    
    return markdown_output

# Function to handle the entire process: generate low-res images and markdown
def generate_markdown_with_temp_images(json_data, temp_dir):
    # Create a temporary directory
    # temp_dir = tempfile.mkdtemp()

    # try:
    # Generate markdown with low-res images
    markdown_result = json_to_markdown_with_low_res_images(json_data, temp_dir)
    return markdown_result
    # finally:
    #     # Clean up temporary directory after use
    #     shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # split img
    # img_path = "./data/DR/multidr/BRSET_img00149.jpg"
    # tmp_path = "./data/tmp/"
    # sub_imgs = split_image(img_path, tmp_path, 2, 2)
    # print(sub_imgs)
    # delete_images(tmp_path)
    
    # 读取JSON文件
    with open("./output/DR_rag.json", "r") as file:
        data = json.load(file)

    # 生成Markdown文档
    markdown_document = generate_markdown_with_temp_images(data, "./data/tmp/low_res")

    # 将Markdown文档写入文件
    with open("./output/DR_rag.md", "w") as file:
        file.write(markdown_document)