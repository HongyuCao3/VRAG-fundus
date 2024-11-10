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
    
def merge_dicts(dicts):
    from collections import defaultdict
    
    # 使用defaultdict来自动处理缺失的键
    merged = defaultdict(lambda: {'score': 0, 'imgs': [], 'metadata': []})
    
    for d in dicts:
        for txt, score, img, meta in zip(d['txt'], d['score'], d['img'], d['metadata']):
            merged[txt]['score'] += score
            merged[txt]['imgs'].append(img)
            merged[txt]['metadata'].append(meta)
    
    # 将defaultdict转换为普通字典
    result = {
        'txt': [],
        'score': [],
        'img': [],
        'metadata': []
    }
    
    for txt, data in merged.items():
        result['txt'].append(txt)
        result['score'].append(data['score'])
        result['img'].extend(data['imgs'])
        result['metadata'].extend(data['metadata'])
    
    return result

def find_longest_matching_class(llm_respond, classes):
    """
    在classes列表中查找与字符串llm_respond匹配的最长类名。
    
    :param llm_respond: 需要匹配的字符串
    :param classes: 包含类名的列表
    :return: 匹配的最长类名，如果没有匹配则返回None
    """
    longest_match = None
    max_length = 0

    for class_name in classes:
        if class_name in llm_respond and len(class_name) > max_length:
            longest_match = class_name
            max_length = len(class_name)
    if longest_match == None:
        print("no matching")
    return longest_match

def find_json_file(folder):
    """
    查找指定文件夹中的JSON文件。
    
    :param folder: 要查找的文件夹路径
    :return: JSON文件的路径，如果未找到则返回None
    """
    for filename in os.listdir(folder):
        if filename.endswith('.json'):
            return os.path.join(folder, filename)
    return None

def find_longest_diagnosis_keys(response, diagnosing_level):
    # 将response转换为小写以确保匹配时不区分大小写
    response_lower = response.lower()
    # 按键的长度降序排序
    sorted_keys = sorted(diagnosing_level.keys(), key=lambda x: len(x), reverse=True)
    # 初始化结果列表
    result = []
    # 遍历排序后的键
    for key in sorted_keys:
        # 将键也转换为小写进行比较
        key_lower = key.lower()
        if key_lower in response_lower:
            # 如果找到匹配，则添加到结果列表中
            result.append(key)
            # 从response中移除已匹配的部分，避免后续误匹配
            response_lower = response_lower.replace(key_lower, '')
            # 由于只匹配一次，可以在这里跳出内层循环
            break
    return result

def convert_abbreviation_to_full_name(abbreviation):
    """
    将输入的简称转换为对应的全称。

    :param abbreviation: 输入的简称
    :return: 对应的全称
    """
    # 定义映射表
    diagnosis_mapping = {
        "Normal": "Normal",
        "Mild NPDR": "mild nonproliferative diabetic retinopathy",
        "Moderate NPDR": "moderate nonproliferative diabetic retinopathy",
        "Severe NPDR": "severe nonproliferative diabetic retinopathy",
        "PDR": "proliferative diabetic retinopathy"
    }
    
    # 转换为全称
    full_name = diagnosis_mapping.get(abbreviation, abbreviation)
    
    return full_name

if __name__ == "__main__":
    # split img
    # img_path = "./data/DR/multidr/BRSET_img00149.jpg"
    # tmp_path = "./data/tmp/"
    # sub_imgs = split_image(img_path, tmp_path, 2, 2)
    # print(sub_imgs)
    # delete_images(tmp_path)
    
    # # 读取JSON文件
    # with open("./output/DR_rag.json", "r") as file:
    #     data = json.load(file)

    # # 生成Markdown文档
    # markdown_document = generate_markdown_with_temp_images(data, "./data/tmp/low_res")

    # # 将Markdown文档写入文件
    # with open("./output/DR_rag.md", "w") as file:
    #     file.write(markdown_document)
    
    # dicts = [
    # {'txt': ['exudates', 'hemorrhage', 'microaneurysm'], 
    #  'score': [0.6606726254525561, 0.6223304584308493, 0.6112649314480555], 
    #  'img': ['./data/lesion/IDRiD_49/exudates.png', './data/lesion/IDRiD_49/hemorrhage.png', './data/lesion/20051020_64945_0100_PP/microaneurysm.png'], 
    #  'metadata': [{}, {}, {}]},
    # {'txt': ['exudates', 'hemorrhage'], 
    #  'score': [0.3393273745474439, 0.3776695415691507], 
    #  'img': ['./data/lesion/IDRiD_50/exudates.png', './data/lesion/IDRiD_50/hemorrhage.png'], 
    #  'metadata': [{'key': 'value'}, {'key2': 'value2'}]}
    # ]

    # # 调用函数
    # merged_dict = merge_dicts(dicts)

    # # 输出结果
    # print(merged_dict)
    llm_respond = "The model predicted severe proliferative diabetic retinopathy."
    classes = ["Normal", "moderate nonproliferative diabetic retinopathy", 
            "severe nonproliferative diabetic retinopathy", "proliferative diabetic retinopathy"]

    matched_class = find_longest_matching_class(llm_respond, classes)
    print(f"Longest matching class: {matched_class}")