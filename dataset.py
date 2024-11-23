import pandas as pd
import os, json, shutil
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class DRDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = f"{self.image_dir}/{self.data_frame.iloc[idx, 0]}.jpg"  # Adjust file extension if needed
        # image = Image.open(img_name)
        diagnosis = self.data_frame.iloc[idx, 1]
        
        # if self.transform:
        #     image = self.transform(image)

        return img_name, diagnosis
    
    def get_sample(self, tgt_dir):
         # 读取CSV文件
        df = self.data_frame

        # 获取每种dis的随机两条数据
        selected_data = df.groupby('dis').apply(lambda x: x.sample(n=2, random_state=1)).reset_index(drop=True)

        # 创建目标目录，如果不存在
        if not os.path.exists(tgt_dir):
            os.makedirs(tgt_dir)

        # 准备JSON数据
        json_data = []

        for index, row in selected_data.iterrows():
            print(row)
            imid = row['imid']
            dis = row['dis']
            
            # 假设图像文件名为imid.jpg
            img_src = f'./data/DR/multidr/{imid}.jpg'
            img_dst = os.path.join(tgt_dir, f'{imid}.jpg')
            
            # 复制图像文件到目标目录
            if os.path.exists(img_src):
                shutil.copy(img_src, img_dst)

                # 添加到JSON数据
                json_data.append({'image_path': img_dst, 'imid': imid, 'dis': dis})

        # 保存JSON文件
        json_file_path = os.path.join(tgt_dir, 'level.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)


class EyeImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0] + '.jpg')
        # image = read_image(img_path)
        diagnosis = self.annotations.iloc[idx, 1]
        finding = self.annotations.iloc[idx, 1]
        modality = self.annotations.iloc[idx, 3]
        dataset = self.annotations.iloc[idx, 4]
        caption = self.annotations.iloc[idx, 5]

        # if self.transform:
        #     image = self.transform(image)

        return img_path, diagnosis,
    # finding, modality, dataset, caption

class MultiModalVQADataset(Dataset):
    def __init__(self, excel_file, root_dir, transform=None):
        """
        Args:
            excel_file (string): Path to the Excel file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations_df = pd.read_excel(excel_file, sheet_name=None)  # Read all sheets
        self.root_dir = root_dir
        self.transform = transform
        
        # Flatten the DataFrame to a single list of dictionaries
        self.samples = []
        for sheet_name, df in self.annotations_df.items():
            for index, row in df.iterrows():
                img_path = os.path.join(self.root_dir, sheet_name, row['Diagnosis'], f"{row['Case number']}.jpg")
                if os.path.exists(img_path):
                    self.samples.append({
                        'img_path': img_path,
                        'diagnosis': row['Diagnosis']
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples[idx]
        img_path = sample['img_path']
        diagnosis = sample['diagnosis']
        
        # Load image
        image = read_image(img_path)
        
        if self.transform:
            image = self.transform(image)

        return img_path, diagnosis
    
# Usage Example
if __name__ == "__main__":
    # Specify the path to your CSV and image directory
    # csv_file = './data/DR/multidr.csv'
    # image_dir = './data/DR/multidr'

    # # Define transformations if needed
    # transform = transforms.Compose([
    #     transforms.Resize((256, 256)),  # Resize to desired size
    #     transforms.ToTensor(),            # Convert image to tensor
    # ])

    # # Create dataset and dataloader
    # dataset = DRDataset(csv_file=csv_file, image_dir=image_dir, transform=transform)
    # # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # # # Iterate through the dataloader
    # # for images, diagnoses in dataloader:
    # #     print(images, diagnoses)
    # dataset.get_sample("./data/level/")
    root_path = "/home/hongyu/"
    # 使用示例
    # csv_file = root_path +'alldataset/cleaned_full.csv'
    # img_dir = root_path + 'alldataset/images'
    # eye_dataset = EyeImageDataset(csv_file=csv_file, img_dir=img_dir)

    # # 获取第一个样本
    # image,diagnosis, finding, modality, dataset, caption = eye_dataset[0]
    # print(f"Image path: {image}, diagnosis: {diagnosis}, Finding: {finding}, Modality: {modality}")
    
    excel_file = root_path + "Visual-RAG-LLaVA-Med/data/"+ 'Multimodal VQA Dataset/Multimodal VQA dataset_1015.xlsx'
    data_dir = root_path + "Visual-RAG-LLaVA-Med/data/" + 'Multimodal VQA Dataset'
    
    # 创建数据集实例
    dataset = MultiModalVQADataset(excel_file, data_dir)
    
    # 获取第一个样本
    image, diagnosis = dataset[0]
    print(f"Image path: {image}, Diagnosis: {diagnosis}")
    print(len(dataset))