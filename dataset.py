import pandas as pd
import os, json, shutil
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
from collections import Counter, OrderedDict
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
    
    def get_entries_by_diagnosis(self, diagnosis, n=None):
        """
        Args:
            diagnosis (string): The diagnosis to filter the dataset entries by.
            n (int, optional): Maximum number of entries to return. If None, all matching entries are returned.

        Returns:
            list: A list of dictionaries, each representing a sample with the specified diagnosis and img_name.
        """
        # Filter annotations DataFrame by the given diagnosis using the correct column name
        filtered_df = self.annotations[self.annotations.iloc[:, 1] == diagnosis]

        # Apply limit if n is specified and valid
        if n is not None and isinstance(n, int) and n > 0:
            filtered_df = filtered_df.head(n)

        # Add 'img_name' column to the filtered DataFrame
        
        filtered_df['img_name'] = filtered_df.iloc[:, 0].astype(str) + '.jpg'

        # Create a list of dictionaries containing 'img_name' and 'diagnosis'
        entries = filtered_df[['img_name']].to_dict(orient='records')
    
        return entries

class MultiModalVQADataset(Dataset):
    def __init__(self, excel_file, root_dir, transform=None, sheet_names = ["CFP", "FFA", "ultrasound", "OCT", "slitlamp"]):
        """
        Args:
            excel_file (string): Path to the Excel file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations_df = pd.read_excel(excel_file, sheet_name=None)  # Read all sheets
        self.root_dir = root_dir
        self.transform = transform
        self.annotations_df = pd.read_excel(excel_file, sheet_name=None)  # Read all sheets
        self.root_dir = root_dir
        self.transform = transform
        
        # Use an OrderedDict to preserve insertion order and remove duplicates based on img_path
        samples_dict = OrderedDict()
        
        for sheet_name, df in self.annotations_df.items():
            # select according to sheet_name
            if sheet_name not in sheet_names:
                print("ignore " + sheet_name + " modal")
                continue
            for index, row in df.iterrows():
                img_path = os.path.join(self.root_dir, sheet_name, row['Diagnosis'], f"{row['Case number']}.jpg")
                if os.path.exists(img_path) and img_path not in samples_dict:
                    samples_dict[img_path] = {
                        'img_path': img_path,
                        'diagnosis': row['Diagnosis']
                    }
        
        # Convert the dictionary back into a list of dictionaries
        self.samples = list(samples_dict.values())

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
    
    def get_entries_by_diagnosis(self, diagnosis):
        """
        Args:
            diagnosis (string): The diagnosis to filter the dataset entries by.

        Returns:
            list: A list of dictionaries, each representing a sample with the specified diagnosis.
        """
        # Filter samples by the given diagnosis and return them as a list
        filtered_samples = [sample for sample in self.samples if sample['diagnosis'] == diagnosis]
        
        return filtered_samples
    
class DiagnosisCounter:
    def __init__(self, dataset):
        """
        Args:
            dataset (MultiModalVQADataset): The dataset to analyze.
        """
        self.dataset = dataset
        self.diagnosis_counts = None
        
    def count_diagnoses(self):
        """Count the number of unique diagnoses in the dataset."""
        diagnoses = [sample['diagnosis'] for sample in self.dataset.samples]
        self.diagnosis_counts = Counter(diagnoses)
        
    def print_diagnoses(self):
        """Print the list of unique diagnoses and their counts."""
        if self.diagnosis_counts is None:
            self.count_diagnoses()
        
        print("Unique diagnoses in the dataset:")
        for diagnosis, count in self.diagnosis_counts.items():
            print(f"{diagnosis}: {count}")
            
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
    sheet_names = ["CFP"]
    dataset = MultiModalVQADataset(excel_file, data_dir, sheet_names=sheet_names)
    
    # 获取第一个样本
    # image, diagnosis = dataset[0]
    # print(f"Image path: {image}, Diagnosis: {diagnosis}")
    # print(len(dataset))
    
    # 计算diagnosis种类
    diagnosis_counter = DiagnosisCounter(dataset)
    diagnosis_counter.print_diagnoses()