import pandas as pd
import os, json, shutil
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

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
                json_data.append({'imid': imid, 'dis': dis})

        # 保存JSON文件
        json_file_path = os.path.join(tgt_dir, 'level.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

# Usage Example
if __name__ == "__main__":
    # Specify the path to your CSV and image directory
    csv_file = './data/DR/multidr.csv'
    image_dir = './data/DR/multidr'

    # Define transformations if needed
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to desired size
        transforms.ToTensor(),            # Convert image to tensor
    ])

    # Create dataset and dataloader
    dataset = DRDataset(csv_file=csv_file, image_dir=image_dir, transform=transform)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # # Iterate through the dataloader
    # for images, diagnoses in dataloader:
    #     print(images, diagnoses)
    dataset.get_sample("./data/level/")