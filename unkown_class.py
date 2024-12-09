import shutil, os
from tqdm import tqdm
from dataset import MultiModalVQADataset, EyeImageDataset

class DiagnosisImageCopier:
    def __init__(self, dataset, target_dir='classic images'):
        """
        Args:
            dataset (EyeImageDataset): An instance of the EyeImageDataset class.
            target_dir (string): The directory where to copy the images. Default is 'classic images'.
        """
        self.dataset = dataset
        self.target_dir = target_dir
        
        # Ensure the target directory exists
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

    def copy_images_for_diagnoses(self, d_list, n=2):
        """
        Args:
            d_list (list): A list of diagnoses to copy images for.
            n (int): Maximum number of entries to copy per diagnosis. Default is 2.
        """
        for diagnosis in d_list:
            # Get the entries for the current diagnosis with a limit of n
            entries = self.dataset.get_entries_by_diagnosis(diagnosis, n=n)
            
            # Create subdirectory for this diagnosis if it doesn't exist
            diag_dir = os.path.join(self.target_dir, diagnosis)
            if not os.path.exists(diag_dir):
                os.makedirs(diag_dir)
                
            # Copy each image to the corresponding diagnosis subdirectory
            for entry in entries:
                img_name = entry['img_name']
                img_path = os.path.join(eye_dataset.img_dir, img_name)
                base_name = os.path.basename(img_path)
                target_path = os.path.join(diag_dir, base_name)
                
                try:
                    shutil.copy(img_path, target_path)
                    print(f"Copied {img_path} to {target_path}")
                except IOError as e:
                    print(f"Unable to copy file {img_path}. Error: {e}")
                    
if __name__ == "__main__":
    root_path = "/home/hongyu/"
    cur_path = "/home/hongyu/Visual-RAG-LLaVA-Med"    
    csv_file = root_path +'alldataset/cleaned_full.csv'
    img_dir = root_path + 'alldataset/images'
    eye_dataset = EyeImageDataset(csv_file=csv_file, img_dir=img_dir)

    print(eye_dataset.get_entries_by_diagnosis("macular hole", 2))
    copier = DiagnosisImageCopier(eye_dataset, target_dir='./data/Classic Images')
    
    # 定义诊断列表
    d_list = ["macular hole"]  # 示例诊断列表
    
    # 复制图像
    copier.copy_images_for_diagnoses(d_list)