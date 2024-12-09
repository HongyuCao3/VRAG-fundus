from dataset import MultiModalVQADataset, EyeImageDataset

if __name__ == "__main__":
    root_path = "/home/hongyu/"
    
    csv_file = root_path +'alldataset/cleaned_full.csv'
    img_dir = root_path + 'alldataset/images'
    eye_dataset = EyeImageDataset(csv_file=csv_file, img_dir=img_dir)

    # 获取第一个样本
    # image,diagnosis, finding, modality, dataset, caption = eye_dataset[0]
    # print(f"Image path: {image}, diagnosis: {diagnosis}, Finding: {finding}, Modality: {modality}")
    
    # 创建数据集实例
    
    # 获取第一个样本
    # image, diagnosis = dataset[0]
    # print(f"Image path: {image}, Diagnosis: {diagnosis}")
    # print(len(dataset))
    
    # 计算diagnosis种类
    # diagnosis_counter = DiagnosisCounter(dataset)
    # diagnosis_counter.print_diagnoses()
    print(eye_dataset.get_entries_by_diagnosis("macular hole", 2))