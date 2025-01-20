from abc import ABC, classmethod

class BaseContextConfig(ABC):
    pass

class BaseContextFormer(ABC):
    def __init__(self):
        pass
    
    def convert_abbreviation_to_full_name(abbreviation: str, diagnosis_mapping:dict):
        """
        将输入的简称转换为对应的全称。

        :param abbreviation: 输入的简称
        :return: 对应的全称
        """
        
        # 转换为全称
        full_name = diagnosis_mapping.get(abbreviation, abbreviation)
        
        return full_name
    
    def convert_full_name_to_abbreviation(full_name: str, diagnosis_mapping: dict):
        """
        将输入的全称转换为对应的简称。

        :param full_name: 输入的全称
        :return: 对应的简称
        """
        # 反转映射表
        reverse_mapping = {v: k for k, v in diagnosis_mapping.items()}
        
        # 转换为简称
        abbreviation = reverse_mapping.get(full_name, full_name)
        
        return abbreviation