from abc import ABC

class BaseContextConfig(ABC):
    pass

class BaseContextFormer(ABC):
    def __init__(self):
        pass
    
    def build_prompt(
        self,
        query: str,
        level_context: str = None,
        crop_lesion_context: str = None,
        diagnosis_context: str = None,
        diagnosis_standard: str = None,
    ):
        parts = []
        if diagnosis_standard:
            parts.append(f"Diagnosing Standard: {diagnosis_standard}\n")
        if level_context:
            parts.append(
                f"The possible diagnosing level and similarity: {level_context}\n"
            )
        if crop_lesion_context:
            parts.append(f"The possible lesion and similarity: {crop_lesion_context}\n")
        if diagnosis_context:
            parts.append(
                f"The possible diagnosis and similarity: {diagnosis_context}\n"
            )
        parts.append(query)

        return "".join(parts)
    
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