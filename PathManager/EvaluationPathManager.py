from pathlib import Path
from abc import ABC
from PathManager.BasePathManager import BasePathConfig, BasePathManager

class EvaluationPathConfig(BasePathConfig):
    def __init__(self):
        super.__init__()
        self.evaluation_save_dir = Path.joinpath(self.root_path, "output")

class EvaluationPathManager(BasePathManager, Path):
    def __init__(self):
        super().__init__()
        self.config = EvaluationPathConfig()
        
    def get_saving_dirs(
        self,
        dataset_name: str,
        model_name: str,
    ) -> dict:
        """get dirs needed in saving results

        Args:
            dataset_name (str): like DR, MultiModalVQA etc.
            model_name (str): like LLava, InternVL etc.

        Returns:
            dict: log, metrics, result -> dirs,
        """
        saving_dir = self.joinpath(self.config.evaluation_save_dir, dataset_name, model_name)
        log_dir = self.joinpath(saving_dir, "log")
        result_dir = self.joinpath(saving_dir, "result")
        return {
            "log": log_dir,
            "metrics": saving_dir,
            "result": result_dir,
        }
        
    def get_saving_path(
        self,
        saving_dir: dict,
        emb_name: str,
        m: int=1,
        n: int=1,
        t_check: float=-1,
        t_filter: float=-1,
        test_num: int=-1,
        sheet_names: list[str] = ["CFP"],
        test_times: int=0
    ) -> dict:
        """get saving path of file with suffix

        Args:
            saving_dir (dict): log, metrics, result -> dirs,
            emb_name (str): like MultiDiseaseCFP etc.
            m (int, optional): img split m. Defaults to 1.
            n (int, optional): img split n. Defaults to 1.
            t_check (float, optional): check prob threshold. Defaults to -1.
            t_filter (float, optional): filter prob threshold. Defaults to -1.
            test_num (int, optional): test number. Defaults to -1.
            sheet_names (list[str], optional): modality list. Defaults to ["CFP"].
            test_times (int, optional): retest times. Defaults to 0.

        Returns:
            dict: log, metrics, result -> paths
        """
        sheet_names_str = "_".join(sheet_names)
        save_tmp=Path(f"{emb_name}_${m}_${n}_rag_${test_num}_filter_${t_filter}_check_${t_check}_modality_${sheet_names_str}_test_{test_times}")
        log_path = self.joinpath(saving_dir["log"], save_tmp.with_suffix(".log"))
        metrics_path = self.joinpath(saving_dir["log"], save_tmp.with_suffix(".csv"))
        result_path = self.joinpath(saving_dir["log"], save_tmp.with_suffix(".json"))
        return {
            "log": log_path,
            "metrics": metrics_path,
            "result": result_path
        }
        
        