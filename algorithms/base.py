from abc import ABC, abstractmethod
import pandas as pd


class BaseScheduler(ABC):
    """
    所有排班算法的抽象基底類別。

    子類別只需實作 run() 方法，接收解析後的工程師與需求 DataFrame，
    回傳 (grid_df, penalty_breakdown) tuple。
    """

    name: str = "未命名算法"

    @abstractmethod
    def run(
        self,
        staff_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        callback=None,
    ) -> tuple:
        """
        執行排班算法。

        Parameters
        ----------
        staff_df : pd.DataFrame
            工程師資料，欄位包含：
            - 人員          : str  工程師名稱
            - primary_group : str  主要班別群組代碼（D / E / N）
            - backup_groups : list[str]  可 backup 的群組代碼清單
            - Date_1 ~ Date_30 : str  預設狀態（'' = 正常、'O' = 休假、'D/E/N' = 指定班別）

        demand_df : pd.DataFrame
            班別需求，欄位包含：
            - Date      : str   日期索引（Date_1 … Date_30）
            - IfWeekend : bool  是否為週末
            - Day       : int   日班需求人數
            - Afternoon : int   午班需求人數
            - Night     : int   晚班需求人數

        callback : callable, optional
            進度回報函數，簽章為 callback(gen, total, fitness)。

        Returns
        -------
        tuple[pd.DataFrame, dict]
            grid_df :
                工程師 × 日期 的 grid，index = 人員，columns = Date_1…Date_30，
                值為班別代碼（'D'=日班、'E'=午班、'N'=晚班）或 'O'（休假）或 ''（排休）。
            penalty_breakdown : dict
                各懲罰項目的分項與合計，鍵值包含：
                total, demand, consecutive_6, transition,
                default_shift, rest_blocks, monthly_off,
                weekend_off, single_off
        """
        raise NotImplementedError
