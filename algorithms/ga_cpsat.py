import pandas as pd

from algorithms.base import BaseScheduler
from algorithms.scheduling import SchedulerGA, GAConfig, INT_TO_SHIFT


class GACpsatScheduler(BaseScheduler):
    """GA + CP-SAT 混合排班算法，包裝 SchedulerGA 使其符合 BaseScheduler 介面。"""

    name = "GA + CP-SAT 混合算法"

    def __init__(self, config: GAConfig | None = None):
        self._config = config or GAConfig()

    def run(self, staff_df: pd.DataFrame, demand_df: pd.DataFrame,
            callback=None) -> tuple:

        eng_df = self._prepare_engineer_df(staff_df)
        solver = SchedulerGA(engineer_df=eng_df, demand_df=demand_df,
                             config=self._config)

        best_schedule, breakdown, _ = solver.run(callback=callback)

        grid_df = self._build_grid_df(solver, best_schedule, staff_df)
        penalty = self._map_penalty(breakdown)
        return grid_df, penalty

    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_engineer_df(staff_df: pd.DataFrame) -> pd.DataFrame:
        """將 app.py 傳入的 staff_df 轉為 SchedulerGA 可讀的格式。

        SchedulerGA 需要「班別群組」為單一班別字母（D/E/N），
        app.py 的 staff_df 可能帶有 primary_group / backup_groups 等額外欄位。
        """
        df = staff_df.copy()
        if "班別群組" not in df.columns and "primary_group" in df.columns:
            df["班別群組"] = df["primary_group"]
        return df

    @staticmethod
    def _build_grid_df(solver: SchedulerGA, schedule,
                       staff_df: pd.DataFrame) -> pd.DataFrame:
        """numpy schedule → 合約規定的 grid_df。"""
        engineers = solver.people
        dates = solver.day_cols

        grid: dict[str, list[str]] = {}
        for ei, eng in enumerate(engineers):
            row_vals: list[str] = []
            for di in range(solver.n_days):
                val = int(schedule[ei, di])
                if val == 0:
                    if solver.fixed_mask[ei, di] and solver.fixed_values[ei, di] == 0:
                        row_vals.append("O")
                    else:
                        row_vals.append("")
                else:
                    row_vals.append(INT_TO_SHIFT[val])
            grid[eng] = row_vals

        result_df = pd.DataFrame(grid, index=dates).T
        result_df.index.name = "人員"
        result_df.columns = dates

        group_map = dict(
            zip(staff_df["人員"],
                staff_df.get("班別群組", staff_df.get("primary_group", "")))
        )
        result_df.insert(0, "班別群組", result_df.index.map(group_map))
        return result_df

    @staticmethod
    def _map_penalty(breakdown: dict) -> dict:
        """將 SchedulerGA 的 breakdown 鍵值映射為合約規定的鍵值。"""
        return {
            "total":         round(breakdown["total_fitness"], 2),
            "demand":        round(breakdown["demand_mismatch"], 2),
            "consecutive_6": round(breakdown["consecutive_6"], 2),
            "transition":    round(breakdown["transition_violation"], 2),
            "default_shift": round(breakdown["preset_violation"], 2),
            "rest_blocks":   round(breakdown["few_two_day_off_blocks"], 2),
            "monthly_off":   round(breakdown["monthly_off_lt9"], 2),
            "weekend_off":   round(breakdown["weekend_off_lt4"], 2),
            "single_off":    round(breakdown["single_off_days"], 2),
        }
