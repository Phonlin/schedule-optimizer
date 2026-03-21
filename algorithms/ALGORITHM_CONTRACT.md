# 新算法開發規範

本文件說明在此排班系統中實作新算法所必須遵守的介面契約、資料格式與注意事項。

---

## 快速開始

### 步驟 1：新建算法檔案

在 `algorithms/` 資料夾中建立 `my_algo.py`，繼承 `BaseScheduler`：

```python
# algorithms/my_algo.py
import pandas as pd
from algorithms.base import BaseScheduler

class MyAlgoScheduler(BaseScheduler):
    name = "我的算法名稱"   # 顯示在網頁下拉選單

    def run(
        self,
        staff_df: pd.DataFrame,
        demand_df: pd.DataFrame,
        callback=None,          # 必須接受此參數（可不使用）
    ) -> tuple:
        ...
        return grid_df, penalty_breakdown
```

### 步驟 2：在 Registry 登記

編輯 `algorithms/__init__.py`，加入兩行：

```python
from algorithms.my_algo import MyAlgoScheduler

REGISTRY = {
    "genetic": GeneticScheduler(),
    "my_algo": MyAlgoScheduler(),   # ← 新增
}
```

完成後重啟 Flask，網頁選單會自動出現新選項，無需改動其他檔案。

---

## 必要類別屬性

| 屬性 | 型別 | 說明 |
|---|---|---|
| `name` | `str` | 顯示於前端算法選單的名稱 |

---

## `run()` 方法簽章

```python
def run(
    self,
    staff_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    callback=None,
) -> tuple[pd.DataFrame, dict]:
```

### 參數：`staff_df`（工程師清單）

由 `app.py` 的 `parse_engineer_list()` 解析後傳入，保證包含以下欄位：

| 欄位名稱 | 型別 | 說明 |
|---|---|---|
| `人員` | `str` | 工程師姓名（唯一識別） |
| `班別群組` | `str` | 原始群組字串（如 `"DN"` 代表主D、可backup N） |
| `primary_group` | `str` | 主要班別代碼：`D`、`E`、`N` |
| `backup_groups` | `list[str]` | 可 backup 的班別代碼清單 |
| `Date_1` … `Date_30` | `str` | 每日預設狀態（見下表） |

**每日欄格值語義：**

| 值 | 意義 |
|---|---|
| `""` (空字串) | 無預設，由算法自由排班 |
| `"O"` | 休假，**不可排任何班別** |
| `"D"` | 預先指定日班，算法應**鎖定**此格 |
| `"E"` | 預先指定午班，算法應**鎖定**此格 |
| `"N"` | 預先指定晚班，算法應**鎖定**此格 |

### 參數：`demand_df`（班別需求）

由 `app.py` 的 `parse_shift_demand()` 解析後傳入，保證包含以下欄位：

| 欄位名稱 | 型別 | 說明 |
|---|---|---|
| `Date` | `str` | 日期索引（`Date_1` … `Date_30`） |
| `IfWeekend` | `bool` | 是否為週末 |
| `Day` | `int` | 日班需求人數 |
| `Afternoon` | `int` | 午班需求人數 |
| `Night` | `int` | 晚班需求人數 |

### 參數：`callback`（進度回報，選用）

算法執行過程中，**每隔一段時間**應呼叫此函數以更新網頁進度條。  
若不使用，直接忽略即可（但使用者在等待期間不會看到任何進度）。

```python
# callback 簽章
callback(
    gen: int,       # 目前已完成的代數（或步驟數）
    total: int,     # 總代數（或步驟數）
    fitness: float, # 目前最佳分數（越低越好）
)

# 呼叫範例
if callback:
    callback(current_step, total_steps, best_score)
```

---

## 回傳值規範

`run()` 必須回傳 `(grid_df, penalty_breakdown)` tuple。

### `grid_df`：排班結果 DataFrame

```
index   = 人員名稱（與 staff_df["人員"] 相同）
columns = Date_1, Date_2, …, Date_30（加上最前面的「班別群組」欄）
values  = 班別代碼（見下表）
```

| 格值 | 意義 |
|---|---|
| `"D"` | 日班 |
| `"E"` | 午班 |
| `"N"` | 晚班 |
| `"O"` | 休假（對應輸入的 O 格） |
| `""` (空字串) | 排休（算法決定休息） |

> **注意**：`"班別群組"` 欄必須存在且為第一欄（從 `staff_df` 帶入即可）。

**建構範例：**

```python
result_df = pd.DataFrame(grid_dict, index=dates).T
result_df.index.name = "人員"
result_df.columns = dates
group_map = dict(zip(staff_df["人員"], staff_df["班別群組"]))
result_df.insert(0, "班別群組", result_df.index.map(group_map))
```

### `penalty_breakdown`：懲罰明細 dict

必須包含以下所有鍵（值為 `float`，四捨五入至小數點後 2 位）：

| 鍵 | 對應懲罰規則 | 權重參考 |
|---|---|---|
| `total` | 所有分項總和 | — |
| `demand` | 需求人數未達標 | 50 / 缺人 |
| `consecutive_6` | 連續上班 ≥6 天 | 1.0 / 次 |
| `transition` | 班別轉換違規（N→D / N→E / E→D） | 1.0 / 次 |
| `default_shift` | 違反預設班別（排在非主要班別） | 0.2 / 人天 |
| `rest_blocks` | 連續休假段數 < 2 | 0.1 / 缺段 |
| `monthly_off` | 月休 < 9 天 | 0.1 / 缺天 |
| `weekend_off` | 周末休 < 4 天 | 0.1 / 缺天 |
| `single_off` | 單休1日（孤立的單日排休） | 0.1 / 次 |

> 若你的算法有額外的懲罰指標，可在 dict 中加入自訂鍵（不影響現有顯示）。

**建構範例：**

```python
penalty_breakdown = {
    "total":         round(total, 2),
    "demand":        round(p_demand, 2),
    "consecutive_6": round(p_consec6, 2),
    "transition":    round(p_trans, 2),
    "default_shift": round(p_default, 2),
    "rest_blocks":   round(p_rest_blk, 2),
    "monthly_off":   round(p_monthly, 2),
    "weekend_off":   round(p_weekend, 2),
    "single_off":    round(p_single, 2),
}
```

---

## 硬性約束（必須遵守）

1. **O 格不可排班**：`staff_df` 中值為 `"O"` 的格子代表休假，算法**不得**將其改為工作班別。
2. **預填班別格鎖定**：值為 `"D"`/`"E"`/`"N"` 的格子為事先指定，算法**不得**更改。
3. **每人每天最多一個班別**：同一工程師在同一天只能有一個基因值（日班/午班/晚班/休息擇一）。
4. **班別代碼必須為規定值**：`grid_df` 的格值只允許 `"D"`, `"E"`, `"N"`, `"O"`, `""` 五種，不可使用其他字串。

---

## 軟性規則（建議但非強制）

這些規則不強制由算法保證，但若違反會反映在 `penalty_breakdown` 中，影響優化目標：

- 每位工程師每月休假 ≥ 9 天
- 每位工程師每月周末休假 ≥ 4 天
- 不出現連續上班 ≥ 6 天
- 不出現無效班別轉換（N→D、N→E、E→D）
- 每位工程師整月至少有 2 段連續休假期
- 避免孤立的單日排休

---

## 完整最小範例

```python
# algorithms/simple_greedy.py
import pandas as pd
from algorithms.base import BaseScheduler

class SimpleGreedyScheduler(BaseScheduler):
    name = "簡單貪婪算法"

    def run(self, staff_df, demand_df, callback=None):
        engineers = staff_df["人員"].tolist()
        dates = demand_df["Date"].tolist()

        # 建立空的 grid（全部排休）
        grid = {eng: [""] * len(dates) for eng in engineers}

        # 鎖定固定格
        for _, row in staff_df.iterrows():
            eng = row["人員"]
            for di, date in enumerate(dates):
                cell = str(row.get(date, "")).strip().upper()
                if cell in ("O", "D", "E", "N"):
                    grid[eng][di] = cell

        # 你的排班邏輯 ...
        # 每處理 10% 就呼叫 callback
        total_steps = len(dates)
        for step, date in enumerate(dates):
            # ... 排班邏輯 ...
            if callback and step % max(1, total_steps // 10) == 0:
                callback(step + 1, total_steps, 0.0)

        # 建立 grid_df
        result_df = pd.DataFrame(grid, index=dates).T
        result_df.index.name = "人員"
        result_df.columns = dates
        group_map = dict(zip(staff_df["人員"], staff_df["班別群組"]))
        result_df.insert(0, "班別群組", result_df.index.map(group_map))

        # 回傳空的懲罰明細（請填入實際計算值）
        penalty_breakdown = {
            "total": 0.0, "demand": 0.0, "consecutive_6": 0.0,
            "transition": 0.0, "default_shift": 0.0, "rest_blocks": 0.0,
            "monthly_off": 0.0, "weekend_off": 0.0, "single_off": 0.0,
        }
        return result_df, penalty_breakdown
```

---

## 班別代碼對照表

| 代碼 | 中文 | 對應 Shift_Demand 欄位 |
|---|---|---|
| `D` | 日班 | `Day` |
| `E` | 午班 | `Afternoon` |
| `N` | 晚班 | `Night` |
| `O` | 休假 | — |
| `""` | 排休 | — |
