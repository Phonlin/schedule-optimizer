"""
Gurobi MIP 參考求解：與 workshop_gurobi_result.py 相同目標與約束。
供網頁與啟發式演算法結果對照（參考最佳／時間限制內最佳）。
"""

from __future__ import annotations

import time
from typing import Any, Callable

import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None  # type: ignore[misc, assignment]
    GRB = None  # type: ignore[misc, assignment]

SHIFTS = ["D", "E", "N", "O"]

LogFn = Callable[[str], None]


def gurobi_available() -> bool:
    return gp is not None


class GurobiReferenceError(Exception):
    pass


def _primary_group_char(row: pd.Series) -> str:
    raw = str(row.get("班別群組", "") or "").strip().upper()
    if not raw:
        return "D"
    c = raw[0]
    return c if c in ("D", "E", "N") else "D"


def prepare_from_frames(
    staff_df: pd.DataFrame, demand_df: pd.DataFrame
) -> dict[str, Any]:
    num_days = len(demand_df)
    if num_days < 1:
        raise GurobiReferenceError("Shift_Demand 沒有資料列。")

    engineers = staff_df["人員"].astype(str).tolist()
    num_eng = len(engineers)
    # 與 parse_engineer_list 一致：以列順序對應索引
    default_group_list = [_primary_group_char(staff_df.iloc[i]) for i in range(num_eng)]
    default_group = dict(zip(engineers, default_group_list))

    date_cols = [f"Date_{d + 1}" for d in range(num_days)]
    for c in date_cols:
        if c not in staff_df.columns:
            raise GurobiReferenceError(f"Engineer_List 缺少欄位 {c}。")

    pre_assigned: dict[int, dict[int, str]] = {}
    for i in range(num_eng):
        row = staff_df.iloc[i]
        for d, col in enumerate(date_cols):
            val = str(row[col]).strip() if pd.notna(row[col]) else ""
            val = val.upper()
            if val in ("D", "E", "N", "O"):
                pre_assigned.setdefault(i, {})[d] = val

    demand_D = demand_df["Day"].astype(int).tolist()
    demand_E = demand_df["Afternoon"].astype(int).tolist()
    demand_N = demand_df["Night"].astype(int).tolist()

    if demand_df["IfWeekend"].dtype == bool:
        is_weekend = demand_df["IfWeekend"].tolist()
    else:
        is_weekend = [
            str(v).strip().upper() == "Y" for v in demand_df["IfWeekend"].tolist()
        ]
    weekend_days = [d for d in range(num_days) if is_weekend[d]]

    return {
        "num_days": num_days,
        "num_eng": num_eng,
        "engineers": engineers,
        "default_group": default_group,
        "default_group_list": default_group_list,
        "pre_assigned": pre_assigned,
        "demand_D": demand_D,
        "demand_E": demand_E,
        "demand_N": demand_N,
        "is_weekend": is_weekend,
        "weekend_days": weekend_days,
        "date_cols": date_cols,
    }


def compute_breakdown(
    schedule: list[list[str]],
    engineers: list[str],
    default_group: dict[str, str],
    is_weekend: list[bool],
    num_days: int,
) -> dict[str, int]:
    num_eng = len(engineers)
    bd = {
        k: 0
        for k in [
            "consecutive_6days",
            "night_to_day_aft",
            "aft_to_day",
            "day_aft_to_night",
            "violate_default",
            "consec_off_lt2",
            "total_off_lt9",
            "weekend_off_lt4",
            "isolated_off",
        ]
    }
    for i in range(num_eng):
        s = schedule[i]
        grp = default_group[engineers[i]]
        consec = 0
        for d in range(num_days):
            if s[d] != "O":
                consec += 1
                bd["consecutive_6days"] += 1 if consec >= 6 else 0
            else:
                consec = 0
        for d in range(num_days - 1):
            t, n = s[d], s[d + 1]
            if t == "N" and n in ("D", "E"):
                bd["night_to_day_aft"] += 1
            if t == "E" and n == "D":
                bd["aft_to_day"] += 1
            if t in ("D", "E") and n == "N":
                bd["day_aft_to_night"] += 1
        for d in range(num_days):
            if s[d] != "O" and s[d] != grp:
                bd["violate_default"] += 1
        off_blocks: list[int] = []
        in_b = False
        bl = 0
        for d in range(num_days):
            if s[d] == "O":
                in_b = True
                bl += 1
            else:
                if in_b:
                    off_blocks.append(bl)
                    in_b = False
                    bl = 0
        if in_b:
            off_blocks.append(bl)
        coc = len(off_blocks)
        if coc < 2:
            bd["consec_off_lt2"] += 2 - coc
        to = s.count("O")
        if to < 9:
            bd["total_off_lt9"] += 9 - to
        wo = sum(1 for d in range(num_days) if is_weekend[d] and s[d] == "O")
        if wo < 4:
            bd["weekend_off_lt4"] += 4 - wo
        for d in range(1, num_days - 1):
            if s[d] == "O" and s[d - 1] != "O" and s[d + 1] != "O":
                bd["isolated_off"] += 1
    return bd


def weighted_total_penalty(bd: dict[str, int]) -> float:
    return (
        bd["consecutive_6days"] * 1.0
        + bd["night_to_day_aft"] * 1.0
        + bd["aft_to_day"] * 1.0
        + bd["day_aft_to_night"] * 1.0
        + bd["violate_default"] * 0.2
        + bd["consec_off_lt2"] * 0.1
        + bd["total_off_lt9"] * 0.1
        + bd["weekend_off_lt4"] * 0.1
        + bd["isolated_off"] * 0.1
    )


def _status_name(status: int) -> str:
    names = {
        getattr(GRB, "LOADED", -1): "LOADED",
        getattr(GRB, "OPTIMAL", -2): "OPTIMAL",
        getattr(GRB, "INFEASIBLE", -3): "INFEASIBLE",
        getattr(GRB, "INF_OR_UNBD", -4): "INF_OR_UNBD",
        getattr(GRB, "UNBOUNDED", -5): "UNBOUNDED",
        getattr(GRB, "CUTOFF", -6): "CUTOFF",
        getattr(GRB, "ITERATION_LIMIT", -7): "ITERATION_LIMIT",
        getattr(GRB, "NODE_LIMIT", -8): "NODE_LIMIT",
        getattr(GRB, "TIME_LIMIT", -9): "TIME_LIMIT",
        getattr(GRB, "SOLUTION_LIMIT", -10): "SOLUTION_LIMIT",
        getattr(GRB, "INTERRUPTED", -11): "INTERRUPTED",
        getattr(GRB, "NUMERIC", -12): "NUMERIC",
        getattr(GRB, "SUBOPTIMAL", -13): "SUBOPTIMAL",
        getattr(GRB, "INPROGRESS", -14): "INPROGRESS",
    }
    return names.get(status, f"STATUS_{status}")


def solve_reference_mip(
    staff_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    *,
    time_limit: float = 60.0,
    mip_gap: float = 0.01,
    log: LogFn | None = None,
) -> dict[str, Any]:
    if not gurobi_available():
        raise GurobiReferenceError(
            "未安裝 gurobipy。請安裝 Gurobi 與對應 Python 套件並設定授權後再試。"
        )

    ctx = prepare_from_frames(staff_df, demand_df)
    num_days = ctx["num_days"]
    num_eng = ctx["num_eng"]
    engineers = ctx["engineers"]
    default_group = ctx["default_group"]
    default_group_list = ctx["default_group_list"]
    pre_assigned = ctx["pre_assigned"]
    demand_D = ctx["demand_D"]
    demand_E = ctx["demand_E"]
    demand_N = ctx["demand_N"]
    is_weekend = ctx["is_weekend"]
    weekend_days = ctx["weekend_days"]

    def L(msg: str) -> None:
        if log:
            log(msg)

    S_IDX = {s: k for k, s in enumerate(SHIFTS)}
    I = range(num_eng)
    D = range(num_days)
    S = range(len(SHIFTS))

    L("建立 Gurobi 模型…")
    m = gp.Model("TSMC_Scheduling_ref")
    m.setParam("TimeLimit", time_limit)
    m.setParam("MIPGap", mip_gap)
    m.setParam("OutputFlag", 0)

    x = m.addVars(num_eng, num_days, len(SHIFTS), vtype=GRB.BINARY, name="x")

    for i in I:
        for d in D:
            m.addConstr(
                gp.quicksum(x[i, d, s] for s in S) == 1, name=f"one_shift_{i}_{d}"
            )

    for i, days in pre_assigned.items():
        for d, sh in days.items():
            m.addConstr(x[i, d, S_IDX[sh]] == 1, name=f"preassign_{i}_{d}")

    for d in D:
        m.addConstr(
            gp.quicksum(x[i, d, S_IDX["D"]] for i in I) == demand_D[d],
            name=f"dem_D_{d}",
        )
        m.addConstr(
            gp.quicksum(x[i, d, S_IDX["E"]] for i in I) == demand_E[d],
            name=f"dem_E_{d}",
        )
        m.addConstr(
            gp.quicksum(x[i, d, S_IDX["N"]] for i in I) == demand_N[d],
            name=f"dem_N_{d}",
        )

    obj_terms = []

    z_consec6 = m.addVars(num_eng, num_days, vtype=GRB.BINARY, name="z_consec6")
    for i in I:
        for d in range(5, num_days):
            working_expr = [
                gp.quicksum(x[i, d2, s] for s in S if SHIFTS[s] != "O")
                for d2 in range(d - 5, d + 1)
            ]
            m.addConstr(
                z_consec6[i, d] >= gp.quicksum(working_expr) - 5,
                name=f"consec6_lb_{i}_{d}",
            )
            for d2 in range(d - 5, d + 1):
                m.addConstr(
                    z_consec6[i, d]
                    <= gp.quicksum(x[i, d2, s] for s in S if SHIFTS[s] != "O"),
                    name=f"consec6_ub_{i}_{d}_{d2}",
                )
            obj_terms.append(1.0 * z_consec6[i, d])

    z_trans = m.addVars(num_eng, num_days - 1, 3, vtype=GRB.BINARY, name="z_trans")
    for i in I:
        for d in range(num_days - 1):
            m.addConstr(
                z_trans[i, d, 0] >= x[i, d, S_IDX["N"]] + x[i, d + 1, S_IDX["D"]] - 1,
                name=f"trans0a_{i}_{d}",
            )
            m.addConstr(
                z_trans[i, d, 0] >= x[i, d, S_IDX["N"]] + x[i, d + 1, S_IDX["E"]] - 1,
                name=f"trans0b_{i}_{d}",
            )
            m.addConstr(
                z_trans[i, d, 1] >= x[i, d, S_IDX["E"]] + x[i, d + 1, S_IDX["D"]] - 1,
                name=f"trans1_{i}_{d}",
            )
            m.addConstr(
                z_trans[i, d, 2] >= x[i, d, S_IDX["D"]] + x[i, d + 1, S_IDX["N"]] - 1,
                name=f"trans2a_{i}_{d}",
            )
            m.addConstr(
                z_trans[i, d, 2] >= x[i, d, S_IDX["E"]] + x[i, d + 1, S_IDX["N"]] - 1,
                name=f"trans2b_{i}_{d}",
            )
            for k in range(3):
                obj_terms.append(1.0 * z_trans[i, d, k])

    z_viol_def = m.addVars(num_eng, num_days, vtype=GRB.BINARY, name="z_viol_def")
    for i in I:
        grp = default_group_list[i]
        non_grp_non_O = [s for s in S if SHIFTS[s] != "O" and SHIFTS[s] != grp]
        for d in D:
            for s in non_grp_non_O:
                m.addConstr(
                    z_viol_def[i, d] >= x[i, d, s], name=f"viol_def_{i}_{d}_{s}"
                )
            obj_terms.append(0.2 * z_viol_def[i, d])

    z_off_short = m.addVars(num_eng, lb=0, vtype=GRB.INTEGER, name="z_off_short")
    for i in I:
        total_off = gp.quicksum(x[i, d, S_IDX["O"]] for d in D)
        m.addConstr(z_off_short[i] >= 9 - total_off, name=f"off_short_{i}")
        obj_terms.append(0.1 * z_off_short[i])

    z_wk_short = m.addVars(num_eng, lb=0, vtype=GRB.INTEGER, name="z_wk_short")
    for i in I:
        wk_off = gp.quicksum(x[i, d, S_IDX["O"]] for d in weekend_days)
        m.addConstr(z_wk_short[i] >= 4 - wk_off, name=f"wk_short_{i}")
        obj_terms.append(0.1 * z_wk_short[i])

    z_iso = m.addVars(num_eng, num_days, vtype=GRB.BINARY, name="z_iso")
    for i in I:
        for d in range(1, num_days - 1):
            m.addConstr(
                z_iso[i, d]
                >= x[i, d, S_IDX["O"]]
                - x[i, d - 1, S_IDX["O"]]
                - x[i, d + 1, S_IDX["O"]],
                name=f"iso_{i}_{d}",
            )
            obj_terms.append(0.1 * z_iso[i, d])

    z_bstart = m.addVars(num_eng, num_days, vtype=GRB.BINARY, name="z_bstart")
    for i in I:
        m.addConstr(z_bstart[i, 0] == x[i, 0, S_IDX["O"]], name=f"bstart0_{i}")
        for d in range(1, num_days):
            m.addConstr(
                z_bstart[i, d] <= x[i, d, S_IDX["O"]], name=f"bstart_ub1_{i}_{d}"
            )
            m.addConstr(
                z_bstart[i, d] <= 1 - x[i, d - 1, S_IDX["O"]],
                name=f"bstart_ub2_{i}_{d}",
            )
            m.addConstr(
                z_bstart[i, d] >= x[i, d, S_IDX["O"]] - x[i, d - 1, S_IDX["O"]],
                name=f"bstart_lb_{i}_{d}",
            )

    z_consec_off_short = m.addVars(
        num_eng, lb=0, vtype=GRB.INTEGER, name="z_consec_off_short"
    )
    for i in I:
        total_blocks = gp.quicksum(z_bstart[i, d] for d in D)
        m.addConstr(
            z_consec_off_short[i] >= 2 - total_blocks, name=f"consec_off_short_{i}"
        )
        obj_terms.append(0.1 * z_consec_off_short[i])

    m.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)

    last_log = [0.0]

    def mip_callback(model: gp.Model, where: int) -> None:
        if where != GRB.Callback.MIPSOL:
            return
        now = time.time()
        if now - last_log[0] < 3.0:
            return
        last_log[0] = now
        try:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            L(f"求解中… 目前可行解目標 ≈ {obj:.4f}")
        except (gp.GurobiError, AttributeError):
            pass

    L(f"開始求解（TimeLimit={time_limit}s, MIPGap={mip_gap}）…")
    m.optimize(mip_callback)

    status = m.Status
    status_str = _status_name(status)

    if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        raise GurobiReferenceError(
            f"Gurobi 未取得可行解（狀態：{status_str}）。"
        )

    obj_val = float(m.ObjVal)
    mip_gap_val = None
    try:
        if getattr(m, "IsMIP", False):
            mip_gap_val = float(m.MIPGap)
    except (gp.GurobiError, AttributeError, TypeError, ValueError):
        pass

    schedule: list[list[str]] = []
    for i in I:
        row: list[str] = []
        for d in D:
            for s in S:
                if x[i, d, s].X > 0.5:
                    row.append(SHIFTS[s])
                    break
        schedule.append(row)

    bd = compute_breakdown(
        schedule, engineers, default_group, is_weekend, num_days
    )
    raw_penalty = weighted_total_penalty(bd)

    L(f"完成。狀態={status_str}，目標值={obj_val:.4f}")

    return {
        "gurobi_obj": obj_val,
        "gurobi_status": status_str,
        "mip_gap": mip_gap_val,
        "breakdown": bd,
        "raw_penalty_from_breakdown": raw_penalty,
        "schedule": schedule,
        "engineers": engineers,
        "num_days": num_days,
        "date_cols": ctx["date_cols"],
    }


def diff_grid_vs_schedule(
    grid_df: pd.DataFrame,
    schedule: list[list[str]],
    engineers: list[str],
    date_cols: list[str],
) -> list[dict[str, str]]:
    by_eng = {engineers[i]: schedule[i] for i in range(len(engineers))}
    diffs: list[dict[str, str]] = []
    dates = [c for c in grid_df.columns if c != "班別群組"]
    for eng in grid_df.index:
        eng_s = str(eng)
        row_s = by_eng.get(eng_s)
        if row_s is None:
            continue
        for d, col in enumerate(dates):
            if d >= len(row_s) or col not in grid_df.columns:
                continue
            ga = str(grid_df.loc[eng].get(col, "") or "").strip().upper()
            ga = ga if ga in ("D", "E", "N", "O") else ("O" if not ga else ga)
            if ga not in ("D", "E", "N", "O"):
                ga = "O"
            gr = row_s[d]
            if ga != gr:
                diffs.append(
                    {"人員": eng_s, "日期": col, "algo": ga, "gurobi": gr}
                )
    return diffs


def breakdown_for_json(bd: dict[str, int]) -> dict[str, Any]:
    trans = (
        bd["night_to_day_aft"] + bd["aft_to_day"] + bd["day_aft_to_night"]
    )
    # 與 genetic 等演算法 penalty 分項同為「加權後分數」
    weighted = {
        "consecutive_6": round(bd["consecutive_6days"] * 1.0, 4),
        "transition": round(trans * 1.0, 4),
        "default_shift": round(bd["violate_default"] * 0.2, 4),
        "rest_blocks": round(bd["consec_off_lt2"] * 0.1, 4),
        "monthly_off": round(bd["total_off_lt9"] * 0.1, 4),
        "weekend_off": round(bd["weekend_off_lt4"] * 0.1, 4),
        "single_off": round(bd["isolated_off"] * 0.1, 4),
    }
    return {
        "consecutive_6": bd["consecutive_6days"],
        "transition_total": trans,
        "night_to_day_aft": bd["night_to_day_aft"],
        "aft_to_day": bd["aft_to_day"],
        "day_aft_to_night": bd["day_aft_to_night"],
        "violate_default": bd["violate_default"],
        "consec_off_lt2": bd["consec_off_lt2"],
        "total_off_lt9": bd["total_off_lt9"],
        "weekend_off_lt4": bd["weekend_off_lt4"],
        "isolated_off": bd["isolated_off"],
        "weighted": weighted,
        "weighted_total": round(weighted_total_penalty(bd), 4),
    }
