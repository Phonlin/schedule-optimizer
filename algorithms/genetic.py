import random
import pandas as pd
from deap import base, creator, tools, algorithms

from algorithms.base import BaseScheduler

# ── 班別代碼 ────────────────────────────────────────────────────────────
# D=日班 / E=午班 / N=晚班
SHIFTS = ["D", "E", "N"]
SHIFT_IDX = {s: i for i, s in enumerate(SHIFTS)}
GROUP_TO_SHIFT = {"D": "D", "E": "E", "N": "N"}

# 班別轉換違規（相鄰連續工作日）：晚→日、晚→午、午→日 作息不足
INVALID_TRANSITIONS: set[tuple[str, str]] = {
    ("N", "D"), ("N", "E"), ("E", "D"),
}

# ── 懲罰權重（對應圖片）───────────────────────────────────────────────
W_CONSEC_6     = 1.0    # 連續上班 ≥6 天（每超過第5天加1次）
W_TRANSITION   = 1.0    # 班別轉換違規（每次）
W_DEFAULT      = 0.2    # 違反預設班別（每人天）
W_REST_BLOCK   = 0.1    # 連續休假段數 <2（每缺1段）
W_MONTHLY      = 0.1    # 月休 <9（每缺1天）
W_WEEKEND      = 0.1    # 周末休 <4（每缺1天）
W_SINGLE_OFF   = 0.1    # 單休1日（每次孤立單日）
# 需求懲罰（硬性）
W_DEMAND_SHORT = 50.0   # 缺人（每缺1人）
W_DEMAND_OVER  = 0.0    # 超額（每超1人）


def _safe_int(val, default=0):
    try:
        return default if pd.isna(val) else int(float(val))
    except (TypeError, ValueError):
        return default


def _run_lengths(indices: list[int]) -> list[int]:
    """將連續索引序列拆成各段長度。"""
    if not indices:
        return []
    runs, length = [], 1
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            length += 1
        else:
            runs.append(length)
            length = 1
    runs.append(length)
    return runs


# ────────────────────────────────────────────────────────────────────────
# Repair 輔助函數
# ────────────────────────────────────────────────────────────────────────

def _would_create_transition(genes: list, ei: int, di: int,
                              shift_gene_val: int, n_days: int) -> bool:
    """檢查將 shift_gene_val 指派給 (ei, di) 是否會產生班別轉換違規。"""
    shift_str = SHIFTS[shift_gene_val - 1]
    base_idx = ei * n_days
    if di > 0:
        prev_g = genes[base_idx + di - 1]
        if prev_g > 0 and (SHIFTS[prev_g - 1], shift_str) in INVALID_TRANSITIONS:
            return True
    if di < n_days - 1:
        next_g = genes[base_idx + di + 1]
        if next_g > 0 and (shift_str, SHIFTS[next_g - 1]) in INVALID_TRANSITIONS:
            return True
    return False


def _consec_work_around(genes: list, ei: int, di: int, n_days: int) -> int:
    """假設 (ei, di) 也工作，計算包含該天的連續工作天數。"""
    base_idx = ei * n_days
    streak = 1
    for d in range(di - 1, -1, -1):
        if genes[base_idx + d] > 0:
            streak += 1
        else:
            break
    for d in range(di + 1, n_days):
        if genes[base_idx + d] > 0:
            streak += 1
        else:
            break
    return streak


def _would_create_single_off(genes: list, ei: int, di: int, n_days: int) -> bool:
    """若將 (ei, di) 從休假改為上班，檢查是否會讓 di-1 或 di+1 變成孤立單休。"""
    base = ei * n_days
    if di >= 2 and genes[base + di - 1] == 0 and genes[base + di - 2] > 0:
        return True
    if di <= n_days - 3 and genes[base + di + 1] == 0 and genes[base + di + 2] > 0:
        return True
    return False


# ────────────────────────────────────────────────────────────────────────
# 修補算子（Enhanced Repair Operator）
# ────────────────────────────────────────────────────────────────────────

def _repair(genes: list, fixed: list, demand: dict, dates: list,
            n_eng: int, n_days: int, primary_shift: list) -> list:
    """
    就地修補：
    1. 超額裁人：某天某班別超過需求，移除多餘人員
    2. 缺人補足：某天某班別不足需求，從休息者中補人
       - 主要班別優先
       - 迴避班別轉換違規
       - 迴避連續工作 ≥6 天
       - 負載均衡
    """
    workload = [
        sum(1 for di in range(n_days) if genes[ei * n_days + di] > 0)
        for ei in range(n_eng)
    ]

    for di, date in enumerate(dates):
        daily = [genes[ei * n_days + di] for ei in range(n_eng)]

        for shi, shift in enumerate(SHIFTS):
            gene_val = shi + 1
            current_count = daily.count(gene_val)
            needed = demand[date][shift] - current_count

            if needed > 0:
                candidates = [
                    ei for ei in range(n_eng)
                    if fixed[ei][di] is None and daily[ei] == 0
                ]
                candidates.sort(key=lambda ei: (
                    0 if primary_shift[ei] == shift else 1,
                    1 if _would_create_transition(genes, ei, di, gene_val, n_days) else 0,
                    1 if _would_create_single_off(genes, ei, di, n_days) else 0,
                    1 if _consec_work_around(genes, ei, di, n_days) >= 5 else 0,
                    workload[ei],
                    random.random(),
                ))
                for ei in candidates[:needed]:
                    genes[ei * n_days + di] = gene_val
                    daily[ei] = gene_val
                    workload[ei] += 1

            elif needed < 0:
                excess = -needed
                removable = [
                    ei for ei in range(n_eng)
                    if fixed[ei][di] is None and daily[ei] == gene_val
                ]
                removable.sort(key=lambda ei: (
                    0 if primary_shift[ei] != shift else 1,
                    -workload[ei],
                    random.random(),
                ))
                for ei in removable[:excess]:
                    genes[ei * n_days + di] = 0
                    daily[ei] = 0
                    workload[ei] -= 1

    return genes


# ────────────────────────────────────────────────────────────────────────
# 初始化
# ────────────────────────────────────────────────────────────────────────

def _make_feasible_genes(fixed: list, demand: dict, dates: list,
                          n_eng: int, n_days: int, primary_shift: list,
                          init_rate: float = 0.0) -> list:
    """
    建立一個可行初始解（保證需求達標）。
    init_rate: 0=全空底，>0 以此比率隨機預填主要班別（增加多樣性）。
    """
    genes = [0] * (n_eng * n_days)
    for ei in range(n_eng):
        for di in range(n_days):
            if fixed[ei][di] is not None:
                genes[ei * n_days + di] = fixed[ei][di]
            elif init_rate > 0 and random.random() < init_rate:
                genes[ei * n_days + di] = SHIFT_IDX[primary_shift[ei]] + 1
    _repair(genes, fixed, demand, dates, n_eng, n_days, primary_shift)
    return genes


def _make_greedy_genes(fixed: list, demand: dict, dates: list,
                        n_eng: int, n_days: int, primary_shift: list) -> list:
    """
    貪心初始化：逐天逐班別，優先指派主班別吻合的工程師，
    同時避免轉換違規和連續過長工作。日期順序隨機化以增加多樣性。
    """
    genes = [0] * (n_eng * n_days)
    for ei in range(n_eng):
        for di in range(n_days):
            if fixed[ei][di] is not None:
                genes[ei * n_days + di] = fixed[ei][di]

    workload = [
        sum(1 for di in range(n_days) if genes[ei * n_days + di] > 0)
        for ei in range(n_eng)
    ]

    day_order = list(range(n_days))
    random.shuffle(day_order)

    for di in day_order:
        date = dates[di]
        daily = [genes[ei * n_days + di] for ei in range(n_eng)]

        for shi, shift in enumerate(SHIFTS):
            gene_val = shi + 1
            needed = demand[date][shift] - daily.count(gene_val)
            if needed <= 0:
                continue

            candidates = [
                ei for ei in range(n_eng)
                if fixed[ei][di] is None and daily[ei] == 0
            ]
            candidates.sort(key=lambda ei: (
                0 if primary_shift[ei] == shift else 1,
                1 if _would_create_transition(genes, ei, di, gene_val, n_days) else 0,
                1 if _would_create_single_off(genes, ei, di, n_days) else 0,
                1 if _consec_work_around(genes, ei, di, n_days) >= 5 else 0,
                workload[ei],
                random.random(),
            ))
            for ei in candidates[:needed]:
                genes[ei * n_days + di] = gene_val
                daily[ei] = gene_val
                workload[ei] += 1

    _repair(genes, fixed, demand, dates, n_eng, n_days, primary_shift)
    return genes


# ────────────────────────────────────────────────────────────────────────
# GeneticScheduler
# ────────────────────────────────────────────────────────────────────────

class GeneticScheduler(BaseScheduler):
    """
    遺傳算法排班器，帶需求感知初始化與 Repair 算子。

    染色體：長度 = n_engineers × n_dates，每格 ∈ {0,1,2,3}
      0 = 休息/休假, 1 = 日班(D), 2 = 午班(E), 3 = 晚班(N)

    懲罰規則：
      1.0  連續上班 ≥6 天
      1.0  班別轉換違規（N→D / N→E / E→D）
      0.2  違反預設班別
      0.1  連續休假段數 <2
      0.1  月休 <9
      0.1  周末休 <4
      0.1  單休 1 日
    """

    name = "遺傳算法（Genetic Algorithm）"

    def __init__(
        self,
        pop_size: int = 300,
        n_gen: int = 300,
        cx_prob: float = 0.7,
        mut_prob: float = 0.4,
        mut_indpb: float = 0.05,
        tournsize: int = 5,
    ):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.mut_indpb = mut_indpb
        self.tournsize = tournsize

    # ------------------------------------------------------------------

    def run(self, staff_df: pd.DataFrame, demand_df: pd.DataFrame,
            callback=None) -> tuple:
        """
        callback(gen: int, total: int, best_fitness: float) — 每隔 LOG_INTERVAL 代呼叫一次。
        """
        engineers = staff_df["人員"].tolist()
        dates = demand_df["Date"].tolist()
        n_eng = len(engineers)
        n_days = len(dates)

        # 需求字典
        demand: dict[str, dict[str, int]] = {}
        for _, row in demand_df.iterrows():
            demand[row["Date"]] = {
                "D": _safe_int(row.get("Day", 0)),
                "E": _safe_int(row.get("Afternoon", 0)),
                "N": _safe_int(row.get("Night", 0)),
            }

        is_weekend = [bool(demand_df.iloc[di]["IfWeekend"]) for di in range(n_days)]

        # 固定格 & 主要班別
        fixed: list[list] = [[None] * n_days for _ in range(n_eng)]
        primary_shift: list[str] = []

        for ei, (_, row) in enumerate(staff_df.iterrows()):
            pg = str(row.get("primary_group", "D")).strip().upper()
            primary_shift.append(GROUP_TO_SHIFT.get(pg, "D"))
            for di, date in enumerate(dates):
                cell = str(row.get(date, "")).strip().upper()
                if cell == "O":
                    fixed[ei][di] = 0
                elif cell in SHIFT_IDX:
                    fixed[ei][di] = SHIFT_IDX[cell] + 1

        # ── DEAP 設定 ────────────────────────────────────────────────
        if not hasattr(creator, "FitnessMinGA"):
            creator.create("FitnessMinGA", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "IndividualGA"):
            creator.create("IndividualGA", list, fitness=creator.FitnessMinGA)

        toolbox = base.Toolbox()

        # 初始化：1/3 全空、1/3 高預填、1/3 貪心
        def _make_ind():
            r = random.random()
            if r < 0.33:
                g = _make_feasible_genes(
                    fixed, demand, dates, n_eng, n_days, primary_shift, 0.0
                )
            elif r < 0.66:
                g = _make_feasible_genes(
                    fixed, demand, dates, n_eng, n_days, primary_shift, 0.5
                )
            else:
                g = _make_greedy_genes(
                    fixed, demand, dates, n_eng, n_days, primary_shift
                )
            return creator.IndividualGA(g)

        toolbox.register("individual", _make_ind)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # ── 懲罰函數 ────────────────────────────────────────────────
        def _compute_penalty(individual, detailed: bool = False):
            count = {d: {"D": 0, "E": 0, "N": 0} for d in dates}
            for ei in range(n_eng):
                for di, date in enumerate(dates):
                    g = individual[ei * n_days + di]
                    if g > 0:
                        count[date][SHIFTS[g - 1]] += 1

            p_demand = 0.0
            for date in dates:
                for sh in SHIFTS:
                    diff = demand[date][sh] - count[date][sh]
                    if diff > 0:
                        p_demand += W_DEMAND_SHORT * diff
                    elif diff < 0:
                        p_demand += W_DEMAND_OVER * (-diff)

            p_consec6 = p_trans = p_default = 0.0
            p_rest_blk = p_monthly = p_weekend = p_single = 0.0

            for ei in range(n_eng):
                work_seq: list[tuple[int, str]] = []
                off_seq: list[int] = []
                for di in range(n_days):
                    g = individual[ei * n_days + di]
                    if g > 0:
                        work_seq.append((di, SHIFTS[g - 1]))
                    else:
                        off_seq.append(di)

                # 連續上班 ≥6 天
                work_idx = [d for d, _ in work_seq]
                for rl in _run_lengths(work_idx):
                    if rl >= 6:
                        p_consec6 += W_CONSEC_6 * (rl - 5)

                # 班別轉換違規
                for k in range(1, len(work_seq)):
                    pd_prev, ps_prev = work_seq[k - 1]
                    pd_curr, ps_curr = work_seq[k]
                    if pd_curr == pd_prev + 1 and (ps_prev, ps_curr) in INVALID_TRANSITIONS:
                        p_trans += W_TRANSITION

                # 違反預設班別
                ps = primary_shift[ei]
                for _, sh in work_seq:
                    if sh != ps:
                        p_default += W_DEFAULT

                # 月休 <9
                off_cnt = len(off_seq)
                if off_cnt < 9:
                    p_monthly += W_MONTHLY * (9 - off_cnt)

                # 周末休 <4
                wkend_off = sum(1 for di in off_seq if is_weekend[di])
                if wkend_off < 4:
                    p_weekend += W_WEEKEND * (4 - wkend_off)

                # 休假段分析
                off_runs = _run_lengths(sorted(off_seq))
                if len(off_runs) < 2:
                    p_rest_blk += W_REST_BLOCK * (2 - len(off_runs))

                # 單休1日：僅懲罰「工作＋休假＋工作」，第一天與最後一天不計
                off_set = set(off_seq)
                for di in off_seq:
                    prev_work = di > 0 and (di - 1) not in off_set
                    next_work = di < n_days - 1 and (di + 1) not in off_set
                    if prev_work and next_work:
                        p_single += W_SINGLE_OFF

            total = (
                p_demand + p_consec6 + p_trans + p_default
                + p_rest_blk + p_monthly + p_weekend + p_single
            )
            if not detailed:
                return total
            return {
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

        # ── 智慧突變算子 ─────────────────────────────────────────────

        def _mutate_random(ind):
            """原始隨機突變。"""
            for idx in range(len(ind)):
                ei = idx // n_days
                di = idx % n_days
                if fixed[ei][di] is not None:
                    continue
                if random.random() < self.mut_indpb:
                    if random.random() < 0.5:
                        ind[idx] = SHIFT_IDX[primary_shift[ei]] + 1
                    else:
                        ind[idx] = random.randint(0, 3)

        def _mutate_swap_shift(ind):
            """同天班別互換：不影響需求滿足，但可降低 default_shift 懲罰。"""
            attempts = max(1, n_days // 3)
            for _ in range(attempts):
                di = random.randint(0, n_days - 1)
                workers = [
                    ei for ei in range(n_eng)
                    if fixed[ei][di] is None and ind[ei * n_days + di] > 0
                ]
                if len(workers) < 2:
                    continue
                e1, e2 = random.sample(workers, 2)
                idx1 = e1 * n_days + di
                idx2 = e2 * n_days + di
                if ind[idx1] != ind[idx2]:
                    ind[idx1], ind[idx2] = ind[idx2], ind[idx1]

        def _mutate_fix_transition(ind):
            """掃描轉換違規並修復其中一處。"""
            violations = []
            for ei in range(n_eng):
                base_idx = ei * n_days
                for di in range(n_days - 1):
                    g_curr = ind[base_idx + di]
                    g_next = ind[base_idx + di + 1]
                    if g_curr > 0 and g_next > 0:
                        if (SHIFTS[g_curr - 1], SHIFTS[g_next - 1]) in INVALID_TRANSITIONS:
                            violations.append((ei, di + 1))
            if not violations:
                return
            ei, di = random.choice(violations)
            if fixed[ei][di] is not None:
                return
            prev_g = ind[ei * n_days + di - 1] if di > 0 else 0
            prev_s = SHIFTS[prev_g - 1] if prev_g > 0 else None
            options = [0]
            for gv in range(1, 4):
                s = SHIFTS[gv - 1]
                if prev_s is None or (prev_s, s) not in INVALID_TRANSITIONS:
                    options.append(gv)
            ps_gv = SHIFT_IDX[primary_shift[ei]] + 1
            if ps_gv in options and random.random() < 0.7:
                ind[ei * n_days + di] = ps_gv
            else:
                ind[ei * n_days + di] = random.choice(options)

        def _mutate_merge_rest(ind):
            """找到單休日，嘗試將相鄰天也改為休息以消除孤立單日。"""
            candidates = []
            for ei in range(n_eng):
                base_idx = ei * n_days
                for di in range(n_days):
                    if ind[base_idx + di] != 0:
                        continue
                    prev_work = di > 0 and ind[base_idx + di - 1] > 0
                    next_work = di < n_days - 1 and ind[base_idx + di + 1] > 0
                    if prev_work and next_work:
                        candidates.append((ei, di))
            if not candidates:
                return
            ei, di = random.choice(candidates)
            adj = []
            if di > 0 and fixed[ei][di - 1] is None:
                adj.append(di - 1)
            if di < n_days - 1 and fixed[ei][di + 1] is None:
                adj.append(di + 1)
            if adj:
                target_di = random.choice(adj)
                ind[ei * n_days + target_di] = 0

        # ── 遺傳算子（每次操作後 repair）────────────────────────────
        def evaluate(ind):
            return (_compute_penalty(ind),)

        def crossover(ind1, ind2):
            for ei in range(n_eng):
                if random.random() < 0.5:
                    start = ei * n_days
                    end = start + n_days
                    ind1[start:end], ind2[start:end] = (
                        ind2[start:end], ind1[start:end]
                    )
            for ei in range(n_eng):
                for di in range(n_days):
                    f = fixed[ei][di]
                    if f is not None:
                        ind1[ei * n_days + di] = f
                        ind2[ei * n_days + di] = f
            _repair(ind1, fixed, demand, dates, n_eng, n_days, primary_shift)
            _repair(ind2, fixed, demand, dates, n_eng, n_days, primary_shift)
            return ind1, ind2

        def mutate(ind):
            r = random.random()
            if r < 0.3:
                _mutate_random(ind)
            elif r < 0.6:
                _mutate_swap_shift(ind)
            elif r < 0.8:
                _mutate_fix_transition(ind)
            else:
                _mutate_merge_rest(ind)
            _repair(ind, fixed, demand, dates, n_eng, n_days, primary_shift)
            return (ind,)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", crossover)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selTournament, tournsize=self.tournsize)

        pop = toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)

        # 菁英保留（10%）
        n_elite = max(1, self.pop_size // 10)

        LOG_INTERVAL = max(1, self.n_gen // 20)

        def ea_with_elitism(population, tb, cxpb, mutpb, ngen, hof, callback=None):
            fits = tb.map(tb.evaluate, population)
            for ind, fit in zip(population, fits):
                ind.fitness.values = fit
            hof.update(population)

            for gen in range(1, ngen + 1):
                offspring = tb.select(population, len(population) - n_elite)
                offspring = algorithms.varAnd(offspring, tb, cxpb, mutpb)
                invalid = [ind for ind in offspring if not ind.fitness.valid]
                fits = tb.map(tb.evaluate, invalid)
                for ind, fit in zip(invalid, fits):
                    ind.fitness.values = fit
                elites = tools.selBest(population, n_elite)
                population[:] = offspring + elites
                hof.update(population)

                if callback and (gen % LOG_INTERVAL == 0 or gen == ngen):
                    callback(gen, ngen, round(hof[0].fitness.values[0], 2))

            return population

        ea_with_elitism(
            pop, toolbox,
            cxpb=self.cx_prob, mutpb=self.mut_prob,
            ngen=self.n_gen, hof=hof,
            callback=callback,
        )

        # ── Local Search（Hill Climbing）─────────────────────────────
        best = list(hof[0])
        best_p = _compute_penalty(best)
        ls_iters = 2000

        for _ in range(ls_iters):
            op = random.random()

            if op < 0.4:
                ei = random.randint(0, n_eng - 1)
                di = random.randint(0, n_days - 1)
                if fixed[ei][di] is not None:
                    continue
                idx = ei * n_days + di
                old_val = best[idx]
                new_val = random.choice([v for v in range(4) if v != old_val])
                best[idx] = new_val
                new_p = _compute_penalty(best)
                if new_p < best_p:
                    best_p = new_p
                else:
                    best[idx] = old_val

            elif op < 0.8:
                di = random.randint(0, n_days - 1)
                e1 = random.randint(0, n_eng - 1)
                e2 = random.randint(0, n_eng - 1)
                if e1 == e2:
                    continue
                if fixed[e1][di] is not None or fixed[e2][di] is not None:
                    continue
                idx1 = e1 * n_days + di
                idx2 = e2 * n_days + di
                if best[idx1] == best[idx2]:
                    continue
                best[idx1], best[idx2] = best[idx2], best[idx1]
                new_p = _compute_penalty(best)
                if new_p < best_p:
                    best_p = new_p
                else:
                    best[idx1], best[idx2] = best[idx2], best[idx1]

            else:
                e1 = random.randint(0, n_eng - 1)
                e2 = random.randint(0, n_eng - 1)
                if e1 == e2:
                    continue
                d_start = random.randint(0, n_days - 1)
                d_end = min(d_start + random.randint(1, 5), n_days)
                can_swap = all(
                    fixed[e1][d] is None and fixed[e2][d] is None
                    for d in range(d_start, d_end)
                )
                if not can_swap:
                    continue
                saved = [
                    (best[e1 * n_days + d], best[e2 * n_days + d])
                    for d in range(d_start, d_end)
                ]
                for d in range(d_start, d_end):
                    i1, i2 = e1 * n_days + d, e2 * n_days + d
                    best[i1], best[i2] = best[i2], best[i1]
                new_p = _compute_penalty(best)
                if new_p < best_p:
                    best_p = new_p
                else:
                    for j, d in enumerate(range(d_start, d_end)):
                        best[e1 * n_days + d] = saved[j][0]
                        best[e2 * n_days + d] = saved[j][1]

        if callback:
            callback(self.n_gen, self.n_gen, round(best_p, 2))

        # ── 最佳個體 → grid DataFrame ────────────────────────────────
        grid = {}
        for ei, eng in enumerate(engineers):
            row_vals = []
            for di in range(n_days):
                g = best[ei * n_days + di]
                if g == 0:
                    row_vals.append("O" if fixed[ei][di] == 0 else "")
                else:
                    row_vals.append(SHIFTS[g - 1])
            grid[eng] = row_vals

        result_df = pd.DataFrame(grid, index=dates).T
        result_df.index.name = "人員"
        result_df.columns = dates

        group_map = dict(
            zip(staff_df["人員"],
                staff_df.get("班別群組", staff_df.get("primary_group", "")))
        )
        result_df.insert(0, "班別群組", result_df.index.map(group_map))

        penalty_breakdown = _compute_penalty(best, detailed=True)
        return result_df, penalty_breakdown
