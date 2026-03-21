import random

import pandas as pd

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

# ── 懲罰權重 ────────────────────────────────────────────────────────────
W_CONSEC_6 = 1.0
W_TRANSITION = 1.0
W_DEFAULT = 0.2
W_REST_BLOCK = 0.1
W_MONTHLY = 0.1
W_WEEKEND = 0.1
W_SINGLE_OFF = 0.1
W_DEMAND_SHORT = 50.0
W_DEMAND_OVER = 0.0


def _safe_int(val, default=0):
    try:
        return default if pd.isna(val) else int(float(val))
    except (TypeError, ValueError):
        return default


def _run_lengths(indices: list[int]) -> list[int]:
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


def _would_create_transition(
    genes: list[int], ei: int, di: int, shift_gene_val: int, n_days: int
) -> bool:
    if shift_gene_val <= 0:
        return False
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


def _consec_work_around(genes: list[int], ei: int, di: int, n_days: int) -> int:
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


def _would_create_single_off(
    genes: list[int], ei: int, di: int, n_days: int
) -> bool:
    """若將 (ei, di) 從休假改為上班，檢查是否會讓 di-1 或 di+1 變成孤立單休。"""
    base = ei * n_days
    if di >= 2 and genes[base + di - 1] == 0 and genes[base + di - 2] > 0:
        return True
    if di <= n_days - 3 and genes[base + di + 1] == 0 and genes[base + di + 2] > 0:
        return True
    return False


def _weekend_off_count(genes: list[int], ei: int, n_days: int, is_weekend: list[bool]) -> int:
    base = ei * n_days
    return sum(1 for di in range(n_days) if is_weekend[di] and genes[base + di] == 0)


def _repair(
    genes: list[int],
    fixed: list[list[int | None]],
    demand: dict[str, dict[str, int]],
    dates: list[str],
    n_eng: int,
    n_days: int,
    primary_shift: list[str],
    is_weekend: list[bool] | None = None,
) -> list[int]:
    workload = [
        sum(1 for di in range(n_days) if genes[ei * n_days + di] > 0)
        for ei in range(n_eng)
    ]

    for di, date in enumerate(dates):
        daily = [genes[ei * n_days + di] for ei in range(n_eng)]
        di_is_wkend = is_weekend[di] if is_weekend else False

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
                    1 if di_is_wkend and is_weekend and _weekend_off_count(genes, ei, n_days, is_weekend) <= 4 else 0,
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


def _make_feasible_genes(
    fixed: list[list[int | None]],
    demand: dict[str, dict[str, int]],
    dates: list[str],
    n_eng: int,
    n_days: int,
    primary_shift: list[str],
    init_rate: float = 0.0,
    is_weekend: list[bool] | None = None,
) -> list[int]:
    genes = [0] * (n_eng * n_days)
    for ei in range(n_eng):
        for di in range(n_days):
            if fixed[ei][di] is not None:
                genes[ei * n_days + di] = fixed[ei][di]
            elif init_rate > 0 and random.random() < init_rate:
                genes[ei * n_days + di] = SHIFT_IDX[primary_shift[ei]] + 1
    _repair(genes, fixed, demand, dates, n_eng, n_days, primary_shift, is_weekend)
    return genes


def _make_greedy_genes(
    fixed: list[list[int | None]],
    demand: dict[str, dict[str, int]],
    dates: list[str],
    n_eng: int,
    n_days: int,
    primary_shift: list[str],
    is_weekend: list[bool] | None = None,
) -> list[int]:
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
        di_is_wkend = is_weekend[di] if is_weekend else False

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
                1 if di_is_wkend and is_weekend and _weekend_off_count(genes, ei, n_days, is_weekend) <= 4 else 0,
                1 if _consec_work_around(genes, ei, di, n_days) >= 5 else 0,
                workload[ei],
                random.random(),
            ))
            for ei in candidates[:needed]:
                genes[ei * n_days + di] = gene_val
                daily[ei] = gene_val
                workload[ei] += 1

    _repair(genes, fixed, demand, dates, n_eng, n_days, primary_shift, is_weekend)
    return genes


def _compute_penalty(
    individual: list[int],
    dates: list[str],
    demand: dict[str, dict[str, int]],
    n_eng: int,
    n_days: int,
    primary_shift: list[str],
    is_weekend: list[bool],
    detailed: bool = False,
):
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

        work_idx = [d for d, _ in work_seq]
        for rl in _run_lengths(work_idx):
            if rl >= 6:
                p_consec6 += W_CONSEC_6 * (rl - 5)

        for k in range(1, len(work_seq)):
            pd_prev, ps_prev = work_seq[k - 1]
            pd_curr, ps_curr = work_seq[k]
            if pd_curr == pd_prev + 1 and (ps_prev, ps_curr) in INVALID_TRANSITIONS:
                p_trans += W_TRANSITION

        ps = primary_shift[ei]
        for _, sh in work_seq:
            if sh != ps:
                p_default += W_DEFAULT

        off_cnt = len(off_seq)
        if off_cnt < 9:
            p_monthly += W_MONTHLY * (9 - off_cnt)

        wkend_off = sum(1 for di in off_seq if is_weekend[di])
        if wkend_off < 4:
            p_weekend += W_WEEKEND * (4 - wkend_off)

        off_runs = _run_lengths(sorted(off_seq))
        if len(off_runs) < 2:
            p_rest_blk += W_REST_BLOCK * (2 - len(off_runs))

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
        "total": round(total, 2),
        "demand": round(p_demand, 2),
        "consecutive_6": round(p_consec6, 2),
        "transition": round(p_trans, 2),
        "default_shift": round(p_default, 2),
        "rest_blocks": round(p_rest_blk, 2),
        "monthly_off": round(p_monthly, 2),
        "weekend_off": round(p_weekend, 2),
        "single_off": round(p_single, 2),
    }


class TabuScheduler(BaseScheduler):
    name = "禁忌搜尋（Tabu Search）"

    def __init__(
        self,
        max_iters: int = 800,
        tabu_tenure: int = 25, # 40 -> 25 從 1 變 0.8
        neighbor_samples: int = 300,
        max_no_improve: int = 300,
        candidate_pool_size: int = 8,
        init_candidates: int = 9, # 讓一開始的選擇更多樣化
        log_interval: int | None = None,
    ):
        self.max_iters = max_iters
        self.tabu_tenure = tabu_tenure
        self.neighbor_samples = neighbor_samples
        self.max_no_improve = max_no_improve
        self.candidate_pool_size = candidate_pool_size
        self.init_candidates = init_candidates
        self.log_interval = log_interval

    def run(self, staff_df: pd.DataFrame, demand_df: pd.DataFrame, callback=None) -> tuple:
        engineers = staff_df["人員"].tolist()
        dates = demand_df["Date"].tolist()
        n_eng = len(engineers)
        n_days = len(dates)

        demand: dict[str, dict[str, int]] = {}
        for _, row in demand_df.iterrows():
            demand[row["Date"]] = {
                "D": _safe_int(row.get("Day", 0)),
                "E": _safe_int(row.get("Afternoon", 0)),
                "N": _safe_int(row.get("Night", 0)),
            }

        is_weekend = [bool(demand_df.iloc[di]["IfWeekend"]) for di in range(n_days)]

        fixed: list[list[int | None]] = [[None] * n_days for _ in range(n_eng)]
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

        def penalty(individual: list[int], detailed: bool = False):
            return _compute_penalty(
                individual,
                dates,
                demand,
                n_eng,
                n_days,
                primary_shift,
                is_weekend,
                detailed=detailed,
            )

        free_cells = [
            (ei, di)
            for ei in range(n_eng)
            for di in range(n_days)
            if fixed[ei][di] is None
        ]

        def enforce_fixed(ind: list[int]) -> None:
            for ei in range(n_eng):
                for di in range(n_days):
                    if fixed[ei][di] is not None:
                        ind[ei * n_days + di] = fixed[ei][di]

        def build_initial_solution() -> list[int]:
            seeds = []
            for idx in range(max(3, self.init_candidates)):
                mode = idx % 3
                if mode == 0:
                    cand = _make_feasible_genes(
                        fixed, demand, dates, n_eng, n_days, primary_shift, 0.0, is_weekend
                    )
                elif mode == 1:
                    cand = _make_feasible_genes(
                        fixed, demand, dates, n_eng, n_days, primary_shift, 0.5, is_weekend
                    )
                else:
                    cand = _make_greedy_genes(
                        fixed, demand, dates, n_eng, n_days, primary_shift, is_weekend
                    )
                seeds.append(cand)
            return min(seeds, key=penalty)

        def candidate_transition_fixes(ind: list[int]) -> list[tuple[int, int]]:
            fixes: list[tuple[int, int]] = []
            for ei in range(n_eng):
                base_idx = ei * n_days
                for di in range(n_days - 1):
                    g_curr = ind[base_idx + di]
                    g_next = ind[base_idx + di + 1]
                    if g_curr > 0 and g_next > 0:
                        if (SHIFTS[g_curr - 1], SHIFTS[g_next - 1]) in INVALID_TRANSITIONS:
                            if fixed[ei][di + 1] is None:
                                fixes.append((ei, di + 1))
            return fixes

        def candidate_single_offs(ind: list[int]) -> list[tuple[int, int]]:
            singles: list[tuple[int, int]] = []
            for ei in range(n_eng):
                base_idx = ei * n_days
                for di in range(1, n_days - 1):
                    if ind[base_idx + di] != 0:
                        continue
                    if ind[base_idx + di - 1] > 0 and ind[base_idx + di + 1] > 0:
                        singles.append((ei, di))
            return singles

        def make_set_move(ind: list[int]):
            if not free_cells:
                return None
            ei, di = random.choice(free_cells)
            idx = ei * n_days + di
            old_val = ind[idx]
            choices = [v for v in range(4) if v != old_val]
            if not choices:
                return None
            if random.random() < 0.6:
                preferred = SHIFT_IDX[primary_shift[ei]] + 1
                new_val = preferred if preferred != old_val else random.choice(choices)
            else:
                new_val = random.choice(choices)
            return ("set", ei, di, old_val, new_val)

        def make_swap_day_move(ind: list[int]):
            if n_eng < 2 or n_days == 0:
                return None
            for _ in range(self.candidate_pool_size):
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
                    return ("swap_day", di, min(e1, e2), max(e1, e2))
            return None

        def make_swap_block_move(ind: list[int]):
            if n_eng < 2 or n_days == 0:
                return None
            for _ in range(self.candidate_pool_size):
                e1, e2 = random.sample(range(n_eng), 2)
                d_start = random.randint(0, n_days - 1)
                d_end = min(d_start + random.randint(1, 5), n_days)
                can_swap = all(
                    fixed[e1][d] is None and fixed[e2][d] is None
                    for d in range(d_start, d_end)
                )
                if can_swap:
                    return ("swap_block", min(e1, e2), max(e1, e2), d_start, d_end)
            return None

        def make_transition_fix_move(ind: list[int]):
            violations = candidate_transition_fixes(ind)
            if not violations:
                return None
            ei, di = random.choice(violations)
            idx = ei * n_days + di
            old_val = ind[idx]
            prev_g = ind[idx - 1] if di > 0 else 0
            prev_s = SHIFTS[prev_g - 1] if prev_g > 0 else None
            options = [0]
            for gv in range(1, 4):
                s = SHIFTS[gv - 1]
                if prev_s is None or (prev_s, s) not in INVALID_TRANSITIONS:
                    options.append(gv)
            options = [v for v in options if v != old_val]
            if not options:
                return None
            preferred = SHIFT_IDX[primary_shift[ei]] + 1
            if preferred in options and random.random() < 0.7:
                new_val = preferred
            else:
                new_val = random.choice(options)
            return ("set", ei, di, old_val, new_val)

        def make_merge_rest_move(ind: list[int]):
            candidates = candidate_single_offs(ind)
            if not candidates:
                return None
            ei, di = random.choice(candidates)
            adj = []
            if di > 0 and fixed[ei][di - 1] is None:
                adj.append(di - 1)
            if di < n_days - 1 and fixed[ei][di + 1] is None:
                adj.append(di + 1)
            if not adj:
                return None
            target_di = random.choice(adj)
            idx = ei * n_days + target_di
            old_val = ind[idx]
            if old_val == 0:
                return None
            return ("set", ei, target_di, old_val, 0)

        def make_weekend_swap_move(ind: list[int]):
            needy = [
                ei for ei in range(n_eng)
                if _weekend_off_count(ind, ei, n_days, is_weekend) < 4
            ]
            if not needy:
                return None
            random.shuffle(needy)
            for ei in needy:
                wkend_work = [
                    di for di in range(n_days)
                    if is_weekend[di] and ind[ei * n_days + di] > 0 and fixed[ei][di] is None
                ]
                if not wkend_work:
                    continue
                di = random.choice(wkend_work)
                shift_val = ind[ei * n_days + di]
                donors = [
                    e2 for e2 in range(n_eng)
                    if e2 != ei
                    and fixed[e2][di] is None
                    and ind[e2 * n_days + di] == 0
                    and _weekend_off_count(ind, e2, n_days, is_weekend) > 4
                ]
                if not donors:
                    donors = [
                        e2 for e2 in range(n_eng)
                        if e2 != ei
                        and fixed[e2][di] is None
                        and ind[e2 * n_days + di] == 0
                        and _weekend_off_count(ind, e2, n_days, is_weekend) >= 4
                    ]
                if not donors:
                    continue
                e2 = min(donors, key=lambda e: (
                    0 if primary_shift[e] == SHIFTS[shift_val - 1] else 1,
                    1 if _would_create_transition(ind, e, di, shift_val, n_days) else 0,
                    1 if _consec_work_around(ind, e, di, n_days) >= 5 else 0,
                    random.random(),
                ))
                return ("weekend_swap", ei, e2, di, shift_val)
            return None

        def apply_move(ind: list[int], move) -> list[int]:
            cand = list(ind)
            kind = move[0]
            if kind == "set":
                _, ei, di, _, new_val = move
                cand[ei * n_days + di] = new_val
            elif kind == "swap_day":
                _, di, e1, e2 = move
                idx1 = e1 * n_days + di
                idx2 = e2 * n_days + di
                cand[idx1], cand[idx2] = cand[idx2], cand[idx1]
            elif kind == "swap_block":
                _, e1, e2, d_start, d_end = move
                for d in range(d_start, d_end):
                    idx1 = e1 * n_days + d
                    idx2 = e2 * n_days + d
                    cand[idx1], cand[idx2] = cand[idx2], cand[idx1]
            elif kind == "weekend_swap":
                _, ei, e2, di, shift_val = move
                cand[ei * n_days + di] = 0
                cand[e2 * n_days + di] = shift_val
            enforce_fixed(cand)
            _repair(cand, fixed, demand, dates, n_eng, n_days, primary_shift, is_weekend)
            enforce_fixed(cand)
            return cand

        def inverse_signature(move):
            kind = move[0]
            if kind == "set":
                _, ei, di, old_val, new_val = move
                return ("set", ei, di, new_val, old_val)
            if kind == "weekend_swap":
                _, ei, e2, di, shift_val = move
                return ("weekend_swap", e2, ei, di, shift_val)
            return move

        move_builders = [
            (0.25, make_set_move),
            (0.20, make_swap_day_move),
            (0.15, make_swap_block_move),
            (0.10, make_transition_fix_move),
            (0.10, make_merge_rest_move),
            (0.20, make_weekend_swap_move),
        ]

        def sample_move(ind: list[int]):
            r = random.random()
            cumulative = 0.0
            for prob, builder in move_builders:
                cumulative += prob
                if r <= cumulative:
                    return builder(ind)
            return move_builders[-1][1](ind)

        def perturb(ind: list[int], strength: int = 8) -> list[int]:
            cand = list(ind)
            targets = [c for c in free_cells if random.random() < strength / max(1, len(free_cells))]
            if len(targets) < strength:
                targets = random.sample(free_cells, min(strength, len(free_cells)))
            for ei, di in targets:
                idx = ei * n_days + di
                if random.random() < 0.5:
                    cand[idx] = 0
                else:
                    cand[idx] = SHIFT_IDX[primary_shift[ei]] + 1
            enforce_fixed(cand)
            _repair(cand, fixed, demand, dates, n_eng, n_days, primary_shift, is_weekend)
            enforce_fixed(cand)
            return cand

        def hill_climb(ind: list[int], ind_p: float, max_rounds: int = 3) -> tuple[list[int], float]:
            improved = True
            for _ in range(max_rounds):
                if not improved:
                    break
                improved = False
                cells = list(free_cells)
                random.shuffle(cells)
                for ei, di in cells:
                    idx = ei * n_days + di
                    old_val = ind[idx]
                    for new_val in range(4):
                        if new_val == old_val:
                            continue
                        ind[idx] = new_val
                        new_p = penalty(ind)
                        if new_p < ind_p:
                            ind_p = new_p
                            old_val = new_val
                            improved = True
                        else:
                            ind[idx] = old_val
            return ind, ind_p

        n_restarts = 5
        iters_per_restart = self.max_iters // n_restarts
        no_improve_limit = self.max_no_improve // n_restarts
        total_iters = self.max_iters
        log_interval = self.log_interval or max(1, total_iters // 20)

        current = build_initial_solution()
        current_p = penalty(current)
        best = list(current)
        best_p = current_p
        global_iter = 0

        for restart in range(n_restarts):
            tabu_until: dict[tuple, int] = {}
            no_improve = 0

            for iteration in range(1, iters_per_restart + 1):
                global_iter += 1

                expired = [sig for sig, expiry in tabu_until.items() if expiry <= iteration]
                for sig in expired:
                    del tabu_until[sig]

                best_neighbor = None
                best_neighbor_p = float("inf")
                best_move = None

                for _ in range(max(1, self.neighbor_samples)):
                    move = sample_move(current)
                    if move is None:
                        continue

                    cand = apply_move(current, move)
                    cand_p = penalty(cand)
                    sig = inverse_signature(move)
                    is_tabu_move = sig in tabu_until
                    aspirated = cand_p < best_p

                    if is_tabu_move and not aspirated:
                        continue

                    if cand_p < best_neighbor_p:
                        best_neighbor = cand
                        best_neighbor_p = cand_p
                        best_move = move

                if best_neighbor is None:
                    no_improve += 1
                    if no_improve >= no_improve_limit:
                        break
                    continue

                current = best_neighbor
                current_p = best_neighbor_p
                tabu_until[inverse_signature(best_move)] = iteration + self.tabu_tenure

                if current_p < best_p:
                    best = list(current)
                    best_p = current_p
                    no_improve = 0
                else:
                    no_improve += 1

                if callback and (global_iter % log_interval == 0):
                    callback(global_iter, total_iters, round(best_p, 2))

                if no_improve >= no_improve_limit:
                    break

            if restart < n_restarts - 1:
                perturb_strength = 6 + restart * 2
                current = perturb(best, strength=perturb_strength)
                current_p = penalty(current)

        best, best_p = hill_climb(best, best_p, max_rounds=3)

        if callback:
            callback(total_iters, total_iters, round(best_p, 2))

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
            zip(
                staff_df["人員"],
                staff_df.get("班別群組", staff_df.get("primary_group", "")),
            )
        )
        result_df.insert(0, "班別群組", result_df.index.map(group_map))

        penalty_breakdown = penalty(best, detailed=True)
        return result_df, penalty_breakdown
