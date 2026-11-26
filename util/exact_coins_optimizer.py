"""Exact coins/step gearset optimizer — Coins objective on the shared engine.

coins/step = coin_value_per_action / expected_steps_per_action. The denominator
is the SAME shared steps() the XP objective uses; only the numerator differs.

Empirically verified against util.coin_value.compute_coin_targets_for_activity
(the coins ground truth the UI displays): coin value per action is

    (1 + double_rewards) * ( C0 + Σ_findaxis  find_stat * k_findaxis )

— multiplicative in double_rewards, LINEAR in every find axis (chest_finding,
find_gems, find_bird_nests, fine_material_finding, and ItemFindingCategory.*),
with non-negative coefficients. That makes the objective monotone in every coin
axis, so the admissible numerator_bound is just the same expression with each
axis raised by its suffix reserve.

C0 and the per-axis marginals k are activity-specific, so they're probed once
per activity in prepare() by evaluating the real coin function at basis points.
(Only the gearset RANKING matters here; the worker recomputes the displayed
coins/1k via calculate_gearset_metrics, so a constant scale on the numerator is
irrelevant.)
"""
import math

from util.exact_gearset_optimizer import ExactGearsetOptimizer

_EXACT_NODE_BUDGET = 5_000_000
last_run_budget_exhausted = False

# MITM (meet-in-the-middle) for the coins objective. The coins numerator is
# monotone in dr + every find axis (all MAX_AXES) and decreasing in the step
# axes (MIN_AXES), so the same Pareto-collapse MITM used for XP is exact here.
# Enabled with side-by-side B&B validation: _run runs BOTH, uses MITM only when
# it matches the B&B, else falls back to B&B (and logs [MITM] / [MITM-BAIL]).
# Coins has more axes than XP, so larger frontiers may bail to B&B more often —
# the validation + bail make that safe (never worse than B&B).
_MITM_ENABLED = True
_DEDUP_RING_ORDER = True
_SLOT_ORDER_STRATEGY = "tools_first"
# MITM-only: skip the side-by-side B&B (it's ~20s/activity for coins and just
# validates). MITM is proven exact for the coins objective (aggregated A_coins
# axis), so use it directly; the B&B still runs only as a bail fallback when a
# frontier exceeds the combine cap. This is where the coins speedup lands.
_MITM_VALIDATE = False

# Additive (linear) coin find axes always considered; ItemFindingCategory.*
# axes present on the character's gear are discovered per-activity in prepare().
_NAMED_FIND_AXES = ("chest_finding", "find_gems", "find_bird_nests", "fine_material_finding")
_PROBE = 0.10   # probe delta for measuring each axis's marginal coin value


class ExactCoinsOptimizer(ExactGearsetOptimizer):
    """Maximize coins/step. Steps axes are shared with the engine; the coin
    numerator axes (double_rewards + find axes) are added per-activity."""

    def __init__(self, include_chests=True):
        super().__init__()
        self.include_chests = include_chests
        self._C0 = 0.0
        self._k = {}
        # Instance axis model — extended in prepare() with the coin axes.
        self.AXES = ("W", "S_add", "S_pct", "B", "dr")
        self.MAX_AXES = ("W", "B", "dr")
        self.MIN_AXES = ("S_add", "S_pct")
        self.CAP_AXIS = "W"
        self.AXIS_MAP = {"work_efficiency": "W", "steps_add": "S_add",
                         "steps_percent": "S_pct", "double_action": "B",
                         "double_rewards": "dr"}

    def prepare(self, activity, character, skill, location, X0, S0, kappa, WE_level):
        from util.coin_value import compute_coin_targets_for_activity, COIN_TARGET, COIN_NO_CHESTS_TARGET
        target_key = COIN_TARGET if self.include_chests else COIN_NO_CHESTS_TARGET

        def coins_step(sd):
            try:
                dr_rates = activity.get_expected_drop_rate(
                    sd, location=location, target_item=None, character=character,
                    include_collectibles_from_character=False)
                ct = compute_coin_targets_for_activity(activity, dr_rates, stats=sd)
                spc = ct.get(target_key)
                return (1.0 / spc) if (spc and spc > 0 and spc != float("inf")) else 0.0
            except Exception:
                return 0.0

        # Discover ItemFindingCategory.* axes that actually appear on owned gear
        # for this skill/location (named find axes are always candidates).
        if_axes = set()
        try:
            for it in character.items:
                if not hasattr(it, "get_stats_for_skill"):
                    continue
                try:
                    st = self._get_item_stats(it, skill, location, character)
                except Exception:
                    continue
                for k in st:
                    if k.startswith("ItemFindingCategory."):
                        if_axes.add(k)
        except Exception:
            pass

        C0 = coins_step({})
        self._C0 = C0
        k = {}
        for ax in list(_NAMED_FIND_AXES) + sorted(if_axes):
            v = coins_step({ax: _PROBE})
            kv = (v - C0) / _PROBE
            if kv > 1e-12:            # keep only coin-positive, monotone axes
                k[ax] = kv
        self._k = k

        # Collapse ALL linear find-axes into ONE aggregated Pareto axis, A_coins
        # (= sum_a k[a]*s[a]). The coins numerator depends on the find-axes ONLY
        # through this weighted sum, so Pareto-dominance on
        # (W, S_add, S_pct, B, dr, A_coins) is EXACT — while keeping the MITM
        # per-group frontiers XP-sized. Adding each find axis as its own
        # dimension (~12+) exploded the frontiers past the MITM combine cap,
        # forcing a bail to the (slow, ~30s/activity) B&B. Find stats are
        # deliberately NOT added to AXIS_MAP — _vec() folds them into A_coins
        # with their k[a] weights.
        self.AXES = ("W", "S_add", "S_pct", "B", "dr", "A_coins")
        self.MAX_AXES = ("W", "B", "dr", "A_coins")

    def _vec(self, stats):
        # Base step/dr axes via AXIS_MAP, then fold the linear find-axes into a
        # single aggregated coin axis A_coins = sum_a k[a]*s[a]. Keeps the axis
        # count at 6 (XP-like) so MITM frontiers stay small and exact.
        v = super()._vec(stats)
        acoins = 0.0
        for a, kv in self._k.items():
            val = stats.get(a)
            if val:
                acoins += kv * float(val)
        v["A_coins"] = acoins
        return v

    def numerator(self, X0, s):
        return (1.0 + s["dr"]) * (self._C0 + s.get("A_coins", 0.0))

    def numerator_bound(self, X0, s, R):
        # Monotone: raise dr and the aggregated coin axis by their reserves.
        return (1.0 + s["dr"] + R.get("dr", 0.)) * (
            self._C0 + s.get("A_coins", 0.0) + R.get("A_coins", 0.0))

    def _mitm_cache_extra_key(self):
        return (
            bool(getattr(self, "_one_time", False)),
            tuple(sorted((a, round(float(v), 10)) for a, v in self._k.items())),
        )

    def _make_eval_fns(self, X0, S0, kappa, WE_level, base_stats):
        AXES = self.AXES
        iW = AXES.index("W"); iSa = AXES.index("S_add"); iSp = AXES.index("S_pct")
        iB = AXES.index("B"); idr = AXES.index("dr"); iAc = AXES.index("A_coins")
        C0 = self._C0
        step_floor = self.STEP_FLOOR
        _ceil = math.ceil
        step_cache = {}
        sc_get = step_cache.get

        def eval_sv(sv):
            w = sv[iW]; sa = sv[iSa]; sp = sv[iSp]; b = sv[iB]
            key = (w, sa, sp, b)
            st = sc_get(key)
            if st is None:
                eff = WE_level + w
                if eff > kappa:
                    eff = kappa
                stp = _ceil((S0 / (1.0 + eff)) * (1.0 + sp) + sa)
                if stp < step_floor:
                    stp = step_floor
                st = stp / (1.0 + b)
                step_cache[key] = st
            return ((1.0 + sv[idr]) * (C0 + sv[iAc])) / st

        def eval_pair(bv, cv):
            w = bv[iW] + cv[iW]; sa = bv[iSa] + cv[iSa]
            sp = bv[iSp] + cv[iSp]; b = bv[iB] + cv[iB]
            key = (w, sa, sp, b)
            st = sc_get(key)
            if st is None:
                eff = WE_level + w
                if eff > kappa:
                    eff = kappa
                stp = _ceil((S0 / (1.0 + eff)) * (1.0 + sp) + sa)
                if stp < step_floor:
                    stp = step_floor
                st = stp / (1.0 + b)
                step_cache[key] = st
            return ((1.0 + bv[idr] + cv[idr]) * (C0 + bv[iAc] + cv[iAc])) / st

        return eval_sv, eval_sv, eval_pair, eval_pair

    def _combine_best(self, FG, FA, reqs, X0, S0, kappa, WE_level, base_stats):
        """Inlined KD-blocked upper-bound-pruned coins FG × FA combine."""
        AXES = self.AXES
        nax = len(AXES)
        s0 = {a: float(base_stats.get(a, 0.)) for a in AXES}
        s0v = tuple(s0[a] for a in AXES)
        iW = AXES.index("W"); iSa = AXES.index("S_add"); iSp = AXES.index("S_pct")
        iB = AXES.index("B"); idr = AXES.index("dr"); iAc = AXES.index("A_coins")
        C0 = self._C0
        s0W = s0v[iW]; s0Sa = s0v[iSa]; s0Sp = s0v[iSp]; s0B = s0v[iB]
        s0dr = s0v[idr]; s0Ac = s0v[iAc]
        max_idxs = tuple(AXES.index(a) for a in self.MAX_AXES)
        min_idxs = tuple(AXES.index(a) for a in self.MIN_AXES)
        rk = tuple(reqs); nrk = len(rk); req_caps = tuple(reqs[k] for k in rk)
        has_reqs = nrk > 0
        step_floor = self.STEP_FLOOR
        _ceil = math.ceil

        def _extract(fr):
            out = []
            for st in fr:
                vt = st.get("vt")
                if vt is None:
                    v = st["v"]; vt = tuple(v[a] for a in AXES)
                kwt = st.get("kwt")
                if kwt is None:
                    kw = st["kw"]; kwt = tuple(kw[k] for k in rk) if has_reqs else ()
                comp = st["comp"]
                out.append((vt, kwt, -len(comp), st.get("tb", 0.0), comp))
            return out

        FGv = _extract(FG); FAv = _extract(FA)
        if not FGv or not FAv:
            return float("-inf"), None, None
        if has_reqs:
            max_kwA = tuple(max(x[1][j] for x in FAv) for j in range(nrk))
            FGv = [x for x in FGv if all(x[1][j] + max_kwA[j] >= req_caps[j] for j in range(nrk))]
            if not FGv:
                return float("-inf"), None, None
            max_kwG = tuple(max(x[1][j] for x in FGv) for j in range(nrk))
            FAv = [x for x in FAv if all(x[1][j] + max_kwG[j] >= req_caps[j] for j in range(nrk))]
            if not FAv:
                return float("-inf"), None, None

        boxA = [0.0] * nax; boxG = [0.0] * nax
        for j in max_idxs:
            boxA[j] = max(x[0][j] for x in FAv); boxG[j] = max(x[0][j] for x in FGv)
        for j in min_idxs:
            boxA[j] = min(x[0][j] for x in FAv); boxG[j] = min(x[0][j] for x in FGv)

        step_cache = {}
        def _steps(w, sa, sp, b):
            key = (w, sa, sp, b)
            st = step_cache.get(key)
            if st is None:
                eff = WE_level + w
                if eff > kappa:
                    eff = kappa
                stp = _ceil((S0 / (1.0 + eff)) * (1.0 + sp) + sa)
                if stp < step_floor:
                    stp = step_floor
                st = stp / (1.0 + b)
                step_cache[key] = st
            return st

        FG_ub = [
            (*x, ((1.0 + s0dr + x[0][idr] + boxA[idr]) * (C0 + s0Ac + x[0][iAc] + boxA[iAc]))
                 / _steps(s0W + x[0][iW] + boxA[iW], s0Sa + x[0][iSa] + boxA[iSa],
                          s0Sp + x[0][iSp] + boxA[iSp], s0B + x[0][iB] + boxA[iB]))
            for x in FGv
        ]
        FA_ub = [
            (*x, ((1.0 + s0dr + boxG[idr] + x[0][idr]) * (C0 + s0Ac + boxG[iAc] + x[0][iAc]))
                 / _steps(s0W + boxG[iW] + x[0][iW], s0Sa + boxG[iSa] + x[0][iSa],
                          s0Sp + boxG[iSp] + x[0][iSp], s0B + boxG[iB] + x[0][iB]))
            for x in FAv
        ]
        FG_ub.sort(key=lambda x: x[5], reverse=True)
        FA_ub.sort(key=lambda x: x[5], reverse=True)

        best_r = float("-inf"); best_tb = float("-inf"); best_nfill = float("-inf")
        best_load = None; best_gv = None; best_av = None

        for gv, kG, nfG, tbG, cG, _ in FG_ub[:16]:
            gW = s0W + gv[iW]; gSa = s0Sa + gv[iSa]; gSp = s0Sp + gv[iSp]; gB = s0B + gv[iB]
            gdr = s0dr + gv[idr]; gAc = s0Ac + gv[iAc]
            for av, kA, nfA, tbA, cA, _ in FA_ub[:16]:
                if has_reqs and any(kG[j] + kA[j] < req_caps[j] for j in range(nrk)):
                    continue
                r = ((1.0 + gdr + av[idr]) * (C0 + gAc + av[iAc])) / _steps(
                    gW + av[iW], gSa + av[iSa], gSp + av[iSp], gB + av[iB])
                if r >= best_r:
                    tb = tbG + tbA; nf = nfG + nfA
                    if r > best_r or (tb, nf) > (best_tb, best_nfill):
                        best_r = r; best_tb = tb; best_nfill = nf
                        best_load = list(cG) + list(cA); best_gv = gv; best_av = av

        if best_load is not None:
            eps_r = best_r / (1.0 + 1e-12) if best_r > 0 else best_r * (1.0 + 1e-12)
            FG_ub = [x for x in FG_ub if x[5] >= eps_r]
            FA_ub = [x for x in FA_ub if x[5] >= eps_r]

        blocks = self._build_kd_blocks(FA_ub, max_idxs, min_idxs, has_reqs, nrk, leaf_size=32)

        for gv, kG, nfG, tbG, cG, ubG in FG_ub:
            if best_load is not None and ubG * (1.0 + 1e-12) < best_r:
                break
            gW = s0W + gv[iW]; gSa = s0Sa + gv[iSa]; gSp = s0Sp + gv[iSp]; gB = s0B + gv[iB]
            gdr = s0dr + gv[idr]; gAc = s0Ac + gv[iAc]
            for b_box, b_kw, b_items in blocks:
                if has_reqs:
                    ok = True
                    for j in range(nrk):
                        if kG[j] + b_kw[j] < req_caps[j]:
                            ok = False
                            break
                    if not ok:
                        continue
                if best_load is not None:
                    bub = ((1.0 + gdr + b_box[idr]) * (C0 + gAc + b_box[iAc])) / _steps(
                        gW + b_box[iW], gSa + b_box[iSa], gSp + b_box[iSp], gB + b_box[iB])
                    if bub * (1.0 + 1e-12) < best_r:
                        continue
                for av, kA, nfA, tbA, cA, _ in b_items:
                    if has_reqs:
                        ok = True
                        for j in range(nrk):
                            if kG[j] + kA[j] < req_caps[j]:
                                ok = False
                                break
                        if not ok:
                            continue
                    r = ((1.0 + gdr + av[idr]) * (C0 + gAc + av[iAc])) / _steps(
                        gW + av[iW], gSa + av[iSa], gSp + av[iSp], gB + av[iB])
                    if r >= best_r:
                        tb = tbG + tbA; nf = nfG + nfA
                        if r > best_r or (tb, nf) > (best_tb, best_nfill):
                            best_r = r; best_tb = tb; best_nfill = nf
                            best_load = list(cG) + list(cA); best_gv = gv; best_av = av
        if best_load is None:
            return float("-inf"), None, None
        bsv = tuple(s0v[j] + best_gv[j] + best_av[j] for j in range(nax))
        return best_r, best_load, dict(zip(AXES, bsv))


def exact_coins_optimize(activity, character, skill=None, location=None, base_stats=None,
                         pets=None, consumables=None, input_item=None, include_chests=True,
                         tool_frontier_cache=None, mitm_cache=None):
    """Return {slot: Item|None} maximizing coins/step, or None to fall back."""
    opt = ExactCoinsOptimizer(include_chests=include_chests)
    opt.NODE_BUDGET = _EXACT_NODE_BUDGET
    opt._MITM_ENABLED = _MITM_ENABLED
    opt.DEDUP_RING_ORDER = _DEDUP_RING_ORDER
    opt.SLOT_ORDER_STRATEGY = _SLOT_ORDER_STRATEGY
    opt._MITM_VALIDATE = _MITM_VALIDATE
    result = opt.optimize(activity, character, skill=skill, location=location,
                          base_stats=base_stats, pets=pets, consumables=consumables,
                          input_item=input_item,
                          tool_frontier_cache=tool_frontier_cache,
                          mitm_cache=mitm_cache)
    globals()["last_run_budget_exhausted"] = opt.last_run_budget_exhausted
    return result
