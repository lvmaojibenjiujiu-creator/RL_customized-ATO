
import math, random, heapq, argparse, os, json, time, copy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


_HAVE_GRB = True
try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    _HAVE_GRB = False
    _GRB_IMPORT_ERR = str(e)



def softplus_stable_np(x):
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))

def to_int_nonneg_safe(arr_f: np.ndarray, cap: Optional[float] = None) -> np.ndarray:
    arr = np.nan_to_num(arr_f, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.rint(arr)
    if cap is not None:
        arr = np.clip(arr, 0.0, float(cap))
    else:
        arr = np.clip(arr, 0.0, np.inf)
    return arr.astype(np.int64)

def to_builtin(x):
    if isinstance(x, dict):
        return {str(k): to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [to_builtin(v) for v in x]
    if isinstance(x, (np.integer, np.int_, np.int64)):
        return int(x)
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if (torch is not None) and isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return x.detach().cpu().item()
        return x.detach().cpu().tolist()
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            return x.item()
        except Exception:
            pass
    return x



@dataclass
class Order:
    product_id: int
    qty: int
    bom_per_unit: np.ndarray
    t_arrive: int
    t_design_done: int
    t_due: int
    realized: bool = False
    assembled_qty: int = 0
    @property
    def remaining(self) -> int:
        return max(0, self.qty - self.assembled_qty)

@dataclass
class State:
    t: int
    component_inventory: np.ndarray
    pipeline_snapshot: Dict[int, List[Tuple[int, int]]]
    realized_backlog: np.ndarray
    unrealized_backlog: np.ndarray
    overdue_backlog: np.ndarray
    last_demand: np.ndarray



def make_component_sets_and_bom_templates(n, m, rng, common_ratio=0.3,
                                          per_product_span=(8, 14),
                                          mean_range_specific=(1, 5),
                                          mean_range_common=(2, 7),
                                          sigma_pct=0.3):
    common_count = max(0, int(round(common_ratio * m)))
    common_components = sorted(rng.choice(m, size=common_count, replace=False).tolist())
    product_sets = {}
    for i in range(n):
        k = int(rng.integers(per_product_span[0], per_product_span[1] + 1))
        chosen = set(common_components)
        remaining = k - len(chosen)
        if remaining > 0:
            pool = [c for c in range(m) if c not in chosen]
            if len(pool) > 0:
                chosen.update(rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist())
        product_sets[i] = sorted(list(chosen))
    bom_templates = {}
    for i in range(n):
        mask = np.zeros(m, dtype=int)
        mask[product_sets[i]] = 1
        mean = np.zeros(m, dtype=float)
        if len(common_components) > 0:
            idx = np.array(common_components, dtype=int)
            mean[idx] = rng.integers(mean_range_common[0], mean_range_common[1] + 1, size=len(idx))
        spec_idx = np.where(mask == 1)[0]
        spec_idx = np.setdiff1d(spec_idx, np.array(common_components, dtype=int))
        if len(spec_idx) > 0:
            mean[spec_idx] = rng.integers(mean_range_specific[0], mean_range_specific[1] + 1, size=len(spec_idx))
        mean = mean * mask
        std = sigma_pct * np.maximum(1.0, mean)
        bom_templates[i] = (mean, std, mask.astype(bool))
    return bom_templates, common_components, product_sets

def make_backorder_costs_strictly_higher_per_unit(bom_templates, comp_order_cost, comp_hold_cost, rng, markup_range=(1.5, 3.0)):
    n = len(bom_templates)
    boc = np.zeros(n, dtype=float)
    base_fallback = max(1.0, float(np.mean(comp_order_cost)))
    lo, hi = markup_range
    for i in range(n):
        mean_i = np.asarray(bom_templates[i][0], dtype=float)
        unit_cost = float(np.dot(mean_i, comp_order_cost) + np.dot((mean_i>0).astype(float), comp_hold_cost)*0.0)
        unit_cost = unit_cost if unit_cost > 0 else base_fallback
        markup = float(rng.uniform(lo, hi))
        boc[i] = max(1.0, unit_cost * markup)
    return boc

def build_example_config(n, m, seed=2025, common_ratio=0.3) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    bom_tmpl, _, _ = make_component_sets_and_bom_templates(n, m, rng, common_ratio=common_ratio)
    demand_rates = rng.uniform(2.0, 8.0, size=n)
    design_leads = rng.integers(2, 4, size=n)
    comp_order_cost = rng.uniform(0.1, 0.3, size=m)
    comp_hold_cost  = rng.uniform(0.01, 0.03, size=m)
    init_inv        = rng.integers(10, 50, size=m)
    backorder_cost  = make_backorder_costs_strictly_higher_per_unit(bom_tmpl, comp_order_cost, comp_hold_cost, rng, (1.5, 3.0))
    for i in range(n):
        mean_i = np.asarray(bom_tmpl[i][0], dtype=float)
        unit_cost = float(np.dot(mean_i, comp_order_cost))
        assert backorder_cost[i] > unit_cost, f"backorder_cost[{i}]={backorder_cost[i]:.3f} <= unit={unit_cost:.3f}"
    return dict(
        n_products=n, m_components=m,
        bom_templates={i:(bom_tmpl[i][0].tolist(), bom_tmpl[i][1].tolist(), bom_tmpl[i][2].tolist()) for i in range(n)},
        design_leads=design_leads.tolist(),
        delivery_window=2,
        demand_rates=demand_rates.tolist(),
        comp_order_cost=comp_order_cost.tolist(),
        comp_hold_cost=comp_hold_cost.tolist(),
        backorder_cost=backorder_cost.tolist(),
        init_component_inventory=init_inv.tolist(),
        meta={"seed": seed, "common_ratio": common_ratio},
    )



class ATOEnvPPO:

    def __init__(self, cfg: Dict[str, Any], rng_seed: Optional[int] = None, lt_range=(1, 3),
                 reward_scale: float = 1000.0,
                 order_deadband_tau: float = 1.0, order_temp: float = 0.5, order_cap: float = 400.0,
                 use_lblr: bool = True):
        self.cfg = cfg
        self.rng = np.random.default_rng(rng_seed)
        self.lt_lo, self.lt_hi = lt_range
        self.use_lblr = bool(use_lblr)

        self.n = int(cfg["n_products"])
        self.m = int(cfg["m_components"])
        self.design_leads = np.array(cfg["design_leads"], dtype=int)
        self.delivery_window = int(cfg["delivery_window"])
        self.comp_order_cost = np.array(cfg["comp_order_cost"], dtype=float)
        self.comp_hold_cost  = np.array(cfg["comp_hold_cost"], dtype=float)
        self.backorder_cost  = np.array(cfg["backorder_cost"], dtype=float)
        self.init_component_inventory = np.array(cfg["init_component_inventory"], dtype=int)

        self.demand_rates = np.array(cfg["demand_rates"], dtype=float)
        self.bom_templates = {
            int(i): (
                np.array(t[0], dtype=float),
                np.array(t[1], dtype=float),
                np.array(t[2], dtype=int).astype(bool),
            )
            for i, t in cfg["bom_templates"].items()
        }

 
        self.co_mat = np.zeros((self.m, self.m), dtype=float)
        for i in range(self.n):
            _, _, mask = self.bom_templates[i]
            idx = np.where(mask)[0]
            for a in idx:
                for b in idx:
                    if a != b: self.co_mat[a, b] = 1.0


        self.alpha_ema = 0.2
        self.ema_mean = np.zeros(self.n)
        self.ema_var  = np.ones(self.n)
        self._ema_initialized = False

        self.order_deadband_tau = float(order_deadband_tau)
        self.order_temp = float(order_temp)
        self.order_cap  = float(order_cap)
        self.reward_scale = float(reward_scale)

        self.reset()

    def _draw_lead_time_for_component(self, j: int) -> int:
        return int(self.rng.integers(self.lt_lo, self.lt_hi + 1))

    def _advance_pipeline_and_receive(self):
        self.last_arrivals[:] = 0
        for j in range(self.m):
            new_list, arrivals = [], 0
            for eta, q in self.pipeline[j]:
                eta -= 1
                if eta <= 0:
                    arrivals += q
                else:
                    new_list.append((eta, q))
            self.pipeline[j] = new_list
            if arrivals > 0:
                self.component_inventory[j] += int(arrivals)
                self.last_arrivals[j] = int(arrivals)

    def _release_realized_orders(self):
        remained = []
        for od in self.unrealized_orders:
            if od.t_design_done <= self.t:
                od.realized = True
                self.realized_orders.append(od)
            else:
                remained.append(od)
        self.unrealized_orders = remained

    def _ingest_new_orders_for_current_t(self):
        D = self.rng.poisson(lam=self.demand_rates, size=self.n).astype(int)
        self.last_demand = D.copy()
        if not self._ema_initialized:
            self.ema_mean = D.astype(float)
            self.ema_var  = np.maximum(1.0, self.ema_mean)
            self._ema_initialized = True
        else:
            prev_mean = self.ema_mean.copy()
            a = float(self.alpha_ema)
            self.ema_mean = (1.0 - a) * self.ema_mean + a * D
            self.ema_var  = (1.0 - a) * self.ema_var  + a * (D - prev_mean) ** 2
        self.ep_total_demand += int(D.sum())
        for i in range(self.n):
            qty = int(D[i]); 
            if qty <= 0: continue
            mean, std, mask = self.bom_templates[i]
            bom = np.zeros(self.m, dtype=int)
            if np.any(mask):
                z = self.rng.normal(loc=mean[mask], scale=std[mask])
                z = np.maximum(0.0, z)
                bom[mask] = np.rint(z).astype(int)
            t_design = self.t + int(self.design_leads[i])
            t_due    = t_design + int(self.delivery_window)
            self.unrealized_orders.append(Order(
                product_id=i, qty=qty, bom_per_unit=bom,
                t_arrive=self.t, t_design_done=t_design, t_due=t_due
            ))


    def _compute_costs(self, order_cost: float) -> Dict[str, float]:
        hold_cost_by_comp = self.component_inventory.astype(float) * self.comp_hold_cost
        hold_cost = float(hold_cost_by_comp.sum())
        overdue_by_prod = np.zeros(self.n, dtype=int)
        for od in self.realized_orders:
            if od.remaining > 0 and self.t > od.t_due:
                overdue_by_prod[od.product_id] += od.remaining
     
        late_cost = float((self.backorder_cost * overdue_by_prod.astype(float)).sum())
        total = order_cost + hold_cost + late_cost
        return dict(order_cost=order_cost, hold_cost=hold_cost, late_cost=late_cost, total_cost=total)

    def _build_state(self) -> State:
        realized_bklg  = np.zeros(self.n, dtype=int)
        unrealized_bklg= np.zeros(self.n, dtype=int)
        overdue_bklg   = np.zeros(self.n, dtype=int)
        for od in self.unrealized_orders:
            unrealized_bklg[od.product_id] += od.qty
        for od in self.realized_orders:
            realized_bklg[od.product_id] += od.remaining
            if od.remaining > 0 and self.t > od.t_due:
                overdue_bklg[od.product_id] += od.remaining
        snap = {j: list(self.pipeline[j]) for j in range(self.m)}  
        return State(
            t=int(self.t),
            component_inventory=self.component_inventory.copy(),
            pipeline_snapshot=snap,
            realized_backlog=realized_bklg,
            unrealized_backlog=unrealized_bklg,
            overdue_backlog=overdue_bklg,
            last_demand=self.last_demand.copy(),
        )


    def _lt_demand_forecast(self, state: State, L_eff: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m = self.m
        mu_comp  = np.zeros(m, dtype=float)
        var_comp = np.zeros(m, dtype=float)
        ema_m = self.ema_mean.astype(float)
        ema_v = self.ema_var.astype(float)
        Ld    = self.design_leads.astype(float)
        for i in range(self.n):
            mean_i, _, mask_i = self.bom_templates[i]
            idx = np.where(mask_i)[0]
            if idx.size == 0: continue
            window_ij = np.maximum(0.0, L_eff[idx] - Ld[i])  # 只计有效揭示窗口
            if np.all(window_ij <= 0): continue
            bom_mean = mean_i[idx].astype(float)
            mu_comp[idx]  += bom_mean * (ema_m[i] * window_ij)
            var_comp[idx] += (bom_mean ** 2) * (ema_v[i] * window_ij)
        sigma_comp = np.sqrt(np.maximum(1e-8, var_comp))
        return mu_comp, sigma_comp

    def _compute_on_order_all(self) -> np.ndarray:
        on_order = np.zeros(self.m, dtype=float)
        for j in range(self.m):
            if len(self.pipeline[j]) > 0:
                on_order[j] = sum(q for eta, q in self.pipeline[j])
        return on_order

    def _component_equiv_backlog(self, state: State) -> np.ndarray:
        B = np.zeros(self.m, dtype=float)
        for i in range(self.n):
            mean_i, _, mask_i = self.bom_templates[i]
            if np.any(mask_i):
                B[mask_i] += state.realized_backlog[i] * mean_i[mask_i]
        return B

    def _place_component_orders(self, q_order_cont: np.ndarray) -> float:
        q = to_int_nonneg_safe(q_order_cont, cap=1e12)
        self.last_orders = q.copy()
        cost = float(np.dot(q.astype(float), self.comp_order_cost))
        for j, qty in enumerate(q):
            if qty <= 0: continue
            lt = max(1, int(self._draw_lead_time_for_component(j)))
            self.pipeline[j].append((lt, int(qty)))
        return cost

    def _assemble_with_lambda(self, lam: np.ndarray) -> np.ndarray:
 
        assembled_by_prod = np.zeros(self.n, dtype=int)
        self.last_ontime_delivered = 0
        if len(self.realized_orders) == 0:
            return assembled_by_prod
        # 计算 b_i^new
        b_new = np.array(self.backorder_cost, dtype=float)
        extra = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            mean_i, _, mask_i = self.bom_templates[i]
            if np.any(mask_i):
                extra[i] = float(self.comp_hold_cost[mask_i].sum())
        b_new = b_new + extra

        heap = []; eta = 0.25
        for idx, od in enumerate(self.realized_orders):
            if od.remaining <= 0: continue
            i = od.product_id
            age = max(0, self.t - od.t_due)
            urg = math.exp(eta * age)
            res_cost = float(np.dot(od.bom_per_unit.astype(float), lam))
            score = urg * float(b_new[i]) - res_cost
            heap.append((-score, idx, urg, res_cost))
        if not heap: return assembled_by_prod
        heapq.heapify(heap)
        inv = self.component_inventory
        while heap:
            _, idx, urg, res_cost = heapq.heappop(heap)
            od = self.realized_orders[idx]; i = od.product_id
            if od.remaining <= 0: continue
            bom = od.bom_per_unit; pos = np.where(bom > 0)[0]
            feas = (min(inv[j] // bom[j] for j in pos) if pos.size > 0 else od.remaining)
            x = int(min(od.remaining, feas))
            if x <= 0: continue
            if pos.size > 0: inv[pos] -= bom[pos] * x
            od.assembled_qty += x; assembled_by_prod[i] += x
            self.ep_total_delivered += x
            if self.t <= od.t_due:
                self.ep_on_time_delivered += x
                self.last_ontime_delivered += x
            if od.remaining > 0:
                heapq.heappush(heap, (-(urg*float(b_new[i]) - res_cost), idx, urg, res_cost))
        return assembled_by_prod

    def reset(self) -> State:
        self.component_inventory = self.init_component_inventory.copy()
        self.pipeline = {j: [] for j in range(self.m)}
        self.unrealized_orders = []; self.realized_orders = []
        self.t = 0
        self.last_demand = np.zeros(self.n, dtype=int)
        self.last_arrivals = np.zeros(self.m, dtype=int)
        self.last_orders   = np.zeros(self.m, dtype=int)

        self._ema_initialized = False
        self.ema_mean[:] = 0.0
        self.ema_var[:]  = 1.0
        self.ep_total_demand = 0
        self.ep_total_delivered = 0
        self.ep_on_time_delivered = 0
        return self._build_state()

    def step(self, action_params: np.ndarray) -> Tuple[State, float, Dict[str, float]]:
        self._advance_pipeline_and_receive()
        self._release_realized_orders()
        self._ingest_new_orders_for_current_t()

        a = np.asarray(action_params, dtype=float)
        m = self.m

   
        if self.use_lblr:
            theta_dim = 6
        else:
            theta_dim = 2
        assert a.shape[0] == m + theta_dim, f"action dim={a.shape[0]} != m + {theta_dim}"
        lam_raw = a[:m].copy()
        theta   = a[m:].copy()

        lam = softplus_stable_np(lam_raw)
        lam = np.clip(lam, 0.0, 1e6)

        L_eff = np.full(self.m, 0.5*(self.lt_lo + self.lt_hi), dtype=float)
        st_now = self._build_state()
        mu_lt, sigma_lt = self._lt_demand_forecast(st_now, L_eff)


        U_comp = np.zeros(self.m, dtype=float)
        for i in range(self.n):
            mean_i, _, mask_i = self.bom_templates[i]
            if np.any(mask_i):
                U_comp[mask_i] += st_now.overdue_backlog[i] * mean_i[mask_i]

        on_order = self._compute_on_order_all()
        B_comp   = self._component_equiv_backlog(st_now)
        IP = self.component_inventory.astype(float) + on_order - B_comp

    
        if self.use_lblr:
            z_lev = softplus_stable_np(theta[:3])   # z0,z1,z2 >=0
            k_lev = softplus_stable_np(theta[3:6])  # k0,k1,k2 >=0
            r = IP / (mu_lt + 1.0)
            cls = np.zeros(self.m, dtype=int)
            cls[r < 1/3] = 0
            cls[(r >= 1/3) & (r < 2/3)] = 1
            cls[r >= 2/3] = 2
            S = mu_lt + z_lev[cls] * sigma_lt + k_lev[cls] * U_comp
        else:
            z = softplus_stable_np(theta[0])
            k = softplus_stable_np(theta[1])
            S = mu_lt + z * sigma_lt + k * U_comp

   
        drive = S - IP
        drive_adj = self.order_temp * (drive - self.order_deadband_tau)
        q_order_cont = softplus_stable_np(drive_adj)
        q_order_cont = np.clip(q_order_cont, 0.0, self.order_cap)
        orders_sum = float(np.sum(q_order_cont))
        order_cost = self._place_component_orders(q_order_cont)

    
        assembled_by_prod = self._assemble_with_lambda(lam)
        delivered_step = int(assembled_by_prod.sum())
        ontime_step = int(self.last_ontime_delivered)

 
        costs = self._compute_costs(order_cost)
        reward_raw = - float(costs["total_cost"])
        reward = reward_raw / max(1.0, self.reward_scale)


        surplus = np.maximum(0.0, IP - S).sum()
        shortage = np.maximum(0.0, S - IP).sum()
        info = dict(costs)
        info.update({
            "demand_step": int(self.last_demand.sum()),
            "delivered_step": delivered_step,
            "ontime_step": ontime_step,
            "orders_sum": float(orders_sum),
            "surplus_sum": float(surplus),
            "shortage_sum": float(shortage),
            "redundancy": float(surplus / (orders_sum + 1e-9)),
            "mu_lt_l1": float(np.sum(mu_lt)),
            "sigma_lt_l1": float(np.sum(sigma_lt)),
            "overdue_pull_l1": float(np.sum(U_comp)),
        })
        self.t += 1
        return self._build_state(), reward, info


class CompHist:
    def __init__(self, m, L):
        self.L = L
        self.buf = np.zeros((L, m, 2), dtype=float)
    def push(self, orders_j: np.ndarray, arrivals_j: np.ndarray):
        self.buf = np.roll(self.buf, shift=-1, axis=0)
        self.buf[-1, :, 0] = orders_j
        self.buf[-1, :, 1] = arrivals_j
    def seq(self) -> np.ndarray:
        return np.transpose(self.buf, (1, 0, 2))

def inbound_histogram(pipeline_snapshot: Dict[int, List[Tuple[int,int]]], m: int):
    inbound_all = np.zeros(m)
    for j in range(m):
        for eta, q in pipeline_snapshot[j]:
            inbound_all[j] += q
    return inbound_all

def build_graph_data(env: ATOEnvPPO, state: State, hist: CompHist, device: torch.device) -> Data:
    n, m = env.n, env.m
    inbound_all = inbound_histogram(state.pipeline_snapshot, m)
    I  = state.component_inventory.astype(float)


    L_eff = np.full(m, 0.5*(env.lt_lo+env.lt_hi), dtype=float)
    dem_lt_cons = env._lt_demand_forecast(state, L_eff)[0]

    comp_avail  = env.co_mat @ (I + inbound_all)

    comp_feat = np.stack([
        I, inbound_all, dem_lt_cons,
        np.array(env.comp_order_cost), np.array(env.comp_hold_cost),
        comp_avail
    ], axis=1)
    d_comp = comp_feat.shape[1]

    realized = state.realized_backlog.astype(float)
    unreal   = state.unrealized_backlog.astype(float)
    overdue  = state.overdue_backlog.astype(float)
    back_c   = np.array(env.backorder_cost, dtype=float)
    prod_feat = np.stack([realized, unreal, overdue, back_c], axis=1)
    d_prod = prod_feat.shape[1]

    d_node = max(d_comp, d_prod)
    comp_pad = np.zeros((m, d_node)); comp_pad[:, :d_comp] = comp_feat
    prod_pad = np.zeros((n, d_node)); prod_pad[:, :d_prod] = prod_feat
    x = np.vstack([comp_pad, prod_pad])
    node_type = np.concatenate([np.zeros(m, dtype=int), np.ones(n, dtype=int)])

 
    real_units = np.zeros(n, dtype=float)
    real_need  = np.zeros((n, m), dtype=float)
    for od in env.realized_orders:
        i = od.product_id; rem = float(od.remaining)
        if rem <= 0: continue
        bom = od.bom_per_unit.astype(float)
        real_units[i] += rem
        real_need[i, :] += rem * bom
    realized_mean = np.zeros((n, m), dtype=float)
    mask_nz = real_units > 0
    if np.any(mask_nz):
        realized_mean[mask_nz, :] = real_need[mask_nz, :] / (real_units[mask_nz].reshape(-1, 1) + 1e-8)

    edges_src, edges_dst, eattr = [], [], []
    for i in range(n):
        mean, std, mask = env.bom_templates[i]
        idx = np.where(mask)[0]
        for j in idx:
            edges_src.append(m + i); edges_dst.append(j)
            eattr.append([mean[j], std[j], realized_mean[i, j], real_need[i, j]])
    if len(edges_src) == 0:
        edges_src, edges_dst, eattr = [0], [0], [[0.0, 0.0, 0.0, 0.0]]

    edge_index = np.vstack([np.array(edges_src, dtype=int), np.array(edges_dst, dtype=int)])
    edge_attr  = np.array(eattr, dtype=float)

    temp_seq_comp = hist.seq()

    data = Data(
        x=torch.tensor(x, dtype=torch.float32, device=device),
        node_type=torch.tensor(node_type, dtype=torch.long, device=device),
        edge_index=torch.tensor(edge_index, dtype=torch.long, device=device),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32, device=device),
    )
    data.comp_idx = torch.arange(0, m, dtype=torch.long, device=device)
    data.prod_idx = torch.arange(m, m + n, dtype=torch.long, device=device)
    data.temp_seq_comp = torch.tensor(temp_seq_comp, dtype=torch.float32, device=device)
    data.inv_col = torch.tensor(0, dtype=torch.long, device=device)
    return data



class TemporalGRU(nn.Module):
    def __init__(self, in_dim, hid):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hid, num_layers=1, batch_first=True)
    def forward(self, seq):
        _, h = self.gru(seq)
        return h.squeeze(0)

class GATEncoder(nn.Module):
    def __init__(self, node_in_dim, edge_dim, hid=64, out=64):
        super().__init__()
        self.type_emb = nn.Embedding(2, 8)
        self.lin_in   = nn.Linear(node_in_dim + 8, hid)
        self.gat1     = GATv2Conv(hid, hid, edge_dim=edge_dim, heads=2, concat=False)
        self.gat2     = GATv2Conv(hid, out, edge_dim=edge_dim, heads=2, concat=False)
        self.lin_g    = nn.Linear(out, 64)
    def forward(self, x, node_type, edge_index, edge_attr):
        x = torch.cat([x, self.type_emb(node_type)], dim=-1)
        x = F.relu(self.lin_in(x))
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = self.gat2(x, edge_index, edge_attr)
        g = torch.tanh(self.lin_g(x.mean(dim=0)))
        return x, g

class ActorGNN(nn.Module):

    def __init__(self, node_dim, edge_dim, use_lblr: bool, temp_in=2, temp_hid=32):
        super().__init__()
        self.use_lblr = use_lblr
        self.temp = TemporalGRU(in_dim=temp_in, hid=temp_hid)
        self.gnn  = GATEncoder(node_in_dim=node_dim + temp_hid, edge_dim=edge_dim, hid=64, out=64)
        self.mu_lambda_head = nn.Sequential(nn.Linear(64 + 64, 64), nn.ReLU(), nn.Linear(64, 1))
        theta_out = 6 if use_lblr else 2
        self.mu_theta_head  = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, theta_out))
        self.logstd_lambda  = nn.Parameter(torch.full((1,), -0.2))
        self.logstd_theta   = nn.Parameter(torch.full((theta_out,), -0.3))
    def forward(self, data: Data):
        m = data.comp_idx.numel()
        Ht = self.temp(data.temp_seq_comp)
        add = torch.zeros(data.x.size(0), Ht.size(1), device=data.x.device)
        add[data.comp_idx] = Ht
        x_aug = torch.cat([data.x, add], dim=-1)
        H, g = self.gnn(x_aug, data.node_type, data.edge_index, data.edge_attr)
        Hc = H[data.comp_idx]
        g_rep = g.unsqueeze(0).repeat(m, 1)
        mu_l = self.mu_lambda_head(torch.cat([Hc, g_rep], dim=-1)).squeeze(-1)   # [m]
        mu_t = self.mu_theta_head(g)                                             # [T]
        ls_l = self.logstd_lambda.expand(m)
        ls_t = self.logstd_theta
        return mu_l, ls_l, mu_t, ls_t
    def act(self, data: Data, deterministic=False):
        mu_l, ls_l, mu_t, ls_t = self.forward(data)
        if deterministic:
            a_l, a_t, logp = mu_l, mu_t, None
        else:
            std_l = ls_l.exp(); std_t = ls_t.exp()
            dist_l = torch.distributions.Normal(mu_l, std_l)
            dist_t = torch.distributions.Normal(mu_t, std_t)
            a_l = dist_l.rsample(); a_t = dist_t.rsample()
            logp = dist_l.log_prob(a_l).sum() + dist_t.log_prob(a_t).sum()
        a = torch.cat([a_l, a_t], dim=0)
        return a, logp

class CriticGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, temp_in=2, temp_hid=32):
        super().__init__()
        self.temp = TemporalGRU(in_dim=temp_in, hid=temp_hid)
        self.gnn  = GATEncoder(node_in_dim=node_dim + temp_hid, edge_dim=edge_dim, hid=64, out=64)
        self.v_head = nn.Sequential(nn.Linear(64 + 64, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, data: Data):
        m = data.comp_idx.numel()
        Ht = self.temp(data.temp_seq_comp)
        add = torch.zeros(data.x.size(0), Ht.size(1), device=data.x.device)
        add[data.comp_idx] = Ht
        x_aug = torch.cat([data.x, add], dim=-1)
        H, g = self.gnn(x_aug, data.node_type, data.edge_index, data.edge_attr)
        Hc = H[data.comp_idx]
        v  = self.v_head(torch.cat([Hc.mean(dim=0), g], dim=-1)).squeeze(-1)
        return v



@dataclass
class PPOStep:
    state_data: Data
    action: torch.Tensor
    logp: float
    value: float
    reward: float
    done: float

class PPOBuffer:
    def __init__(self):
        self.storage: List[PPOStep] = []
    def push(self, data, action, logp, value, reward, done):
        self.storage.append(PPOStep(
            state_data=data,
            action=action.detach().cpu(),
            logp=float(logp) if logp is not None else float(0.0),
            value=float(value),
            reward=float(reward),
            done=float(done)
        ))
    def __len__(self): return len(self.storage)
    def __getitem__(self, idx): return self.storage[idx]
    def clear(self): self.storage.clear()

def compute_gae_from_buffer(buf: PPOBuffer, last_value: float, gamma=0.99, lam=0.95, device="cpu"):
    T = len(buf)
    rewards = torch.tensor([it.reward for it in buf], dtype=torch.float32, device=device)
    dones   = torch.tensor([it.done   for it in buf], dtype=torch.float32, device=device)
    values  = torch.tensor([it.value  for it in buf], dtype=torch.float32, device=device)
    adv = torch.zeros(T, dtype=torch.float32, device=device)
    last_adv = 0.0
    for t in reversed(range(T)):
        v_next = last_value if t==T-1 else values[t+1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma*v_next*mask - values[t]
        last_adv = delta + gamma*lam*mask*last_adv
        adv[t] = last_adv
    ret = adv + values
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret, values

def ppo_update_gnn(actor: ActorGNN, critic: CriticGNN, optimizer, buf: PPOBuffer,
                   clip_ratio=0.2, vf_coef=0.1, ent_coef=1e-3,
                   epochs=20, minibatch=64, max_grad_norm=0.7, device="cpu",
                   use_value_clipping=True):
    if len(buf) == 0: return 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        v_last = critic.forward(buf[-1].state_data)
        last_value = float(v_last.detach().cpu())
    adv, ret, v_old_all = compute_gae_from_buffer(buf, last_value, gamma=0.99, lam=0.95, device=device)

    idx = list(range(len(buf)))
    total_pg_ep, total_v_ep, n_mb = 0.0, 0.0, 0
    for _ in range(epochs):
        random.shuffle(idx)
        for start in range(0, len(idx), minibatch):
            sl = idx[start:start+minibatch]
            optimizer.zero_grad()
            pg_loss_acc = 0.0; v_loss_acc = 0.0

            for k in sl:
                it = buf[k]
                data = it.state_data
                old_logp = torch.tensor(it.logp,  dtype=torch.float32, device=device)
                old_act  = it.action.to(device)
                v_pred = critic.forward(data)

                mu_l, ls_l, mu_t, ls_t = actor.forward(data)
                std_l = ls_l.exp(); std_t = ls_t.exp()
                m = mu_l.numel()
                a_l = old_act[:m]; a_t = old_act[m:]
                dist_l = torch.distributions.Normal(mu_l, std_l)
                dist_t = torch.distributions.Normal(mu_t, std_t)
                logp = dist_l.log_prob(a_l).sum() + dist_t.log_prob(a_t).sum()
                entropy = (dist_l.entropy().mean() + dist_t.entropy().mean())

                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * adv[k]
                surr2 = torch.clamp(ratio, 1.0-clip_ratio, 1.0+clip_ratio) * adv[k]
                pg_loss = -torch.min(surr1, surr2)

                target = ret[k].detach()
                if use_value_clipping:
                    v_old = v_old_all[k].detach()
                    v_clip = torch.clamp(v_pred - v_old, -0.2, 0.2) + v_old
                    v_loss = torch.max((v_pred - target) ** 2, (v_clip - target) ** 2).mean()
                else:
                    v_loss = F.mse_loss(v_pred, target)

                loss = pg_loss + vf_coef*v_loss - ent_coef*entropy
                loss.backward()
                pg_loss_acc += float(pg_loss.detach().cpu())
                v_loss_acc  += float(v_loss.detach().cpu())

            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), max_grad_norm)
            optimizer.step()
            total_pg_ep += pg_loss_acc / max(1, len(sl))
            total_v_ep  += v_loss_acc  / max(1, len(sl))
            n_mb += 1

    with torch.no_grad():
        old_logps = torch.tensor([it.logp for it in buf], dtype=torch.float32, device=device)
        new_logps, v_preds = [], []
        for it in buf:
            data = it.state_data
            mu_l, ls_l, mu_t, ls_t = actor.forward(data)
            std_l = ls_l.exp(); std_t = ls_t.exp()
            m = mu_l.numel()
            a_l = it.action[:m].to(device); a_t = it.action[m:].to(device)
            dist_l = torch.distributions.Normal(mu_l, std_l)
            dist_t = torch.distributions.Normal(mu_t, std_t)
            logp = dist_l.log_prob(a_l).sum() + dist_t.log_prob(a_t).sum()
            new_logps.append(logp)
            v_preds.append(critic.forward(data))
        new_logps = torch.stack(new_logps)
        v_preds = torch.stack(v_preds).squeeze(-1)
        approx_kl = torch.mean(old_logps - new_logps).clamp_min(0).item()
        ret_all = torch.tensor([ret[i].item() for i in range(len(buf))], dtype=torch.float32, device=device)
        ev = (1.0 - torch.var(ret_all - v_preds) / (torch.var(ret_all) + 1e-8)).item()

    return total_pg_ep / max(1, n_mb), total_v_ep / max(1, n_mb), approx_kl, ev



def save_checkpoint(path: str, actor: ActorGNN, critic: CriticGNN, optimizer: torch.optim.Optimizer, meta: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "optimizer": optimizer.state_dict(), "meta": meta}, path)
    print(f"[SAVE] checkpoint -> {path}")

def load_checkpoint(path: str, actor: ActorGNN, critic: CriticGNN, optimizer: Optional[torch.optim.Optimizer] = None):
    ckpt = torch.load(path, map_location="cpu")
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    meta = ckpt.get("meta", {})
    print(f"[LOAD] checkpoint <- {path} | meta={meta}")
    return meta

def train_ppo_gnn(seed=0, n=5, m=20, T_ep=40, episodes=30, L_seq=8,
                  update_epochs=20, lr=3e-4, device="cpu",
                  save_path: Optional[str]=None, use_lblr: bool=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    cfg = build_example_config(n=n, m=m, seed=2026+seed, common_ratio=0.3)
    env = ATOEnvPPO(cfg, rng_seed=2024+seed, lt_range=(1,3),
                    reward_scale=1000.0, order_deadband_tau=1.0, order_temp=0.5, order_cap=1000.0,
                    use_lblr=use_lblr)

    st = env.reset()
    hist = CompHist(m, L_seq); hist.push(env.last_orders, env.last_arrivals)
    sample = build_graph_data(env, st, hist, device=torch.device(device))
    node_dim = sample.x.size(1); edge_dim = sample.edge_attr.size(1)

    actor  = ActorGNN(node_dim=node_dim, edge_dim=edge_dim, use_lblr=use_lblr, temp_in=2, temp_hid=32).to(device)
    critic = CriticGNN(node_dim=node_dim, edge_dim=edge_dim, temp_in=2, temp_hid=32).to(device)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    rewards=[]
    for ep in range(1, episodes+1):
        st = env.reset()
        hist = CompHist(m, L_seq); hist.push(env.last_orders, env.last_arrivals)
        buf = PPOBuffer()

        demand_sum = delivered_sum = ontime_sum = 0
        orders_sum = surplus_sum = 0.0
        mu_l1_sum = sigma_l1_sum = U_l1_sum = 0.0
        ep_ret = 0.0

        for t in range(T_ep):
            data = build_graph_data(env, st, hist, device=torch.device(device))
            with torch.no_grad():
                a, logp = actor.act(data, deterministic=False)
                v = critic.forward(data)
            st_next, reward, info = env.step(a.detach().cpu().numpy())
            buf.push(data, a, logp, v, reward, float(t==T_ep-1))
            ep_ret += reward
            hist.push(env.last_orders, env.last_arrivals)
            st = st_next

            demand_sum   += int(info.get("demand_step", 0))
            delivered_sum+= int(info.get("delivered_step", 0))
            ontime_sum   += int(info.get("ontime_step", 0))
            orders_sum   += float(info.get("orders_sum", 0.0))
            surplus_sum  += float(info.get("surplus_sum", 0.0))
            mu_l1_sum    += float(info.get("mu_lt_l1", 0.0))
            sigma_l1_sum += float(info.get("sigma_lt_l1", 0.0))
            U_l1_sum     += float(info.get("overdue_pull_l1", 0.0))

        pg_loss, v_loss, kl, ev = ppo_update_gnn(actor, critic, optimizer, buf,
                                                 epochs=update_epochs, minibatch=64,
                                                 device=device, vf_coef=0.1, ent_coef=1e-3,
                                                 use_value_clipping=True)

        fill = (delivered_sum / max(1, demand_sum))
        ontime = (ontime_sum / max(1, demand_sum))
        redundancy = (surplus_sum / max(1e-9, orders_sum))
        print(f"[EP {ep:03d}] "
              f"return={ep_ret:8.2f} | pg={pg_loss:7.4f} | v={v_loss:12.4f} | kl={kl:7.5f} | ev={ev:5.3f} | "
              f"demand={demand_sum:.1f} | delivered={delivered_sum:.1f} (fill={fill:.3f}, ontime={ontime:.3f}) | "
              f"orders={orders_sum:.1f} | surplus={surplus_sum:.1f} (redundancy={redundancy:.3f}) | "
              f"||mu||1={mu_l1_sum:.1f} ||sigma||1={sigma_l1_sum:.1f} ||U||1={U_l1_sum:.1f}")

        if save_path and (ep % max(1, episodes//5) == 0 or ep == episodes):
            meta = dict(seed=seed, n=n, m=m, T_ep=T_ep, episodes=ep, L_seq=L_seq,
                        update_epochs=update_epochs, lr=lr, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                        use_lblr=use_lblr)
            save_checkpoint(save_path, actor, critic, optimizer, meta)
        rewards.append(ep_ret)
    return actor, critic, env, rewards

@torch.no_grad()
def rollout_policy(actor: ActorGNN, critic: CriticGNN, env: ATOEnvPPO, T_ep: int, L_seq: int, device="cpu", deterministic=True):
    st = env.reset()
    hist = CompHist(env.m, L_seq); hist.push(env.last_orders, env.last_arrivals)
    ep_ret = 0.0
    metrics = dict(demand=0, delivered=0, ontime=0, orders=0.0, surplus=0.0)
    for t in range(T_ep):
        data = build_graph_data(env, st, hist, device=torch.device(device))
        a, _ = actor.act(data, deterministic=deterministic)
        st, reward, info = env.step(a.detach().cpu().numpy())
        hist.push(env.last_orders, env.last_arrivals)
        ep_ret += reward
        metrics["demand"]   += int(info.get("demand_step", 0))
        metrics["delivered"]+= int(info.get("delivered_step", 0))
        metrics["ontime"]   += int(info.get("ontime_step", 0))
        metrics["orders"]   += float(info.get("orders_sum", 0.0))
        metrics["surplus"]  += float(info.get("surplus_sum", 0.0))
    fill = metrics["delivered"]/max(1,metrics["demand"])
    ontime = metrics["ontime"]/max(1,metrics["demand"])
    redundancy = metrics["surplus"]/max(1e-9, metrics["orders"])
    return ep_ret, fill, ontime, redundancy, metrics

"""
# ------------------ CLI ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train","eval"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--T", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=1330)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ckpt", type=str, default="ckpt_RLBR.pt")
    parser.add_argument("--eval_seeds", type=int, default=5)
    parser.add_argument("--use_lblr", action="store_true")
    args, _ = parser.parse_known_args()

    if args.mode == "train":
        actor, critic, env = train_ppo_gnn(seed=args.seed, n=args.n, m=args.m,
                                           T_ep=args.T, episodes=args.episodes,
                                           L_seq=8, device=args.device,
                                           save_path=args.ckpt, use_lblr=args.use_lblr)
        print("[TRAIN DONE]")

    elif args.mode == "eval":
        cfg = build_example_config(n=args.n, m=args.m, seed=args.seed+2026, common_ratio=0.3)
        env = ATOEnvPPO(cfg, rng_seed=args.seed, lt_range=(1,3), use_lblr=args.use_lblr)
        st = env.reset()
        hist = CompHist(env.m, 8); hist.push(env.last_orders, env.last_arrivals)
        sample = build_graph_data(env, st, hist, device=torch.device(args.device))
        node_dim = sample.x.size(1); edge_dim = sample.edge_attr.size(1)
        actor = ActorGNN(node_dim=node_dim, edge_dim=edge_dim, use_lblr=args.use_lblr, temp_in=2, temp_hid=32).to(args.device)
        critic= CriticGNN(node_dim=node_dim, edge_dim=edge_dim, temp_in=2, temp_hid=32).to(args.device)
        if args.ckpt and os.path.exists(args.ckpt):
            load_checkpoint(args.ckpt, actor, critic, optimizer=None)
        ep_ret, fill, ontime, red, _ = rollout_policy(actor, critic, env, args.T, 8, device=args.device, deterministic=True)
        print(f"[EVAL] return={ep_ret:.2f} | fill={fill:.3f} | ontime={ontime:.3f} | redundancy={red:.3f}")
    else:
        raise ValueError("unknown mode")
"""








