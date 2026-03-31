# -*- coding: utf-8 -*-


import argparse, os, math, time, csv, copy, random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gurobipy as gp
from gurobipy import GRB

from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data


import ato_rl_lblr as RL



ALGO_KEYS = ["RL_BR", "DTP", "NVD", "Oracle", "RL2"]
PLOT_KEYS = ["cost_total","cost_order","cost_hold","cost_late",
             "service_fill","ontime","redundancy","mismatch","pi"]


X_LABEL_MAP = {
    "rho": "Demand correlation (rho)",
    "bom_std_ratio": "BOM uncertainty (std/mean)",
    "common_ratio": "Common component ratio",
    "delivery_window": "Delivery window",
    "design_lead": "Design lead time",
    "lt_hi": "Max replenishment lead time",
    "lt_lo": "Min replenishment lead time",
    "price_scale_back": "Backorder penalty scale",
    "season_amp": "Seasonality amplitude",
}
Y_LABEL_MAP = {
    "cost_total": "Total cost",
    "cost_order": "Ordering cost",
    "cost_hold": "Holding cost",
    "cost_late": "Late cost",
    "service_fill": "Fill rate",
    "ontime": "On-time rate",
    "redundancy": "Redundancy",
    "mismatch": "Mismatch",
    "pi": "PI = Total cost / Oracle total cost",
}



def ensure_dir(p): os.makedirs(p, exist_ok=True)

def set_seeds(seed:int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def inv_std_norm_cdf(p:float)->float:
    p=min(max(p,1e-12),1-1e-12)
    a=[-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00]
    b=[-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01]
    c=[-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00]
    d=[ 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow=0.02425; phigh=1-plow
    if p<plow:
        q=math.sqrt(-2*math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q)+1
    if p>phigh:
        q=math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q)+1
    q=p-0.5; r=q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*q + a[5]*q) / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r + b[4])*r + 1)



@dataclass
class ScenarioCfg:
    seed:int; n_products:int; m_components:int; T:int
    lt_lo:int; lt_hi:int; delivery_window:int
    design_leads:np.ndarray; demand_rates:np.ndarray
    bom_mean:np.ndarray; bom_std:np.ndarray
    comp_order_cost:np.ndarray; comp_hold_cost:np.ndarray
    backorder_cost:np.ndarray; init_inventory:np.ndarray
    meta:dict

@dataclass
class Scenario:
    cfg:ScenarioCfg
    D:np.ndarray
    BOM_mu:np.ndarray
    BOM_std:np.ndarray
    bom_streams:Dict[Tuple[int,int], List[int]]
    leadtime_streams:Dict[int, List[int]]


def _gaussian_copula_correlated_uniform(n:int, T:int, rho:float, seed:int):
    rng=np.random.default_rng(seed); rho=float(np.clip(rho,-0.99,0.99))
    C=(1-rho)*np.eye(n)+rho*np.ones((n,n)); L=np.linalg.cholesky(C)
    Z=rng.standard_normal((n,T)); Zc=L@Z
    U=0.5*(1+torch.erf(torch.tensor(Zc/np.sqrt(2.0))).numpy())
    return np.clip(U,1e-8,1-1e-8)

def _seasonal_base(rate:np.ndarray, T:int, amp=0.3, period=12, phase=0.0):
    t=np.arange(T)
    mult=1.0+amp*np.sin(2*np.pi*(t/period)+phase)
    mult=np.maximum(0.1,mult)
    return rate.reshape(-1,1)*mult.reshape(1,-1)

def _demand_path(n:int,T:int,seed:int,family:str,base_rate:np.ndarray,
                 rho:float=0.0, season_amp:float=0.3,
                 season_period:int=12, season_phase:float=0.0):
    rng=np.random.default_rng(seed); family=family.lower()
    if family in ["poisson","seasonal"]:
        if family=="seasonal":
            lam=_seasonal_base(base_rate,T,amp=season_amp,period=season_period,phase=season_phase)
        else:
            lam=base_rate.reshape(-1,1).repeat(T,1)
        if abs(rho)>1e-8:
            U=_gaussian_copula_correlated_uniform(n,T,rho,seed+11)
            D=np.zeros((n,T),int)
            for i in range(n):
                for t in range(T):
                    rate=lam[i,t]; u=U[i,t]
                    k=0; p=math.exp(-rate); cdf=p
                    while cdf<u and k<10000:
                        k+=1; p*=rate/k; cdf+=p
                    D[i,t]=k
        else:
            D=rng.poisson(lam=lam)
        return D.astype(int)
    raise ValueError("Unsupported demand family")

def make_default_scenario(n=6, m=20, T=40, seed=2025,
                          demand_family="poisson", rho=0.0,
                          bom_std_ratio=0.3, common_ratio=0.3,
                          delivery_window=2, design_lead=3,
                          lt_lo=1, lt_hi=3,
                          price_scale_order=1.0,
                          price_scale_hold=1.0,
                          price_scale_back=1.0,
                          season_amp:float=0.3,
                          season_period:int=12,
                          season_phase:float=0.0):
    rng=np.random.default_rng(seed)
    demand_rates=rng.uniform(2.0,8.0,size=n)
    design_leads=np.full(n,int(design_lead),dtype=int)

    bom_mean=np.zeros((n,m),dtype=int)
    bom_std =np.zeros((n,m),dtype=float)

    common=rng.choice(m,size=max(1,int(common_ratio*m)),replace=False)
    for i in range(n):
        mask=np.zeros(m,bool); mask[common]=True
        pool=[j for j in range(m) if j not in common]
        if len(pool)>0:
            extra=rng.choice(pool,size=min(len(pool),rng.integers(8,14)),replace=False)
            mask[extra]=True
        bom_mean[i,common]=rng.integers(2,7,size=len(common))
        idx=np.where(mask)[0]
        bom_mean[i,idx]=np.maximum(bom_mean[i,idx],rng.integers(1,5,size=len(idx)))
        bom_std[i,idx]=bom_std_ratio*np.maximum(1.0,bom_mean[i,idx])

    comp_order_cost=price_scale_order*rng.uniform(0.1,0.3,size=m)
    comp_hold_cost =price_scale_hold *rng.uniform(0.01,0.03,size=m)
    init_inventory =rng.integers(10,50,size=m)

    backorder=np.zeros(n)
    for i in range(n):
        unit_cost=float(np.dot(bom_mean[i],comp_order_cost))
        backorder[i]=price_scale_back*max(1.0,unit_cost*rng.uniform(1.5,3.0))

    cfg=ScenarioCfg(
        seed=seed,n_products=n,m_components=m,T=T,
        lt_lo=int(lt_lo),lt_hi=int(lt_hi),
        delivery_window=int(delivery_window),
        design_leads=design_leads,demand_rates=demand_rates,
        bom_mean=bom_mean,bom_std=bom_std,
        comp_order_cost=comp_order_cost,comp_hold_cost=comp_hold_cost,
        backorder_cost=backorder,init_inventory=init_inventory,
        meta={
            "common_ratio":float(common_ratio),
            "demand_family":str(demand_family).lower(),
            "season_amp":float(season_amp),
            "season_period":int(season_period),
            "season_phase":float(season_phase),
        }
    )

    D=_demand_path(n,T,seed+7,demand_family,demand_rates,
                   rho=rho, season_amp=season_amp,
                   season_period=season_period,
                   season_phase=season_phase)

    leadtime_streams={
        j: np.random.default_rng(seed+100+j).integers(cfg.lt_lo,cfg.lt_hi+1,size=5000).tolist()
        for j in range(m)
    }

    bom_streams={}
    rng2=np.random.default_rng(seed+300)
    for i in range(n):
        for j in range(m):
            mu=bom_mean[i,j]; sd=bom_std[i,j]
            if sd<=0 or mu<=0:
                bom_streams[(i,j)]=[0]*5000
            else:
                z=np.maximum(0.0, rng2.normal(loc=mu, scale=sd, size=5000))
                bom_streams[(i,j)]=np.rint(z).astype(int).tolist()

    return Scenario(cfg, D, bom_mean.copy(), bom_std.copy(), bom_streams, leadtime_streams)

def scenario_to_cfg(scn:Scenario)->Dict[str,Any]:
    c=scn.cfg; n,m=c.n_products,c.m_components
    bom_templates={}
    for i in range(n):
        mean=c.bom_mean[i].astype(float); std=c.bom_std[i].astype(float)
        mask=(mean>0.0)
        bom_templates[i]=(mean.tolist(),std.tolist(),mask.astype(bool).tolist())
    meta=c.meta
    return dict(
        n_products=n,
        m_components=m,
        bom_templates=bom_templates,
        design_leads=c.design_leads.astype(int).tolist(),
        delivery_window=int(c.delivery_window),
        demand_rates=c.demand_rates.astype(float).tolist(),
        comp_order_cost=c.comp_order_cost.astype(float).tolist(),
        comp_hold_cost=c.comp_hold_cost.astype(float).tolist(),
        backorder_cost=c.backorder_cost.astype(float).tolist(),
        init_component_inventory=c.init_inventory.astype(int).tolist(),
        meta={
            "seed": c.seed,
            "demand_family": meta.get("demand_family","poisson"),
            "common_ratio": meta.get("common_ratio",0.3),
            "season_amp": meta.get("season_amp",0.0),
            "season_period": meta.get("season_period",12),
            "season_phase": meta.get("season_phase",0.0),
        }
    )



class EnvUnifiedReplay(RL.ATOEnvPPO):
    def __init__(self, cfg, scn:Scenario, rng_seed=0, use_lblr=True,
                 reward_scale=1000.0, order_deadband_tau=1.0, order_temp=0.5, order_cap=5000.0):

        # set before super().__init__
        self._cfg = cfg
        meta_cfg = cfg.get("meta", {}) if isinstance(cfg, dict) else {}
        meta_scn = scn.cfg.meta if hasattr(scn, "cfg") else {}

        self._demand_family = str(meta_cfg.get("demand_family", meta_scn.get("demand_family","poisson"))).lower()
        self._season_amp    = float(meta_cfg.get("season_amp",    meta_scn.get("season_amp",0.0)))
        self._season_period = float(meta_cfg.get("season_period", meta_scn.get("season_period",12)))
        self._season_phase  = float(meta_cfg.get("season_phase",  meta_scn.get("season_phase",0.0)))

        # base rates at t=0
        self._base_demand_rates = np.array(cfg["demand_rates"], float) if (isinstance(cfg, dict) and "demand_rates" in cfg) else np.array(scn.cfg.demand_rates, float)

        # placeholders (in case reset called early)
        self.stat_cost_total = 0.0
        self.stat_cost_order = 0.0
        self.stat_cost_hold  = 0.0
        self.stat_cost_late  = 0.0
        self.stat_orders_sum   = 0.0
        self.stat_surplus_sum  = 0.0
        self.stat_shortage_sum = 0.0
        self.last_ontime_delivered = 0
        self.ep_total_delivered = 0
        self.ep_on_time_delivered = 0
        self.ep_total_demand = 0
        # ===== mix mismatch (TV distance) stats =====
        self.stat_mix_mismatch_sum = 0.0
        self.stat_mix_mismatch_cnt = 0



        super().__init__(cfg, rng_seed=rng_seed, lt_range=(scn.cfg.lt_lo, scn.cfg.lt_hi),
                         reward_scale=reward_scale,
                         order_deadband_tau=order_deadband_tau,
                         order_temp=order_temp,
                         order_cap=order_cap,
                         use_lblr=use_lblr)

        # keep scenario streams
        self._D = scn.D.astype(int)
        self._T = scn.cfg.T
        self._bom_streams = copy.deepcopy(scn.bom_streams)
        self._lt_streams  = copy.deepcopy(scn.leadtime_streams)

        self._apply_time_varying_demand_rates()

    def reset(self):
        st = super().reset()

        self.stat_cost_total = 0.0
        self.stat_cost_order = 0.0
        self.stat_cost_hold  = 0.0
        self.stat_cost_late  = 0.0
        self.stat_orders_sum   = 0.0
        self.stat_surplus_sum  = 0.0
        self.stat_shortage_sum = 0.0
        self.last_ontime_delivered = 0
        self.ep_total_delivered = 0
        self.ep_on_time_delivered = 0
        self.ep_total_demand = 0
        self.stat_mix_mismatch_sum = 0.0
        self.stat_mix_mismatch_cnt = 0


        self._apply_time_varying_demand_rates()
        return st

    def _apply_time_varying_demand_rates(self):
        family = str(getattr(self, "_demand_family", "poisson")).lower()
        amp    = float(getattr(self, "_season_amp", 0.0))
        period = float(getattr(self, "_season_period", 12.0))
        phase  = float(getattr(self, "_season_phase", 0.0))
        t = int(getattr(self, "t", 0))

        if family == "seasonal" and abs(amp) > 1e-12:
            mult = 1.0 + amp * math.sin(2.0 * math.pi * (t / period) + phase)
            mult = max(0.1, float(mult))
        else:
            mult = 1.0

        rates = np.array(self._base_demand_rates, float) * mult

        # push to typical locations parent may read
        try:
            self.demand_rates = rates
        except Exception:
            pass
        if isinstance(getattr(self, "_cfg", None), dict):
            self._cfg["demand_rates"] = rates.tolist()

    def _accumulate_cost(self, info: Dict[str, float], reward=None):
        if isinstance(info, dict) and all(k in info for k in ("order_cost","hold_cost","late_cost")):
            oc=float(info["order_cost"]); hc=float(info["hold_cost"]); lc=float(info["late_cost"])
            self.stat_cost_order += oc
            self.stat_cost_hold  += hc
            self.stat_cost_late  += lc
            self.stat_cost_total  = self.stat_cost_order + self.stat_cost_hold + self.stat_cost_late
        elif reward is not None:
            self.stat_cost_total += float(-reward * max(1.0, self.reward_scale))

    def step(self, action_params: np.ndarray):
        self._apply_time_varying_demand_rates()
        st_next, reward, info = super().step(action_params)
        self._accumulate_cost(info if isinstance(info,dict) else {}, reward=reward)

        on_order = self._compute_on_order_all()
        B_comp   = self._component_equiv_backlog(self._build_state())
        IP       = self.component_inventory.astype(float) + on_order - B_comp
        self.stat_surplus_sum  += float(np.maximum(0.0, IP).sum())
        self.stat_shortage_sum += float(np.maximum(0.0,-IP).sum())
        if isinstance(info,dict):
            self.stat_orders_sum += float(info.get("orders_sum", 0.0))

        mm = self._mix_mismatch_tv()
        #mm = self._mismatch_mix_induced(H=int(self.lt_hi))
        if np.isfinite(mm):
            self.stat_mix_mismatch_sum += float(mm)
            self.stat_mix_mismatch_cnt += 1

        return st_next, reward, info



    def step_truecost_with_q(self, q_order:np.ndarray, threads=0, timelimit=60.0):
        self._apply_time_varying_demand_rates()

        self._advance_pipeline_and_receive()
        self._release_realized_orders()
        self._ingest_new_orders_for_current_t()

        order_cost = self._place_component_orders(q_order)

 
        if hasattr(self, "assemble_now"):
            _ = self.assemble_now()
            costs = self._compute_costs(order_cost)
        else:
            _ = self._assemble_now_with_gurobi_compat(threads=threads, timelimit=timelimit)
            costs = self._compute_costs(order_cost)

        total_cost = float(costs["total_cost"])
        reward = - total_cost / max(1.0, self.reward_scale)
        self._accumulate_cost(costs, reward=reward)

        on_order=self._compute_on_order_all()
        st=self._build_state()
        B_comp=self._component_equiv_backlog(st)
        IP=self.component_inventory.astype(float)+on_order-B_comp
        orders_sum=float(np.sum(q_order))
        self.stat_orders_sum += orders_sum
        self.stat_surplus_sum  += float(np.maximum(0.0, IP).sum())
        self.stat_shortage_sum += float(np.maximum(0.0,-IP).sum())

        mm = self._mix_mismatch_tv()
        if np.isfinite(mm):
            self.stat_mix_mismatch_sum += float(mm)
            self.stat_mix_mismatch_cnt += 1

        info=dict(costs)
        info.update({"orders_sum": orders_sum})
        self.t += 1
        

        return self._build_state(), reward, info

    def _compute_on_order_all(self) -> np.ndarray:
        on_order = np.zeros(self.m, dtype=float)
        pl = getattr(self, "pipeline", None)
        if isinstance(pl, dict):
            for j in range(self.m):
                for eta, q in pl.get(j, []):
                    on_order[j] += float(q)
        return on_order

    def _component_equiv_backlog(self, state) -> np.ndarray:
        b = np.zeros(self.m, dtype=float)
        if state is None:
            state = self._build_state()
        for i in range(self.n):
            mean, _, _ = self.bom_templates[i]
            b += np.asarray(mean, float) * float(state.realized_backlog[i])
        return b
    
    def _on_order_within(self, H: int) -> np.ndarray:
        
        H = int(max(0, H))
        oo = np.zeros(self.m, dtype=float)
        pl = getattr(self, "pipeline", None)
        if isinstance(pl, dict):
            for j in range(self.m):
                for eta, q in pl.get(j, []):
                    if int(eta) <= H:
                        oo[j] += float(q)
        elif isinstance(pl, list):
            for j in range(min(self.m, len(pl))):
                for eta, q in pl[j]:
                    if int(eta) <= H:
                        oo[j] += float(q)
        return oo

    def _req_components_realized(self) -> np.ndarray:
        req = np.zeros(self.m, dtype=float)
        ros = getattr(self, "realized_orders", [])
        t = int(getattr(self, "t", 0))
        for od in ros:
            rem = float(getattr(od, "remaining", 0.0))
            if rem <= 0:
                continue
            t_done = int(getattr(od, "t_design_done", -10**9))
            if t < t_done:
                continue
            bom = np.asarray(getattr(od, "bom_per_unit", np.zeros(self.m)), dtype=float)
            if bom.size != self.m:
                bom2 = np.zeros(self.m, dtype=float)
                bom2[:min(self.m, bom.size)] = bom[:min(self.m, bom.size)]
                bom = bom2
            req += rem * bom
        return req
    
    def _get_order_remaining(self, od) -> float:
   
        if hasattr(od, "remaining"):
            return float(getattr(od, "remaining", 0.0))
        if hasattr(od, "remaining_qty"):
            return float(getattr(od, "remaining_qty", 0.0))
        if hasattr(od, "rem"):
            return float(getattr(od, "rem", 0.0))
        return 0.0

    def _get_order_bom(self, od) -> Optional[np.ndarray]:
        bom = getattr(od, "bom_per_unit", None)
        if bom is None:
            bom = getattr(od, "bom", None)
        if bom is None:
            return None
        bom = np.asarray(bom, dtype=float)
        if bom.size != self.m:
            b = np.zeros(self.m, dtype=float)
            b[:min(self.m, bom.size)] = bom[:min(self.m, bom.size)]
            bom = b
        return bom

    def _req_components_realized_or_fallback(self, state=None) -> np.ndarray:

        req = np.zeros(self.m, dtype=float)
        used = False
        ros = getattr(self, "realized_orders", [])
        t = int(getattr(self, "t", 0))
        for od in ros:
            rem = self._get_order_remaining(od)
            if rem <= 0:
                continue
            t_done = int(getattr(od, "t_design_done", -10**9))
            if t < t_done:
                continue
            bom = self._get_order_bom(od)
            if bom is None:
                continue
            req += rem * bom
            used = True

        if used:
            return req

 
        if state is None:
            state = self._build_state()
        rb = np.asarray(getattr(state, "realized_backlog", np.zeros(self.n)), dtype=float)
        req = np.zeros(self.m, dtype=float)
        for i in range(self.n):
            mean_i, _, _ = self.bom_templates[i]
            req += np.asarray(mean_i, float) * float(rb[i])
        return req


    def _mismatch_mix_induced(self, H: Optional[int] = None) -> float:
  
        if H is None:
            H = int(getattr(self, "lt_hi", 0))

        req = self._req_components_realized()
        req_sum = float(req.sum())
        if req_sum <= 1e-9:
            return 0.0

        avail = self.component_inventory.astype(float) + self._on_order_within(H)
        avail_sum = float(avail.sum())
        if avail_sum <= 1e-9:
            return 0.0

        p = avail / avail_sum
        q = req / req_sum
        tv = 0.5 * float(np.abs(p - q).sum())

        shortage_ratio = float(np.maximum(0.0, req - avail).sum()) / max(1e-9, req_sum)
        return float(tv * shortage_ratio)


    def _mix_mismatch_tv(self, state=None) -> float:
        if state is None:
            state = self._build_state()

        req = self._req_components_realized_or_fallback(state)
        rs = float(req.sum())
        if rs <= 1e-9:
            return np.nan

        avail = self.component_inventory.astype(float) + self._compute_on_order_all()
        avs = float(avail.sum())
        if avs <= 1e-9:
            return np.nan

        p = avail / avs
        r = req / rs
        return 0.5 * float(np.abs(p - r).sum())



    def _assemble_now_with_gurobi_compat(self, threads=0, timelimit=60.0):

        realized = getattr(self, "realized_orders", [])
        if not realized:
            self.last_ontime_delivered = 0
            return np.zeros(self.n, dtype=int)

        idxs=[]
        for ridx, od in enumerate(realized):
            rem=float(getattr(od,"remaining",0.0))
            t_done=int(getattr(od,"t_design_done",10**9))
            if rem>0 and int(self.t) >= t_done:
                idxs.append(ridx)
        if len(idxs)==0:
            self.last_ontime_delivered = 0
            return np.zeros(self.n, dtype=int)

        model = gp.Model("alloc_once_eval")
        model.Params.OutputFlag = 0
        if threads and threads>0: model.Params.Threads = threads
        if timelimit and timelimit>0: model.Params.TimeLimit = timelimit

        y = model.addVars(len(idxs), vtype=GRB.INTEGER, lb=0, name="y")

        inv0 = self.component_inventory.astype(int).copy()

        # component constraints
        for j in range(self.m):
            expr=[]
            for k, ridx in enumerate(idxs):
                od=realized[ridx]
                bom=np.asarray(getattr(od,"bom_per_unit",np.zeros(self.m)), dtype=int)
                bj=int(bom[j]) if j < len(bom) else 0
                if bj>0:
                    expr.append(bj*y[k])
            if expr:
                model.addConstr(gp.quicksum(expr) <= int(inv0[j]), name=f"inv_{j}")

        # each order cap
        for k, ridx in enumerate(idxs):
            rem=int(round(float(getattr(realized[ridx],"remaining",0.0))))
            model.addConstr(y[k] <= rem, name=f"rem_{k}")

        # objective: urgency * backorder cost
        obj=0.0
        for k, ridx in enumerate(idxs):
            od=realized[ridx]
            i=int(getattr(od,"product_id",0))
            t_due=int(getattr(od,"t_due",int(self.t)))
            if int(self.t) <= t_due:
                urg=math.exp(-0.25*(t_due-int(self.t)))
            else:
                urg=math.exp( 0.40*(int(self.t)-t_due))
            obj += float(self.backorder_cost[i]) * urg * y[k]
        model.setObjective(obj, GRB.MAXIMIZE)
        model.optimize()

        assembled_by_prod=np.zeros(self.n, dtype=int)
        self.last_ontime_delivered = 0
        if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return assembled_by_prod

        inv = self.component_inventory
        for k, ridx in enumerate(idxs):
            yk = int(round(y[k].X)) if y[k].X is not None else 0
            if yk<=0: continue
            od=realized[ridx]
            i=int(getattr(od,"product_id",0))
            bom=np.asarray(getattr(od,"bom_per_unit",np.zeros(self.m)), dtype=int)
            pos=np.where(bom>0)[0]
            if pos.size>0:
                inv[pos] -= bom[pos]*yk

            # update order
            if hasattr(od,"assembled_qty"):
                try: od.assembled_qty += yk
                except Exception: pass

            assembled_by_prod[i] += yk

            self.ep_total_delivered += yk
            t_due=int(getattr(od,"t_due",int(self.t)))
            if int(self.t) <= t_due:
                self.ep_on_time_delivered += yk
                self.last_ontime_delivered += yk
        return assembled_by_prod


# ================== Metrics ================== #
def metrics_finalize(env:EnvUnifiedReplay)->Dict[str,float]:
    fill = env.ep_total_delivered / max(1, env.ep_total_demand)
    ontime = env.ep_on_time_delivered / max(1, env.ep_total_demand)
    redundancy = env.stat_surplus_sum / max(1.0, env.stat_orders_sum if env.stat_orders_sum>0 else 1.0)
    #mismatch   = env.stat_shortage_sum / max(1.0, env.stat_orders_sum if env.stat_orders_sum>0 else 1.0)
    mismatch = env.stat_mix_mismatch_sum / max(1, env.stat_mix_mismatch_cnt)

    return dict(
        cost_total=env.stat_cost_total,
        cost_order=env.stat_cost_order,
        cost_hold =env.stat_cost_hold,
        cost_late =env.stat_cost_late,
        service_fill=fill,
        ontime=ontime,
        redundancy=redundancy,
        mismatch=mismatch
    )

def ci95(vals:List[float])->Tuple[float,float]:
    arr=np.asarray(vals,float); arr=arr[np.isfinite(arr)]
    if arr.size==0: return np.nan,0.0
    m=float(np.nanmean(arr))
    s=float(np.nanstd(arr,ddof=1)) if arr.size>1 else 0.0
    return m, 1.96*s/max(1.0,np.sqrt(arr.size))


# ================== Oracle ================== #
def _build_cohort_bom(scn:Scenario)->Dict[Tuple[int,int,int],int]:
    n,m,T=scn.cfg.n_products, scn.cfg.m_components, scn.cfg.T
    streams=copy.deepcopy(scn.bom_streams); a={}
    for i in range(n):
        for tau in range(T):
            for j in range(m):
                if scn.cfg.bom_mean[i,j]>0:
                    val=int(streams[(i,j)].pop(0)) if streams[(i,j)] else int(round(max(0.0, scn.cfg.bom_mean[i,j])))
                else:
                    val=0
                a[(i,j,tau)]=val
    return a

def solve_oracle_gurobi(scn: Scenario, threads=0, timelimit=60.0, mipgap=0.02) -> Dict[str, float]:
    n,m,T=scn.cfg.n_products, scn.cfg.m_components, scn.cfg.T
    a=_build_cohort_bom(scn)
    model=gp.Model("oracle_ato"); model.Params.OutputFlag=0
    if threads and threads>0: model.Params.Threads=threads
    if timelimit and timelimit>0: model.Params.TimeLimit=timelimit
    if mipgap is not None: model.Params.MIPGap=mipgap

    inv=model.addVars(m,T+1,vtype=GRB.INTEGER,lb=0,name="inv")
    q  =model.addVars(m,T,  vtype=GRB.INTEGER,lb=0,name="q")
    y  =model.addVars(n,T,T,vtype=GRB.INTEGER,lb=0,name="y")
    u  =model.addVars(n,T,T+1,vtype=GRB.INTEGER,lb=0,name="u")

    for j in range(m): model.addConstr(inv[j,0]==int(scn.cfg.init_inventory[j]))
    for i in range(n):
        for tau in range(T): model.addConstr(u[i,tau,0]==0)

    Lmin,Lmax=int(scn.cfg.lt_lo),int(scn.cfg.lt_hi)
    if Lmax<Lmin: Lmin,Lmax=Lmax,Lmin
    K=max(0,Lmax-Lmin+1)
    qsplit=model.addVars(m,T,K,vtype=GRB.INTEGER,lb=0,name="qsplit")
    if K>0:
        for j in range(m):
            for t in range(T):
                model.addConstr(gp.quicksum(qsplit[j,t,k] for k in range(K))==q[j,t])
    else:
        for j in range(m):
            for t in range(T):
                model.addConstr(q[j,t]==0)

    for t in range(T):
        for j in range(m):
            if K>0 and (t-Lmin)>=0:
                arrivals=gp.quicksum(
                    qsplit[j,tau,(t-tau-Lmin)]
                    for tau in range(max(0,t-Lmax), t-Lmin+1)
                )
            else:
                arrivals=0
            cons=gp.quicksum(
                y[i,t,tau]*int(a[(i,j,tau)])
                for i in range(n)
                for tau in range(0,t+1)
            )
            model.addConstr(inv[j,t+1]==inv[j,t]+arrivals-cons)

    Ld=scn.cfg.design_leads; D=scn.D; W=int(scn.cfg.delivery_window)
    for i in range(n):
        for tau in range(T):
            model.addConstr(gp.quicksum(y[i,t,tau] for t in range(tau,T))<=int(D[i,tau]))

    for i in range(n):
        Li=int(Ld[i])
        for tau in range(T):
            for t in range(T):
                arrival=int(D[i,tau]) if (t==tau+Li) else 0
                model.addConstr(u[i,tau,t+1]==u[i,tau,t]+arrival-y[i,t,tau])

    order_cost=gp.quicksum(q[j,t]*float(scn.cfg.comp_order_cost[j]) for j in range(m) for t in range(T))
    hold_cost =gp.quicksum(inv[j,t]*float(scn.cfg.comp_hold_cost[j])  for j in range(m) for t in range(1,T+1))

    late_terms=[]
    for i in range(n):
        Li=int(Ld[i]); bi=float(scn.cfg.backorder_cost[i])
        for tau in range(T):
            due_hi=tau+Li+W
            for t in range(1,T+1):
                if t>due_hi:
                    late_terms.append(bi*u[i,tau,t])
    late_cost=gp.quicksum(late_terms) if late_terms else 0.0

    model.setObjective(order_cost+hold_cost+late_cost, GRB.MINIMIZE)
    model.optimize()

    res=dict(cost_total=np.nan,cost_order=np.nan,cost_hold=np.nan,cost_late=np.nan,
             service_fill=np.nan, ontime=np.nan, redundancy=np.nan, mismatch=np.nan)
    has_sol=(getattr(model,"SolCount",0) is not None and model.SolCount>0)
    if not has_sol:
        print(f"[WARN] Oracle no solution (Status={model.Status}, SolCount={getattr(model,'SolCount',0)})")
        return res

    try:
        res["cost_total"]=float(model.objVal)
        res["cost_order"]=float(order_cost.getValue())
        res["cost_hold"]=float(hold_cost.getValue())
        res["cost_late"]=float(late_cost.getValue()) if hasattr(late_cost,"getValue") else np.nan

        yv=np.zeros((n,T,T),float)
        for i in range(n):
            for tau in range(T):
                for t in range(tau,T):
                    val=y[i,t,tau].X
                    yv[i,t,tau]=max(0.0,val) if val is not None else 0.0
        Dtot=float(np.sum(D))
        if Dtot>0:
            served=yv.sum()
            served_on=0.0
            for i in range(n):
                Li=int(Ld[i])
                for tau in range(T):
                    lo=tau+Li; hi=min(T, lo+W)
                    if lo<T:
                        served_on += yv[i, lo:hi, tau].sum()
            res["service_fill"]=served/Dtot
            res["ontime"]=served_on/Dtot
    except Exception as e:
        print(f"[WARN] Oracle readback failed: {e}")
    return res


# ================== Controllers ================== #
class NVD_Controller:
    """
    Cost-structured NVD with decomposed underage cost:
      cu_j = sum_i back_i / BOM_ij   (only if BOM_ij>0)
    """
    def __init__(self, capQ: float = 2000.0,
                 z_clip: float = 2.33,
                 F_cap: float = 0.98,
                 F_min: float = 1e-4,
                 var_scale: float = 1.0):
        self.capQ=float(capQ)
        self.z_clip=float(z_clip)
        self.F_cap=float(F_cap)
        self.F_min=float(F_min)
        self.var_scale=float(var_scale)
        self._fitted=False
        self._S=None

    def _invz(self,p:float)->float:
        z=inv_std_norm_cdf(p)
        return max(-self.z_clip, min(self.z_clip, z))

    def _fit(self, env:EnvUnifiedReplay):
        if self._fitted: return
        if not isinstance(env._cfg, dict):
            raise RuntimeError("env._cfg must be dict")

        n,m=env.n,env.m
        mu_prod=np.array(env._cfg["demand_rates"], float)  # time-varying baseline (t=0 used after reset)
        var_prod=np.maximum(1.0, mu_prod)*self.var_scale

        L_eff=0.5*(env.lt_lo+env.lt_hi)
        L_eff=1
        bom_mu=np.array([[env.bom_templates[i][0][j] for j in range(m)] for i in range(n)], float)

        mu_c=(bom_mu.T @ mu_prod)*L_eff
        var_c=((bom_mu**2).T @ var_prod)*L_eff
        sigma_c=np.sqrt(np.maximum(1e-8,var_c))

        back_c=np.array(env.backorder_cost, float)  # (n,)

        # decomposed by division
        mask_pos = (bom_mu > 0)
        safe_bom = np.where(mask_pos, bom_mu, 1.0)
        cu_mat = np.where(mask_pos, back_c.reshape(-1,1) / safe_bom, 0.0)
        cu = cu_mat.sum(axis=0)  # (m,)

        co=np.array(env.comp_hold_cost, float)*L_eff
        F = cu / np.maximum(cu + co, 1e-9)
        F = np.clip(F, self.F_min, self.F_cap)
        z = np.array([self._invz(p) for p in F])

        self._S = np.maximum(0.0, mu_c + z*sigma_c)
        self._fitted=True

    def act(self, env:EnvUnifiedReplay, threads=0, timelimit=60.0):
        self._fit(env)
        st=env._build_state()
        on_order=env._compute_on_order_all()
        Bc=env._component_equiv_backlog(st)
        IP=env.component_inventory.astype(float)+on_order-Bc
        q=np.clip(self._S - IP, 0.0, self.capQ)
        return env.step_truecost_with_q(q, threads=threads, timelimit=timelimit)
    
class DTP_Controller:

    def __init__(self,
                 capQ: float = 2000.0,
                 use_max_lead: bool = True,
                 smooth: bool = True,
                 smooth_extra: Optional[int] = None,
                 use_onorder_horizon: str = "H2",   # "H" or "H2"
                 # --- NEW: generic safety stock (forecast-only) ---
                 use_forecast_base: bool = True,
                 base_F0: float = 0.8,             # service level for generic base-stock
                 base_var_scale: float = 1.0,       # var = var_scale * mean (Poisson approx)
                 base_strength: float = 1.0,        # multiply base-stock level (keep =1.0 by default)
                 base_min_mu: float = 0.0           # optionally only stock components with mu_base >= threshold
                 ):
        self.capQ = float(capQ)
        self.use_max_lead = bool(use_max_lead)
        self.smooth = bool(smooth)
        self.smooth_extra = smooth_extra
        assert use_onorder_horizon in ["H", "H2"]
        self.use_onorder_horizon = use_onorder_horizon

        self.use_forecast_base = bool(use_forecast_base)
        self.base_F0 = float(base_F0)
        self.base_var_scale = float(base_var_scale)
        self.base_strength = float(base_strength)
        self.base_min_mu = float(base_min_mu)

        # cache (recomputed each period because seasonal demand changes)
        self._cached_base = None
        self._cached_base_t = None

    @staticmethod
    def _get_pipeline_list(env, j: int):
        pl = getattr(env, "pipeline", None)
        if isinstance(pl, dict):
            return pl.get(j, [])
        if isinstance(pl, list) and j < len(pl):
            return pl[j]
        return []

    def _on_order_within(self, env, H: int) -> np.ndarray:
        H = int(max(0, H))
        oo = np.zeros(env.m, dtype=float)
        for j in range(env.m):
            for eta, q in self._get_pipeline_list(env, j):
                if int(eta) <= H:
                    oo[j] += float(q)
        return oo

    def _get_delivery_window(self, env) -> int:
        if hasattr(env, "delivery_window"):
            try:
                return int(env.delivery_window)
            except Exception:
                pass
        if isinstance(getattr(env, "_cfg", None), dict):
            try:
                return int(env._cfg.get("delivery_window", 0))
            except Exception:
                pass
        return 0

    def _compute_forecast_base_stock(self, env, t: int) -> np.ndarray:
  
        if not self.use_forecast_base:
            return np.zeros(env.m, dtype=float)

        # cache per period t (because seasonal demand changes with t)
        if self._cached_base_t == t and self._cached_base is not None:
            return self._cached_base

        # current product demand rates (already seasonal-adjusted in env._cfg by EnvUnifiedReplay)
        if isinstance(getattr(env, "_cfg", None), dict) and "demand_rates" in env._cfg:
            lam = np.array(env._cfg["demand_rates"], dtype=float)
        else:
            lam = np.array(getattr(env, "demand_rates", np.ones(env.n)), dtype=float)

        var_prod = np.maximum(1.0, lam) * self.base_var_scale
        L_eff = 0.5 * (env.lt_lo + env.lt_hi)

        # template BOM mean matrix (n x m)
        bom_mu = np.array([[env.bom_templates[i][0][j] for j in range(env.m)]
                           for i in range(env.n)], dtype=float)

        mu_base = (bom_mu.T @ lam) * L_eff
        var_base = ((bom_mu ** 2).T @ var_prod) * L_eff
        sigma_base = np.sqrt(np.maximum(1e-8, var_base))

        # fixed service level -> z
        z = float(inv_std_norm_cdf(self.base_F0))
        # clip z to avoid explosion
        z = float(np.clip(z, -2.33, 2.33))

        S_base = mu_base + z * sigma_base
        S_base = np.maximum(0.0, S_base)

        # optional: only stock sufficiently “common” components
        if self.base_min_mu > 0:
            S_base = np.where(mu_base >= self.base_min_mu, S_base, 0.0)

        S_base = self.base_strength * S_base

        self._cached_base = S_base
        self._cached_base_t = t
        return S_base

    def act(self, env: "EnvUnifiedReplay", threads=0, timelimit=60.0):
        t = int(getattr(env, "t", 0))

        # horizon H: worst-case replenishment LT by default
        H = int(env.lt_hi) if self.use_max_lead else int(round(0.5 * (env.lt_lo + env.lt_hi)))

        # smoothing horizon H2: H + delivery_window by default
        if self.smooth:
            extra = self.smooth_extra
            if extra is None:
                extra = self._get_delivery_window(env)
            H2 = H + int(max(0, extra))
        else:
            H2 = H

        # ----- realized-order requirement (BOM known only) -----
        req = np.zeros(env.m, dtype=float)
        ros = getattr(env, "realized_orders", [])
        for od in ros:
            rem = float(getattr(od, "remaining", 0.0))
            if rem <= 0:
                continue
            t_done = int(getattr(od, "t_design_done", -10**9))
            if t < t_done:
                continue

            due = int(getattr(od, "t_due", t))
            slack = due - t

            # weight (bucket smoothing)
            if slack <= H:
                w = 1.0
            elif self.smooth and (H2 > H) and (slack <= H2):
                w = float(H2 - slack) / float(H2 - H)
            else:
                w = 0.0

            if w <= 0.0:
                continue

            bom = np.asarray(getattr(od, "bom_per_unit", np.zeros(env.m)), dtype=float)
            req += w * rem * bom

        # ----- generic forecast-only base-stock (independent of design-stage orders) -----
        S_base = self._compute_forecast_base_stock(env, t)

        # combine targets: max avoids double-counting forecast demand vs realized backlog
        target = np.maximum(S_base, req)

        # ----- deadline-aware on-order -----
        if self.use_onorder_horizon == "H2":
            oo = self._on_order_within(env, H2)
        else:
            oo = self._on_order_within(env, H)

        IP = env.component_inventory.astype(float) + oo

        q = np.clip(target - IP, 0.0, self.capQ)
        return env.step_truecost_with_q(q, threads=threads, timelimit=timelimit)


"""
class DTP_Controller:

    def __init__(self,
                 capQ: float = 2000.0,
                 use_max_lead: bool = True,
                 smooth: bool = True,
                 smooth_extra: Optional[int] = None,
                 use_onorder_horizon: str = "H"  # "H" or "H2"
                 ):
        self.capQ = float(capQ)
        self.use_max_lead = bool(use_max_lead)
        self.smooth = bool(smooth)
        self.smooth_extra = smooth_extra  # if None -> use delivery_window
        assert use_onorder_horizon in ["H", "H2"]
        self.use_onorder_horizon = use_onorder_horizon

    @staticmethod
    def _get_pipeline_list(env, j: int):

        pl = getattr(env, "pipeline", None)
        if isinstance(pl, dict):
            return pl.get(j, [])
        if isinstance(pl, list) and j < len(pl):
            return pl[j]
        return []

    def _on_order_within(self, env, H: int) -> np.ndarray:
        
        H = int(max(0, H))
        oo = np.zeros(env.m, dtype=float)
        for j in range(env.m):
            for eta, q in self._get_pipeline_list(env, j):
                if int(eta) <= H:
                    oo[j] += float(q)
        return oo

    def _get_delivery_window(self, env) -> int:
        # try env attribute first, then cfg dict
        if hasattr(env, "delivery_window"):
            try:
                return int(env.delivery_window)
            except Exception:
                pass
        if isinstance(getattr(env, "_cfg", None), dict):
            try:
                return int(env._cfg.get("delivery_window", 0))
            except Exception:
                pass
        return 0

    def act(self, env: "EnvUnifiedReplay", threads=0, timelimit=60.0):
        t = int(getattr(env, "t", 0))
        # horizon H: worst-case replenishment LT by default
        H = int(env.lt_hi) if self.use_max_lead else int(round(0.5 * (env.lt_lo + env.lt_hi)))

        # smoothing horizon H2: H + (delivery window) by default
        if self.smooth:
            extra = self.smooth_extra
            if extra is None:
                extra = self._get_delivery_window(env)
            H2 = H + int(max(0, extra))
        else:
            H2 = H

        req = np.zeros(env.m, dtype=float)

        # ONLY realized orders (BOM revealed)
        ros = getattr(env, "realized_orders", [])
        for od in ros:
            rem = float(getattr(od, "remaining", 0.0))
            if rem <= 0:
                continue
            t_done = int(getattr(od, "t_design_done", -10**9))
            if t < t_done:
                # should not happen if realized_orders is consistent, but keep safe
                continue

            due = int(getattr(od, "t_due", t))
            slack = due - t  # can be negative if overdue

            # weight function alpha(slack)
            if slack <= H:
                w = 1.0
            elif self.smooth and (H2 > H) and (slack <= H2):
                # linear decay from 1 at H to 0 at H2
                w = float(H2 - slack) / float(H2 - H)
            else:
                w = 0.0

            if w <= 0.0:
                continue

            bom = np.asarray(getattr(od, "bom_per_unit", np.zeros(env.m)), dtype=float)
            req += w * rem * bom

        # Deadline-aware on-order: only count pipeline arriving within H (or H2)
        if self.use_onorder_horizon == "H2":
            oo = self._on_order_within(env, H2)
        else:
            oo = self._on_order_within(env, H)

        IP = env.component_inventory.astype(float) + oo

        q = np.clip(req - IP, 0.0, self.capQ)
        return env.step_truecost_with_q(q, threads=threads, timelimit=timelimit)
"""

class TemporalGRU(nn.Module):
    def __init__(self,in_dim,hid): super().__init__(); self.gru=nn.GRU(in_dim,hid,batch_first=True)
    def forward(self,seq): _,h=self.gru(seq); return h.squeeze(0)

class GATEncoder(nn.Module):
    def __init__(self,node_in_dim,edge_dim,hid=64,out=64):
        super().__init__()
        self.type_emb=nn.Embedding(2,8)
        self.lin_in=nn.Linear(node_in_dim+8,hid)
        self.gat1=GATv2Conv(hid,hid,edge_dim=edge_dim,heads=2,concat=False)
        self.gat2=GATv2Conv(hid,out,edge_dim=edge_dim,heads=2,concat=False)
        self.lin_g=nn.Linear(out,64)
    def forward(self,x,node_type,edge_index,edge_attr):
        x=torch.cat([x,self.type_emb(node_type)],dim=-1)
        x=F.relu(self.lin_in(x))
        x=F.elu(self.gat1(x,edge_index,edge_attr))
        x=self.gat2(x,edge_index,edge_attr)
        g=torch.tanh(self.lin_g(x.mean(dim=0)))
        return x,g

class RL2_ActorGNN(nn.Module):
    def __init__(self,node_dim,edge_dim,temp_in=2,temp_hid=32):
        super().__init__()
        self.temp=TemporalGRU(temp_in,temp_hid)
        self.gnn =GATEncoder(node_dim+temp_hid,edge_dim,64,64)
        self.mu_lambda_head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        self.mu_theta_head =nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,6))
        self.logstd_lambda=nn.Parameter(torch.full((1,),-0.2))
        self.logstd_theta =nn.Parameter(torch.full((6,),-0.2))
    def forward(self,data:Data):
        m=data.comp_idx.numel()
        Ht=self.temp(data.temp_seq_comp)
        add=torch.zeros(data.x.size(0),Ht.size(1),device=data.x.device)
        add[data.comp_idx]=Ht
        H,g=self.gnn(torch.cat([data.x,add],dim=-1),data.node_type,data.edge_index,data.edge_attr)
        Hc=H[data.comp_idx]; g_rep=g.unsqueeze(0).repeat(m,1)
        mu_l=self.mu_lambda_head(torch.cat([Hc,g_rep],dim=-1)).squeeze(-1)
        mu_t=self.mu_theta_head(g)
        ls_l=self.logstd_lambda.expand(m); ls_t=self.logstd_theta
        return mu_l, ls_l, mu_t, ls_t
    @torch.no_grad()
    def act(self,data:Data,deterministic=True):
        mu_l,ls_l,mu_t,ls_t=self.forward(data)
        if deterministic:
            a_l,a_t=mu_l,mu_t
        else:
            a_l=torch.distributions.Normal(mu_l,ls_l.exp()).rsample()
            a_t=torch.distributions.Normal(mu_t,ls_t.exp()).rsample()
        return torch.cat([a_l,a_t],dim=0), None

class CompHist:
    def __init__(self,m:int,L:int):
        self.L=L; self.buf=np.zeros((L,m,2),dtype=float)
    def push(self,orders_j:np.ndarray,arrivals_j:np.ndarray):
        self.buf=np.roll(self.buf,shift=-1,axis=0)
        self.buf[-1,:,0]=orders_j
        self.buf[-1,:,1]=arrivals_j
    def seq(self)->np.ndarray:
        return np.transpose(self.buf,(1,0,2))

def _get_pipeline_snapshot(env, state) -> Dict[int, List[Tuple[int,int]]]:
    ps = getattr(state, "pipeline_snapshot", None)
    if isinstance(ps, dict):
        return ps
    pl = getattr(env, "pipeline", None)
    if isinstance(pl, dict):
        snap = {}
        for j in range(env.m):
            snap[j] = [(int(eta), int(q)) for eta, q in pl.get(j, [])]
        return snap
    return {j: [] for j in range(env.m)}

def _inbound_histogram(pipeline_snapshot: Dict[int, List[Tuple[int,int]]], m:int, L_eff:np.ndarray):
    inbound1=np.zeros(m); inbound2=np.zeros(m); inbound3p=np.zeros(m); cover=np.zeros(m)
    for j in range(m):
        for eta,q in pipeline_snapshot.get(j, []):
            eta=int(eta); q=float(q)
            if eta<=1: inbound1[j]+=q
            elif eta==2: inbound2[j]+=q
            else: inbound3p[j]+=q
            if eta<=float(L_eff[j]): cover[j]+=q
    return inbound1,inbound2,inbound3p,cover

def _dem_lt_consume_compat(env:EnvUnifiedReplay, state, L_eff:np.ndarray, phi:float=0.5):
    if hasattr(env,"_lt_demand_forecast"):
        try:
            dem,_=env._lt_demand_forecast(state,L_eff)
            return np.asarray(dem,float)
        except Exception:
            pass
    lam=np.array(env._cfg["demand_rates"],float) if isinstance(env._cfg, dict) else np.array(env.demand_rates,float)
    Ld=np.array(env.design_leads,float)
    RB=state.realized_backlog.astype(float)
    m=env.m; res=np.zeros(m,float)
    for i in range(env.n):
        mean_i,_,mask_i=env.bom_templates[i]
        mean_i=np.asarray(mean_i,float)
        mask_i=np.asarray(mask_i,bool)
        idx=np.where(mask_i)[0]
        if idx.size==0: continue
        window=np.maximum(0.0,L_eff[idx]-Ld[i])
        need_i=lam[i]*window+phi*RB[i]
        res[idx]+=mean_i[idx]*need_i
    res+=0.3*np.sqrt(np.maximum(1.0,res))
    return res

def build_graph_data_rl2(env:EnvUnifiedReplay, state, hist:CompHist, device:torch.device)->Data:
    n,m=env.n,env.m
    L_eff=np.full(m,0.5*(env.lt_lo+env.lt_hi),float)
    pipeline_snapshot = _get_pipeline_snapshot(env, state)
    inbound1,inbound2,inbound3p,inboundL=_inbound_histogram(pipeline_snapshot,m,L_eff)

    I=state.component_inventory.astype(float)
    dem_lt_cons=_dem_lt_consume_compat(env,state,L_eff,phi=0.5)

    co_mat=np.zeros((m,m),float)
    for i in range(n):
        _,_,mask=env.bom_templates[i]
        idx=np.where(np.asarray(mask,bool))[0]
        for a in idx:
            for b in idx:
                if a!=b: co_mat[a,b]=1.0
    comp_avail=co_mat@(I+inboundL)

    # 9-dim component features
    comp_feat=np.stack([I,inbound1,inbound2,inbound3p,inboundL,dem_lt_cons,
                        np.array(env.comp_order_cost,float),np.array(env.comp_hold_cost,float),comp_avail],axis=1)
    d_comp=comp_feat.shape[1]  # should be 9

    realized=state.realized_backlog.astype(float)
    unreal=state.unrealized_backlog.astype(float)
    overdue=state.overdue_backlog.astype(float)
    back_c=np.array(env.backorder_cost,float)
    prod_feat=np.stack([realized,unreal,overdue,back_c],axis=1)  # 4-dim
    d_prod=prod_feat.shape[1]

    d_node=max(d_comp,d_prod)  # should be 9
    comp_pad=np.zeros((m,d_node)); comp_pad[:,:d_comp]=comp_feat
    prod_pad=np.zeros((n,d_node)); prod_pad[:,:d_prod]=prod_feat
    x=np.vstack([comp_pad,prod_pad])
    node_type=np.concatenate([np.zeros(m,int),np.ones(n,int)])

    # realized BOM summaries for edges
    real_units=np.zeros(n,float)
    real_need=np.zeros((n,m),float)
    for od in getattr(env,"realized_orders",[]):
        rem=float(getattr(od,"remaining",0.0))
        if rem<=0: continue
        i=int(getattr(od,"product_id",0))
        bom=np.asarray(getattr(od,"bom_per_unit",np.zeros(m)),float)
        real_units[i]+=rem
        real_need[i,:]+=rem*bom
    realized_mean=np.zeros((n,m),float)
    mask_nz=real_units>0
    if np.any(mask_nz):
        realized_mean[mask_nz,:]=real_need[mask_nz,:]/(real_units[mask_nz].reshape(-1,1)+1e-8)

    edges_src,edges_dst,eattr=[],[],[]
    for i in range(n):
        mean,std,mask=env.bom_templates[i]
        mean=np.asarray(mean,float); std=np.asarray(std,float); mask=np.asarray(mask,bool)
        idx=np.where(mask)[0]
        for j in idx:
            edges_src.append(m+i)
            edges_dst.append(j)
            eattr.append([mean[j], std[j], realized_mean[i,j], real_need[i,j]])
    if len(edges_src)==0:
        edges_src,edges_dst,eattr=[0],[0],[[0.0,0.0,0.0,0.0]]

    edge_index=np.vstack([np.array(edges_src,int),np.array(edges_dst,int)])
    edge_attr=np.array(eattr,float)

    data=Data(
        x=torch.tensor(x,dtype=torch.float32,device=device),
        node_type=torch.tensor(node_type,dtype=torch.long,device=device),
        edge_index=torch.tensor(edge_index,dtype=torch.long,device=device),
        edge_attr=torch.tensor(edge_attr,dtype=torch.float32,device=device),
    )
    data.comp_idx=torch.arange(0,m,dtype=torch.long,device=device)
    data.prod_idx=torch.arange(m,m+n,dtype=torch.long,device=device)
    data.temp_seq_comp=torch.tensor(hist.seq(),dtype=torch.float32,device=device)

    # hard check: do NOT silently patch dimensions
    if data.x.size(1) != 9 or data.edge_attr.size(1) != 4:
        raise RuntimeError(f"RL2 graph dim mismatch: node_dim={data.x.size(1)} (want 9), edge_dim={data.edge_attr.size(1)} (want 4).")
    return data

class RL2_Wrapper:
    def __init__(self, ckpt_path:str, device:str="cpu", L_seq:int=8):
        self.ckpt_path=ckpt_path
        self.device=torch.device(device)
        self.L_seq=int(L_seq)
        self.actor=None

    def _lazy_build(self, env:EnvUnifiedReplay):
        st=env.reset()
        hist=CompHist(env.m,self.L_seq)
        hist.push(env.last_orders, env.last_arrivals)
        sample=build_graph_data_rl2(env, st, hist, device=self.device)
        node_dim=int(sample.x.size(1))
        edge_dim=int(sample.edge_attr.size(1))

        self.actor=RL2_ActorGNN(node_dim=node_dim, edge_dim=edge_dim, temp_in=2, temp_hid=32).to(self.device)
        sd=torch.load(self.ckpt_path, map_location=self.device)
        if isinstance(sd,dict) and "actor" in sd:
            actor_sd=sd["actor"]
        elif isinstance(sd,dict) and "state_dict" in sd:
            actor_sd=sd["state_dict"]
        else:
            actor_sd=sd
        self.actor.load_state_dict(actor_sd, strict=True)
        self.actor.eval()

    @torch.no_grad()
    def evaluate_once(self, scn:Scenario):
        env=EnvUnifiedReplay(scenario_to_cfg(scn), scn, rng_seed=scn.cfg.seed, use_lblr=True)
        if self.actor is None:
            self._lazy_build(env)
        st=env.reset()
        hist=CompHist(env.m,self.L_seq)
        hist.push(env.last_orders, env.last_arrivals)
        for _ in range(scn.cfg.T):
            data=build_graph_data_rl2(env, st, hist, device=self.device)
            a,_=self.actor.act(data, deterministic=True)
            st,_,_ = env.step(a.detach().cpu().numpy())
            hist.push(env.last_orders, env.last_arrivals)
        return metrics_finalize(env)



class RLBR_Wrapper:

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        self.ckpt_path = ckpt_path
        self.device = torch.device(device)
        self.actor = None
        self.critic = None
        self.L_seq = 8
        self.use_lblr = True
        self._train_cfg = None
        self._raw_ckpt = None

    def load(self):
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self._raw_ckpt = ckpt
        meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
        self.use_lblr = bool(meta.get("use_lblr", True))
        self.L_seq = int(meta.get("L_seq", 8))
        self._train_cfg = meta.get("cfg", None)

    def _ensure_actor_on_env(self, env:EnvUnifiedReplay):
        if self.actor is not None:
            return
        if self._raw_ckpt is None:
            self.load()

        st = env.reset()
        hist = RL.CompHist(env.m, self.L_seq)
        hist.push(env.last_orders, env.last_arrivals)
        sample = RL.build_graph_data(env, st, hist, device=self.device)
        node_dim = sample.x.size(1); edge_dim = sample.edge_attr.size(1)

        actor = RL.ActorGNN(node_dim=node_dim, edge_dim=edge_dim, use_lblr=self.use_lblr, temp_in=2, temp_hid=32).to(self.device)
        critic= RL.CriticGNN(node_dim=node_dim, edge_dim=edge_dim, temp_in=2, temp_hid=32).to(self.device)

        sd = self._raw_ckpt
        sd_actor = sd.get("actor", sd if isinstance(sd, dict) else sd)
        actor.load_state_dict(sd_actor, strict=False)
        if isinstance(sd, dict) and "critic" in sd:
            critic.load_state_dict(sd["critic"], strict=False)

        self.actor = actor.eval()
        self.critic= critic.eval()

    @torch.no_grad()
    def evaluate_once(self, scn: Scenario):
        scfg = scenario_to_cfg(scn)
        cfg_for_env = copy.deepcopy(scfg)

        if isinstance(self._train_cfg, dict):
            # fill missing keys only
            for k,v in self._train_cfg.items():
                if k not in cfg_for_env:
                    cfg_for_env[k] = copy.deepcopy(v)
            # meta: scenario overrides
            meta = dict(self._train_cfg.get("meta", {}))
            meta.update(scfg.get("meta", {}))
            cfg_for_env["meta"] = meta

        env = EnvUnifiedReplay(cfg_for_env, scn, rng_seed=scn.cfg.seed, use_lblr=self.use_lblr)
        if self.actor is None:
            self._ensure_actor_on_env(env)

        st = env.reset()
        hist = RL.CompHist(env.m, self.L_seq)
        hist.push(env.last_orders, env.last_arrivals)
        for _ in range(scn.cfg.T):
            data = RL.build_graph_data(env, st, hist, device=self.device)
            a, _ = self.actor.act(data, deterministic=True)
            st, _, _ = env.step(a.detach().cpu().numpy())
            hist.push(env.last_orders, env.last_arrivals)
        return metrics_finalize(env)


# ================== Eval ================== #
def _mean_ci(lst:List[dict])->Dict[str,Tuple[float,float]]:
    out={}
    for k in PLOT_KEYS:
        out[k]=ci95([x.get(k,np.nan) for x in lst])
    return out

def _attach_pi(raw:Dict[str,List[dict]]):
    or_list = raw.get("Oracle", [])
    if not or_list:
        for name in raw:
            for rec in raw[name]:
                rec["pi"]=np.nan
        return raw
    for i, or_rec in enumerate(or_list):
        or_cost=float(or_rec.get("cost_total", np.nan))
        for name, recs in raw.items():
            if i>=len(recs): continue
            cost=float(recs[i].get("cost_total", np.nan))
            recs[i]["pi"] = (cost/or_cost) if (np.isfinite(cost) and np.isfinite(or_cost) and or_cost>0) else np.nan
    return raw

def eval_rlbr_per_seed(base_params,demand_family,seeds,rlbr_ckpt,device,**_):
    agent=RLBR_Wrapper(rlbr_ckpt, device=device); agent.load()
    recs=[]
    for sd in seeds:
        scn=make_default_scenario(seed=2025+sd, demand_family=demand_family, **base_params)
        recs.append(agent.evaluate_once(scn))
    return recs

def eval_dtp_per_seed(base_params,demand_family,seeds,**mipkw):
    recs=[]
    for sd in seeds:
        scn=make_default_scenario(seed=2025+sd, demand_family=demand_family, **base_params)
        env=EnvUnifiedReplay(scenario_to_cfg(scn), scn, rng_seed=scn.cfg.seed, use_lblr=True)
        env.reset()
        ctrl=DTP_Controller(capQ=2000.0, use_max_lead=True)
        for _ in range(scn.cfg.T):
            ctrl.act(env, threads=mipkw.get("threads",0), timelimit=mipkw.get("timelimit",60.0))
        recs.append(metrics_finalize(env))
    return recs

def eval_nvd_per_seed(base_params,demand_family,seeds,**mipkw):
    recs=[]
    for sd in seeds:
        scn=make_default_scenario(seed=2025+sd, demand_family=demand_family, **base_params)
        env=EnvUnifiedReplay(scenario_to_cfg(scn), scn, rng_seed=scn.cfg.seed, use_lblr=True)
        env.reset()
        ctrl=NVD_Controller(capQ=2000.0)
        for _ in range(scn.cfg.T):
            ctrl.act(env, threads=mipkw.get("threads",0), timelimit=mipkw.get("timelimit",60.0))
        recs.append(metrics_finalize(env))
    return recs

def eval_oracle_per_seed(base_params,demand_family,seeds,**mipkw):
    recs=[]
    for sd in seeds:
        scn=make_default_scenario(seed=2025+sd, demand_family=demand_family, **base_params)
        r=solve_oracle_gurobi(scn, **mipkw)
        recs.append(dict(
            cost_total=r.get("cost_total",np.nan),
            cost_order=r.get("cost_order",np.nan),
            cost_hold =r.get("cost_hold",np.nan),
            cost_late =r.get("cost_late",np.nan),
            service_fill=r.get("service_fill",np.nan),
            ontime=r.get("ontime",np.nan),
            redundancy=np.nan,
            mismatch=np.nan
        ))
    return recs

def eval_rl2_per_seed(base_params,demand_family,seeds,rl2_ckpt,device,**_):
    if not (rl2_ckpt and os.path.exists(rl2_ckpt)):
        print(f"[WARN] RL2 ckpt not found: {rl2_ckpt}")
        return [{k:np.nan for k in PLOT_KEYS} for _ in seeds]
    agent=RL2_Wrapper(rl2_ckpt, device=device, L_seq=8)
    recs=[]
    for sd in seeds:
        scn=make_default_scenario(seed=2025+sd, demand_family=demand_family, **base_params)
        recs.append(agent.evaluate_once(scn))
    return recs

def evaluate_methods(base_params,demand_family,seeds,rlbr_ckpt,device,rl2_ckpt=None,**mipkw):
    rl_lst = eval_rlbr_per_seed(base_params,demand_family,seeds,rlbr_ckpt,device,**mipkw)
    dt_lst = eval_dtp_per_seed(base_params,demand_family,seeds,**mipkw)
    nv_lst = eval_nvd_per_seed(base_params,demand_family,seeds,**mipkw)
    or_lst = eval_oracle_per_seed(base_params,demand_family,seeds,**mipkw)
    r2_lst = eval_rl2_per_seed(base_params,demand_family,seeds,rl2_ckpt,device,**mipkw) if rl2_ckpt else [{k:np.nan for k in PLOT_KEYS} for _ in seeds]

    raw={"RL_BR":rl_lst,"DTP":dt_lst,"NVD":nv_lst,"Oracle":or_lst,"RL2":r2_lst}
    raw=_attach_pi(raw)
    mean_ci={name:_mean_ci(lst) for name,lst in raw.items()}
    return mean_ci, raw



def plot_boxplots(results_per_seed: Dict[str, List[dict]], outdir: str):
    ensure_dir(outdir)
    USED_KEYS = ["RL_BR","DTP","NVD"]#,"RL2"]

    csv_rows=[]
    n_metrics=len(PLOT_KEYS)
    ncols=4 if n_metrics<=8 else 3
    nrows=math.ceil(n_metrics/ncols)
    fig, axes = plt.subplots(nrows,ncols,figsize=(4*ncols,3*nrows))
    axes=np.array(axes).reshape(-1)

    for idx,k in enumerate(PLOT_KEYS):
        data=[]; labels=[]
        for key in USED_KEYS:
            if key not in results_per_seed: continue
            series=[rec.get(k,np.nan) for rec in results_per_seed[key]]
            data.append(series); labels.append(key)
            for si,v in enumerate(series):
                csv_rows.append([key, key, k, si, v])
        ax=axes[idx]
        if data:
            ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_title(Y_LABEL_MAP.get(k,k))
        ax.grid(True, alpha=0.3)

    for j in range(len(PLOT_KEYS), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    p_img=os.path.join(outdir,"boxplots_selected.png")
    plt.savefig(p_img,dpi=170); plt.close()
    print(f"[PLOT] saved: {p_img}")

    p_csv=os.path.join(outdir,"boxplots_selected.csv")
    with open(p_csv,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["algorithm","display_name","metric","seed_index","value"])
        w.writerows(csv_rows)
    print(f"[CSV]  saved: {p_csv}")

def _grid_for(sweep:str):
    if sweep=="rho": return np.linspace(0.0,0.9,6)
    if sweep=="bom_std_ratio": return np.linspace(0.15,0.60,7)
    if sweep=="common_ratio": return np.linspace(0.1,0.8,8)
    if sweep=="delivery_window": return np.array([1,2,3,4,5,6])
    if sweep=="design_lead": return np.array([0,1,2,3,4,5])
    if sweep=="lt_hi": return np.array([2,3,4,5,6])
    if sweep=="price_scale_back": return np.array([1.0,1.5,2.0,2.5,3.0])
    if sweep=="season_amp": return np.linspace(0.0,0.8,9)
    return np.linspace(0.0,0.9,4)

def run_sensitivity_multi(base_params, demand_family, seeds, rlbr_ckpt, device,
                          sweep: str, outdir: str, rl2_ckpt=None, **mipkw):
    grid=_grid_for(sweep)
    outdir=os.path.join(outdir, f"sens_{sweep}")
    ensure_dir(outdir)

    series_mu={k:[] for k in ALGO_KEYS}
    series_ci={k:[] for k in ALGO_KEYS}

    for gv in grid:
        bp=dict(base_params)
        if sweep in ["rho","bom_std_ratio","common_ratio","price_scale_back","season_amp"]:
            bp[sweep]=float(gv)
        elif sweep in ["delivery_window","design_lead","lt_hi"]:
            bp[sweep]=int(gv)

        mean_ci,_ = evaluate_methods(bp, demand_family, seeds, rlbr_ckpt, device, rl2_ckpt=rl2_ckpt, **mipkw)
        for name in ALGO_KEYS:
            series_mu[name].append({k:mean_ci[name][k][0] for k in PLOT_KEYS})
            series_ci[name].append({k:mean_ci[name][k][1] for k in PLOT_KEYS})

    USED_KEYS=["RL_BR","DTP","NVD"]
    markers={"RL_BR":"o","DTP":"s","NVD":"^"}

    for metric in PLOT_KEYS:
        plt.figure(figsize=(7.0,4.2))
        header=["grid"]+USED_KEYS
        rows=[]
        for name in USED_KEYS:
            y=[rec.get(metric,np.nan) for rec in series_mu[name]]
            c=[rec.get(metric,0.0) for rec in series_ci[name]]
            plt.plot(grid,y,marker=markers.get(name,"o"),label=name,linewidth=1.6)
            y_np=np.array(y); c_np=np.array(c)
            plt.fill_between(grid, y_np-c_np, y_np+c_np, alpha=0.15)

        plt.xlabel(X_LABEL_MAP.get(sweep,sweep))
        plt.ylabel(Y_LABEL_MAP.get(metric,metric))
        plt.grid(True, alpha=0.3); plt.ylim(bottom=0)
        plt.legend()
        p_img=os.path.join(outdir,f"line_{sweep}_{metric}_selected.png")
        plt.tight_layout(); plt.savefig(p_img,dpi=170); plt.close()
        print(f"[SENS] plot saved: {p_img}")

        for i,g in enumerate(grid):
            row=[g]
            for name in USED_KEYS:
                row.append(series_mu[name][i].get(metric,np.nan))
            rows.append(row)
        p_csv=os.path.join(outdir,f"line_{sweep}_{metric}_selected.csv")
        with open(p_csv,"w",newline="") as f:
            w=csv.writer(f); w.writerow(header); w.writerows(rows)
        print(f"[SENS] CSV saved: {p_csv}")



def run_all(seed=0, n=6, m=20, T=40, eval_seeds=10, device="cpu",
            demand_family="poisson",
            season_amp=0.3, season_period=12, season_phase=0.0,
            rlbr_ckpt:str=None, rl2_ckpt:str=None,
            do_sensitivity=True, sweep:str="all", outdir:str="figs",
            threads:int=4, time_limit:float=60.0, mip_gap:float=0.02):

    if not (rlbr_ckpt and os.path.exists(rlbr_ckpt)):
        raise FileNotFoundError(f"RL_BR checkpoint not found: {rlbr_ckpt}")

    set_seeds(seed)

    base_params=dict(
        n=n, m=m, T=T,
        rho=0.3,
        bom_std_ratio=0.3,
        common_ratio=0.3,
        delivery_window=2,
        design_lead=3,
        lt_lo=3,
        lt_hi=8,
        price_scale_order=2.0,
        price_scale_hold=1.0,
        price_scale_back=2.0,
        season_amp=float(season_amp),
        season_period=int(season_period),
        season_phase=float(season_phase),
    )

    seeds=[seed+i for i in range(eval_seeds)]
    mipkw=dict(threads=threads, timelimit=time_limit, mipgap=mip_gap)

    print(f"[CFG] demand_family={demand_family}, season_amp={season_amp}, n={n}, m={m}, T={T}, seeds={seeds}")
    mean_ci, raw = evaluate_methods(base_params, demand_family, seeds,
                                    rlbr_ckpt, device,
                                    rl2_ckpt=rl2_ckpt, **mipkw)

    def pretty(name, d):
        print(f"\n{name}")
        for k in PLOT_KEYS:
            mval,ci=d.get(k,(np.nan,0.0))
            if isinstance(mval,float) and np.isnan(mval):
                print(f"{k:>14s}:    nan ± 0.00")
            else:
                print(f"{k:>14s}: {mval:10.2f} ± {ci:.2f}")

    print("\n=== Final Evaluation (RL_BR / DTP / NVD / Oracle / RL2) ===")
    for name in ALGO_KEYS:
        pretty(name, mean_ci[name])

    ensure_dir(outdir)
    plot_boxplots(raw, outdir)

    if do_sensitivity:
        if sweep.strip().lower()=="all":
            for fac in ["rho","bom_std_ratio","common_ratio",
                        "delivery_window","design_lead","lt_hi",
                        "price_scale_back","season_amp"]:
                run_sensitivity_multi(base_params, demand_family, seeds,
                                      rlbr_ckpt, device, fac, outdir,
                                      rl2_ckpt=rl2_ckpt, **mipkw)
        else:
            run_sensitivity_multi(base_params, demand_family, seeds,
                                  rlbr_ckpt, device, sweep, outdir,
                                  rl2_ckpt=rl2_ckpt, **mipkw)



































