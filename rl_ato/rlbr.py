from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .config import PPOConfig
from .env import ATOEnv, ControlAction, Observation
from .scenario import ProblemInstance

try:
    from torch_geometric.nn import GATConv
except Exception:
    GATConv = None


def obs_to_tensors(obs: Observation, device: torch.device | str) -> Dict[str, torch.Tensor]:
    return {
        "comp_features": torch.as_tensor(obs.comp_features, dtype=torch.float32, device=device),
        "prod_features": torch.as_tensor(obs.prod_features, dtype=torch.float32, device=device),
        "edge_index": torch.as_tensor(obs.edge_index, dtype=torch.long, device=device),
        "edge_attr": torch.as_tensor(obs.edge_attr, dtype=torch.float32, device=device),
        "history": torch.as_tensor(obs.history, dtype=torch.float32, device=device),
        "d_hat": torch.as_tensor(obs.d_hat, dtype=torch.float32, device=device),
        "sigma_hat": torch.as_tensor(obs.sigma_hat, dtype=torch.float32, device=device),
        "urgency": torch.as_tensor(obs.urgency, dtype=torch.float32, device=device),
    }


def obs_batch_to_tensors(observations: Sequence[Observation], device: torch.device | str) -> Dict[str, torch.Tensor]:
    if not observations:
        raise ValueError("observations must be non-empty")
    node_count = observations[0].prod_features.shape[0] + observations[0].comp_features.shape[0]
    edge_indices = []
    edge_attrs = []
    for b, obs in enumerate(observations):
        if obs.edge_index.size:
            edge_indices.append(
                torch.as_tensor(obs.edge_index, dtype=torch.long, device=device) + b * node_count
            )
            edge_attrs.append(torch.as_tensor(obs.edge_attr, dtype=torch.float32, device=device))
    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.zeros((0, 3), dtype=torch.float32, device=device)
    return {
        "comp_features": torch.as_tensor(
            np.stack([obs.comp_features for obs in observations]), dtype=torch.float32, device=device
        ),
        "prod_features": torch.as_tensor(
            np.stack([obs.prod_features for obs in observations]), dtype=torch.float32, device=device
        ),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "history": torch.as_tensor(
            np.stack([obs.history for obs in observations]), dtype=torch.float32, device=device
        ),
        "d_hat": torch.as_tensor(np.stack([obs.d_hat for obs in observations]), dtype=torch.float32, device=device),
        "sigma_hat": torch.as_tensor(
            np.stack([obs.sigma_hat for obs in observations]), dtype=torch.float32, device=device
        ),
        "urgency": torch.as_tensor(
            np.stack([obs.urgency for obs in observations]), dtype=torch.float32, device=device
        ),
    }


class _MessageLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, edge_dim: int):
        super().__init__()
        if GATConv is None:
            self.gat = None
            self.self_linear = nn.Linear(hidden, hidden)
            self.msg_linear = nn.Linear(hidden + edge_dim, hidden)
        else:
            out_channels = max(1, hidden // max(1, heads))
            self.ignore_edge_attr = False
            try:
                self.gat = GATConv(
                    hidden,
                    out_channels,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    add_self_loops=True,
                )
            except TypeError:
                self.gat = GATConv(hidden, out_channels, heads=heads, concat=True, add_self_loops=True)
                self.ignore_edge_attr = True
            gat_out = out_channels * heads
            self.out = nn.Identity() if gat_out == hidden else nn.Linear(gat_out, hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if self.gat is not None:
            if getattr(self, "ignore_edge_attr", False):
                return self.out(self.gat(x, edge_index))
            return self.out(self.gat(x, edge_index, edge_attr))
        out = self.self_linear(x)
        if edge_index.numel() == 0:
            return out
        src, dst = edge_index
        msg = self.msg_linear(torch.cat([x[src], edge_attr], dim=-1))
        agg = torch.zeros_like(out)
        agg.index_add_(0, dst, msg)
        degree = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        degree.index_add_(0, dst, torch.ones((dst.shape[0], 1), dtype=x.dtype, device=x.device))
        return out + agg / degree.clamp_min(1.0)


class RLBRActorCritic(nn.Module):
    def __init__(self, instance: ProblemInstance, config: PPOConfig):
        super().__init__()
        self.instance = instance
        self.config = config
        hidden = config.gat_hidden_width
        self.history_gru = nn.GRU(input_size=2, hidden_size=config.gru_hidden_size, batch_first=True)
        self.comp_proj = nn.Linear(5 + config.gru_hidden_size, hidden)
        self.prod_proj = nn.Linear(4, hidden)
        self.layers = nn.ModuleList(
            [_MessageLayer(hidden, config.attention_heads, edge_dim=3) for _ in range(config.gat_layers)]
        )
        self.context_proj = nn.Linear(hidden, config.context_dim)
        self.comp_head = nn.Sequential(
            nn.Linear(hidden + config.context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 5),
        )
        self.eta_head = nn.Sequential(nn.Linear(config.context_dim, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.value_head = nn.Sequential(
            nn.Linear(hidden + config.context_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.action_dim = 4 * instance.J + 1
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -0.5))

    def forward(self, tensors: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        comp = tensors["comp_features"]
        prod = tensors["prod_features"]
        history = tensors["history"]
        _gru_out, h_n = self.history_gru(history)
        hist_emb = h_n[-1]
        comp_x = F.relu(self.comp_proj(torch.cat([comp, hist_emb], dim=-1)))
        prod_x = F.relu(self.prod_proj(prod))
        x = torch.cat([prod_x, comp_x], dim=0)
        edge_index = tensors["edge_index"]
        edge_attr = tensors["edge_attr"]
        for layer in self.layers:
            x = F.relu(layer(x, edge_index, edge_attr))
        context = torch.tanh(self.context_proj(x.mean(dim=0, keepdim=True))).squeeze(0)
        comp_emb = x[self.instance.I : self.instance.I + self.instance.J]
        comp_context = context.expand(self.instance.J, -1)
        raw = self.comp_head(torch.cat([comp_emb, comp_context], dim=-1))
        eta_mean = self.eta_head(context).reshape(1)
        action_mean = torch.cat([raw[:, :4].reshape(-1), eta_mean], dim=0)
        value = self.value_head(torch.cat([comp_emb.mean(dim=0), context], dim=-1)).squeeze(-1)
        aux = {
            "raw": raw,
            "alpha": F.softplus(raw[:, 4]),
            "context": context,
            "comp_emb": comp_emb,
        }
        return action_mean, value, aux

    def forward_batch(
        self, tensors: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        comp = tensors["comp_features"]
        prod = tensors["prod_features"]
        history = tensors["history"]
        batch_size = comp.shape[0]
        I, J = self.instance.I, self.instance.J

        history_flat = history.reshape(batch_size * J, history.shape[-2], history.shape[-1])
        _gru_out, h_n = self.history_gru(history_flat)
        hist_emb = h_n[-1].reshape(batch_size, J, -1)
        comp_x = F.relu(self.comp_proj(torch.cat([comp, hist_emb], dim=-1)))
        prod_x = F.relu(self.prod_proj(prod))
        x = torch.cat([prod_x, comp_x], dim=1).reshape(batch_size * (I + J), -1)
        edge_index = tensors["edge_index"]
        edge_attr = tensors["edge_attr"]
        for layer in self.layers:
            x = F.relu(layer(x, edge_index, edge_attr))
        x = x.reshape(batch_size, I + J, -1)
        context = torch.tanh(self.context_proj(x.mean(dim=1)))
        comp_emb = x[:, I : I + J]
        comp_context = context[:, None, :].expand(-1, J, -1)
        raw = self.comp_head(torch.cat([comp_emb, comp_context], dim=-1))
        eta_mean = self.eta_head(context).reshape(batch_size, 1)
        action_mean = torch.cat([raw[:, :, :4].reshape(batch_size, -1), eta_mean], dim=1)
        value = self.value_head(torch.cat([comp_emb.mean(dim=1), context], dim=-1)).squeeze(-1)
        aux = {
            "raw": raw,
            "alpha": F.softplus(raw[:, :, 4]),
            "context": context,
            "comp_emb": comp_emb,
        }
        return action_mean, value, aux

    def dist_value(self, tensors: Dict[str, torch.Tensor]) -> tuple[torch.distributions.Normal, torch.Tensor, Dict[str, torch.Tensor]]:
        mean, value, aux = self.forward(tensors)
        std = torch.exp(self.log_std).clamp(1e-4, 5.0)
        return torch.distributions.Normal(mean, std), value, aux

    def dist_value_batch(
        self, tensors: Dict[str, torch.Tensor]
    ) -> tuple[torch.distributions.Normal, torch.Tensor, Dict[str, torch.Tensor]]:
        mean, value, aux = self.forward_batch(tensors)
        std = torch.exp(self.log_std).clamp(1e-4, 5.0)
        return torch.distributions.Normal(mean, std), value, aux

    def deterministic_rule_params(
        self, tensors: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, _value, aux = self.forward(tensors)
        return self.rule_params_from_latent(mean, aux)

    def rule_params_from_latent(
        self, latent: torch.Tensor, aux: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        J = self.instance.J
        flat = latent[: 4 * J].reshape(J, 4)
        lam = F.softplus(flat[:, 0])
        w_mu = F.softplus(flat[:, 1])
        w_sigma = F.softplus(flat[:, 2])
        w_u = F.softplus(flat[:, 3])
        eta = self.config.eta_max * torch.sigmoid(latent[-1])
        alpha = aux["alpha"]
        return lam, w_mu, w_sigma, w_u, eta, alpha


@dataclass
class RLBRActionInfo:
    latent: np.ndarray
    logprob: float
    value: float


class RLBRPolicy:
    name = "RLBR"

    def __init__(self, instance: ProblemInstance, config: PPOConfig, model: RLBRActorCritic | None = None):
        self.instance = instance
        self.config = config
        self.device = torch.device(config.device)
        self.model = model if model is not None else RLBRActorCritic(instance, config)
        self.model.to(self.device)
        self._nvd_base_stock: np.ndarray | None = None
        if float(getattr(config, "nvd_floor_scale", 0.0)) > 1e-9:
            self._ensure_nvd_base_stock()

    def act(self, env: ATOEnv, obs: Observation) -> ControlAction:
        action, _info = self.act_with_info(env, obs, deterministic=True)
        return action

    @torch.no_grad()
    def act_with_info(
        self, env: ATOEnv, obs: Observation, deterministic: bool = False
    ) -> tuple[ControlAction, RLBRActionInfo]:
        tensors = obs_to_tensors(obs, self.device)
        dist, value, aux = self.model.dist_value(tensors)
        latent = dist.mean if deterministic else dist.sample()
        logprob = dist.log_prob(latent).sum()
        action = self._latent_to_control(env, obs, latent, aux)
        return action, RLBRActionInfo(
            latent=latent.detach().cpu().numpy(),
            logprob=float(logprob.detach().cpu()),
            value=float(value.detach().cpu()),
        )

    def _latent_to_control(
        self, env: ATOEnv, obs: Observation, latent: torch.Tensor, aux: Dict[str, torch.Tensor]
    ) -> ControlAction:
        lam, w_mu, w_sigma, w_u, eta, alpha = self.model.rule_params_from_latent(latent, aux)
        lam_np = lam.detach().cpu().numpy() * float(self.config.shadow_price_scale)
        eta_value = float(eta.detach().cpu())
        scores: Dict[Tuple[int, int], float] = {}
        for i, s, _remaining, bom in obs.revealed:
            due = s + int(self.instance.design_lead_times[i]) + self.instance.delivery_window
            urgency = np.exp(eta_value * max(0, obs.t - due))
            b_aug = self.instance.backlog_costs[i] + float(np.dot(self.instance.holding_costs, bom > 1e-12))
            scores[(i, s)] = float(urgency * b_aug - np.dot(lam_np, bom))
        allocations = self._allocate(env, obs, scores)
        end_inventory = env.inventory.copy()
        assert env.scenario is not None
        for i, s, qty in allocations:
            end_inventory -= env.scenario.realized_bom[int(i), int(s)] * float(qty)
        end_inventory = np.maximum(end_inventory, 0.0)
        s_target = (
            alpha
            + w_mu * torch.as_tensor(obs.d_hat, dtype=torch.float32, device=self.device)
            + w_sigma * torch.as_tensor(obs.sigma_hat, dtype=torch.float32, device=self.device)
            + w_u * torch.as_tensor(obs.urgency, dtype=torch.float32, device=self.device)
        )
        target = s_target.detach().cpu().numpy() * float(self.config.base_stock_scale)
        if np.min(self.instance.design_lead_times) <= 0:
            target *= float(self.config.short_design_scale)
        floor = (
            float(self.config.base_stock_floor) * obs.d_hat
            + float(self.config.base_stock_safety) * obs.sigma_hat
        )
        if np.any(floor > 1e-9):
            target = np.maximum(target, floor)
        nvd_floor_scale = float(getattr(self.config, "nvd_floor_scale", 0.0))
        if nvd_floor_scale > 1e-9:
            target = np.maximum(target, self._nvd_floor_target(env, allocations, nvd_floor_scale))
        dtp_blend = self._adaptive_dtp_blend()
        if dtp_blend > 1e-9:
            dtp_target = self._dtp_lookahead_target(env, allocations)
            target = (1.0 - dtp_blend) * target + dtp_blend * dtp_target
        ip = end_inventory + obs.outstanding
        max_order = max(10.0, obs.scale * max(2, self.instance.max_replenishment_lead_time) * 4.0)
        orders = np.clip(np.maximum(target - ip, 0.0), 0.0, max_order)
        return ControlAction(allocations=allocations, orders=orders)

    def _adaptive_dtp_blend(self) -> float:
        strength = float(self.config.adaptive_dtp_blend)
        if strength <= 0.0:
            return 0.0
        window_easy = max(0.0, (float(self.instance.delivery_window) - 3.0) / 3.0)
        lead_easy = max(0.0, (5.0 - float(self.instance.max_replenishment_lead_time)) / 3.0)
        return float(np.clip(strength * max(window_easy, lead_easy), 0.0, 1.0))

    def _allocate(
        self,
        env: ATOEnv,
        obs: Observation,
        scores: Dict[Tuple[int, int], float],
    ) -> List[Tuple[int, int, float]]:
        solver = str(getattr(self.config, "allocation_solver", "greedy")).lower()
        if solver == "greedy":
            return env.greedy_allocate(scores)
        from .policies import solve_obca_allocation, solve_weighted_allocation

        mode = str(getattr(self.config, "allocation_weight_mode", "rlbr")).lower()
        if mode == "obca":
            return solve_obca_allocation(env, obs, beta_late=1.0, solver=solver)
        if mode in {"urgency", "due-urgency", "rlbr-urgency"}:
            return self._urgency_weighted_allocation(env, obs, solver=solver)
        cohorts = [(i, s, rem, bom) for i, s, rem, bom in obs.revealed if rem > 1e-9]
        weights = [scores.get((int(i), int(s)), 0.0) for i, s, _rem, _bom in cohorts]
        return solve_weighted_allocation(cohorts, obs.inventory, weights, self.instance.J, solver=solver)

    def _urgency_weighted_allocation(
        self,
        env: ATOEnv,
        obs: Observation,
        solver: str,
    ) -> List[Tuple[int, int, float]]:
        from .policies import solve_weighted_allocation

        cohorts = [(i, s, rem, bom) for i, s, rem, bom in obs.revealed if rem > 1e-9]
        if not cohorts:
            return []
        due_weight = float(getattr(self.config, "allocation_due_weight", 0.0))
        late_weight = float(getattr(self.config, "allocation_late_weight", 1.0))
        holding_aug = bool(getattr(self.config, "allocation_holding_augmented", False))
        horizon = max(1.0, float(self.instance.max_replenishment_lead_time))
        weights: List[float] = []
        for i, s, _rem, bom in cohorts:
            due = s + int(self.instance.design_lead_times[i]) + self.instance.delivery_window
            overdue = max(0.0, float(env.t - due))
            until_due = max(0.0, float(due - env.t))
            near_due = max(0.0, horizon - until_due) / horizon
            base = float(self.instance.backlog_costs[i])
            if holding_aug:
                base += float(np.dot(self.instance.holding_costs, bom > 1e-12))
            weights.append(base * (1.0 + late_weight * overdue + due_weight * near_due))
        return solve_weighted_allocation(cohorts, obs.inventory, weights, self.instance.J, solver=solver)

    def _ensure_nvd_base_stock(self) -> np.ndarray:
        if self._nvd_base_stock is None:
            from .policies import NVDPolicy

            self._nvd_base_stock = np.asarray(NVDPolicy(self.instance).base_stock, dtype=float)
        return self._nvd_base_stock

    def _nvd_floor_target(
        self,
        env: ATOEnv,
        allocations: List[Tuple[int, int, float]],
        safety_scale: float,
    ) -> np.ndarray:
        remaining_after = env.remaining.copy()
        for i, s, qty in allocations:
            remaining_after[int(i), int(s)] = max(0.0, remaining_after[int(i), int(s)] - float(qty))
        if bool(getattr(self.config, "actual_revealed_bom_replenishment", False)):
            known_requirement = self._actual_revealed_known_requirement(env, remaining_after)
        else:
            known_requirement = remaining_after.sum(axis=1) @ self.instance.template_bom
        return float(safety_scale) * self._ensure_nvd_base_stock() + known_requirement

    def _actual_revealed_known_requirement(self, env: ATOEnv, remaining_after: np.ndarray) -> np.ndarray:
        assert env.scenario is not None
        requirement = np.zeros(self.instance.J, dtype=float)
        for i in range(self.instance.I):
            for s in range(env.t + 1):
                rem = float(remaining_after[i, s])
                if rem <= 1e-9:
                    continue
                if env.revealed[i, s]:
                    requirement += env.scenario.realized_bom[i, s] * rem
                else:
                    requirement += self.instance.template_bom[i] * rem
        return requirement

    def _dtp_lookahead_target(self, env: ATOEnv, allocations: List[Tuple[int, int, float]]) -> np.ndarray:
        assert env.scenario is not None
        remaining_after = env.remaining.copy()
        for i, s, qty in allocations:
            remaining_after[int(i), int(s)] = max(0.0, remaining_after[int(i), int(s)] - float(qty))
        target = np.zeros(self.instance.J, dtype=float)
        horizon = env.t + self.instance.max_replenishment_lead_time
        for i in range(self.instance.I):
            for s in range(env.t + 1):
                if not env.revealed[i, s] or remaining_after[i, s] <= 1e-9:
                    continue
                due = s + int(self.instance.design_lead_times[i]) + self.instance.delivery_window
                if due <= horizon:
                    target += env.scenario.realized_bom[i, s] * remaining_after[i, s]
        return target

    def save(self, path: str) -> None:
        torch.save({"model_state": self.model.state_dict(), "ppo_config": asdict(self.config)}, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state"])
