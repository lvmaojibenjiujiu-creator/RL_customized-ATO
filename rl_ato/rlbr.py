from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, scatter, softmax

from .config import PPOConfig
from .env import ATOEnv, ControlAction, Observation
from .scenario import ProblemInstance


def augmented_backlog_penalty(instance: ProblemInstance, product: int) -> float:
    return float(
        instance.backlog_costs[product]
        + np.dot(instance.holding_costs, instance.template_bom[product])
    )


def obs_to_tensors(obs: Observation, device: torch.device | str) -> Dict[str, torch.Tensor]:
    return {
        "comp_features": torch.as_tensor(obs.comp_features, dtype=torch.float32, device=device),
        "prod_features": torch.as_tensor(obs.prod_features, dtype=torch.float32, device=device),
        "edge_index": torch.as_tensor(obs.edge_index, dtype=torch.long, device=device),
        "edge_attr": torch.as_tensor(obs.edge_attr, dtype=torch.float32, device=device),
        "history": torch.as_tensor(obs.history, dtype=torch.float32, device=device),
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
        "urgency": torch.as_tensor(
            np.stack([obs.urgency for obs in observations]), dtype=torch.float32, device=device
        ),
    }


class _MessageLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, edge_dim: int):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden width must be divisible by attention heads")
        self.heads = int(heads)
        self.width = int(hidden // heads)
        self.message = nn.Linear(hidden + edge_dim, hidden, bias=False)
        self.destination = nn.Linear(hidden, hidden, bias=False)
        self.attention_source = nn.Parameter(torch.empty(heads, self.width))
        self.attention_destination = nn.Parameter(torch.empty(heads, self.width))
        self.bias = nn.Parameter(torch.zeros(hidden))
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_destination)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        edge_index, edge_attr = add_self_loops(
            edge_index,
            edge_attr,
            fill_value=0.0,
            num_nodes=x.shape[0],
        )
        src, dst = edge_index
        message = self.message(torch.cat([x[src], edge_attr], dim=-1)).view(-1, self.heads, self.width)
        destination = self.destination(x[dst]).view(-1, self.heads, self.width)
        logits = F.leaky_relu(
            (message * self.attention_source).sum(dim=-1)
            + (destination * self.attention_destination).sum(dim=-1),
            negative_slope=0.2,
        )
        weights = softmax(logits, dst, num_nodes=x.shape[0])
        values = message * weights.unsqueeze(-1)
        aggregated = scatter(values, dst, dim=0, dim_size=x.shape[0], reduce="sum")
        return aggregated.reshape(x.shape[0], -1) + self.bias


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
        self.demand_feature_head = nn.Sequential(
            nn.Linear(hidden + config.context_dim, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 2),
        )
        expected_lead_time = float(instance.expected_lead_time)
        component_mean = instance.template_bom.T @ instance.demand_lambdas
        lead_values = np.arange(
            instance.min_replenishment_lead_time,
            instance.max_replenishment_lead_time + 1,
            dtype=float,
        )
        lead_variance = float(lead_values.var())
        second_moment = np.square(instance.template_bom) * (1.0 + float(instance.bom_cv) ** 2)
        component_variance = second_moment.T @ instance.demand_lambdas
        lead_demand_mean = expected_lead_time * component_mean
        lead_demand_variance = expected_lead_time * component_variance + lead_variance * np.square(component_mean)
        self.register_buffer(
            "demand_prior",
            torch.as_tensor(
                np.maximum(lead_demand_mean / float(instance.feature_scale), 1e-3),
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "deviation_prior",
            torch.as_tensor(
                np.maximum(np.sqrt(lead_demand_variance) / float(instance.feature_scale), 1e-3),
                dtype=torch.float32,
            ),
        )
        self.alpha_raw = nn.Parameter(torch.full((instance.J,), self._inverse_softplus(0.1)))
        self.comp_head = nn.Sequential(
            nn.Linear(hidden + config.context_dim + 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),
        )
        self.eta_head = nn.Sequential(nn.Linear(config.context_dim, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.value_head = nn.Sequential(
            nn.Linear(hidden + config.context_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.action_dim = 4 * instance.J + 1
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -0.5))
        nn.init.normal_(self.demand_feature_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.demand_feature_head[-1].bias)
        nn.init.normal_(self.comp_head[-1].weight, mean=0.0, std=1e-3)
        self.comp_head[-1].bias.data.copy_(
            torch.tensor(
                [
                    self._inverse_softplus(1.0),
                    self._inverse_softplus(1.0),
                    self._inverse_softplus(0.1),
                    self._inverse_softplus(0.1),
                ],
                dtype=torch.float32,
            )
        )
        nn.init.normal_(self.eta_head[-1].weight, mean=0.0, std=1e-3)
        eta_fraction = min(max(0.1 / max(float(config.eta_max), 1e-9), 1e-4), 1.0 - 1e-4)
        self.eta_head[-1].bias.data.fill_(float(np.log(eta_fraction / (1.0 - eta_fraction))))

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        return float(np.log(np.expm1(float(value))))

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
        encoded = torch.cat([comp_emb, comp_context], dim=-1)
        learned_adjustment = torch.exp(torch.clamp(self.demand_feature_head(encoded), -4.0, 4.0))
        learned_normalized = learned_adjustment * torch.stack(
            [self.demand_prior, self.deviation_prior],
            dim=-1,
        )
        alpha = F.softplus(self.alpha_raw)
        raw = self.comp_head(
            torch.cat([encoded, learned_normalized, alpha.unsqueeze(-1)], dim=-1)
        )
        eta_mean = self.eta_head(context).reshape(1)
        action_mean = torch.cat(
            [raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3], eta_mean],
            dim=0,
        )
        value = self.value_head(torch.cat([comp_emb.mean(dim=0), context], dim=-1)).squeeze(-1)
        aux = {
            "raw": raw,
            "alpha": alpha,
            "d_hat": learned_normalized[:, 0] * float(self.instance.feature_scale),
            "sigma_hat": learned_normalized[:, 1] * float(self.instance.feature_scale),
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
        encoded = torch.cat([comp_emb, comp_context], dim=-1)
        learned_adjustment = torch.exp(torch.clamp(self.demand_feature_head(encoded), -4.0, 4.0))
        prior = torch.stack([self.demand_prior, self.deviation_prior], dim=-1)
        learned_normalized = learned_adjustment * prior.unsqueeze(0)
        alpha = F.softplus(self.alpha_raw)[None, :].expand(batch_size, -1)
        raw = self.comp_head(
            torch.cat([encoded, learned_normalized, alpha.unsqueeze(-1)], dim=-1)
        )
        eta_mean = self.eta_head(context).reshape(batch_size, 1)
        action_mean = torch.cat(
            [raw[:, :, 0], raw[:, :, 1], raw[:, :, 2], raw[:, :, 3], eta_mean],
            dim=1,
        )
        value = self.value_head(torch.cat([comp_emb.mean(dim=1), context], dim=-1)).squeeze(-1)
        aux = {
            "raw": raw,
            "alpha": alpha,
            "d_hat": learned_normalized[:, :, 0] * float(self.instance.feature_scale),
            "sigma_hat": learned_normalized[:, :, 1] * float(self.instance.feature_scale),
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
        lam = F.softplus(latent[0:J])
        w_mu = F.softplus(latent[J : 2 * J])
        w_sigma = F.softplus(latent[2 * J : 3 * J])
        w_u = F.softplus(latent[3 * J : 4 * J])
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
        lam_np = lam.detach().cpu().numpy()
        eta_value = float(eta.detach().cpu())
        scores = self._rationing_scores(obs, lam_np, eta_value)
        allocations = self._allocate(env, obs, scores)
        end_inventory = env.inventory.copy()
        assert env.scenario is not None
        for i, s, qty in allocations:
            end_inventory -= env.scenario.realized_bom[int(i), int(s)] * float(qty)
        end_inventory = np.maximum(end_inventory, 0.0)
        s_target = (
            alpha
            + w_mu * aux["d_hat"]
            + w_sigma * aux["sigma_hat"]
            + w_u * torch.as_tensor(obs.urgency, dtype=torch.float32, device=self.device)
        )
        target = s_target.detach().cpu().numpy()
        ip = end_inventory + obs.outstanding
        orders = np.maximum(0.0, np.ceil(target - ip - 1e-9))
        return ControlAction(allocations=allocations, orders=orders)

    def _rationing_scores(
        self,
        obs: Observation,
        shadow_prices: np.ndarray,
        eta: float,
    ) -> Dict[Tuple[int, int], float]:
        prices = np.asarray(shadow_prices, dtype=float)
        scores: Dict[Tuple[int, int], float] = {}
        for product, cohort_period, _remaining, realized_bom in obs.revealed:
            due = (
                cohort_period
                + int(self.instance.design_lead_times[product])
                + self.instance.delivery_window
            )
            urgency = np.exp(float(eta) * max(0, obs.t - due))
            scores[(product, cohort_period)] = float(
                urgency * augmented_backlog_penalty(self.instance, product)
                - np.dot(prices, realized_bom)
            )
        return scores

    def _allocate(
        self,
        env: ATOEnv,
        obs: Observation,
        scores: Dict[Tuple[int, int], float],
    ) -> List[Tuple[int, int, float]]:
        return env.greedy_allocate(scores)

    def save(self, path: str) -> None:
        torch.save({"model_state": self.model.state_dict(), "ppo_config": asdict(self.config)}, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state"])
