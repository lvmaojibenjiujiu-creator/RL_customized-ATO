from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .config import ExperimentConfig, PPOConfig
from .env import ATOEnv, Observation
from .rlbr import RLBRPolicy, obs_batch_to_tensors, obs_to_tensors
from .scenario import ProblemInstance, ScenarioGenerator


@dataclass
class Transition:
    obs: Observation
    latent: np.ndarray
    old_logprob: float
    value: float
    reward: float
    raw_reward: float
    done: bool
    advantage: float = 0.0
    ret: float = 0.0


def train_rlbr(
    instance: ProblemInstance,
    exp_config: ExperimentConfig,
    ppo_config: PPOConfig,
    progress: bool = True,
) -> tuple[RLBRPolicy, List[Dict[str, float]]]:
    network_seed = int(ppo_config.network_seed)
    if network_seed < 0:
        network_seed = int(instance.seeds["network_initialization"])
    torch.manual_seed(network_seed)
    np.random.seed(network_seed)
    policy = RLBRPolicy(instance, ppo_config)
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=ppo_config.learning_rate, betas=(0.9, 0.999))
    generator = ScenarioGenerator(instance, seed=instance.seeds["training_environment"])
    history: List[Dict[str, float]] = []
    episodes_done = 0

    while episodes_done < exp_config.train_episodes:
        rollout: List[Transition] = []
        episode_costs = []
        for _ in range(min(ppo_config.rollout_episodes, exp_config.train_episodes - episodes_done)):
            episode = _collect_episode(policy, instance, generator)
            _compute_gae(episode, ppo_config.gamma, ppo_config.gae_lambda)
            rollout.extend(episode)
            episode_costs.append(
                -sum((ppo_config.gamma**period) * tr.raw_reward for period, tr in enumerate(episode))
            )
            episodes_done += 1

        stats = _ppo_update(policy, optimizer, rollout, ppo_config, episodes_done / max(1, exp_config.train_episodes))
        stats["episode"] = episodes_done
        stats["mean_cost"] = float(np.mean(episode_costs))
        history.append(stats)
        if progress:
            print(
                f"episode={episodes_done:5d} cost={stats['mean_cost']:.2f} "
                f"loss={stats['loss']:.4f} policy={stats['policy_loss']:.4f} value={stats['value_loss']:.4f}"
            )
    return policy, history


def _collect_episode(
    policy: RLBRPolicy,
    instance: ProblemInstance,
    generator: ScenarioGenerator,
) -> List[Transition]:
    env = ATOEnv(instance)
    obs = env.reset(generator.sample())
    done = False
    episode: List[Transition] = []
    while not done:
        action, info = policy.act_with_info(env, obs, deterministic=False)
        next_obs, reward, done, _info = env.step(action)
        episode.append(
            Transition(
                obs=obs,
                latent=info.latent,
                old_logprob=info.logprob,
                value=info.value,
                reward=float(reward),
                raw_reward=float(reward),
                done=done,
            )
        )
        if next_obs is not None:
            obs = next_obs
    return episode


def _compute_gae(episode: List[Transition], gamma: float, lam: float) -> None:
    next_value = 0.0
    gae = 0.0
    for tr in reversed(episode):
        mask = 0.0 if tr.done else 1.0
        delta = tr.reward + gamma * next_value * mask - tr.value
        gae = delta + gamma * lam * mask * gae
        tr.advantage = gae
        tr.ret = tr.advantage + tr.value
        next_value = tr.value


def _ppo_update(
    policy: RLBRPolicy,
    optimizer: torch.optim.Optimizer,
    rollout: Sequence[Transition],
    config: PPOConfig,
    progress_frac: float,
) -> Dict[str, float]:
    device = policy.device
    advantages = torch.as_tensor([tr.advantage for tr in rollout], dtype=torch.float32, device=device)
    returns = torch.as_tensor([tr.ret for tr in rollout], dtype=torch.float32, device=device)
    old_logprobs = torch.as_tensor([tr.old_logprob for tr in rollout], dtype=torch.float32, device=device)
    latents = torch.as_tensor(np.stack([tr.latent for tr in rollout]), dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    n = len(rollout)
    batch_size = min(config.batch_size, n)
    last_stats: Dict[str, float] = {}
    clip_range = _anneal(config.clip_start, config.clip_end, progress_frac)
    entropy_coef = _anneal(config.entropy_start, config.entropy_end, progress_frac)

    for _epoch in range(config.epochs):
        perm = torch.randperm(n, device=device).cpu().numpy()
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch_obs = [rollout[int(row)].obs for row in idx]
            tensors = obs_batch_to_tensors(batch_obs, device)
            dist, values_t, _aux = policy.model.dist_value_batch(tensors)
            logprobs_t = dist.log_prob(latents[idx]).sum(dim=-1)
            entropy_t = dist.entropy().sum(dim=-1).mean()
            values_t = values_t.reshape(-1)
            adv_t = advantages[idx]
            ret_t = returns[idx]
            old_t = old_logprobs[idx]
            ratio = torch.exp(logprobs_t - old_t)
            unclipped = ratio * adv_t
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_t
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values_t, ret_t)
            base_loss = policy_loss + config.value_coef * value_loss - entropy_coef * entropy_t
            optimizer.zero_grad(set_to_none=True)
            base_loss.backward()

            reg_totals = {"reg_mon": 0.0, "reg_lip": 0.0, "reg_rat": 0.0}
            reg_loss_value = 0.0
            for observation in batch_obs:
                state_reg_loss, state_reg_parts = _structural_regularization(
                    policy,
                    [observation],
                    config,
                    progress_frac,
                )
                scaled_state_loss = state_reg_loss / float(len(batch_obs))
                if scaled_state_loss.requires_grad:
                    scaled_state_loss.backward()
                reg_loss_value += float(state_reg_loss.detach().cpu()) / float(len(batch_obs))
                for key in reg_totals:
                    reg_totals[key] += state_reg_parts[key] / float(len(batch_obs))
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), config.max_grad_norm)
            optimizer.step()
            last_stats = {
                "loss": float(base_loss.detach().cpu()) + reg_loss_value,
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy_t.detach().cpu()),
                "clip_range": float(clip_range),
                "entropy_coef": float(entropy_coef),
                **reg_totals,
            }
    return last_stats


def _anneal(start: float, end: float, progress_frac: float) -> float:
    progress_frac = float(np.clip(progress_frac, 0.0, 1.0))
    return start + (end - start) * progress_frac


def _coavailability_from_normalized_state(
    instance: ProblemInstance,
    inventory_normalized: torch.Tensor,
    outstanding_normalized: torch.Tensor,
) -> torch.Tensor:
    stock = inventory_normalized + outstanding_normalized
    values: List[torch.Tensor] = []
    for component in range(instance.J):
        contributions: List[torch.Tensor] = []
        for product in range(instance.I):
            if not instance.support[product, component]:
                continue
            family_components = np.flatnonzero(instance.support[product])
            complements = family_components[family_components != component]
            if len(complements) == 0:
                continue
            complement_index = torch.as_tensor(complements, dtype=torch.long, device=stock.device)
            denominators = torch.as_tensor(
                instance.template_bom[product, complements],
                dtype=stock.dtype,
                device=stock.device,
            ).clamp_min(1e-9)
            supported_units = torch.min(stock[complement_index] / denominators)
            contributions.append(float(instance.template_bom[product, component]) * supported_units)
        if contributions:
            values.append(torch.stack(contributions).sum())
        else:
            values.append(stock.new_zeros(()))
    return torch.stack(values)


def _structural_regularization(
    policy: RLBRPolicy, observations: Sequence[Observation], config: PPOConfig, progress_frac: float
) -> tuple[torch.Tensor, Dict[str, float]]:
    device = policy.device
    inst = policy.instance
    comp = inst.complement_matrix
    alpha_mon = _anneal(config.alpha_mon_start, config.alpha_mon_end, progress_frac)
    alpha_lip = _anneal(config.alpha_lip_start, config.alpha_lip_end, progress_frac)
    alpha_rat = _anneal(config.alpha_rat_start, config.alpha_rat_end, progress_frac)
    if not observations or not comp.any() or (alpha_mon + alpha_lip + alpha_rat) <= 0:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return zero, {"reg_mon": 0.0, "reg_lip": 0.0, "reg_rat": 0.0}

    mon_by_state: List[torch.Tensor] = []
    lip_by_state: List[torch.Tensor] = []
    rat_by_state: List[torch.Tensor] = []
    for obs in observations:
        tensors = obs_to_tensors(obs, device)
        base_features = tensors["comp_features"].clone().detach()
        inventory_normalized = base_features[:, 0].clone().requires_grad_(True)
        coavailability = _coavailability_from_normalized_state(
            inst,
            inventory_normalized,
            base_features[:, 1],
        )
        comp_features = torch.cat(
            [
                inventory_normalized[:, None],
                base_features[:, 1:4],
                coavailability[:, None],
            ],
            dim=1,
        )
        tensors["comp_features"] = comp_features
        mean, _value, aux = policy.model.forward(tensors)
        lam, w_mu, w_sigma, w_u, _eta, alpha = policy.model.rule_params_from_latent(mean, aux)
        urgency = tensors["urgency"]
        s_target = alpha + w_mu * aux["d_hat"] + w_sigma * aux["sigma_hat"] + w_u * urgency
        state_mon = torch.zeros((), dtype=torch.float32, device=device)
        state_lip = torch.zeros((), dtype=torch.float32, device=device)
        state_rat = torch.zeros((), dtype=torch.float32, device=device)
        for j in range(inst.J):
            comps = np.flatnonzero(comp[j])
            if len(comps) == 0:
                continue
            grad_s = torch.autograd.grad(
                s_target[j], inventory_normalized, retain_graph=True, create_graph=True, allow_unused=True
            )[0]
            grad_l = torch.autograd.grad(
                lam[j], inventory_normalized, retain_graph=True, create_graph=True, allow_unused=True
            )[0]
            if grad_s is None or grad_l is None:
                continue
            ds = grad_s[comps] / float(inst.feature_scale)
            dl = grad_l[comps] / float(inst.feature_scale)
            state_mon = state_mon + F.relu(-ds).sum()
            state_lip = state_lip + F.relu(ds.sum() - 1.0)
            state_rat = state_rat + F.relu(dl).sum()
        mon_by_state.append(state_mon)
        lip_by_state.append(state_lip)
        rat_by_state.append(state_rat)

    def _mean_or_zero(xs: List[torch.Tensor]) -> torch.Tensor:
        if not xs:
            return torch.zeros((), dtype=torch.float32, device=device)
        return torch.stack(xs).mean()

    mon = _mean_or_zero(mon_by_state)
    lip = _mean_or_zero(lip_by_state)
    rat = _mean_or_zero(rat_by_state)
    total = alpha_mon * mon + alpha_lip * lip + alpha_rat * rat
    return total, {
        "reg_mon": float(mon.detach().cpu()),
        "reg_lip": float(lip.detach().cpu()),
        "reg_rat": float(rat.detach().cpu()),
    }
