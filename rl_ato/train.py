from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    validation_scenarios: Sequence | None = None,
    validation_interval: int = 0,
    best_path: str | None = None,
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
    best_validation_cost = float("inf")

    while episodes_done < exp_config.train_episodes:
        rollout: List[Transition] = []
        episode_costs = []
        for _ in range(min(ppo_config.rollout_episodes, exp_config.train_episodes - episodes_done)):
            episode = _collect_episode(policy, instance, generator, ppo_config.reward_scale)
            _compute_gae(episode, ppo_config.gamma, ppo_config.gae_lambda)
            rollout.extend(episode)
            episode_costs.append(-sum(tr.raw_reward for tr in episode))
            episodes_done += 1

        stats = _ppo_update(policy, optimizer, rollout, ppo_config, episodes_done / max(1, exp_config.train_episodes))
        stats["episode"] = episodes_done
        stats["mean_cost"] = float(np.mean(episode_costs))
        if validation_scenarios and validation_interval > 0 and (
            episodes_done % validation_interval == 0 or episodes_done >= exp_config.train_episodes
        ):
            from .evaluate import evaluate_policy

            validation = evaluate_policy(policy, instance, validation_scenarios)
            validation_cost = float(validation["cost"].mean())
            stats["validation_cost"] = validation_cost
            if validation_cost < best_validation_cost:
                best_validation_cost = validation_cost
                if best_path is not None:
                    Path(best_path).parent.mkdir(parents=True, exist_ok=True)
                    policy.save(best_path)
            stats["best_validation_cost"] = best_validation_cost
        history.append(stats)
        if progress:
            val_msg = ""
            if "validation_cost" in stats:
                val_msg = f" val={stats['validation_cost']:.2f} best={stats['best_validation_cost']:.2f}"
            print(
                f"episode={episodes_done:5d} cost={stats['mean_cost']:.2f} "
                f"loss={stats['loss']:.4f} policy={stats['policy_loss']:.4f} value={stats['value_loss']:.4f}"
                f"{val_msg}"
            )
    if best_path is not None and Path(best_path).exists():
        policy.load(best_path)
    return policy, history


def _collect_episode(
    policy: RLBRPolicy, instance: ProblemInstance, generator: ScenarioGenerator, reward_scale: float
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
                reward=float(reward) * float(reward_scale),
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
            clipped = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * adv_t
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values_t, ret_t)
            reg_loss, reg_parts = _structural_regularization(policy, [rollout[int(i)].obs for i in idx], config, progress_frac)
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_t + reg_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.model.parameters(), config.max_grad_norm)
            optimizer.step()
            last_stats = {
                "loss": float(loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy_t.detach().cpu()),
                **reg_parts,
            }
    return last_stats


def _anneal(start: float, end: float, progress_frac: float) -> float:
    progress_frac = float(np.clip(progress_frac, 0.0, 1.0))
    return start + (end - start) * progress_frac


def _structural_regularization(
    policy: RLBRPolicy, observations: Sequence[Observation], config: PPOConfig, progress_frac: float
) -> tuple[torch.Tensor, Dict[str, float]]:
    device = policy.device
    inst = policy.instance
    comp = inst.complement_matrix
    alpha_mon = _anneal(config.alpha_mon_start, config.alpha_mon_end, progress_frac)
    alpha_lip = _anneal(config.alpha_lip_start, config.alpha_lip_end, progress_frac)
    alpha_rat = _anneal(config.alpha_rat_start, config.alpha_rat_end, progress_frac)
    sample_obs = observations[: max(0, min(config.regularization_samples, len(observations)))]
    if not sample_obs or not comp.any() or (alpha_mon + alpha_lip + alpha_rat) <= 0:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return zero, {"reg_mon": 0.0, "reg_lip": 0.0, "reg_rat": 0.0}

    mon_terms: List[torch.Tensor] = []
    lip_terms: List[torch.Tensor] = []
    rat_terms: List[torch.Tensor] = []
    for obs in sample_obs:
        tensors = obs_to_tensors(obs, device)
        comp_features = tensors["comp_features"].clone().detach().requires_grad_(True)
        tensors["comp_features"] = comp_features
        lam, w_mu, w_sigma, w_u, _eta, alpha = policy.model.deterministic_rule_params(tensors)
        d_hat = tensors["d_hat"]
        sigma_hat = tensors["sigma_hat"]
        urgency = tensors["urgency"]
        s_target = alpha + w_mu * d_hat + w_sigma * sigma_hat + w_u * urgency
        for j in range(inst.J):
            comps = np.flatnonzero(comp[j])
            if len(comps) == 0:
                continue
            grad_s = torch.autograd.grad(
                s_target[j], comp_features, retain_graph=True, create_graph=True, allow_unused=True
            )[0]
            grad_l = torch.autograd.grad(
                lam[j], comp_features, retain_graph=True, create_graph=True, allow_unused=True
            )[0]
            if grad_s is None or grad_l is None:
                continue
            ds = grad_s[comps, 0]
            dl = grad_l[comps, 0]
            mon_terms.append(F.relu(-ds).sum())
            lip_terms.append(F.relu(ds.sum() - 1.0))
            rat_terms.append(F.relu(dl).sum())

    def _mean_or_zero(xs: List[torch.Tensor]) -> torch.Tensor:
        if not xs:
            return torch.zeros((), dtype=torch.float32, device=device)
        return torch.stack(xs).mean()

    mon = _mean_or_zero(mon_terms)
    lip = _mean_or_zero(lip_terms)
    rat = _mean_or_zero(rat_terms)
    total = alpha_mon * mon + alpha_lip * lip + alpha_rat * rat
    return total, {
        "reg_mon": float(mon.detach().cpu()),
        "reg_lip": float(lip.detach().cpu()),
        "reg_rat": float(rat.detach().cpu()),
    }
