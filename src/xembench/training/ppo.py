"""CleanRL-style PPO on GPU-vectorized ManiSkill envs (no SB3: its CPU VecEnv
and numpy round-trips waste ManiSkill's torch-native batched sim).

Two modes share the same GAE/update core:

train_ppo_state   — dense-reward expert on flat state obs, NATIVE actions.
                    This is how G1 demonstration experts are produced (B2).
train_ppo_visual  — sparse-reward fine-tuning of a trained BCPolicy on
                    unified obs, CANONICAL actions through the adapter, with
                    an L2 pull toward the frozen BC actor (policy-collapse
                    guard) and a per-step penalty for efficiency.

Both run unchanged with num_envs=1 CPU sim (Mac smoke) and num_envs>=1024
GPU sim (5090).
"""

import copy
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from xembench.constants import A_CANON
from xembench.embodiment import get_adapter
from xembench.envs import make_env
from xembench.envs.wrappers import UnifiedObsWrapper
from xembench.language.tokenizer import get_tokenizer
from xembench.models.ac_policy import StateActorCritic, VisualActorCritic
from xembench.models.bc_policy import BCPolicy
from xembench.utils.config import (
    dump_run_config,
    load_yaml,
    repo_root,
    resolve_path,
)


# --------------------------------------------------------------------------
# shared core
# --------------------------------------------------------------------------
def compute_gae(rewards, values, dones, next_value, next_done, gamma, lam):
    """rewards/values/dones: [T, N]; returns (advantages, returns) [T, N]."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = 0
    for t in reversed(range(T)):
        if t == T - 1:
            nonterminal = 1.0 - next_done.float()
            nextval = next_value
        else:
            nonterminal = 1.0 - dones[t + 1].float()
            nextval = values[t + 1]
        delta = rewards[t] + gamma * nextval * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


def ppo_update(agent, optimizer, batch_fn, n_minibatches, epochs, clip_coef,
               ent_coef, vf_coef, max_grad_norm, extra_loss_fn=None,
               target_kl=None):
    """batch_fn(mb_idx) -> (logprob_new, entropy, value_new, logprob_old,
    advantage, return_). extra_loss_fn(mb_idx) -> scalar (e.g. BC reg).
    target_kl: stop the epoch loop once the epoch-mean approx KL exceeds it —
    staged success bonuses make the return scale jump when a policy starts
    succeeding, and unbounded epochs then produce policy-wrecking KL spikes."""
    stats = {}
    idx_all = torch.randperm(batch_fn.batch_size, device=batch_fn.device)
    mb_size = batch_fn.batch_size // n_minibatches
    for _ in range(epochs):
        epoch_kls = []
        for start in range(0, batch_fn.batch_size, mb_size):
            mb = idx_all[start:start + mb_size]
            newlogprob, entropy, newvalue, oldlogprob, adv, ret = batch_fn(mb)
            logratio = newlogprob - oldlogprob
            ratio = logratio.exp()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg = torch.max(-adv * ratio,
                           -adv * ratio.clamp(1 - clip_coef, 1 + clip_coef)).mean()
            v_loss = 0.5 * F.mse_loss(newvalue.view(-1), ret)
            ent = entropy.mean()
            loss = pg - ent_coef * ent + vf_coef * v_loss
            if extra_loss_fn is not None:
                reg = extra_loss_fn(mb)
                loss = loss + reg
                stats["bc_reg"] = float(reg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                max_grad_norm,
            )
            optimizer.step()
            epoch_kls.append(float(((ratio - 1) - logratio).mean()))
            stats.update(pg_loss=float(pg), v_loss=float(v_loss), entropy=float(ent),
                         approx_kl=epoch_kls[-1])
        if target_kl is not None and sum(epoch_kls) / len(epoch_kls) > target_kl:
            break
    return stats


class _BatchFn:
    """Small closure holder so ppo_update can shuffle minibatches."""

    def __init__(self, fn, batch_size, device):
        self.fn, self.batch_size, self.device = fn, batch_size, device

    def __call__(self, mb):
        return self.fn(mb)


def _episode_stats(infos, device):
    """Per-episode success outcomes (0/1 tensor) from ManiSkillVectorEnv
    final_info, one entry per episode that ended this step. Consumers pool
    these in a rolling buffer spanning several episode horizons: successes
    terminate EARLY while failures only appear at horizon truncation, so any
    window shorter than a full horizon can contain only successes and read
    100% no matter how bad the policy is (this once picked a best.pt from a
    handful of lucky episodes)."""
    if "final_info" not in infos:
        return None
    fi = infos["final_info"]
    mask = infos.get("_final_info", torch.ones(1, dtype=torch.bool))
    if isinstance(fi, dict):  # ManiSkill: dict of batched tensors
        succ = fi.get("success")
        if succ is not None:
            mask = torch.as_tensor(mask, device=succ.device)
            return succ[mask].float() if mask.any() else None
    return None


# --------------------------------------------------------------------------
# mode 1: flat-state expert (native actions, dense reward)
# --------------------------------------------------------------------------
def train_ppo_state(cfg: dict) -> Path:
    device = torch.device(cfg.get("device") if cfg.get("device") not in (None, "auto")
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    tasks = load_yaml(repo_root() / "configs" / "tasks.yaml")["tasks"]
    task = tasks[cfg["task"]]
    out_dir = Path(cfg["out_dir"]).resolve()  # SAPIEN may chdir; stay absolute
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_run_config(cfg, out_dir)
    writer = SummaryWriter(out_dir / "tb")

    raw = make_env(task["env_id"], num_envs=cfg["num_envs"], obs_mode="state",
                   reward_mode=cfg.get("reward_mode", "dense"),
                   **(cfg.get("env_kwargs") or {}))
    envs = ManiSkillVectorEnv(raw, auto_reset=True, ignore_terminations=False,
                              record_metrics=False)
    obs, _ = envs.reset(seed=cfg["seed"])
    obs = obs.to(device)
    state_dim = obs.shape[-1]
    action_dim = envs.single_action_space.shape[-1]
    N = cfg["num_envs"]
    T = cfg["num_steps"]

    agent = StateActorCritic(state_dim, action_dim, hidden=cfg.get("hidden", 256)).to(device)
    if cfg.get("init_ckpt"):
        # warm start (recorded in the run's config.yaml): continue training an
        # earlier expert instead of re-exploring from scratch
        ckpt = torch.load(resolve_path(cfg["init_ckpt"]), map_location=device,
                          weights_only=True)
        assert ckpt["state_dim"] == state_dim and ckpt["action_dim"] == action_dim, (
            f"init_ckpt dims {ckpt['state_dim']}/{ckpt['action_dim']} != "
            f"env {state_dim}/{action_dim}")
        agent.load_state_dict(ckpt["model"])
        print(f"warm start from {cfg['init_ckpt']}")
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg["lr"], eps=1e-5)

    buf_obs = torch.zeros(T, N, state_dim, device=device)
    buf_act = torch.zeros(T, N, action_dim, device=device)
    buf_logp = torch.zeros(T, N, device=device)
    buf_rew = torch.zeros(T, N, device=device)
    buf_done = torch.zeros(T, N, device=device)
    buf_val = torch.zeros(T, N, device=device)
    next_done = torch.zeros(N, device=device)

    global_step, best_success = 0, -1.0
    n_updates = cfg["total_timesteps"] // (T * N)
    # rolling per-episode outcomes across >=4 batches of episodes; sr is nan
    # until enough episodes (incl. horizon-truncated failures) are pooled
    recent = deque(maxlen=max(1024, 4 * N))
    min_episodes = recent.maxlen // 8
    start = time.time()
    for update in range(1, n_updates + 1):
        for t in range(T):
            global_step += N
            buf_obs[t] = obs
            buf_done[t] = next_done
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(obs)
            buf_act[t], buf_logp[t], buf_val[t] = action, logp, value.view(-1)
            nobs, reward, term, trunc, infos = envs.step(action.cpu())
            obs = nobs.to(device)
            reward = torch.as_tensor(reward, device=device).view(-1)
            term = torch.as_tensor(term, device=device).view(-1)
            trunc = torch.as_tensor(trunc, device=device).view(-1)
            # bootstrap truncated (time-limit) episodes with V(final_obs)
            if trunc.any() and "final_observation" in infos:
                with torch.no_grad():
                    fv = agent.get_value(
                        torch.as_tensor(infos["final_observation"], device=device)
                    ).view(-1)
                reward = reward + cfg["gamma"] * fv * trunc.float()
            buf_rew[t] = reward
            next_done = (term | trunc).float()
            outcomes = _episode_stats(infos, device)
            if outcomes is not None:
                recent.extend(outcomes.tolist())

        with torch.no_grad():
            next_value = agent.get_value(obs).view(-1)
        adv, ret = compute_gae(buf_rew, buf_val, buf_done, next_value, next_done,
                               cfg["gamma"], cfg["gae_lambda"])

        b_obs = buf_obs.reshape(-1, state_dim)
        b_act = buf_act.reshape(-1, action_dim)
        b_logp, b_adv, b_ret = buf_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)

        def batch(mb):
            _, newlogp, ent, newval = agent.get_action_and_value(b_obs[mb], b_act[mb])
            return newlogp, ent, newval, b_logp[mb], b_adv[mb], b_ret[mb]

        stats = ppo_update(
            agent, optimizer, _BatchFn(batch, T * N, device),
            cfg["num_minibatches"], cfg["update_epochs"], cfg["clip_coef"],
            cfg["ent_coef"], cfg["vf_coef"], cfg["max_grad_norm"],
            target_kl=cfg.get("target_kl"),
        )

        sr = (sum(recent) / len(recent)
              if len(recent) >= min_episodes else float("nan"))
        if update % cfg.get("log_every", 1) == 0:
            sps = int(global_step / (time.time() - start))
            print(f"update {update}/{n_updates} step {global_step} "
                  f"success {sr:.2%} kl {stats['approx_kl']:.4f} sps {sps}")
            writer.add_scalar("charts/success_rate", sr, global_step)
            for k, v in stats.items():
                writer.add_scalar(f"losses/{k}", v, global_step)
        if len(recent) >= min_episodes and sr >= best_success:
            best_success = sr
            torch.save(dict(model=agent.state_dict(), state_dim=state_dim,
                            action_dim=action_dim, task=cfg["task"]),
                       out_dir / "best.pt")

    torch.save(dict(model=agent.state_dict(), state_dim=state_dim,
                    action_dim=action_dim, task=cfg["task"]), out_dir / "last.pt")
    envs.close()
    writer.close()
    print(f"expert done: best success {best_success:.2%} -> {out_dir}")
    return out_dir / "best.pt"


# --------------------------------------------------------------------------
# mode 2: sparse-reward fine-tuning of a BC policy (canonical actions)
# --------------------------------------------------------------------------
OBS_KEYS = ("rgb", "proprio", "proprio_mask", "tokens", "token_len", "embodiment_id")


def train_ppo_visual(cfg: dict) -> Path:
    device = torch.device(cfg.get("device") if cfg.get("device") not in (None, "auto")
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    tasks = load_yaml(repo_root() / "configs" / "tasks.yaml")["tasks"]
    task = tasks[cfg["task"]]
    adapter = get_adapter(task["embodiment"])
    out_dir = Path(cfg["out_dir"]).resolve()  # SAPIEN may chdir; stay absolute
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_run_config(cfg, out_dir)
    writer = SummaryWriter(out_dir / "tb")

    raw = make_env(task["env_id"], num_envs=cfg["num_envs"], obs_mode="rgb",
                   reward_mode="sparse", **(cfg.get("env_kwargs") or {}))
    vec = ManiSkillVectorEnv(raw, auto_reset=True, ignore_terminations=False)
    envs = UnifiedObsWrapper(
        vec, adapter, task["camera"], task["family"], static_slots=task.get("slots"),
        instruction_mode="train", paraphrase_p=cfg.get("paraphrase_p", 0.7),
        language_seed=cfg["seed"],
    )

    bc = BCPolicy(vocab_size=len(get_tokenizer()), vision=cfg.get("vision", "nature_cnn"))
    bc.load_state_dict(torch.load(cfg["bc_ckpt"], map_location="cpu", weights_only=True))
    agent = VisualActorCritic(bc, init_logstd=cfg.get("init_logstd", -2.0)).to(device)
    frozen = copy.deepcopy(agent.bc).to(device).eval()
    for p in frozen.parameters():
        p.requires_grad = False
    if cfg.get("freeze_vision", True):
        for p in agent.bc.vision.parameters():
            p.requires_grad = False
    optimizer = torch.optim.Adam(
        [p for p in agent.parameters() if p.requires_grad], lr=cfg["lr"], eps=1e-5
    )

    N, T = cfg["num_envs"], cfg["num_steps"]
    amask = adapter.action_mask.to(device).expand(N, -1)

    obs, _ = envs.reset(seed=cfg["seed"])
    obs = {k: obs[k].to(device) for k in OBS_KEYS}
    buf = {k: torch.zeros(T, *obs[k].shape, dtype=obs[k].dtype, device=device)
           for k in OBS_KEYS}
    buf_act = torch.zeros(T, N, A_CANON, device=device)
    buf_logp = torch.zeros(T, N, device=device)
    buf_rew = torch.zeros(T, N, device=device)
    buf_done = torch.zeros(T, N, device=device)
    buf_val = torch.zeros(T, N, device=device)
    next_done = torch.zeros(N, device=device)

    global_step, best_success = 0, -1.0
    n_updates = cfg["total_timesteps"] // (T * N)
    recent = deque(maxlen=max(1024, 4 * N))
    min_episodes = recent.maxlen // 8
    mb_mask = adapter.action_mask.to(device)
    for update in range(1, n_updates + 1):
        for t in range(T):
            global_step += N
            for k in OBS_KEYS:
                buf[k][t] = obs[k]
            buf_done[t] = next_done
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(obs, amask)
            buf_act[t], buf_logp[t], buf_val[t] = action, logp, value.view(-1)
            native = adapter.to_native_action(action).cpu()
            nobs, reward, term, trunc, infos = envs.step(native)
            obs = {k: nobs[k].to(device) for k in OBS_KEYS}
            reward = torch.as_tensor(reward, device=device).view(-1).float()
            reward = reward - cfg.get("step_penalty", 0.01)
            term = torch.as_tensor(term, device=device).view(-1)
            trunc = torch.as_tensor(trunc, device=device).view(-1)
            buf_rew[t] = reward
            next_done = (term | trunc).float()
            outcomes = _episode_stats(infos, device)
            if outcomes is not None:
                recent.extend(outcomes.tolist())

        with torch.no_grad():
            next_value = agent.get_value(obs).view(-1)
        adv, ret = compute_gae(buf_rew, buf_val, buf_done, next_value, next_done,
                               cfg["gamma"], cfg["gae_lambda"])

        flat = {k: buf[k].reshape(-1, *buf[k].shape[2:]) for k in OBS_KEYS}
        b_act = buf_act.reshape(-1, A_CANON)
        b_logp, b_adv, b_ret = buf_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        B = T * N

        def batch(mb):
            mb_obs = {k: flat[k][mb] for k in OBS_KEYS}
            _, newlogp, ent, newval = agent.get_action_and_value(
                mb_obs, mb_mask.expand(len(mb), -1), b_act[mb]
            )
            return newlogp, ent, newval, b_logp[mb], b_adv[mb], b_ret[mb]

        def bc_reg(mb):
            mb_obs = {k: flat[k][mb] for k in OBS_KEYS}
            with torch.no_grad():
                ref = frozen(mb_obs)
            cur = torch.tanh(agent.bc.head(agent.bc.encode(mb_obs)))
            return cfg.get("bc_reg_coef", 1.0) * (
                ((cur - ref) ** 2 * mb_mask).sum(-1) / mb_mask.sum()
            ).mean()

        stats = ppo_update(
            agent, optimizer, _BatchFn(batch, B, device),
            cfg["num_minibatches"], cfg["update_epochs"], cfg["clip_coef"],
            cfg["ent_coef"], cfg["vf_coef"], cfg["max_grad_norm"],
            extra_loss_fn=bc_reg, target_kl=cfg.get("target_kl"),
        )

        sr = (sum(recent) / len(recent)
              if len(recent) >= min_episodes else float("nan"))
        print(f"update {update}/{n_updates} step {global_step} success {sr:.2%} "
              f"kl {stats['approx_kl']:.4f} bc_reg {stats.get('bc_reg', 0):.4f}")
        writer.add_scalar("charts/success_rate", sr, global_step)
        for k, v in stats.items():
            writer.add_scalar(f"losses/{k}", v, global_step)
        if len(recent) >= min_episodes and sr >= best_success:
            best_success = sr
            torch.save(agent.bc.state_dict(), out_dir / "best.pt")

    torch.save(agent.bc.state_dict(), out_dir / "last.pt")
    envs.close()
    writer.close()
    print(f"finetune done: best success {best_success:.2%} -> {out_dir}")
    return out_dir / "best.pt"
