"""Behavior cloning trainer: masked SmoothL1 on canonical actions.

Step-based (not epoch-based) so tiny smoke configs and full runs share one
code path. Checkpoints: best.pt (lowest val loss) and last.pt; TensorBoard
scalars under <out_dir>/tb.
"""

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from xembench.data.dataset import DemoDataset, split_by_episode
from xembench.language.tokenizer import get_tokenizer
from xembench.models.bc_policy import BCPolicy, masked_bc_loss
from xembench.utils.config import dump_run_config
from xembench.utils.torch_utils import auto_device, count_params, to_device


def random_shift(rgb: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ-style random translation for uint8 [B,3,H,W] images."""
    B, C, H, W = rgb.shape
    padded = torch.nn.functional.pad(rgb.float(), (pad,) * 4, mode="replicate")
    dx = torch.randint(0, 2 * pad + 1, (B,), device=rgb.device)
    dy = torch.randint(0, 2 * pad + 1, (B,), device=rgb.device)
    out = torch.empty_like(rgb)
    for i in range(B):  # small B-loop, negligible next to the conv forward
        out[i] = padded[i, :, dy[i] : dy[i] + H, dx[i] : dx[i] + W].to(rgb.dtype)
    return out


def _cycle(loader: DataLoader):
    while True:
        yield from loader


@torch.no_grad()
def _validate(model: BCPolicy, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        pred = model(batch)
        total += masked_bc_loss(pred, batch["action"], batch["action_mask"]).item()
        n += 1
    model.train()
    return total / max(n, 1)


def train_bc(cfg: dict) -> dict:
    device = auto_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["out_dir"]).resolve()  # SAPIEN may chdir; stay absolute
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_run_config(cfg, out_dir)
    writer = SummaryWriter(out_dir / "tb")

    dataset = DemoDataset(
        cfg["demos"],
        instruction_mode="train" if cfg.get("paraphrase_augment", True) else "canonical",
        paraphrase_p=cfg.get("paraphrase_p", 0.7),
        preload=cfg.get("preload", False),
        chunk=cfg.get("chunk", 1),
    )
    train_set, val_set = split_by_episode(dataset, cfg.get("val_fraction", 0.1),
                                          seed=cfg["seed"])
    loader_kwargs = dict(
        batch_size=cfg["batch_size"], num_workers=cfg.get("num_workers", 0),
        drop_last=False, persistent_workers=cfg.get("num_workers", 0) > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    model = BCPolicy(
        vocab_size=len(get_tokenizer()), vision=cfg.get("vision", "nature_cnn"),
        chunk=cfg.get("chunk", 1),
    ).to(device)
    print(f"BCPolicy[{cfg.get('vision', 'nature_cnn')}] params: {count_params(model):,} "
          f"| train {len(train_set)} val {len(val_set)} transitions | device {device}")

    steps = cfg["steps"]
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps))
    )

    best_val = float("inf")
    batches = _cycle(train_loader)
    for step in range(1, steps + 1):
        batch = to_device(next(batches), device)
        if cfg.get("augment", True):
            batch["rgb"] = random_shift(batch["rgb"])
        loss = masked_bc_loss(model(batch), batch["action"], batch["action_mask"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % cfg.get("log_every", 50) == 0:
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
        if step % cfg.get("eval_every", 500) == 0 or step == steps:
            val_loss = _validate(model, val_loader, device)
            writer.add_scalar("val/loss", val_loss, step)
            print(f"step {step:6d} | train {loss.item():.4f} | val {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), out_dir / "best.pt")

    torch.save(model.state_dict(), out_dir / "last.pt")
    writer.close()
    return {"best_val_loss": best_val, "final_train_loss": loss.item()}
