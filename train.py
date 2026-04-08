
import os
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from basics.utils import Linear, Embedding, RMSNorm, TransformerBlock, save_checkpoint, load_checkpoint
from basics.loss import cross_entropy
from basics.optimizer import AdamW, cosine_schedule
from basics.model import TransformerLM

# ─────────────────────────────────────────────────────────────────────────────
# 2.  数据加载
# ─────────────────────────────────────────────────────────────────────────────

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x = torch.stack([
        torch.from_numpy(dataset[i     : i + context_length    ].astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(dataset[i + 1 : i + context_length + 1].astype(np.int64))
        for i in ix
    ])
    return x.to(device), y.to(device)



# ─────────────────────────────────────────────────────────────────────────────
# 3.  验证 loss 估算
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_val_loss(
    model: TransformerLM,
    val_data: np.ndarray,
    batch_size: int,
    context_length: int,
    val_steps: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(val_steps):
        x, y = get_batch(val_data, batch_size, context_length, device)
        logits = model(x)                              # (B, T, V)
        loss   = cross_entropy(logits, y)              # cross_entropy 支持 (..., V) 输入
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  画 loss 曲线
# ─────────────────────────────────────────────────────────────────────────────

def plot_and_save(log: dict, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")          # 无 GUI 环境也能保存
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # 左图：loss vs steps
        ax = axes[0]
        ax.plot(log["train_step"], log["train_loss"], label="train", alpha=0.7, linewidth=1)
        if log["val_step"]:
            ax.plot(log["val_step"], log["val_loss"],
                    label="val", marker="o", linewidth=2, markersize=4)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Loss vs Steps")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 右图：loss vs wall-clock time（秒）
        ax = axes[1]
        ax.plot(log["train_time"], log["train_loss"], label="train", alpha=0.7, linewidth=1)
        if log["val_time"]:
            ax.plot(log["val_time"], log["val_loss"],
                    label="val", marker="o", linewidth=2, markersize=4)
        ax.set_xlabel("Wall-clock time (s)")
        ax.set_ylabel("Loss")
        ax.set_title("Loss vs Wall-Clock Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(out_dir, "loss_curves.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[plot] saved → {path}")
    except ImportError:
        print("[plot] matplotlib not found, skipping")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  命令行参数
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train a Transformer LM")

    # 数据 & 输出
    p.add_argument("--train_path", required=True, help="Path to tokenized train .npy")
    p.add_argument("--val_path", required=True, help="Path to tokenized val .npy")
    p.add_argument("--out_dir", default="runs", help="Directory for checkpoints & logs")

    # 模型超参
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)

    # 训练超参
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--total_steps", type=int, default=20000,
                   help="Total training steps. "
                        "With batch=64, ctx=256 this ≈ 327M tokens.")
    p.add_argument("--lr_max", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int,   default=1000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # 日志 & 保存
    p.add_argument("--log_interval",  type=int, default=100,
                   help="Print train loss every N steps")
    p.add_argument("--val_interval",  type=int, default=500,
                   help="Compute val loss every N steps")
    p.add_argument("--save_interval", type=int, default=2000,
                   help="Save checkpoint every N steps")
    p.add_argument("--val_steps",     type=int, default=20,
                   help="Number of batches used to estimate val loss")

    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  主训练循环
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── 加载数据（memmap 节省内存）────────────────────────────────────────────
    train_data = np.load(args.train_path, mmap_mode="r")
    val_data = np.load(args.val_path, mmap_mode="r")
    print(f"Train tokens : {len(train_data):,}")
    print(f"Val tokens : {len(val_data):,}")

    # ── 模型 ──────────────────────────────────────────────────────────────────
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters   : {n_params:,}  ({n_params/1e6:.1f}M)")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # ── 可选：从 checkpoint 恢复 ──────────────────────────────────────────────
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from step {start_step}")

    # ── 日志容器 ──────────────────────────────────────────────────────────────
    log = {
        "train_step": [], "train_loss": [], "train_time": [],
        "val_step":   [], "val_loss":   [], "val_time":   [],
    }

    # ── 训练 ──────────────────────────────────────────────────────────────────
    model.train()
    t_start = time.time()
    running_loss = 0.0

    print(f"\nStarting training on {device} for {args.total_steps} steps …\n")

    for step in range(start_step, args.total_steps):

        # 1. 计算当前 lr 并写入 optimizer
        lr = cosine_schedule(
            step,
            max_lr=args.lr_max,
            min_lr=args.lr_min,
            tw=args.warmup_steps,
            tc=args.total_steps,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # 2. 采样 batch
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)

        # 3. Forward + loss
        logits = model(x)                    # (B, T, V)
        loss  = cross_entropy(logits, y)    # scalar

        # 4. Backward
        optimizer.zero_grad()
        loss.backward()

        # 5. 梯度裁剪（用你 utils 里的函数）
        # _manual_clip_grad_norm(model.parameters(), args.grad_clip)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # 6. 参数更新
        optimizer.step()

        running_loss += loss.item()

        # ── 打印 train loss ───────────────────────────────────────────────────
        if (step + 1) % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            elapsed  = time.time() - t_start
            tokens   = (step + 1) * args.batch_size * args.context_length

            log["train_step"].append(step + 1)
            log["train_loss"].append(avg_loss)
            log["train_time"].append(elapsed)

            print(
                f"step {step+1:6d}/{args.total_steps}"
                f"  loss {avg_loss:.4f}"
                f"  lr {lr:.2e}"
                f"  tokens {tokens/1e6:.1f}M"
                f"  elapsed {elapsed/60:.1f}min"
            )
            running_loss = 0.0

        # ── 验证 loss ─────────────────────────────────────────────────────────
        if (step + 1) % args.val_interval == 0:
            val_loss = estimate_val_loss(
                model, val_data,
                args.batch_size, args.context_length,
                args.val_steps, device,
            )
            elapsed = time.time() - t_start
            log["val_step"].append(step + 1)
            log["val_loss"].append(val_loss)
            log["val_time"].append(elapsed)
            print(f"  ▶ val_loss {val_loss:.4f}")

        # ── 保存 checkpoint ───────────────────────────────────────────────────
        if (step + 1) % args.save_interval == 0:
            ckpt = os.path.join(args.out_dir, f"step_{step+1:06d}.pt")
            save_checkpoint(model, optimizer, step + 1, ckpt)
            print(f"  ✓ checkpoint → {ckpt}")

    # ── 训练结束：保存最终 checkpoint + 画图 ──────────────────────────────────
    final_ckpt = os.path.join(args.out_dir, "final.pt")
    save_checkpoint(model, optimizer, args.total_steps, final_ckpt)
    print(f"\nTraining done. Final checkpoint → {final_ckpt}")

    plot_and_save(log, args.out_dir)


if __name__ == "__main__":
    main()