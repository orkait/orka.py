"""VQ-QAT training driver (prototype). Fine-tunes a student whose allocated
linears are VQ-quantized on forward, distilling from a frozen fp16 teacher.

Usage:
  python -m orka.qat_train MODEL_DIR ALLOC_JSON CORPUS.txt OUT_DIR \
      --steps 300 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_corpus(tokenizer, path: Path, seq_len: int, max_seqs: int):
    text = path.read_text()
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n = (ids.shape[0] // seq_len)
    n = min(n, max_seqs)
    return ids[: n * seq_len].reshape(n, seq_len)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("allocation")
    ap.add_argument("corpus")
    ap.add_argument("out_dir")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--commit", type=float, default=0.25)
    ap.add_argument("--cb-weight", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--reassign-every", type=int, default=1,
                    help="re-run the hard VQ argmin every N optimizer steps (cache idx "
                         "between). 1 = exact (only skips redundant re-search within a "
                         "grad-accum window); >1 trades ~0.3%%/step assignment staleness "
                         "for ~Nx fewer argmin passes - the QAT forward is argmin-bound.")
    ap.add_argument("--max-seqs", type=int, default=256)
    ap.add_argument("--optim8bit", action="store_true",
                    help="use bitsandbytes AdamW8bit (m+v in int8) to fit small GPUs")
    ap.add_argument("--student-bf16", action="store_true",
                    help="load student backbone in bf16 (shadow weights + codebooks "
                         "stay fp32); frozen layers are bf16 -> saves ~1GB on small GPUs")
    ap.add_argument("--checkpoint-quantize", action="store_true",
                    help="gradient-checkpoint each layer's quantize() - frees the "
                         "~8GB of weight-sized straight-through intermediates "
                         "(recomputed in backward); bit-identical training")
    ap.add_argument("--ckpt-dir", default="",
                    help="directory to write/read a resumable training checkpoint "
                         "(shadow + codebooks + optimizer + step). Empty = disabled.")
    ap.add_argument("--ckpt-every", type=int, default=25,
                    help="save a resume checkpoint every N optimizer steps")
    ap.add_argument("--resume", action="store_true",
                    help="resume from --ckpt-dir/qat_ckpt.pt if it exists")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="accumulate this many micro-batches per optimizer step "
                         "(recovers effective batch size at low VRAM)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from orka.qat import build_qat_student, collect_codebook_loss

    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    allocation = json.loads(Path(args.allocation).read_text())

    print("loading teacher (frozen bf16)...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.bfloat16
    ).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("loading student + wrapping allocated linears...", flush=True)
    student_dtype = torch.bfloat16 if args.student_bf16 else torch.float32
    student = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, dtype=student_dtype
    ).to(dev)
    student.config.use_cache = False
    # Model-level checkpointing would nest inside the per-quantize checkpoint
    # (and clobber its cb_loss output tracking), so use one or the other.
    if not args.checkpoint_quantize:
        try:
            student.gradient_checkpointing_enable()
        except Exception:
            pass
    wrapped = build_qat_student(student, allocation, group_size=args.group_size,
                                commitment=args.commit, checkpoint=args.checkpoint_quantize,
                                reassign_every=args.reassign_every)
    print(f"  wrapped {len(wrapped)} linears", flush=True)

    # Train only the quantized layers' shadow weights + codebooks.
    train_params = []
    for qat in wrapped.values():
        train_params.append(qat.shadow)
        train_params.append(qat.scales)
        for cb in qat.codebooks:
            train_params.append(cb)
        if qat.bias is not None:
            train_params.append(qat.bias)
    for p in student.parameters():
        p.requires_grad_(False)
    for p in train_params:
        p.requires_grad_(True)
    if args.optim8bit:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(train_params, lr=args.lr)
        print("optimizer: bitsandbytes AdamW8bit (int8 m+v)", flush=True)
    else:
        opt = torch.optim.AdamW(train_params, lr=args.lr)

    # Warmup + cosine decay - longer runs converge better with a schedule.
    warmup = max(10, args.steps // 20)

    def lr_at(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    corpus = _load_corpus(tok, Path(args.corpus), args.seq_len, args.max_seqs).to(dev)
    print(f"corpus: {corpus.shape[0]} sequences x {args.seq_len} tokens", flush=True)

    student.train()
    import time
    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)
    accum = max(1, args.grad_accum)

    # --- resume checkpoint (shadow + codebooks + optimizer + step + RNG) ---
    # The trainable state lives in student.state_dict() (QATVQLinear shadow +
    # codebooks are nn.Parameters) plus the optimizer state. Saving all of it
    # every --ckpt-every steps lets a killed run continue instead of restart.
    ckpt_path = Path(args.ckpt_dir) / "qat_ckpt.pt" if args.ckpt_dir else None
    start_step = 0
    if ckpt_path and args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev)
        student.load_state_dict(ck["student"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        rng.set_state(ck["rng"].cpu().to(torch.uint8))  # set_state needs a CPU ByteTensor
        start_step = ck["step"]
        print(f"resumed from {ckpt_path} at step {start_step}", flush=True)

    def _save_ckpt(step):
        if not ckpt_path:
            return
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"student": student.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "rng": rng.get_state(), "step": step}, tmp)
        tmp.replace(ckpt_path)   # atomic - a kill mid-write never corrupts the ckpt

    for step in range(start_step, args.steps):
        # Gradient accumulation: sum grads over `accum` micro-batches, then one
        # optimizer step. Recovers a larger effective batch (accum * batch) on a
        # small GPU where only batch=1 fits in VRAM.
        opt.zero_grad(set_to_none=True)
        # Accumulate the logged losses as GPU tensors and sync (.item()) ONLY on a
        # log step. A per-step .item() forces a GPU->CPU sync every micro-batch (19 of
        # every 20 wasted, since they are never printed), serializing the device and
        # blocking the CPU from queuing the next step's kernels.
        log_step = step % 20 == 0 or step == args.steps - 1
        kl_acc = torch.zeros((), device=dev)
        cb_acc = torch.zeros((), device=dev)
        for micro in range(accum):
            idx = torch.randint(0, corpus.shape[0], (args.batch,), generator=rng).to(dev)
            batch = corpus[idx]
            with torch.no_grad():
                t_logits = teacher(batch).logits.float()
            s_logits = student(batch).logits

            T = args.temp
            sl = s_logits.reshape(-1, s_logits.shape[-1])
            tl = t_logits.reshape(-1, t_logits.shape[-1])
            kl = F.kl_div(
                F.log_softmax(sl / T, dim=-1),
                F.softmax(tl / T, dim=-1),
                reduction="batchmean", log_target=False,
            ) * (T * T)
            del t_logits, tl
            cb_loss = collect_codebook_loss(wrapped)
            loss = (kl + args.cb_weight * cb_loss) / accum
            loss.backward()
            if log_step:
                kl_acc = kl_acc + kl.detach()
                cb_acc = cb_acc + cb_loss.detach()

        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        sched.step()

        if log_step:
            print(f"step {step:4d}  kl={kl_acc.item() / accum:.4f}  cb={cb_acc.item() / accum:.4f}  lr={sched.get_last_lr()[0]:.2e}", flush=True)
        # checkpoint AFTER the step so resume restarts at the next unfinished step
        if ckpt_path and ((step + 1) % args.ckpt_every == 0 or step == args.steps - 1):
            _save_ckpt(step + 1)

    print("materializing quantized weights into a dense HF dir...", flush=True)
    student.eval()
    with torch.no_grad():
        for name, qat in wrapped.items():
            w = qat.materialized_weight().to(torch.float32)
            parent = student.get_submodule(name.rsplit(".", 1)[0])
            import torch.nn as nn
            lin = nn.Linear(qat.in_features, qat.out_features,
                            bias=qat.bias is not None).to(dev)
            lin.weight.data.copy_(w)
            if qat.bias is not None:
                lin.bias.data.copy_(qat.bias.data)
            setattr(parent, name.rsplit(".", 1)[-1], lin)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    student.to(torch.bfloat16).save_pretrained(str(out))
    tok.save_pretrained(str(out))
    # copy non-weight sidecars (config already saved by save_pretrained)
    print(f"saved QAT student -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
