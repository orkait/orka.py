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
    ap.add_argument("--max-seqs", type=int, default=256)
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
    student = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, dtype=torch.float32
    ).to(dev)
    student.config.use_cache = False
    try:
        student.gradient_checkpointing_enable()
    except Exception:
        pass
    wrapped = build_qat_student(student, allocation, group_size=args.group_size,
                                commitment=args.commit)
    print(f"  wrapped {len(wrapped)} linears", flush=True)

    # Train only the quantized layers' shadow weights + codebooks.
    train_params = []
    for qat in wrapped.values():
        train_params.append(qat.shadow)
        for cb in qat.codebooks:
            train_params.append(cb)
        if qat.bias is not None:
            train_params.append(qat.bias)
    for p in student.parameters():
        p.requires_grad_(False)
    for p in train_params:
        p.requires_grad_(True)
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
    for step in range(args.steps):
        idx = torch.randint(0, corpus.shape[0], (args.batch,), generator=rng).to(dev)
        batch = corpus[idx]
        with torch.no_grad():
            t_logits = teacher(batch).logits.float()
        s_logits = student(batch).logits

        T = args.temp
        # flatten tokens; KL per token averaged
        sl = s_logits.reshape(-1, s_logits.shape[-1])
        tl = t_logits.reshape(-1, t_logits.shape[-1])
        kl = F.kl_div(
            F.log_softmax(sl / T, dim=-1),
            F.softmax(tl / T, dim=-1),
            reduction="batchmean", log_target=False,
        ) * (T * T)
        del t_logits, tl
        cb_loss = collect_codebook_loss(wrapped)
        loss = kl + args.cb_weight * cb_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        sched.step()

        if step % 20 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  kl={kl.item():.4f}  cb={cb_loss.item():.4f}  lr={sched.get_last_lr()[0]:.2e}", flush=True)

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
