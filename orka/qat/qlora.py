"""QLoRA-style quality recovery for a quantized model (the industry approach).

Instead of full-model fake-quant QAT (fp32 shadow + full grads + teacher = ~16GB
on a 1.5B model, the memory-heaviest path), this freezes the already-quantized
base weights and trains small low-rank LoRA adapters on top, distilling from the
fp16 teacher. Trainable params are a few million (adapters), not billions, so it
fits a 10GB cap with room to spare and runs fast.

  frozen quantized W_q  +  B @ A * (alpha/r)      (B init 0 -> starts == base)

The adapters recover the output error the quantizer introduced. At eval the
adapter delta is merged into the weight (W_q + BA*scaling); the cost is a small
bpw bump (~0.3 bpw at rank 16 on 1536-dim linears), reported in the manifest.

Usage:
  python -m orka.qat.qlora MODEL_DIR QUANTIZED_SAFETENSORS CORPUS.txt OUT_DIR \
      --steps 300 --rank 16 --device cuda
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLoRALinear(nn.Module):
    """Frozen quantized linear + trainable low-rank adapter. B is zero-init so the
    initial delta is exactly zero -> training starts at the quantized base."""

    def __init__(self, w_q: torch.Tensor, bias: torch.Tensor | None, rank: int, alpha: float):
        super().__init__()
        self.out_features, self.in_features = w_q.shape
        self.register_buffer("weight", w_q)                     # frozen quantized base
        self.register_buffer("bias", bias if bias is not None else None)
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=w_q.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=w_q.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        delta = (x @ self.lora_A.t().to(x.dtype)) @ self.lora_B.t().to(x.dtype)
        return base + delta * self.scaling

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        return self.weight.float() + (self.lora_B @ self.lora_A) * self.scaling


def build_qlora_student(model: nn.Module, quantized_sd: dict, rank: int, alpha: float) -> dict:
    """Replace each attn/mlp Linear with a QLoRALinear whose frozen base is the
    quantized weight from ``quantized_sd``. Returns {module_name: QLoRALinear}."""
    wrapped = {}
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not ("self_attn" in full_name or "mlp" in full_name):
            continue
        wname = full_name + ".weight"
        if wname not in quantized_sd:
            continue
        w_q = quantized_sd[wname].to(module.weight.device, dtype=module.weight.dtype)
        bias = module.bias.data if module.bias is not None else None
        ql = QLoRALinear(w_q, bias, rank, alpha)
        parent = model.get_submodule(full_name.rsplit(".", 1)[0])
        setattr(parent, full_name.rsplit(".", 1)[-1], ql)
        wrapped[full_name] = ql
    return wrapped


def _load_corpus(tokenizer, path: Path, seq_len: int, max_seqs: int):
    ids = tokenizer(path.read_text(), return_tensors="pt").input_ids[0]
    n = min(ids.shape[0] // seq_len, max_seqs)
    return ids[: n * seq_len].reshape(n, seq_len)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("quantized", help="dense safetensors of the quantized base weights")
    ap.add_argument("corpus")
    ap.add_argument("out_dir")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--max-seqs", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)

    print("loading quantized base weights...", flush=True)
    quantized_sd = load_file(args.quantized)

    print("loading student (frozen quantized base + LoRA adapters)...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, dtype=torch.bfloat16).to(dev)
    student.config.use_cache = False
    wrapped = build_qlora_student(student, quantized_sd, args.rank, args.alpha)
    print(f"  wrapped {len(wrapped)} linears, rank={args.rank}", flush=True)
    del quantized_sd

    train_params = []
    for ql in wrapped.values():
        train_params += [ql.lora_A, ql.lora_B]
    for p in student.parameters():
        p.requires_grad_(False)
    for p in train_params:
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in train_params)
    print(f"  trainable adapter params: {n_train/1e6:.2f}M", flush=True)

    print("loading teacher (frozen bf16)...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, dtype=torch.bfloat16).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(train_params, lr=args.lr)
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
    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)
    accum = max(1, args.grad_accum)
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        log_step = step % 20 == 0 or step == args.steps - 1
        kl_acc = torch.zeros((), device=dev)
        for _ in range(accum):
            idx = torch.randint(0, corpus.shape[0], (args.batch,), generator=rng).to(dev)
            batch = corpus[idx]
            with torch.no_grad():
                t_logits = teacher(batch).logits.float()
            s_logits = student(batch).logits
            T = args.temp
            kl = F.kl_div(
                F.log_softmax(s_logits.reshape(-1, s_logits.shape[-1]) / T, dim=-1),
                F.softmax(t_logits.reshape(-1, t_logits.shape[-1]) / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)
            del t_logits
            (kl / accum).backward()
            if log_step:
                kl_acc = kl_acc + kl.detach()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        sched.step()
        if log_step:
            print(f"step {step:4d}  kl={kl_acc.item()/accum:.4f}  lr={sched.get_last_lr()[0]:.2e}", flush=True)

    print("merging adapters into the weights, materializing dense HF dir...", flush=True)
    del teacher  # teacher is done; free ~3GB before the fp32 merge materialization
    torch.cuda.empty_cache()
    student.eval()
    with torch.no_grad():
        for name, ql in wrapped.items():
            w = ql.merged_weight().to(torch.float32)
            parent = student.get_submodule(name.rsplit(".", 1)[0])
            lin = nn.Linear(ql.in_features, ql.out_features, bias=ql.bias is not None).to(dev)
            lin.weight.data.copy_(w)
            if ql.bias is not None:
                lin.bias.data.copy_(ql.bias)
            setattr(parent, name.rsplit(".", 1)[-1], lin)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    student.to(torch.bfloat16).save_pretrained(str(out))
    tok.save_pretrained(str(out))
    print(f"saved QLoRA-recovered model -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
