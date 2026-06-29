"""CPU-offloaded AdamW for VRAM-bound QAT.

Keeps the optimizer moments (m, v) in host RAM and runs the update on the CPU,
so none of the optimizer state lives on the GPU. Only the gradient (moved to
CPU) and the resulting parameter delta (moved back) cross the bus each step.

Why not ``bnb.optim.PagedAdamW8bit``: paged state uses CUDA unified memory,
which still counts against the GPU allocator (~2.6GB resident for a 1.5B model)
and tips a 10GB-capped card over at the optimizer step. A true CPU optimizer
removes that, letting full-model 1.5B QAT fit under the cap. Costs ~5GB of
PCIe traffic/step (grad out + delta in) - a few minutes over a 300-step run.

State is fp32 even when params are bf16 (standard mixed-precision: bf16 master
+ fp32 moments) and lives on the CPU, so it does not consume GPU memory.
"""
from __future__ import annotations

import torch


class CPUOffloadAdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        # m, v held on CPU in fp32, keyed by parameter identity.
        self._state: dict[int, dict[str, torch.Tensor]] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.detach().to("cpu", dtype=torch.float32)
            st = self._state.get(id(p))
            if st is None:
                st = {"m": torch.zeros_like(g), "v": torch.zeros_like(g),
                      "p32": p.detach().to("cpu", dtype=torch.float32)}
                self._state[id(p)] = st
            m, v, p32 = st["m"], st["v"], st["p32"]
            m.mul_(self.b1).add_(g, alpha=1.0 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1.0 - self.b2)
            denom = (v / bc2).sqrt_().add_(self.eps)
            # decoupled weight decay (AdamW) on the fp32 master
            if self.wd:
                p32.mul_(1.0 - self.lr * self.wd)
            p32.addcdiv_(m / bc1, denom, value=-self.lr)
            p.data.copy_(p32.to(p.device, dtype=p.dtype))

    def state_dict(self) -> dict:
        return {"t": self.t, "state": self._state}

    def load_state_dict(self, sd: dict) -> None:
        self.t = sd["t"]
        self._state = sd["state"]
