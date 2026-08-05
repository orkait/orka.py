# Vendored research sources

Third-party code kept here for study, not built or imported by orka.

| dir | upstream | commit | fetched | license |
|---|---|---|---|---|
| `unsloth/` | https://github.com/unslothai/unsloth | `949b3466e03b6089e3e17ca91121e060025bf96b` | 2026-08-05 04:07:35 -0700 | Apache-2.0 (LICENSE retained in-tree) |

`.git` was removed at the user's request. The commit above is the provenance record;
LICENSE and COPYING are preserved unmodified inside the vendored tree.

## Why it is here

To understand how llama.cpp GGUF quants beat orka RVQ on the same model
(measured: Q4_K_M 1.674 GB / PPL 9.8191 vs orka 1.739 GB / PPL 10.9784).
