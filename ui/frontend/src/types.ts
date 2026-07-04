// Mirrors ui/backend/schema.py (the journey contract). Keep in sync.
export type Source = "estimated" | "measured";
export type Treatment = "quantize" | "keep_fp16" | "skip_error_comp";

export interface ModelMeta {
  name: string;
  params_total: number;
  dtype: string;
  vocab_size: number | null;
  tie_word_embeddings: boolean;
  fp16_bytes: number;
}
export interface FamilyBreakdown { family: string; params: number; pct: number; role: string; }
export interface ModuleEntry { name: string; shape: number[]; family: string; treatment: Treatment; }
export interface LayerBlock { index: number; modules: ModuleEntry[]; }
export interface Architecture {
  arch_class: string;
  flags: Record<string, boolean>;
  param_breakdown: FamilyBreakdown[];
  layers: LayerBlock[];
  partial: boolean;
}
export interface Stage { id: string; title: string; summary: string; }
export interface Trick {
  id: string; label: string; kind: "scalar" | "toggle";
  default: number | boolean; applies: boolean;
  why: string; warn: string | null; gated_by: string | null;
}
export interface Result {
  source: Source; bpw: number; ratio: number; fp16_mb: number; orka_mb: number;
  ppl_base: number | null; ppl_orka: number | null; ppl_ratio: number | null;
  trusted: boolean | null; trust_reason: string | null; notes: string[];
}
export interface Journey {
  schema_version: number;
  model: ModelMeta;
  architecture: Architecture;
  pipeline: Stage[];
  tricks: Trick[];
  result: Result;
}

export interface RDPoint { bpw: number; sqnr: number; }
export interface TensorProbe {
  name: string;
  shape: number[];
  dtype: string;
  sampled_elems: number;
  std: number; mean: number; vmin: number; vmax: number; outlier_pct: number;
  distribution: number[];
  dist_range: number[];
  codebook_values: number[];
  rd: RDPoint[];
  utilization: number[];
  entropy_bits: number; entropy_max: number;
  weights_block: number[][];
  error_block: number[][];
  sqnr_3bpw: number; error_pct: number;
  vectors3d: number[][];
  centroids3d: number[][];
}
