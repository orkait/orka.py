"""Semantic Reconstruction Engine: Ingestion, Trie Discovery, and Clustering."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

def build_morphological_trie(token_strings: list[tuple[str, int]]):
    """Phase 2: Unsupervised discovery of linguistic roots."""
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.token_id = None
            self.count = 0

    root = TrieNode()
    for s, tid in token_strings:
        if not isinstance(s, str): continue
        node = root
        for char in s:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        node.token_id = tid
    
    roots = []
    def find_productive_nodes(node, current_str, depth):
        # A productive node is one that has many descendants (it is a root)
        if node.count >= 10 and depth >= 3:
            roots.append({
                "root": current_str,
                "count": node.count,
                "depth": depth
            })
        for char, child in node.children.items():
            find_productive_nodes(child, current_str + char, depth + 1)
            
    find_productive_nodes(root, '', 0)
    return sorted(roots, key=lambda x: x["count"], reverse=True)


def find_semantic_hubs(embeddings: np.ndarray, threshold: float = 0.999):
    """Phase 3: Finding semantic neighborhoods (Concept Merging hubs)."""
    import torch
    
    # Move to GPU if possible for massive matrix math
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Clustering {embeddings.shape[0]} vectors on {device}...", flush=True)
    
    t_emb = torch.from_numpy(embeddings).to(device)
    # Normalize for cosine similarity
    t_emb = torch.nn.functional.normalize(t_emb, p=2, dim=1)
    
    # We will find the most unique concepts first
    # This is a simplified Concept Merging discovery
    hubs = []
    processed = torch.zeros(t_emb.shape[0], dtype=torch.bool, device=device)
    
    # Scan ranges where redundancy is common
    # 1. First 2500 tokens (structural/linguistic)
    # 2. Last 2500 tokens (often padding/unused/special)
    scan_ranges = [
        range(min(2500, t_emb.shape[0])),
        range(max(0, t_emb.shape[0] - 2500), t_emb.shape[0])
    ]
    
    for r in scan_ranges:
        for i in r:
            if processed[i]: continue
            
            query = t_emb[i:i+1]
            sims = torch.mm(query, t_emb.T).squeeze(0)
            
            matches = torch.where(sims > threshold)[0]
            if len(matches) > 1:
                hubs.append({
                    "master_tid": int(i),
                    "member_count": int(len(matches)),
                    "member_tids": matches.cpu().tolist(),
                    "avg_similarity": float(sims[matches].mean().item())
                })
                processed[matches] = True
            
    return sorted(hubs, key=lambda x: x["member_count"], reverse=True)


def profile_architecture(model_dir: Path) -> dict:
    """Phase 0: Generic architectural profiling."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return {"type": "unknown"}
    
    with open(config_path) as f:
        cfg = json.load(f)
    
    m_type = cfg.get("model_type", "unknown")
    num_layers = cfg.get("num_hidden_layers", 0)
    hidden_size = cfg.get("hidden_size", 0)
    
    # Detect MoE
    num_experts = cfg.get("num_experts", cfg.get("n_routed_experts", 0))
    is_moe = num_experts > 0
    
    return {
        "architecture": m_type,
        "is_moe": is_moe,
        "experts": num_experts,
        "layers": num_layers,
        "hidden_dim": hidden_size,
        "params_estimate_millions": (cfg.get("num_parameters", 0)) // 1_000_000
    }


def cmd_sem_analyze(args: argparse.Namespace) -> int:
    """Entry point for orka sem-analyze (Phases 0-3)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    model_dir = Path(args.model_dir)
    
    print(f"--- Phase 0: Architectural Profiling ---", flush=True)
    profile = profile_architecture(model_dir)
    print(f"  Type: {profile['architecture'].upper()} ({'MoE' if profile['is_moe'] else 'Dense'})", flush=True)
    if profile["is_moe"]:
        print(f"  Experts: {profile['experts']}", flush=True)
    
    print(f"--- Phase 1: Ingesting {model_dir.name} ---", flush=True)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), 
            torch_dtype=torch.float32, 
            device_map="cpu", 
            trust_remote_code=True
        )
        embeddings = model.get_input_embeddings().weight.detach().numpy()
        vocab_size, hidden_dim = embeddings.shape
        print(f"  Vocab Size: {vocab_size}, Hidden Dim: {hidden_dim}", flush=True)
    except Exception as exc:
        print(f"Error during ingestion: {exc}")
        return 1

    print(f"--- Phase 2: Morphological Trie Discovery ---", flush=True)
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    roots = build_morphological_trie(sorted_vocab)
    print(f"  Discovered {len(roots)} productive linguistic roots.", flush=True)
    
    print(f"--- Phase 3: Geometric Neighborhood Discovery ---", flush=True)
    hubs = find_semantic_hubs(embeddings)
    print(f"  Identified {len(hubs)} distinct semantic hubs (Concept Clusters).", flush=True)
    
    analysis = {
        "model": model_dir.name,
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "productive_roots": roots[:1000],
        "semantic_hubs": hubs,
        "summary": {
            "total_roots": len(roots),
            "total_hubs": len(hubs),
            "redundant_tokens": sum(h["member_count"] - 1 for h in hubs),
            "potential_vocab_reduction_pct": (sum(h["member_count"] - 1 for h in hubs) / vocab_size) * 100
        },
        "status": "Phases 1-3 Complete"
    }
    
    Path(args.out).write_text(json.dumps(analysis, indent=2) + "\n")
    print(f"Analysis saved to {args.out}", flush=True)

    # NEW: Automatic Sensitivity Map Generation for Orka Pack
    if getattr(args, "save_sensitivity_map", None):
        print(f"--- Generating Sensitivity Map for Pack ---", flush=True)
        # We define pillars as:
        # 1. Master tokens of semantic hubs (The unique concepts)
        # 2. Top 500 productive morphological roots
        top_tokens = set()
        for hub in hubs:
            top_tokens.add(hub["master_tid"])
        
        # Add tokens that ARE roots themselves (terminating at a productive node)
        # This is simplified: we take the first 1000 IDs for now as a heuristic
        # In Phase 4 we will do exact mapping.
        for tid in range(min(1000, vocab_size)):
            top_tokens.add(tid)

        s_map = {
            "top_tokens": sorted(list(top_tokens)),
            "layers": []
        }
        Path(args.save_sensitivity_map).write_text(json.dumps(s_map, indent=2) + "\n")
        print(f"Sensitivity map saved to {args.save_sensitivity_map}", flush=True)

    return 0

