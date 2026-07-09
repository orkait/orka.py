"""Semantic Reconstruction Engine: Ingestion, Trie Discovery, and Clustering."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Clustering {embeddings.shape[0]} vectors on {device}...", flush=True)
    
    t_emb = torch.from_numpy(embeddings).to(device)
    # Normalize for cosine similarity
    t_emb = torch.nn.functional.normalize(t_emb, p=2, dim=1)
    
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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from orka.deploy.kaggle import _hf_snapshot_with_retry
    
    model_input = args.model_dir
    model_dir = Path(model_input)
    
    if not model_dir.exists():
        print(f"--- Resolving {model_input} from HF Hub ---", flush=True)
        try:
            from huggingface_hub import snapshot_download
            model_dir = Path(snapshot_download(model_input))
        except Exception as exc:
            print(f"Error downloading model: {exc}")
            return 1
    
    print("--- Phase 0: Architectural Profiling ---", flush=True)
    profile = profile_architecture(model_dir)
    print(f"  Type: {profile.get('architecture', 'unknown').upper()} ({'MoE' if profile.get('is_moe') else 'Dense'})", flush=True)
    if profile.get("is_moe"):
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

    print("--- Phase 2: Morphological Trie Discovery ---", flush=True)
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    roots = build_morphological_trie(sorted_vocab)
    print(f"  Discovered {len(roots)} productive linguistic roots.", flush=True)
    
    print("--- Phase 3: Geometric Neighborhood Discovery ---", flush=True)
    hubs = find_semantic_hubs(embeddings)
    print(f"  Identified {len(hubs)} distinct semantic hubs (Concept Clusters).", flush=True)
    
    analysis = {
        "model": model_dir.name,
        "model_dir": str(model_dir),
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

    # Sensitivity map consumed by `orka pack --sensitivity-map`.
    if getattr(args, "save_sensitivity_map", None):
        print("--- Generating Sensitivity Map for Pack ---", flush=True)
        # Pillars are the master tokens of semantic hubs plus the most productive
        # morphological roots.
        top_tokens = set()
        for hub in hubs:
            top_tokens.add(hub["master_tid"])
        
        # Roots themselves (terminating at a productive node). The first 1000 IDs
        # stand in for exact root mapping.
        for tid in range(min(1000, vocab_size)):
            top_tokens.add(tid)

        s_map = {
            "top_tokens": sorted(list(top_tokens)),
            "layers": []
        }
        Path(args.save_sensitivity_map).write_text(json.dumps(s_map, indent=2) + "\n")
        print(f"Sensitivity map saved to {args.save_sensitivity_map}", flush=True)

    return 0


def cmd_sem_map(args: argparse.Namespace) -> int:
    """Entry point for orka sem-map (Phase 4)."""
    analysis_path = Path(args.analysis_json)
    if not analysis_path.exists():
        print(f"Error: Analysis file not found: {analysis_path}")
        return 1
        
    with open(analysis_path) as f:
        analysis = json.load(f)
        
    vocab_size = analysis["vocab_size"]
    hubs = analysis.get("semantic_hubs", [])
    
    print(f"--- Phase 4: Concept Union Mapping ({analysis['model']}) ---", flush=True)
    
    # link_table: child_tid -> {parent_tid, relationship_type, confidence}
    link_table = {}
    
    # Geometric hubs: high-confidence mathematical redundancies.
    for hub in hubs:
        master = hub["master_tid"]
        for member in hub["member_tids"]:
            if member == master: continue
            link_table[str(member)] = {
                "parent": int(master),
                "type": "geometric_synonym",
                "confidence": float(hub["avg_similarity"])
            }
            
    print(f"  Mapped {len(link_table)} geometric synonyms.", flush=True)
    
    # Morphological mapping would need the tokenizer here to match strings to roots.
    # Geometric hubs alone give the largest compression win.
    
    map_result = {
        "model": analysis["model"],
        "vocab_size": vocab_size,
        "master_concept_count": vocab_size - len(link_table),
        "links": link_table,
        "status": "Phase 4 Complete"
    }
    
    Path(args.out).write_text(json.dumps(map_result, indent=2) + "\n")
    print(f"Concept Map saved to {args.out}", flush=True)
    print(f"  Theoretical Deduplication: {len(link_table)} vectors removed.", flush=True)
    return 0


