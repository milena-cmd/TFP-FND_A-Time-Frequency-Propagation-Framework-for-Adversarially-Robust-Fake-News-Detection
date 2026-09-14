"""TFP-FND Dataset.

Each __getitem__ returns the *raw* (untokenized, unbatched) material for the four
conditions used in Joint Robust Optimization (Section "Joint Robust Optimization"):
  - clean       : original text, original cascade
  - content     : attacked text (one sampled accepted variant), original cascade
  - propagation : original text, attacked cascade (one sampled kappa/family)
  - joint       : attacked text, attacked cascade

At eval time (attack_mode="clean" or a single fixed attack), only the relevant
condition(s) are populated and the others are set to None so evaluation scripts can
request exactly the condition they need without paying for the rest.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import Dataset

from src.data.hypergraph import build_relations, truncate_prefix
from src.data.historical import HistoricalContext


class TFPFNDDataset(Dataset):
    def __init__(self, processed_dir: Path, split: dict, partition: str,
                 historical_ctx: HistoricalContext, tau_seconds: Optional[float] = None,
                 window_seconds: float = 600.0, max_reply_depth: int = 5,
                 sync_threshold_seconds: float = 60.0,
                 content_attack_cache: Optional[Path] = None,
                 propagation_attack_file: Optional[Path] = None,
                 include_attacks: bool = True, seed: int = 0):
        self.processed_dir = processed_dir
        self.partition = partition
        self.historical_ctx = historical_ctx
        self.tau_seconds = tau_seconds
        self.window_seconds = window_seconds
        self.max_reply_depth = max_reply_depth
        self.sync_threshold_seconds = sync_threshold_seconds
        self.include_attacks = include_attacks
        self.rng = random.Random(seed)

        with open(processed_dir / "instances.jsonl", "r", encoding="utf-8") as f:
            all_instances = {json.loads(l)["source_id"]: json.loads(l) for l in f if l.strip()}
        self.ids = [i for i in split[partition] if i in all_instances]
        self.by_id = {i: all_instances[i] for i in self.ids}

        self.content_variants: Dict[str, List[dict]] = {}
        if include_attacks and content_attack_cache and content_attack_cache.exists():
            with open(content_attack_cache, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if rec["variants"]:
                        self.content_variants[rec["source_id"]] = rec["variants"]

        self.propagation_attacked: Dict[str, dict] = {}
        if include_attacks and propagation_attack_file and propagation_attack_file.exists():
            with open(propagation_attack_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    self.propagation_attacked[rec["source_id"]] = rec

    def __len__(self) -> int:
        return len(self.ids)

    def _relations_for(self, cascade: List[dict], source_id: str, inst: dict) -> Dict:
        cascade = truncate_prefix(cascade, float(inst.get("source_ts") or 0.0), self.tau_seconds)
        source_user = inst.get("author", source_id)
        return build_relations(
            cascade, source_id, source_user, float(inst.get("source_ts") or 0.0),
            self.historical_ctx.community_of, self.window_seconds, self.max_reply_depth,
            self.sync_threshold_seconds,
        ), cascade

    def __getitem__(self, idx: int) -> Dict:
        sid = self.ids[idx]
        inst = self.by_id[sid]
        label = inst["label"]

        clean_relations, clean_cascade = self._relations_for(inst["cascade"], sid, inst)
        item = {
            "source_id": sid,
            "label": label,
            "clean_text": inst["text"],
            "clean_cascade": clean_cascade,
            "clean_relations": clean_relations,
        }

        content_text = None
        if sid in self.content_variants:
            variant = self.rng.choice(self.content_variants[sid])
            content_text = variant["text"]
        item["content_text"] = content_text

        prop_cascade, prop_relations = None, None
        if sid in self.propagation_attacked:
            attacked_inst = self.propagation_attacked[sid]
            prop_cascade_raw = attacked_inst["cascade"]
            prop_relations, prop_cascade = self._relations_for(prop_cascade_raw, sid, attacked_inst)
        item["propagation_cascade"] = prop_cascade
        item["propagation_relations"] = prop_relations

        return item
