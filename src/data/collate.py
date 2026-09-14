"""Collate function for TFPFNDDataset.

Text is tokenized here (BERT tokenizer); cascades/relations are left as raw Python
structures because their size varies per instance and per relation -- padding and
hyperedge-index construction happen inside the model's temporal/propagation view
modules, which build a fresh padded batch tensor on every forward call.

`has_content_attack` / `has_propagation_attack` tell the loss functions which samples
in the batch have a *real* attacked variant vs. a clean fallback (when no accepted
content variant or no generated propagation attack exists for that instance) --
consistency losses are only computed over samples where the attack is real.
"""
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizerBase


def make_collate_fn(tokenizer: PreTrainedTokenizerBase, max_seq_len: int = 128):
    def collate_fn(batch: List[Dict]) -> Dict:
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        source_ids = [b["source_id"] for b in batch]

        clean_texts = [b["clean_text"] for b in batch]
        content_texts = [b["content_text"] or b["clean_text"] for b in batch]
        has_content_attack = torch.tensor(
            [b["content_text"] is not None for b in batch], dtype=torch.bool
        )
        has_propagation_attack = torch.tensor(
            [b["propagation_cascade"] is not None for b in batch], dtype=torch.bool
        )

        clean_cascades = [b["clean_cascade"] for b in batch]
        clean_relations = [b["clean_relations"] for b in batch]
        prop_cascades = [b["propagation_cascade"] or b["clean_cascade"] for b in batch]
        prop_relations = [b["propagation_relations"] or b["clean_relations"] for b in batch]

        def _tok(texts: List[str]) -> Dict[str, torch.Tensor]:
            enc = tokenizer(
                texts, padding=True, truncation=True, max_length=max_seq_len,
                return_tensors="pt",
            )
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

        return {
            "source_id": source_ids,
            "label": labels,
            "has_content_attack": has_content_attack,
            "has_propagation_attack": has_propagation_attack,
            "conditions": {
                # a=0 (clean): clean text, clean cascade
                "clean": {**_tok(clean_texts), "cascades": clean_cascades, "relations": clean_relations},
                # a=c: attacked text, clean cascade
                "content": {**_tok(content_texts), "cascades": clean_cascades, "relations": clean_relations},
                # a=p: clean text, attacked cascade
                "propagation": {**_tok(clean_texts), "cascades": prop_cascades, "relations": prop_relations},
                # a=cp: attacked text, attacked cascade
                "joint": {**_tok(content_texts), "cascades": prop_cascades, "relations": prop_relations},
            },
        }

    return collate_fn
