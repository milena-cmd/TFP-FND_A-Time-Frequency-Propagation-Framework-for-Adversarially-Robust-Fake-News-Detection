"""Content-level adversarial attack generator.

Implements Section "Adversarial Content Transformation":
  - 5 transformation families: paraphrasing, syntactic restructuring, lexical
    substitution, sensational rewriting, neutral rewriting.
  - Candidates generated with GPT-4 (gpt-4-0613, temperature 0.7, top_p 0.9,
    max_tokens 200), 5 candidates per source text.
  - Each candidate is filtered with a bidirectional NLI check (DeBERTa-large-mnli):
    keep only if entailment > entail_thresh AND contradiction < contra_thresh in BOTH
    directions (original->transformed and transformed->original).
  - Attacks are generated strictly after split assignment (run this only after
    src/preprocessing/splits.py); a perturbed sample always inherits its source's split
    membership -- this script never touches split files, only reads them.
  - Results are cached by (source_id, model, temperature) so repeated runs don't
    re-spend API budget; delete the cache file to force regeneration.

Requires OPENAI_API_KEY in the environment for generation. NLI filtering requires a
local `transformers` model (downloaded on first use).
"""
import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

TRANSFORMATION_FAMILIES = [
    "paraphrasing", "syntactic restructuring", "lexical substitution",
    "sensational rewriting", "neutral rewriting",
]

PROMPT_TEMPLATE = """You will rewrite a short news/social-media post using the
"{family}" transformation while strictly preserving its factual claim (do not add,
remove, or change any fact, name, date, or number). Output only the rewritten text,
with no preamble or quotation marks.

Original: {text}
Rewritten:"""


def _cache_path(processed_dir: Path) -> Path:
    cache_dir = processed_dir / "attack_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "content_attacks.jsonl"


def _load_cache(cache_path: Path) -> Dict[str, dict]:
    cache = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cache[rec["cache_key"]] = rec
    return cache


def _cache_key(source_id: str, model: str, temperature: float) -> str:
    raw = f"{source_id}|{model}|{temperature}"
    return hashlib.sha256(raw.encode()).hexdigest()


class NLIFilter:
    """Bidirectional entailment/contradiction filter, DeBERTa-large-mnli by default."""

    def __init__(self, model_name: str = "microsoft/deberta-large-mnli", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # MNLI label order for this checkpoint: 0=contradiction, 1=neutral, 2=entailment
        self.contra_idx, self.neutral_idx, self.entail_idx = 0, 1, 2

    @torch.no_grad()
    def probs(self, premise: str, hypothesis: str) -> Dict[str, float]:
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt",
                                 truncation=True, max_length=256).to(self.device)
        logits = self.model(**inputs).logits[0]
        p = torch.softmax(logits, dim=-1).cpu().tolist()
        return {"contradiction": p[self.contra_idx], "neutral": p[self.neutral_idx],
                "entailment": p[self.entail_idx]}

    def accept(self, original: str, transformed: str, entail_thresh: float,
               contra_thresh: float) -> bool:
        fwd = self.probs(original, transformed)
        bwd = self.probs(transformed, original)
        return (
            fwd["entailment"] > entail_thresh and fwd["contradiction"] < contra_thresh and
            bwd["entailment"] > entail_thresh and bwd["contradiction"] < contra_thresh
        )


def _generate_candidates_openai(text: str, family: str, model: str, temperature: float,
                                 top_p: float, max_tokens: int, num_candidates: int) -> List[str]:
    from openai import AuthenticationError, OpenAI  # imported lazily so the module is importable without the package
    client = OpenAI()
    prompt = PROMPT_TEMPLATE.format(family=family, text=text)
    candidates = []
    for _ in range(num_candidates):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            )
            candidates.append(resp.choices[0].message.content.strip())
        except AuthenticationError:
            # Every remaining call in this run would fail identically (bad/placeholder
            # key) -- re-raise instead of retrying thousands of calls one at a time.
            raise
        except Exception as exc:  # pragma: no cover - transient network/service errors
            log.warning("Generation failed for family=%s: %s", family, exc)
            time.sleep(2)
    return candidates


def _check_api_key() -> None:
    """Fail fast with a clear message if OPENAI_API_KEY is missing or invalid, before
    committing to a run that could otherwise make thousands of identical failing calls
    (e.g. ~30,000 for a full 1,198-instance x 5-family x 5-candidate PHEME run) before
    anyone notices and interrupts it.
    """
    from openai import AuthenticationError, OpenAI
    client = OpenAI()
    try:
        client.models.list()  # cheapest possible authenticated call; costs nothing
    except AuthenticationError as exc:
        raise EnvironmentError(
            "OPENAI_API_KEY is set but was rejected by the API (invalid or placeholder "
            "key). Set it to your real key, e.g.: os.environ['OPENAI_API_KEY'] = "
            "'sk-proj-...your actual key...' -- not the literal example string 'sk-...' "
            f"from documentation. Original error: {exc}"
        ) from exc


def _check_model_available(model: str) -> None:
    """Fail fast if the requested model is unavailable (deprecated/retired snapshot,
    or not enabled for this account) -- otherwise every single generation call fails
    identically with a 404, and without this check that's only discovered after
    burning through the whole retry-and-warn loop for every instance/family/candidate.

    Dated model snapshots are retired by the provider from time to time, so the model
    name is checked once before any generation call is made.
    """
    from openai import NotFoundError, OpenAI
    client = OpenAI()
    try:
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=1,
        )
    except NotFoundError as exc:
        available = []
        try:
            available = sorted(
                m.id for m in client.models.list().data
                if "gpt" in m.id.lower() or "o1" in m.id.lower() or "o3" in m.id.lower()
            )
        except Exception:  # pragma: no cover - best-effort suggestion only
            pass
        suggestion = (
            f" Models currently available to this account include: {', '.join(available[:15])}"
            f"{' ...' if len(available) > 15 else ''}"
        ) if available else ""
        raise EnvironmentError(
            f"Model '{model}' is not available to this API key (likely deprecated/"
            f"retired, or not enabled for this account/org). Pass --model with a "
            f"currently available model instead.{suggestion} Original error: {exc}"
        ) from exc


def run(processed_dir: Path, model: str, temperature: float, top_p: float,
        max_tokens: int, num_candidates: int, nli_model: str, entail_thresh: float,
        contra_thresh: float, limit: Optional[int] = None) -> None:
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("Set OPENAI_API_KEY before running content attack generation.")
    _check_api_key()
    _check_model_available(model)

    with open(processed_dir / "instances.jsonl", "r", encoding="utf-8") as f:
        instances = [json.loads(line) for line in f if line.strip()]
    if limit:
        instances = instances[:limit]

    cache_path = _cache_path(processed_dir)
    cache = _load_cache(cache_path)
    nli = NLIFilter(nli_model)

    n_generated, n_accepted, n_from_cache = 0, 0, 0
    with open(cache_path, "a", encoding="utf-8") as out_f:
        for inst in instances:
            key = _cache_key(inst["source_id"], model, temperature)
            if key in cache:
                n_from_cache += 1
                continue

            accepted_variants = []
            any_candidate_generated = False
            for family in TRANSFORMATION_FAMILIES:
                candidates = _generate_candidates_openai(
                    inst["text"], family, model, temperature, top_p, max_tokens, num_candidates,
                )
                n_generated += len(candidates)
                if candidates:
                    any_candidate_generated = True
                for cand in candidates:
                    if nli.accept(inst["text"], cand, entail_thresh, contra_thresh):
                        accepted_variants.append({"family": family, "text": cand})
                        n_accepted += 1

            if not any_candidate_generated:
                # Generation itself produced nothing for every family -- almost always a
                # transient failure (bad key, rate limit, network) rather than a genuine
                # "NLI rejected everything" outcome. Don't cache this as done: caching it
                # would make every future run skip this instance forever, even after
                # whatever caused the failure is fixed. A real "0 accepted" outcome (NLI
                # legitimately rejected every generated candidate) still gets cached below,
                # since in that case candidates. were actually generated.
                log.warning(
                    "source_id=%s produced zero candidates across all %d families -- "
                    "not caching, will retry on next run.",
                    inst["source_id"], len(TRANSFORMATION_FAMILIES),
                )
                continue

            record = {
                "cache_key": key,
                "source_id": inst["source_id"],
                "model": model,
                "temperature": temperature,
                "variants": accepted_variants,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    log.info(
        "Content attack generation done: %d generated, %d accepted (%.1f%%), %d served from cache",
        n_generated, n_accepted, 100 * n_accepted / max(1, n_generated), n_from_cache,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4-0613")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--num_candidates", type=int, default=5)
    parser.add_argument("--nli_model", default="microsoft/deberta-large-mnli")
    parser.add_argument("--entail_thresh", type=float, default=0.85)
    parser.add_argument("--contra_thresh", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional cap on number of source instances, for smoke-testing.")
    args = parser.parse_args()
    run(args.processed_dir, args.model, args.temperature, args.top_p, args.max_tokens,
        args.num_candidates, args.nli_model, args.entail_thresh, args.contra_thresh, args.limit)
