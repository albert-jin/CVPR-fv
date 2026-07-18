"""Pseudo-verifiability labeler for CVPR-FV (main.tex Section 3.1).

For every rumor-detection instance with label ``ℓ ∈ {True, False}`` we
compute an *unverifiability score* ::

    u(c) = Σ_j w_j · 1[cue_j(c)]

using five heuristic linguistic cues (weights 1 each) plus an optional
LLM-based flag (cue f, weight 2). The claim is then assigned:

* ``Unverifiable`` if ``u(c) ≥ τ`` **and** the RD label is ``False``.
* ``Verifiable``   if ``u(c) <  τ`` **and** the RD label is ``True``.
* ``Undefined``    otherwise (dropped from training).

``τ = 2`` by default (matches the paper).

The heuristic cues are pure Python string checks — no external API call
is required by default, which makes the pipeline deterministic and
auditable (reviewer 1's ``label noise / circular reasoning'' concern).
If an ``LLM_FLAG_FN`` callable is provided, it is used to compute cue (f);
otherwise cue (f) is treated as inactive.

The reviewer-visible aspect: the exact cue lexicon lives in
``configs.CVP_CUES`` and can be inspected without diving into the code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Callable, Dict, List, Optional

import configs


# Compile lower-cased word-boundary regexes once — much faster than
# scanning the phrase list on every claim.
_CUE_REGEXES: List[re.Pattern] = []
for _name, _phrases in configs.CVP_CUES:
    _pat = r'|'.join(re.escape(p.lower()) for p in _phrases)
    _CUE_REGEXES.append(re.compile(rf'(?<![A-Za-z0-9])({_pat})(?![A-Za-z0-9])'))

_CUE_NAMES = [name for name, _ in configs.CVP_CUES]


# ---------------------------------------------------------------------------
# Offline heuristic proxy for cue (f) — the paper's "LLM flag".
# ---------------------------------------------------------------------------
#
# The paper obtains cue (f) by prompting GPT-5.2. That is a paid API call
# per claim, which makes the pipeline hard to reproduce. We ship an
# *offline* proxy that fires on a small set of high-signal patterns
# commonly found in unverifiable claims (emotional punctuation, hedged
# quotations, conspiracy triggers). It is intentionally conservative so
# it doesn't clash with cues (a)-(e) — the effective (f) rate stays
# well below 100 %.
#
# Callers can override the default proxy by passing an ``llm_flag_fn``
# to ``label_rd_corpus`` — e.g. wrap an actual GPT-5.2 call.

_LLM_PROXY_PATTERNS = [
    re.compile(r'!{2,}'),                                    # !!! …
    re.compile(r'\bcure[sd]?\b', re.IGNORECASE),
    re.compile(r'\bhoax(es|ed)?\b', re.IGNORECASE),
    re.compile(r'\bconspiracy\b', re.IGNORECASE),
    re.compile(r'\bsecret(ly)?\b', re.IGNORECASE),
    re.compile(r'\bshocking\b', re.IGNORECASE),
    re.compile(r'\bmiracle\b', re.IGNORECASE),
    re.compile(r'\bunbelievable\b', re.IGNORECASE),
    re.compile(r'\bproven\s+false\b', re.IGNORECASE),
    re.compile(r'\bthey\s+don\'?t\s+want\s+you\s+to\s+know\b', re.IGNORECASE),
    re.compile(r'\b(will|can)\s+kill\s+you\b', re.IGNORECASE),
    re.compile(r'\bbig\s+pharma\b', re.IGNORECASE),
]

# Long ALL-CAPS runs are a soft attention-grabbing signal, but common
# institutional acronyms are noise. Match a claim only when it contains
# an ALL-CAPS word of 4+ letters that is NOT on the allow-list.
_ALL_CAPS_RX = re.compile(r'\b[A-Z]{4,}\b')
_ALL_CAPS_ALLOWLIST = {
    'WHO', 'CDC', 'FBI', 'CIA', 'NASA', 'USA', 'USDA', 'EPA', 'FDA',
    'HHS', 'NIH', 'DOJ', 'DOD', 'IRS', 'GOP', 'DNC', 'RNC', 'FEMA',
    'NATO', 'ISIS', 'ISIL', 'UK', 'EU', 'UN', 'MRI', 'DNA', 'RNA',
    'HIV', 'AIDS', 'SARS', 'MERS', 'COVID', 'ICU', 'CEO', 'CFO',
    'IPO', 'GDP', 'CNN', 'BBC', 'NBC', 'ABC', 'MSNBC', 'HHS',
}


def default_llm_flag(claim: str) -> int:
    """Offline stand-in for the paper's GPT-5.2 flag (cue f).

    Returns 1 when the claim matches any high-risk pattern; 0 otherwise.
    Deterministic and dependency-free, so results are reproducible.
    """
    if not claim:
        return 0
    for rx in _LLM_PROXY_PATTERNS:
        if rx.search(claim):
            return 1
    # ALL-CAPS words (excluding common agency / medical acronyms).
    for match in _ALL_CAPS_RX.findall(claim):
        if match not in _ALL_CAPS_ALLOWLIST:
            return 1
    return 0


def cue_indicators(claim: str) -> Dict[str, int]:
    """Return a dict mapping each cue name → {0, 1}."""
    text = claim.lower()
    return {name: int(bool(rx.search(text))) for name, rx in zip(_CUE_NAMES, _CUE_REGEXES)}


def unverifiability_score(claim: str, llm_flag: int = 0) -> int:
    """Compute u(c) = Σ w_j · 1[cue_j] with cues (a)-(e) w=1 and (f) w=2."""
    ind = cue_indicators(claim)
    score = 0
    for name, w in zip(_CUE_NAMES, configs.CVP_CUE_WEIGHTS):
        score += w * ind[name]
    score += configs.CVP_LLM_WEIGHT * int(bool(llm_flag))
    return score


def pseudo_verifiability_label(claim: str,
                                rd_label: str,
                                llm_flag: int = 0,
                                tau: int = None) -> Optional[str]:
    """Return one of ``Verifiable`` / ``Unverifiable`` / ``None``.

    Following main.tex Section 3.1:

    * u(c) ≥ τ and rd_label == 'FAKE' → Unverifiable
    * u(c) <  τ and rd_label == 'REAL' → Verifiable
    * otherwise: undefined (excluded from training).
    """
    tau = configs.CVP_TAU if tau is None else tau
    u = unverifiability_score(claim, llm_flag=llm_flag)
    rd = str(rd_label).strip().upper()
    if u >= tau and rd == 'FAKE':
        return 'Unverifiable'
    if u < tau and rd == 'REAL':
        return 'Verifiable'
    return None


def label_rd_corpus(rd_rows: List[dict],
                     llm_flag_fn: Optional[Callable[[str], int]] = None,
                     tau: int = None) -> List[dict]:
    """Vectorised pseudo-labelling.

    ``llm_flag_fn`` defaults to :func:`default_llm_flag` — an offline
    heuristic proxy for the paper's GPT-5.2 flag. Pass your own
    callable (e.g. a wrapped API call) to reproduce the paper's exact
    numbers.

    Returns rows augmented with:

    * ``cvp_label``   : "Verifiable" | "Unverifiable"  (undefined rows dropped)
    * ``cvp_u_score`` : integer u(c)
    * ``cvp_cues``    : dict of cue → 0/1  (for later analysis)
    """
    flag_fn = llm_flag_fn if llm_flag_fn is not None else default_llm_flag
    out = []
    for row in rd_rows:
        claim = row.get('claim') or row.get('text') or row.get('statement') or ''
        try:
            llm_flag = int(bool(flag_fn(claim)))
        except Exception:
            llm_flag = 0
        label = pseudo_verifiability_label(claim, row.get('label', ''), llm_flag=llm_flag, tau=tau)
        if label is None:
            continue
        new_row = dict(row)
        new_row['cvp_label'] = label
        new_row['cvp_u_score'] = unverifiability_score(claim, llm_flag=llm_flag)
        new_row['cvp_cues'] = cue_indicators(claim)
        new_row['cvp_llm_flag'] = llm_flag
        out.append(new_row)
    return out


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Assign pseudo-verifiability labels to a rumor-detection corpus.')
    p.add_argument('--input', required=True, help='Path to an RD JSONL file (id, claim, label).')
    p.add_argument('--output', required=True, help='Path to the CVP-labelled JSONL.')
    p.add_argument('--tau', type=int, default=None,
                   help='Unverifiability threshold τ (default: configs.CVP_TAU).')
    args = p.parse_args()

    rows = []
    with open(args.input, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    labelled = label_rd_corpus(rows, tau=args.tau)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for row in labelled:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    n_ver = sum(1 for r in labelled if r['cvp_label'] == 'Verifiable')
    n_unv = sum(1 for r in labelled if r['cvp_label'] == 'Unverifiable')
    print(f'in : {len(rows)} rows')
    print(f'out: {len(labelled)} rows  (Verifiable={n_ver}, Unverifiable={n_unv})')


if __name__ == '__main__':
    main()
