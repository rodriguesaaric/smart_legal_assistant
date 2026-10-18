"""
Multi-Criteria Clause Scoring with 7 Scoring Heads.

Scores each clause on: Ambiguity (A), Maliciousness (M), Readability (R),
Transparency (T), Inconsistency (I), Fairness (F), Specificity (S).

Uses heuristic feature extraction + optional Random Forest regression.
"""

from typing import List, Dict, Optional
import re
import math
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# ── Lexicons ──────────────────────────────────────────────────────────
MODAL_WORDS = {"may", "might", "could", "should", "would", "can", "shall", "must"}
HEDGING_WORDS = {"may", "might", "could", "seems", "appears", "suggests", "possible",
                 "possibly", "likely", "unlikely", "approximate", "approximately"}
VAGUE_WORDS = {"some", "several", "various", "etc", "etc.", "certain", "reasonable",
               "appropriate", "adequate", "substantial", "material"}

MALICIOUS_PHRASES = {
    "at our sole discretion", "without notice", "without prior notice",
    "hold harmless", "indemnif", "no liability", "not liable", "not be liable",
    "sole discretion", "as we see fit", "we may terminate", "we will not",
    "we are not", "waive", "waiver", "forfeit", "penalty", "liquidated damages",
    "automatic renewal", "non-refundable", "irrevocable",
}

SNEAK_IN_PHRASES = {
    "modify without notice", "change without notice", "amend without notice",
    "right to modify", "right to change", "updated from time to time",
    "deemed to have accepted", "binding arbitration", "waive right to",
    "class action waiver", "jury trial waiver", "mandatory arbitration",
    "auto-renew", "automatically renew", "opt-out", "without prior consent",
}

FAIRNESS_NEGATIVE = {
    "unilateral", "sole discretion", "exclusive right", "at any time without",
    "no obligation", "unlimited liability", "disproportionate", "one-sided",
    "as it deems fit", "without cause", "unconditionally",
}

# Exclusivity / absolute restriction phrases
EXCLUSIVITY_PHRASES = {
    "exclusively allocated", "exclusively branded", "exclusively created",
    "exclusive presence", "no other sponsor", "no third-party", "no third party",
    "any third-party", "any third party",
    "shall not appear", "shall not be permitted", "no branding",
    "no sponsor logos", "no association", "shall not have any association",
    "no other platform", "shall not be displayed",
    "shall be no branding", "not be permitted",
}

# Mandatory obligation / performance guarantee phrases
OBLIGATION_PHRASES = {
    "shall ensure a minimum", "ensure a minimum of", "minimum of",
    "mandatory entry", "mandatory requirement", "required to register",
    "shall be required", "must be prominently", "will be validated",
    "shall ensure that", "shall guarantee", "guarantees that",
}

# Unilateral control / approval phrases
CONTROL_PHRASES = {
    "reserves the right", "right to approve", "right to disallow",
    "prior written approval", "require prior", "retain the right",
    "at its discretion", "in its sole judgment", "final say",
    "right to reject", "approval from the brand", "right to disapprove",
}

INCONSISTENCY_TRIGGERS = {"unless", "except", "notwithstanding", "however", "but",
                          "contrary", "subject to", "provided that"}
DEFINITIONS_TRIGGERS = {"means", "defined as", "definition", "shall mean",
                        "has the meaning", "for purposes of"}

# ── Clause-type detection ─────────────────────────────────────────────

# Patterns that indicate a clause is a definition block / address / boilerplate
DEFINITION_BLOCK_PATTERN = re.compile(
    r'(?i)(?:'
    r'(?:^|\n)\s*\d*\.?\s*definitions?\b'
    r'|"[^"]+"\s+(?:means|shall mean|has the meaning)'
    r'|\bfor purposes of this\b'
    r')',
)

ADDRESS_PATTERN = re.compile(
    r'(?:'
    r'(?:floor|suite|sector|road|street|avenue|lane|phase|plot|building|centre|center)'
    r'|(?:\b\d{4,6}\b)'
    r'|\b(?:pin|zip|postal)\s*(?:code)?\s*[:—-]?\s*\d'
    r'|(?:email\s*(?:id)?[:—]?\s*\S+@\S+)'
    r'|(?:dated\s+\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4})'
    r')',
    re.IGNORECASE
)

BOILERPLATE_HEADS = {
    "definitions", "interpretation", "recitals", "schedules",
    "annexure", "preamble", "table of contents", "signatures",
    "execution", "notices", "counterparts",
}

STANDARD_OPERATIONAL_HEADS = {
    "payment", "fees", "invoicing", "maintenance", "support",
    "delivery", "timeline", "reporting", "audit", "insurance",
    "warranty", "representations", "covenants", "conditions precedent",
    "service level", "sla",
}


def classify_clause_type(clause_text: str) -> str:
    """
    Classify a clause into one of:
      - "definition"  : pure definitions / glossary
      - "address"     : address blocks, contact info
      - "boilerplate" : standard boilerplate (notices, counterparts)
      - "standard"    : normal operational clause
      - "substantive" : clauses with legal risk potential
    """
    lower = clause_text.lower().strip()
    first_line = lower.split('\n')[0].strip()

    # Address blocks: contain physical addresses, emails, pin codes
    addr_hits = len(ADDRESS_PATTERN.findall(clause_text))
    if addr_hits >= 2:
        return "address"

    # Definition blocks: many quoted "term" means patterns
    definition_matches = len(re.findall(r'"[^"]+"[\s\S]{0,15}(?:means|shall mean)', lower))
    if definition_matches >= 2:
        return "definition"
    if DEFINITION_BLOCK_PATTERN.search(clause_text) and definition_matches >= 1:
        return "definition"

    # Check first-line heading against known boilerplate sections
    heading = re.sub(r'^[\d.]+\s*', '', first_line).strip()
    for bp in BOILERPLATE_HEADS:
        if heading.startswith(bp):
            return "boilerplate"

    # Standard operational clauses
    for op in STANDARD_OPERATIONAL_HEADS:
        if heading.startswith(op):
            return "standard"

    return "substantive"


# ── Feature extraction ────────────────────────────────────────────────

def tokenize_words(text: str):
    return re.findall(r"[A-Za-z']{2,}", text.lower())


def sentence_split(text: str):
    return [p for p in re.split(r'[.!?;]+\s*', text.strip()) if p]


def count_syllables(word):
    word = re.sub(r'[^a-z]', '', word.lower())
    if not word:
        return 0
    syll = len(re.findall(r'[aeiouy]+', word))
    if word.endswith("e"):
        syll = max(1, syll - 1)
    return max(syll, 1)


def extract_features(clause: str) -> Dict[str, float]:
    words = tokenize_words(clause)
    n_words = max(len(words), 1)
    sentences = sentence_split(clause)
    n_sent = max(1, len(sentences))
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    avg_sent_len = n_words / n_sent
    long_words = sum(1 for w in words if len(w) > 6)
    complex_words = sum(1 for w in words if count_syllables(w) >= 3)
    caps_words = sum(1 for w in re.findall(r"\b[A-Z]{2,}\b", clause))
    modal_count = sum(1 for w in words if w in MODAL_WORDS)
    hedge_count = sum(1 for w in words if w in HEDGING_WORDS)
    vague_count = sum(1 for w in words if w in VAGUE_WORDS)
    lower = clause.lower()
    mal_count = sum(1 for ph in MALICIOUS_PHRASES if ph in lower)
    sneak_count = sum(1 for ph in SNEAK_IN_PHRASES if ph in lower)
    fairness_neg = sum(1 for ph in FAIRNESS_NEGATIVE if ph in lower)
    exclusivity_count = sum(1 for ph in EXCLUSIVITY_PHRASES if ph in lower)
    obligation_count = sum(1 for ph in OBLIGATION_PHRASES if ph in lower)
    control_count = sum(1 for ph in CONTROL_PHRASES if ph in lower)
    inc_count = sum(1 for ph in INCONSISTENCY_TRIGGERS if ph in lower)
    def_count = sum(1 for ph in DEFINITIONS_TRIGGERS if ph in lower)
    numeric_count = len(re.findall(r'\b\d+(?:[.,]\d+)?\b', clause))
    percent_count = len(re.findall(r'\b\d+%|\b\d+\spercent', lower))
    passive_approx = int(bool(
        re.search(r'\b(be|is|are|was|were|been|being)\b', lower) and
        re.search(r'\b\w+ed\b', lower)
    ))
    pronouns = {"it", "they", "them", "their", "its", "this", "that", "these", "those"}
    pronoun_count = sum(1 for w in words if w in pronouns)
    # Normalize counts by clause length
    norm = n_words / 100.0

    return {
        "n_words": n_words, "avg_word_len": avg_word_len, "avg_sent_len": avg_sent_len,
        "long_words": long_words, "complex_words": complex_words, "caps_words": caps_words,
        "modal_count": modal_count, "hedge_count": hedge_count, "vague_count": vague_count,
        "mal_count": mal_count, "sneak_count": sneak_count, "fairness_neg": fairness_neg,
        "exclusivity_count": exclusivity_count, "obligation_count": obligation_count,
        "control_count": control_count,
        "inc_count": inc_count, "def_count": def_count,
        "numeric_count": numeric_count, "percent_count": percent_count,
        "passive_approx": passive_approx, "pronoun_count": pronoun_count,
        "sentence_count": n_sent,
        # Normalized rates (per 100 words)
        "modal_rate": modal_count / norm if norm else 0,
        "hedge_rate": hedge_count / norm if norm else 0,
        "vague_rate": vague_count / norm if norm else 0,
    }


# ── 7-Head Heuristic Scoring ─────────────────────────────────────────

def heuristic_scores(features: Dict[str, float]) -> Dict[str, float]:
    """
    Produce scores for 7 multi-criteria heads:
      A = Ambiguity, M = Maliciousness, R = Readability,
      T = Transparency, I = Inconsistency, F = Fairness, S = Specificity

    All scores in [0, 1] where HIGHER = MORE of that property.
    For R, T, F, S: higher = BETTER (less risky).
    For A, M, I: higher = WORSE (more risky).
    """
    # ── Ambiguity (higher = more ambiguous = riskier) ──
    A_raw = (
        0.3 * features["modal_rate"] +
        0.4 * features["hedge_rate"] +
        0.5 * features["vague_rate"] +
        0.15 * features["pronoun_count"] / max(features["n_words"] / 50, 1) +
        0.3 * features["passive_approx"] +
        0.01 * max(0, features["avg_sent_len"] - 25)
    )

    # ── Maliciousness (higher = more malicious = riskier) ──
    # Triggers on malicious phrases, sneak-ins, exclusivity, obligations, and control
    M_raw = (
        1.5 * features["mal_count"] +
        1.0 * features["sneak_count"] +
        0.5 * features["fairness_neg"] +
        0.9 * features["exclusivity_count"] +
        0.6 * features["obligation_count"] +
        0.9 * features["control_count"]
    )

    # ── Readability / Clarity (higher = clearer = LESS risky) ──
    R_raw = (
        1.0
        - 0.015 * max(0, features["avg_sent_len"] - 15)
        - 0.02 * features["complex_words"] / max(features["n_words"] / 20, 1)
        - 0.1 * features["passive_approx"]
        - 0.005 * features["long_words"] / max(features["n_words"] / 20, 1)
    )

    # ── Transparency (higher = more transparent = LESS risky) ──
    T_raw = (
        0.4 +
        0.15 * min(features["numeric_count"], 5) +
        0.2 * min(features["percent_count"], 3) +
        0.15 * min(features["def_count"], 4) -
        0.15 * features["vague_count"]
    )

    # ── Inconsistency (higher = more inconsistent = riskier) ──
    I_raw = (
        0.5 * features["inc_count"] +
        0.1 * features["modal_count"] +
        0.15 * features["hedge_count"]
    )

    # ── Fairness (higher = fairer = LESS risky) ──
    F_raw = (
        0.7
        - 0.25 * features["fairness_neg"]
        - 0.15 * features["mal_count"]
        - 0.2 * features["sneak_count"]
        - 0.15 * features["exclusivity_count"]
        - 0.2 * features["control_count"]
    )

    # ── Specificity (higher = more specific = LESS risky) ──
    S_raw = (
        0.3 +
        0.12 * min(features["numeric_count"], 5) +
        0.15 * min(features["percent_count"], 3) +
        0.1 * min(features["def_count"], 4) +
        0.08 * (features["avg_word_len"] > 5)
        - 0.2 * features["vague_rate"]
    )

    # ── Squash to [0, 1] via sigmoid ──
    # Center at 0.5 with steep slope so only extreme values get extreme scores
    def squash(x, center=0.5, steepness=3.0):
        return 1.0 / (1.0 + math.exp(-steepness * (x - center)))

    return {
        "A": float(np.clip(squash(A_raw, center=1.2, steepness=2.5), 0, 1)),
        "M": float(np.clip(squash(M_raw, center=0.8, steepness=2.0), 0, 1)),
        "R": float(np.clip(R_raw, 0, 1)),
        "T": float(np.clip(T_raw, 0, 1)),
        "I": float(np.clip(squash(I_raw, center=1.5, steepness=2.0), 0, 1)),
        "F": float(np.clip(F_raw, 0, 1)),
        "S": float(np.clip(S_raw, 0, 1)),
    }


class ClauseScorer:
    def __init__(self):
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
        self.models = {}
        self.feature_names = None

    def fit(self, clauses: List[str], labels: Optional[List[Dict[str, float]]] = None):
        X_meta = [extract_features(c) for c in clauses]
        df_meta = pd.DataFrame(X_meta).fillna(0)
        self.feature_names = df_meta.columns.tolist()
        X_tfidf = self.tfidf.fit_transform(clauses)
        X = hstack([X_tfidf, df_meta.values])
        if labels is None:
            print("No labels provided — using heuristics only.")
            return
        Ys = {k: np.array([lab.get(k, np.nan) for lab in labels], dtype=float)
              for k in ["A", "M", "R", "T", "I", "F", "S"]}
        for k, y in Ys.items():
            mask = ~np.isnan(y)
            if mask.sum() < 5:
                print(f"Skipping {k}, not enough labeled examples")
                continue
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X[mask], y[mask])
            self.models[k] = rf
            print(f"Trained model for {k} on {mask.sum()} examples.")

    def predict(self, clause: str) -> Dict[str, float]:
        feat = extract_features(clause)
        heur = heuristic_scores(feat)
        X = hstack([self.tfidf.transform([clause]), np.array([list(feat.values())])])
        preds = {}
        for k in ["A", "M", "R", "T", "I", "F", "S"]:
            if k in self.models:
                raw = self.models[k].predict(X)[0]
                preds[k] = float(min(1.0, max(0.0, raw)))
            else:
                preds[k] = heur[k]
        return preds

    def predict_batch(self, clauses: List[str]) -> List[Dict[str, float]]:
        return [self.predict(c) for c in clauses]


if __name__ == "__main__":
    sample_json = [
        {"id": 1, "clause": '"Agreement" means these terms and conditions. "Venue" means the event location.'},
        {"id": 2, "clause": "Payment of INR 10,000 shall be made within 30 days from invoice date."},
        {"id": 3, "clause": "The Company may terminate this Agreement at any time without prior notice."},
        {"id": 4, "clause": "All disputes shall be finally settled by binding arbitration."},
        {"id": 5, "clause": "The Organiser shall ensure a minimum of 3000 registrations through the platform."},
        {"id": 6, "clause": "No other sponsor branding shall appear at the Venue at any time."},
        {"id": 7, "clause": "The Company reserves the right to modify these terms without notice."},
    ]
    df_demo = pd.DataFrame(sample_json)
    scorer = ClauseScorer()
    scorer.fit(df_demo["clause"].tolist(), labels=None)
    scores = scorer.predict_batch(df_demo["clause"].tolist())
    df_scores = pd.DataFrame(scores)
    result = pd.concat([df_demo, df_scores.round(3)], axis=1)
    print(result.to_string(index=False))
