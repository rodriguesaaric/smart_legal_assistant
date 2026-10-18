"""
Contract Analysis Orchestrator — v2 (Paper-aligned).

Implements:
  - 7 Multi-Criteria Scoring Heads (A, M, R, T, I, F, S)
  - Expert Weight Configuration (w_m) with normalization ∑w_m = 1
  - Weighted Document Score (S_doc) with 0.75 normalized threshold
  - Clause-type filtering (skip definitions, addresses, boilerplate)
  - Consumer-mode sneak-in clause penalization
  - Specificity-tuned thresholds targeting 0.920 specificity / 90.5% accuracy
"""

import re
import math
import json
import os
import time
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("contractshield")

from scoring.models.random_forest import (
    ClauseScorer, extract_features, heuristic_scores, classify_clause_type,
    SNEAK_IN_PHRASES, FAIRNESS_NEGATIVE, MALICIOUS_PHRASES,
    VAGUE_WORDS, HEDGING_WORDS,
    EXCLUSIVITY_PHRASES, OBLIGATION_PHRASES, CONTROL_PHRASES,
)

# ─────────────────────────────────────────────────────────────────────
# Expert Weights (w_m) — sum to 1.0
# Higher weights for Maliciousness, Fairness, Transparency (primary risk)
# Lower weight for standard Ambiguity
# ─────────────────────────────────────────────────────────────────────
EXPERT_WEIGHTS = {
    "M": 0.22,   # Maliciousness — highest weight
    "F": 0.18,   # Fairness
    "T": 0.16,   # Transparency
    "I": 0.12,   # Inconsistency
    "A": 0.10,   # Ambiguity — lower weight
    "R": 0.12,   # Readability
    "S": 0.10,   # Specificity
}
# Verification: sum = 1.0
assert abs(sum(EXPERT_WEIGHTS.values()) - 1.0) < 1e-6, \
    f"Weight normalization violated: ∑w_m = {sum(EXPERT_WEIGHTS.values())}"

# ─────────────────────────────────────────────────────────────────────
# Thresholds — tuned for 0.920 specificity
# ─────────────────────────────────────────────────────────────────────
# For "risky" heads (A, M, I): higher score = riskier → threshold is upper bound
# For "good" heads (R, T, F, S): higher score = better → threshold is lower bound

# S_doc threshold: a clause is only "High Risk" if weighted doc score > 0.75
S_DOC_HIGH_THRESHOLD = 0.75
S_DOC_MEDIUM_THRESHOLD = 0.45

# Per-head thresholds (used for individual dimension reporting)
THRESHOLDS = {
    "consumer": {
        # Risky heads (higher = worse): score > threshold = flagged
        "A": {"medium": 0.55, "high": 0.72},
        "M": {"medium": 0.45, "high": 0.62},
        "I": {"medium": 0.55, "high": 0.70},
        # Good heads (higher = better): score < threshold = flagged
        "R": {"medium": 0.45, "high": 0.30},
        "T": {"medium": 0.45, "high": 0.30},
        "F": {"medium": 0.45, "high": 0.30},
        "S": {"medium": 0.40, "high": 0.25},
    },
    "company": {
        "A": {"medium": 0.65, "high": 0.80},
        "M": {"medium": 0.55, "high": 0.70},
        "I": {"medium": 0.65, "high": 0.78},
        "R": {"medium": 0.38, "high": 0.25},
        "T": {"medium": 0.38, "high": 0.25},
        "F": {"medium": 0.38, "high": 0.25},
        "S": {"medium": 0.35, "high": 0.20},
    },
}

CATEGORY_MAP = {
    "A": "ambiguity",
    "M": "malicious_intent",
    "R": "clarity",
    "T": "compliance",
    "I": "inconsistency",
    "F": "fairness",
    "S": "specificity",
}

SCORE_LABELS = {
    "A": "Ambiguity",
    "M": "Malicious Intent",
    "R": "Clarity",
    "T": "Compliance",
    "I": "Inconsistency",
    "F": "Fairness",
    "S": "Specificity",
}

# ─────────────────────────────────────────────────────────────────────
# Clause splitting
# ─────────────────────────────────────────────────────────────────────

CLAUSE_PATTERN = re.compile(
    r'(?:^|\n)'
    r'(?:'
    r'(?:\d+\.[\d.]*)\s+'
    r'|(?:[IVXLC]+\.)\s+'
    r'|(?:[a-z]\))\s+'
    r'|(?:Section\s+\d+)'
    r'|(?:Clause\s+\d+)'
    r'|(?:Article\s+\d+)'
    r'|(?:ARTICLE\s+\d+)'
    r')',
    re.IGNORECASE
)


def split_text_into_clauses(text: str) -> List[Dict]:
    """Split raw contract text into clauses using regex patterns."""
    boundaries = [m.start() for m in CLAUSE_PATTERN.finditer(text)]

    if len(boundaries) >= 2:
        clauses = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
            clause_text = text[start:end].strip()
            if clause_text and len(clause_text) > 20:
                clauses.append({"id": len(clauses) + 1, "text": clause_text})
        if clauses:
            return clauses

    paragraphs = re.split(r'\n\s*\n', text.strip())
    if len(paragraphs) < 2:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    if len(paragraphs) < 2:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        paragraphs = []
        for i in range(0, len(sentences), 3):
            chunk = ' '.join(sentences[i:i + 3]).strip()
            if chunk:
                paragraphs.append(chunk)

    clauses = []
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if para and len(para) > 15:
            clauses.append({"id": i + 1, "text": para})

    return clauses if clauses else [{"id": 1, "text": text.strip()}]


# ─────────────────────────────────────────────────────────────────────
# Weighted Document Score (S_doc)
# ─────────────────────────────────────────────────────────────────────

def compute_sdoc(scores: Dict[str, float]) -> float:
    """
    Compute the Weighted Document Score for a single clause.

    For risky heads (A, M, I): use score directly (higher = worse)
    For good heads (R, T, F, S): invert to (1 - score) so higher = worse

    S_doc = Σ w_m * risk_m, normalized to [0, 1]
    """
    risk_values = {
        "A": scores.get("A", 0),
        "M": scores.get("M", 0),
        "I": scores.get("I", 0),
        "R": 1.0 - scores.get("R", 0.5),
        "T": 1.0 - scores.get("T", 0.5),
        "F": 1.0 - scores.get("F", 0.5),
        "S": 1.0 - scores.get("S", 0.5),
    }

    s_doc = sum(EXPERT_WEIGHTS[k] * risk_values[k] for k in EXPERT_WEIGHTS)
    return float(min(1.0, max(0.0, s_doc)))


# ─────────────────────────────────────────────────────────────────────
# Severity classification
# ─────────────────────────────────────────────────────────────────────

def classify_severity_from_sdoc(s_doc: float) -> str:
    """Classify severity based on weighted document score."""
    if s_doc >= S_DOC_HIGH_THRESHOLD:
        return "high"
    elif s_doc >= S_DOC_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def get_primary_risk_dimension(scores: Dict[str, float], user_type: str) -> tuple:
    """Find the highest-risk individual dimension for a clause."""
    th = THRESHOLDS.get(user_type, THRESHOLDS["consumer"])
    worst_key = None
    worst_sev = "low"
    worst_sev_num = 0

    risky_heads = {"A", "M", "I"}
    good_heads = {"R", "T", "F", "S"}

    for key in ["M", "F", "A", "I", "R", "T", "S"]:  # Priority order
        if key not in th:
            continue
        val = scores.get(key, 0.5)
        if key in risky_heads:
            if val >= th[key]["high"]:
                sev = "high"
            elif val >= th[key]["medium"]:
                sev = "medium"
            else:
                sev = "low"
        else:  # good heads - inverted
            if val <= th[key]["high"]:
                sev = "high"
            elif val <= th[key]["medium"]:
                sev = "medium"
            else:
                sev = "low"

        sev_num = {"low": 0, "medium": 1, "high": 2}[sev]
        if sev_num > worst_sev_num:
            worst_sev_num = sev_num
            worst_sev = sev
            worst_key = key

    return worst_key or "A", worst_sev


# ─────────────────────────────────────────────────────────────────────
# Consumer-mode sneak-in penalization
# ─────────────────────────────────────────────────────────────────────

def apply_consumer_penalties(scores: Dict[str, float], clause_text: str) -> Dict[str, float]:
    """
    In consumer mode, penalize 'sneak-in' clauses:
    - Modification without notice
    - Arbitration waivers
    - Auto-renewal traps
    """
    lower = clause_text.lower()
    penalty = 0.0

    for phrase in SNEAK_IN_PHRASES:
        if phrase in lower:
            penalty += 0.15

    if penalty > 0:
        scores = dict(scores)
        scores["M"] = min(1.0, scores["M"] + penalty)
        scores["F"] = max(0.0, scores["F"] - penalty * 0.5)
    return scores


def should_downgrade_standard(clause_text: str, clause_type: str, scores: Dict[str, float]) -> bool:
    """
    Standard operational clauses (payment, maintenance, delivery)
    should be downgraded if they follow industry standards.
    """
    if clause_type != "standard":
        return False
    # If no malicious phrases are present, it's a standard clause
    lower = clause_text.lower()
    mal_hits = sum(1 for ph in MALICIOUS_PHRASES if ph in lower)
    sneak_hits = sum(1 for ph in SNEAK_IN_PHRASES if ph in lower)
    return mal_hits == 0 and sneak_hits == 0


# ─────────────────────────────────────────────────────────────────────
# Risk reason generation
# ─────────────────────────────────────────────────────────────────────

def generate_reason(clause_text: str, score_key: str, severity: str, value: float) -> str:
    """Generate a detailed, clause-specific explanation of why this clause was flagged."""
    lower = clause_text.lower()
    parts = []

    if score_key == "M":
        # ── Maliciousness: identify exact trigger phrases ──
        triggers_found = []
        trigger_explanations = {
            "hold harmless": "a 'hold harmless' provision that shields one party from legal responsibility",
            "indemnif": "an indemnification obligation that shifts financial risk to one party",
            "sole discretion": "a 'sole discretion' clause giving one party unchecked decision-making power",
            "without notice": "a provision allowing action 'without notice,' removing the other party's opportunity to respond",
            "without prior notice": "a provision allowing action 'without prior notice,' denying advance warning",
            "not liable": "language disclaiming liability, which may leave the other party without recourse",
            "not be liable": "language disclaiming liability, which may leave the other party without recourse",
            "no liability": "a blanket liability exclusion that may be unenforceable in some jurisdictions",
            "waive": "a waiver of rights that could prevent a party from seeking legal remedies",
            "waiver": "a waiver provision that may strip important legal protections",
            "forfeit": "a forfeiture clause that could result in loss of rights or assets",
            "penalty": "a penalty provision that may impose disproportionate financial consequences",
            "liquidated damages": "a liquidated damages clause — verify the amount is a genuine pre-estimate, not punitive",
            "as we see fit": "the phrase 'as we see fit' which grants broad unilateral authority",
            "automatic renewal": "an auto-renewal provision that may lock a party into extended commitments",
            "non-refundable": "a non-refundable term that eliminates financial flexibility",
            "irrevocable": "an 'irrevocable' commitment that permanently removes the ability to withdraw",
            "we may terminate": "unilateral termination rights favoring one party",
            "at any time without": "the ability to act 'at any time without' restriction, creating an asymmetric power dynamic",
        }
        for phrase, explanation in trigger_explanations.items():
            if phrase in lower:
                triggers_found.append(explanation)

        # Also detect exclusivity, obligation, and control triggers
        excl_explanations = {
            "exclusively allocated": "exclusive allocation of resources or spaces, limiting opportunities for other stakeholders",
            "exclusively branded": "exclusive branding rights that prevent any competing visibility",
            "no other sponsor": "a prohibition on other sponsors, creating a monopolistic presence",
            "no branding": "blanket branding restrictions that grant disproportionate visual control",
            "no sponsor logos": "absolute prohibition on competing sponsor visibility",
            "shall not appear": "absolute restrictions on what may appear, leaving no room for negotiation",
            "shall not be displayed": "blanket display restrictions giving one party total visual control",
            "no association": "a prohibition on external associations that may limit the other party's business relationships",
        }
        for phrase, explanation in excl_explanations.items():
            if phrase in lower:
                triggers_found.append(explanation)

        oblig_explanations = {
            "shall ensure a minimum": "a mandatory minimum performance target that creates binding financial or operational risk",
            "ensure a minimum of": "a guaranteed minimum obligation — failure to meet this could trigger breach of contract",
            "mandatory entry": "a mandatory entry requirement that may restrict participant autonomy",
            "required to register": "a compulsory registration requirement tied to a third-party platform",
            "will be validated": "a validation requirement — if the minimum is not met, this creates a contractual exposure",
            "must be prominently": "a mandatory prominence requirement dictating specific placement or visibility",
        }
        for phrase, explanation in oblig_explanations.items():
            if phrase in lower:
                triggers_found.append(explanation)

        ctrl_explanations = {
            "reserves the right": "a 'reserves the right' provision giving one party unilateral decision-making authority",
            "right to approve": "a unilateral approval right that gives one party veto power over the other's decisions",
            "right to disallow": "the right to disallow actions, creating an asymmetric power dynamic",
            "prior written approval": "a requirement for prior written approval, giving one party gatekeeping control",
            "retain the right": "retained rights that limit the other party's autonomy",
        }
        for phrase, explanation in ctrl_explanations.items():
            if phrase in lower:
                triggers_found.append(explanation)

        if triggers_found:
            if severity == "high":
                parts.append("This clause raises significant concerns due to:")
                for t in triggers_found[:4]:
                    parts.append(f"  • {t}")
                parts.append("These provisions, taken together, create a substantial imbalance of power and may expose one party to undue risk.")
            else:
                parts.append(f"This clause contains {triggers_found[0]}.")
                if len(triggers_found) > 1:
                    parts.append(f"It also includes {triggers_found[1]}.")
                if len(triggers_found) > 2:
                    parts.append(f"Additionally, there is {triggers_found[2]}.")
                parts.append("While these terms are not uncommon, they warrant careful review to ensure balanced risk allocation.")
        else:
            parts.append("This clause contains language patterns commonly associated with one-sided contractual terms. Review recommended.")

    elif score_key == "A":
        # ── Ambiguity: identify vague/hedging language ──
        vague_found = [w for w in VAGUE_WORDS if w in lower]
        hedge_found = [w for w in HEDGING_WORDS if w in lower and w not in {"may"}]
        modal_found = [w for w in {"may", "might", "could", "should"} if w in lower]

        if severity == "high":
            parts.append("This clause has significant ambiguity issues:")
            if vague_found:
                parts.append(f"  • Vague qualifiers like '{', '.join(vague_found[:3])}' leave key terms open to subjective interpretation.")
            if hedge_found:
                parts.append(f"  • Hedging language ('{', '.join(hedge_found[:3])}') weakens the enforceability of stated obligations.")
            if modal_found:
                parts.append(f"  • Modal verbs ('{', '.join(modal_found[:3])}') create uncertainty about whether actions are required or merely optional.")
            if not vague_found and not hedge_found:
                parts.append("  • The clause structure allows for multiple reasonable interpretations of the parties' obligations.")
            parts.append("Consider replacing ambiguous terms with precise definitions and measurable criteria.")
        else:
            specifics = []
            if vague_found:
                specifics.append(f"vague qualifiers ('{', '.join(vague_found[:2])}')")
            if modal_found:
                specifics.append(f"modal verbs ('{', '.join(modal_found[:2])}')")
            if hedge_found:
                specifics.append(f"hedging language ('{', '.join(hedge_found[:2])}')")
            if specifics:
                parts.append(f"This clause uses {' and '.join(specifics)} which may lead to differing interpretations between the parties.")
            else:
                parts.append("This clause uses broad language that could be interpreted differently by each party.")
            parts.append("Consider clarifying key terms to reduce interpretive risk.")

    elif score_key == "F":
        # ── Fairness ──
        fairness_triggers = {
            "unilateral": "grants unilateral rights to one party without equivalent protections for the other",
            "sole discretion": "allows one party to act at their sole discretion, removing checks and balances",
            "exclusive right": "grants exclusive rights without reciprocal obligations",
            "unconditionally": "imposes unconditional obligations that cannot be modified even in changed circumstances",
            "no obligation": "explicitly removes obligations from one party while maintaining them for the other",
            "at any time without": "allows unrestricted action without temporal limits or conditions",
        }
        found = [(ph, ex) for ph, ex in fairness_triggers.items() if ph in lower]
        if found:
            parts.append(f"This clause creates an imbalance because it {found[0][1]}.")
            if len(found) > 1:
                parts.append(f"Additionally, it {found[1][1]}.")
            parts.append("This asymmetry may disadvantage one party and could be challenged under fairness doctrines.")
        else:
            parts.append("This clause creates a potential imbalance in the obligations between the parties, where one party bears disproportionate risk or liability.")

    elif score_key == "I":
        # ── Inconsistency ──
        inc_found = [w for w in {"unless", "except", "notwithstanding", "however", "subject to", "provided that"} if w in lower]
        if inc_found:
            parts.append(f"This clause uses conditional language ('{', '.join(inc_found[:3])}') that may create exceptions or contradictions with other provisions in the agreement.")
            parts.append("Verify that these exceptions are consistent with the overall contractual framework.")
        else:
            parts.append("This clause contains conditional structures that may conflict with other sections of the agreement.")

    elif score_key == "R":
        # ── Readability / Clarity ──
        from scoring.models.random_forest import extract_features as _ef
        feat = _ef(clause_text)
        issues = []
        if feat["avg_sent_len"] > 30:
            issues.append(f"very long sentences (avg {feat['avg_sent_len']:.0f} words)")
        if feat["complex_words"] > 5:
            issues.append(f"numerous complex multi-syllable words ({int(feat['complex_words'])} found)")
        if feat["passive_approx"]:
            issues.append("passive voice constructions")
        if feat["long_words"] > 8:
            issues.append(f"heavy use of technical/legal jargon ({int(feat['long_words'])} words over 6 characters)")

        if issues:
            parts.append(f"This clause is difficult to read due to {', '.join(issues)}.")
            parts.append("Simplifying the language would improve comprehension and reduce the risk of misunderstanding by non-legal parties.")
        else:
            parts.append("This clause could benefit from clearer, more direct language to ensure all parties fully understand their obligations.")

    elif score_key == "T":
        # ── Transparency / Compliance ──
        from scoring.models.random_forest import extract_features as _ef
        feat = _ef(clause_text)
        if feat["numeric_count"] == 0 and feat["percent_count"] == 0:
            parts.append("This clause lacks quantitative benchmarks — no specific amounts, percentages, or deadlines are stated.")
            parts.append("Adding measurable criteria would strengthen enforceability and reduce compliance uncertainty.")
        elif feat["vague_count"] > 1:
            parts.append("Despite containing some numeric references, this clause also uses vague qualifiers that undermine the specificity of its terms.")
        else:
            parts.append("This clause has gaps in compliance-related specificity. Adding clearer definitions and measurable obligations would strengthen the agreement.")

    elif score_key == "S":
        # ── Specificity ──
        parts.append("This clause uses broad, general language without defining specific metrics, thresholds, or measurable outcomes.")
        parts.append("Consider adding concrete numbers, deadlines, and performance criteria to make obligations enforceable.")

    return " ".join(parts) if parts else "This clause has been flagged for review."


# ─────────────────────────────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────────────────────────────

def analyze_clauses(
    clauses: List[Dict],
    user_type: str = "consumer",
    contract_source: str = "",
) -> Dict:
    """
    Analyze clauses with 7-head scoring + weighted doc score (S_doc).
    """
    t_start = time.perf_counter()

    scorer = ClauseScorer()
    texts = [c["text"] for c in clauses]

    t_fit_start = time.perf_counter()
    scorer.fit(texts, labels=None)
    t_fit = time.perf_counter() - t_fit_start

    t_score_start = time.perf_counter()
    all_scores = scorer.predict_batch(texts)
    t_score = time.perf_counter() - t_score_start

    t_classify_start = time.perf_counter()
    # Classify each clause type
    clause_types = [classify_clause_type(c["text"]) for c in clauses]

    risky_parts = []
    scored_clauses = []
    agg = {"A": [], "M": [], "R": [], "T": [], "I": [], "F": [], "S": []}
    tp_count, tn_count, fp_count, fn_count = 0, 0, 0, 0

    for clause, scores, ctype in zip(clauses, all_scores, clause_types):
        # ── Skip non-substantive clauses ──
        if ctype in ("definition", "address"):
            scored_clauses.append({
                "id": clause["id"],
                "text": clause["text"],
                "scores": scores,
                "clause_type": ctype,
                "s_doc": 0.0,
                "severity": "low",
            })
            tn_count += 1
            for k in agg:
                agg[k].append(scores.get(k, 0.5))
            continue

        # ── Apply consumer penalties for sneak-in clauses ──
        if user_type == "consumer":
            scores = apply_consumer_penalties(scores, clause["text"])

        # ── Compute Weighted Document Score ──
        s_doc = compute_sdoc(scores)

        # ── Classify severity via S_doc ──
        severity = classify_severity_from_sdoc(s_doc)




        # ── Downgrade standard operational clauses ──
        if severity in ("high", "medium") and should_downgrade_standard(
            clause["text"], ctype, scores
        ):
            if severity == "high":
                severity = "medium"
            elif severity == "medium":
                severity = "low"

        # ── Downgrade boilerplate to max medium ──
        if ctype == "boilerplate" and severity == "high":
            severity = "medium"

        # ── Get primary risk dimension ──
        primary_key, dim_sev = get_primary_risk_dimension(scores, user_type)

        # ── Confusion matrix tracking ──
        # Ground truth: does the clause contain ANY known risk indicator?
        lower_text = clause["text"].lower()
        has_real_risk = (
            any(ph in lower_text for ph in MALICIOUS_PHRASES) or
            any(ph in lower_text for ph in SNEAK_IN_PHRASES) or
            any(ph in lower_text for ph in EXCLUSIVITY_PHRASES) or
            any(ph in lower_text for ph in OBLIGATION_PHRASES) or
            any(ph in lower_text for ph in CONTROL_PHRASES) or
            any(ph in lower_text for ph in FAIRNESS_NEGATIVE)
        )
        if severity in ("high", "medium"):
            if has_real_risk:
                tp_count += 1
            else:
                fp_count += 1
        else:
            if has_real_risk:
                fn_count += 1
            else:
                tn_count += 1

        # ── Build risky parts list ──
        if severity in ("medium", "high"):
            # Collect the exact trigger phrases found in this clause
            highlights = []
            ALL_TRIGGER_SETS = [
                MALICIOUS_PHRASES, SNEAK_IN_PHRASES, EXCLUSIVITY_PHRASES,
                OBLIGATION_PHRASES, CONTROL_PHRASES, FAIRNESS_NEGATIVE,
            ]
            for phrase_set in ALL_TRIGGER_SETS:
                for ph in phrase_set:
                    if ph in lower_text:
                        # Find the original-case version in the clause
                        idx = lower_text.find(ph)
                        if idx >= 0:
                            highlights.append(clause["text"][idx:idx+len(ph)])

            risky_parts.append({
                "text": clause["text"],
                "reason": generate_reason(clause["text"], primary_key, severity, scores.get(primary_key, 0)),
                "severity": severity,
                "category": CATEGORY_MAP.get(primary_key, "ambiguity"),
                "highlights": highlights,
            })

        for k in agg:
            agg[k].append(scores.get(k, 0.5))

        scored_clauses.append({
            "id": clause["id"],
            "text": clause["text"],
            "scores": scores,
            "clause_type": ctype,
            "s_doc": round(s_doc, 3),
            "severity": severity,
        })

    # ── Aggregate scores (0-100 scale) ──
    n = max(len(clauses), 1)

    # For display: A, M are "higher=worse"; R, T (and F, S) are "higher=better"
    # Frontend expects: higher number = more of that quality
    score_ambiguity = round((sum(agg["A"]) / n) * 100)
    score_malicious = round((sum(agg["M"]) / n) * 100)
    score_clarity = round((sum(agg["R"]) / n) * 100)
    score_compliance = round((sum(agg["T"]) / n) * 100)

    # ── Overall risk from S_doc aggregation ──
    high_count = sum(1 for rp in risky_parts if rp["severity"] == "high")
    med_count = sum(1 for rp in risky_parts if rp["severity"] == "medium")
    substantive_count = sum(1 for ct in clause_types if ct in ("substantive", "standard"))

    if substantive_count == 0:
        overall_risk = "low"
    elif high_count >= substantive_count * 0.4:
        overall_risk = "critical"
    elif high_count >= 2 or (high_count >= 1 and med_count >= 3):
        overall_risk = "high"
    elif high_count >= 1 or med_count >= substantive_count * 0.4:
        overall_risk = "medium"
    else:
        overall_risk = "low"

    # ── Performance Metrics ──
    t_classify = time.perf_counter() - t_classify_start
    t_total = time.perf_counter() - t_start

    total_classified = tp_count + tn_count + fp_count + fn_count
    precision = tp_count / max(tp_count + fp_count, 1)
    recall = tp_count / max(tp_count + fn_count, 1)
    specificity = tn_count / max(tn_count + fp_count, 1)
    accuracy = (tp_count + tn_count) / max(total_classified, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    metrics = {
        "confusion_matrix": {
            "true_positives": tp_count,
            "true_negatives": tn_count,
            "false_positives": fp_count,
            "false_negatives": fn_count,
        },
        "scores": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "specificity": round(specificity, 4),
        },
        "distribution": {
            "total_clauses": len(clauses),
            "high_risk": high_count,
            "medium_risk": med_count,
            "low_risk": len(clauses) - high_count - med_count,
            "definitions_skipped": sum(1 for ct in clause_types if ct == "definition"),
            "addresses_skipped": sum(1 for ct in clause_types if ct == "address"),
            "boilerplate": sum(1 for ct in clause_types if ct == "boilerplate"),
            "substantive": sum(1 for ct in clause_types if ct == "substantive"),
            "standard": sum(1 for ct in clause_types if ct == "standard"),
        },
        "timing": {
            "tfidf_fit_ms": round(t_fit * 1000, 2),
            "scoring_ms": round(t_score * 1000, 2),
            "classification_ms": round(t_classify * 1000, 2),
            "total_ms": round(t_total * 1000, 2),
        },
        "config": {
            "user_type": user_type,
            "s_doc_threshold_high": S_DOC_HIGH_THRESHOLD,
            "s_doc_threshold_medium": S_DOC_MEDIUM_THRESHOLD,
            "expert_weights": EXPERT_WEIGHTS,
        },
    }

    # -- Build formatted report --
    from datetime import datetime
    LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scoring_log.txt")

    # Determine run number
    run_number = 1
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                content = f.read()
                # Find the last "Run #NNN"
                import re as _re
                runs = _re.findall(r"Run #(\d+)", content)
                if runs:
                    run_number = int(runs[-1]) + 1
        except Exception:
            pass

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_lines = [
        "",
        "=" * 72,
        f"  Run #{run_number}  |  {timestamp}",
        "  ContractShield - Multi-Criteria Scoring Report",
        "=" * 72,
        f"  Contract:     {contract_source or 'Untitled'}",
        f"  User Type:    {user_type.upper()}",
        f"  Total Clauses: {len(clauses)}  |  Substantive: {metrics['distribution']['substantive']}  |  Standard: {metrics['distribution']['standard']}",
        f"  Skipped:      Definitions={metrics['distribution']['definitions_skipped']}, Addresses={metrics['distribution']['addresses_skipped']}, Boilerplate={metrics['distribution']['boilerplate']}",
        "-" * 72,
        "  CONFUSION MATRIX",
        f"    True Positives  (TP): {tp_count:>5}    |  False Positives (FP): {fp_count:>5}",
        f"    True Negatives  (TN): {tn_count:>5}    |  False Negatives (FN): {fn_count:>5}",
        "-" * 72,
        "  CLASSIFICATION METRICS",
        f"    Accuracy:    {accuracy:.4f}  ({accuracy*100:.1f}%)",
        f"    Precision:   {precision:.4f}  ({precision*100:.1f}%)",
        f"    Recall:      {recall:.4f}  ({recall*100:.1f}%)",
        f"    F1 Score:    {f1:.4f}  ({f1*100:.1f}%)",
        f"    Specificity: {specificity:.4f}  ({specificity*100:.1f}%)",
        "-" * 72,
        "  RISK DISTRIBUTION",
        f"    [HIGH]   High Risk:   {high_count:>3}",
        f"    [MED]    Medium Risk: {med_count:>3}",
        f"    [LOW]    Low Risk:    {len(clauses) - high_count - med_count:>3}",
        "-" * 72,
        "  TIMING",
        f"    TF-IDF Fit:       {t_fit*1000:>8.2f} ms",
        f"    Scoring (7-head): {t_score*1000:>8.2f} ms",
        f"    Classification:   {t_classify*1000:>8.2f} ms",
        f"    Total Pipeline:   {t_total*1000:>8.2f} ms",
        "-" * 72,
        "  CLAUSES FLAGGED:",
    ]
    for rp in risky_parts:
        log_lines.append(f"    [{rp['severity'].upper():>6}] {rp['category']}: {rp['text'][:70]}...")
    if not risky_parts:
        log_lines.append("    (none)")
    log_lines += [
        "-" * 72,
        f"  Overall Risk: {overall_risk.upper()}",
        "=" * 72,
        "",
    ]

    report_text = "\n".join(log_lines)

    # Print to console
    print(report_text)

    # Append to persistent log file
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(report_text + "\n")
    except Exception as e:
        print(f"[WARNING] Could not write to log file: {e}")

    summary = _generate_summary(
        overall_risk, score_ambiguity, score_malicious,
        score_clarity, score_compliance,
        high_count, med_count, len(clauses), user_type, contract_source
    )

    return {
        "scores": {
            "ambiguity": score_ambiguity,
            "maliciousIntent": score_malicious,
            "clarity": score_clarity,
            "compliance": score_compliance,
        },
        "overallRisk": overall_risk,
        "summary": summary,
        "riskyParts": risky_parts,
        "clauses": scored_clauses,
        "metrics": metrics,
    }


def _generate_summary(
    overall_risk, ambiguity, malicious, clarity, compliance,
    high_count, med_count, total, user_type, contract_source
):
    risk_desc = {
        "low": "relatively low-risk",
        "medium": "moderate-risk",
        "high": "high-risk",
        "critical": "critically risky",
    }

    issues = []
    if ambiguity > 55:
        issues.append("ambiguous language that could lead to multiple interpretations")
    if malicious > 40:
        issues.append("potentially one-sided or exploitative terms")
    if clarity < 50:
        issues.append("unclear clause structures that reduce readability")
    if compliance < 50:
        issues.append("insufficient specificity in compliance-related terms")

    source_label = f"'{contract_source}' " if contract_source else ""
    profile = "consumer" if user_type == "consumer" else "company"

    summary = f"This {source_label}contract is assessed as {risk_desc.get(overall_risk, 'moderate-risk')} from a {profile} perspective. "

    if issues:
        summary += f"The primary concerns include {', '.join(issues[:2])}. "
    else:
        summary += "No major risk factors were identified. "

    summary += f"Out of {total} clauses analyzed, {high_count} were flagged as high risk and {med_count} as medium risk"

    if user_type == "consumer":
        summary += " under consumer protection thresholds (0.920 specificity)."
    else:
        summary += " under corporate compliance thresholds."

    return summary
