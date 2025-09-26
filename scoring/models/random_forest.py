from typing import List, Dict, Optional
import re
import math
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

MODAL_WORDS = {"may","might","could","should","would","can","shall","must"}
HEDGING_WORDS = {"may","might","could","seems","appears","suggests","possible","possibly","likely","unlikely","approximate","approximately"}
VAGUE_WORDS = {"some","several","various","etc","etc."}
MALICIOUS_PHRASES = {
    "at our sole discretion", "without notice", "hold harmless", "indemnif", 
    "no liability", "not liable", "not be liable", "sole discretion", "as we see fit",
    "we may", "we will not", "we are not", "waive", "waiver", "forfeit", "penalty"
}
INCONSISTENCY_TRIGGERS = {"unless","except","notwithstanding","however","but","contrary","subject to"}
DEFINITIONS_TRIGGERS = {"means","defined as","definition","shall mean"}

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
    n_words = len(words)
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
    mal_count = sum(1 for ph in MALICIOUS_PHRASES if ph in clause.lower())
    inc_count = sum(1 for ph in INCONSISTENCY_TRIGGERS if ph in clause.lower())
    def_count = sum(1 for ph in DEFINITIONS_TRIGGERS if ph in clause.lower())
    numeric_count = len(re.findall(r'\b\d+(?:[\.,]\d+)?\b', clause))
    percent_count = len(re.findall(r'\b\d+%|\b\d+\spercent', clause.lower()))
    passive_approx = int(bool(re.search(r'\b(be|is|are|was|were|been|being)\b', clause.lower()) and re.search(r'\b\w+ed\b', clause.lower())))
    pronouns = {"it","they","them","their","its","this","that","these","those","here","there"}
    pronoun_count = sum(1 for w in words if w in pronouns)
    features = {
        "n_words": n_words, "avg_word_len": avg_word_len, "avg_sent_len": avg_sent_len,
        "long_words": long_words, "complex_words": complex_words, "caps_words": caps_words,
        "modal_count": modal_count, "hedge_count": hedge_count, "vague_count": vague_count,
        "mal_count": mal_count, "inc_count": inc_count, "def_count": def_count,
        "numeric_count": numeric_count, "percent_count": percent_count, "passive_approx": passive_approx,
        "pronoun_count": pronoun_count, "sentence_count": n_sent
    }
    return features


def heuristic_scores(features: Dict[str,float]) -> Dict[str,float]:
    A_raw = 0.5*features["modal_count"] + 0.6*features["hedge_count"] + 0.4*features["vague_count"] + \
            0.3*features["pronoun_count"] + 0.6*features["passive_approx"] + 0.02*max(0, features["avg_sent_len"]-20)
    M_raw = 1.5*features["mal_count"] + 0.8*(features["caps_words"]>3) + 0.3*max(0,5-features["def_count"])
    R_raw = -0.03*features["avg_sent_len"] -0.05*features["complex_words"] -0.2*features["passive_approx"] -0.02*features["long_words"]
    T_raw = 0.6*features["numeric_count"] + 0.8*features["percent_count"] + 0.8*features["def_count"] - 0.4*features["vague_count"]
    I_raw = 0.7*features["inc_count"] + 0.2*features["modal_count"] + 0.3*features["hedge_count"] + 0.02*features["sentence_count"]

    def squash(x): return 1.0 / (1.0 + math.exp(-0.6*(x-0.5)))
    A = float(squash(A_raw))
    M = float(squash(M_raw))
    R = float(squash(-R_raw))
    T = float(squash(T_raw))
    I = float(squash(I_raw))
    return {"A": A, "M": M, "R": R, "T": T, "I": I}

class ClauseScorer:
    def __init__(self):
        self.tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
        self.models = {}
        self.feature_names = None

    def fit(self, clauses: List[str], labels: Optional[List[Dict[str,float]]]=None):
        X_meta = [extract_features(c) for c in clauses]
        df_meta = pd.DataFrame(X_meta).fillna(0)
        self.feature_names = df_meta.columns.tolist()
        X_tfidf = self.tfidf.fit_transform(clauses)
        X = hstack([X_tfidf, df_meta.values])
        if labels is None:
            print("No labels provided — using heuristics only.")
            return
        Ys = {k: np.array([lab.get(k, np.nan) for lab in labels], dtype=float) for k in ["A","M","R","T","I"]}
        for k, y in Ys.items():
            mask = ~np.isnan(y)
            if mask.sum() < 5:
                print(f"Skipping {k}, not enough labeled examples")
                continue
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X[mask], y[mask])
            self.models[k] = rf
            print(f"Trained model for {k} on {mask.sum()} examples.")

    def predict(self, clause: str) -> Dict[str,float]:
        feat = extract_features(clause)
        heur = heuristic_scores(feat)
        X = hstack([self.tfidf.transform([clause]), np.array([list(feat.values())])])
        preds = {}
        for k in ["A","M","R","T","I"]:
            if k in self.models:
                raw = self.models[k].predict(X)[0]
                preds[k] = float(min(1.0,max(0.0,raw)))
            else:
                preds[k] = heur[k]
        return preds

    def predict_batch(self, clauses: List[str]) -> List[Dict[str,float]]:
        return [self.predict(c) for c in clauses]

if __name__ == "__main__":
    sample_json = [
        {"id":1,"clause":"The Company may terminate this Agreement at any time without prior notice."},
        {"id":2,"clause":"Payment of INR 10,000 shall be made within 30 days from invoice date."},
        {"id":3,"clause":"The Contractor will use reasonable efforts to perform the Services."},
        {"id":4,"clause":"All disputes shall be finally settled by arbitration in accordance with the Rules."},
        {"id":5,"clause":"Unless otherwise provided, the parties agree to abide by the applicable laws."}
    ]
    df_demo = pd.DataFrame(sample_json)
    scorer = ClauseScorer()
    scorer.fit(df_demo["clause"].tolist(), labels=None)
    scores = scorer.predict_batch(df_demo["clause"].tolist())
    df_scores = pd.DataFrame(scores)
    result = pd.concat([df_demo, df_scores], axis=1)
    print(result.to_string(index=False))
    result.to_json("clause_scores_demo.json", orient="records", indent=2)
    print("Scored clauses saved to clause_scores_demo.json")
