"""Parse scoring_log.txt and compute average metrics across all runs."""
import re

with open("scoring_log.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Parse all runs
runs = re.split(r"={72}", content)

metrics_all = []
for block in runs:
    acc_m = re.search(r"Accuracy:\s+([\d.]+)\s+\(([\d.]+)%\)", block)
    prec_m = re.search(r"Precision:\s+([\d.]+)\s+\(([\d.]+)%\)", block)
    rec_m = re.search(r"Recall:\s+([\d.]+)\s+\(([\d.]+)%\)", block)
    f1_m = re.search(r"F1 Score:\s+([\d.]+)\s+\(([\d.]+)%\)", block)
    spec_m = re.search(r"Specificity:\s+([\d.]+)\s+\(([\d.]+)%\)", block)
    tp_m = re.search(r"True Positives\s+\(TP\):\s+(\d+)", block)
    tn_m = re.search(r"True Negatives\s+\(TN\):\s+(\d+)", block)
    fp_m = re.search(r"False Positives\s+\(FP\):\s+(\d+)", block)
    fn_m = re.search(r"False Negatives\s+\(FN\):\s+(\d+)", block)
    run_m = re.search(r"Run #(\d+)", block)
    contract_m = re.search(r"Contract:\s+(.+?)$", block, re.MULTILINE)
    user_m = re.search(r"User Type:\s+(\w+)", block)
    clauses_m = re.search(r"Total Clauses:\s+(\d+)", block)
    high_m = re.search(r"High Risk:\s+(\d+)", block)
    med_m = re.search(r"Medium Risk:\s+(\d+)", block)
    low_m = re.search(r"Low Risk:\s+(\d+)", block)
    
    # Timing
    tfidf_m = re.search(r"TF-IDF Fit:\s+([\d.]+)\s+ms", block)
    scoring_m = re.search(r"Scoring \(7-head\):\s+([\d.]+)\s+ms", block)
    class_m = re.search(r"Classification:\s+([\d.]+)\s+ms", block)
    total_m = re.search(r"Total Pipeline:\s+([\d.]+)\s+ms", block)

    if acc_m and prec_m and rec_m and f1_m and spec_m:
        metrics_all.append({
            "run": int(run_m.group(1)) if run_m else 0,
            "contract": contract_m.group(1).strip() if contract_m else "?",
            "user_type": user_m.group(1) if user_m else "?",
            "clauses": int(clauses_m.group(1)) if clauses_m else 0,
            "accuracy": float(acc_m.group(1)),
            "precision": float(prec_m.group(1)),
            "recall": float(rec_m.group(1)),
            "f1": float(f1_m.group(1)),
            "specificity": float(spec_m.group(1)),
            "tp": int(tp_m.group(1)) if tp_m else 0,
            "tn": int(tn_m.group(1)) if tn_m else 0,
            "fp": int(fp_m.group(1)) if fp_m else 0,
            "fn": int(fn_m.group(1)) if fn_m else 0,
            "high": int(high_m.group(1)) if high_m else 0,
            "med": int(med_m.group(1)) if med_m else 0,
            "low": int(low_m.group(1)) if low_m else 0,
            "tfidf_ms": float(tfidf_m.group(1)) if tfidf_m else 0,
            "scoring_ms": float(scoring_m.group(1)) if scoring_m else 0,
            "class_ms": float(class_m.group(1)) if class_m else 0,
            "total_ms": float(total_m.group(1)) if total_m else 0,
        })

print(f"Total runs parsed: {len(metrics_all)}")
print()

# ── Per-run table ──
print("=" * 120)
print(f"{'Run':>4} | {'Contract':<30} | {'Type':<8} | {'Cl':>3} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'Spec':>6} | {'TP':>3} {'TN':>3} {'FP':>3} {'FN':>3}")
print("-" * 120)
for m in metrics_all:
    print(f"{m['run']:>4} | {m['contract'][:30]:<30} | {m['user_type']:<8} | {m['clauses']:>3} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['specificity']:.4f} | {m['tp']:>3} {m['tn']:>3} {m['fp']:>3} {m['fn']:>3}")

print("=" * 120)
print()

# ── Averages across ALL runs ──
n = len(metrics_all)
avg_acc = sum(m["accuracy"] for m in metrics_all) / n
avg_prec = sum(m["precision"] for m in metrics_all) / n
avg_rec = sum(m["recall"] for m in metrics_all) / n
avg_f1 = sum(m["f1"] for m in metrics_all) / n
avg_spec = sum(m["specificity"] for m in metrics_all) / n

total_tp = sum(m["tp"] for m in metrics_all)
total_tn = sum(m["tn"] for m in metrics_all)
total_fp = sum(m["fp"] for m in metrics_all)
total_fn = sum(m["fn"] for m in metrics_all)

# Micro-averaged metrics (from pooled confusion matrix)
micro_prec = total_tp / max(total_tp + total_fp, 1)
micro_rec = total_tp / max(total_tp + total_fn, 1)
micro_f1 = 2 * micro_prec * micro_rec / max(micro_prec + micro_rec, 1e-9)
micro_acc = (total_tp + total_tn) / max(total_tp + total_tn + total_fp + total_fn, 1)
micro_spec = total_tn / max(total_tn + total_fp, 1)

print("╔══════════════════════════════════════════════════════════════════╗")
print("║         MACRO-AVERAGED METRICS (Mean of all 36 runs)           ║")
print("╠══════════════════════════════════════════════════════════════════╣")
print(f"║  Accuracy:      {avg_acc:.4f}  ({avg_acc*100:.1f}%)                            ║")
print(f"║  Precision:     {avg_prec:.4f}  ({avg_prec*100:.1f}%)                            ║")
print(f"║  Recall:        {avg_rec:.4f}  ({avg_rec*100:.1f}%)                            ║")
print(f"║  F1 Score:      {avg_f1:.4f}  ({avg_f1*100:.1f}%)                            ║")
print(f"║  Specificity:   {avg_spec:.4f}  ({avg_spec*100:.1f}%)                            ║")
print("╚══════════════════════════════════════════════════════════════════╝")
print()

print("╔══════════════════════════════════════════════════════════════════╗")
print("║     MICRO-AVERAGED METRICS (Pooled Confusion Matrix)           ║")
print("╠══════════════════════════════════════════════════════════════════╣")
print(f"║  Pooled TP: {total_tp:>5}   |   Pooled FP: {total_fp:>5}                    ║")
print(f"║  Pooled TN: {total_tn:>5}   |   Pooled FN: {total_fn:>5}                    ║")
print(f"║                                                                ║")
print(f"║  Accuracy:      {micro_acc:.4f}  ({micro_acc*100:.1f}%)                            ║")
print(f"║  Precision:     {micro_prec:.4f}  ({micro_prec*100:.1f}%)                            ║")
print(f"║  Recall:        {micro_rec:.4f}  ({micro_rec*100:.1f}%)                            ║")
print(f"║  F1 Score:      {micro_f1:.4f}  ({micro_f1*100:.1f}%)                            ║")
print(f"║  Specificity:   {micro_spec:.4f}  ({micro_spec*100:.1f}%)                            ║")
print("╚══════════════════════════════════════════════════════════════════╝")
print()

# ── Exclude Run 18 (0 positives — no risk clauses present) for fairer average ──
valid_runs = [m for m in metrics_all if m["tp"] + m["fn"] > 0]
n_valid = len(valid_runs)
v_acc = sum(m["accuracy"] for m in valid_runs) / n_valid
v_prec = sum(m["precision"] for m in valid_runs) / n_valid
v_rec = sum(m["recall"] for m in valid_runs) / n_valid
v_f1 = sum(m["f1"] for m in valid_runs) / n_valid
v_spec = sum(m["specificity"] for m in valid_runs) / n_valid

print("╔══════════════════════════════════════════════════════════════════╗")
print(f"║  MACRO-AVG (excl. Run #18 — no positives) [{n_valid} runs]         ║")
print("╠══════════════════════════════════════════════════════════════════╣")
print(f"║  Accuracy:      {v_acc:.4f}  ({v_acc*100:.1f}%)                            ║")
print(f"║  Precision:     {v_prec:.4f}  ({v_prec*100:.1f}%)                            ║")
print(f"║  Recall:        {v_rec:.4f}  ({v_rec*100:.1f}%)                            ║")
print(f"║  F1 Score:      {v_f1:.4f}  ({v_f1*100:.1f}%)                            ║")
print(f"║  Specificity:   {v_spec:.4f}  ({v_spec*100:.1f}%)                            ║")
print("╚══════════════════════════════════════════════════════════════════╝")
print()

# ── Timing averages ──
avg_tfidf = sum(m["tfidf_ms"] for m in metrics_all) / n
avg_scoring = sum(m["scoring_ms"] for m in metrics_all) / n
avg_class = sum(m["class_ms"] for m in metrics_all) / n
avg_total = sum(m["total_ms"] for m in metrics_all) / n

print("╔══════════════════════════════════════════════════════════════════╗")
print("║              AVERAGE TIMING (across 36 runs)                   ║")
print("╠══════════════════════════════════════════════════════════════════╣")
print(f"║  TF-IDF Fit:        {avg_tfidf:>8.2f} ms                              ║")
print(f"║  Scoring (7-head):  {avg_scoring:>8.2f} ms                              ║")
print(f"║  Classification:    {avg_class:>8.2f} ms                              ║")
print(f"║  Total Pipeline:    {avg_total:>8.2f} ms                              ║")
print("╚══════════════════════════════════════════════════════════════════╝")
print()

# ── Per unique contract type averages ──
from collections import defaultdict
by_type = defaultdict(list)
for m in metrics_all:
    key = f"{m['contract']} ({m['user_type']}, {m['clauses']}cl)"
    by_type[key].append(m)

print("╔══════════════════════════════════════════════════════════════════╗")
print("║         AVERAGES BY CONTRACT TYPE                              ║")
print("╠══════════════════════════════════════════════════════════════════╣")
for key, runs_list in by_type.items():
    n_r = len(runs_list)
    a = sum(m["accuracy"] for m in runs_list) / n_r
    p = sum(m["precision"] for m in runs_list) / n_r
    r = sum(m["recall"] for m in runs_list) / n_r
    f = sum(m["f1"] for m in runs_list) / n_r
    s = sum(m["specificity"] for m in runs_list) / n_r
    print(f"║  {key[:55]:<55}  ║")
    print(f"║    Runs={n_r}  Acc={a*100:.1f}%  Prec={p*100:.1f}%  Rec={r*100:.1f}%  F1={f*100:.1f}%  Spec={s*100:.1f}%  ║")
    print(f"║                                                                ║")
print("╚══════════════════════════════════════════════════════════════════╝")
