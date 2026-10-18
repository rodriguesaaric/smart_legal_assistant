"""Debug clause 7 scoring."""
from scoring.models.random_forest import extract_features, heuristic_scores

clause = "There shall be no branding on the three audience-facing sides of the venue by any third-party sponsor, festival, or college."
feat = extract_features(clause)
scores = heuristic_scores(feat)
print("Features:")
for k, v in feat.items():
    if v != 0:
        print(f"  {k}: {v}")
print(f"\nScores: {scores}")
print(f"\nExclusivity: {feat.get('exclusivity_count', 0)}")
print(f"Obligation: {feat.get('obligation_count', 0)}")
print(f"Control: {feat.get('control_count', 0)}")
