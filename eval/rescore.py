"""
eval/rescore.py
Re-score saved results.json against updated metric functions without re-running the agent.
"""
import json, os, sys, re

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")


def metric_uncertainty_acknowledged(answer, q_type):
    if q_type != "insufficient_evidence":
        return "N/A"
    signals = [
        "cannot", "unclear", "insufficient", "uncertain", "unable", "no information",
        "not known", "speculative", "not confirmed", "unknown", "lacks", "lack sufficient",
        "not publicly", "not available", "suggest", "further research", "follow-up",
        "not yet known", "has not been", "not disclosed", "not announced",
    ]
    return 1 if any(s in answer.lower() for s in signals) else 0


def metric_conflict_flagged(answer, q_type):
    if q_type != "conflict":
        return "N/A"
    signals = [
        "disagree", "conflict", "contradict", "however, another", "varies",
        "different sources", "while", "whereas", "on the other hand",
        "some sources", "other sources", "reported differently", "not agree",
        "discrepancy", "inconsistent",
    ]
    return 1 if any(s in answer.lower() for s in signals) else 0


def metric_citation_rate(answer):
    return 1 if re.search(r"\[\d+\]", answer) else 0


def metric_source_diversity(citation_map):
    if not citation_map:
        return 0
    domains = {v.get("domain", "") for v in citation_map.values() if v.get("domain")}
    return len(domains)


def metric_answer_length_ok(answer):
    wc = len(answer.split())
    return 1 if 50 <= wc <= 600 else 0


with open(RESULTS_PATH, encoding='utf-8') as f:
    data = json.load(f)

rows = []
for item in data["results"]:
    answer = item.get("answer", "")
    q_type = item["type"]
    citation_map = item.get("citation_map", {})

    cite = metric_citation_rate(answer)
    div = metric_source_diversity(citation_map)
    unc = metric_uncertainty_acknowledged(answer, q_type)
    conf = metric_conflict_flagged(answer, q_type)
    length = metric_answer_length_ok(answer)

    item["metrics"] = {
        "citation_rate": cite,
        "source_diversity": div,
        "uncertainty_acknowledged": unc,
        "conflict_flagged": conf,
        "answer_length_ok": length,
    }
    rows.append((item["id"], q_type, cite, div, unc, conf, length))

# Save updated results
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# Print table
print("\n" + "=" * 70)
print(f"{'ID':<5} {'Type':<22} {'Cite':<6} {'Div':<5} {'Unc':<5} {'Conf':<6} {'Len':<5}")
print("-" * 70)
for row in rows:
    qid, qtype, cite, div, unc, conf, length = row
    print(f"{qid:<5} {qtype:<22} {cite:<6} {div:<5} {str(unc):<5} {str(conf):<6} {length:<5}")
print("=" * 70)

def avg_numeric(vals):
    nums = [v for v in vals if v != "N/A"]
    return sum(nums) / len(nums) if nums else 0.0

cite_avg = avg_numeric([r[2] for r in rows])
div_avg = avg_numeric([r[3] for r in rows])
unc_avg = avg_numeric([r[4] for r in rows])
conf_avg = avg_numeric([r[5] for r in rows])
len_avg = avg_numeric([r[6] for r in rows])

print(f"\nAGGREGATES: Cite={cite_avg:.2f} | Diversity={div_avg:.2f} | "
      f"Uncertainty={unc_avg:.2f} | Conflict={conf_avg:.2f} | Length OK={len_avg:.2f}")
print(f"\nUpdated results saved to: {RESULTS_PATH}")
