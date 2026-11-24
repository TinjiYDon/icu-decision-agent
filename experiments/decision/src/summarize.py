"""Print a compact summary of a metrics_*.json file (console convenience only).

Usage: python experiments/decision/src/summarize.py [tag]
"""
import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"


def main(tag="full"):
    m = json.loads((RES / f"metrics_{tag}.json").read_text(encoding="utf-8"))
    print(f"== metrics_{tag}.json  test n={m['meta']['n_test']} pos={m['meta']['n_test_pos']} "
          f"prev={m['meta']['test_prevalence']:.4f}")
    hdr = f"{'model':4s} {'AUPRC':>7s} {'95%CI':>17s} {'AUROC':>7s} {'95%CI':>17s} " \
          f"{'Brier':>8s} {'ECE15':>7s} {'slope':>6s} {'sens@50':>7s}"
    print(hdr)
    for k, v in m["models"].items():
        ci = v["ci"]
        s50 = v["budgets"][1]["sensitivity"]
        s50s = f"{s50:7.3f}" if s50 is not None else f"{'n/a':>7s}"
        print(f"{k:4s} {v['auprc']:7.4f} [{ci['auprc']['lo']:6.4f},{ci['auprc']['hi']:6.4f}] "
              f"{v['auroc']:7.4f} [{ci['auroc']['lo']:6.4f},{ci['auroc']['hi']:6.4f}] "
              f"{v['brier']:8.5f} {v['ece15']:7.4f} {v['cal_slope']:6.3f} {s50s}")
    print("\n-- comparisons")
    for k, c in m["comparisons"].items():
        d = c["auprc_delta"]
        da = c["auroc_delta"]
        print(f"{k:10s} dAUPRC {d['abs']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}] "
              f"p_boot={d['p_two_sided']:.4f} | dAUROC {da['abs']:+.4f} "
              f"[{da['lo']:+.4f},{da['hi']:+.4f}] DeLong p={c['delong']['p']:.3g} | "
              f"dBrier {c['brier_delta_abs']:+.5f} dECE {c['ece_delta_abs']:+.4f}")
    print("\n-- budget sensitivity (per 1000 stay-hours)")
    for k, v in m["models"].items():
        s = " ".join(f"{b['budget_per_1000']}:" + (f"{b['sensitivity']:.3f}" if b['sensitivity'] is not None else "n/a") for b in v["budgets"])
        print(f"{k:4s} {s}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "full")
