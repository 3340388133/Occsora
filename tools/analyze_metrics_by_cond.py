#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def _percentile(xs: List[float], q: float) -> float:
    xs = [x for x in xs if _is_finite(x)]
    if not xs:
        return float("nan")
    xs.sort()
    if q <= 0:
        return xs[0]
    if q >= 100:
        return xs[-1]
    pos = (len(xs) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1 - w) + xs[hi] * w


@dataclass
class CondRow:
    cond: int
    miou: float
    cd: float
    occ_ratio: float
    fid: float
    fvd: float


def load_by_cond(path: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for method, cond_map in (raw or {}).items():
        if not isinstance(cond_map, dict):
            continue
        fixed: Dict[int, Dict[str, Any]] = {}
        for k, v in cond_map.items():
            try:
                cond = int(k)
            except Exception:
                continue
            if isinstance(v, dict):
                fixed[cond] = v
        out[str(method)] = fixed
    return out


def collect_rows(method_map: Dict[int, Dict[str, Any]]) -> List[CondRow]:
    rows: List[CondRow] = []
    for cond, m in sorted(method_map.items(), key=lambda kv: kv[0]):
        rows.append(
            CondRow(
                cond=cond,
                miou=_to_float(m.get("miou")),
                cd=_to_float(m.get("cd")),
                occ_ratio=_to_float(m.get("physics_occ_ratio", m.get("occ_ratio"))),
                fid=_to_float(m.get("fid")),
                fvd=_to_float(m.get("fvd")),
            )
        )
    return rows


def summarize(rows: List[CondRow], empty_occ_thr: float, zero_miou_thr: float) -> Dict[str, Any]:
    miou = [r.miou for r in rows]
    cd = [r.cd for r in rows]
    occ = [r.occ_ratio for r in rows]
    fid = [r.fid for r in rows]
    fvd = [r.fvd for r in rows]

    def frac(pred: Iterable[bool]) -> float:
        pred_l = list(pred)
        return float(sum(pred_l)) / float(len(pred_l)) if pred_l else float("nan")

    empty_occ = [(_is_finite(r.occ_ratio) and r.occ_ratio < empty_occ_thr) for r in rows]
    zero_miou = [(_is_finite(r.miou) and r.miou <= zero_miou_thr) for r in rows]
    bad_cd = [(_is_finite(r.cd) and r.cd >= 900.0) for r in rows]

    return {
        "n": len(rows),
        "occ_empty_frac": frac(empty_occ),
        "miou_zero_frac": frac(zero_miou),
        "cd_ge_900_frac": frac(bad_cd),
        "miou_p50": _percentile(miou, 50),
        "miou_p10": _percentile(miou, 10),
        "miou_p90": _percentile(miou, 90),
        "occ_p50": _percentile(occ, 50),
        "occ_p10": _percentile(occ, 10),
        "occ_p90": _percentile(occ, 90),
        "cd_p50": _percentile(cd, 50),
        "cd_p10": _percentile(cd, 10),
        "cd_p90": _percentile(cd, 90),
        "fid_p50": _percentile(fid, 50),
        "fvd_p50": _percentile(fvd, 50),
    }


def top_anomalies(
    rows: List[CondRow],
    empty_occ_thr: float,
    zero_miou_thr: float,
    topk: int,
) -> List[CondRow]:
    # 排序规则：优先全空，其次mIoU接近0，其次CD大
    def score(r: CondRow) -> Tuple[int, float, float]:
        empty = int(_is_finite(r.occ_ratio) and r.occ_ratio < empty_occ_thr)
        zero = int(_is_finite(r.miou) and r.miou <= zero_miou_thr)
        cd = r.cd if _is_finite(r.cd) else float("inf")
        # empty/zero 越大越异常，cd 越大越异常
        return (empty + zero, -r.miou if _is_finite(r.miou) else 0.0, cd)

    ranked = sorted(rows, key=score, reverse=True)
    return ranked[: max(0, int(topk))]


def fmt(x: float, nd: int = 6) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf"
    return f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze metrics_by_cond.json for collapse/empty-occupancy symptoms")
    ap.add_argument("--json", required=True, help="Path to metrics_by_cond.json")
    ap.add_argument("--methods", default="baseline,stca,sads,full", help="Comma-separated methods to show")
    ap.add_argument("--empty-occ-thr", type=float, default=1e-6, help="occ_ratio below this treated as empty")
    ap.add_argument("--zero-miou-thr", type=float, default=0.0, help="mIoU <= this treated as zero")
    ap.add_argument("--topk", type=int, default=8, help="Show top-K anomalous conditions per method")
    args = ap.parse_args()

    path = os.path.abspath(args.json)
    data = load_by_cond(path)
    want = [m.strip() for m in (args.methods or "").split(",") if m.strip()]

    print(f"[analyze] file: {path}")
    print(f"[analyze] methods: {', '.join(want) if want else '(all)'}")
    print(f"[analyze] empty_occ_thr={args.empty_occ_thr:g}  zero_miou_thr={args.zero_miou_thr:g}")

    methods = want if want else sorted(data.keys())
    for method in methods:
        if method not in data:
            print(f"\n== {method} == (missing)")
            continue

        rows = collect_rows(data[method])
        s = summarize(rows, empty_occ_thr=args.empty_occ_thr, zero_miou_thr=args.zero_miou_thr)

        print(f"\n== {method} ==")
        print(
            "n={n} | occ_empty={oe:.1%} | miou_zero={mz:.1%} | cd>=900={c9:.1%}".format(
                n=s["n"], oe=s["occ_empty_frac"], mz=s["miou_zero_frac"], c9=s["cd_ge_900_frac"]
            )
        )
        print(
            "p50: miou={miou}  occ={occ}  cd={cd}  fid={fid}  fvd={fvd}".format(
                miou=fmt(s["miou_p50"], 6),
                occ=fmt(s["occ_p50"], 6),
                cd=fmt(s["cd_p50"], 4),
                fid=fmt(s["fid_p50"], 4),
                fvd=fmt(s["fvd_p50"], 4),
            )
        )
        print(
            "p10/p90: miou={a}/{b}  occ={c}/{d}  cd={e}/{f}".format(
                a=fmt(s["miou_p10"], 6),
                b=fmt(s["miou_p90"], 6),
                c=fmt(s["occ_p10"], 6),
                d=fmt(s["occ_p90"], 6),
                e=fmt(s["cd_p10"], 4),
                f=fmt(s["cd_p90"], 4),
            )
        )

        bad = top_anomalies(rows, empty_occ_thr=args.empty_occ_thr, zero_miou_thr=args.zero_miou_thr, topk=args.topk)
        if bad:
            print("top anomalies: cond  miou  occ_ratio  cd  fid  fvd")
            for r in bad:
                print(
                    f"  {r.cond:>4d}  {fmt(r.miou, 6):>10}  {fmt(r.occ_ratio, 8):>12}  {fmt(r.cd, 4):>10}"
                    f"  {fmt(r.fid, 4):>8}  {fmt(r.fvd, 4):>8}"
                )


if __name__ == "__main__":
    main()
