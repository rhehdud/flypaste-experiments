#!/usr/bin/env python3
"""mb_stream.py가 저장한 예측(out/mb_stream_preds_*.parquet)으로 종족 단위 5겹 교차 평가를 한다 (학습 없음).

한 번 나눈 종족 절반(새 종족 25종)으로는 0.1~0.5점 차이를 가르기 어렵다.
새 환경 종족을 새 종족·익숙한 종족 각각 5묶음으로 나누고, 묶음마다 나머지 4묶음의 EV 오차로 두 구획 τ를 골라 그 묶음을 채점한다.
모든 종족이 "고를 때 안 쓴 채점"을 한 번씩 받는다. 묶음 나누기를 N_REPEATS번 반복해 평균.
차이의 구간은 종족 단위 부트스트랩(나누기 반복을 평균한 개체별 차이).

  python src/mb_stream.py        # 먼저 (예측 저장)
  python src/mb_stream_cv.py     # 새 반응 코드 예측 (기본). --same-noise면 학습 코드를 재사용한 예측
τ는 최빈 배분 EV 오차로 고르고, 예측에 있으면 기대 오차 최소·성격 앎 EV 오차도 같은 τ로 채점한다.
"""
import argparse

import numpy as np
import pandas as pd

from mb_stream import PRED_KINDS, VARIANTS, stream_path
from mb_task import EV, NATURE, OLD_SOURCE, err, features, load_mons
from spread import odd_combo

N_FOLDS = 5
N_REPEATS = 20
N_BOOT = 2000
BASE = "한 번씩"


def species_boot(d, sp, mask, rng):
    u, t = np.unique(sp[mask], return_inverse=True)
    w = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=N_BOOT)[:, t]
    b = (w * d[mask]).sum(1) / w.sum(1)
    return d[mask].mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), len(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--same-noise", action="store_true", help="mb_stream --same-noise 예측으로 평가")
    same_noise = ap.parse_args().same_noise
    mo = load_mons()
    X, vocab = features(mo)
    X = X.tocsr()
    tr = np.where(mo.source == OLD_SOURCE)[0]
    kind = np.array([v[:2] for v in vocab])
    sp_feat = np.array([[f for f in X.indices[X.indptr[i]:X.indptr[i + 1]] if kind[f] == "sp"][0] for i in range(len(mo))])
    counts = np.asarray(X[tr].sum(0)).ravel()

    res = pd.read_parquet(stream_path(same_noise))
    print(f"예측: {stream_path(same_noise)}")
    res["ev_err"] = err(res[EV].to_numpy(), mo[EV].to_numpy()[res.row])
    extra = [pre for pre in list(PRED_KINDS)[1:] if pre + EV[0] in res]
    for pre in extra:
        res[f"ev_err_{pre}"] = err(res[[pre + c for c in EV]].to_numpy(), mo[EV].to_numpy()[res.row])
    metrics = ["ev_err", *[f"ev_err_{pre}" for pre in extra], "nature_hit", "odd"]
    res["nature_hit"] = (res.nature.to_numpy() == mo[NATURE].to_numpy()[res.row]).astype(float)
    res["odd"] = odd_combo(res[EV].to_numpy(), res.nature.to_numpy()).astype(float)
    per = res.groupby(["variant", "row"])[metrics].mean()  # 순서 평균
    rows = per.loc[BASE].index.to_numpy()
    sp, new = sp_feat[rows], counts[sp_feat[rows]] == 0
    duals = [v for v in VARIANTS if VARIANTS[v]["learn"] == "dual" and v in per.index.get_level_values(0)]
    others = [v for v in VARIANTS if VARIANTS[v]["learn"] != "dual" and v in per.index.get_level_values(0)]
    E = {v: per.loc[v].reindex(rows) for v in others + duals}

    picked, cv = [], {m: np.zeros(len(rows)) for m in metrics}
    for r in range(N_REPEATS):
        rng = np.random.default_rng(r)
        fold = np.zeros(len(rows), int)
        for is_new in (True, False):
            spp = np.unique(sp[new == is_new])
            f_of = dict(zip(spp[rng.permutation(len(spp))], np.arange(len(spp)) % N_FOLDS))
            m = new == is_new
            fold[m] = [f_of[s] for s in sp[m]]
        for k in range(N_FOLDS):
            out = fold == k
            best = min(duals, key=lambda v: E[v].ev_err.to_numpy()[~out].mean())  # 나머지 4묶음 개체 평균
            picked.append(best)
            for m in cv:
                cv[m][out] += E[best][m].to_numpy()[out] / N_REPEATS
    cvname = "두 구획 (설정을 나머지 묶음으로 선택)"
    E[cvname] = pd.DataFrame(cv, index=rows)

    print(f"새 환경 종족 {N_FOLDS}겹 교차 평가 x {N_REPEATS}번 | 새 종족 {len(np.unique(sp[new]))}종 {new.sum()}마리, "
          f"익숙한 종족 {len(np.unique(sp[~new]))}종 {(~new).sum()}마리")
    print("뽑힌 두 구획 횟수: " + ", ".join(f"{v} {c}" for v, c in pd.Series(picked).value_counts().items()))
    groups = {"전체": np.ones(len(rows), bool), "새 종족": new, "익숙한 종족": ~new}
    names = others + [cvname] + duals
    shown = ([("ev_err", "EV 오차 (최빈)", "{:.2f}")] + [(f"ev_err_{pre}", f"EV 오차 ({PRED_KINDS[pre]})", "{:.2f}") for pre in extra]
             + [("nature_hit", "성격 정확도", "{:.1%}"), ("odd", "어색한 조합", "{:.1%}")])
    for metric, label, f in shown:
        t = pd.DataFrame({g: {v: E[v][metric].to_numpy()[m].mean() for v in names} for g, m in groups.items()})
        print(f"\n[{label}] (고정 τ 줄은 참고: 전체 새 환경으로 본 값)")
        print(t.to_string(float_format=f.format))

    rng = np.random.default_rng(0)
    print(f"\n[{BASE} 대비 차이, 종족 단위 부트스트랩 95% 구간]")
    for v in [v for v in [cvname, "새로움(패턴 0.5)"] if v in E]:
        cells = []
        for g, m in groups.items():
            e, lo, hi, n = species_boot((E[v].ev_err - E[BASE].ev_err).to_numpy(), sp, m, rng)
            h, hlo, hhi, _ = species_boot((E[v].nature_hit - E[BASE].nature_hit).to_numpy(), sp, m, rng)
            cell = f"{g}({n}종) EV {e:+.2f} [{lo:+.2f}, {hi:+.2f}] 성격 {h:+.1%} [{hlo:+.1%}, {hhi:+.1%}]"
            for pre in extra:
                d = (E[v][f"ev_err_{pre}"] - E[BASE][f"ev_err_{pre}"]).to_numpy()
                x, xlo, xhi, _ = species_boot(d, sp, m, rng)
                cell += f" | {PRED_KINDS[pre]} {x:+.2f} [{xlo:+.2f}, {xhi:+.2f}]"
            cells.append(cell)
        print(f"  {v}:\n    " + "\n    ".join(cells))


if __name__ == "__main__":
    main()
