#!/usr/bin/env python3
"""저장된 최종 모델을 **고정한 채** TEST를 예측만 해서 지표를 내고 `out/test/metrics_{TAG}.json`에 저장한다.

표나 그림에 쓰는 숫자는 이 JSON에서 읽는다. 숫자를 다른 코드에 적어 두지 않는다 —
설정이나 데이터가 바뀌면 조용히 낡기 때문이다.

**TEST 취급.** 가중치를 갱신하지 않고, 이 숫자로 설계·설정값을 고르지 않는다 (η는 따로 고른 값).
읽기는 `load_test_mons()`로만. 실행할 때마다 TEST를 한 번 더 보는 것이므로 횟수를 기록한다.

**채점 대상.** TEST 전체에서 **입력도 답도 학습 데이터와 같은 개체를 뺀다** (외울 수 있는 중복).

**비교 줄.** `입력 x 학습 규칙` 네 칸으로 읽는다. 종족 최빈 배분(암기표)이 바닥이다.

  OTS 특징    + 로지스틱 회귀   초파리를 아예 쓰지 않는다
  버섯체 코드 + 로지스틱 회귀   배선으로 코드는 만들고, 읽는 것은 로지스틱
  버섯체 코드 + 도파민 학습     기본 방법

버섯체 코드 줄이 OTS 줄보다 좋으면 배선이 만든 희소 코드가 기여한 것이고,
도파민 줄이 버섯체 코드 줄보다 좋으면 학습 규칙이 기여한 것이다. --no-compare로 로지스틱을 뺀다.

  python src/mb_final.py          # 먼저 한 번 (최종 모델 저장)
  python src/mb_test_metrics.py
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from mb import (DURATION_MS, RATE_HZ, TAG, TEAM_ENCOUNTER_SEED0, TEAM_RATE_HZ, TEAM_SEED0, TEST_SEED0, kc_codes,
                kc_row_codes, load_mb, model_path)
from mb_task import EV, NATURE, TEST_DIR, features, load_mons, load_test_mons, logreg_classes, team_union_features
from model_io import load_model
from spread import STATS, decode_ev, ev_given_nature, fine_value_hits, odd_combo, paste_scores, predict

ETA = 3.0  # 다른 데이터에서 고른 값. TEST로 고르지 않는다


def species_mode_spread(train, species):
    """학습 데이터에서 그 종족의 최빈 EV 배분 (학습에 없는 종족은 학습 전체 최빈). 두 갈래 공통 기준선."""
    gm = train[EV].value_counts().index[0]
    mode = train.groupby("species")[EV].apply(lambda d: d.value_counts().index[0])
    return np.array([mode.get(s, gm) for s in species])


def per_stat(ev, Y):
    """스탯별 평균 절대오차와 치우침(예측 − 정답). 하나로 퉁치지 않고 어느 스탯에서 새는지 본다."""
    return np.abs(ev - Y).mean(0), (ev - Y).mean(0)


def logreg_rows(train, test, X_train, X_test, vocab_all, rates_test, natt, W, groups, model, args):
    """같은 과제를 로지스틱 회귀가 읽을 때 -> {줄 이름: (EV 배분, 성격)}.

    두 벌을 낸다. OTS 특징을 그대로 읽는 것과, 버섯체가 만든 Kenyon cell 코드를 읽는 것.
    초파리와 나란히 놓으면 이득이 코드에서 온 것인지 도파민 규칙에서 온 것인지 갈린다.
    학습은 학습 데이터만 쓴다 (`A[tr]`, `Y[tr]`, `nat[tr]`). TEST 정답은 성격 맥락으로만 들어간다.
    """
    n_tr, n_te = len(train), len(test)
    tr, te = np.arange(n_tr), np.arange(n_tr, n_tr + n_te)
    Y_all = np.vstack([train[EV].to_numpy(), test[EV].to_numpy()])
    nat_all = np.concatenate([train[NATURE].to_numpy(), natt])

    # OTS 특징: 초파리와 정보량을 맞춘다 (그 마리 + 같은 팀 5마리를 tm_ 접두어로).
    # 투사뉴런 배정과 무관하므로 모델 vocab을 따르지 않고 여기서 새로 만든다.
    _, v0 = features(train, team="all")
    X_te_ots, v_all = features(test, team="all", vocab=v0)   # TEST에만 있는 이름은 뒤에 붙는다
    X_tr_ots, _ = features(train, team="all", vocab=v_all)   # 그 이름까지 열을 맞춰 다시 만든다
    A_ots = sp.vstack([sp.csr_matrix(X_tr_ots), sp.csr_matrix(X_te_ots)]).tocsr()

    # 버섯체 코드: 학습 쪽은 고유 입력마다 한 번 (mb_final과 같은 캐시)
    codes, inv = kc_codes(X_train, W, groups, f"out/kc_codes_real_{TAG}.npz", n_jobs=args.jobs)
    rates_tr = codes.astype(np.float64)[inv] / (DURATION_MS / 1000)
    if bool(model["team"]):
        Xt_tr = team_union_features(train, X_train, list(model["vocab"]))
        tc, tinv = kc_codes(Xt_tr, W, groups, f"out/kc_codes_real_{TAG}_team{TEAM_RATE_HZ}Hz.npz",
                            seed0=TEAM_SEED0, rate=TEAM_RATE_HZ, n_jobs=args.jobs)
        rates_tr = np.hstack([rates_tr, tc.astype(np.float64)[tinv] / (DURATION_MS / 1000)])
    A_kc = sp.csr_matrix(np.vstack([rates_tr, rates_test]))

    out = {}
    for label, A in [("OTS 특징", A_ots), ("버섯체 코드", A_kc)]:
        t0 = time.time()
        print(f"  [{label} + 로지스틱 회귀] 성격 학습 중 (입력 열 {A.shape[1]:,}개)...", flush=True)
        na_lr = logreg_classes(A, nat_all, tr, te)
        print(f"    성격 끝 ({time.time() - t0:.0f}초). EV 6스탯 학습 중...", flush=True)
        # nat_preds가 목록이라 한 번 학습으로 두 갈래를 다 낸다 (성격 주어짐 / 스스로 예측)
        ev_known, ev_pred = ev_given_nature(A, Y_all, nat_all, tr, te, [natt, na_lr])
        print(f"    EV 끝 (합계 {(time.time() - t0) / 60:.1f}분)", flush=True)
        out[f"{label} + 로지스틱 회귀 · 성격을 알 때"] = (ev_known, natt)
        out[f"{label} + 로지스틱 회귀 · 성격을 모를 때"] = (ev_pred, na_lr)
    return out


def nature_prf(pred, true, natures):
    rows = []
    for n in natures:
        tp = int(((pred == n) & (true == n)).sum())
        fp = int(((pred == n) & (true != n)).sum())
        fn = int(((pred != n) & (true == n)).sum())
        if tp + fn == 0 and tp + fp == 0:
            continue
        p = tp / (tp + fp) if tp + fp else np.nan
        r = tp / (tp + fn) if tp + fn else np.nan
        f = 2 * p * r / (p + r) if p and r and (p + r) else 0.0
        rows.append({"성격": n, "정답 수": tp + fn, "예측 수": tp + fp, "정밀도": p, "재현율": r, "F1": f})
    return pd.DataFrame(rows).sort_values("정답 수", ascending=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="모델 파일 (기본 설정의 이름)")
    ap.add_argument("--eta", type=float, default=ETA, help=f"EV 고르는 규칙의 세기 (기본 {ETA:g})")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--team-code", choices=["encounter", "unique"], default="encounter",
                    help="TEST 팀 자극 코드 뽑기. encounter(기본, mb.py가 적어 둔 시험용 규약) = 개체마다 새 반응. "
                         "unique = 학습처럼 고유 팀 입력마다 한 번")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="자극 잡음을 다른 뽑기로 (기본 0). 같은 모델·같은 채점 집합에서 "
                         "잡음만 바꿔 지표가 얼마나 흔들리는지 재는 용도")
    ap.add_argument("--no-compare", action="store_true",
                    help="로지스틱 회귀 비교 줄을 빼고 초파리와 종족 최빈만 낸다")
    ap.add_argument("--out", default=None, help=f"지표 JSON (기본 {TEST_DIR}/metrics_{TAG}.json)")
    args = ap.parse_args()
    off = args.seed_offset * 10_000
    sfx = "" if not off else f"_off{args.seed_offset}"
    model = load_model(args.model or model_path())
    out_json = args.out or f"{TEST_DIR}/metrics_{TAG}{sfx}.json"

    train, test = load_mons(), load_test_mons()
    assert not set(test.team_id) & set(train.team_id), "TEST 팀이 학습에 섞여 있다"
    vocab = list(model["vocab"])
    X_train, _ = features(train, vocab=vocab)
    X_test, vocab_all = features(test, vocab=vocab)  # 학습 열 번호 유지, TEST에만 있는 이름은 뒤에
    Yt, natt = test[EV].to_numpy(), test[NATURE].to_numpy()
    print(f"모델 {args.model or model_path()} (학습 {', '.join(map(str, model['train_sources']))} "
          f"{int(model['n_train']):,}마리 / {int(model['n_teams']):,}팀) | TEST {len(test)}마리 {test.team_id.nunique()}팀 "
          f"| TEST에만 있는 특징 이름 {len(vocab_all) - len(vocab)}개", flush=True)

    # --- 입력도 답도 학습과 같은 개체를 뺀다 (외울 수 있는 중복)
    train_ans = {}
    for i in range(X_train.shape[0]):
        train_ans.setdefault(tuple(np.sort(X_train[i].indices)), set()).add(
            (train[NATURE].iat[i], tuple(train[EV].iloc[i])))
    trows = [tuple(np.sort(X_test[i].indices)) for i in range(X_test.shape[0])]
    dup = np.array([r in train_ans and (na, tuple(y)) in train_ans[r] for r, na, y in zip(trows, natt, Yt)])
    keep = ~dup
    new_sp = ~test.species.isin(set(train.species)).to_numpy()
    print(f"채점 대상: {int(keep.sum())}마리 (입력·답이 학습과 같은 중복 {int(dup.sum())}마리 제외, "
          f"{dup.mean():.1%}) | 그중 처음 보는 종족 {int(new_sp[keep].sum())}마리 ({new_sp[keep].mean():.1%})", flush=True)

    # --- 고정 모델로 예측만 (개체마다 새 반응, 팀 자극도 개체마다 새로)
    W, groups = load_mb()
    codes = kc_row_codes(X_test, np.arange(len(test)), W, groups,
                         f"{TEST_DIR}/kc_codes_test_{TAG}_rows_seed{TEST_SEED0 + off}.npz",
                         seed0=TEST_SEED0 + off, n_jobs=args.jobs)
    rates = codes.astype(np.float64) / (DURATION_MS / 1000)
    if bool(model["team"]):
        Xt = team_union_features(test, X_test, vocab_all)
        if args.team_code == "unique":
            tc, tinv = kc_codes(Xt, W, groups, f"{TEST_DIR}/kc_codes_test_{TAG}_team{TEAM_RATE_HZ}Hz_uniq{off}.npz",
                                seed0=TEAM_SEED0 + off, rate=TEAM_RATE_HZ, n_jobs=args.jobs)
            tcodes = tc[tinv]
        else:
            tcodes = kc_row_codes(Xt, np.arange(len(test)), W, groups,
                                  f"{TEST_DIR}/kc_codes_test_{TAG}_team{TEAM_RATE_HZ}Hz_rows_seed{TEAM_ENCOUNTER_SEED0 + off}.npz",
                                  seed0=TEAM_ENCOUNTER_SEED0 + off, rate=TEAM_RATE_HZ, n_jobs=args.jobs)
        rates = np.hstack([rates, tcodes.astype(np.float64) / (DURATION_MS / 1000)])
    print(f"입력 열 {rates.shape[1]:,}개 (그 마리 {RATE_HZ}Hz"
          f"{f' + 팀 {TEAM_RATE_HZ}Hz' if bool(model['team']) else ''})", flush=True)

    natures = np.array(model["natures"])
    p_nat, lp_unknown = predict(rates, model)            # 성격도 맞힌다 (성격 확률만큼 맥락)
    _, lp_known = predict(rates, model, given=natt)      # 성격을 확정 맥락으로 주고 EV만
    na_pred = natures[p_nat.argmax(1)]
    base_ev = species_mode_spread(train, test.species.to_numpy())
    unconv = ~(base_ev == Yt).all(1)  # 정답이 종족 최빈 배분과 다른 개체
    base_na = train.groupby("species")[NATURE].agg(lambda s: s.mode().iat[0]).reindex(
        test.species.to_numpy()).fillna(train[NATURE].mode().iat[0]).to_numpy()

    FLY, BASE = "버섯체 코드 + 도파민 학습", "종족 최빈 배분"
    TRACKS = {f"{FLY} · 성격을 알 때": (decode_ev(lp_known, "eta", args.eta), natt),
              f"{FLY} · 성격을 모를 때": (decode_ev(lp_unknown, "eta", args.eta), na_pred)}
    if not args.no_compare:
        TRACKS = {**logreg_rows(train, test, X_train, X_test, vocab_all, rates, natt,
                                W, groups, model, args), **TRACKS}
    ALL = {**TRACKS, BASE: (base_ev, base_na)}
    rows, metrics = {}, {}
    for name, (ev, na) in ALL.items():
        sc = paste_scores(ev[keep], na[keep], Yt[keep], natt[keep])
        rows[name] = {"개체": int(keep.sum()), **sc}
        e_, y_ = ev[keep], Yt[keep]
        metrics[name] = {"ev_error": float(sc["EV 오차"]), "exact": float(sc["EV 완전일치"]),
                         "within10": float((np.abs(e_ - y_).sum(1) / 2 <= 10).mean()),
                         # "어디에 줄지": EV를 준 칸/안 준 칸이 6스탯 모두 정답과 같은 비율
                         "pattern": float(((e_ > 0) == (y_ > 0)).all(1).mean()),
                         # "얼마나 줄지": 정답이 EV를 준 칸 중에서 값까지 정확히 맞힌 비율
                         "given_cell_exact": float((e_[y_ > 0] == y_[y_ > 0]).mean()),
                         # 어색한 조합: 그 방법이 낸 성격 기준(자기 답의 앞뒤가 맞는가) / 정답 성격 기준(둘 다 기록)
                         "odd_combo": float(sc["어색한 조합"]),
                         "odd_combo_vs_true": float(odd_combo(e_, natt[keep]).mean()),
                         "nature_acc": float(sc["성격 정확도"]),
                         "new_species_ev_error": float((np.abs(ev - Yt).sum(1) / 2)[keep & new_sp].mean()),
                         "seen_species_ev_error": float((np.abs(ev - Yt).sum(1) / 2)[keep & ~new_sp].mean()),
                         "new_species_nature_acc": float((na == natt)[keep & new_sp].mean()),
                         "seen_species_nature_acc": float((na == natt)[keep & ~new_sp].mean()),
                         # 정답이 그 종족의 최빈 배분과 다른 개체 = 암기표가 정의상 틀리는 쪽
                         "unconventional_ev_error": float((np.abs(ev - Yt).sum(1) / 2)[keep & unconv].mean())}
    cols = ["EV 오차", "EV 완전일치", "세부 값 정확도", "성격 정확도", "EV·성격 완전일치", "맞춤 배분 예측", "어색한 조합"]
    t = pd.DataFrame(rows).T
    print(f"\n=== TEST {int(keep.sum())}마리, 고정 모델 예측만 (EV 규칙 η={args.eta:g}) ===")
    print(pd.concat([t["개체"].astype(int), t[cols].astype(float)], axis=1).to_string(
        formatters={c: ("{:.2f}".format if c.startswith("EV 오차") else "{:.1%}".format) for c in cols}))

    print("\n--- 스탯별 EV 절대오차 / 치우침(예측−정답) ---")
    for name, (ev, _) in ALL.items():
        mae, bias = per_stat(ev[keep], Yt[keep])
        print(f"{name:34s} " + "  ".join(f"{s} {m:4.1f}/{b:+5.1f}" for s, m, b in zip(STATS, mae, bias)))
    print("\n--- 스탯별 세부 값(정답 3~29) 정확도 ---")
    for name, (ev, _) in ALL.items():
        n_fine, n_hit = fine_value_hits(ev[keep], Yt[keep])
        print(f"{name:34s} " + "  ".join(f"{s} {h/n:5.1%}" if n else f"{s}     -" for s, h, n in zip(STATS, n_hit, n_fine)))
    print("\n--- 처음 보는 종족 / 학습에 있던 종족 EV 오차 ---")
    for name in metrics:
        m = metrics[name]
        print(f"{name:34s} 처음 {m['new_species_ev_error']:5.2f}  아는 종족 {m['seen_species_ev_error']:5.2f}")

    print(f"\n--- 성격별 정밀도·재현율·F1 (성격을 모를 때, {int(keep.sum())}마리) ---")
    print(nature_prf(na_pred[keep], natt[keep], natures).to_string(
        index=False, formatters={c: "{:.1%}".format for c in ("정밀도", "재현율", "F1")}))

    payload = {"tag": TAG, "model": args.model or model_path(), "eta": args.eta,
               "seed_offset": args.seed_offset, "n_scored": int(keep.sum()),
               "n_test": int(len(test)), "n_dup_excluded": int(dup.sum()),
               "n_new_species": int(new_sp[keep].sum()), "n_unconventional": int((keep & unconv).sum()),
               "n_train": int(model["n_train"]),
               "n_teams": int(model["n_teams"]), "train_sources": [str(s) for s in model["train_sources"]],
               "baseline_label": BASE, "tracks": metrics}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n지표 저장: {out_json}  (figs/make_figures.py가 이 파일을 읽는다)")


if __name__ == "__main__":
    sys.exit(main())
