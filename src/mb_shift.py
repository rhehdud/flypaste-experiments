#!/usr/bin/env python3
"""새 포켓몬·기술이 풀렸을 때: 옛 환경으로 학습 -> 새 환경으로 시험. 초파리(성격·EV 도파민 학습)는 mb_learn과 같은 방식.

새 환경에는 옛 환경에 없던 종족·기술·아이템·특성이 있다. 시험 입력 두 가지를 같은 학습 결과로 비교한다.
  이름 그대로: 처음 보는 이름의 투사뉴런도 자극
  처음 보는 이름 뺌: 학습 데이터(옛 환경)에 한 번도 안 나온 특징은 자극하지 않는다
새 환경 개체를 새 종족 / 새 기술(종족은 앎) / 새 아이템·특성만 / 전부 아는 이름으로 나눠 채점한다.
새 환경 개체의 Kenyon cell 코드는 개체마다 새 반응 (mb_stream과 같은 out/kc_codes_*_new_rows_seed*.npz). 옛 환경과 입력이 같아도 학습 코드를
다시 쓰지 않는다 (잡음까지 같으면 점수가 부풀려진다). 처음 보는 이름을 뺀 입력도 같은 개체의 새 반응(같은 시드)이라
뺄 이름이 없는 개체는 원래 새 반응 코드와 같고, 뺀 개체만 새로 계산해 out/kc_codes_*_new_rows_seen_only_seed*.npz에 캐시.
--same-noise면 학습 코드를 재사용하고, 이름 뺀 입력은 기존 캐시에 없는 것만 고유 입력마다 한 번 계산한다 (out/kc_codes_*_seen_only.npz).
예측 규칙: 성격 확률만큼 맥락(최빈 / 기대 오차 최소), 확정 성격 맥락, 완전히 함께, 성격 앎(정답 성격 맥락, 최빈 / 기대 오차 최소).

  python src/mb_shift.py                # 새 반응 코드
  python src/mb_shift.py --same-noise   # 학습 코드 재사용, 비교용
"""
import argparse
import pathlib
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from mb import DURATION_MS, ENCOUNTER_SEED0, PN_MIN_KC_SYNAPSES, PNS_PER_FEATURE, input_pns, kc_code, kc_codes, kc_row_codes, load_mb, pn_channels
from mb_learn import ETAS, N_JOBS, context_rate, kc_mbon_synapses, mb_ev_job, mb_nature_job, nature_output_setup
from mb_task import EV, NATURE, NEW_SOURCE, OLD_SOURCE, features, load_mons
from spread import nature_bits, paste_scores

METHODS = ["성격 확률만큼 맥락", "성격 확률만큼 맥락, 기대 오차 최소", "확정 성격 맥락", "완전히 함께", "성격 앎", "성격 앎, 기대 오차 최소"]


def seen_only_row_codes(X, tr, te, enc_codes, W, groups, cache):
    """시험 개체에서 학습 때 안 나온 특징을 뺀 입력의 개체별 새 반응 코드 (len(te), Kenyon cell 수)와 뺀 특징 수.
    시드는 원래 입력의 새 반응과 같다(ENCOUNTER_SEED0 + 행). 뺄 특징이 없으면 입력·시드가 같아 enc_codes(원래 입력 새 반응) 그대로."""
    seen = np.asarray(X[tr].sum(0)).ravel() > 0
    X_seen = (X @ sp.diags(seen.astype(np.float64))).tocsr()
    X_seen.eliminate_zeros()
    X_seen.sort_indices()
    n_dropped = X[te].getnnz(axis=1) - X_seen[te].getnnz(axis=1)
    codes = enc_codes.copy()
    k = n_dropped > 0
    codes[k] = kc_row_codes(X_seen, te[k], W, groups, cache)
    return codes, n_dropped


def seen_only_codes(X, W, groups, tr, te, codes, cache):
    """--same-noise: 시험 개체에서 학습 때 안 나온 특징을 뺀 입력의 Kenyon cell 코드 (len(te), Kenyon cell 수).
    학습 캐시에 같은 입력이 있으면 그 코드를 재사용."""
    rows = [tuple(np.sort(X[i].indices)) for i in range(X.shape[0])]
    uniq = sorted(set(rows))
    known = {u: codes[j] for j, u in enumerate(uniq)}
    seen = np.asarray(X[tr].sum(0)).ravel() > 0
    masked = [tuple(f for f in rows[i] if seen[f]) for i in te]
    new = sorted(set(masked) - set(known))

    cache = pathlib.Path(cache)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(map(tuple, z["inputs"])) == new:
            known.update(zip(new, z["codes"]))
            new = []
    if new:
        ch = pn_channels(X.shape[1], input_pns(W, groups))
        t0, got, step = time.time(), [], max(1, len(new) // 20)
        jobs = (delayed(kc_code)(W, groups, ch, list(u), seed=len(uniq) + k) for k, u in enumerate(new))
        for k, c in enumerate(Parallel(n_jobs=N_JOBS, return_as="generator")(jobs), 1):
            got.append(c)
            if k % step == 0 or k == len(new):
                print(f"  처음 보는 이름 뺀 입력 Kenyon cell 코드 {k:,}/{len(new):,} (경과 {(time.time() - t0) / 60:.1f}분)", flush=True)
        got = np.stack(got).astype(np.uint16)
        np.savez_compressed(cache, codes=got, inputs=np.array(new, dtype=object))
        known.update(zip(new, got))
    return np.stack([known[m] for m in masked]), np.array([len(r) - len(m) for r, m in zip((rows[i] for i in te), masked)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--same-noise", action="store_true", help="새 환경 개체도 학습 코드를 재사용 (비교용)")
    ap.add_argument("--select", choices=["error", "logloss"], default="error",
                    help="학습률·반복 수 고르는 검증 채점 (mb_learn --select와 같음). logloss면 예측 파일 이름 끝 _sel-logloss")
    ap.add_argument("--etas", default=",".join(f"{e:g}" for e in ETAS), help="학습률 후보, 쉼표로 (mb_learn --etas와 같음)")
    ap.add_argument("--decay-rare", type=float, default=0.0,
                    help="드물게 쓰이는 Kenyon cell만 감쇠: 학습 데이터에서 그 Kenyon cell을 켜는 개체 수가 "
                         "중앙값보다 적으면 이 값으로, 많으면 비례해 약하게 감쇠 (--decay와 배타)")
    ap.add_argument("--decay", type=float, default=0.0,
                    help="시냅스 감쇠: 개체 하나를 배울 때마다 Kenyon cell->출력 가중치가 (1-decay)배. "
                         "드물게 나오는 특징의 가중치가 덜 쌓인다 (0이면 지금까지와 같음)")
    args = ap.parse_args()
    if args.decay:
        print(f"시냅스 감쇠 {args.decay:g} (반감기 {np.log(2) / args.decay:,.0f}마리)", flush=True)
    etas = sorted((float(x) for x in args.etas.split(",")), reverse=True)
    mo = load_mons()
    arch = mo.archetype_key.to_numpy()
    X, vocab = features(mo)
    Y = mo[EV].to_numpy()
    nat = mo[NATURE].to_numpy()
    natures, nat_id = np.unique(nat, return_inverse=True)
    bits = nature_bits(nat)
    tr, te = np.where(mo.source == OLD_SOURCE)[0], np.where(mo.source == NEW_SOURCE)[0]

    seen = np.asarray(X[tr].sum(0)).ravel() > 0
    kind = np.array([v[:2] for v in vocab])
    Xte = X[te].tocsr()
    has = lambda k: np.asarray(Xte[:, np.where(~seen & (kind == k))[0]].sum(1)).ravel() > 0
    new_sp, new_mv, new_other = has("sp"), has("mv"), has("it") | has("ab")
    subgroup = np.select([new_sp, new_mv, new_other], ["새 종족", "새 기술(종족은 앎)", "새 아이템·특성만"], "전부 아는 이름")
    print(f"학습 옛 환경 {len(tr):,}마리, 시험 새 환경 {len(te):,}마리 | 옛 환경에 없는 특징: 종족 {(~seen & (kind == 'sp')).sum()}, "
          f"기술 {(~seen & (kind == 'mv')).sum()}, 아이템 {(~seen & (kind == 'it')).sum()}, 특성 {(~seen & (kind == 'ab')).sum()}")
    print("새 환경 개체 분류:", pd.Series(subgroup).value_counts().to_dict(), flush=True)

    W, groups = load_mb()
    tag = f"{DURATION_MS}ms_pn{PN_MIN_KC_SYNAPSES}x{PNS_PER_FEATURE}"
    codes, inv = kc_codes(X, W, groups, f"out/kc_codes_real_{tag}.npz")
    per_row = codes[inv]
    if args.same_noise:
        masked, n_dropped = seen_only_codes(X, W, groups, tr, te, codes, f"out/kc_codes_real_{tag}_seen_only.npz")
    else:  # 새 환경 행은 개체마다 새 반응 (mb_stream과 같은 코드). 옛 환경 학습은 원래 코드 그대로
        per_row = per_row.copy()
        per_row[te] = kc_row_codes(X, te, W, groups, f"out/kc_codes_real_{tag}_new_rows_seed{ENCOUNTER_SEED0}.npz")
        masked, n_dropped = seen_only_row_codes(X, tr, te, per_row[te], W, groups,
                                                f"out/kc_codes_real_{tag}_new_rows_seen_only_seed{ENCOUNTER_SEED0}.npz")
    print(f"새 환경 코드: {'학습 코드 재사용 (같은 잡음)' if args.same_noise else '개체마다 새 반응'} | "
          f"처음 보는 이름 뺀 개수: {pd.Series(n_dropped).value_counts().sort_index().to_dict()}", flush=True)

    # 원래 개체 N행 뒤에 시험 개체의 "이름 뺀" 코드를 붙인다. 학습은 앞쪽 옛 환경 행만 쓰므로 학습 결과는 하나
    n = len(mo)
    rates = sp.vstack([sp.csr_matrix(per_row.astype(np.float64)), sp.csr_matrix(masked.astype(np.float64))]).tocsr()
    rates = rates / (DURATION_MS / 1000)
    S = kc_mbon_synapses(W, groups)
    conn, axes = nature_output_setup(S, natures)
    te_all = np.concatenate([te, n + np.arange(len(te))])

    t0 = time.time()
    decay = args.decay
    if args.decay_rare:
        c = np.asarray((rates[tr] > 0).sum(0)).ravel().astype(float)   # 학습 개체 몇 마리가 이 Kenyon cell을 켜나
        med = np.median(c[c > 0])
        decay = args.decay_rare * np.minimum(1.0, med / np.maximum(c, 1.0))
        print(f"드문 시냅스만 감쇠 {args.decay_rare:g} | Kenyon cell을 켜는 개체 수 중앙값 {med:.0f}, "
              f"감쇠가 붙는 Kenyon cell {(decay > args.decay_rare / 2).sum():,}/{len(c):,}", flush=True)
    val_logp, te_logp, fit_n = mb_nature_job(rates, nat_id, len(natures), conn, arch, tr, te_all, "옛 환경 성격", axes=axes, etas=etas, select=args.select, decay=decay)
    print(f"[1/2] 성격({fit_n['kind']}) 학습 끝 (eta={fit_n['eta']:g}, {fit_n['epochs']}회, 경과 {(time.time() - t0) / 60:.1f}분)", flush=True)
    soft, soft_min_err, score, spreads, fit_e, known_min_err, _, _ = mb_ev_job(rates, Y, bits, natures, val_logp, te_logp, arch, tr, te_all,
                                                                         context_rate(rates, tr), "옛 환경 EV", etas=etas, te_nat_id=np.tile(nat_id[te], 2),
                                                                         select=args.select, decay=decay)
    print(f"[2/2] EV 학습 끝 (eta={fit_e['eta']:g}, {fit_e['epochs']}회, 경과 {(time.time() - t0) / 60:.1f}분)", flush=True)

    rows, saved = {}, []
    for label, sl in [("이름 그대로", slice(0, len(te))), ("처음 보는 이름 뺌", slice(len(te), 2 * len(te)))]:
        lp, first = te_logp[sl], te_logp[sl].argmax(1)
        joint = (lp + score[sl]).argmax(1)
        rr = np.arange(len(te))
        preds = {"성격 확률만큼 맥락": (soft[sl], natures[first]),
                 "성격 확률만큼 맥락, 기대 오차 최소": (soft_min_err[sl], natures[first]),
                 "확정 성격 맥락": (spreads[sl][rr, first], natures[first]),
                 "완전히 함께": (spreads[sl][rr, joint], natures[joint]),
                 "성격 앎": (spreads[sl][rr, nat_id[te]], nat[te]),
                 "성격 앎, 기대 오차 최소": (known_min_err[sl], nat[te])}
        for how in METHODS:
            ev, na = preds[how]
            for g in ["전체", "새 종족", "새 기술(종족은 앎)", "새 아이템·특성만", "전부 아는 이름"]:
                m = np.ones(len(te), bool) if g == "전체" else subgroup == g
                if m.any():
                    rows[(label, how, g)] = {"개체": m.sum(), **paste_scores(ev[m], na[m], Y[te][m], nat[te][m])}
            saved.append(pd.DataFrame({"row": te, "input": label, "method": how, "subgroup": subgroup, "nature": na,
                                       **{c: ev[:, s] for s, c in enumerate(EV)}}))
    out = (f"out/mb_shift_preds_{tag}{'' if args.same_noise else '_encounter'}{'' if not args.decay else f'_decay{args.decay:g}'}{'' if not args.decay_rare else f'_rare{args.decay_rare:g}'}{'' if etas == ETAS else '_eta' + '_'.join(f'{e:g}' for e in etas)}"
           f"{'' if args.select == 'error' else '_sel-' + args.select}.parquet")
    pd.concat(saved, ignore_index=True).to_parquet(out)

    table = pd.DataFrame(rows).T
    cols = ["EV 오차", "EV 오차(성격 맞힘)", "EV 오차(성격 틀림)", "EV 완전일치", "세부 값 정확도", "성격 정확도", "EV·성격 완전일치",
            "맞춤 배분 예측", "어색한 조합"]
    fmt = {c: ("{:.2f}".format if c.startswith("EV 오차") else "{:.1%}".format) for c in cols}
    for how in METHODS:
        t = table.xs(how, level=1)
        print(f"\n=== 옛 환경 학습 -> 새 환경 시험: {how} ===")
        print(pd.concat([t["개체"].astype(int), t[cols].astype(float)], axis=1).to_string(formatters=fmt))
    print(f"\n시험 예측 저장: {out}")


if __name__ == "__main__":
    main()
