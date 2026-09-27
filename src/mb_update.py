#!/usr/bin/env python3
"""새 환경 데이터가 조금씩 모일 때 업데이트 방식 비교: 계속 배우기 vs 모아서 한꺼번에 다시 학습.
출처 이름으로 가른다: OLD_SOURCE = 먼저 모인 데이터, NEW_SOURCE = 나중에 모인 데이터 (mb_task.py).

나누기: 새 환경을 archetype 5겹으로. 4겹 = 새로 모인 데이터(학습 몫), 1겹 = 평가 (모든 방식이 안 봄). 5번 돌려 새 환경 전체를 한 번씩 채점.
학습 몫은 팀 단위 무작위 순서(fold마다 시드)의 앞부분 FRACS 비율 (팀 경계에서 자름, 0.1 ⊂ 0.3 ⊂ ... ⊂ 1.0).
같은 fold·같은 비율이면 모든 방식이 같은 페이스트를 배운다.
방식:
  고정: 옛 환경으로만 학습 (비율 0)
  계속 배우기: 옛 환경 학습 -> 학습 몫을 그 순서대로 한 마리씩 (mb_stream.stream_job 그대로, 평가 개체는 배우지 않고 예측만)
    두 구획 τ=0.5 (예측 때 옛 환경 코드와 최대 코사인을 씀) / 한 번씩 (문 없음, 저장 데이터 안 씀)
  한꺼번에: 옛 환경 + 학습 몫을 처음부터 함께 (mb_learn fit_nature·fit_ev, 학습률·반복 수를 안쪽 검증으로 고름)
  새 환경만 한꺼번에: 학습 몫만 (옛 환경 데이터가 돕는지 방해하는지)
코드: 옛 환경 행 = 고유 입력 코드, 새 환경 행(학습 몫·평가) = 개체마다 새 반응 (mb_stream과 같음). 모든 방식이 같은 코드를 쓴다.
예측 기록: 성격 먼저 -> EV (성격 확률만큼 맥락) 최빈 / 기대 오차 최소, 성격 앎 두 규칙. 시험 데이터는 쓰지 않는다.

작업 하나가 끝날 때마다 예측을 out/mb_update_parts_*/에 바로 저장한다 (옛 환경 학습 결과도). 다시 실행하면 저장된 작업은 건너뛴다.
실행 중에도 --report로 지금까지 저장된 작업만으로 부분 표를 볼 수 있다 (비율마다 모든 방식이 가진 평가 개체만 비교).

  python src/mb_update.py            # 학습·예측 -> 작업별 저장 + out/mb_update_preds_*.parquet, 이어서 표
  python src/mb_update.py --report   # 저장된 예측으로 표만 (중간 결과도 가능)
  python src/mb_update.py --teams 5,10,20,30   # 비율 대신 새로 모인 팀 수 (희소한 새 환경). 저장은 *_teams
"""
import argparse
import glob
import os
import time

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from mb import TAG, DURATION_MS, ENCOUNTER_SEED0, PN_MIN_KC_SYNAPSES, PNS_PER_FEATURE, kc_codes, kc_row_codes, load_mb
from mb_learn import context_rate, fit_ev, fit_nature, kc_mbon_synapses, nature_output_setup
from mb_stream import NOVEL_THETA, similar_counts, stream_job
from mb_task import EV, NATURE, NEW_SOURCE, OLD_SOURCE, err, features, load_mons
from spread import nature_bits

FRACS = [0.1, 0.3, 0.5, 0.7, 1.0]
N_FOLDS = 5
N_WORKERS = 9
STREAM = {"계속: 두 구획 τ=0.5": {"learn": "dual", "theta": NOVEL_THETA, "tau": 0.5, "gate_mode": "self"},
          "계속: 한 번씩": {"learn": "once"}}
ARMS = ["고정 (옛 환경만)", *STREAM, "한꺼번에: 옛 환경+새 환경", "한꺼번에: 새 환경만"]
SLUG = dict(zip(ARMS, ["fixed", "stream_dual", "stream_once", "batch_both", "batch_new"]))
OUT = f"out/mb_update_preds_{TAG}.parquet"
PARTS = f"out/mb_update_parts_{TAG}"
EV_SHIFT = 6  # EV 경향이 바뀐 종족: 옛 환경·새 환경 평균 EV 차이(|차이| 합 / 2)가 이 값 이상
UNIT = "비율"  # --teams면 "팀 수": FRACS가 비율 대신 학습 몫 앞에서부터 쓸 팀 수


def use_teams(counts):
    """학습 몫 크기를 팀 수로 (희소한 새 환경). 저장 위치를 비율 실행과 나눈다."""
    global FRACS, UNIT, OUT, PARTS
    FRACS, UNIT = [float(c) for c in counts], "팀 수"
    OUT, PARTS = OUT.replace(".parquet", "_teams.parquet"), PARTS + "_teams"


def splits(mo):
    """[(학습 몫 전체 순서(팀 단위 무작위), 평가 행)] fold마다, {(fold, 비율 또는 팀 수): 학습할 페이스트 수}.
    같은 fold의 팀 순서는 비율 실행과 같다 (시드 100 + fold) → 팀 수 k개 = 비율 실행의 앞부분과 같은 팀들."""
    mb = np.where(mo.source == NEW_SOURCE)[0]
    team = mo.team_id.to_numpy()
    out, n_of = [], {}
    for f, (lr, ev) in enumerate(GroupKFold(n_splits=N_FOLDS).split(mb, groups=mo.archetype_key.to_numpy()[mb])):
        learn, evalr = mb[lr], mb[ev]
        teams = np.unique(team[learn])
        rank = dict(zip(np.random.default_rng(100 + f).permutation(teams), range(len(teams))))
        order = learn[np.lexsort((learn, [rank[t] for t in team[learn]]))]
        cum = np.cumsum(pd.Series(team[order]).groupby(team[order], sort=False).size().to_numpy())  # 팀 순서대로 누적 페이스트 수
        for fr in FRACS:
            if UNIT == "팀 수":
                n_of[(f, fr)] = int(cum[int(fr) - 1])
            else:
                n_of[(f, fr)] = int(cum[min(np.searchsorted(cum, fr * len(order) - 1e-9), len(cum) - 1)])
        out.append((order, evalr))
    return out, n_of


def part_path(arm, fold, frac):
    return f"{PARTS}/{SLUG[arm]}_fold{fold}_frac{frac:g}.parquet"


def save_part(df, path):
    """다른 프로세스(--report)가 반쯤 쓴 파일을 읽지 않도록 임시 이름으로 쓰고 바꾼다."""
    tmp = path + ".tmp"
    df.to_parquet(tmp)
    os.replace(tmp, path)


def tagged(key, fn, *args, **kwargs):
    return key, fn(*args, **kwargs)


def fit_job(rates, Y, bits, nat_id, natures, conn, axes, arch, rows, label):
    """행들로 처음부터 학습 (mb_learn과 같은 절차) -> (성격 출력층, EV 출력층, 성격 학습 정보, EV 학습 정보, 맥락 세기)."""
    nature_net, val_logp, fit_n = fit_nature(rates, nat_id, len(natures), conn, arch, rows, f"{label} 성격", axes=axes)
    ctx_rate = context_rate(rates, rows)
    ev_net, fit_e = fit_ev(rates, Y, bits, natures, val_logp, arch, rows, ctx_rate, f"{label} EV")
    return nature_net, ev_net, fit_n, fit_e, ctx_rate


def batch_job(rates, Y, bits, nat_id, natures, conn, axes, arch, X, sp_feat, train_rows, eval_rows, label):
    """학습 행으로 처음부터 학습 -> 평가 행 예측만."""
    nature_net, ev_net, fit_n, fit_e, ctx_rate = fit_job(rates, Y, bits, nat_id, natures, conn, axes, arch, train_rows, label)
    counts = np.asarray(X[train_rows].sum(0)).ravel()
    df = stream_job(nature_net, ev_net, fit_n, fit_e, rates, Y, bits, nat_id, natures, X, sp_feat, counts, eval_rows, {"learn": "none"},
                    None, None, ctx_rate, label, quiet=True, learn_mask=np.zeros(len(eval_rows), bool))
    return df, {"ep_n": fit_n["epochs"], "ep_e": fit_e["epochs"]}


def run():
    mo = load_mons()
    arch = mo.archetype_key.to_numpy()
    X, vocab = features(mo)
    X = X.tocsr()
    Y, nat = mo[EV].to_numpy(), mo[NATURE].to_numpy()
    natures, nat_id = np.unique(nat, return_inverse=True)
    bits = nature_bits(nat)
    ma, mb = np.where(mo.source == OLD_SOURCE)[0], np.where(mo.source == NEW_SOURCE)[0]
    kind = np.array([v[:2] for v in vocab])
    sp_feat = np.array([[f for f in X.indices[X.indptr[i]:X.indptr[i + 1]] if kind[f] == "sp"][0] for i in range(len(mo))])

    W, groups = load_mb()
    codes, inv = kc_codes(X, W, groups, f"out/kc_codes_real_{TAG}.npz")
    per_row = codes[inv].copy()
    per_row[mb] = kc_row_codes(X, mb, W, groups, f"out/kc_codes_real_{TAG}_new_rows_seed{ENCOUNTER_SEED0}.npz")
    rates = sp.csr_matrix(per_row.astype(np.float64) / (DURATION_MS / 1000))
    C = (per_row / np.maximum(np.linalg.norm(per_row, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    conn, axes = nature_output_setup(kc_mbon_synapses(W, groups), natures)
    sp_, n_of = splits(mo)
    print("학습 몫 페이스트 수 (fold별): " + " / ".join(f"{fr:g}: " + ",".join(str(n_of[(f, fr)]) for f in range(N_FOLDS)) for fr in FRACS), flush=True)
    os.makedirs(PARTS, exist_ok=True)
    fold_of = np.full(len(mo), -1)
    for f, (_, evalr) in enumerate(sp_):
        fold_of[evalr] = f

    t0 = time.time()
    common = (rates, Y, bits, nat_id, natures, conn, axes, arch)
    # 1단계: 옛 환경 학습(고정·계속 배우기의 출발점) + 한꺼번에 다시 학습 전부. 저장된 작업은 건너뜀
    ma_path = f"{PARTS}/old_nets.joblib"
    jobs, n_all = [], 1 + 2 * N_FOLDS * len(FRACS)
    if not os.path.exists(ma_path):
        jobs.append(delayed(tagged)(("옛 환경",), fit_job, *common, ma, "[옛 환경]"))
    for f, (order, evalr) in enumerate(sp_):
        for fr in FRACS:
            learn = order[:n_of[(f, fr)]]
            for arm, rows in [("한꺼번에: 옛 환경+새 환경", np.concatenate([ma, learn])), ("한꺼번에: 새 환경만", learn)]:
                if not os.path.exists(part_path(arm, f, fr)):
                    jobs.append(delayed(tagged)((arm, f, fr), batch_job, *common, X, sp_feat, np.sort(rows), evalr, f"[{arm}, fold {f}, {fr:g}]"))
    print(f"[1단계] 옛 환경 학습 + 한꺼번에 다시 학습 {n_all}개 중 {len(jobs)}개 시작 (저장돼 건너뜀 {n_all - len(jobs)}, 작업자 {N_WORKERS})", flush=True)
    for k, (key, res) in enumerate(Parallel(n_jobs=N_WORKERS, return_as="generator_unordered")(jobs), 1):
        if key == ("옛 환경",):
            joblib.dump(res, ma_path)
            print(f"[1단계 {k}/{len(jobs)}] 옛 환경 학습 끝 (성격 eta={res[2]['eta']:g} {res[2]['epochs']}회, EV eta={res[3]['eta']:g} {res[3]['epochs']}회, "
                  f"경과 {(time.time() - t0) / 60:.1f}분)", flush=True)
            continue
        df, fit = res
        save_part(df.assign(arm=key[0], fold=key[1], frac=key[2]), part_path(*key))
        print(f"[1단계 {k}/{len(jobs)}] {key[0]} fold {key[1]} {UNIT} {key[2]:g} 끝·저장 (성격 {fit['ep_n']}회, EV {fit['ep_e']}회, "
              f"경과 {(time.time() - t0) / 60:.1f}분)", flush=True)

    # 2단계: 고정 예측 + 계속 배우기 (옛 환경 학습 뒤 학습 몫 앞부분을 한 마리씩 -> 평가 개체 예측만). 저장된 작업은 건너뜀
    nature_net, ev_net, fit_n, fit_e, ctx_rate = joblib.load(ma_path)
    counts_ma = np.asarray(X[ma].sum(0)).ravel()
    train_sim = np.zeros(len(mo), np.float32)
    train_sim[mb] = (C[mb] @ C[ma].T).max(1)
    jobs, n_all = [], 1 + len(STREAM) * N_FOLDS * len(FRACS)
    if not os.path.exists(part_path("고정 (옛 환경만)", "all", 0)):
        jobs.append(delayed(tagged)(("고정 (옛 환경만)", "all", 0), stream_job, nature_net, ev_net, fit_n, fit_e, rates, Y, bits, nat_id, natures, X, sp_feat,
                                    counts_ma, mb, {"learn": "none"}, None, None, ctx_rate, "[고정]", quiet=True, learn_mask=np.zeros(len(mb), bool)))
    for f, (order, evalr) in enumerate(sp_):
        todo = [(arm, fr) for fr in FRACS for arm in STREAM if not os.path.exists(part_path(arm, f, fr))]
        if not todo:
            continue
        full = np.concatenate([order, evalr])
        fam = similar_counts(C, ma, full, [NOVEL_THETA])[0][NOVEL_THETA]  # 앞선 페이스트만 세므로 앞부분 순서에서도 같은 값
        for arm, fr in todo:
            n, cfg = n_of[(f, fr)], STREAM[arm]
            seq = np.concatenate([order[:n], evalr])
            fam_seq = np.concatenate([fam[:n], fam[len(order):]])  # 평가 개체 값은 배우지 않으므로 쓰이지 않음
            mask = np.r_[np.ones(n, bool), np.zeros(len(evalr), bool)]
            jobs.append(delayed(tagged)((arm, f, fr), stream_job, nature_net, ev_net, fit_n, fit_e, rates, Y, bits, nat_id, natures, X, sp_feat, counts_ma,
                                        seq, cfg, fam_seq if "theta" in cfg else None, None, ctx_rate, f"[{arm}, fold {f}, {fr:g}]",
                                        quiet=True, train_sim=train_sim[seq], learn_mask=mask))
    print(f"[2단계] 고정 예측 + 계속 배우기 {n_all}개 중 {len(jobs)}개 시작 (저장돼 건너뜀 {n_all - len(jobs)}, 경과 {(time.time() - t0) / 60:.1f}분)", flush=True)
    for k, (key, df) in enumerate(Parallel(n_jobs=N_WORKERS, return_as="generator_unordered")(jobs), 1):
        if key[0] == "고정 (옛 환경만)":
            df = df.assign(arm=key[0], fold=fold_of[df.row.to_numpy()], frac=0.0)
        else:
            df = df[~df.learned].assign(arm=key[0], fold=key[1], frac=key[2])
        save_part(df, part_path(*key))
        print(f"[2단계 {k}/{len(jobs)}] {key[0]} fold {key[1]} {UNIT} {key[2]:g} 끝·저장 (경과 {(time.time() - t0) / 60:.1f}분)", flush=True)
    res = load_results()
    res.to_parquet(OUT)
    print(f"예측 저장: {OUT} (작업 {len(glob.glob(f'{PARTS}/*.parquet'))}개, 경과 {(time.time() - t0) / 60:.1f}분)", flush=True)


def load_results():
    """작업별 저장이 있으면 그것을, 없으면 합친 파일을 읽는다."""
    files = sorted(glob.glob(f"{PARTS}/*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True) if files else pd.read_parquet(OUT)


def report():
    mo = load_mons()
    X, vocab = features(mo)
    X = X.tocsr()
    Y, nat, species = mo[EV].to_numpy(), mo[NATURE].to_numpy(), mo.species.to_numpy()
    ma, mb = np.where(mo.source == OLD_SOURCE)[0], np.where(mo.source == NEW_SOURCE)[0]
    kind = np.array([v[:2] for v in vocab])
    sp_feat = np.array([[f for f in X.indices[X.indptr[i]:X.indptr[i + 1]] if kind[f] == "sp"][0] for i in range(len(mo))])
    res = load_results()
    sp_, n_of = splits(mo)

    cov = res.groupby(["frac", "arm"]).fold.nunique().unstack("arm").reindex(columns=ARMS)
    full = (cov.fillna(0) == N_FOLDS).to_numpy().sum() == 1 + (len(ARMS) - 1) * len(FRACS)
    print(f"저장된 작업 ({UNIT}별 fold 수, 5면 끝):\n" + cov.fillna(0).astype(int).to_string())
    if not full:
        print(f"※ 중간 결과: {UNIT}마다 모든 방식이 예측한 평가 개체만 비교 (고정은 새 환경 전체)")
    keep = [res[res.frac == 0]]
    for fr, d in res[res.frac > 0].groupby("frac"):
        cnt = d.groupby("row").arm.nunique()
        keep.append(d[d.row.isin(cnt.index[cnt == d.arm.nunique()])])
    res = pd.concat(keep, ignore_index=True)

    cnt_ma = np.asarray(X[ma].sum(0)).ravel()
    new_sp = cnt_ma[sp_feat] == 0
    new_tool = np.array([not new_sp[i] and any(cnt_ma[f] == 0 and kind[f] != "sp" for f in X.indices[X.indptr[i]:X.indptr[i + 1]]) for i in range(len(mo))])
    n_ma, n_mb = pd.Series(species[ma]).value_counts(), pd.Series(species[mb]).value_counts()
    both = [s for s in n_ma.index.intersection(n_mb.index) if n_ma[s] >= 20 and n_mb[s] >= 20]
    # 성격 경향이 바뀐 종족: 옛 환경·새 환경 모두 20마리 이상, 성격 최빈이 다르고, 옛 환경 최빈 성격의 비율이 새 환경에서 15%p 이상 줄어듦
    share = lambda idx: pd.crosstab(species[idx], nat[idx], normalize="index")
    s_ma, s_mb = share(ma), share(mb)
    changed_sp = {s for s in both if s_mb.loc[s].idxmax() != s_ma.loc[s].idxmax()
                  and s_ma.loc[s, s_ma.loc[s].idxmax()] - s_mb.loc[s].get(s_ma.loc[s].idxmax(), 0.0) >= 0.15}
    # EV 경향이 바뀐 종족: 옛 환경·새 환경 모두 20마리 이상, 평균 EV 차이(|차이| 합 / 2)가 EV_SHIFT 이상
    mean_ev = lambda idx: pd.DataFrame(Y[idx], columns=EV).groupby(species[idx]).mean()
    e_ma, e_mb = mean_ev(ma), mean_ev(mb)
    shift = {s: (e_ma.loc[s] - e_mb.loc[s]).abs().sum() / 2 for s in both}
    ev_changed_sp = {s for s, v in shift.items() if v >= EV_SHIFT}
    changed, ev_changed = np.isin(species, list(changed_sp)), np.isin(species, list(ev_changed_sp))
    print(f"새 환경 {len(mb):,}마리: 새 종족 {new_sp[mb].sum()}, 새 도구 {new_tool[mb].sum()}, 아는 종족·새 도구 없음 {(~new_sp & ~new_tool)[mb].sum()}")
    print(f"성격 경향이 바뀐 종족(양쪽 20마리 이상, 최빈 바뀜·옛 환경 최빈 비율 15%p 이상 줄어듦) {len(changed_sp)}종 {changed[mb].sum()}마리: "
          + ", ".join(f"{s} {s_ma.loc[s].idxmax()}→{s_mb.loc[s].idxmax()}" for s in sorted(changed_sp)))
    print(f"EV 경향이 바뀐 종족(양쪽 20마리 이상, 평균 EV 차이 {EV_SHIFT} 이상) {len(ev_changed_sp)}종 {ev_changed[mb].sum()}마리: "
          + ", ".join(f"{s} {shift[s]:.1f}" for s in sorted(ev_changed_sp, key=lambda s: -shift[s])))

    # 학습 몫 안의 같은 종족 수 (평가 개체마다, fold·비율별)
    seen = {}
    for f, (order, evalr) in enumerate(sp_):
        for fr in FRACS:
            vc = pd.Series(species[order[:n_of[(f, fr)]]]).value_counts()
            seen[(f, fr)] = dict(zip(evalr, vc.reindex(species[evalr]).fillna(0).to_numpy(int)))
    r = res.row.to_numpy()
    res["ev_mode"] = err(res[EV].to_numpy(), Y[r])
    res["ev_min"] = err(res[["min_" + c for c in EV]].to_numpy(), Y[r])
    res["nat_hit"] = res.nature.to_numpy() == nat[r]
    res["seen"] = [0 if fr == 0 else seen[(f, fr)][i] for f, fr, i in zip(res.fold, res.frac, r)]
    fracs = [0.0, *FRACS]
    groups = {"전체": np.ones(len(mo), bool), "새 종족": new_sp, "새 도구 (아는 종족)": new_tool, "아는 종족·새 도구 없음": ~new_sp & ~new_tool,
              "성격 경향이 바뀐 종족": changed, "EV 경향이 바뀐 종족": ev_changed}

    def table(sub, title):
        t = sub.groupby(["frac", "arm"])[["nat_hit", "ev_min", "ev_mode"]].mean()
        cell = t.apply(lambda x: f"{x.nat_hit:.1%} / {x.ev_min:.2f} / {x.ev_mode:.2f}", axis=1).unstack("arm")
        fixed = cell.loc[0.0, "고정 (옛 환경만)"] if (0.0 in cell.index and "고정 (옛 환경만)" in cell.columns) else "–"
        cell = cell.reindex(index=fracs, columns=ARMS[1:])
        cell.loc[0.0] = fixed  # 비율 0 = 옛 환경만 학습 = 모든 방식의 출발점
        n = sub[sub.frac > 0].groupby("frac").row.nunique()
        cell.index = ["0"] + [f"{fr:g} ({n.get(fr, 0):,})" for fr in FRACS]
        print(f"\n[{title}] 칸 = 성격 / EV 기대 오차 최소 / EV 최빈. 행 = 새 환경 학습 몫 중 쓴 {UNIT} (0 = 옛 환경만, 괄호는 비교한 평가 개체 수)")
        print(cell.fillna("–").to_string())

    for g, mask in groups.items():
        table(res[mask[r]], f"{g} ({mask[mb].sum():,}마리)")

    print(f"\n[새 종족: 학습 몫에 같은 종족이 몇 마리 있었나별] 칸 = 성격 / EV 기대 오차 최소 (개체 수 = {UNIT} 전체 합)")
    sub = res[new_sp[r] & (res.frac > 0)].copy()
    if len(sub):
        sub["bin"] = pd.cut(sub.seen, [-1, 0, 4, 19, 10**9], labels=["0", "1~4", "5~19", "20+"])
        t = sub.groupby(["bin", "arm"], observed=True)[["nat_hit", "ev_min"]].mean()
        n = sub[sub.arm == sub.arm.iloc[0]].groupby("bin", observed=True).size()
        cell = t.apply(lambda x: f"{x.nat_hit:.1%} / {x.ev_min:.2f}", axis=1).unstack("arm").reindex(columns=ARMS[1:])
        cell.index = [f"{b} ({n.get(b, 0)})" for b in cell.index]
        print(cell.fillna("–").to_string())

    print("\n[짝지은 차이, 종족 부트스트랩 95%] 음수 EV = 앞 방식이 나음, 양수 성격 = 앞 방식이 나음 (두 방식 모두 있는 평가 개체만)")
    rng = np.random.default_rng(0)
    u, ti = np.unique(species[mb], return_inverse=True)
    Wb = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=2000)[:, ti]  # (2000, 새 환경 행)
    pairs = [("한꺼번에: 옛 환경+새 환경", "계속: 두 구획 τ=0.5"), ("한꺼번에: 옛 환경+새 환경", "계속: 한 번씩"), ("계속: 두 구획 τ=0.5", "계속: 한 번씩"),
             ("한꺼번에: 새 환경만", "한꺼번에: 옛 환경+새 환경")]
    for a, b in pairs:
        for g in ["전체", "새 종족", "성격 경향이 바뀐 종족", "EV 경향이 바뀐 종족"]:
            cells = []
            for fr in FRACS:
                A = res[(res.arm == a) & (res.frac == fr)].set_index("row").reindex(mb)
                B = res[(res.arm == b) & (res.frac == fr)].set_index("row").reindex(mb)
                out = []
                for col in ["ev_min", "nat_hit"]:
                    d = A[col].to_numpy(float) - B[col].to_numpy(float)
                    ok = groups[g][mb] & ~np.isnan(d)
                    if not ok.any():
                        break
                    w = Wb * ok[None, :]
                    bs = (w * np.nan_to_num(d)).sum(1) / np.maximum(w.sum(1), 1)
                    scale = 100 if col == "nat_hit" else 1
                    out.append(f"{d[ok].mean() * scale:+.2f} [{np.percentile(bs, 2.5) * scale:+.2f}, {np.percentile(bs, 97.5) * scale:+.2f}]")
                if len(out) == 2:
                    cells.append(f"{fr:g}: EV {out[0]}, 성격 {out[1]}%p ({ok.sum():,}마리)")
            if cells:
                print(f"  {a} − {b} | {g}\n    " + "\n    ".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="저장된 예측으로 표만 (실행 중이면 지금까지 저장된 작업만)")
    ap.add_argument("--teams", default="", help="비율 대신 학습 몫 앞에서부터 쓸 팀 수, 쉼표로 (예: 5,10,20,30). 저장 위치가 *_teams로 따로")
    args = ap.parse_args()
    if args.teams:
        use_teams(args.teams.split(","))
    if not args.report:
        run()
    report()
