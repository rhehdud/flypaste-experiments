#!/usr/bin/env python3
"""환경이 계속 바뀔 때: 옛 환경으로 학습한 초파리가 새 환경 페이스트를 한 마리씩 "먼저 예측 -> 정답을 보고 학습"한다 (순차 평가).

새 이름도 빼지 않고 그대로 넣는다. 예측 직후 정답을 받는다고 가정하므로 "그 종족의 공개 페이스트가 N개 쌓였을 때"로 읽는다.
새 환경 파일에 날짜가 없어 순서가 시간순인지 알 수 없으므로 무작위 순서 N_ORDERS번의 평균.
예측은 mb_learn의 "성격 먼저 -> EV (성격 확률만큼 맥락)". 학습은 mb_learn과 같은 도파민 규칙, 맥락은 실제 성격.

익숙함 = 이미 배운 페이스트(옛 환경 + 스트림에서 앞서 배운 것) 중 Kenyon cell 발화 패턴의 코사인 유사도가 θ 이상인 수.
새로움 g = 1 / (1 + 익숙함). R = 옛 환경 학습에서 뽑힌 반복 수.
새 페이스트를 배우는 방식 (초파리 변형):
  한 번씩: 출력층(느린 구획)이 1회
  새로움(패턴): 출력층이 max(1, R x g)회
  빠른·느린 구획: 느린 구획 = 한 번씩. 빠른 구획 = 0에서 시작하는 별도 출력층.
    빠른 구획은 합친 점수(느린 + 빠른)의 오차로 R x g회(반올림, g는 θ=0.5 기준 새로움) 배우고,
    배운 페이스트의 Kenyon cell 패턴을 세기 g로 기억한다. 매 페이스트마다 빠른 구획 가중치와 기억 세기를 (1 - 망각)배.
    출력 점수 = 느린 구획 + 문 x 빠른 구획. m = max_기억 (세기 x 현재 패턴과의 코사인).
    빠른 구획이 기억한 낯선 페이스트와 닮은 입력에서만 켜진다.
    기억 세기 두 가지:
      새로움 g: 위의 g를 그대로 기억 세기로 쓴다
      학습 기억(τ): 학습 데이터 중 가장 닮은 페이스트의 코사인이 τ 미만일 때만
        빠른 구획이 배우고(max(1, R x g)회) 세기 1로 기억. 느린 구획은 스트림 페이스트를 1회씩만 배워 아직 모르므로,
        같은 종족 페이스트가 쌓여도 학습 데이터 기준 낯섦은 줄지 않는다
    문 세 가지:
      비례: clip((m - 켜짐) / (1 - 켜짐), 0, 1)
      계단: m >= 켜짐이면 1, 아니면 0
      자기 낯섦: 기억 유사도 대신 지금 입력의 학습 데이터 최대 코사인 < τ이면 1 (빠른 구획이 배우는 기준과 같음).
        느린 구획이 오프라인 재학습으로 새 페이스트를 익히기 전까지 새 종족에서 계속 열린다
  두 구획(온전한 복사): 느린 구획 = 한 번씩. 빠른 구획 = 시작 때 학습된 가중치를 복사하고 모든 페이스트를 새로움 규칙
    (max(1, R x g)회, 자기 오차)으로 배움 = 새로움 변형과 같은 네트워크. 출력 = (1 - 문) x 느린 + 문 x 빠른,
    문 = 지금 입력의 학습 데이터 최대 코사인 < τ이면 1 (자기 낯섦).
    실제 버섯체도 구획마다 학습 속도·기억 유지가 다르고, MBON-α'3처럼 낯선 Kenyon cell 패턴에 반응하는 출력 뉴런이 있다.

새로 들어오는 새 환경 페이스트의 Kenyon cell 코드는 개체마다 따로 시뮬레이션한 새 반응(mb.kc_row_codes, 기본). 같은 반응으로 예측하고 바로 배운다.
옛 환경과 입력이 같아도 학습 코드를 그대로 다시 쓰지 않는다 (잡음까지 같으면 점수가 부풀려진다). --same-noise면 학습 코드를 재사용한다.
예측은 세 가지를 함께 기록한다 (배우는 것은 같음): 성격 확률만큼 맥락의 최빈 배분(EV 열), 같은 확률의 기대 오차 최소 배분(min_ 열),
성격을 알 때 정답 성격 맥락의 최빈·기대 오차 최소 배분(known_·known_min_ 열).

설정은 시험 데이터에서 고르지 않도록 새 환경을 종족 단위로 반씩 나눈다 (새 종족·익숙한 종족 각각 무작위 반).
튜닝 절반의 EV 오차로 고르고, 다른 절반(그 종족들은 튜닝 채점에 안 쓰임)에서 확인한다. 스트림 자체는 새 환경 전체로 한 번 흐른다.
확인 절반은 여러 번 보지 않도록 기본으로는 튜닝 절반만 출력하고, --confirm일 때만 확인 절반을 출력한다.

  python src/mb_stream.py              # 튜닝 절반만 (새 반응 코드)
  python src/mb_stream.py --confirm    # 설계를 정한 뒤 한 번: 확인 절반까지
  python src/mb_stream.py --same-noise # 새 환경도 학습 코드 재사용, 비교용
"""
import argparse
import copy
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.special import log_softmax, softmax

from mb import DURATION_MS, ENCOUNTER_SEED0, PN_MIN_KC_SYNAPSES, PNS_PER_FEATURE, kc_codes, kc_row_codes, load_mb
from mb_learn import N_JOBS, DopamineOutput, context_rate, fit_ev, fit_nature, kc_mbon_synapses, nature_output_setup
from mb_task import EV, NATURE, NEW_SOURCE, OLD_SOURCE, err, features, load_mons
from spread import EV_CAP, best_spread, expected_error, nature_bits, odd_combo

N_ORDERS = 3
NOVEL_THETA = 0.5  # 새로움 기준
# 두 구획: 학습 데이터 최대 코사인이 이 값 미만이면 빠른 구획 출력. 0.99 = 학습에 거의 같은 입력이 있을 때만 느린 구획
DUAL_TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
# 보조층 빠른 구획("learn": "fast")은 목록에서 뺐다. 성격 21구획 출력층에서만 돌아간다
# 빠른 구획 반복 수 배수 k: 빠른 구획 반복 = max(1, round(R x g x k)). 빈 목록이면 k=1만 쓴다
FAST_MULTS = []
VARIANTS = {"한 번씩": {"learn": "once"},
            "새로움(패턴 0.5)": {"learn": "novel", "theta": NOVEL_THETA},
            **{f"두 구획(낯섦 τ={t:g})": {"learn": "dual", "theta": NOVEL_THETA, "tau": t, "gate_mode": "self"} for t in DUAL_TAUS},
            **{f"두 구획(낯섦 τ=0.5, 빠른 반복 x{k:g})": {"learn": "dual", "theta": NOVEL_THETA, "tau": 0.5, "gate_mode": "self", "fast_mult": k}
               for k in FAST_MULTS}}
SEEN_BINS = [(0, 0), (1, 4), (5, 19), (20, 10**9)]
PRED_KINDS = {"": "최빈", "min_": "기대 오차 최소", "known_": "성격 앎", "known_min_": "성격 앎, 기대 오차 최소"}  # 열 접두어 -> 이름


def stream_path(same_noise):
    return f"out/mb_stream_preds_{DURATION_MS}ms_pn{PN_MIN_KC_SYNAPSES}x{PNS_PER_FEATURE}{'' if same_noise else '_encounter'}.parquet"


def similar_counts(C, tr, order, thetas):
    """({θ: 스트림 k번째 개체와 코사인 θ 이상인 이미 배운 페이스트 수 (옛 환경 전체 + 스트림에서 앞선 것)},
    스트림 개체끼리 코사인 (새 환경, 새 환경) 순서대로). C는 L2 정규화 코드."""
    cur = C[order]
    s_past = cur @ C[tr].T                              # (새 환경, 옛 환경)
    s_prev = cur @ cur.T                                # (새 환경, 새 환경), 순서대로
    earlier = np.tril(np.ones((len(order), len(order)), dtype=bool), -1)
    return {t: (s_past >= t).sum(1) + ((s_prev >= t) & earlier).sum(1) for t in thetas}, s_prev


def practise_fast(fast, slow, active, rates, truth, ctx):
    """빠른 구획만 합친 점수(느린 + 빠른)의 오차로 한 번 배운다 (배우는 페이스트는 막 기억했으므로 문이 열린 상태)."""
    s = slow.b - rates @ slow.w[active] + fast.b - rates @ fast.w[active]
    if ctx is not None:
        s = s - ctx @ slow.u - ctx @ fast.u
    dopamine = softmax(s.reshape(fast.shape), axis=1)
    dopamine[np.arange(fast.shape[0]), truth] -= 1
    dopamine = dopamine.ravel()
    w = np.maximum(fast.w[active] + fast.eta * np.outer(rates, dopamine), 0)
    fast.w[active] = w if fast.conn is None else w * fast.conn[active]
    if ctx is not None:
        fast.u = np.maximum(fast.u + fast.eta * np.outer(ctx, dopamine), 0)
    fast.b -= fast.eta * dopamine


def stream_job(nature_net, ev_net, fit_n, fit_e, rates, Y, bits, nat_id, natures, X, sp_feat, counts, order, cfg,
               familiar, s_prev, ctx_rate, label, quiet=False, train_sim=None, learn_mask=None):
    """familiar: 스트림 순서대로의 익숙함 (similar_counts[θ]), 한 번씩이면 None. s_prev: 빠른 구획일 때 스트림 개체끼리 코사인.
    train_sim: 스트림 순서대로 학습 데이터와의 최대 코사인 (기억 세기 "train"·자기 낯섦 문일 때).
    learn_mask: 스트림 순서대로 정답을 보고 배울지 (False면 예측만 하고 배우지도, 본 수를 세지도 않음). None이면 모두 배움.
    cfg["learn"] == "none"이면 배우지 않고 예측만 (고정). quiet면 스트림 안 진행률을 찍지 않는다."""
    # 변형·순서마다 따로 배우도록 복사 (병렬 작업에 넘어온 큰 배열은 읽기 전용일 수 있어 쓸 수 있는 사본으로)
    nature_net, ev_net = copy.deepcopy(nature_net), copy.deepcopy(ev_net)
    for net in (nature_net, ev_net):
        net.w, net.u, net.b = np.array(net.w), np.array(net.u), np.array(net.b)
    fast = cfg["learn"] == "fast"
    dual = cfg["learn"] == "dual"
    if fast:
        assert isinstance(nature_net, DopamineOutput), "보조층 빠른 구획은 성격 21구획 출력층에서만 (합친 점수 학습이 그 구조 기준)"
        nature_fast = DopamineOutput(rates.shape[1], 1, len(natures), fit_n["eta"], conn=nature_net.conn)
        ev_fast = DopamineOutput(rates.shape[1], 6, EV_CAP + 1, fit_e["eta"], n_ctx=bits.shape[1])
        strength = np.zeros(len(order))  # 스트림 k번째 페이스트를 빠른 구획이 기억한 세기
    if dual:
        nature_fast, ev_fast = copy.deepcopy(nature_net), copy.deepcopy(ev_net)
    counts = counts.copy()
    cand = nature_bits(natures) * ctx_rate
    rec, t0, step = [], time.time(), max(1, len(order) // 5)
    for k, m in enumerate(order):
        R = rates[m]
        g = 1.0 / (1 + familiar[k]) if familiar is not None else 0.0
        gate = 0.0
        mode = cfg.get("gate_mode", "ramp")
        if (fast or dual) and mode == "self":
            gate = float(train_sim[k] < cfg["tau"])
        elif fast and k:
            m_sim = (strength[:k] * s_prev[k, :k]).max()
            gate = float(m_sim >= cfg["gate"]) if mode == "step" else float(np.clip((m_sim - cfg["gate"]) / (1 - cfg["gate"]), 0, 1))
        if dual:  # 두 구획: 문에 따라 느린·빠른 구획 점수를 섞음
            mix = lambda slow_s, fast_s: fast_s() if gate == 1 else slow_s() if gate == 0 else (1 - gate) * slow_s() + gate * fast_s()
        else:  # 보조층: 느린 구획 + 문 x 빠른 구획
            mix = lambda slow_s, fast_s: slow_s() + (gate * fast_s() if gate else 0)
        s_nat = mix(lambda: nature_net.scores(R)[0, 0], lambda: nature_fast.scores(R)[0, 0])
        logp = log_softmax(s_nat)
        ctx = (np.exp(logp) @ cand)[None]
        lp_ev = log_softmax(mix(lambda: ev_net.scores(R, ctx), lambda: ev_fast.scores(R, ctx)), axis=2)
        ctx_known = cand[nat_id[m]][None]
        lp_known = log_softmax(mix(lambda: ev_net.scores(R, ctx_known), lambda: ev_fast.scores(R, ctx_known)), axis=2)
        spreads = {"": best_spread(lp_ev)[1][0], "min_": best_spread(-expected_error(np.exp(lp_ev)))[1][0],
                   "known_": best_spread(lp_known)[1][0], "known_min_": best_spread(-expected_error(np.exp(lp_known)))[1][0]}
        feats = X.indices[X.indptr[m]:X.indptr[m + 1]]
        learn = learn_mask is None or bool(learn_mask[k])
        rec.append({"row": m, "step": k + 1, "nature": natures[logp.argmax()],
                    **{pre + c: v for pre, sp_ in spreads.items() for c, v in zip(EV, sp_)},
                    "species_seen": counts[sp_feat[m]], "g": g, "gate": gate, "learned": learn})
        if not learn:
            continue

        lo, hi = rates.indptr[m], rates.indptr[m + 1]
        active, data, ctx_true = rates.indices[lo:hi], rates.data[lo:hi], bits[m] * ctx_rate
        reps_n = reps_e = 0 if cfg["learn"] == "none" else 1
        if cfg["learn"] == "novel":
            reps_n, reps_e = (max(1, round(fit["epochs"] * g)) for fit in (fit_n, fit_e))
        for _ in range(reps_n):
            nature_net.practise(active, data, nat_id[m:m + 1])
        for _ in range(reps_e):
            ev_net.practise(active, data, Y[m], ctx_true)
        if dual:
            k_fast = cfg.get("fast_mult", 1)
            for _ in range(max(1, round(fit_n["epochs"] * g * k_fast))):
                nature_fast.practise(active, data, nat_id[m:m + 1])
            for _ in range(max(1, round(fit_e["epochs"] * g * k_fast))):
                ev_fast.practise(active, data, Y[m], ctx_true)
        if fast:
            if cfg["decay"]:
                strength *= 1 - cfg["decay"]
                for net in (nature_fast, ev_fast):
                    net.w *= 1 - cfg["decay"]
                    net.u *= 1 - cfg["decay"]
                    net.b *= 1 - cfg["decay"]
            if cfg.get("memory", "novelty") == "novelty":
                reps_n, reps_e = round(fit_n["epochs"] * g), round(fit_e["epochs"] * g)
                if reps_n or reps_e:
                    strength[k] = g
            elif train_sim[k] < cfg["tau"]:
                reps_n, reps_e = max(1, round(fit_n["epochs"] * g)), max(1, round(fit_e["epochs"] * g))
                strength[k] = 1.0
            else:
                reps_n = reps_e = 0
            for _ in range(reps_n):
                practise_fast(nature_fast, nature_net, active, data, nat_id[m:m + 1], None)
            for _ in range(reps_e):
                practise_fast(ev_fast, ev_net, active, data, Y[m], ctx_true)
        counts[feats] += 1
        if not quiet and (k + 1) % step == 0:
            print(f"  {label} {k + 1:,}/{len(order):,} (경과 {(time.time() - t0) / 60:.1f}분)", flush=True)
    return pd.DataFrame(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="확인 절반까지 출력 (설계를 정한 뒤 한 번만)")
    ap.add_argument("--same-noise", action="store_true", help="새 환경 페이스트도 학습 코드를 재사용 (비교용)")
    args = ap.parse_args()
    shown = "확인" if args.confirm else "튜닝"
    mo = load_mons()
    arch = mo.archetype_key.to_numpy()
    X, vocab = features(mo)
    X = X.tocsr()
    Y = mo[EV].to_numpy()
    nat = mo[NATURE].to_numpy()
    natures, nat_id = np.unique(nat, return_inverse=True)
    bits = nature_bits(nat)
    tr, te = np.where(mo.source == OLD_SOURCE)[0], np.where(mo.source == NEW_SOURCE)[0]
    kind = np.array([v[:2] for v in vocab])
    sp_feat = np.array([[f for f in X.indices[X.indptr[i]:X.indptr[i + 1]] if kind[f] == "sp"][0] for i in range(len(mo))])

    W, groups = load_mb()
    tag = f"{DURATION_MS}ms_pn{PN_MIN_KC_SYNAPSES}x{PNS_PER_FEATURE}"
    codes, inv = kc_codes(X, W, groups, f"out/kc_codes_real_{tag}.npz")
    per_row = codes[inv]
    if not args.same_noise:  # 새 환경 행은 개체마다 새 반응. 옛 환경 학습은 원래 코드 그대로
        per_row = per_row.copy()
        per_row[te] = kc_row_codes(X, te, W, groups, f"out/kc_codes_real_{tag}_new_rows_seed{ENCOUNTER_SEED0}.npz")
    rates = sp.csr_matrix(per_row.astype(np.float64) / (DURATION_MS / 1000))
    C = (per_row / np.maximum(np.linalg.norm(per_row, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    print(f"새 환경 코드: {'학습 코드 재사용 (같은 잡음)' if args.same_noise else '개체마다 새 반응'}", flush=True)
    S = kc_mbon_synapses(W, groups)
    conn, axes = nature_output_setup(S, natures)
    ctx_rate = context_rate(rates, tr)

    # 종족 단위로 새 환경을 튜닝/확인 절반으로 (새 종족, 익숙한 종족 각각 무작위 반)
    counts = np.asarray(X[tr].sum(0)).ravel()
    rng = np.random.default_rng(0)
    half_of = {}
    for is_new in (True, False):
        spp = np.unique(sp_feat[te][(counts[sp_feat[te]] == 0) == is_new])
        for s in spp[rng.permutation(len(spp))[:len(spp) // 2]]:
            half_of[s] = "튜닝"
    half = pd.Series(["튜닝" if half_of.get(s) else "확인" for s in sp_feat], index=range(len(mo)))

    t0 = time.time()
    nature_net, val_logp, fit_n = fit_nature(rates, nat_id, len(natures), conn, arch, tr, "옛 환경 성격", axes=axes)
    print(f"[옛 환경 학습 1/2] 성격({fit_n['kind']}) 끝 (eta={fit_n['eta']:g}, {fit_n['epochs']}회, 경과 {(time.time() - t0) / 60:.1f}분)", flush=True)
    ev_net, fit_e = fit_ev(rates, Y, bits, natures, val_logp, arch, tr, ctx_rate, "옛 환경 EV")
    print(f"[옛 환경 학습 2/2] EV 끝 (eta={fit_e['eta']:g}, {fit_e['epochs']}회, 경과 {(time.time() - t0) / 60:.1f}분)", flush=True)

    orders = [np.random.default_rng(s).permutation(te) for s in range(N_ORDERS)]
    thetas = sorted({cfg["theta"] for cfg in VARIANTS.values() if "theta" in cfg})
    fams, sims = zip(*[similar_counts(C, tr, o, thetas) for o in orders])
    train_sim = np.zeros(len(mo))
    train_sim[te] = (C[te] @ C[tr].T).max(1)
    print(f"[익숙함 계산 끝] 경과 {(time.time() - t0) / 60:.1f}분", flush=True)
    keys = [(v, s) for v in VARIANTS for s in range(N_ORDERS)]
    jobs = [delayed(stream_job)(nature_net, ev_net, fit_n, fit_e, rates, Y, bits, nat_id, natures, X, sp_feat, counts,
                                orders[s], VARIANTS[v], fams[s][VARIANTS[v]["theta"]] if "theta" in VARIANTS[v] else None,
                                sims[s] if VARIANTS[v]["learn"] == "fast" else None, ctx_rate, f"[{v}, 순서 {s + 1}]",
                                quiet=True, train_sim=train_sim[orders[s]])
            for v, s in keys]
    got = []
    for k, df in enumerate(Parallel(n_jobs=N_JOBS, return_as="generator")(jobs), 1):
        got.append(df.assign(variant=keys[k - 1][0], order=keys[k - 1][1]))
        print(f"  [순차 평가 {k}/{len(jobs)}] {keys[k - 1][0]}, 순서 {keys[k - 1][1] + 1} 끝 (경과 {(time.time() - t0) / 60:.1f}분)", flush=True)
    res = pd.concat(got, ignore_index=True)
    out = stream_path(args.same_noise)
    res.to_parquet(out)

    pred, true = res[EV].to_numpy(), Y[res.row]
    res["ev_err"] = err(pred, true)
    for pre in list(PRED_KINDS)[1:]:
        res[f"ev_err_{pre}"] = err(res[[pre + c for c in EV]].to_numpy(), true)
    res["nature_hit"] = res.nature.to_numpy() == nat[res.row]
    res["odd"] = odd_combo(pred, res.nature.to_numpy())
    res["species"] = np.where(counts[sp_feat[res.row]] == 0, "새 종족", "익숙한 종족")
    res["half"] = half[res.row].to_numpy()
    bin_of = lambda n: next(f"{lo}" if lo == hi else (f"{lo}+" if hi >= 10**9 else f"{lo}~{hi}") for lo, hi in SEEN_BINS if lo <= n <= hi)
    res["species_bin"] = res.species_seen.map(bin_of)
    bins = [bin_of(lo) for lo, _ in SEEN_BINS]
    names = list(VARIANTS)

    def mean(df, by):
        return df.groupby([*by, "order"])[["ev_err", "nature_hit", "odd"]].mean().groupby(by).mean()

    tune = mean(res[res.half == "튜닝"], ["variant"])
    new_names = [v for v in names if VARIANTS[v]["learn"] == "dual"]
    chosen = tune.loc[new_names, "ev_err"].idxmin()
    fmt = lambda c: "{:.2f}".format if c.endswith("EV 오차") else "{:.1%}".format
    ren = {"ev_err": "EV 오차", "nature_hit": "성격 정확도", "odd": "어색한 조합"}

    first = res[(res.order == 0) & (res.variant == names[0])]
    print(f"\nMA 학습 -> 새 환경 순차 평가, 무작위 순서 {N_ORDERS}번 평균. 개체 수: "
          + ", ".join(f"{h} {g}: {((first.half == h) & (first.species == g)).sum()}" for h in ["튜닝", "확인"] for g in ["새 종족", "익숙한 종족"]))
    print(f"튜닝 절반 EV 오차로 고른 두 구획: {chosen}" + ("  <- τ 후보 끝값" if VARIANTS[chosen]["tau"] in (DUAL_TAUS[0], DUAL_TAUS[-1]) else ""))
    halves = ["튜닝", "확인"] if shown == "확인" else ["튜닝"]
    for h in halves:
        sub = res[res.half == h]
        t = pd.concat({"전체": mean(sub, ["variant"]), **{g: mean(sub[sub.species == g], ["variant"]) for g in ["새 종족", "익숙한 종족"]}},
                      axis=1).reindex(names)
        t.columns = [f"{g} {ren[m]}" for g, m in t.columns]
        print(f"\n=== {h} 절반 ===")
        print(t.to_string(formatters={c: fmt(c) for c in t.columns}))

    for v in [chosen]:
        c = res[(res.half == shown) & (res.variant == v)]
        print(f"\n{v} {shown} 절반 빠른 구획 문 평균: "
              + ", ".join(f"{g} {c[c.species == g].gate.mean():.3f} (0.5 이상 {(c[c.species == g].gate >= 0.5).mean():.1%})"
                          for g in ["새 종족", "익숙한 종족"]))

    focus = ["한 번씩", "새로움(패턴 0.5)", chosen]
    sub = res[(res.half == shown) & res.variant.isin(focus)]
    for g in ["새 종족", "익숙한 종족"]:
        s = sub[sub.species == g]
        n = s[(s.order == 0) & (s.variant == names[0])].groupby("species_bin").size()
        cols = [b for b in bins if b in n.index]
        print(f"\n=== {shown} 절반, {g}: 그 종족을 전에 본 페이스트 수별 ({', '.join(f'{c}: {n[c]}마리' for c in cols)}) ===")
        g_mean = mean(s, ["variant", "species_bin"])
        for metric in ["ev_err", "nature_hit"]:
            print(f"[{ren[metric]}]")
            print(g_mean[metric].unstack("species_bin").reindex(index=focus, columns=cols)
                  .to_string(float_format=fmt(ren[metric])))
    extra = [f"ev_err_{pre}" for pre in list(PRED_KINDS)[1:]]
    for h in halves:
        sub = res[(res.half == h) & res.variant.isin(focus)]
        t = pd.concat({g: sub[sub.species == g].groupby(["variant", "order"])[["ev_err", *extra]].mean().groupby("variant").mean()
                       for g in ["새 종족", "익숙한 종족"]}, axis=1).reindex(focus)
        t.columns = [f"{g} {PRED_KINDS[c[7:]] if c != 'ev_err' else '최빈'}" for g, c in t.columns]
        print(f"\n=== {h} 절반, 고르는 규칙별 EV 오차 ===")
        print(t.to_string(float_format="{:.2f}".format))
    print(f"\n예측 저장: {out}")


if __name__ == "__main__":
    main()
