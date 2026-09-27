"""MaleCNS LIF 시뮬레이터 — Shiu et al. 2024 (philshiu/Drosophila_brain_model) 방식.

    dv/dt = (-52mV - v + g) / 20ms,  dg/dt = -g / 5ms   (불응기 중 고정, 들어온 시냅스 입력은 버림)
    spike: v > -45mV -> v = -52mV, g = 0, 불응기 2.2ms
    시냅스: 1.8ms 뒤 g += 접촉수 x 0.275mV x 전달물질 부호
    자극: Poisson(rate)마다 v += 250 x 0.275mV, 자극받는 뉴런은 불응기 0
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

DT = 0.1          # ms
TAU_V = 20.0      # ms
TAU_G = 5.0       # ms
V_REST = -52.0    # mV
V_THRESH = -45.0  # mV
REFRACTORY_MS = 2.2
DELAY_MS = 1.8
CONTACT_MV = 0.275
POISSON_KICK_MV = 250 * CONTACT_MV

NT_SIGN = {"acetylcholine": 1, "dopamine": 1, "octopamine": 1, "serotonin": 1,
           "gaba": -1, "glutamate": -1, "histamine": -1}


def build_weight_matrix(nodes, edges, nt):
    """W[post, pre] = mV (CSC), bodyId -> 행렬 인덱스."""
    idx = pd.Series(np.arange(len(nodes)), index=nodes.index)
    sign = nt.set_index("body")["consensus_nt"].map(NT_SIGN).reindex(nodes.index).fillna(1).to_numpy()

    pre = idx.loc[edges.body_pre].to_numpy()
    post = idx.loc[edges.body_post].to_numpy()
    strength = edges.weight.to_numpy() * CONTACT_MV * sign[pre]

    n = len(nodes)
    return sp.coo_matrix((strength, (post, pre)), shape=(n, n)).tocsc(), idx


def shuffle_targets(W, seed):
    """연결의 도착 뉴런만 섞은 W: in/out degree, 가중치, 부호는 보존."""
    c = W.tocoo()
    rows = np.random.default_rng(seed).permutation(c.row)
    return sp.coo_matrix((c.data, (rows, c.col)), shape=W.shape).tocsc()


def run(W, stim_idx, rate_hz, duration_ms, seed=0, align_noise=False):
    """stim_idx에 Poisson 입력을 주고 duration_ms 동안의 뉴런별 스파이크 수."""
    return run_schedule(W, [(stim_idx, 0.0, duration_ms)], rate_hz, duration_ms, seed, align_noise)[0]


def run_schedule(W, schedule, rate_hz, duration_ms, seed=0, align_noise=False):
    """schedule = [(자극 뉴런 인덱스, 시작 ms, 끝 ms), ...] 시간 순, 겹치지 않게. 구간 밖에서는 자극 없음 (뇌 상태는 이어짐).
    반환: (뉴런별 스파이크 수 전체, 구간별 스파이크 수 (구간 수, 뉴런 수)). 구간 k = k 시작부터 k+1 시작(마지막은 끝)까지.

    align_noise: 기본(False)은 자극 집합 크기만큼 뽑으므로, 자극 집합이 다르면 **같은 seed라도 공유 뉴런이 받는
    잡음이 다르다**. True면 매 스텝 전체 뉴런 수만큼 뽑아 번호로 색인하므로, 자극 집합이 달라도 같은 뉴런은
    같은 입력을 받는다 → 자극만 바꾼 비교를 할 때 쓴다. 학습·추론 경로는 전부 기본값(False)이고
    (out/kc_codes_* 캐시도 그 값으로 만들어졌다), True는 분석 전용이다."""
    W = W.tocsc()
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    stims = [np.asarray(s) for s, _, _ in schedule]
    starts = [round(t0 / DT) for _, t0, _ in schedule]
    ends = [round(t1 / DT) for _, _, t1 in schedule]

    base_refr = round(REFRACTORY_MS / DT)
    refr = np.full(n, base_refr)
    delay = round(DELAY_MS / DT)
    em, eg = np.exp(-DT / TAU_V), np.exp(-DT / TAU_G)
    a = TAU_G / (TAU_G - TAU_V)
    p_kick = rate_hz * DT / 1000

    v = np.full(n, V_REST)
    g = np.zeros(n)
    ready = np.zeros(n, dtype=np.int64)
    counts = np.zeros(n, dtype=np.int64)
    seg_counts = np.zeros((len(schedule), n), dtype=np.int64)
    buf = np.zeros((delay, n))
    k, stim_idx = -1, None

    for step in range(round(duration_ms / DT)):
        if k + 1 < len(schedule) and step >= starts[k + 1]:
            k += 1
            stim_idx = stims[k]
            refr[:] = base_refr
            refr[stim_idx] = 0
        if stim_idx is not None and step >= ends[k]:
            refr[stim_idx] = base_refr
            stim_idx = None

        nr = step >= ready
        v = np.where(nr, V_REST + (v - V_REST - a * g) * em + a * g * eg, v)
        g = np.where(nr, g * eg, g)
        spiking = np.flatnonzero(nr & (v > V_THRESH))

        slot = step % delay
        g += np.where(nr, buf[slot], 0.0)
        buf[slot] = 0.0
        if align_noise:  # 자극 여부와 무관하게 매 스텝 n개를 뽑아야 뉴런 번호 - 난수가 어긋나지 않는다
            u = rng.random(n)
            if stim_idx is not None:
                v[stim_idx] += POISSON_KICK_MV * (u[stim_idx] < p_kick)
        elif stim_idx is not None:
            v[stim_idx] += POISSON_KICK_MV * (rng.random(stim_idx.size) < p_kick)

        if spiking.size:
            counts[spiking] += 1
            if k >= 0:
                seg_counts[k, spiking] += 1
            v[spiking] = V_REST
            g[spiking] = 0.0
            ready[spiking] = step + 1 + refr[spiking]
            out = W[:, spiking]
            buf[slot] += np.bincount(out.indices, weights=out.data, minlength=n)

    return counts, seg_counts
