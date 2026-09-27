#!/usr/bin/env python3
"""MaleCNS 버섯체 부분망: 포켓몬 특징 -> 투사뉴런(ALPN) Poisson 입력 -> Kenyon cell 스파이크 코드.

부분망 = ALPN, Kenyon cell, MBON, DAN, APL, DPM 사이의 연결만 (바깥 뉴런은 끈 것과 같음).
특징 하나당 투사뉴런 PNS_PER_FEATURE개를 시드 고정 무작위로 배정한다. 후보는 Kenyon cell에 실제로 신호를 보내는
투사뉴런만이다 (input_pns). 투사뉴런 전체에 배정하면 억제성 투사뉴런 등 절반 넘게 버섯체에 거의 닿지 않는다.

  python src/mb.py    # 부분망 크기, Kenyon cell 희소도, 속도 확인
"""
import hashlib
import pathlib
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from connectome import DATA_DIR
from model_io import EXT as MODEL_EXT
from simulate import CONTACT_MV, build_weight_matrix, run, run_schedule

MB_CLASSES = ["ALPN", "Kenyon_Cell", "MBON", "DAN"]
MB_TYPES = ["APL", "DPM"]
# 투사뉴런 686개의 Kenyon cell 흥분성 시냅스 수: 0개 372, 1~49개 71, 50~99개 6, 100개 이상 237 (억제성 부호 11).
# 둘로 갈리는 골짜기(50)를 기준으로 삼는다 -> 241개. 0개인 372개 중 203개가 억제성(GABA) 투사뉴런.
PN_MIN_KC_SYNAPSES = 50
# 개체 24마리 표본의 발화한 Kenyon cell 비율: 특징당 5개 5.7%, 7개 8.5%, 10개 12.1%, 14개 17.8%, 20개 37% (희소 코드가 깨짐).
# 5개: 가장 희소한 쪽.
PNS_PER_FEATURE = 5
RATE_HZ = 150
# 1초: 200ms면 APL 억제로 문턱 근처에 걸린 Kenyon cell이 시드마다 바뀌어 같은 입력끼리 스파이크 수 상관 0.87, 1초면 0.97.
# (200ms x 5회 합산도 0.96으로 같지만, 매번 뇌를 초기화하는 것보다 냄새를 계속 맡는 쪽이 자연스럽다.
#  Shiu 모델엔 적응이 없어 1초 내내 같은 세기로 반응한다는 점은 실제 초파리와 다르다.)
DURATION_MS = 1000
# 한 번의 시뮬레이션 안에서 특징을 차례로 자극할 때 특징 사이 쉬는 구간. 막 20ms·시냅스 5ms 시간상수보다 충분히 길다.
# 표본 30마리: 특징당 200ms면 순서를 섞어도 코드 상관 0.827로 잡음 기준(같은 순서·다른 시드) 0.832와 같다. 500ms는 0.916 vs 0.930
SEQ_GAP_MS = 100
# 시험 개체용 새 잡음 코드의 시드 시작값 (학습 0~, 예전 TEST 10^6~와 겹치지 않음). 같은 입력을 다시 맡은 반응.
# 학습 코드를 그대로 쓰면 잡음까지 같아 교차검증 점수가 부풀려진다
FRESH_SEED0 = 2_000_000
# 순차 평가에서 새로 들어오는 페이스트는 개체마다 따로 시뮬레이션 (시드 = ENCOUNTER_SEED0 + 개체 행 번호). 같은 입력이 다시 와도 새 반응
ENCOUNTER_SEED0 = 3_000_000
# TEST 개체도 개체마다 새 반응 (시드 = TEST_SEED0 + TEST 행 번호). 위 시드들과 겹치지 않음
TEST_SEED0 = 4_000_000
# 팀 자극 (그 마리 + 팀원 5마리 특징의 투사뉴런 합집합). 학습 코드는 고유 팀 입력마다 TEAM_SEED0 + 번호,
# 시험 코드는 개체마다 새 반응 TEAM_ENCOUNTER_SEED0 + 행. 투사뉴런 약 54%가 켜져 150Hz면 Kenyon cell 약 80%가 발화(폭주)하므로
# 세기를 낮춰 발화 비율을 한 마리 자극(약 5.5%)과 같은 5%로 맞춘다: 40팀 표본 60Hz 4.7%, 62Hz 5.2% -> 61Hz
TEAM_SEED0 = 5_000_000
TEAM_ENCOUNTER_SEED0 = 6_000_000
TEAM_RATE_HZ = 61

# 학습률 후보와 모델 파일 이름. 파일 이름 규칙이 추론에도 필요해서 여기 둔다 (추론이 학습 모듈을 안 부르게).
ETAS = [1e-4, 3e-5, 1e-5]              # mb_learn 기본 후보
FINAL_ETAS = [1e-4, 3e-5, 1e-5, 3e-6]  # mb_final 기본
TAG = f"{DURATION_MS}ms_pn{PN_MIN_KC_SYNAPSES}x{PNS_PER_FEATURE}"
DEFAULT_TAG = "1000ms_pn50x5"  # 기본 설정의 TAG. 여기서 벗어나면 모델 이름에 해시가 붙는다


def model_path(kind="updown", team=True, select="logloss", select_ev=None, etas=FINAL_ETAS):
    """기본 설정이면 out/fly_final.safetensors. 설정이 다르면 다른 이름이 되어 덮어쓰지 않는다.

    자주 바꾸는 것(성격 출력층·팀 자극·검증 채점)만 이름에 그대로 적고, 나머지(TAG, 학습률 후보)는
    다를 때만 짧은 해시를 붙인다. 전체 설정은 파일 안 메타데이터에 들어 있으므로 이름은 구분만 하면 된다."""
    select_ev = select_ev or select
    bits = []
    if kind != "updown":
        bits.append(kind)
    if not team:
        bits.append("no-team")
    if (select, select_ev) != ("logloss", "logloss"):
        bits.append(f"sel-{select}" if select == select_ev else f"sel-nat-{select}-ev-{select_ev}")
    if list(etas) != list(FINAL_ETAS) or TAG != DEFAULT_TAG:
        seed = f"{TAG}|" + ",".join(f"{e:g}" for e in etas)
        bits.append(hashlib.sha1(seed.encode()).hexdigest()[:6])
    return f"out/fly_final{''.join('_' + b for b in bits)}{MODEL_EXT}"


SEQ_KIND_ORDER = ["sp", "it", "ab", "mv"]  # OTS 읽는 순서: 종족 -> 아이템 -> 특성 -> 기술 (같은 종류는 특징 번호 순)
CACHE = pathlib.Path("out/mb_graph.npz")


def load_mb():
    """(W, groups): W는 부분망 가중치(CSC), groups는 역할별 행렬 인덱스."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        W = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        return W, z["groups"].item()

    ann = pd.read_feather(DATA_DIR / "annotations.feather")
    nodes = ann[ann.superclass.notna()].set_index("bodyId")
    keep = nodes["class"].isin(MB_CLASSES) | nodes.type.isin(MB_TYPES)
    nodes = nodes[keep]
    edges = pd.read_feather(DATA_DIR / "edges.feather")
    edges = edges[edges.body_pre.isin(nodes.index) & edges.body_post.isin(nodes.index)]
    nt = pd.read_feather(DATA_DIR / "neurotransmitters.feather")
    W, idx = build_weight_matrix(nodes, edges, nt)

    role = nodes["class"].where(nodes["class"].isin(MB_CLASSES), nodes.type)
    groups = {r: idx[role == r].to_numpy() for r in MB_CLASSES + MB_TYPES}
    groups["bodyId"] = nodes.index.to_numpy()
    np.savez(CACHE, data=W.data, indices=W.indices, indptr=W.indptr, shape=W.shape,
             groups=np.array(groups, dtype=object))
    return W, groups


def shuffle_pn_kc(W, groups, seed=0):
    """투사뉴런->Kenyon cell 연결의 도착 Kenyon cell만 섞은 W. 투사뉴런별 출력 수·가중치, APL 등 나머지는 그대로."""
    c = W.tocoo()
    pn = np.zeros(W.shape[0], dtype=bool)
    kc = np.zeros(W.shape[0], dtype=bool)
    pn[groups["ALPN"]] = True
    kc[groups["Kenyon_Cell"]] = True
    block = pn[c.col] & kc[c.row]
    rows = c.row.copy()
    rows[block] = np.random.default_rng(seed).permutation(rows[block])
    return sp.coo_matrix((c.data, (rows, c.col)), shape=W.shape).tocsc()


def input_pns(W, groups):
    """Kenyon cell로 흥분성 시냅스를 PN_MIN_KC_SYNAPSES개 이상 보내는 투사뉴런.
    shuffle_pn_kc는 투사뉴런별 출력 수·가중치를 보존하므로 섞은 배선에서도 같은 투사뉴런이 뽑힌다."""
    W = W.tocsc()
    kc = np.zeros(W.shape[0], dtype=bool)
    kc[groups["Kenyon_Cell"]] = True
    pns = groups["ALPN"]
    syn = np.array([W.data[W.indptr[j]:W.indptr[j + 1]][kc[W.indices[W.indptr[j]:W.indptr[j + 1]]]].sum()
                    for j in pns]) / CONTACT_MV
    return pns[syn >= PN_MIN_KC_SYNAPSES]


def pn_channels(n_features, pns, seed=0):
    """특징 i -> 투사뉴런 인덱스 PNS_PER_FEATURE개."""
    rng = np.random.default_rng(seed)
    return [rng.choice(pns, PNS_PER_FEATURE, replace=False) for _ in range(n_features)]


def kc_code(W, groups, channels, active_features, seed=0, rate=RATE_HZ):
    """활성 특징들의 투사뉴런을 자극했을 때 Kenyon cell별 스파이크 수. rate = 자극 투사뉴런 Poisson 입력 세기(Hz)."""
    stim = np.unique(np.concatenate([channels[f] for f in active_features]))
    return run(W, stim, rate, DURATION_MS, seed=seed)[groups["Kenyon_Cell"]]


def kc_codes(X, W, groups, cache, n_jobs=8, seed0=0, rate=RATE_HZ):
    """(고유 입력별 Kenyon cell 코드 uint16, 개체 -> 고유 입력 인덱스). 고유 입력마다 한 번만 시뮬레이션.
    시드 = seed0 + 고유 입력 번호. seed0=FRESH_SEED0이면 같은 입력의 새 잡음 코드.
    캐시는 시드·특징 수(투사뉴런 배정)·고유 입력 목록이 모두 같을 때만 쓴다."""
    import hashlib

    from joblib import Parallel, delayed

    rows = [tuple(np.sort(X[i].indices)) for i in range(X.shape[0])]
    uniq = sorted(set(rows))
    pos = {u: j for j, u in enumerate(uniq)}
    inv = np.array([pos[r] for r in rows])
    tag = f"{seed0}|{X.shape[1]}|" + ("" if rate == RATE_HZ else f"{rate}Hz|")  # 기본 세기면 예전 캐시 키 그대로
    key = hashlib.sha1(tag.encode() + b"".join(np.asarray(u, np.int64).tobytes() + b"|" for u in uniq)).hexdigest()
    cache = pathlib.Path(cache)
    if cache.exists():
        z = np.load(cache)
        if "key" in z.files and str(z["key"]) == key:
            return z["codes"], inv

    ch = pn_channels(X.shape[1], input_pns(W, groups))
    t0, codes, step = time.time(), [], max(1, len(uniq) // 20)
    for j, c in enumerate(Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(kc_code)(W, groups, ch, list(u), seed=seed0 + j, rate=rate) for j, u in enumerate(uniq)), 1):
        codes.append(c)
        if j % step == 0 or j == len(uniq):
            print(f"  Kenyon cell 코드 {j:,}/{len(uniq):,} (경과 {(time.time() - t0) / 60:.1f}분)", flush=True)
    codes = np.stack(codes).astype(np.uint16)
    np.savez_compressed(cache, codes=codes, key=key)
    return codes, inv


def kc_row_codes(X, rows, W, groups, cache, seed0=ENCOUNTER_SEED0, n_jobs=8, rate=RATE_HZ):
    """개체 행마다 따로 시뮬레이션한 코드 (len(rows), Kenyon cell 수) uint16. 시드 = seed0 + 행 번호 (입력이 같아도 행마다 새 잡음).
    캐시는 행 번호·시드·특징 수(투사뉴런 배정)와 입력(행별 활성 특징 번호)이 같을 때만 쓴다."""
    import hashlib

    from joblib import Parallel, delayed

    rows = np.asarray(rows)
    tag = f"{seed0}|{X.shape[1]}|" + ("" if rate == RATE_HZ else f"{rate}Hz|")
    key = hashlib.sha1(tag.encode() + b"".join(np.asarray(np.sort(X[int(r)].indices), np.int64).tobytes() + b"|" for r in rows)).hexdigest()
    cache = pathlib.Path(cache)
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["rows"], rows) and "key" in z.files and str(z["key"]) == key:
            return z["codes"]
    ch = pn_channels(X.shape[1], input_pns(W, groups))
    t0, codes, step = time.time(), [], max(1, len(rows) // 20)
    for k, c in enumerate(Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(kc_code)(W, groups, ch, list(X[r].indices), seed=seed0 + int(r), rate=rate) for r in rows), 1):
        codes.append(c)
        if k % step == 0 or k == len(rows):
            print(f"  개체별 Kenyon cell 코드 {k:,}/{len(rows):,} (경과 {(time.time() - t0) / 60:.1f}분)", flush=True)
    codes = np.stack(codes).astype(np.uint16)
    np.savez_compressed(cache, codes=codes, rows=rows, key=key)
    return codes


def kc_feature_codes(vocab, W, groups, cache, n_jobs=8):
    """특징마다 투사뉴런만 따로 자극한 Kenyon cell 코드 (특징 수, Kenyon cell 수) uint16. 특징 i는 시드 i.
    개체 코드 = 가진 특징들의 코드 합 (특징을 한 번에 하나씩 차례로 맡고 MBON이 시행 동안 적분).
    한꺼번에 자극하면 특징끼리 투사뉴런·Kenyon cell 경쟁에서 섞여 정보를 잃는다."""
    from joblib import Parallel, delayed

    cache = pathlib.Path(cache)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["vocab"]) == list(vocab):
            return z["codes"]

    ch = pn_channels(len(vocab), input_pns(W, groups))
    t0, codes, step = time.time(), [], max(1, len(vocab) // 10)
    for i, c in enumerate(Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(kc_code)(W, groups, ch, [f], seed=f) for f in range(len(vocab))), 1):
        codes.append(c)
        if i % step == 0 or i == len(vocab):
            print(f"  특징별 Kenyon cell 코드 {i:,}/{len(vocab):,} (경과 {(time.time() - t0) / 60:.1f}분)", flush=True)
    codes = np.stack(codes).astype(np.uint16)
    np.savez_compressed(cache, codes=codes, vocab=np.array(vocab, dtype=object))
    return codes


def sequence_order(active_features, vocab):
    rank = {k: i for i, k in enumerate(SEQ_KIND_ORDER)}
    return sorted(active_features, key=lambda f: (rank[vocab[f][:2]], f))


def kc_sequence_code(W, groups, channels, ordered_features, stim_ms, seed=0):
    """한 번의 시뮬레이션 안에서 특징마다 stim_ms 자극 + SEQ_GAP_MS 쉼을 차례로 (뇌 상태 이어짐). Kenyon cell별 전체 스파이크 수."""
    period = stim_ms + SEQ_GAP_MS
    schedule = [(channels[f], k * period, k * period + stim_ms) for k, f in enumerate(ordered_features)]
    return run_schedule(W, schedule, RATE_HZ, len(ordered_features) * period, seed)[0][groups["Kenyon_Cell"]]


def kc_sequence_codes(X, vocab, W, groups, cache, stim_ms, n_jobs=8):
    """kc_codes와 같은 형식 (고유 입력별 코드 uint16, 개체 -> 고유 입력 인덱스). 입력마다 특징을 차례로 자극."""
    from joblib import Parallel, delayed

    rows = [tuple(np.sort(X[i].indices)) for i in range(X.shape[0])]
    uniq = sorted(set(rows))
    pos = {u: j for j, u in enumerate(uniq)}
    inv = np.array([pos[r] for r in rows])
    keys = np.array([" ".join(vocab[f] for f in u) for u in uniq], dtype=object)
    cache = pathlib.Path(cache)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if np.array_equal(z["inputs"], keys):
            return z["codes"], inv

    ch = pn_channels(X.shape[1], input_pns(W, groups))
    t0, codes, step = time.time(), [], max(1, len(uniq) // 20)
    for j, c in enumerate(Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(kc_sequence_code)(W, groups, ch, sequence_order(u, vocab), stim_ms, seed=j) for j, u in enumerate(uniq)), 1):
        codes.append(c)
        if j % step == 0 or j == len(uniq):
            print(f"  차례 자극({stim_ms}ms) Kenyon cell 코드 {j:,}/{len(uniq):,} (경과 {(time.time() - t0) / 60:.1f}분)", flush=True)
    codes = np.stack(codes).astype(np.uint16)
    np.savez_compressed(cache, codes=codes, inputs=keys)
    return codes, inv


def main():
    from mb_task import load_mons, features

    t0 = time.time()
    W, groups = load_mb()
    print(f"부분망 로드 {time.time() - t0:.1f}s | 뉴런 {W.shape[0]:,} | 연결 {W.nnz:,}")
    for r in MB_CLASSES + MB_TYPES:
        print(f"  {r:12s} {len(groups[r]):5d}")

    mo = load_mons()
    X, vocab = features(mo)
    pns = input_pns(W, groups)
    ch = pn_channels(len(vocab), pns)
    sample = np.random.default_rng(0).choice(len(mo), 20, replace=False)

    t0 = time.time()
    frac, spikes, pn_active = [], [], []
    for i in sample:
        act = X[i].indices
        pn_active.append(len(np.unique(np.concatenate([ch[f] for f in act]))))
        c = kc_code(W, groups, ch, act, seed=int(i))
        frac.append((c > 0).mean())
        spikes.append(c.sum())
    per = (time.time() - t0) / len(sample)
    print(f"\n특징 1개당 투사뉴런 {PNS_PER_FEATURE}개, {RATE_HZ}Hz, {DURATION_MS}ms, 개체 {len(sample)}개 표본")
    print(f"  자극된 투사뉴런 수 평균 {np.mean(pn_active):.0f} / 후보 {len(pns)} (전체 {len(groups['ALPN'])})")
    print(f"  발화한 Kenyon cell 비율 평균 {np.mean(frac):.1%} (범위 {np.min(frac):.1%}~{np.max(frac):.1%})")
    print(f"  Kenyon cell 스파이크 합 평균 {np.mean(spikes):,.0f}")
    print(f"  개체당 {per:.2f}s -> 고유 입력 {len({tuple(X[i].indices) for i in range(len(mo))}):,}개 전체 예상 "
          f"{per * len({tuple(X[i].indices) for i in range(len(mo))}) / 60:.0f}분 (단일 코어)")


if __name__ == "__main__":
    main()
