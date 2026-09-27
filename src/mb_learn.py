#!/usr/bin/env python3
"""버섯체로 페이스트(성격 + EV) 고르기. 성격과 EV 모두 도파민 규칙으로 학습한다.

출력 뉴런 묶음: 묶음마다 가장 덜 흥분한 뉴런이 답.
  점수 = 흥분성 b - Kenyon cell 발화율 @ W - 맥락 입력 @ U   (W, U >= 0, 0에서 시작)
  도파민 = 묶음별 softmax(점수) - 정답 표시  (크로스엔트로피 기울기, 바깥에서 뉴런마다 준다)
  W += eta x 발화율 x 도파민,  U += eta x 맥락 x 도파민,  b -= eta x 도파민   (한 개체씩)
성격: 실제 MBON을 성격 21종 구획으로 나누고, 실제로 연결된 Kenyon cell->MBON 쌍만 학습한다.
EV: 스탯마다 0~32 값 하나씩 가상 출력 뉴런 33개 (6 x 33 = 198개, MBON 97개로는 담을 수 없어 실제 초파리에 없는 뉴런).
  모든 Kenyon cell에서 입력을 받고, 성격의 올림/내림 표시 10개는 Kenyon cell을 거치지 않는 맥락 뉴런으로 들어간다.
  맥락 뉴런 최대 발화율 = Kenyon cell 코드 전체 세기 sqrt(평균 발화율 제곱합, 약 146Hz): 맥락 쪽 한 번 학습이 점수를 바꾸는 양이
  Kenyon cell 쪽과 같다.
  학습 때 맥락은 실제 성격(최대 세기). 검증·시험 때는 초파리 성격 출력층의 확률만큼 발화한다:
  맥락 = 최대 발화율 x sum_성격 P(성격) x 올림/내림 표시. 확신하면 최대 세기, 헷갈리면 후보 성격들의 표시가 나눠서 약하게 켜진다.
  묶음별 softmax를 확률로 보고 합 66·스탯당 32 이하 조합을 동적계획법으로 찾는다.
고르는 방식 (같은 EV 출력층, 추가 학습 없음). 성격은 성격 출력층에서 확률이 가장 높은 것, 단 "완전히 함께"·"성격 앎"은 예외:
  성격 먼저 -> EV (성격 확률만큼 맥락): 위의 확률 가중 맥락으로 EV 배분. 검증 기준도 이것
  성격 먼저 -> EV (확정 성격 맥락): 가장 높은 성격 하나를 최대 세기로
  완전히 함께: 성격 후보마다 확정 맥락으로 EV 배분을 뽑아 log P(성격) + log P(배분 | 성격)이 가장 큰 쌍
  성격 앎 -> EV (정답 성격 맥락): 성격을 알려 준 경우. 성격 출력층을 건너뛰고 정답 성격을 최대 세기로 (학습 때와 같은 맥락)
학습률·반복 수는 학습 fold 안의 검증용 아키타입(15%)으로 고른다.
  --select error(기본): 성격은 검증 오답률, EV는 확률이 가장 높은 배분의 EV 오차로 고른다.
  --select logloss: 성격은 정답 성격 −log 확률 평균, EV는 스탯별 정답 값 −log 확률 합의 평균 (확률 하한 1e-4).
  --select-ev: EV만 따로 고르는 채점 (기본은 --select와 같음). 예: --select logloss --select-ev error
학습률은 큰 값부터 시도하고, 검증 오차가 PATIENCE회 동안 새 최저를 못 찍으면 멈춘다.
상한까지 계속 나아진 학습률이 나오면 더 작은 학습률은 건너뛴다. 뽑힌 값이 후보 끝값이거나 상한까지 나아진 경우 경고한다.
Kenyon cell 코드: Kenyon cell에 연결된 투사뉴런(mb.input_pns)을 DURATION_MS(1초) 자극한 발화율(Hz). 입력은 그 마리만.
  --code together(기본): 특징 7개의 투사뉴런을 한꺼번에 자극. --code perfeature: 특징마다 따로 1초씩 자극한 스파이크 수의 합 (mb.kc_feature_codes).
  --code sequence: 한 번의 시뮬레이션 안에서 특징마다 --seq-ms 자극 + 쉼을 차례로 (mb.kc_sequence_codes), 발화율 = 스파이크 수 / 특징당 자극 시간.
시험 예측은 out/mb_preds_*.parquet에 저장한다 (다시 돌리지 않고 분석하기 위해).
시험 fold는 두 가지 코드로 채점한다 (학습·검증은 저장된 코드 그대로):
  같은 잡음: 학습과 같은 저장 코드. 입력이 같으면 잡음까지 같아 점수가 부풀려진다
  새 반응 (방법 이름 끝 [새 반응]): 개체마다 따로 다시 시뮬레이션한 코드 (시드 mb.ENCOUNTER_SEED0 + 행, mb_stream·mb_shift와 같은 코드). 기본 지표
  새 반응 2번 평균 (--test-draws 2): 위 새 반응과 고유 입력마다 한 번 뽑은 코드의 평균 = 두 번 맡은 반응. 둘 다 학습 잡음과 독립
  --code together만 새 반응을 계산한다.
학습 반응 섞기 (--train-draws 2): 개체를 배울 때마다 두 코드 중 하나를 무작위로 쓴다 (같은 입력을 다시 만날 때 다른 반응).
  한 회차에 배우는 수는 같다. 검증은 원래 코드를 쓴다.
성격 확률 온도 (--ctx-temps): EV 맥락의 성격 확률을 softmax(log p / T)로 바꿔 채점을 추가한다 (학습은 그대로).

기본은 초파리(도파민 학습)만 돌린다. 목표는 초파리 성능을 끌어올리는 것이다.
초파리가 아닌 비교 기준(종족별 최빈값, OTS 특징 + 로지스틱 회귀, 버섯체 코드 + 로지스틱 회귀)은 필요할 때만 --compare로 켠다.
섞은 배선(투사뉴런->Kenyon cell 도착 Kenyon cell만 섞음)은 찾은 방법이 초파리 배선을 활용하는지 확인할 때만 --shuffle로 켠다.

  python src/mb_learn.py [--compare] [--shuffle]
"""
import argparse
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.special import log_softmax, softmax
from sklearn.model_selection import GroupShuffleSplit

from mb import (DURATION_MS, ENCOUNTER_SEED0, ETAS, FRESH_SEED0, PN_MIN_KC_SYNAPSES, PNS_PER_FEATURE, SEQ_GAP_MS, TEAM_ENCOUNTER_SEED0, TEAM_RATE_HZ, TEAM_SEED0,
                kc_codes, kc_feature_codes, kc_row_codes, kc_sequence_codes, load_mb, shuffle_pn_kc)
from mb_task import EV, NATURE, NEW_SOURCE, OLD_SOURCE, err, features, folds, load_mons, logreg_classes, species_mode_nature, species_mode_predict, team_union_features
from simulate import CONTACT_MV
from spread import (EV_CAP, NATURE_EFFECT, STATS, best_spread, nature_axes, ev_given_nature, expected_error, fine_value_hits, format_table,
                    nature_bits, paste_scores)

# 큰 값부터 시도한다
MAX_EPOCHS = 300
PATIENCE = 50  # 회차 사이 검증 오차가 흔들려서 짧게 잡지 않는다
N_JOBS = 8
LOGP_FLOOR = np.log(1e-4)  # --select logloss: 한 개체가 검증 손실을 좌우하지 않게 확률 하한


def kc_mbon_synapses(W, groups):
    """(Kenyon cell x MBON) 시냅스 접촉 수."""
    W = W.tocsr()
    return (abs(W[groups["MBON"]][:, groups["Kenyon_Cell"]]).T / CONTACT_MV).toarray().astype(np.float32)


def compartments(S, k):
    """MBON을 Kenyon cell 입력 수가 비슷하도록 k개 구획에 배분 -> (MBON x k) 소속 행렬."""
    kc_in = (S > 0).sum(0)
    load = np.zeros(k)
    M = np.zeros((S.shape[1], k), dtype=np.float32)
    for m in np.argsort(-kc_in, kind="stable"):
        if kc_in[m]:
            c = load.argmin()
            M[m, c] = 1
            load[c] += kc_in[m]
    return M


class DopamineOutput:
    """출력 뉴런 n_groups x group_size개. conn: (Kenyon cell x 출력) 실제 연결 여부 (None이면 모두 연결)."""

    def __init__(self, n_kc, n_groups, group_size, eta, conn=None, n_ctx=0, target=None, decay=0.0):
        """target: (group_size, group_size) 정답 값 -> 도파민에서 뺄 분포. None이면 정답 뉴런만 1.
        decay: 개체 하나를 배울 때마다 모든 Kenyon cell->출력 시냅스가 (1-decay)배로 준다 (시냅스 감쇠).
        실제로는 건드리는 행만 '마지막으로 쓴 뒤 지난 개체 수'만큼 한꺼번에 줄여서 같은 결과를 싸게 낸다.
        드물게 나오는 특징은 갱신 사이 간격이 길어 가중치가 덜 쌓인다."""
        n_out = n_groups * group_size
        self.shape, self.eta, self.conn, self.target = (n_groups, group_size), eta, conn, target
        self.decay, self.t, self.last = _decay_vec(decay, n_kc), 0, np.zeros(n_kc, np.int64)
        self.w = np.zeros((n_kc, n_out))
        self.u = np.zeros((n_ctx, n_out))
        self.b = np.zeros(n_out)

    def _catch_up(self):
        """밀린 감쇠를 모든 행에 반영."""
        if self.decay is not None:
            self.w *= (1.0 - self.decay[:, None]) ** (self.t - self.last)[:, None]
            self.last[:] = self.t

    def scores(self, rates, ctx=None):
        """(n, n_groups, group_size)."""
        self._catch_up()
        s = self.b - rates @ self.w
        if ctx is not None:
            s = s - ctx @ self.u
        return s.reshape(-1, *self.shape)

    def practise(self, active, rates, truth, ctx=None):
        if self.decay is not None:
            self.t += 1
            self.w[active] *= (1.0 - self.decay[active, None]) ** (self.t - self.last[active])[:, None]
            self.last[active] = self.t
        s = self.b - rates @ self.w[active]
        if ctx is not None:
            s = s - ctx @ self.u
        dopamine = softmax(s.reshape(self.shape), axis=1)
        if getattr(self, "target", None) is None:
            dopamine[np.arange(self.shape[0]), truth] -= 1
        else:
            dopamine -= self.target[truth]
        dopamine = dopamine.ravel()
        w = np.maximum(self.w[active] + self.eta * np.outer(rates, dopamine), 0)
        self.w[active] = w if self.conn is None else w * self.conn[active]
        if ctx is not None:
            self.u = np.maximum(self.u + self.eta * np.outer(ctx, dopamine), 0)
        self.b -= self.eta * dopamine


def ev_target(sigma):
    """이웃 값도 배우는 EV 도파민 정답: 정답 y에서 값 v까지 exp(-(v-y)^2 / 2σ^2)를 0~32 안에서 합 1로. σ=0이면 None (정답만).
    예: σ=1이면 정답 0.40, ±1 0.24, ±2 0.05. 정답 근처 출력 뉴런도 거리에 따라 억제가 조금 풀린다."""
    if not sigma:
        return None
    v = np.arange(EV_CAP + 1)
    k = np.exp(-(v[None, :] - v[:, None]) ** 2 / (2 * sigma ** 2))
    return k / k.sum(1, keepdims=True)


NATURE_KIND = "updown"  # 올림·내림 12구획


def nature_output_setup(S, natures, kind=NATURE_KIND):
    """(conn, axes): 성격 출력층의 (Kenyon cell x 구획) 실제 연결 여부와 올림·내림 번호 (21구획이면 axes는 None)."""
    if kind == "updown":
        return ((S @ compartments(S, 12)) > 0).astype(np.float64), nature_axes(natures)
    return ((S @ compartments(S, len(natures))) > 0).astype(np.float64), None


def _decay_vec(decay, n_kc):
    """감쇠를 Kenyon cell마다의 값으로. 0이거나 전부 0이면 None (감쇠 안 함)."""
    if decay is None:
        return None
    d = np.asarray(decay, float)
    if not np.any(d > 0):
        return None
    return np.full(n_kc, float(d)) if d.ndim == 0 else d.astype(float)


DECAY = 0.0  # 시냅스 감쇠 (0이면 지금까지와 같음). mb_learn/mb_shift의 --decay가 바꾼다


def nature_output(n_kc, natures, eta, conn, axes, decay=0.0):
    return (UpDownNatureOutput(n_kc, axes, eta, conn=conn, decay=decay) if axes is not None
            else DopamineOutput(n_kc, 1, len(natures), eta, conn=conn, decay=decay))


class UpDownNatureOutput:
    """성격 출력을 올림 구획 6개 + 내림 구획 6개(MBON 12구획)로 나눈다.
    점수(성격) = b(성격) - 발화율 @ W[:, 올림(성격)] - 발화율 @ W[:, 6 + 내림(성격)].
    도파민(구획) = 그 구획을 쓰는 성격들의 softmax 확률 합 - 정답 표시. 예: "특공 내림" 구획은 고집·명랑·장난꾸러기·신중이 함께 배운다."""

    def __init__(self, n_kc, axes, eta, conn=None, decay=0.0):
        self.up, self.down = axes
        n_nat = len(self.up)
        self.shape, self.eta, self.conn = (1, n_nat), eta, conn
        self.decay, self.t, self.last = _decay_vec(decay, n_kc), 0, np.zeros(n_kc, np.int64)
        self.w = np.zeros((n_kc, 12))
        self.u = np.zeros((0, n_nat))
        self.b = np.zeros(n_nat)
        self.M = np.zeros((n_nat, 12))  # 성격 -> 구획
        self.M[np.arange(n_nat), self.up] = 1
        self.M[np.arange(n_nat), 6 + self.down] = 1

    def _catch_up(self):
        if self.decay is not None:
            self.w *= (1.0 - self.decay[:, None]) ** (self.t - self.last)[:, None]
            self.last[:] = self.t

    def scores(self, rates, ctx=None):
        self._catch_up()
        return (self.b - (rates @ self.w) @ self.M.T).reshape(-1, *self.shape)

    def practise(self, active, rates, truth, ctx=None):
        if self.decay is not None:
            self.t += 1
            self.w[active] *= (1.0 - self.decay[active, None]) ** (self.t - self.last[active])[:, None]
            self.last[active] = self.t
        s = self.b - (rates @ self.w[active]) @ self.M.T
        dopamine = softmax(s)
        dopamine[truth[0]] -= 1
        w = np.maximum(self.w[active] + self.eta * np.outer(rates, dopamine @ self.M), 0)
        self.w[active] = w if self.conn is None else w * self.conn[active]
        self.b -= self.eta * dopamine


def run_jobs(jobs, what):
    """jobs: delayed(tagged)(key, ...) 목록 -> {key: 결과}. 끝날 때마다 진행률을 찍는다."""
    t0, got = time.time(), {}
    for k, (key, res) in enumerate(Parallel(n_jobs=N_JOBS, return_as="generator_unordered")(jobs), 1):
        got[key] = res
        print(f"[{what} {k}/{len(jobs)}] {key} 끝 (경과 {(time.time() - t0) / 60:.1f}분)", flush=True)
    return got


def tagged(key, fn, *args):
    return key, fn(*args)


def onehot_logp(ids, n):
    """정답 하나만 확률 1인 log 확률 (성격 공개 과제에서 확정 맥락으로 쓴다)."""
    lp = np.full((len(ids), n), -1e9)
    lp[np.arange(len(ids)), ids] = 0.0
    return lp


def inner_split(arch, tr, seed=0):
    """학습 fold -> (학습용 85%, 검증용 아키타입 15%). 성격·EV가 같은 분할을 쓴다."""
    itr, iva = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed).split(tr, groups=arch[tr]))
    return tr[itr], tr[iva]


def fit_dopamine(make, rates, targets, arch, tr, evaluate, ctx=None, seed=0, label="", n_jobs=1, etas=None, draw_offsets=None):
    """학습률·반복 수를 검증용 아키타입으로 고르고 학습 fold 전체로 다시 학습한다.
    evaluate(net) -> (검증 손실, 그 회차에 남길 값). 반환: (출력층, {val_loss, eta, epochs, capped, val_out}).
    n_jobs > 1이면 학습률들을 동시에 돌리고 "상한까지 나아진 학습률보다 작은 값은 건너뜀"을 끝나고 적용한다 (결과는 같다).
    etas: 학습률 후보 (큰 값부터, 없으면 ETAS).
    draw_offsets: 학습 반응 행 오프셋 목록 (예: [0, k x N]). 개체를 배울 때마다 그중 하나를 무작위로 골라 rates[i + 오프셋]을 쓴다
    (같은 입력의 다른 잡음 반응). None이면 rates[i]만. 검증은 늘 원래 행."""
    etas = ETAS if etas is None else etas
    itr, _ = inner_split(arch, tr, seed)
    t0 = time.time()

    def train(idx, eta, epochs, validate=False):
        net = make(eta)
        rng = np.random.default_rng(seed)
        draw_rng = np.random.default_rng([seed, 1]) if draw_offsets is not None else None  # 순서 난수와 따로 (개체 순서는 그대로)
        best = {"val_loss": np.inf, "epochs": 0, "val_out": None}
        for ep in range(1, epochs + 1):
            order = rng.permutation(idx)
            rows = order if draw_rng is None else order + np.asarray(draw_offsets)[draw_rng.integers(len(draw_offsets), size=len(order))]
            for i, r in zip(order, rows):
                lo, hi = rates.indptr[r], rates.indptr[r + 1]
                net.practise(rates.indices[lo:hi], rates.data[lo:hi], targets[i], None if ctx is None else ctx[i])
            if validate:
                loss, out = evaluate(net)
                if loss < best["val_loss"]:
                    best = {"val_loss": loss, "epochs": ep, "val_out": out}
                elif ep - best["epochs"] >= PATIENCE:
                    return net, best, False
        return net, best, validate

    def search(k, eta):
        _, best, capped = train(itr, eta, MAX_EPOCHS, validate=True)
        print(f"  {label} 학습률 {k + 1}/{len(etas)} (eta={eta:g}) 끝: 최저 {best['val_loss']:.4f} @ {best['epochs']}회"
              f"{' (상한까지 나아지는 중)' if capped else ''}, 경과 {(time.time() - t0) / 60:.1f}분", flush=True)
        return {**best, "eta": eta, "capped": capped}

    tried = []
    if n_jobs > 1:
        tried = Parallel(n_jobs=min(n_jobs, len(etas)))(delayed(search)(k, eta) for k, eta in enumerate(etas))
    for k, eta in enumerate(etas):
        if n_jobs <= 1:
            tried.append(search(k, eta))
        if tried[k]["capped"] and k + 1 < len(etas):
            print(f"  {label} 학습률 {eta:g}가 상한까지 나아져 더 작은 학습률은 {'쓰지 않음' if n_jobs > 1 else '건너뜀'}", flush=True)
            tried = tried[:k + 1]
            break

    fit = min(tried, key=lambda t: t["val_loss"])
    net = train(tr, fit["eta"], fit["epochs"])[0]
    print(f"  {label} 다시 학습 끝 (eta={fit['eta']:g}, {fit['epochs']}회), 경과 {(time.time() - t0) / 60:.1f}분", flush=True)
    return net, fit


def dopamine_spreads(net, rates, ctx):
    return best_spread(log_softmax(net.scores(rates, ctx), axis=2))[1]


def fit_nature(rates, nat_id, n_natures, conn, arch, tr, label="", n_jobs=1, axes=None, etas=None, draw_offsets=None, select="error",
               decay=None):
    """(성격 출력층, 검증용 개체 성격 log 확률, 학습 정보).
    검증용 값은 고른 학습률·반복 수에서 학습 fold 85%로 학습한 출력층의 것 (EV 검증에 쓴다).
    axes(nature_axes)를 주면 올림·내림 12구획 출력층 (conn은 (Kenyon cell, 12)), 없으면 성격마다 한 구획.
    select: 검증 채점 "error"(오답률) / "logloss"(정답 성격 −log 확률 평균)."""
    _, iva = inner_split(arch, tr)

    def evaluate(net):
        s = net.scores(rates[iva])[:, 0]
        lp = log_softmax(s, axis=1)
        if select == "logloss":
            return -np.maximum(lp[np.arange(len(iva)), nat_id[iva]], LOGP_FLOOR).mean(), lp
        return (s.argmax(1) != nat_id[iva]).mean(), lp

    dec = DECAY if decay is None else decay  # 닫힘변수로 잡아 워커 프로세스까지 값이 간다
    make = ((lambda eta: UpDownNatureOutput(rates.shape[1], axes, eta, conn=conn, decay=dec)) if axes is not None
            else (lambda eta: DopamineOutput(rates.shape[1], 1, n_natures, eta, conn=conn, decay=dec)))
    net, fit = fit_dopamine(make, rates, nat_id[:, None], arch, tr, evaluate, label=label, n_jobs=n_jobs, etas=etas, draw_offsets=draw_offsets)
    fit["kind"] = "updown" if axes is not None else "21"
    return net, fit.pop("val_out"), fit


def mb_nature_job(rates, nat_id, n_natures, conn, arch, tr, te, label="", axes=None, etas=None, draw_offsets=None, select="error",
                  decay=None):
    """(검증용 개체 성격 log 확률, 시험 개체 성격 log 확률 (n, 성격 수), 학습 정보)."""
    net, val_logp, fit = fit_nature(rates, nat_id, n_natures, conn, arch, tr, label, axes=axes, etas=etas, draw_offsets=draw_offsets,
                                    select=select, decay=decay)
    return val_logp, log_softmax(net.scores(rates[te])[:, 0], axis=1), fit


def context_rate(rates, tr):
    """Kenyon cell 코드 전체 세기 = 학습 fold의 sqrt(평균 발화율 제곱합)."""
    return float(np.sqrt(np.asarray(rates[tr].multiply(rates[tr]).sum(1)).mean()))


def fit_ev(rates, Y, bits, natures, val_nat_logp, arch, tr, ctx_rate, label="", n_jobs=1, sigma=0, etas=None, draw_offsets=None,
           select="error", decay=None):
    """(EV 출력층, 학습 정보). 학습 맥락 = 실제 성격 표시 x ctx_rate. 검증 맥락 = 초파리 성격 확률만큼 (확률 가중 표시 x ctx_rate).
    sigma > 0이면 이웃 값도 배운다 (ev_target). 검증은 σ와 무관하게
    select "error": 확률이 가장 높은 배분의 EV 오차 / "logloss": 스탯별 정답 값 −log 확률 합의 평균."""
    _, iva = inner_split(arch, tr)
    val_ctx = np.exp(val_nat_logp) @ (nature_bits(natures) * ctx_rate)

    def evaluate(net):
        if select == "logloss":
            lp = log_softmax(net.scores(rates[iva], val_ctx), axis=2)
            return -np.maximum(np.take_along_axis(lp, Y[iva][:, :, None], axis=2)[:, :, 0], LOGP_FLOOR).sum(1).mean(), None
        return err(dopamine_spreads(net, rates[iva], val_ctx), Y[iva]).mean(), None

    target = ev_target(sigma)
    dec = DECAY if decay is None else decay
    net, fit = fit_dopamine(lambda eta: DopamineOutput(rates.shape[1], 6, EV_CAP + 1, eta, n_ctx=bits.shape[1], target=target, decay=dec),
                            rates, Y, arch, tr, evaluate, ctx=bits * ctx_rate, label=label, n_jobs=n_jobs, etas=etas, draw_offsets=draw_offsets)
    fit.pop("val_out")
    return net, fit


def mb_ev_job(rates, Y, bits, natures, val_nat_logp, te_nat_logp, arch, tr, te, ctx_rate, label="", sigma=0, etas=None,
              te_nat_id=None, draw_offsets=None, temps=(), return_lp=False, select="error", decay=None):
    """반환: (확률 가중 맥락으로 고른 시험 배분 (n, 6), 같은 확률에서 기대 오차가 가장 작은 배분 (n, 6),
          성격 후보별 확정 맥락의 최대 log P(배분 | 성격) (n, m), 그 배분 (n, m, 6), 학습 정보,
          te_nat_id(시험 개체 정답 성격 번호)를 주면 정답 성격 맥락에서 기대 오차가 가장 작은 배분 (n, 6), 아니면 None,
          {T: (최빈 배분, 기대 오차 최소 배분)} — temps의 온도마다 성격 확률을 softmax(log p / T)로 날카롭게(T<1) 한 맥락).
    정답 성격 맥락의 최빈 배분은 spreads[행, te_nat_id]. T=1은 첫 두 값과 같고, T→0은 확정 성격 맥락에 가까워진다.
    return_lp면 끝에 확률 가중 맥락의 스탯별 log 확률 (n, 6, 33) float32와 정답 성격 맥락의 같은 확률(te_nat_id 없으면 None)을 붙인다
    (고르는 규칙을 학습 없이 바꿔 보기용)."""
    net, fit = fit_ev(rates, Y, bits, natures, val_nat_logp, arch, tr, ctx_rate, label, sigma=sigma, etas=etas, draw_offsets=draw_offsets,
                      select=select, decay=decay)
    cand = nature_bits(natures) * ctx_rate  # (m, 10)
    lp = log_softmax(net.scores(rates[te], np.exp(te_nat_logp) @ cand), axis=2)
    soft = best_spread(lp)[1]
    soft_min_err = best_spread(-expected_error(np.exp(lp)))[1]
    s = (net.b - np.asarray(rates[te] @ net.w))[:, None, :] - (cand @ net.u)[None]
    lp_cand = log_softmax(s.reshape(len(te), len(natures), 6, EV_CAP + 1), axis=3)
    score, spreads = best_spread(lp_cand)
    known_min_err = None
    if te_nat_id is not None:
        known_min_err = best_spread(-expected_error(np.exp(lp_cand[np.arange(len(te)), te_nat_id])))[1]
    sharp = {}
    for T in temps:
        lp_t = log_softmax(net.scores(rates[te], softmax(te_nat_logp / T, axis=1) @ cand), axis=2)
        sharp[T] = (best_spread(lp_t)[1], best_spread(-expected_error(np.exp(lp_t)))[1])
    def nature_from(ev):  # 배분을 먼저 확정하고, 그 배분이 가장 그럴듯한 성격으로 되돌려 고른다
        fit_lp = np.take_along_axis(lp_cand, ev[:, None, :, None], axis=3)[:, :, :, 0].sum(2)  # (n, 성격) log P(배분 | 성격)
        # (배분만, 성격 확률도 더함, 무게 λ를 나중에 바꿔 보려고 행렬 그대로도 돌려준다)
        return fit_lp.argmax(1), (te_nat_logp + fit_lp).argmax(1), fit_lp.astype(np.float32)

    back = {"최빈": nature_from(soft), "기대 오차 최소": nature_from(soft_min_err)}
    out = (soft, soft_min_err, score, spreads, fit, known_min_err, sharp, back)
    if not return_lp:
        return out
    known_lp = None if te_nat_id is None else lp_cand[np.arange(len(te)), te_nat_id].astype(np.float32)
    return (*out, lp.astype(np.float32), known_lp)


def report_nature(got, data, kinds, kind_label, F, X, mo, nat_id, natures):
    """성격 출력층끼리 시험 fold 성격 정확도 비교. 일반화를 보려고 학습 fold에 같은 입력이 없는 개체, 드문 종족(학습 fold 20마리 미만)을 따로."""
    rows = []
    up, down = nature_axes(natures)
    species = mo.species.to_numpy()
    for f, (tr, te) in enumerate(F):
        seen = {tuple(np.sort(X[i].indices)) for i in tr}
        fresh = np.array([tuple(np.sort(X[i].indices)) not in seen for i in te])
        n_sp = pd.Series(species[tr]).value_counts()
        rare = np.array([n_sp.get(s, 0) < 20 for s in species[te]])
        for name in data:
            for kind in kinds:
                pred = got[("nature", name, kind, f)][1][:len(te)].argmax(1)  # 앞쪽 = 같은 잡음
                hit, t = pred == nat_id[te], nat_id[te]
                rows.append({"출력층": f"{name}{kind_label[kind]}", "fold": f, "성격 정확도": hit.mean(),
                             "올림 스탯 정확도": (up[pred] == up[t]).mean(), "내림 스탯 정확도": (down[pred] == down[t]).mean(),
                             "같은 입력 없음": hit[fresh].mean(), "드문 종족(<20)": hit[rare].mean(),
                             "뽑힌 학습률·반복": f"{got[('nature', name, kind, f)][2]['eta']:g}/{got[('nature', name, kind, f)][2]['epochs']}"})
    t = pd.DataFrame(rows)
    num = ["성격 정확도", "올림 스탯 정확도", "내림 스탯 정확도", "같은 입력 없음", "드문 종족(<20)"]
    summary = t.groupby("출력층")[num].agg(["mean", "std"])
    print(f"\n성격만 archetype GroupKFold 5 (fold 평균 ± 표준편차). 같은 입력 없음 = 학습 fold에 입력이 완전히 같은 개체가 없음")
    print(pd.DataFrame({c: summary[c]["mean"].map("{:.1%}".format) + " ± " + summary[c]["std"].map("{:.1%}".format) for c in num}).to_string())
    print(t.pivot(index="fold", columns="출력층", values="뽑힌 학습률·반복").to_string())


def brainless_job(X, Y, nat, tr, te):
    na = logreg_classes(X, nat, tr, te)
    return ev_given_nature(X, Y, nat, tr, te, [na])[0], na


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffle", action="store_true", help="섞은 배선 대조군도 돌린다")
    ap.add_argument("--compare", action="store_true", help="초파리가 아닌 비교 기준(종족별 최빈값, 로지스틱 회귀)도 돌린다")
    ap.add_argument("--nature", choices=["21", "updown", "both"], default=NATURE_KIND,
                    help="성격 출력층: 성격마다 한 구획(21) / 올림·내림 12구획(updown, 기본) / 둘 다")
    ap.add_argument("--nature-only", action="store_true", help="성격 학습까지만 돌리고 성격 정확도를 비교한다")
    ap.add_argument("--ev-sigma", default="0", help="EV 이웃 값 학습 폭 σ, 쉼표로 여러 개 (0 = 정답 값만). 예: 0,1,2")
    ap.add_argument("--etas", default=",".join(f"{e:g}" for e in ETAS), help="학습률 후보, 큰 값부터 쉼표로. 예: 3e-5,1e-5,3e-6")
    ap.add_argument("--code", choices=["together", "perfeature", "sequence"], default="together",
                    help="Kenyon cell 코드: 특징을 한꺼번에 자극(기본) / 특징마다 따로 자극한 코드의 합 / 한 시뮬레이션 안에서 차례로 자극")
    ap.add_argument("--seq-ms", type=int, default=200, help="--code sequence의 특징당 자극 시간(ms)")
    ap.add_argument("--test-draws", type=int, choices=[1, 2], default=1,
                    help="2면 시험 개체를 새 반응 2번의 평균으로도 채점 (학습은 그대로, 한꺼번에 코드만)")
    ap.add_argument("--train-draws", type=int, choices=[1, 2], default=1,
                    help="2면 학습 때 개체를 배울 때마다 원래 코드 / 새 잡음 코드 중 하나를 무작위로 (검증·시험 코드는 그대로, 한꺼번에 코드만)")
    ap.add_argument("--ctx-temps", default="", help="EV 맥락용 성격 확률 온도, 쉼표로 (T<1이면 날카롭게, 학습은 그대로). 예: 2,0.7,0.5,0.3,0.1")
    ap.add_argument("--save-lp", action="store_true",
                    help="[새 반응] 시험 개체의 성격 log 확률과 EV 스탯별 log 확률(성격 확률만큼 맥락 ev_lp, 정답 성격 맥락 known_lp)을 out/mb_lp_*.npz에 저장 (학습은 그대로)")
    ap.add_argument("--team", action="store_true",
                    help=f"팀 자극 코드(그 마리 + 팀원 5마리 합집합, {TEAM_RATE_HZ}Hz로 발화 5%)를 그 마리 코드 옆에 붙인다. 출력 가중치는 따로, 성격 연결은 같은 Kenyon cell이라 그대로")
    ap.add_argument("--team-scale", type=float, default=1.0, help="--team 코드 발화율에 곱할 배율 (기본 1 = 시뮬레이션 발화율 그대로)")
    ap.add_argument("--select", choices=["error", "logloss"], default="error",
                    help="학습률·반복 수 고르는 검증 채점: 오답률·EV 오차(기본) / 로그 손실. logloss면 예측 파일 이름 끝 _sel-logloss")
    ap.add_argument("--nature-given", action="store_true",
                    help="대회용 과제: OTS에 성격이 공개라고 보고 성격을 학습하지 않는다. 정답 성격을 확정 맥락으로 주고 EV만 배운다. 파일 이름 끝 _natgiven")
    ap.add_argument("--select-ev", choices=["error", "logloss"], default=None,
                    help="EV만 따로 고르는 검증 채점 (기본 --select와 같음). 다르면 예측 파일 이름 끝 _sel-nat-{성격}-ev-{EV}")
    args = ap.parse_args()
    select_ev = args.select_ev or args.select
    select_tag = "" if (args.select, select_ev) == ("error", "error") else (
        f"_sel-{args.select}" if args.select == select_ev else f"_sel-nat-{args.select}-ev-{select_ev}")
    temps = tuple(float(x) for x in args.ctx_temps.split(",") if x)
    if args.nature_given and (args.nature == "both" or args.nature_only or args.shuffle):
        ap.error("--nature-given은 성격 출력층을 안 쓰므로 --nature both / --nature-only / --shuffle와 같이 쓸 수 없다")
    if args.team and (args.code != "together" or args.test_draws == 2 or args.train_draws == 2 or args.shuffle):
        ap.error("--team은 한꺼번에 코드에서 --test-draws 1, --train-draws 1, 섞은 배선 없이만")
    if args.train_draws == 2 and (args.test_draws == 2 or args.code != "together"):
        ap.error("--train-draws 2는 한꺼번에 코드에서 --test-draws 1과만 (2번 평균 시험 코드가 학습 반응을 포함하게 됨)")
    kinds = ["21", "updown"] if args.nature == "both" else [args.nature]
    kind_label = {"21": "", "updown": " (올림·내림 성격)"}
    sigmas = [float(x) for x in args.ev_sigma.split(",")]
    etas = sorted((float(x) for x in args.etas.split(",")), reverse=True)
    sigma_label = lambda sg: f" (이웃 σ={sg:g})" if sg else ""

    mo = load_mons()
    arch = mo.archetype_key.to_numpy()
    X, vocab = features(mo)
    Y = mo[EV].to_numpy()
    nat = mo[NATURE].to_numpy()
    natures, nat_id = np.unique(nat, return_inverse=True)
    bits = nature_bits(nat)
    F = folds(mo)

    W, groups = load_mb()
    nets = {"real": W, **({"shuffle_pnkc": shuffle_pn_kc(W, groups, 0)} if args.shuffle else {})}
    data = {}
    code_tag = f"{DURATION_MS}ms_pn{PN_MIN_KC_SYNAPSES}x{PNS_PER_FEATURE}"
    for name, Wn in nets.items():
        fresh = []  # 시험 채점용 추가 코드: 개체마다 새 반응, (--test-draws 2) 새 반응 2번 평균. 한꺼번에 코드만
        if args.code == "sequence":
            codes, inv = kc_sequence_codes(X, vocab, Wn, groups, f"out/kc_codes_{name}_{code_tag}_seq{args.seq_ms}ms_gap{SEQ_GAP_MS}.npz",
                                           args.seq_ms)
            rates = sp.csr_matrix(codes.astype(np.float64) / (args.seq_ms / 1000))[inv]
        elif args.code == "perfeature":
            per_f = kc_feature_codes(vocab, Wn, groups, f"out/kc_codes_{name}_{code_tag}_perfeature.npz")
            rates = sp.csr_matrix(X @ sp.csr_matrix(per_f.astype(np.float64) / (DURATION_MS / 1000)))
        else:
            codes, inv = kc_codes(X, Wn, groups, f"out/kc_codes_{name}_{code_tag}.npz")
            rates = sp.csr_matrix(codes.astype(np.float64) / (DURATION_MS / 1000))[inv]
            fcodes = np.zeros((len(mo), codes.shape[1]), np.uint16)
            for src in [NEW_SOURCE, OLD_SOURCE]:  # 새 환경 행은 mb_stream이 만든 캐시와 같은 파일
                rows = np.where(mo.source == src)[0]
                fcodes[rows] = kc_row_codes(X, rows, Wn, groups, f"out/kc_codes_{name}_{code_tag}_{src.lower()}_rows_seed{ENCOUNTER_SEED0}.npz")
            fresh = [sp.csr_matrix(fcodes.astype(np.float64) / (DURATION_MS / 1000))]
            if args.team:  # 팀 자극: 학습 코드는 고유 팀 입력마다 한 번, 시험 코드는 개체마다 새 반응. 같은 Kenyon cell의 두 번째 반응이라 열을 옆에 붙임
                Xt = team_union_features(mo, X, vocab)
                tcodes, tinv = kc_codes(Xt, Wn, groups, f"out/kc_codes_{name}_{code_tag}_team{TEAM_RATE_HZ}Hz.npz", seed0=TEAM_SEED0, rate=TEAM_RATE_HZ)
                tf = kc_row_codes(Xt, np.arange(len(mo)), Wn, groups, f"out/kc_codes_{name}_{code_tag}_team{TEAM_RATE_HZ}Hz_rows_seed{TEAM_ENCOUNTER_SEED0}.npz",
                                  seed0=TEAM_ENCOUNTER_SEED0, rate=TEAM_RATE_HZ)
                scale = args.team_scale / (DURATION_MS / 1000)
                trates = sp.csr_matrix(tcodes.astype(np.float64) * scale)[tinv]
                print(f"[{name}] 팀 코드: 고유 팀 입력 {len(tcodes):,}개, 발화한 Kenyon cell 비율 평균 {trates.getnnz(axis=1).mean() / trates.shape[1]:.1%}, "
                      f"발화율 평균 {trates.data.mean():.1f}Hz (배율 {args.team_scale:g})", flush=True)
                rates = sp.hstack([rates, trates]).tocsr()
                fresh = [sp.hstack([fresh[0], sp.csr_matrix(tf.astype(np.float64) * scale)]).tocsr()]
            if args.test_draws == 2:  # 두 번째 반응: 새 잡음 코드 (고유 입력마다 FRESH_SEED0 + 번호)
                f2, _ = kc_codes(X, Wn, groups, f"out/kc_codes_{name}_{code_tag}_seed{FRESH_SEED0}.npz", seed0=FRESH_SEED0)
                fresh.append(sp.csr_matrix((fcodes.astype(np.float64) + f2[inv]) / 2 / (DURATION_MS / 1000)))
        train_extra = []  # --train-draws 2: 학습 때 무작위로 섞어 쓸 두 번째 반응 (시험 [새 반응]과 독립)
        if args.train_draws == 2:
            f2, _ = kc_codes(X, Wn, groups, f"out/kc_codes_{name}_{code_tag}_seed{FRESH_SEED0}.npz", seed0=FRESH_SEED0)
            train_extra = [sp.csr_matrix(f2[inv].astype(np.float64) / (DURATION_MS / 1000))]
        S = kc_mbon_synapses(Wn, groups)
        print(f"[{name}] 발화한 Kenyon cell 비율 평균 {rates.getnnz(axis=1).mean() / rates.shape[1]:.1%}, "
              f"발화한 세포의 평균 발화율 {rates.data.mean():.1f}Hz", flush=True)
        # 추가 코드는 원래 개체 N행 뒤에 차례로 N행씩 붙인다: 시험 채점 코드들, 그 뒤에 학습용 두 번째 반응.
        # 검증은 앞쪽 원래 행만 쓰고, 학습은 draw_offsets가 없으면 원래 행만 쓰므로 학습 결과는 같다
        n_views = 1 + len(fresh)
        draw_offsets = [0] + [(n_views + k) * len(mo) for k in range(len(train_extra))] if train_extra else None
        conns = {k: nature_output_setup(S, natures, k)[0] for k in ["21", "updown"]}
        if args.team:  # 팀 코드 열도 같은 Kenyon cell -> 같은 MBON 연결
            conns = {k: np.vstack([c, c]) for k, c in conns.items()}
        data[name] = (sp.vstack([rates, *fresh, *train_extra]).tocsr(), conns, n_views, draw_offsets)

    n = len(mo)
    test_rows = lambda te, n_views: np.concatenate([te + k * n for k in range(n_views)])

    # 1단계: 성격. EV 검증에 초파리가 고른 성격이 필요해서 먼저 돌린다
    axes = nature_axes(natures)
    jobs = []
    if args.nature_given:  # 대회용(성격 공개): 성격 출력층을 학습하지 않고 정답 성격을 확정 맥락으로 준다. 검증·시험 모두 정답 성격
        got = {}
        for f, (tr, te) in enumerate(F):
            for name, (_, _, n_views, _) in data.items():
                for kind in kinds:
                    got[("nature", name, kind, f)] = (onehot_logp(nat_id[inner_split(arch, tr)[1]], len(natures)),
                                                      onehot_logp(nat_id[test_rows(te, n_views) % n], len(natures)),
                                                      {"kind": kind, "eta": 0.0, "epochs": 0, "capped": False})
        print("성격 공개 과제: 성격 학습 건너뜀 (정답 성격을 맥락으로)", flush=True)
    for f, (tr, te) in enumerate(F):
        if args.nature_given:
            break
        for name, (rates, conns, n_views, draw_offsets) in data.items():
            for kind in kinds:
                jobs.append(delayed(tagged)(("nature", name, kind, f), mb_nature_job, rates, nat_id, len(natures), conns[kind],
                                            arch, tr, test_rows(te, n_views), f"fold {f} {name} 성격{kind_label[kind]}",
                                            axes if kind == "updown" else None, etas, draw_offsets, args.select))
            if args.compare:
                jobs.append(delayed(tagged)(("lr_nature", name, f), logreg_classes, rates, nat, tr, te))
        if args.compare:
            jobs.append(delayed(tagged)(("brainless", None, f), brainless_job, X, Y, nat, tr, te))
    if not args.nature_given:
        got = run_jobs(jobs, "1단계: 성격 학습")
    if args.nature_only:
        report_nature(got, data, kinds, kind_label, F, X, mo, nat_id, natures)
        return

    # 2단계: EV
    jobs, ctx_rates = [], {}
    for f, (tr, te) in enumerate(F):
        for name, (rates, _, n_views, draw_offsets) in data.items():
            ctx_rates[(name, f)] = context_rate(rates, tr)
            rows_te = test_rows(te, n_views)
            for kind in kinds:
                val_logp, te_logp, _ = got[("nature", name, kind, f)]
                for sg in sigmas:
                    jobs.append(delayed(tagged)(("ev", name, kind, sg, f), mb_ev_job, rates, Y, bits, natures, val_logp, te_logp,
                                                arch, tr, rows_te, ctx_rates[(name, f)],
                                                f"fold {f} {name} EV{kind_label[kind]}{sigma_label(sg)}", sg, etas, nat_id[rows_te % n],
                                                draw_offsets, temps, args.save_lp, select_ev))
    if args.compare:
        jobs += [delayed(tagged)(("lr_ev", name, f), ev_given_nature, data[name][0], Y, nat, *F[f], [got[("lr_nature", name, f)]])
                 for f in range(len(F)) for name in data]
    got.update(run_jobs(jobs, "2단계: EV 학습"))

    def note(fit):
        return (f"eta={fit['eta']:g} {fit['epochs']}회" + ("  <- 학습률 후보 끝값" if fit["eta"] in (etas[0], etas[-1]) else "")
                + ("  <- 상한까지 계속 나아짐" if fit["capped"] else ""))

    rows, fine, saved, lps = defaultdict(list), defaultdict(lambda: np.zeros((2, 6))), [], defaultdict(list)
    for f, (tr, te) in enumerate(F):
        if args.compare:
            rows["종족별 최빈값"].append(paste_scores(species_mode_predict(mo, tr, te), species_mode_nature(mo, tr, te), Y[te], nat[te]))
            rows["OTS 특징 + 로지스틱 회귀"].append(paste_scores(*got[("brainless", None, f)], Y[te], nat[te]))
        for name, label in [("real", "진짜"), ("shuffle_pnkc", "섞은")]:
            if name not in data:
                continue
            if args.compare:
                rows[f"{label} 버섯체 코드 + 로지스틱 회귀"].append(
                    paste_scores(got[("lr_ev", name, f)][0], got[("lr_nature", name, f)], Y[te], nat[te]))

            views = [(["", " [새 반응]", " [새 반응 2번 평균]"][k], slice(k * len(te), (k + 1) * len(te))) for k in range(data[name][2])]
            for kind in kinds:
                _, nat_logp_all, fit_n = got[("nature", name, kind, f)]
                for sg in sigmas:
                    ev_all = got[("ev", name, kind, sg, f)]
                    fit_e = ev_all[4]
                    for noise, sl in views:
                        nat_logp = nat_logp_all[sl]
                        soft, soft_min_err, score, spreads, known_min_err = (x[sl] for x in (ev_all[0], ev_all[1], ev_all[2], ev_all[3], ev_all[5]))
                        sharp = {T: (ev_t[sl], ev_t_min[sl]) for T, (ev_t, ev_t_min) in ev_all[6].items()}
                        back = {k: (a[sl], b[sl], c[sl]) for k, (a, b, c) in ev_all[7].items()}
                        if args.save_lp and name == "real" and noise == " [새 반응]":
                            for key, val in [("row", te), ("fold", np.full(len(te), f)), ("nat_logp", nat_logp.astype(np.float32)), ("ev_lp", ev_all[8][sl]),
                                             ("known_lp", ev_all[9][sl]), ("fit_lp_mode", back["최빈"][2]),
                                             ("fit_lp_minerr", back["기대 오차 최소"][2]), ("ev_mode", soft), ("ev_minerr", soft_min_err)]:
                                lps[(kind, sg, key)].append(val)
                        first = nat_logp.argmax(1)
                        joint = (nat_logp + score).argmax(1)
                        rr = np.arange(len(te))
                        top = np.exp(nat_logp.max(1))
                        print(f"  fold {f} {name}{kind_label[kind]}{sigma_label(sg)}{noise} 성격 {note(fit_n)} | EV(맥락 최대 {ctx_rates[(name, f)]:.1f}Hz) "
                              f"{note(fit_e)} | 가장 높은 성격 확률 중앙값 {np.median(top):.2f} (맞힘 {np.median(top[first == nat_id[te]]):.2f}, "
                              f"틀림 {np.median(top[first != nat_id[te]]):.2f})")

                        decodings = [("성격 먼저 -> EV (성격 확률만큼 맥락)", (soft, natures[first])),
                                     ("성격 먼저 -> EV (성격 확률만큼 맥락, 기대 오차 최소)", (soft_min_err, natures[first])),
                                     ("성격 먼저 -> EV (확정 성격 맥락)", (spreads[rr, first], natures[first])),
                                     ("완전히 함께", (spreads[rr, joint], natures[joint])),
                                     ("성격 앎 -> EV (정답 성격 맥락)", (spreads[rr, nat_id[te]], nat[te])),
                                     ("성격 앎 -> EV (정답 성격 맥락, 기대 오차 최소)", (known_min_err, nat[te])),
                                     ("EV 먼저 (최빈) -> 성격 (배분만)", (soft, natures[back["최빈"][0]])),
                                     ("EV 먼저 (최빈) -> 성격 (성격 확률도)", (soft, natures[back["최빈"][1]])),
                                     ("EV 먼저 (기대 오차 최소) -> 성격 (배분만)", (soft_min_err, natures[back["기대 오차 최소"][0]])),
                                     ("EV 먼저 (기대 오차 최소) -> 성격 (성격 확률도)", (soft_min_err, natures[back["기대 오차 최소"][1]]))]
                        for T, (ev_t, ev_t_min) in sharp.items():
                            decodings += [(f"성격 먼저 -> EV (성격 확률 T={T:g} 맥락)", (ev_t, natures[first])),
                                          (f"성격 먼저 -> EV (성격 확률 T={T:g} 맥락, 기대 오차 최소)", (ev_t_min, natures[first]))]
                        for how, (ev, na) in decodings:
                            method = f"{label} 초파리{kind_label[kind]}{sigma_label(sg)}: {how}{noise}"
                            rows[method].append(paste_scores(ev, na, Y[te], nat[te]))
                            fine[method] += fine_value_hits(ev, Y[te])
                            saved.append(pd.DataFrame({"row": te, "fold": f, "method": method, "nature": na,
                                                       **{c: ev[:, s] for s, c in enumerate(EV)}}))

    code_suffix = {"together": "", "perfeature": "_perfeature", "sequence": f"_seq{args.seq_ms}ms"}[args.code]
    out = (f"out/mb_preds_{code_tag}{code_suffix}{'_draws2' if args.test_draws == 2 else ''}{'_train2' if args.train_draws == 2 else ''}{'_temps' if temps else ''}"
           f"{'' if etas == ETAS else '_eta' + '_'.join(f'{e:g}' for e in etas)}{select_tag}"
           f"{f'_team{TEAM_RATE_HZ}Hz' + ('' if args.team_scale == 1 else f'_x{args.team_scale:g}') if args.team else ''}"
           f"{'_natgiven' if args.nature_given else ''}.parquet")  # --save-lp는 예측이 같아 이름 그대로
    pd.concat(saved, ignore_index=True).to_parquet(out)
    for kind, sg in {(k, s) for k, s, _ in lps}:
        lp_out = out.replace("mb_preds_", "mb_lp_").replace(".parquet", f"_{kind}_sigma{sg:g}.npz")
        np.savez_compressed(lp_out, natures=natures, **{key: np.concatenate(lps[(kind, sg, key)])
                                                        for key in ["row", "fold", "nat_logp", "ev_lp", "known_lp", "fit_lp_mode", "fit_lp_minerr",
                                                                    "ev_mode", "ev_minerr"]})
        print(f"시험 log 확률 저장: {lp_out}")

    table = pd.DataFrame({k: {**pd.DataFrame(v).mean().to_dict(),
                              "EV 오차 ±": pd.DataFrame(v)["EV 오차"].std(ddof=0),
                              "성격 ±": pd.DataFrame(v)["성격 정확도"].std(ddof=0)} for k, v in rows.items()}).T
    cols = ["EV 오차", "EV 오차 ±", "EV 오차(정답이 맞춤 배분)", "EV 오차(성격 맞힘)", "EV 오차(성격 틀림)", "EV 완전일치",
            "세부 값 정확도", "성격 정확도", "성격 ±", "EV·성격 완전일치", "맞춤 배분 예측", "어색한 조합"]
    print(f"\narchetype GroupKFold 5, 정상 개체 {len(mo):,}, 그 마리만 입력. 세부 값 = 정답이 3~29인 스탯 칸\n")
    print(format_table(table[cols]))
    print("\n스탯별 세부 값 정확도 (정답이 3~29인 칸 중 값을 정확히 맞힌 비율)")
    print(pd.DataFrame({m: dict(zip(STATS, h / n)) for m, (n, h) in fine.items()}).T.to_string(float_format="{:.1%}".format))
    print(f"\n시험 예측 저장: {out}")


if __name__ == "__main__":
    main()
