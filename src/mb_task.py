#!/usr/bin/env python3
"""과제 준비: 입력 특징, 교차검증 분할, 오차, 초파리를 쓰지 않는 기준선 도구.

예측은 한 마리씩, 입력은 OTS 전체(그 마리 + 같은 팀 나머지 5마리)를 쓸 수 있다.
출력은 EV 배분(합 66, 스탯당 32 이하)과 성격. 고르는 방법은 spread.py.
평가: archetype_key GroupKFold 5, 정상 개체만. EV 오차 = |예측-정답| 합 / 2 (66점 만점), 성격은 정확도.

  python src/mb_task.py    # 데이터 요약, 종족별 최빈값 기준선, 입력이 같은 개체끼리 정답 일치율
"""
from collections import defaultdict

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

EV = [f"ev_{s}" for s in ["hp", "atk", "def", "spa", "spd", "spe"]]
NATURE = "nature"
TEAM_INPUTS = {None: [], "species": ["sp"], "all": ["sp", "it", "ab", "mv"]}
TEAM_PREFIX = "tm_"
INPUTS = [("그 마리만", None), ("+ 팀 동료 종족", "species"), ("+ 팀 동료 전체", "all")]
MLP_HIDDEN = 512
MLP_ALPHAS = [1e-3, 1e-1, 1.0, 10.0]
MLP_MAX_EPOCHS = 40


# 시험 전용 데이터. 어떤 학습·선택에도 쓰지 않는다. 원본 엑셀은 저장소에 없다(공개하지 않음):
# 각자 가진 파일을 pipeline.py로 가공해 TEST_DIR에만 둔다 -- `python src/pipeline.py <파일>:TEST --out-dir out/test`
# 경로는 환경변수 FLY_TEST_DIR로 바꿀 수 있다
TEST_SOURCE = os.environ.get("FLY_TEST_SOURCE", "TEST")
TEST_DIR = os.environ.get("FLY_TEST_DIR", "out/test")
# 환경 변화 실험(mb_shift·mb_stream·mb_stream_cv·mb_update)은 학습 데이터가 두 시기로 나뉘어 있어야 한다.
# 먼저 모인 쪽을 OLD_SOURCE, 나중에 모인 쪽을 NEW_SOURCE 이름으로 가공해 둔다.
OLD_SOURCE = os.environ.get("FLY_OLD_SOURCE", "OLD")
NEW_SOURCE = os.environ.get("FLY_NEW_SOURCE", "NEW")


def load_mons(root="out"):
    """학습용 정상 개체 + archetype_key + 같은 팀 나머지 마리(mates: 슬롯별 종족·아이템·특성·기술).

    mons_clean에 같은 팀의 같은 슬롯이 두 번 들어간 행(15팀)이 있어 (team_id, slot)당 한 행만 남긴다.
    팀 동료는 EV가 비정상인 마리도 포함한다 (OTS에는 보이므로).
    TEST 데이터가 섞여 있으면 멈춘다. 시험 데이터는 load_test_mons()로만 읽는다.
    """
    mo = _load(root)
    if (mo.source == TEST_SOURCE).any():
        raise RuntimeError(f"{root}/mons_clean.parquet에 {TEST_SOURCE} 데이터가 섞여 있다. TEST는 학습에 쓰지 않는다")
    return mo


def load_test_mons():
    """시험 전용 개체 (TEST_DIR). 평가에만 쓴다."""
    mo = _load(TEST_DIR)
    if not (mo.source == TEST_SOURCE).all():
        raise RuntimeError(f"{TEST_DIR}에 {TEST_SOURCE} 아닌 데이터가 있다")
    return mo


def _load(root):
    mo = pd.read_parquet(f"{root}/mons_clean.parquet").drop_duplicates(["team_id", "slot"])
    t = pd.read_parquet(f"{root}/teams_clean.parquet")
    team = {tid: list(zip(d.slot, d.species, d.item, d.ability, d.moves)) for tid, d in mo.groupby("team_id")}
    mo = mo.assign(mates=[[m[1:] for m in team[tid] if m[0] != s] for tid, s in zip(mo.team_id, mo.slot)])
    return mo[mo.valid].merge(t[["team_id", "archetype_key"]], on="team_id").reset_index(drop=True)


def mon_tokens(species, item, ability, moves, prefix="", kinds=("sp", "it", "ab", "mv")):
    toks = {"sp": [f"sp:{species}"], "it": [f"it:{item}"], "ab": [f"ab:{ability}"], "mv": [f"mv:{m}" for m in moves]}
    return [prefix + tok for k in kinds for tok in toks[k]]


def features(mo, team=None, vocab=None):
    """이진 특징과 특징 이름. 그 마리의 종족·아이템·특성·기술 + (team) 팀 동료 5마리 것을 tm_ 접두어로 합친 multi-hot.
    vocab(특징 이름 목록)을 주면 그 순서를 지키고, 거기 없는 이름은 뒤에 붙인다 (학습 때의 투사뉴런 배정을 유지)."""
    tokens = [sorted({*mon_tokens(r.species, r.item, r.ability, r.moves),
                      *[tok for m in r.mates for tok in mon_tokens(*m, TEAM_PREFIX, TEAM_INPUTS[team])]})
              for r in mo.itertuples()]
    names = list(vocab or []) + sorted({tok for row in tokens for tok in row} - set(vocab or []))
    vocab = {tok: i for i, tok in enumerate(names)}
    rows = np.repeat(np.arange(len(tokens)), [len(r) for r in tokens])
    cols = np.array([vocab[tok] for row in tokens for tok in row])
    X = sp.csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(tokens), len(vocab)))
    return X, list(vocab)


def team_union_features(mo, X, vocab):
    """팀 자극 입력: 그 마리 특징 + 같은 팀 나머지 5마리의 종족·아이템·특성·기술을 같은 특징 번호(같은 투사뉴런)로 합친 이진 행렬.
    vocab에 없는 이름(학습 개체에 한 번도 안 나온 것)은 뺀다. 같은 팀 6마리는 같은 행이 된다."""
    col = {tok: i for i, tok in enumerate(vocab)}
    rows, cols = [], []
    for i, ms in enumerate(mo.mates):
        f = set(X[i].indices) | {col[t] for m in ms for t in mon_tokens(*m) if t in col}
        rows += [i] * len(f); cols += sorted(f)
    return sp.csr_matrix((np.ones(len(cols)), (rows, cols)), shape=X.shape)


def folds(mo, n=5):
    return list(GroupKFold(n_splits=n).split(mo, groups=mo.archetype_key))


def err(pred, true):
    return np.abs(pred - true).sum(1) / 2


def species_mode_predict(mo, tr, te):
    """종족별로 가장 흔한 실제 EV 배분 (없는 종족은 전체 최빈 배분)."""
    tr_df = mo.iloc[tr]
    gm = tr_df[EV].value_counts().index[0]
    mode = tr_df.groupby("species")[EV].apply(lambda d: d.value_counts().index[0])
    return np.array([mode.get(s, gm) for s in mo.species.iloc[te]])


def species_mode_nature(mo, tr, te):
    tr_df = mo.iloc[tr]
    mode = tr_df.groupby("species")[NATURE].agg(lambda s: s.mode().iloc[0])
    gm = tr_df[NATURE].mode().iloc[0]
    return np.array([mode.get(s, gm) for s in mo.species.iloc[te]])


def logreg(max_iter=2000):
    """특징마다 학습 데이터 최댓값으로 나눈 뒤 로지스틱 회귀 (0/1 특징은 그대로, 발화율은 수렴이 빨라짐)."""
    return make_pipeline(MaxAbsScaler(), LogisticRegression(max_iter=max_iter))


def logreg_classes(X, y, tr, te):
    clf = logreg().fit(X[tr], y[tr])
    return clf.classes_[clf.predict_proba(X[te]).argmax(1)]


def mlp_classes(X, y, n_classes, arch, tr, te, loss):
    """은닉층 1개 MLP. alpha·반복 수는 학습 fold 안 검증용 아키타입(15%)에서 loss(예측, 인덱스)가 가장 낮은 값."""
    itr, iva = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0).split(tr, groups=arch[tr]))
    itr, iva = tr[itr], tr[iva]

    def train(fit_idx, alpha, epochs, eval_idx=None):
        clf = MLPClassifier(hidden_layer_sizes=(MLP_HIDDEN,), alpha=alpha, random_state=0)
        Xf, yf = X[fit_idx], y[fit_idx]
        scores = []
        for _ in range(epochs):
            clf.partial_fit(Xf, yf, classes=np.arange(n_classes))
            if eval_idx is not None:
                scores.append(loss(clf.predict_proba(X[eval_idx]).argmax(1), eval_idx))
        return clf, scores

    _, alpha, epochs = min((s, a, e + 1) for a in MLP_ALPHAS
                           for e, s in enumerate(train(itr, a, MLP_MAX_EPOCHS, iva)[1]))
    return train(tr, alpha, epochs)[0].predict_proba(X[te]).argmax(1)


def ambiguity(X, y):
    """입력이 완전히 같은 개체가 2마리 이상인 묶음의 (개체 수, 묶음 안 최빈 정답 비율). 어떤 모델도 이 묶음에서 이 비율을 못 넘는다."""
    groups = defaultdict(list)
    for i in range(X.shape[0]):
        groups[tuple(np.sort(X[i].indices))].append(i)
    multi = [g for g in groups.values() if len(g) > 1]
    n = sum(map(len, multi))
    return n, sum(np.bincount(y[g]).max() for g in multi) / max(n, 1)


def main():
    mo = load_mons()
    Y = mo[EV].to_numpy()
    F = folds(mo)
    _, vocab = features(mo)
    print(f"정상 개체 {len(mo):,} | 특징 {len(vocab):,} | 아키타입 {mo.archetype_key.nunique():,} | "
          f"서로 다른 EV 배분 {len(np.unique(Y, axis=0)):,}가지\n")

    e, exact, nat = [], [], []
    for tr, te in F:
        pred = species_mode_predict(mo, tr, te)
        e.append(err(pred, Y[te]).mean())
        exact.append((pred == Y[te]).all(1).mean())
        nat.append((species_mode_nature(mo, tr, te) == mo[NATURE].iloc[te].to_numpy()).mean())
    print(f"종족별 최빈값: EV 오차 {np.mean(e):.2f} ± {np.std(e):.2f} | EV 완전일치 {np.mean(exact):.1%} | 성격 정확도 {np.mean(nat):.1%}\n")

    spread_id = np.unique(Y, axis=0, return_inverse=True)[1].ravel()
    nature_id = np.unique(mo[NATURE], return_inverse=True)[1]
    rows = []
    for label, team in INPUTS:
        Xt, _ = features(mo, team)
        n_same, cap_ev = ambiguity(Xt, spread_id)
        rows.append({"입력": label, "입력 같은 개체": n_same, "그 안 최빈 EV 배분": cap_ev, "그 안 최빈 성격": ambiguity(Xt, nature_id)[1]})
    print("=== 입력이 완전히 같은 개체끼리 정답 일치 (모델이 넘을 수 없는 선) ===")
    print(pd.DataFrame(rows).to_string(index=False, formatters={"그 안 최빈 EV 배분": "{:.1%}".format, "그 안 최빈 성격": "{:.1%}".format}))


if __name__ == "__main__":
    main()
