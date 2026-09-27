#!/usr/bin/env python3
"""EV 배분과 성격 고르기. 정해진 배분 목록 없이 규칙(합 66, 스탯당 32 이하)에 맞는 모든 배분을 낼 수 있다.

EV_s 모델: 입력에 "성격이 Atk~Spe를 올리나/내리나" 표시 10개를 붙인 분류기 (0~32 중 하나). 학습 때는 실제 성격.
EV 배분: 스탯별 log 확률의 합이 가장 큰 합법 조합을 동적계획법으로 찾는다.
  스탯끼리는 입력과 성격이 주어지면 독립이라고 가정한다 (합 66 제약만 스탯을 묶는다).
기본 순서는 "성격 먼저 -> 그 성격을 조건으로 EV".
비교: 스탯별 EV + 성격 따로, 성격 먼저 -> EV, EV 먼저 -> 성격, 완전히 함께 (P(성격) x P(EV|성격) 최대).

  python src/spread.py
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.special import log_softmax

from mb_task import EV, NATURE, err, features, folds, load_mons, logreg

STATS = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]
EV_TOTAL, EV_CAP = 66, 32
NATURE_EFFECT = {  # 성격 -> (올리는 스탯, 내리는 스탯)
    "Hardy": (None, None), "Lonely": ("Atk", "Def"), "Brave": ("Atk", "Spe"), "Adamant": ("Atk", "SpA"), "Naughty": ("Atk", "SpD"),
    "Bold": ("Def", "Atk"), "Docile": (None, None), "Relaxed": ("Def", "Spe"), "Impish": ("Def", "SpA"), "Lax": ("Def", "SpD"),
    "Timid": ("Spe", "Atk"), "Hasty": ("Spe", "Def"), "Serious": (None, None), "Jolly": ("Spe", "SpA"), "Naive": ("Spe", "SpD"),
    "Modest": ("SpA", "Atk"), "Mild": ("SpA", "Def"), "Quiet": ("SpA", "Spe"), "Bashful": (None, None), "Rash": ("SpA", "SpD"),
    "Calm": ("SpD", "Atk"), "Gentle": ("SpD", "Def"), "Sassy": ("SpD", "Spe"), "Careful": ("SpD", "SpA"), "Quirky": (None, None),
}


def nature_axes(natures):
    """성격마다 (올림 구획, 내림 구획) 번호. 0 = 없음(무보정), 1~5 = Atk~Spe."""
    idx = lambda s: 0 if s is None else STATS.index(s)
    return np.array([idx(NATURE_EFFECT[n][0]) for n in natures]), np.array([idx(NATURE_EFFECT[n][1]) for n in natures])
NEG = -1e9


def nature_bits(names):
    """(n, 10): Atk~Spe 각각 성격이 올리나(앞 5개), 내리나(뒤 5개)."""
    out = np.zeros((len(names), 10))
    for i, name in enumerate(names):
        up, down = NATURE_EFFECT[name]
        if up:
            out[i, STATS.index(up) - 1] = 1
        if down:
            out[i, 4 + STATS.index(down)] = 1
    return out


def with_bits(A, names):
    return sp.hstack([sp.csr_matrix(A), sp.csr_matrix(nature_bits(names))]).tocsr()


def odd_combo(spreads, names):
    """성격이 올리는 스탯에 EV 2 이하, 또는 내리는 스탯에 EV 30 이상인 조합."""
    b = nature_bits(names).astype(bool)
    return (b[:, :5] & (spreads[:, 1:] <= 2)).any(1) | (b[:, 5:] & (spreads[:, 1:] >= 30)).any(1)


def is_custom(spreads):
    """3~29 사이 값이 하나라도 있는 배분 (몰빵·자투리 1~2만으로 된 배분이 아님)."""
    return ((spreads >= 3) & (spreads <= 29)).any(1)


def logreg_proba(Xtr, ytr, Xte):
    clf = logreg().fit(Xtr, ytr)
    return clf.classes_, clf.predict_proba(Xte)


def expected_error(p):
    """p (..., 6, 33) 스탯별 값 확률 -> 값마다 기대 오차 sum_u p(u)|v - u| (..., 6, 33).
    best_spread(-expected_error(p))는 규칙을 지키는 배분 중 기대 EV 오차(L1)가 가장 작은 것."""
    v = np.arange(EV_CAP + 1)
    return p @ np.abs(v[:, None] - v[None, :]).astype(float)


def best_spread(logp):
    """logp (..., 6, 33) -> (합 66·스탯당 32 이하 조합 중 최대 log 확률 (...), 그 조합 (..., 6))."""
    shape = logp.shape[:-2]
    dp = np.full(shape + (EV_TOTAL + 1,), NEG)
    dp[..., 0] = 0.0
    choice = np.zeros((6,) + shape + (EV_TOTAL + 1,), dtype=np.int8)
    for s in range(6):
        new = np.full_like(dp, NEG)
        for v in range(EV_CAP + 1):
            cand = np.full_like(dp, NEG)
            cand[..., v:] = dp[..., :EV_TOTAL + 1 - v] + logp[..., s, v, None]
            better = cand > new
            new = np.where(better, cand, new)
            choice[s] = np.where(better, v, choice[s])
        dp = new
    spread = np.zeros(shape + (6,), dtype=int)
    t = np.full(shape, EV_TOTAL)
    for s in reversed(range(6)):
        v = np.take_along_axis(choice[s], t[..., None], -1)[..., 0]
        spread[..., s] = v
        t = t - v
    return dp[..., EV_TOTAL], spread


def predict(rates, model, given=None):
    """(성격 확률 (n, 성격 수), 스탯별 EV 값 log 확률 (n, 6, 33)). given(성격 이름 목록)을 주면 그 성격을 확정 맥락으로 쓴다."""
    natures = model["natures"]
    if str(model["nature_kind"]) == "updown":
        up, down = nature_axes(natures)
        M = np.zeros((len(natures), 12))
        M[np.arange(len(natures)), up] = 1
        M[np.arange(len(natures)), 6 + down] = 1
        s_nat = model["nature_b"] - (rates @ model["nature_w"]) @ M.T
    else:
        s_nat = model["nature_b"] - rates @ model["nature_w"]
    logp = log_softmax(s_nat, axis=1)
    ctx = (nature_bits(given) if given is not None else np.exp(logp) @ nature_bits(natures)) * float(model["ctx_rate"])  # 성격 확률만큼 맥락
    s_ev = model["ev_b"] - rates @ model["ev_w"] - ctx @ model["ev_u"]
    return np.exp(logp), log_softmax(s_ev.reshape(len(rates), 6, EV_CAP + 1), axis=2)


def decode_ev(lp, how="eta", eta=3.0):
    """스탯별 log 확률 (n, 6, 33) -> 규칙(합 66·스탯당 32 이하)에 맞는 배분 (n, 6).
    eta: -기대오차 + eta x log p. 확률이 두 봉우리로 갈리면 그 사이 기대 오차가 평평해져
    기대 오차 최소가 아무도 안 쓰는 중간값을 내는데, log p를 조금 섞으면 그 구간에서만 봉우리 쪽으로 붙는다."""
    lp = np.asarray(lp, np.float64)
    if how == "mode":
        return best_spread(lp)[1]
    score = -expected_error(np.exp(lp))
    return best_spread(score if how == "minerr" else score + eta * lp)[1]


def ev_log_probs(Xtr, Ytr, Xtes, fit_proba=logreg_proba):
    """스탯마다 분류기를 한 번 학습해 여러 test 행렬에 적용 -> [log P(EV_s = v) (n, 6, 33)]."""
    out = [np.full((X.shape[0], 6, EV_CAP + 1), NEG) for X in Xtes]
    stacked = sp.vstack([sp.csr_matrix(X) for X in Xtes]).tocsr()
    bounds = np.cumsum([0] + [X.shape[0] for X in Xtes])
    for s in range(6):
        classes, p = fit_proba(Xtr, Ytr[:, s], stacked)
        lp = np.log(np.clip(p, 1e-12, None))
        for k, o in enumerate(out):
            o[:, s, classes] = lp[bounds[k]:bounds[k + 1]]
    return out


def ev_given_nature(A, Y, nat, tr, te, nat_preds, fit_proba=logreg_proba):
    """성격 예측(여러 벌 가능)을 조건으로 고른 EV 배분 목록. EV 모델은 실제 성격으로 한 번만 학습한다."""
    logps = ev_log_probs(with_bits(A[tr], nat[tr]), Y[tr], [with_bits(A[te], n) for n in nat_preds], fit_proba)
    return [best_spread(lp)[1] for lp in logps]


def fine_value_hits(ev, Y):
    """(스탯별 정답이 3~29인 칸 수 (6,), 그중 값을 정확히 맞힌 수 (6,))."""
    fine = (Y >= 3) & (Y <= 29)
    return fine.sum(0), (fine & (ev == Y)).sum(0)


def paste_scores(ev, na, Y, nat):
    e = err(ev, Y)
    exact = (ev == Y).all(1)
    custom = is_custom(Y)
    hit = na == nat
    n_fine, n_hit = fine_value_hits(ev, Y)
    return {"EV 오차": e.mean(), "EV 오차(정답이 맞춤 배분)": e[custom].mean(), "EV 오차(정답이 몰빵 배분)": e[~custom].mean(),
            "EV 오차(성격 맞힘)": e[hit].mean(), "EV 오차(성격 틀림)": e[~hit].mean(),
            "EV 완전일치": exact.mean(), "세부 값 정확도": n_hit.sum() / n_fine.sum(), "성격 정확도": hit.mean(),
            "EV·성격 완전일치": (exact & hit).mean(), "맞춤 배분 예측": is_custom(ev).mean(), "어색한 조합": odd_combo(ev, na).mean()}


def format_table(table):
    return table.to_string(na_rep="-", formatters={c: ("{:.2f}".format if c.startswith("EV 오차") else "{:.1%}".format)
                                                   for c in table.columns})


def eval_fold(X, Y, nat, tr, te):
    natures, p = logreg_proba(X[tr], nat[tr], X[te])
    logp_nat = np.log(np.clip(p, 1e-12, None))
    na_first = natures[p.argmax(1)]

    # 모든 성격 후보를 조건으로 한 EV log 확률 (n, m, 6, 33): "완전히 함께"에 필요
    lp_all = ev_log_probs(with_bits(X[tr], nat[tr]), Y[tr], [with_bits(X[te], [n] * len(te)) for n in natures])
    lp_all = np.stack(lp_all, axis=1)
    score, spreads = best_spread(lp_all)
    rows = np.arange(len(te))
    ev_first_nat = spreads[rows, p.argmax(1)]
    joint = (logp_nat + score).argmax(1)

    # 성격 조건 없는 EV, 그 EV를 붙여 고른 성격
    lp_free = ev_log_probs(X[tr], Y[tr], [X[te]])[0]
    ev_free = best_spread(lp_free)[1]
    ev_feat = lambda S: np.hstack([S / EV_CAP, S <= 2, S >= 30]).astype(float)
    nat2, p2 = logreg_proba(sp.hstack([X[tr], sp.csr_matrix(ev_feat(Y[tr]))]).tocsr(), nat[tr],
                            sp.hstack([X[te], sp.csr_matrix(ev_feat(ev_free))]).tocsr())

    preds = {"스탯별 EV + 성격 따로": (ev_free, na_first),
             "성격 먼저 -> EV": (ev_first_nat, na_first),
             "EV 먼저 -> 성격": (ev_free, nat2[p2.argmax(1)]),
             "완전히 함께": (spreads[rows, joint], natures[joint])}
    return {label: paste_scores(ev, na, Y[te], nat[te]) for label, (ev, na) in preds.items()}, preds


def main():
    mo = load_mons()
    X, _ = features(mo)
    Y = mo[EV].to_numpy()
    nat = mo[NATURE].to_numpy()
    F = folds(mo)
    res = Parallel(n_jobs=5)(delayed(eval_fold)(X, Y, nat, tr, te) for tr, te in F)

    table = pd.DataFrame({label: pd.DataFrame([r[0][label] for r in res]).mean() for label in res[0][0]}).T
    print(f"archetype GroupKFold 5, 정상 개체 {len(mo):,}, 그 마리만 입력, 로지스틱 회귀 (초파리 없음)")
    print(f"실제 데이터: 맞춤 배분(3~29 값 포함) {is_custom(Y).mean():.1%} | 어색한 조합 {odd_combo(Y, nat).mean():.1%} "
          f"(성격이 올리는 스탯에 EV 2 이하 또는 내리는 스탯에 EV 30 이상)\n")
    print(format_table(table))

    te = F[0][1]
    ev, na = res[0][1]["성격 먼저 -> EV"]
    line = lambda v: " / ".join(f"{int(x)} {s}" for x, s in zip(v, STATS) if x)
    print("\n=== 예시 (fold 0, 성격 먼저 -> EV) ===")
    for i in np.random.default_rng(3).choice(len(te), 6, replace=False):
        r = mo.iloc[te[i]]
        print(f"{r.species} @ {r['item']} / {r.ability} / {', '.join(r.moves)}")
        print(f"  예측: {na[i]} | {line(ev[i])}")
        print(f"  정답: {r.nature} | {line(Y[te[i]])}   (EV 오차 {err(ev[i][None], Y[te[i]][None])[0]:.0f}점)")


if __name__ == "__main__":
    main()
