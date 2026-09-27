#!/usr/bin/env python3
"""최종 초파리를 시험 데이터로 평가한다. TEST는 시험 전용이다: 학습·학습률 선택·설계 선택 어디에도 쓰지 않는다.

최종 모델: 학습 데이터(옛 환경 + 새 환경) 전체로 mb_learn과 같은 방식(성격 -> EV 도파민 학습, 학습률·반복 수는 학습 데이터 안
검증용 아키타입으로 선택). out/에 저장한다 (학습 출처를 함께 기록).
시험: 모델을 고정한 채 TEST 개체를 예측만 한다 (정답으로 가중치를 바꾸지 않음).
특징 이름은 학습 때 순서를 유지하고 TEST에만 있는 이름을 뒤에 붙인다 (학습 때의 투사뉴런 배정 그대로).
TEST 개체의 Kenyon cell 코드는 개체마다 따로 시뮬레이션한 새 반응이다. 학습에 같은 입력이 있어도 학습 코드를 다시 쓰지
않는다 (잡음까지 같으면 점수가 부풀려진다). out/test/에 캐시.
예측 규칙: 성격 확률만큼 맥락(최빈 / 기대 오차 최소), 확정 성격 맥락, 완전히 함께, 성격 앎(정답 성격 맥락, 최빈 / 기대 오차 최소).

  python src/pipeline.py <시험용 엑셀>:TEST --out-dir out/test   # 먼저 한 번 (엑셀은 저장소에 없다)
  python src/mb_test.py
"""
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import log_softmax

from mb import TAG, DURATION_MS, PN_MIN_KC_SYNAPSES, PNS_PER_FEATURE, TEST_SEED0, kc_codes, kc_row_codes, load_mb
from mb_learn import N_JOBS, NATURE_KIND, context_rate, fit_ev, fit_nature, kc_mbon_synapses, nature_output_setup
from mb_task import EV, NATURE, TEST_DIR, TEST_SOURCE, features, load_mons, load_test_mons
from model_io import EXT as MODEL_EXT, save_model
from spread import EV_CAP, STATS, best_spread, expected_error, fine_value_hits, nature_bits, paste_scores

def kind_suffix(kind=NATURE_KIND):
    """성격 21구획의 산출물은 이름 그대로, 올림·내림은 _updown을 붙인다."""
    return "" if kind == "21" else f"_{kind}"


def model_path(kind=NATURE_KIND):
    return f"out/fly_test{kind_suffix(kind)}{MODEL_EXT}"


def test_codes(X_test, W, groups):
    """(TEST 개체마다 따로 시뮬레이션한 새 반응 Kenyon cell 코드 (TEST 수, Kenyon cell 수), 행별 입력 = 활성 특징 번호 튜플).
    시드 = TEST_SEED0 + TEST 행 번호. 캐시는 입력이 같을 때만 쓴다 (mb.kc_row_codes)."""
    X_test = X_test.tocsr()
    rows = [tuple(np.sort(X_test[i].indices)) for i in range(X_test.shape[0])]
    codes = kc_row_codes(X_test, np.arange(X_test.shape[0]), W, groups, f"{TEST_DIR}/kc_codes_test_{TAG}_rows_seed{TEST_SEED0}.npz",
                         seed0=TEST_SEED0)
    return codes, rows


def main():
    train, test = load_mons(), load_test_mons()
    assert not set(test.team_id) & set(train.team_id)
    X_train, vocab = features(train)
    X_test, vocab_all = features(test, vocab=vocab)  # 학습 특징의 열 번호는 그대로, TEST에만 있는 이름은 뒤에
    Y, nat = train[EV].to_numpy(), train[NATURE].to_numpy()
    natures, nat_id = np.unique(nat, return_inverse=True)
    bits = nature_bits(nat)
    arch = train.archetype_key.to_numpy()
    tr = np.arange(len(train))
    print(f"학습 {len(train):,}마리 (출처 {train.source.value_counts().to_dict()}) | 시험 {len(test)}마리, {test.team_id.nunique()}팀 "
          f"| TEST에만 있는 특징 이름 {len(vocab_all) - len(vocab)}개", flush=True)

    W, groups = load_mb()
    codes, inv = kc_codes(X_train, W, groups, f"out/kc_codes_real_{TAG}.npz")
    train_uniq = sorted({tuple(np.sort(X_train[i].indices)) for i in range(X_train.shape[0])})
    tcodes, trows = test_codes(X_test, W, groups)
    rates = sp.csr_matrix(codes.astype(np.float64) / (DURATION_MS / 1000))[inv]
    rates_test = sp.csr_matrix(tcodes.astype(np.float64) / (DURATION_MS / 1000))
    S = kc_mbon_synapses(W, groups)
    conn, axes = nature_output_setup(S, natures)
    ctx_rate = context_rate(rates, tr)

    t0 = time.time()
    nature_net, val_logp, fit_n = fit_nature(rates, nat_id, len(natures), conn, arch, tr, "최종 성격", n_jobs=N_JOBS, axes=axes)
    print(f"[최종 학습 1/2] 성격({fit_n['kind']}) 끝 (eta={fit_n['eta']:g}, {fit_n['epochs']}회, 경과 {(time.time() - t0) / 60:.1f}분)", flush=True)
    ev_net, fit_e = fit_ev(rates, Y, bits, natures, val_logp, arch, tr, ctx_rate, "최종 EV", n_jobs=N_JOBS)
    print(f"[최종 학습 2/2] EV 끝 (eta={fit_e['eta']:g}, {fit_e['epochs']}회, 경과 {(time.time() - t0) / 60:.1f}분)", flush=True)
    model = model_path(fit_n["kind"])
    save_model(model, nature_kind=fit_n["kind"], nature_w=nature_net.w, nature_b=nature_net.b, ev_w=ev_net.w, ev_u=ev_net.u, ev_b=ev_net.b,
                        ctx_rate=ctx_rate, natures=natures, vocab=vocab,
                        train_sources=np.array(sorted(train.source.unique())), n_train=len(train),
                        nature_fit=np.array([fit_n["eta"], fit_n["epochs"]]), ev_fit=np.array([fit_e["eta"], fit_e["epochs"]]))
    assert TEST_SOURCE not in set(train.source)

    # 시험: 고정된 모델로 예측만
    Yt, natt = test[EV].to_numpy(), test[NATURE].to_numpy()
    logp = log_softmax(nature_net.scores(rates_test)[:, 0], axis=1)
    first = logp.argmax(1)
    cand = nature_bits(natures) * ctx_rate
    lp_soft = log_softmax(ev_net.scores(rates_test, np.exp(logp) @ cand), axis=2)
    s = (ev_net.b - np.asarray(rates_test @ ev_net.w))[:, None, :] - (cand @ ev_net.u)[None]
    score, spreads = best_spread(log_softmax(s.reshape(len(test), len(natures), 6, EV_CAP + 1), axis=3))
    joint = (logp + score).argmax(1)
    rr = np.arange(len(test))
    lp_known = log_softmax(ev_net.scores(rates_test, nature_bits(natt) * ctx_rate), axis=2)  # 정답 성격 맥락 (학습에 없는 성격이어도 됨)
    methods = {"성격 확률만큼 맥락": (best_spread(lp_soft)[1], natures[first]),
               "성격 확률만큼 맥락, 기대 오차 최소": (best_spread(-expected_error(np.exp(lp_soft)))[1], natures[first]),
               "확정 성격 맥락": (spreads[rr, first], natures[first]),
               "완전히 함께": (spreads[rr, joint], natures[joint]),
               "성격 앎": (best_spread(lp_known)[1], natt),
               "성격 앎, 기대 오차 최소": (best_spread(-expected_error(np.exp(lp_known)))[1], natt)}

    train_species, train_inputs = set(train.species), set(train_uniq)
    groups_ = {"전체": np.ones(len(test), bool),
               "새 종족": ~test.species.isin(train_species).to_numpy(),
               "아는 종족": test.species.isin(train_species).to_numpy(),
               "  └ 학습에 같은 입력 있음": np.array([r in train_inputs for r in trows]),
               "  └ 입력이 처음 (아는 종족)": test.species.isin(train_species).to_numpy() & np.array([r not in train_inputs for r in trows])}
    cols = ["EV 오차", "EV 오차(성격 맞힘)", "EV 오차(성격 틀림)", "EV 완전일치", "세부 값 정확도", "성격 정확도", "EV·성격 완전일치",
            "맞춤 배분 예측", "어색한 조합"]
    fmt = {c: ("{:.2f}".format if c.startswith("EV 오차") else "{:.1%}".format) for c in cols}
    saved = []
    print(f"\n최종 초파리(학습 {', '.join(sorted(train.source.unique()))}, 고정) -> TEST {len(test)}마리")
    for how, (ev, na) in methods.items():
        rows = {g: {"개체": int(m.sum()), **paste_scores(ev[m], na[m], Yt[m], natt[m])} for g, m in groups_.items() if m.any()}
        t = pd.DataFrame(rows).T
        print(f"\n=== {how} ===")
        print(pd.concat([t["개체"].astype(int), t[cols].astype(float)], axis=1).to_string(formatters=fmt))
        n_fine, n_hit = fine_value_hits(ev, Yt)
        print("스탯별 세부 값 정확도: " + ", ".join(f"{s} {h / n:.1%}" if n else f"{s} -" for s, h, n in zip(STATS, n_hit, n_fine)))
        saved.append(pd.DataFrame({"team_id": test.team_id, "slot": test.slot, "species": test.species, "method": how,
                                   "nature_pred": na, "nature_true": natt, **{f"{c}_pred": ev[:, i] for i, c in enumerate(EV)},
                                   **{f"{c}_true": Yt[:, i] for i, c in enumerate(EV)}}))
    out = f"{TEST_DIR}/mb_test_preds_{TAG}{kind_suffix(fit_n['kind'])}_encounter.parquet"
    pd.concat(saved, ignore_index=True).to_parquet(out)
    print(f"\n최종 모델 저장: {model} | 시험 예측 저장: {out}")


if __name__ == "__main__":
    main()
