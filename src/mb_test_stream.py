#!/usr/bin/env python3
"""최종 초파리(mb_test.py가 저장한 out/fly_final_*.npz)의 버리는 사본으로 TEST에서 새 페이스트 배우기 방법을 평가한다.

TEST 팀을 무작위로 두 무리(A, B)로 나눠 A팀들 페이스트를 한 마리씩 "예측 -> 정답 보고 배우기"로
흘린 뒤, 배우지 않고 B팀들을 채점한다. 반대(B로 배우고 A 채점)도 해서 모든 개체가 한 번씩 채점된다. 나누기를 N_SPLITS번 반복.
- TEST 정답으로 배우는 것은 나누기·방향·변형마다 새로 만든 **버리는 사본**뿐이다. 최종 모델 파일은 읽기만 하고(실행 전후 해시 비교),
  배운 가중치는 저장하지 않는다. 채점하는 팀의 정답은 배우지 않는다
- 변형과 설정은 새 환경(`mb_stream.py`)에서 정한 그대로다. **이 결과로 설정을 고르지 않는다** (틀리는 유형 찾기에만 쓴다)
- 반복 수 R은 mb_stream과 같은 규칙으로 최종 학습에서 뽑힌 반복 수
- TEST 개체의 Kenyon cell 코드는 개체마다 새 반응 (mb_test.test_codes). 채점 파일 이름에 _encounter
- 예측 규칙: 성격 확률만큼 맥락의 최빈(기본) / 기대 오차 최소, 성격 앎의 최빈 / 기대 오차 최소 (배우는 것은 같음)

  python src/mb_test.py          # 먼저 한 번 (최종 모델 저장)
  python src/mb_test_stream.py
"""
import hashlib
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from mb import DURATION_MS, kc_codes, load_mb
from mb_learn import N_JOBS, DopamineOutput, kc_mbon_synapses, nature_output, nature_output_setup
from mb_stream import NOVEL_THETA, PRED_KINDS, VARIANTS as STREAM_VARIANTS, similar_counts, stream_job
from mb_task import EV, NATURE, TEST_DIR, TEST_SOURCE, err, features, load_mons, load_test_mons
from model_io import load_model
from mb_test import TAG, kind_suffix, model_path, test_codes
from spread import EV_CAP, nature_bits, odd_combo

N_SPLITS = 10  # 팀 2겹 나누기 반복 수
# mb_stream.VARIANTS 이름. 새 환경에서 평가를 끝낸 방법만 넣는다
METHODS = ["한 번씩", "새로움(패턴 0.5)", "두 구획(낯섦 τ=0.5)"]
VARIANTS = {"고정 (배우지 않음)": {"learn": "none"}, **{m: STREAM_VARIANTS[m] for m in METHODS}}
assert all(cfg["learn"] != "fast" for cfg in VARIANTS.values()), "보조층 빠른 구획(기억 유사도 문)은 스트림 개체끼리 코사인이 필요해 지원 안 함"
N_BOOT = 2000


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def team_boot(d, team, mask, rng):
    """행별 값 d의 mask 평균과 팀 단위 부트스트랩 95% 구간 (팀을 뽑으면 그 팀의 모든 행이 따라온다)."""
    t = np.unique(team, return_inverse=True)[1]
    n_t = t.max() + 1
    w = rng.multinomial(n_t, np.full(n_t, 1 / n_t), size=N_BOOT)[:, t] * mask
    b = (w * d).sum(1) / np.maximum(w.sum(1), 1)
    return d[mask].mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    model = model_path()
    digest = sha256(model)
    z = load_model(model)
    nature_kind = str(z.get("nature_kind", "21"))
    train, test = load_mons(), load_test_mons()
    assert TEST_SOURCE not in set(z["train_sources"]) and int(z["n_train"]) == len(train)
    assert not set(test.team_id) & set(train.team_id)
    X_train, vocab = features(train)
    assert vocab == list(z["vocab"])
    X_test, vocab_all = features(test, vocab=vocab)
    natures = z["natures"]
    assert list(natures) == list(np.unique(train[NATURE]))

    W, groups = load_mb()
    codes, inv = kc_codes(X_train, W, groups, f"out/kc_codes_real_{TAG}.npz")
    train_uniq = sorted({tuple(np.sort(X_train[i].indices)) for i in range(X_train.shape[0])})
    tcodes, trows = test_codes(X_test, W, groups)

    # 학습 개체 뒤에 TEST 개체를 붙인 한 묶음 (스트림은 TEST 행만 흐르고, 학습 행은 익숙함 계산에만)
    n_tr, n_te = len(train), len(test)
    X_train = X_train.tocsr()
    X_train.resize((n_tr, len(vocab_all)))
    X = sp.vstack([X_train, X_test]).tocsr()
    all_codes = np.vstack([codes[inv], tcodes])
    rates = sp.csr_matrix(all_codes.astype(np.float64) / (DURATION_MS / 1000))
    C = (all_codes / np.maximum(np.linalg.norm(all_codes, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    Y = np.vstack([train[EV].to_numpy(), test[EV].to_numpy()])
    nat = np.concatenate([train[NATURE].to_numpy(), test[NATURE].to_numpy()])
    nat_id = np.searchsorted(natures, nat)
    assert (natures[nat_id] == nat).all()
    bits = nature_bits(nat)
    kind = np.array([v[:2] for v in vocab_all])
    sp_feat = np.array([[f for f in X.indices[X.indptr[i]:X.indptr[i + 1]] if kind[f] == "sp"][0] for i in range(len(nat))])
    tr, te = np.arange(n_tr), np.arange(n_tr, n_tr + n_te)
    counts = np.asarray(X[tr].sum(0)).ravel()
    train_sim = np.zeros(len(nat))
    train_sim[te] = (C[te] @ C[tr].T).max(1)

    # 저장된 최종 모델을 그대로 복원
    S = kc_mbon_synapses(W, groups)
    conn, axes = nature_output_setup(S, natures, nature_kind)
    fit_n = {"eta": float(z["nature_fit"][0]), "epochs": int(z["nature_fit"][1])}
    fit_e = {"eta": float(z["ev_fit"][0]), "epochs": int(z["ev_fit"][1])}
    nature_net = nature_output(rates.shape[1], natures, fit_n["eta"], conn, axes)
    nature_net.w, nature_net.b = z["nature_w"], z["nature_b"]
    assert not (nature_net.w * (1 - conn)).any()
    ev_net = DopamineOutput(rates.shape[1], 6, EV_CAP + 1, fit_e["eta"], n_ctx=bits.shape[1])
    ev_net.w, ev_net.u, ev_net.b = z["ev_w"], z["ev_u"], z["ev_b"]
    ctx_rate = float(z["ctx_rate"])
    print(f"최종 모델 {model} (성격 {nature_kind}, 학습 {', '.join(z['train_sources'])} {int(z['n_train']):,}마리, 성격 eta={fit_n['eta']:g} "
          f"{fit_n['epochs']}회, EV eta={fit_e['eta']:g} {fit_e['epochs']}회) -> TEST {n_te}마리, 팀 2겹 x {N_SPLITS}번", flush=True)

    # 나누기: 팀을 무작위로 반씩. 방향마다 배우는 팀 개체(무작위 순서) 뒤에 채점 팀 개체
    team = test.team_id.to_numpy()
    teams = np.unique(team)
    streams = []
    for r in range(N_SPLITS):
        rng = np.random.default_rng(r)
        half = set(teams[rng.permutation(len(teams))[:len(teams) // 2]])
        in_a = np.array([t in half for t in team])
        for f, learn_part in enumerate([in_a, ~in_a]):
            order = np.concatenate([rng.permutation(te[learn_part]), rng.permutation(te[~learn_part])])
            streams.append((r, f, order, np.arange(len(order)) < learn_part.sum()))
    t0 = time.time()
    fams = [similar_counts(C, tr, o, [NOVEL_THETA])[0][NOVEL_THETA] for _, _, o, _ in streams]
    keys = [(v, i) for v in VARIANTS for i in range(len(streams))]
    jobs = [delayed(stream_job)(nature_net, ev_net, fit_n, fit_e, rates, Y, bits, nat_id, natures, X, sp_feat, counts,
                                streams[i][2], VARIANTS[v], fams[i] if "theta" in VARIANTS[v] else None, None, ctx_rate, "",
                                quiet=True, train_sim=train_sim[streams[i][2]], learn_mask=streams[i][3])
            for v, i in keys]
    got = []
    for k, df in enumerate(Parallel(n_jobs=N_JOBS, return_as="generator")(jobs), 1):
        v, i = keys[k - 1]
        got.append(df.assign(variant=v, split=streams[i][0], direction=streams[i][1]))
        if k % len(streams) == 0:
            print(f"  [{k}/{len(jobs)}] {v} 끝 (경과 {(time.time() - t0) / 60:.1f}분)", flush=True)
    res = pd.concat(got, ignore_index=True)
    res = res[~res.learned].reset_index(drop=True)  # 채점은 배우지 않은 팀만

    idx = res.row.to_numpy() - n_tr
    same_input = np.array([r in set(train_uniq) for r in trows])
    pred, true = res[EV].to_numpy(), Y[res.row]
    res["ev_err"] = err(pred, true)
    extra = list(PRED_KINDS)[1:]
    for pre in extra:
        res[f"ev_err_{pre}"] = err(res[[pre + c for c in EV]].to_numpy(), true)
    res["nature_hit"] = res.nature.to_numpy() == nat[res.row]
    res["odd"] = odd_combo(pred, res.nature.to_numpy())
    res["team_id"], res["species"] = team[idx], test.species.to_numpy()[idx]
    is_new = counts[sp_feat[res.row]] == 0
    res["group"] = np.where(is_new, np.where(res.species_seen > 0, "새 종족·배운 팀에 있음", "새 종족·배운 팀에 없음"),
                            np.where(same_input[idx], "아는 종족·같은 입력", "아는 종족·처음 입력"))
    out = f"{TEST_DIR}/mb_test_split_preds_{TAG}{kind_suffix(nature_kind)}_encounter.parquet"
    res.assign(test_row=idx).drop(columns=["row"]).to_parquet(out)

    names = list(VARIANTS)
    base = res[res.variant == names[0]]
    same = all((base.groupby("row")[c].nunique() == 1).all() for c in [pre + e for pre in PRED_KINDS for e in EV] + ["nature"])
    print(f"\n고정은 나누기와 무관하게 같은 예측: {same}")
    group_masks = {"전체": lambda d: np.ones(len(d), bool), "새 종족": lambda d: d.group.str.startswith("새 종족").to_numpy(),
                   "  └ 배운 팀에 있음": lambda d: (d.group == "새 종족·배운 팀에 있음").to_numpy(),
                   "  └ 배운 팀에 없음": lambda d: (d.group == "새 종족·배운 팀에 없음").to_numpy(),
                   "아는 종족": lambda d: d.group.str.startswith("아는 종족").to_numpy(),
                   "  └ 같은 입력": lambda d: (d.group == "아는 종족·같은 입력").to_numpy(),
                   "  └ 처음 입력": lambda d: (d.group == "아는 종족·처음 입력").to_numpy()}
    n_rows = len(base) / N_SPLITS
    print(f"\nTEST 팀 2겹 평가 (배운 팀 절반 -> 나머지 팀 채점, {N_SPLITS}번 평균). 나누기당 평균 개체: "
          + ", ".join(f"{g.strip(' └')} {fn(base).sum() / N_SPLITS:.1f}" for g, fn in group_masks.items()) + f" (합 {n_rows:.0f})")
    for metric, name, f in ([("ev_err", "EV 오차 (최빈)", "{:.2f}")] + [(f"ev_err_{pre}", f"EV 오차 ({PRED_KINDS[pre]})", "{:.2f}") for pre in extra]
                            + [("nature_hit", "성격 정확도", "{:.1%}"), ("odd", "어색한 조합", "{:.1%}")]):
        t = pd.DataFrame({g: {v: res[res.variant == v][metric].to_numpy()[fn(res[res.variant == v])].mean() for v in names}
                          for g, fn in group_masks.items()})
        print(f"\n[{name}]")
        print(t.to_string(float_format=f.format))

    # 고정 대비 차이 (같은 나누기·같은 개체끼리), 팀 단위 부트스트랩 95% 구간
    key = ["split", "row"]
    b = base.set_index(key)
    rng = np.random.default_rng(0)
    print("\n[고정 대비 차이, 팀 단위 부트스트랩 95% 구간] EV 오차는 낮을수록, 성격은 높을수록 좋음")
    for v in names[1:]:
        s = res[res.variant == v].set_index(key).reindex(b.index)
        cells = []
        for g in ["전체", "새 종족", "  └ 배운 팀에 있음", "아는 종족"]:
            m = group_masks[g](s.reset_index())
            e, lo, hi = team_boot((s.ev_err - b.ev_err).to_numpy(), s.team_id.to_numpy(), m, rng)
            h, hlo, hhi = team_boot((s.nature_hit.astype(float) - b.nature_hit.astype(float)).to_numpy(), s.team_id.to_numpy(), m, rng)
            cell = f"{g.strip(' └')} EV {e:+.2f} [{lo:+.2f}, {hi:+.2f}] 성격 {h:+.1%} [{hlo:+.1%}, {hhi:+.1%}]"
            for pre in extra:
                x, xlo, xhi = team_boot((s[f"ev_err_{pre}"] - b[f"ev_err_{pre}"]).to_numpy(), s.team_id.to_numpy(), m, rng)
                cell += f" | {PRED_KINDS[pre]} {x:+.2f} [{xlo:+.2f}, {xhi:+.2f}]"
            cells.append(cell)
        print(f"  {v}:\n    " + "\n    ".join(cells))

    assert sha256(model) == digest, "최종 모델 파일이 바뀌었다"
    print(f"\n최종 모델 파일 변경 없음 (sha256 {digest[:12]}). 배운 사본은 저장하지 않음 | 채점 예측 저장: {out}")


if __name__ == "__main__":
    main()
