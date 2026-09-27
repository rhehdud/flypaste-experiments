#!/usr/bin/env python3
"""최종 초파리를 학습 데이터(옛 환경 + 새 환경) 전체로 학습해서 저장만 한다. **TEST는 열지 않는다** (평가는 mb_test.py / mb_test_stream.py).

기본 설정: 올림·내림 성격 출력층, 학습률·반복 수는 학습 데이터 안 검증용 아키타입(15%)에서 로그 손실로 고른다.
입력은 그 마리 코드 + 같은 팀 6마리 특징 합집합을 TEAM_RATE_HZ로 자극한 코드.
학습 Kenyon cell 코드는 고유 입력마다 한 번 (out/kc_codes_*.npz 캐시, mb_learn.py와 같은 파일을 쓴다).

저장물 하나에 예측에 필요한 것이 전부 들어간다: 성격·EV 가중치, 성격 목록, 특징 이름(투사뉴런 배정 순서),
맥락 세기, 팀 자극 여부·세기, 시뮬레이션 설정.

  python src/mb_final.py                                  # 기본 설정
  python src/mb_final.py --no-team --select error         # 다른 설정 (파일 이름이 달라진다)
"""
import argparse
import time

import numpy as np
import scipy.sparse as sp

from mb import (DURATION_MS, FINAL_ETAS, PN_MIN_KC_SYNAPSES, PNS_PER_FEATURE, RATE_HZ, TAG, TEAM_RATE_HZ, TEAM_SEED0,
                kc_codes, load_mb, model_path)
from mb_learn import N_JOBS, context_rate, fit_ev, fit_nature, kc_mbon_synapses, nature_output_setup
from mb_task import EV, NATURE, TEST_SOURCE, features, load_mons, team_union_features
from model_io import save_model
from spread import nature_bits



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-team", dest="team", action="store_false", help="팀 자극 코드를 쓰지 않는다 (그 마리 코드만)")
    ap.add_argument("--select", choices=["error", "logloss"], default="logloss", help="학습률·반복 수 고르는 검증 채점 (기본 로그 손실)")
    ap.add_argument("--select-ev", choices=["error", "logloss"], default=None, help="EV만 따로 (기본 --select와 같음)")
    ap.add_argument("--etas", default=",".join(f"{e:g}" for e in FINAL_ETAS), help="학습률 후보, 큰 값부터 쉼표로")
    ap.add_argument("--nature", choices=["21", "updown"], default="updown", help="성격 출력층 (기본 올림·내림 12구획)")
    args = ap.parse_args()
    select_ev = args.select_ev or args.select
    etas = sorted((float(x) for x in args.etas.split(",")), reverse=True)

    mo = load_mons()  # 옛 환경 + 새 환경. load_mons()는 TEST가 섞이면 멈춘다
    assert TEST_SOURCE not in set(mo.source), "TEST는 최종 학습에 쓰지 않는다"
    X, vocab = features(mo)
    Y, nat = mo[EV].to_numpy(), mo[NATURE].to_numpy()
    natures, nat_id = np.unique(nat, return_inverse=True)
    bits, arch, tr = nature_bits(nat), mo.archetype_key.to_numpy(), np.arange(len(mo))
    print(f"학습 {len(mo):,}마리 (출처 {mo.source.value_counts().to_dict()}), 팀 {mo.team_id.nunique():,}개, 특징 이름 {len(vocab):,}개, "
          f"성격 {len(natures)}가지 | 설정: 성격 {args.nature}, 검증 채점 성격 {args.select}·EV {select_ev}, 팀 자극 "
          f"{f'{TEAM_RATE_HZ}Hz' if args.team else '안 씀'}, 학습률 후보 {etas}", flush=True)

    W, groups = load_mb()
    codes, inv = kc_codes(X, W, groups, f"out/kc_codes_real_{TAG}.npz")
    rates = sp.csr_matrix(codes.astype(np.float64) / (DURATION_MS / 1000))[inv]
    if args.team:  # 팀 자극: 고유 팀 입력마다 한 번, 같은 Kenyon cell의 두 번째 반응이라 열을 옆에 붙인다
        Xt = team_union_features(mo, X, vocab)
        tcodes, tinv = kc_codes(Xt, W, groups, f"out/kc_codes_real_{TAG}_team{TEAM_RATE_HZ}Hz.npz", seed0=TEAM_SEED0, rate=TEAM_RATE_HZ)
        trates = sp.csr_matrix(tcodes.astype(np.float64) / (DURATION_MS / 1000))[tinv]
        rates = sp.hstack([rates, trates]).tocsr()
        print(f"팀 코드: 고유 팀 입력 {len(tcodes):,}개, 발화한 Kenyon cell 비율 평균 "
              f"{trates.getnnz(axis=1).mean() / trates.shape[1]:.1%}", flush=True)
    print(f"입력 열 {rates.shape[1]:,}개, 발화한 Kenyon cell 비율 평균 {rates.getnnz(axis=1).mean() / rates.shape[1]:.1%}", flush=True)

    S = kc_mbon_synapses(W, groups)
    conn, axes = nature_output_setup(S, natures, args.nature)
    if args.team:  # 팀 코드 열도 같은 Kenyon cell -> 같은 MBON 연결
        conn = np.vstack([conn, conn])
    ctx_rate = context_rate(rates, tr)

    t0 = time.time()
    nature_net, val_logp, fit_n = fit_nature(rates, nat_id, len(natures), conn, arch, tr, "최종 성격", n_jobs=N_JOBS, axes=axes,
                                             etas=etas, select=args.select)
    print(f"[최종 학습 1/2] 성격({fit_n['kind']}) 끝 (eta={fit_n['eta']:g}, {fit_n['epochs']}회, 경과 {(time.time() - t0) / 60:.1f}분)", flush=True)
    ev_net, fit_e = fit_ev(rates, Y, bits, natures, val_logp, arch, tr, ctx_rate, "최종 EV", n_jobs=N_JOBS, etas=etas, select=select_ev)
    print(f"[최종 학습 2/2] EV 끝 (eta={fit_e['eta']:g}, {fit_e['epochs']}회, 경과 {(time.time() - t0) / 60:.1f}분)", flush=True)

    out = model_path(fit_n["kind"], args.team, args.select, select_ev, etas)
    save_model(out, nature_kind=fit_n["kind"], nature_w=nature_net.w, nature_b=nature_net.b,
                        ev_w=ev_net.w, ev_u=ev_net.u, ev_b=ev_net.b, ctx_rate=ctx_rate, natures=natures,
                        vocab=vocab, team=args.team, team_rate=TEAM_RATE_HZ, rate=RATE_HZ,
                        duration_ms=DURATION_MS, pn_min_kc=PN_MIN_KC_SYNAPSES, pns_per_feature=PNS_PER_FEATURE,
                        select=args.select, select_ev=select_ev, etas=np.array(etas),
                        nature_fit=np.array([fit_n["eta"], fit_n["epochs"]]), ev_fit=np.array([fit_e["eta"], fit_e["epochs"]]),
                        train_sources=np.array(sorted(mo.source.unique())), n_train=len(mo), n_teams=mo.team_id.nunique())
    print(f"\n최종 모델 저장: {out} (학습 {', '.join(sorted(mo.source.unique()))} {len(mo):,}마리). 추론은 python src/mb_paste.py <OTS 파일>")


if __name__ == "__main__":
    main()
