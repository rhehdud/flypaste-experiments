import numpy as np, pandas as pd, duckdb
from mb_task import EV as C, NATURE, err, folds, load_mons, species_mode_nature, species_mode_predict


def main():
    mo=load_mons()   # 정상 개체만(테라·IVs 오염, EV 합계/cap 불일치, 성격 없음 등 6개 플래그 제외), (team_id, slot) 중복 행 제거
    m=mo[["species",*C]]

    con=duckdb.connect(); con.register("m",m)
    print("=== species freq ===")
    print(con.execute("select count(*) n, count(distinct species) sp from m").df().to_string(index=False))
    print(con.execute("""select species, count(*) n from m group by 1 order by 2 desc limit 8""").df().to_string(index=False))
    print("\n=== within-species spread diversity ===")
    q=con.execute(f"""
     select species, count(*) n, count(distinct concat_ws('/',{','.join(C)})) uniq
     from m group by 1 having n>=20 order by n desc limit 10""").df()
    q["uniq_ratio"]=(q.uniq/q.n).round(3); print(q.to_string(index=False))

    # archetype GroupKFold 5. EV 예측은 전부 실제로 존재하는 배분(합 66): 종족별/전체에서 가장 흔한 배분 통째로 (스탯마다 따로 최빈값을 뽑지 않음)
    Y=mo[C].to_numpy(); nat=mo[NATURE].to_numpy()
    rows=[]
    for tr,te in folds(mo):
        t=mo.iloc[tr]
        gm=np.array(t[C].value_counts().index[0]); gn=t[NATURE].mode().iloc[0]
        preds={"species_mode":(species_mode_predict(mo,tr,te),species_mode_nature(mo,tr,te)),
               "global_mode":(np.tile(gm,(len(te),1)),np.full(len(te),gn)),
               "uniform":(np.full((len(te),6),11),None)}
        for k,(ev,na) in preds.items():
            assert (ev.sum(1)==66).all() and (ev<=32).all(), k
            rows.append({"method":k,"ev_err":err(ev,Y[te]).mean(),"ev_exact":(ev==Y[te]).all(1).mean(),
                         "nature_acc":np.nan if na is None else (na==nat[te]).mean()})
    r=pd.DataFrame(rows).groupby("method",sort=False)
    print(f"\n=== baseline: {len(mo):,} mons, 5 folds, archetype-grouped ===")
    print(f"{'':14s} {'EV wrong/66':>14s} {'EV exact':>9s} {'nature acc':>11s}")
    for k,d in r:
        print(f"{k:14s} {d.ev_err.mean():6.2f} ± {d.ev_err.std(ddof=0):.2f} {d.ev_exact.mean():9.1%} "
              f"{'-' if d.nature_acc.isna().all() else format(d.nature_acc.mean(),'.1%'):>11s}")


if __name__ == "__main__":
    main()
