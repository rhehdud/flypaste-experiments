#!/usr/bin/env python3
"""VGC Champions paste pipeline: xlsx -> normalized parquet + xlsx.

  python pipeline.py old.xlsx:OLD new.xlsx:NEW --out-dir out

Input : xlsx with an OTS and/or `paste` column, one team per row.
Output: teams_clean.parquet / mons_clean.parquet  (all rows, validity flagged)
        normalized.xlsx                           (OTS + paste, valid teams only)
"""
import argparse, hashlib, pathlib, re, sys, unicodedata
import numpy as np, pandas as pd

STATS = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]
KEY   = {"HP":"hp", "Atk":"atk", "Def":"def", "SpA":"spa", "SpD":"spd", "Spe":"spe"}
S     = list(KEY.values())
LBL   = {v: k for k, v in KEY.items()}
C     = [f"ev_{x}" for x in S]
EV_TOTAL, EV_CAP = 66, 32          # Champions 룰: EV 합계 66, 스탯당 32 상한


# ---------------------------------------------------------------- parsing
def norm(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = " ".join(unicodedata.normalize("NFKC", str(s)).split())
    return s or None


MEGA_BASE = {"Floette-Mega": "Floette-Eternal"}  # 기본 이름이 폼 이름인 경우


def base_species(s):
    """메가 폼 이름 -> 기본 이름 (Pyroar-Mega -> Pyroar, Charizard-Mega-Y -> Charizard). OTS에는 메가 진화 전 이름이 보이고,
    메가 정보는 아이템(메가스톤)에 남는다. 같은 포켓몬의 종족 특징이 이름 표기로 갈라지지 않게 한다."""
    if s is None:
        return None
    return MEGA_BASE.get(s) or re.sub(r"-Mega(-[XYZ])?$", "", s)


def parse_mon(block):
    """One Showdown-format block -> dict. Handles nickname, gender, no-item, no-level."""
    lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
    if not lines:
        return None
    m = {"item": None, "ability": None, "nature": None, "level": None,
         "tera": None, "nickname": None, "moves": [], "has_ivs": False}
    for k in S:
        m["ev_" + k] = 0

    left, _, item = lines[0].partition(" @ ")
    m["item"] = norm(item)
    left = left.strip()

    g = re.search(r"\((M|F)\)\s*$", left)
    if g:
        left = left[:g.start()].strip()
    p = re.search(r"\(([^()]+)\)\s*$", left)
    if p:
        m["nickname"] = norm(left[:p.start()])
        m["species"]  = base_species(norm(p.group(1)))
    else:
        m["species"] = base_species(norm(left))

    for l in lines[1:]:
        if l.startswith("- "):
            m["moves"].append(norm(l[2:]))
        elif l.startswith("Ability:"):
            m["ability"] = norm(l.split(":", 1)[1])
        elif l.startswith("Level:"):
            m["level"] = int(l.split(":", 1)[1].strip())
        elif l.startswith("Tera Type:"):
            m["tera"] = norm(l.split(":", 1)[1])
        elif l.startswith("EVs:"):
            for part in l.split(":", 1)[1].split("/"):
                tok = part.strip().split()
                if len(tok) == 2 and tok[1] in KEY:
                    m["ev_" + KEY[tok[1]]] = int(tok[0])
        elif l.startswith("IVs:"):
            m["has_ivs"] = True          # Champions엔 IV 없음 -> 타 포맷 오염
        elif l.endswith("Nature"):
            m["nature"] = norm(l.rsplit(" ", 1)[0])
    m["ev_total"] = sum(m["ev_" + k] for k in S)
    return m


CONT = ("Ability:", "Level:", "EVs:", "Tera Type:", "IVs:",
        "Shiny:", "Happiness:", "Dynamax Level:", "Gigantamax:", "- ")


def split_mons(text):
    """빈 줄 구분자가 없는 페이스트도 처리. 헤더 줄(= 속성 줄이 아닌 줄)에서 끊는다."""
    blocks, cur = [], []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if not (s.startswith(CONT) or s.endswith("Nature")) and cur:
            blocks.append("\n".join(cur)); cur = []
        cur.append(s)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def parse_team(text):
    ms = (parse_mon(b) for b in split_mons(text))
    # 속성 줄이 하나도 없는 블록(페이스트 끝의 잔여 기호 등)은 개체가 아니다
    return [x for x in ms if x and (x["ability"] or x["moves"])]


def sha(x, n=12):
    return hashlib.sha1(x.encode()).hexdigest()[:n]


def load(specs):
    """specs: list of 'path:SOURCE'. Returns (teams, mons) raw frames."""
    teams, mons, fails = [], [], 0
    for spec in specs:
        path, _, source = spec.rpartition(":")
        if not path:
            path, source = spec, pathlib.Path(spec).stem
        df = pd.read_excel(path)
        if "paste" not in df.columns:
            sys.exit(f"{path}: no 'paste' column (found {list(df.columns)})")
        for raw in df["paste"].dropna().astype(str):
            ms = parse_team(raw)
            if not ms:
                fails += 1
                continue
            tid  = sha(re.sub(r"\s+", " ", raw).strip())
            arch = sha("|".join(sorted(x["species"] for x in ms)))
            teams.append({"team_id": tid, "archetype_key": arch,
                          "source": source, "paste_raw": raw})
            for i, x in enumerate(ms):
                mons.append({**x, "team_id": tid, "slot": i, "source": source})
        print(f"  {path} -> {source}: {len(df)} rows", file=sys.stderr)
    if fails:
        print(f"  parse failures: {fails}", file=sys.stderr)
    t  = pd.DataFrame(teams).drop_duplicates("team_id")
    mo = pd.DataFrame(mons)
    return t, mo[mo.team_id.isin(t.team_id)].copy()


# ------------------------------------------------------------ normalizing
def normalize(t, mo):
    for c in ["species", "item", "ability", "nature", "tera"]:
        mo[c] = mo[c].map(norm)
    mo["level"]  = mo["level"].fillna(50).astype(int)
    mo["item"]   = mo["item"].fillna("None")
    mo["tera"]   = mo["tera"].fillna("None")
    mo["moves"]        = mo["moves"].apply(list)
    mo["moves_sorted"] = mo["moves"].apply(sorted)
    mo = mo.drop(columns=["nickname"])

    mo["n_moves"]       = mo["moves"].apply(len)
    mo["flag_ev_total"] = mo.ev_total != EV_TOTAL
    mo["flag_ev_cap"]   = (mo[C] > EV_CAP).any(axis=1)
    mo["flag_moves"]    = ~mo.n_moves.between(1, 4)
    mo["flag_ivs"]      = mo.has_ivs
    mo["flag_tera"]     = mo.tera != "None"
    mo["flag_nature"]   = mo.nature.isna()
    mo["valid"] = ~(mo.flag_ev_total | mo.flag_ev_cap | mo.flag_moves | mo.flag_nature | mo.flag_ivs | mo.flag_tera)

    mo["mon_key"] = mo.apply(lambda r: sha("|".join(
        [r["species"], r["item"], r["ability"], r["tera"], *r["moves_sorted"]])), axis=1)

    mo = mo.sort_values(["team_id", "slot"])
    mo["_paste"] = mo.apply(paste_mon_text, axis=1)
    mo["_ots"]   = mo.apply(lambda r: paste_mon_text(r, with_spread=False), axis=1)
    g = mo.groupby("team_id")
    t = t.merge(g._ots.apply("\n\n".join).rename("ots_text"), on="team_id")
    t = t.merge(g._paste.apply("\n\n".join).rename("paste_norm"), on="team_id")
    t["ots_key"]   = t.ots_text.map(sha)
    t["all_valid"] = t.team_id.map(g.valid.all())
    t["n_valid"]   = t.team_id.map(g.valid.sum())
    return t, mo.drop(columns=["_paste", "_ots"])


def paste_mon_text(r, with_spread=True):
    """Rebuild one Showdown block. with_spread=False -> OTS (no EVs, no nature)."""
    head = r["species"]
    if r["item"] != "None":
        head += f" @ {r['item']}"
    L = [head, f"Ability: {r['ability']}", f"Level: {r['level']}"]
    if r["tera"] != "None":
        L.append(f"Tera Type: {r['tera']}")
    if with_spread:
        ev = " / ".join(f"{r['ev_' + k]} {LBL[k]}" for k in S if r["ev_" + k] > 0)
        L += [f"EVs: {ev}", f"{r['nature']} Nature"]
    return "\n".join(L + [f"- {m}" for m in r["moves"]])


# --------------------------------------------------------------- exporting
def to_xlsx(t, path):
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    df = t[t.all_valid][["OTS", "paste"]] if "OTS" in t else \
         t[t.all_valid].rename(columns={"ots_text": "OTS", "paste_norm": "paste"})[["OTS", "paste"]]
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="시트1", index=False)
    wb = load_workbook(path); ws = wb.active
    for c in ws[1]:
        c.font = Font(name="Arial", bold=True, size=10)
        c.fill = PatternFill("solid", fgColor="DDE5F0")
    ws.freeze_panes = "A2"
    for col in ("A", "B"):
        ws.column_dimensions[col].width = 64
        for c in ws[col][1:]:
            c.font = Font(name="Arial", size=10)
            c.alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(path)
    return len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="path.xlsx:SOURCE")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    t, mo = load(a.inputs)
    t, mo = normalize(t, mo)
    t.to_parquet(out / "teams_clean.parquet", index=False)
    mo.to_parquet(out / "mons_clean.parquet", index=False)
    n = to_xlsx(t, out / "normalized.xlsx")

    print(f"\nteams {len(t)} | mons {len(mo)} | species {mo.species.nunique()} "
          f"| archetypes {t.archetype_key.nunique()}")
    print(f"valid teams {int(t.all_valid.sum())} ({t.all_valid.mean():.1%}) | "
          f"valid mons {int(mo.valid.sum())} ({mo.valid.mean():.1%})")
    print("flags  ev_total:%d  ev_cap:%d  moves:%d  nature:%d  ivs:%d  tera:%d" % (
        mo.flag_ev_total.sum(), mo.flag_ev_cap.sum(),
        mo.flag_moves.sum(), mo.flag_nature.sum(), mo.flag_ivs.sum(), mo.flag_tera.sum()))
    print(t.groupby("source").agg(teams=("team_id", "size"),
                                  valid=("all_valid", "sum")).to_string())
    print(f"\nwrote {out}/normalized.xlsx ({n} teams), teams_clean.parquet, mons_clean.parquet")


if __name__ == "__main__":
    main()
