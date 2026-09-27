#!/usr/bin/env python3
"""MaleCNS v1.0 커넥톰 로더.

connectome_data/malecns_v1/*.feather (원본, 필터 전)에서
시뮬레이션에 쓸 노드·엣지만 골라낸다.

데이터셋: Janelia FlyEM MaleCNS v1.0
  소개·인용   https://www.janelia.org/project-team/flyem/male-cns-connectome
  파일 목록   https://male-cns.janelia.org/download/   (= janelia-flyem.github.io/male-cns/download/)
그 페이지의 feather 7개 중 아래 3개만 쓴다. `minconf-0.5`는 시냅스 신뢰도 0.5 이상만 남긴
판이라, 이름을 안 맞추면 다른 판을 받아 노드·엣지 수부터 달라진다.

  mkdir -p connectome_data/malecns_v1 && cd connectome_data/malecns_v1
  B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
  curl -L -o annotations.feather       "$B/body-annotations-male-cns-v1.0-minconf-0.5.feather"
  curl -L -o neurotransmitters.feather "$B/body-neurotransmitters-male-cns-v1.0.feather"
  curl -L -o edges.feather             "$B/connectome-weights-male-cns-v1.0-minconf-0.5.feather"

  로컬 이름                 원본 이름                                                    바이트  sha256
  annotations.feather       body-annotations-male-cns-v1.0-minconf-0.5.feather        14,483,314  2177e246...
  neurotransmitters.feather body-neurotransmitters-male-cns-v1.0.feather              43,282,834  95c92892...
  edges.feather             connectome-weights-male-cns-v1.0-minconf-0.5.feather   1,051,241,946  e35da783...

브라우저로 같은 데이터를 보려면 neuPrint (https://neuprint.janelia.org/?dataset=male-cns:v1.0).

노드 정책: superclass가 있는 엔트리만 (없으면 glia/비신경 개체) -> 166,700개
엣지 정책: 양쪽 다 retained 노드인 엣지만 -> 25,582,938개

  python src/connectome.py
"""
import pathlib
import pandas as pd

DATA_DIR = pathlib.Path("connectome_data/malecns_v1")

N_NEURONS_EXPECTED = 166_700
N_EDGES_EXPECTED = 25_582_938


def load():
    """(nodes, edges) 반환. nodes: bodyId 인덱스의 annotations. edges: body_pre/body_post/weight."""
    ann = pd.read_feather(DATA_DIR / "annotations.feather")
    edges = pd.read_feather(DATA_DIR / "edges.feather")
    nt = pd.read_feather(DATA_DIR / "neurotransmitters.feather")

    nodes = ann[ann.superclass.notna()].set_index("bodyId")
    retained = nodes.index
    edges = edges[edges.body_pre.isin(retained) & edges.body_post.isin(retained)].reset_index(drop=True)

    return nodes, edges, nt


def main():
    nodes, edges, _ = load()
    print(f"nodes: {len(nodes):,} (기대값 {N_NEURONS_EXPECTED:,}) {'OK' if len(nodes) == N_NEURONS_EXPECTED else 'MISMATCH'}")
    print(f"edges: {len(edges):,} (기대값 {N_EDGES_EXPECTED:,}) {'OK' if len(edges) == N_EDGES_EXPECTED else 'MISMATCH'}")
    print()
    print("superclass 분포:")
    print(nodes.superclass.value_counts())


if __name__ == "__main__":
    main()
