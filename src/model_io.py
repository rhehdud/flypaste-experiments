#!/usr/bin/env python3
"""학습된 출력층을 파일로 저장·읽기. 숫자 배열은 safetensors 텐서, 나머지(문자열·스칼라·설정)는 JSON 메타데이터.

`.npz` 모델도 읽는다 (allow_pickle=False).
"""
import json
import pathlib

import numpy as np

EXT = ".safetensors"


def save_model(path, **fields):
    """숫자 배열(1차원 이상)은 텐서로, 그 밖의 값은 JSON 메타데이터로 나눠 저장한다."""
    from safetensors.numpy import save_file

    tensors, meta = {}, {}
    for k, v in fields.items():
        a = np.asarray(v)
        if a.ndim >= 1 and a.dtype.kind in "fiub":      # 숫자·불리언 배열만 텐서
            tensors[k] = np.ascontiguousarray(a)
        else:                                            # 문자열 배열, 스칼라, 설정값
            meta[k] = json.dumps(a.tolist())
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=meta)
    return path


def load_model(path):
    """{이름: 값}. 텐서는 ndarray, 메타데이터는 파이썬 값(str/int/float/bool/list)으로 돌려준다."""
    path = str(path)
    if path.endswith(".npz"):                            # 예전 파일 호환. pickle은 쓰지 않는다
        with np.load(path, allow_pickle=False) as z:
            m = {k: z[k] for k in z.files}
        for k, v in list(m.items()):
            if isinstance(v, np.ndarray) and (v.dtype.kind in "US" or v.ndim == 0):
                m[k] = v.tolist()
        return m

    from safetensors import safe_open
    from safetensors.numpy import load_file

    m = dict(load_file(path))
    with safe_open(path, framework="numpy") as f:
        for k, v in (f.metadata() or {}).items():
            m[k] = json.loads(v)
    return m
