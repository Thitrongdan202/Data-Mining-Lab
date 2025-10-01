"""Các hàm hỗ trợ Rough Set (thuộc tính rời rạc)."""
from __future__ import annotations
from typing import List, Set
import pandas as pd


def indiscernibility(df: pd.DataFrame, attrs: List[str]) -> List[Set[int]]:
    """Tạo các lớp tương đương theo danh sách thuộc tính."""
    groups = df.groupby(attrs, dropna=False).groups
    return [set(idx) for idx in groups.values()]


def discernibility_matrix(df: pd.DataFrame, cond_attrs: List[str], decision_attr: str) -> List[Set[str]]:
    """Ma trận phân biệt: tập thuộc tính phân biệt các cặp khác quyết định."""
    matrix = []
    n = len(df)
    for i in range(n):
        for j in range(i + 1, n):
            if df.loc[i, decision_attr] != df.loc[j, decision_attr]:
                diff = {
                    a for a in cond_attrs
                    if df.loc[i, a] != df.loc[j, a]
                }
                if diff:
                    matrix.append(diff)
    return matrix


def greedy_reduct(df: pd.DataFrame, cond_attrs: List[str], decision_attr: str) -> List[str]:
    """Thuật toán tham lam chọn thuộc tính phủ nhiều cặp nhất."""
    matrix = discernibility_matrix(df, cond_attrs, decision_attr)
    uncovered = list(matrix)
    reduct: List[str] = []
    while uncovered:
        counts = {a: 0 for a in cond_attrs if a not in reduct}
        for diff in uncovered:
            for a in diff:
                if a in counts:
                    counts[a] += 1
        if not counts:
            break
        best = max(counts, key=counts.get)
        reduct.append(best)
        uncovered = [d for d in uncovered if best not in d]
    return reduct


def positive_region(df: pd.DataFrame, cond_attrs: List[str], decision_attr: str) -> float:
    """Tính chỉ số Positive Region."""
    pos = set()
    classes = indiscernibility(df, cond_attrs)
    for cls in classes:
        decisions = df.loc[list(cls), decision_attr].unique()
        if len(decisions) == 1:
            pos.update(cls)
    return len(pos) / len(df)