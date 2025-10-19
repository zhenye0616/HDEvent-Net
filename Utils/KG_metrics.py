#!/usr/bin/env python3
"""Compute structural diagnostics for a Knowledge Graph represented as triple files."""

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

Triple = Tuple[str, str, str]


def load_triples(path: Path) -> List[Triple]:
    triples: List[Triple] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Malformed triple line in {path}: {line}")
            triples.append(tuple(parts))
    return triples


def compute_degrees(triples: Iterable[Triple]) -> Dict[str, int]:
    degree: Dict[str, int] = defaultdict(int)
    for h, _, t in triples:
        degree[h] += 1
        degree[t] += 1
    return degree


def describe_degree(degree: Dict[str, int]) -> Dict[str, float]:
    if not degree:
        return {
            "num_entities": 0,
            "avg_degree": 0.0,
            "max_degree": 0,
            "median_degree": 0.0,
            "p90_degree": 0.0,
            "pct_leq_2": 0.0,
        }
    values = sorted(degree.values())
    n = len(values)
    avg = sum(values) / n
    max_deg = values[-1]
    median = values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])
    p90 = values[int(math.floor(0.9 * (n - 1)))]
    pct_leq_2 = sum(1 for v in values if v <= 2) / n * 100
    return {
        "num_entities": n,
        "avg_degree": avg,
        "max_degree": max_deg,
        "median_degree": median,
        "p90_degree": p90,
        "pct_leq_2": pct_leq_2,
    }


def relation_statistics(triples: Iterable[Triple]) -> Tuple[Counter, float]:
    counter = Counter()
    for _, r, _ in triples:
        counter[r] += 1
    total = sum(counter.values())
    if total == 0:
        return counter, 0.0
    entropy = -sum((cnt / total) * math.log(cnt / total, 2) for cnt in counter.values() if cnt)
    return counter, entropy


def compute_fanout(triples: Iterable[Triple]) -> Dict[str, Dict[str, float]]:
    per_rel_head = defaultdict(lambda: defaultdict(set))
    per_rel_tail = defaultdict(lambda: defaultdict(set))
    for h, r, t in triples:
        per_rel_head[r][h].add(t)
        per_rel_tail[r][t].add(h)

    fanout = {}
    for rel in per_rel_head:
        heads = per_rel_head[rel]
        tails = per_rel_tail[rel]
        avg_tail = sum(len(ts) for ts in heads.values()) / len(heads) if heads else 0.0
        avg_head = sum(len(hs) for hs in tails.values()) / len(tails) if tails else 0.0
        fanout[rel] = {
            "avg_tails_per_head": avg_tail,
            "avg_heads_per_tail": avg_head,
        }
    return fanout


def component_stats(triples: Iterable[Triple]) -> Dict[str, float]:
    parent: Dict[str, str] = {}
    size: Dict[str, int] = {}

    def find(x: str) -> str:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if size[rx] < size[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        size[rx] += size[ry]

    def add_node(x: str) -> None:
        if x not in parent:
            parent[x] = x
            size[x] = 1

    for h, _, t in triples:
        add_node(h)
        add_node(t)
        union(h, t)

    if not parent:
        return {
            "num_components": 0,
            "largest_component_ratio": 0.0,
        }

    comp_sizes = Counter(find(node) for node in parent)
    largest = max(comp_sizes.values())
    total = sum(comp_sizes.values())
    return {
        "num_components": len(comp_sizes),
        "largest_component_ratio": largest / total * 100,
    }


def split_coverage(train_triples: Set[Triple], eval_triples: Iterable[Triple]) -> Dict[str, float]:
    train_heads = defaultdict(set)
    train_tails = defaultdict(set)
    for h, r, t in train_triples:
        train_heads[h].add(r)
        train_tails[t].add(r)

    covered = 0
    multi_relation = 0
    total = 0
    for h, _, t in eval_triples:
        total += 1
        if h in train_heads or t in train_tails:
            covered += 1
        if (len(train_heads.get(h, set())) > 1) or (len(train_tails.get(t, set())) > 1):
            multi_relation += 1
    return {
        "entity_seen_ratio": covered / total * 100 if total else 0.0,
        "multi_relation_ratio": multi_relation / total * 100 if total else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute KG diagnostics from triple files.")
    parser.add_argument("data_root", type=Path, help="Directory containing train.txt/val.txt/test.txt")
    parser.add_argument(
        "--temporal-relations",
        nargs="*",
        default=[],
        help="Relations to treat as temporal and exclude from the optional filtered stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.data_root / "train.txt"
    val_path = args.data_root / "val.txt"
    test_path = args.data_root / "test.txt"

    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing triple file: {path}")

    train = load_triples(train_path)
    val = load_triples(val_path)
    test = load_triples(test_path)

    all_triples = train + val + test

    print("=== Triple Counts ===")
    print(f"train: {len(train)}  val: {len(val)}  test: {len(test)}  total: {len(all_triples)}")

    print("\n=== Degree Statistics (full graph) ===")
    degree = compute_degrees(all_triples)
    for key, value in describe_degree(degree).items():
        print(f"{key}: {value}")

    print("\n=== Relation Frequency ===")
    rel_counter, entropy = relation_statistics(all_triples)
    for rel, count in rel_counter.most_common():
        print(f"{rel}: {count}")
    print(f"entropy(bits): {entropy:.4f}")

    print("\n=== Relation Fan-out (avg tails/head & heads/tail) ===")
    fanout = compute_fanout(all_triples)
    for rel, stats in fanout.items():
        print(f"{rel}: tails/head={stats['avg_tails_per_head']:.2f}, heads/tail={stats['avg_heads_per_tail']:.2f}")

    print("\n=== Component Statistics ===")
    comp = component_stats(all_triples)
    for key, value in comp.items():
        print(f"{key}: {value}")

    print("\n=== Split Coverage (val/test vs train) ===")
    train_set = set(train)
    coverage_val = split_coverage(train_set, val)
    coverage_test = split_coverage(train_set, test)
    print(f"val: {coverage_val}")
    print(f"test: {coverage_test}")

    if args.temporal_relations:
        print("\n=== Degree Statistics without temporal relations ===")
        temporal = set(args.temporal_relations)
        filtered = [tr for tr in all_triples if tr[1] not in temporal]
        degree_filtered = compute_degrees(filtered)
        for key, value in describe_degree(degree_filtered).items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
