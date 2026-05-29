"""
Complexity scorer for the autoresearch loop.

Implements the gate from CLAUDE.md:

    score = AST_node_count + 40 * cyclomatic_complexity

The 40x coefficient was chosen in the original autoresearcher design so AST
node count and cyclomatic complexity contribute roughly equally for typical
FSRS source files. Cyclomatic complexity follows McCabe: 1 base path per
function, +1 per branching construct:

    if / elif                             → +1 each
    for / while / async for               → +1 each
    except handler                        → +1 each
    each match case (3.10+)               → +1 each
    each `and` / `or` operand beyond 1st  → +1
    each `if` filter inside a comprehension → +1
    each comprehension generator          → +1
    ternary (a if b else c)               → +1
    assert                                → +1

Standalone Python — no third-party deps. Runs on the host (no Docker
needed).

Usage:
    python -m src.autoresearch.complexity \\
        src/models/fsrs_v7.py \\
        src/models/fsrs_v7_interval_penalty.py \\
        src/main/run.py \\
        src/main/config.py

Outputs JSON to stdout with per-file and total stats.

Use --baseline to compare against a previous run and gate accept/reject:
    # produce baseline
    python -m src.autoresearch.complexity src/models/fsrs_v7.py > baseline.json
    # ... modify code ...
    # check delta (exits 1 if over budget)
    python -m src.autoresearch.complexity src/models/fsrs_v7.py \\
        --baseline baseline.json --max-pct 5.0
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileStats:
    path: str
    ast_nodes: int
    cyclomatic: int
    score: int  # ast_nodes + 40 * cyclomatic


class CyclomaticVisitor(ast.NodeVisitor):
    """Compute McCabe cyclomatic complexity over an AST."""

    def __init__(self) -> None:
        self.complexity: int = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # `a and b and c` adds 2 branches beyond the base; same for `or`
        self.complexity += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_match_case(self, node: ast.AST) -> None:  # py3.10+
        self.complexity += 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        # comprehension generator + each `if` filter
        self.complexity += 1 + len(node.ifs)
        self.generic_visit(node)


def score_source(source: str, filename: str = "<string>") -> tuple[int, int]:
    """Return (ast_nodes, cyclomatic) for a Python source string."""
    tree = ast.parse(source, filename=filename)
    ast_nodes = sum(1 for _ in ast.walk(tree))
    visitor = CyclomaticVisitor()
    visitor.visit(tree)
    return ast_nodes, visitor.complexity


def score_file(path: Path) -> FileStats:
    source = path.read_text(encoding="utf-8")
    ast_nodes, cyc = score_source(source, str(path))
    return FileStats(
        path=str(path),
        ast_nodes=ast_nodes,
        cyclomatic=cyc,
        score=ast_nodes + 40 * cyc,
    )


def score_paths(paths: list[Path]) -> dict:
    """Score a list of files and return the aggregate JSON-serialisable dict."""
    files = [score_file(p) for p in paths]
    total_nodes = sum(f.ast_nodes for f in files)
    total_cyc = sum(f.cyclomatic for f in files)
    return {
        "files": [asdict(f) for f in files],
        "total_ast_nodes": total_nodes,
        "total_cyclomatic": total_cyc,
        "total_score": total_nodes + 40 * total_cyc,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute FSRS-style complexity score over Python files.",
    )
    parser.add_argument("paths", nargs="+", help="Python files to score")
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help=(
            "Path to a previous output JSON. If provided, compares total_score "
            "and exits 1 when the increase exceeds --max-pct."
        ),
    )
    parser.add_argument(
        "--max-pct",
        type=float,
        default=5.0,
        help="Max allowed percent increase vs baseline (default 5.0).",
    )
    args = parser.parse_args(argv)

    result = score_paths([Path(p) for p in args.paths])

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        prev = baseline["total_score"]
        pct = 100.0 * (result["total_score"] - prev) / prev if prev else 0.0
        result["baseline_score"] = prev
        result["pct_change"] = pct
        result["over_budget"] = pct > args.max_pct
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 1 if pct > args.max_pct else 0

    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
