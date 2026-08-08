"""Fusion reward based on reference semantics assigned to generated launches.

The reference defragmentation plans define adjacent operator boundaries that
should be fused. Generated Torch operations provide semantic anchors, while
Triton/CUDA launches are identified structurally rather than by name.
Correctness is enforced by the caller, so semantics already inferred by the
DefragmentationFramework may constrain which contiguous residual semantics
belong to each launch. Unknown custom launches remain conservative wildcards.

A boundary is proven fused only when every valid alignment assigns both sides
to the same custom launch. Missing, opaque, ambiguous, or unparsable generated
graphs leave the boundary unknown and therefore neutral. Exact ordered
semantics in one launched custom-kernel name may prove an unknown boundary.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import tempfile
import threading
import fcntl
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Default location: sibling checkout of the KernelGYM repo.
# .../daniel/KernelGYM/drkernel/kernel/rewards/fusion_helper.py -> .../daniel/DefragmentationFramework
_DEFAULT_DEFRAG_ROOT = Path(__file__).resolve().parents[4] / "DefragmentationFramework"
_DEFAULT_REFERENCE_CACHE_ROOT = (
    Path(__file__).resolve().parents[4]
    / "Results"
    / "RL-Training"
    / "reference_fusion_cache"
)
_REFERENCE_CONTEXT_VERSION = "reference-export-v5-scoped-library-anchors"

# Plan actions whose first/last semantic is a producer anchor.
_ANCHOR_FIRST_ACTIONS = {"fuse_epilogue", "fuse_pointwise_diamond"}
_ANCHOR_LAST_ACTIONS = {"fuse_prologue"}

# Entries ending with "*" match by prefix: "conv*" covers conv1d/2d/3d,
# conv_transpose*, convolution, ... as emitted by the semantic lift.
# Optimized-library convolution and linear calls are not custom-fusion targets,
# so they receive no additional anchor bonus by default.
DEFAULT_ANCHOR_SEMANTICS: Tuple[str, ...] = ()

_NAME_TOKEN_ALIASES = {
    "addmm": "linear",
    "batchnorm": "batch_norm",
    "bn": "batch_norm",
    "hswish": "hardswish",
    "maximum": "max",
    "minimum": "min",
    "multiply": "mul",
}

_import_lock = threading.Lock()
_defrag_modules: Optional[Dict[str, Any]] = None
_defrag_import_error: Optional[str] = None


def _load_defrag_modules(defrag_framework_path: Optional[str] = None) -> Dict[str, Any]:
    """Import the DefragmentationFramework core modules once, thread-safe.

    The framework is a plain checkout (imported as ``scripts.*`` with its
    repo root on sys.path), so we append the root lazily on first use.
    """
    global _defrag_modules, _defrag_import_error
    with _import_lock:
        if _defrag_modules is not None:
            return _defrag_modules
        if _defrag_import_error is not None:
            raise ImportError(_defrag_import_error)

        root = Path(
            defrag_framework_path
            or os.environ.get("DEFRAG_FRAMEWORK_PATH")
            or _DEFAULT_DEFRAG_ROOT
        )
        try:
            if not (root / "scripts" / "common").is_dir():
                raise ImportError(f"DefragmentationFramework not found at {root}")
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from scripts.ast_based.program_trace import parse_program
            from scripts.common.defrag_plan import plan_defragmentation
            from scripts.common.fragment_analyze import analyze_fragments
            from scripts.common.semantic_lift import lift_graph
            from scripts.torch_export_based.export_trace import export_file
        except Exception as e:  # noqa: BLE001
            _defrag_import_error = f"failed to import DefragmentationFramework from {root}: {e}"
            raise ImportError(_defrag_import_error) from e

        _defrag_modules = {
            "parse_program": parse_program,
            "lift_graph": lift_graph,
            "analyze_fragments": analyze_fragments,
            "plan_defragmentation": plan_defragmentation,
            "export_file": export_file,
        }
        return _defrag_modules


def generate_fusion_plans(
    source_code: str,
    defrag_framework_path: Optional[str] = None,
    parse_mode: str = "defrag",
) -> List[Dict[str, Any]]:
    """Run the static defragmentation pipeline and return fusion-only plans."""
    _, plans = analyze_fusion_source(source_code, defrag_framework_path, parse_mode)
    return plans


def analyze_fusion_source(
    source_code: str,
    defrag_framework_path: Optional[str] = None,
    parse_mode: str = "defrag",
    include_library_anchors: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return the lifted graph and its remaining fusion opportunities."""
    mods = _load_defrag_modules(defrag_framework_path)
    graph = mods["parse_program"](source_code, mode=parse_mode).to_dict()
    lifted = mods["lift_graph"](graph)
    fragments = mods["analyze_fragments"](lifted)
    plans = mods["plan_defragmentation"](
        fragments,
        lifted,
        fusion_only=True,
        include_library_anchors=include_library_anchors,
    )
    return lifted, list(plans.get("plans", []))


def _plans_from_lifted_graph(
    lifted: Dict[str, Any],
    defrag_framework_path: Optional[str] = None,
    include_library_anchors: bool = False,
) -> List[Dict[str, Any]]:
    mods = _load_defrag_modules(defrag_framework_path)
    fragments = mods["analyze_fragments"](lifted)
    plans = mods["plan_defragmentation"](
        fragments,
        lifted,
        fusion_only=True,
        include_library_anchors=include_library_anchors,
    )
    return list(plans.get("plans", []))


def _reference_prompt_boundaries(
    plans: Sequence[Dict[str, Any]],
) -> List[Tuple[str, str]]:
    boundaries = []
    seen = set()
    for plan in plans:
        if plan.get("fallback_for"):
            continue
        semantics = [str(item) for item in plan.get("semantics", [])]
        for left, right in zip(semantics, semantics[1:]):
            boundary = (left, right)
            if boundary not in seen:
                seen.add(boundary)
                boundaries.append(boundary)
    return boundaries


def _reference_context_cache_root() -> Path:
    configured = os.environ.get("REFERENCE_FUSION_CACHE_DIR")
    return Path(configured) if configured else _DEFAULT_REFERENCE_CACHE_ROOT


def _reference_context_key(
    reference_code: str, include_library_anchors: bool
) -> str:
    payload = (
        f"{_REFERENCE_CONTEXT_VERSION}\0{include_library_anchors}"
        f"\0{reference_code}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _build_reference_fusion_context(
    reference_code: str,
    defrag_framework_path: Optional[str],
    include_library_anchors: bool,
) -> Dict[str, Any]:
    source_hash = hashlib.sha256(reference_code.encode()).hexdigest()
    analysis_mode = "export"
    export_error = None
    try:
        mods = _load_defrag_modules(defrag_framework_path)
        with tempfile.TemporaryDirectory(prefix="reference_fusion_export_") as temp_dir:
            reference_path = Path(temp_dir) / "reference.py"
            reference_path.write_text(reference_code)
            program, export_metadata = mods["export_file"](
                reference_path,
                model_class="Model",
                use_meta=True,
                strict=False,
                return_metadata=True,
            )
        lifted = mods["lift_graph"](program.to_dict())
        plans = _plans_from_lifted_graph(
            lifted,
            defrag_framework_path,
            include_library_anchors=include_library_anchors,
        )
        analysis_mode = str(export_metadata.get("mode", "export"))
    except Exception as exc:  # noqa: BLE001
        export_error = f"{type(exc).__name__}: {exc}"[:2000]
        lifted, plans = analyze_fusion_source(
            reference_code,
            defrag_framework_path,
            include_library_anchors=include_library_anchors,
        )
        analysis_mode = "ast_fallback"
    return {
        "version": _REFERENCE_CONTEXT_VERSION,
        "include_library_anchors": include_library_anchors,
        "source_hash": source_hash,
        "analysis_mode": analysis_mode,
        "export_error": export_error,
        "lifted_graph": _compact_lifted_graph(lifted, 4096),
        "fusion_plans": plans,
        "prompt_boundaries": [
            [left, right] for left, right in _reference_prompt_boundaries(plans)
        ],
    }


@lru_cache(maxsize=512)
def reference_fusion_context(
    reference_code: str,
    defrag_framework_path: Optional[str] = None,
    include_library_anchors: bool = False,
) -> Dict[str, Any]:
    """Return one shared, versioned reference graph and plan bundle."""
    reference_code = reference_code or ""
    cache_root = _reference_context_cache_root()
    cache_key = _reference_context_key(
        reference_code, include_library_anchors
    )
    cache_path = cache_root / f"{cache_key}.json"
    lock_path = cache_root / f"{cache_key}.lock"
    cache_root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if cache_path.is_file():
                try:
                    cached = json.loads(cache_path.read_text())
                    if (
                        cached.get("version") == _REFERENCE_CONTEXT_VERSION
                        and cached.get("include_library_anchors")
                        is include_library_anchors
                    ):
                        return cached
                except Exception:  # noqa: BLE001
                    pass
            context = _build_reference_fusion_context(
                reference_code,
                defrag_framework_path,
                include_library_anchors,
            )
            temporary_path = cache_path.with_suffix(
                f".{os.getpid()}.tmp"
            )
            temporary_path.write_text(json.dumps(context, sort_keys=True))
            os.replace(temporary_path, cache_path)
            return context
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def plan_signature(plan: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    """Graph-independent identity of a fusion plan for cross-kernel matching."""
    return (str(plan.get("action", "")), tuple(plan.get("semantics", [])))


def anchor_semantic_matches(semantic: Optional[str], patterns: Sequence[str]) -> bool:
    """True when the semantic matches a pattern ("foo" exact, "foo*" prefix)."""
    if not semantic:
        return False
    for pattern in patterns:
        pattern = str(pattern)
        if pattern.endswith("*"):
            if semantic.startswith(pattern[:-1]):
                return True
        elif semantic == pattern:
            return True
    return False


def plan_anchor_semantic(plan: Dict[str, Any]) -> Optional[str]:
    """Return the producer/anchor semantic of an anchored fusion plan."""
    semantics = list(plan.get("semantics", []))
    if not semantics:
        return None
    action = plan.get("action")
    if action in _ANCHOR_FIRST_ACTIONS:
        return semantics[0]
    if action in _ANCHOR_LAST_ACTIONS:
        return semantics[-1]
    return None


def format_fusion_plan(plan: Dict[str, Any]) -> str:
    """Compact single-line rendering of a fusion plan for log output."""
    action = str(plan.get("action", "?"))
    semantics = "+".join(plan.get("semantics", [])) or "?"
    anchor = plan_anchor_semantic(plan)
    anchor_note = f" anchor={anchor}" if anchor else ""
    return f"{action}[{semantics}]{anchor_note}"


@lru_cache(maxsize=512)
def reference_fusion_plans_text(
    reference_code: str, defrag_framework_path: Optional[str] = None
) -> Optional[str]:
    """Render unique adjacent fusion boundaries as an ordered bullet list.

    Used to inject in-context fusion guidance into the first-turn prompt when
    the fusion reward is active. Returns None when the reference has no fusion
    plans or cannot be analyzed, so callers can skip the guidance block.
    Cached because the same reference is rolled out n_samples times per batch.
    """
    try:
        context = reference_fusion_context(
            reference_code or "", defrag_framework_path
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"reference fusion plan generation for prompt failed: {e}")
        return None
    boundaries = [
        (str(item[0]), str(item[1]))
        for item in context.get("prompt_boundaries", [])
        if isinstance(item, list) and len(item) == 2
    ]
    if not boundaries:
        return None
    return "\n".join(f"- {left} -> {right}" for left, right in boundaries)


def _is_custom_launch(node: Dict[str, Any]) -> bool:
    return (
        node.get("backend") in {"triton", "cuda_extension"}
        and node.get("original_kind") in {"triton_wrapper", "cuda_extension"}
    )


def _custom_lifted_semantics(node: Dict[str, Any]) -> Tuple[str, ...]:
    """Return semantics supplied by the shared semantic-lifting pipeline."""
    semantics = tuple(str(item) for item in node.get("semantics", []))
    if not semantics or semantics == ("unknown",):
        return ()
    return semantics


def _canonical_name_tokens(value: str) -> Tuple[str, ...]:
    """Return conservative semantic tokens from a semantic or launch name."""
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value))
    raw_tokens = re.findall(r"[a-z0-9]+", expanded.lower())
    tokens: List[str] = []
    index = 0
    while index < len(raw_tokens):
        pair = tuple(raw_tokens[index : index + 2])
        if pair == ("batch", "norm"):
            tokens.append("batch_norm")
            index += 2
            continue
        if pair == ("leaky", "relu"):
            tokens.append("leaky_relu")
            index += 2
            continue
        if pair in {("mask", "fill"), ("masked", "fill")}:
            tokens.append("masked_fill")
            index += 2
            continue
        token = _NAME_TOKEN_ALIASES.get(raw_tokens[index], raw_tokens[index])
        if token in {"conv1d", "conv2d", "conv3d"}:
            token = "conv"
        tokens.append(token)
        index += 1
    return tuple(tokens)


def _canonical_semantic(value: str) -> str:
    return _NAME_TOKEN_ALIASES.get(str(value), str(value))


def _name_proves_boundary(
    left: str, right: str, generated_graph: Optional[Dict[str, Any]]
) -> bool:
    """True when exactly one launched custom node names the ordered boundary."""
    target = (*_canonical_name_tokens(left), *_canonical_name_tokens(right))
    if not target:
        return False
    matching_launches = 0
    for node in (generated_graph or {}).get("lifted_nodes", []):
        if not _is_custom_launch(node):
            continue
        tokens = _canonical_name_tokens(str(node.get("name", "")))
        if any(
            tokens[index : index + len(target)] == target
            for index in range(len(tokens) - len(target) + 1)
        ):
            matching_launches += 1
    return matching_launches == 1


def _visible_semantics(node: Dict[str, Any]) -> Optional[List[str]]:
    """Return non-custom semantics safe to use as graph anchors."""
    if _is_custom_launch(node):
        return []
    node_name = str(node.get("name", ""))
    if node_name.startswith(
        (
            "aten.empty.",
            "aten.empty_like.",
            "aten.new_empty.",
            "aten.new_empty_strided.",
        )
    ):
        return []
    semantics = [str(item) for item in node.get("semantics", [])]
    if node.get("op_category") == "metadata":
        return []
    if not semantics or semantics == ["unknown"] or node.get("opacity") == "opaque":
        return None
    return semantics


def _enumerate_execution_paths(
    lifted_graph: Dict[str, Any], max_paths: int = 256
) -> Tuple[List[List[Dict[str, Any]]], bool]:
    nodes = {
        node["node_id"]: node
        for node in lifted_graph.get("lifted_nodes", [])
        if node.get("node_id")
    }
    if not nodes:
        return [], False

    successors = {node_id: [] for node_id in nodes}
    predecessors = {node_id: [] for node_id in nodes}
    for edge in lifted_graph.get("edges", []):
        src, dst = edge.get("src"), edge.get("dst")
        if src in nodes and dst in nodes:
            successors[src].append(dst)
            predecessors[dst].append(src)

    roots = [node_id for node_id in nodes if not predecessors[node_id]]
    returned_nodes = set(lifted_graph.get("returns", [])).intersection(nodes)
    terminal_nodes = {node_id for node_id in nodes if not successors[node_id]}
    sinks = returned_nodes | terminal_nodes
    paths: List[List[Dict[str, Any]]] = []
    truncated = False

    def visit(node_id: str, path: List[str]) -> None:
        nonlocal truncated
        if len(paths) >= max_paths:
            truncated = True
            return
        if node_id in path:
            return
        next_path = [*path, node_id]
        if not successors[node_id]:
            if node_id in sinks:
                paths.append([nodes[item] for item in next_path])
            return
        for successor in successors[node_id]:
            visit(successor, next_path)

    for root_index, root in enumerate(roots):
        visit(root, [])
        if len(paths) >= max_paths:
            if root_index < len(roots) - 1:
                truncated = True
            break
    return paths, truncated


def _align_semantics_to_path(
    reference_semantics: Sequence[str],
    path: Sequence[Dict[str, Any]],
    max_alignments: int = 256,
) -> Tuple[List[Tuple[str, ...]], bool]:
    """Align one reference chain to a contiguous generated path segment."""
    target = tuple(str(item) for item in reference_semantics)
    alignments: set[Tuple[str, ...]] = set()
    truncated = False

    def next_visible_semantics(node_index: int) -> Tuple[str, ...]:
        for following_node in path[node_index:]:
            if _is_custom_launch(following_node):
                return ()
            semantics = _visible_semantics(following_node)
            if semantics is None:
                return ()
            if semantics:
                return tuple(semantics)
        return ()

    def consume(node_index: int, semantic_index: int, assigned: Tuple[str, ...]) -> None:
        nonlocal truncated
        if len(alignments) >= max_alignments:
            truncated = True
            return
        if semantic_index == len(target):
            alignments.add(assigned)
            return
        if node_index >= len(path):
            return

        node = path[node_index]
        launch_id = str(node.get("node_id", ""))
        if _is_custom_launch(node):
            lifted_semantics = _custom_lifted_semantics(node)
            if node.get("profiled_execution_slot"):
                remaining = len(target) - semantic_index
                for count in range(1, remaining + 1):
                    candidate = target[semantic_index : semantic_index + count]
                    if lifted_semantics:
                        if node.get("semantic_constraint_unordered"):
                            available = Counter(candidate)
                            required = Counter(lifted_semantics)
                            if any(
                                available[semantic] < required_count
                                for semantic, required_count in required.items()
                            ):
                                continue
                        elif not any(
                            candidate[offset : offset + len(lifted_semantics)]
                            == lifted_semantics
                            for offset in range(
                                len(candidate) - len(lifted_semantics) + 1
                            )
                        ):
                            continue
                    consume(
                        node_index + 1,
                        semantic_index + count,
                        (*assigned, *((launch_id,) * count)),
                    )
                return
            if lifted_semantics:
                width = len(lifted_semantics)
                if target[semantic_index : semantic_index + width] == lifted_semantics:
                    following_semantics = next_visible_semantics(node_index + 1)
                    if (
                        semantic_index + width == len(target)
                        and set(lifted_semantics).intersection(following_semantics)
                    ):
                        return
                    consume(
                        node_index + 1,
                        semantic_index + width,
                        (*assigned, *((launch_id,) * width)),
                    )
                return
            if not node.get("allow_opaque_wildcard", True):
                return
            remaining = len(target) - semantic_index
            for count in range(1, remaining + 1):
                consume(
                    node_index + 1,
                    semantic_index + count,
                    (*assigned, *((launch_id,) * count)),
                )
            return

        semantics = _visible_semantics(node)
        if semantics is None:
            return
        if not semantics:
            consume(node_index + 1, semantic_index, assigned)
            return

        width = len(semantics)
        if target[semantic_index : semantic_index + width] == tuple(semantics):
            consume(
                node_index + 1,
                semantic_index + width,
                (*assigned, *((launch_id,) * width)),
            )

    for start in range(len(path)):
        consume(start, 0, ())
        if len(alignments) >= max_alignments:
            truncated = True
            break
    return sorted(alignments), truncated


def _path_is_relevant(
    path: Sequence[Dict[str, Any]], reference_semantics: Sequence[str], only_path: bool
) -> bool:
    if only_path:
        return True
    target = set(str(item) for item in reference_semantics)
    return any(
        target.intersection(str(item) for item in node.get("semantics", []))
        for node in path
    )


def _path_mentions_all_semantics(
    path: Sequence[Dict[str, Any]], reference_semantics: Sequence[str]
) -> bool:
    """True when a path already contains all exact reference semantics."""
    mentioned: List[str] = []
    for node in path:
        if _is_custom_launch(node):
            mentioned.extend(_custom_lifted_semantics(node))
            continue
        semantics = _visible_semantics(node)
        if semantics:
            mentioned.extend(semantics)
    available = Counter(mentioned)
    required = Counter(str(item) for item in reference_semantics)
    return all(available[semantic] >= count for semantic, count in required.items())


def _boundary_key(
    plan: Dict[str, Any], index: int, left: str, right: str
) -> Tuple[str, str, str, str]:
    node_ids = [str(item) for item in plan.get("node_ids", [])]
    if len(node_ids) == len(plan.get("semantics", [])):
        return node_ids[index], node_ids[index + 1], left, right
    return str(plan.get("fragment_id", "")), str(index), left, right


def classify_fusion_boundaries(
    reference_plans: List[Dict[str, Any]],
    generated_graph: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Classify unique adjacent reference boundaries using launch alignment."""
    paths, paths_truncated = _enumerate_execution_paths(generated_graph or {})
    custom_ids = {
        str(node.get("node_id"))
        for node in (generated_graph or {}).get("lifted_nodes", [])
        if _is_custom_launch(node)
    }
    boundaries: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

    for plan in reference_plans:
        semantics = [str(item) for item in plan.get("semantics", [])]
        if len(semantics) < 2:
            continue
        alignments: List[Tuple[str, ...]] = []
        occurrence_unknown = paths_truncated
        occurrence_truncated = paths_truncated
        occurrence_name_fallback_allowed = not any(
            _path_mentions_all_semantics(path, semantics) for path in paths
        )
        for path in paths:
            path_alignments, alignments_truncated = _align_semantics_to_path(
                semantics, path
            )
            alignments.extend(path_alignments)
            occurrence_truncated = occurrence_truncated or alignments_truncated
            if alignments_truncated or (
                not path_alignments
                and _path_is_relevant(path, semantics, only_path=len(paths) == 1)
            ):
                occurrence_unknown = True

        anchor = plan_anchor_semantic(plan)
        for index, (left, right) in enumerate(zip(semantics, semantics[1:])):
            key = _boundary_key(plan, index, left, right)
            evidence = [
                assignment[index] == assignment[index + 1]
                and assignment[index] in custom_ids
                for assignment in alignments
            ]
            if any(not item for item in evidence):
                occurrence_status = "not_fused"
            elif occurrence_unknown or not evidence:
                occurrence_status = "unknown"
            else:
                occurrence_status = "proven_fused"
            boundary_anchor = anchor if anchor in {left, right} else None
            if key not in boundaries:
                boundaries[key] = {
                    "left_semantic": left,
                    "right_semantic": right,
                    "status": occurrence_status,
                    "proven_fused": occurrence_status == "proven_fused",
                    "anchor_semantic": boundary_anchor,
                    "alignment_count": len(alignments),
                    "analysis_truncated": occurrence_truncated,
                    "proof_source": (
                        "graph" if occurrence_status == "proven_fused" else None
                    ),
                    "name_fallback_allowed": occurrence_name_fallback_allowed,
                }
            else:
                statuses = {boundaries[key]["status"], occurrence_status}
                if "not_fused" in statuses:
                    boundaries[key]["status"] = "not_fused"
                elif "unknown" in statuses:
                    boundaries[key]["status"] = "unknown"
                else:
                    boundaries[key]["status"] = "proven_fused"
                boundaries[key]["proven_fused"] = (
                    boundaries[key]["status"] == "proven_fused"
                )
                boundaries[key]["alignment_count"] += len(alignments)
                boundaries[key]["analysis_truncated"] = (
                    boundaries[key]["analysis_truncated"] or occurrence_truncated
                )
                boundaries[key]["name_fallback_allowed"] = (
                    boundaries[key]["name_fallback_allowed"]
                    and occurrence_name_fallback_allowed
                )
                if boundaries[key]["anchor_semantic"] is None:
                    boundaries[key]["anchor_semantic"] = boundary_anchor

    for boundary in boundaries.values():
        if (
            boundary["status"] == "unknown"
            and boundary["alignment_count"] == 0
            and not boundary["analysis_truncated"]
            and boundary["name_fallback_allowed"]
            and _name_proves_boundary(
                boundary["left_semantic"],
                boundary["right_semantic"],
                generated_graph,
            )
        ):
            boundary["status"] = "proven_fused"
            boundary["proven_fused"] = True
            boundary["proof_source"] = "name"

    return list(boundaries.values())


def format_fusion_boundary(boundary: Dict[str, Any]) -> str:
    status = str(boundary.get("status", "unknown"))
    source = boundary.get("proof_source")
    source_note = f";source={source}" if source else ""
    return (
        f"{boundary.get('left_semantic', '?')}|{boundary.get('right_semantic', '?')}"
        f"[{status};alignments={boundary.get('alignment_count', 0)}{source_note}]"
    )


@dataclass
class FusionDetectionContext:
    reference_code: str
    kernel_code: str
    correctness: bool = False
    profiler_metadata: Dict[str, Any] = field(default_factory=dict)
    sample_extra_info: Dict[str, Any] = field(default_factory=dict)
    defrag_framework_path: Optional[str] = None
    max_graph_nodes: int = 2048


@dataclass
class FusionDetectionResult:
    detector: str
    reference_plans: List[Dict[str, Any]] = field(default_factory=list)
    generated_plans: List[Dict[str, Any]] = field(default_factory=list)
    boundaries: List[Dict[str, Any]] = field(default_factory=list)
    reference_graph: Optional[Dict[str, Any]] = None
    generated_graph: Optional[Dict[str, Any]] = None
    profiler_launches: List[str] = field(default_factory=list)
    plan_alignments: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    skipped_reason: Optional[str] = None
    analysis_truncated: bool = False
    reference_analysis_mode: Optional[str] = None
    reference_export_error: Optional[str] = None
    generated_export_mode: Optional[str] = None
    generated_export_warning: Optional[str] = None

    def artifact(self) -> Dict[str, Any]:
        return {
            "detector": self.detector,
            "reference_fusion_plans": self.reference_plans,
            "inferred_generated_fusion_plans": self.generated_plans,
            "reference_graph": self.reference_graph,
            "generated_graph": self.generated_graph,
            "profiler_launches": self.profiler_launches,
            "plan_alignments": self.plan_alignments,
            "boundaries": self.boundaries,
            "analysis_truncated": self.analysis_truncated,
            "error": self.error,
            "skipped_reason": self.skipped_reason,
            "reference_analysis_mode": self.reference_analysis_mode,
            "reference_export_error": self.reference_export_error,
            "generated_export_mode": self.generated_export_mode,
            "generated_export_warning": self.generated_export_warning,
        }


class AstFusionDetector:
    name = "ast"

    def detect(self, context: FusionDetectionContext) -> FusionDetectionResult:
        result = FusionDetectionResult(detector=self.name)
        try:
            reference_graph, result.reference_plans = analyze_fusion_source(
                context.reference_code, context.defrag_framework_path
            )
            result.reference_graph = _compact_lifted_graph(
                reference_graph, context.max_graph_nodes
            )
        except Exception as exc:  # noqa: BLE001
            result.error = (
                f"reference plan generation failed: {type(exc).__name__}: {exc}"
            )
            return result
        generated_graph = None
        try:
            generated_graph, generated_plans = analyze_fusion_source(
                context.kernel_code, context.defrag_framework_path
            )
            result.generated_graph = _compact_lifted_graph(
                generated_graph, context.max_graph_nodes
            )
            result.generated_plans = generated_plans
        except Exception as exc:  # noqa: BLE001
            result.error = (
                f"generated plan generation failed: {type(exc).__name__}: {exc}"
            )
        try:
            result.boundaries = classify_fusion_boundaries(
                result.reference_plans, generated_graph
            )
        except Exception as exc:  # noqa: BLE001
            result.error = (
                f"fusion boundary classification failed: {type(exc).__name__}: {exc}"
            )
            result.boundaries = classify_fusion_boundaries(
                result.reference_plans, None
            )
        result.analysis_truncated = any(
            item.get("analysis_truncated", False) for item in result.boundaries
        )
        return result


def _compact_lifted_graph(
    graph: Optional[Dict[str, Any]], max_nodes: int
) -> Optional[Dict[str, Any]]:
    if graph is None:
        return None
    nodes = list(graph.get("lifted_nodes", []))
    if len(nodes) > max_nodes:
        raise ValueError(f"graph_node_limit_exceeded:{len(nodes)}>{max_nodes}")
    compact_nodes = []
    for node in nodes:
        compact_nodes.append(
            {
                "node_id": node.get("node_id"),
                "name": node.get("name"),
                "original_kind": node.get("original_kind"),
                "backend": node.get("backend"),
                "semantics": list(node.get("semantics", [])),
                "op_category": node.get("op_category"),
                "opacity": node.get("opacity"),
            }
        )
    node_ids = {node.get("node_id") for node in compact_nodes}
    return {
        "lifted_nodes": compact_nodes,
        "edges": [
            {"src": edge.get("src"), "dst": edge.get("dst")}
            for edge in graph.get("edges", [])
            if edge.get("src") in node_ids or edge.get("dst") in node_ids
        ],
        "returns": list(graph.get("returns", [])),
    }


def _export_generated_lifted_graph(
    context: FusionDetectionContext,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    mods = _load_defrag_modules(context.defrag_framework_path)
    with tempfile.TemporaryDirectory(prefix="fusion_export_") as temp_dir:
        root = Path(temp_dir)
        reference_path = root / "reference.py"
        generated_path = root / "generated.py"
        reference_path.write_text(context.reference_code)
        generated_path.write_text(context.kernel_code)
        try:
            generated_program, export_metadata = mods["export_file"](
                generated_path,
                model_class="ModelNew",
                factory_path=reference_path,
                use_meta=True,
                strict=False,
                draft_fallback=True,
                return_metadata=True,
            )
            generated_graph = mods["lift_graph"](generated_program.to_dict())
        except Exception as export_exc:  # noqa: BLE001
            try:
                generated_graph, _ = analyze_fusion_source(
                    context.kernel_code, context.defrag_framework_path
                )
            except Exception as ast_exc:  # noqa: BLE001
                raise RuntimeError(
                    "generated export and AST fallback failed: "
                    f"{type(export_exc).__name__}: {export_exc}; "
                    f"{type(ast_exc).__name__}: {ast_exc}"
                ) from ast_exc
            export_metadata = {
                "mode": "ast_fallback",
                "warning": f"{type(export_exc).__name__}: {export_exc}",
            }
    _compact_lifted_graph(generated_graph, context.max_graph_nodes)
    return generated_graph, export_metadata


def _profiler_launch_names(metadata: Dict[str, Any]) -> List[str]:
    names = metadata.get("triton_kernel_in_profiling")
    if isinstance(names, list) and names:
        return [str(name) for name in names]
    profiling = metadata.get("profiling", {})
    kernels = profiling.get("kernels", []) if isinstance(profiling, dict) else []
    return [
        str(kernel.get("name", ""))
        for kernel in kernels
        if isinstance(kernel, dict) and kernel.get("name")
    ]


def _launch_semantic_name(name: str) -> str:
    match = re.match(r"triton_kernel::(.+?)__[0-9a-fA-F]+$", str(name))
    return match.group(1) if match else str(name)


def _profiler_confirms_launch(export_name: str, profiler_names: Sequence[str]) -> bool:
    semantic_name = _launch_semantic_name(export_name).lower()
    if not semantic_name or semantic_name == "unknown":
        return False
    return any(
        semantic_name in str(profile_name).lower()
        or str(profile_name).lower() in semantic_name
        for profile_name in profiler_names
    )


def _matched_profiler_launch_names(
    generated_graph: Dict[str, Any], metadata: Dict[str, Any]
) -> List[str]:
    profiler_names = _profiler_launch_names(metadata)
    export_names = [
        str(node.get("name", ""))
        for node in generated_graph.get("lifted_nodes", [])
        if _is_custom_launch(node)
    ]
    matched = []
    for profiler_name in profiler_names:
        if any(
            _profiler_confirms_launch(export_name, [profiler_name])
            for export_name in export_names
        ):
            matched.append(profiler_name)
    return matched


def _confirmed_launch_occurrences(
    generated_graph: Dict[str, Any], profiler_names: Sequence[str]
) -> Dict[str, Tuple[str, int]]:
    """Map confirmed custom nodes to export-ordered launch occurrences.

    Kernel profiler summaries are aggregated and commonly sorted by runtime,
    not execution order. They confirm presence only; FX/export node order
    supplies the occurrence order.
    """
    custom_nodes = [
        node
        for node in generated_graph.get("lifted_nodes", [])
        if _is_custom_launch(node)
    ]
    occurrences: Dict[str, Tuple[str, int]] = {}
    for export_index, node in enumerate(custom_nodes):
        export_name = str(node.get("name", ""))
        if not _profiler_confirms_launch(export_name, profiler_names):
            continue
        node_id = str(node.get("node_id", ""))
        occurrences[node_id] = (
            f"{node_id}@export:{export_index}",
            export_index,
        )
    return occurrences


def _unknown_boundaries(
    reference_plans: List[Dict[str, Any]], reason: str
) -> List[Dict[str, Any]]:
    boundaries: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for plan in reference_plans:
        semantics = [str(item) for item in plan.get("semantics", [])]
        anchor = plan_anchor_semantic(plan)
        for index, (left, right) in enumerate(zip(semantics, semantics[1:])):
            key = _boundary_key(plan, index, left, right)
            if key in boundaries:
                continue
            boundaries[key] = {
                "left_semantic": left,
                "right_semantic": right,
                "status": "unknown",
                "proven_fused": False,
                "anchor_semantic": anchor if anchor in {left, right} else None,
                "alignment_count": 0,
                "analysis_truncated": False,
                "proof_source": None,
                "reference_plan_id": plan.get("plan_id"),
                "unknown_reason": reason,
            }
    return list(boundaries.values())


def _named_semantic_span(
    launch_name: str, reference_semantics: Sequence[str]
) -> Optional[Tuple[str, ...]]:
    launch_tokens = _canonical_name_tokens(_launch_semantic_name(launch_name))
    matches: List[Tuple[int, int, Tuple[str, ...]]] = []
    for start in range(len(reference_semantics)):
        for end in range(start + 1, len(reference_semantics) + 1):
            semantics = tuple(str(item) for item in reference_semantics[start:end])
            target = tuple(
                token
                for semantic in semantics
                for token in _canonical_name_tokens(semantic)
            )
            if target and any(
                launch_tokens[index : index + len(target)] == target
                for index in range(len(launch_tokens) - len(target) + 1)
            ):
                matches.append((start, end, semantics))
    if not matches:
        return None
    max_width = max(end - start for start, end, _ in matches)
    longest = [item for item in matches if item[1] - item[0] == max_width]
    spans = {item[2] for item in longest}
    span = next(iter(spans)) if len(spans) == 1 else None
    return span


def _profiled_paths(
    generated_graph: Dict[str, Any],
    reference_semantics: Sequence[str],
    profiler_names: Sequence[str],
    allow_name_evidence: bool,
    allow_opaque_wildcards: bool,
) -> Tuple[List[List[Dict[str, Any]]], bool, List[str]]:
    occurrences = _confirmed_launch_occurrences(
        generated_graph, profiler_names
    )
    execution_graph = {
        **generated_graph,
        "edges": [
            dict(edge) for edge in generated_graph.get("edges", [])
        ],
    }
    ordered_nodes = [
        node_id
        for node_id, _ in sorted(
            occurrences.items(), key=lambda item: item[1][1]
        )
    ]
    existing_edges = {
        (str(edge.get("src", "")), str(edge.get("dst", "")))
        for edge in execution_graph["edges"]
    }
    for left_id, right_id in zip(ordered_nodes, ordered_nodes[1:]):
        if (
            (left_id, right_id) not in existing_edges
            and (right_id, left_id) not in existing_edges
        ):
            execution_graph["edges"].append(
                {"src": left_id, "dst": right_id, "kind": "profiler_order"}
            )
    paths, truncated = _enumerate_execution_paths(execution_graph)
    confirmed_ids: List[str] = []
    prepared_paths: List[List[Dict[str, Any]]] = []
    for path in paths:
        prepared = []
        for original in path:
            node = dict(original)
            if _is_custom_launch(node):
                launch_name = str(node.get("name", ""))
                occurrence = occurrences.get(str(node.get("node_id", "")))
                node["profiler_confirmed"] = occurrence is not None
                if occurrence is not None:
                    node["node_id"] = occurrence[0]
                node["lifted_semantics"] = list(node.get("semantics", []))
                if allow_name_evidence:
                    semantic_span = _named_semantic_span(
                        launch_name, reference_semantics
                    )
                else:
                    lifted = [
                        _canonical_semantic(item)
                        for item in node["lifted_semantics"]
                        if item and item != "unknown"
                    ]
                    semantic_span = tuple(lifted) or None
                node["semantics"] = list(semantic_span or ["unknown"])
                node["opacity"] = "partial" if semantic_span else "opaque"
                node["allow_opaque_wildcard"] = allow_opaque_wildcards
                node["profiled_execution_slot"] = (
                    node["profiler_confirmed"] and allow_opaque_wildcards
                )
                node["semantic_constraint_unordered"] = (
                    launch_name.startswith("triton_kernel::")
                )
                if node["profiler_confirmed"]:
                    confirmed_ids.append(str(node.get("node_id", "")))
            else:
                node["semantics"] = [
                    _canonical_semantic(item)
                    for item in node.get("semantics", [])
                ]
            prepared.append(node)
        prepared_paths.append(prepared)
    return prepared_paths, truncated, confirmed_ids


def _contains_name_tokens(
    launch_tokens: Sequence[str], target_tokens: Sequence[str]
) -> bool:
    return bool(target_tokens) and any(
        tuple(launch_tokens[index : index + len(target_tokens)])
        == tuple(target_tokens)
        for index in range(len(launch_tokens) - len(target_tokens) + 1)
    )


def _named_launch_evidence_by_position(
    generated_graph: Dict[str, Any],
    reference_semantics: Sequence[str],
    profiler_names: Sequence[str],
) -> Tuple[List[set[str]], List[set[str]], Dict[str, int]]:
    """Return position-aware name evidence for unresolved alignments.

    Semantic occurrences are indexed by their position in the reference plan,
    so repeated operations remain distinct. A boundary launch must name the
    ordered adjacent pair; merely containing both tokens in another order is
    not fusion evidence.
    """
    semantic_launch_ids = [set() for _ in reference_semantics]
    boundary_launch_ids = [
        set() for _ in range(max(0, len(reference_semantics) - 1))
    ]
    launch_order: Dict[str, int] = {}
    occurrences = _confirmed_launch_occurrences(
        generated_graph, profiler_names
    )
    for node in generated_graph.get("lifted_nodes", []):
        if not _is_custom_launch(node):
            continue
        launch_name = str(node.get("name", ""))
        occurrence = occurrences.get(str(node.get("node_id", "")))
        if occurrence is None:
            continue
        launch_tokens = _canonical_name_tokens(_launch_semantic_name(launch_name))
        launch_id, profile_index = occurrence
        launch_order[launch_id] = profile_index
        for index, semantic in enumerate(reference_semantics):
            if _contains_name_tokens(
                launch_tokens, _canonical_name_tokens(semantic)
            ):
                semantic_launch_ids[index].add(launch_id)
        for index, (left, right) in enumerate(
            zip(reference_semantics, reference_semantics[1:])
        ):
            target = (
                *_canonical_name_tokens(left),
                *_canonical_name_tokens(right),
            )
            repeated = (
                _canonical_name_tokens(left) == _canonical_name_tokens(right)
                and _contains_name_tokens(
                    launch_tokens, _canonical_name_tokens(left)
                )
                and any(
                    token in {"twice", "double", "2"}
                    for token in launch_tokens
                )
            )
            if _contains_name_tokens(launch_tokens, target) or repeated:
                boundary_launch_ids[index].add(launch_id)
    return semantic_launch_ids, boundary_launch_ids, launch_order


def _unique_visible_assignments(
    generated_graph: Dict[str, Any], reference_semantics: Sequence[str]
) -> Dict[int, str]:
    visible_ids: Dict[str, set[str]] = {}
    for node in generated_graph.get("lifted_nodes", []):
        semantics = _visible_semantics(node)
        if not semantics:
            continue
        node_id = str(node.get("node_id", ""))
        for semantic in semantics:
            visible_ids.setdefault(_canonical_semantic(semantic), set()).add(node_id)
    required = {}
    for index, semantic in enumerate(reference_semantics):
        candidates = visible_ids.get(_canonical_semantic(str(semantic)), set())
        if len(candidates) == 1:
            required[index] = next(iter(candidates))
    return required


def _reference_contexts_for_plan(
    plan: Dict[str, Any],
    reference_graph: Optional[Dict[str, Any]],
    max_contexts: int = 32,
) -> Tuple[List[Tuple[List[str], List[int]]], bool]:
    """Return reference-path semantics and plan positions for one plan."""
    plan_semantics = [
        _canonical_semantic(str(item)) for item in plan.get("semantics", [])
    ]
    plan_node_ids = [str(item) for item in plan.get("node_ids", [])]
    fallback = [(plan_semantics, list(range(len(plan_semantics))))]
    if (
        not reference_graph
        or len(plan_node_ids) != len(plan_semantics)
        or not plan_semantics
    ):
        return fallback, False

    paths, truncated = _enumerate_execution_paths(
        reference_graph, max_paths=max_contexts
    )
    contexts: List[Tuple[List[str], List[int]]] = []
    for path in paths:
        occurrences: List[Tuple[str, str]] = []
        for node in path:
            if node.get("op_category") == "metadata":
                continue
            node_id = str(node.get("node_id", ""))
            for semantic in node.get("semantics", []):
                canonical = _canonical_semantic(str(semantic))
                if canonical and canonical != "unknown":
                    occurrences.append((node_id, canonical))
        positions: List[int] = []
        cursor = 0
        for plan_node_id, plan_semantic in zip(
            plan_node_ids, plan_semantics
        ):
            match = next(
                (
                    index
                    for index in range(cursor, len(occurrences))
                    if occurrences[index]
                    == (plan_node_id, plan_semantic)
                ),
                None,
            )
            if match is None:
                positions = []
                break
            positions.append(match)
            cursor = match + 1
        if positions:
            context_semantics = [semantic for _, semantic in occurrences]
            item = (context_semantics, positions)
            if item not in contexts:
                contexts.append(item)
    return (contexts or fallback), truncated


def _classify_profiled_boundaries(
    reference_plans: List[Dict[str, Any]],
    generated_graph: Dict[str, Any],
    profiler_names: Sequence[str],
    allow_name_evidence: bool,
    allow_opaque_wildcards: bool = True,
    reference_graph: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    boundaries: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    plan_alignments: List[Dict[str, Any]] = []
    inferred_plans: List[Dict[str, Any]] = []

    for plan in reference_plans:
        if plan.get("fallback_for"):
            continue
        semantics = [str(item) for item in plan.get("semantics", [])]
        if len(semantics) < 2:
            continue
        contexts, context_truncated = _reference_contexts_for_plan(
            plan, reference_graph
        )
        (
            named_launch_ids,
            named_boundary_launch_ids,
            launch_order,
        ) = (
            _named_launch_evidence_by_position(
                generated_graph, semantics, profiler_names
            )
            if allow_name_evidence
            else (
                [set() for _ in semantics],
                [set() for _ in range(max(0, len(semantics) - 1))],
                {},
            )
        )
        required_visible_assignments = _unique_visible_assignments(
            generated_graph, semantics
        )
        alignments: List[Tuple[str, ...]] = []
        execution_group_counts: List[int] = []
        alignment_truncated = context_truncated
        confirmed: set[str] = set()
        for alignment_semantics, plan_positions in contexts:
            paths, paths_truncated, confirmed_ids = _profiled_paths(
                generated_graph,
                alignment_semantics,
                profiler_names,
                False,
                allow_opaque_wildcards,
            )
            confirmed.update(confirmed_ids)
            alignment_truncated = alignment_truncated or paths_truncated
            context_visible_assignments = _unique_visible_assignments(
                generated_graph, alignment_semantics
            )
            for path in paths:
                unconfirmed = [
                    node
                    for node in path
                    if _is_custom_launch(node)
                    and not node.get("profiler_confirmed")
                ]
                if unconfirmed:
                    continue
                path_alignments, truncated = _align_semantics_to_path(
                    alignment_semantics, path
                )
                required_custom_ids = {
                    str(node.get("node_id", ""))
                    for node in path
                    if _is_custom_launch(node)
                    and node.get("profiler_confirmed")
                }
                if required_custom_ids:
                    path_alignments = [
                        assignment
                        for assignment in path_alignments
                        if required_custom_ids.issubset(assignment)
                    ]
                required_visible_ids = {
                    str(node.get("node_id", ""))
                    for node in path
                    if not _is_custom_launch(node)
                    and (_visible_semantics(node) or [])
                    and set(_visible_semantics(node) or []).intersection(
                        alignment_semantics
                    )
                }
                if required_visible_ids:
                    path_alignments = [
                        assignment
                        for assignment in path_alignments
                        if required_visible_ids.issubset(assignment)
                    ]
                if context_visible_assignments:
                    path_alignments = [
                        assignment
                        for assignment in path_alignments
                        if all(
                            assignment[index] == node_id
                            for index, node_id in context_visible_assignments.items()
                        )
                    ]
                for assignment in path_alignments:
                    plan_assignment = tuple(
                        assignment[index] for index in plan_positions
                    )
                    alignments.append(plan_assignment)
                    # A lifted Python/Torch op and a Triton/CUDA call each have
                    # one node id. Repeated ids mean one execution group covers
                    # multiple reference semantics; consecutive kernel calls
                    # retain distinct ids and therefore count separately.
                    first_position = min(plan_positions)
                    last_position = max(plan_positions)
                    region_group_ids = {
                        assignment[index]
                        for index in range(first_position, last_position + 1)
                        if assignment[index]
                    }
                    execution_group_counts.append(len(region_group_ids))
                alignment_truncated = alignment_truncated or truncated

        unique_alignments = sorted(set(alignments))
        anchor = plan_anchor_semantic(plan)
        plan_record = {
            "plan_id": plan.get("plan_id"),
            "action": plan.get("action"),
            "semantics": semantics,
            "assignments": [list(item) for item in unique_alignments[:64]],
            "assignment_count": len(unique_alignments),
            "execution_group_counts": sorted(set(execution_group_counts)),
            "analysis_truncated": alignment_truncated or len(unique_alignments) > 64,
        }
        plan_alignments.append(plan_record)
        if len(unique_alignments) == 1:
            assignment = unique_alignments[0]
            launch_groups: Dict[str, List[str]] = {}
            for semantic, launch_id in zip(semantics, assignment):
                if launch_id in confirmed:
                    launch_groups.setdefault(launch_id, []).append(semantic)
            for launch_id, assigned_semantics in launch_groups.items():
                inferred_plans.append(
                    {
                        "reference_plan_id": plan.get("plan_id"),
                        "launch_id": launch_id,
                        "semantics": assigned_semantics,
                    }
                )

        for index, (left, right) in enumerate(zip(semantics, semantics[1:])):
            key = _boundary_key(plan, index, left, right)
            left_named = named_launch_ids[index]
            right_named = named_launch_ids[index + 1]
            boundary_named = named_boundary_launch_ids[index]
            left_visible = required_visible_assignments.get(index)
            right_visible = required_visible_assignments.get(index + 1)
            left_locations = set(left_named)
            right_locations = set(right_named)
            if left_visible:
                left_locations.add(left_visible)
            if right_visible:
                right_locations.add(right_visible)
            visible_name_conflict = bool(
                (left_visible and boundary_named)
                or (right_visible and boundary_named)
            )
            named_fused = len(boundary_named) == 1
            named_pairs = {
                (left_id, right_id)
                for left_id in left_locations
                for right_id in right_locations
                if (
                    left_id == right_id
                    and left_id in boundary_named
                )
                or (
                    left_id != right_id
                    and launch_order.get(left_id, -1)
                    < launch_order.get(right_id, 1 << 30)
                )
            }
            named_split = bool(named_pairs) and all(
                left_id != right_id for left_id, right_id in named_pairs
            )
            evidence = [
                assignment[index] == assignment[index + 1]
                and assignment[index] in confirmed
                for assignment in unique_alignments
            ]
            constrained_alignments = [
                assignment
                for assignment in unique_alignments
                if (
                    not left_named
                    or assignment[index] in left_named
                    or assignment[index] == left_visible
                )
                and (
                    not right_named
                    or assignment[index + 1] in right_named
                    or assignment[index + 1] == right_visible
                )
                and (
                    not boundary_named
                    or (
                        assignment[index] == assignment[index + 1]
                        and assignment[index] in boundary_named
                    )
                )
            ]
            constrained_evidence = [
                assignment[index] == assignment[index + 1]
                and assignment[index] in confirmed
                for assignment in constrained_alignments
            ]
            fused_launches = {
                assignment[index]
                for assignment, fused in zip(unique_alignments, evidence)
                if fused
            }
            if visible_name_conflict or named_split:
                status = "not_fused"
                proof_source = None
            elif evidence and not any(evidence):
                status = "not_fused"
                proof_source = None
            elif (
                evidence
                and all(evidence)
                and len(fused_launches) == 1
                and not alignment_truncated
            ):
                status = "proven_fused"
                proof_source = "profiled_export"
            elif alignment_truncated:
                status = "unknown"
                proof_source = None
            elif (
                constrained_evidence
                and all(constrained_evidence)
                and not visible_name_conflict
            ):
                status = "proven_fused"
                proof_source = "profiled_name"
            elif constrained_evidence and not any(constrained_evidence):
                status = "not_fused"
                proof_source = None
            elif evidence:
                status = "unknown"
                proof_source = None
            elif named_fused:
                status = "proven_fused"
                proof_source = "profiled_name"
            else:
                status = "unknown"
                proof_source = None
            record = {
                "left_semantic": left,
                "right_semantic": right,
                "status": status,
                "proven_fused": status == "proven_fused",
                "anchor_semantic": anchor if anchor in {left, right} else None,
                "alignment_count": len(unique_alignments),
                "analysis_truncated": alignment_truncated,
                "proof_source": proof_source,
                "reference_plan_id": plan.get("plan_id"),
            }
            previous = boundaries.get(key)
            if previous is None:
                boundaries[key] = record
            else:
                statuses = {previous["status"], status}
                if "not_fused" in statuses:
                    merged_status = "not_fused"
                elif "unknown" in statuses:
                    merged_status = "unknown"
                else:
                    merged_status = "proven_fused"
                previous["status"] = merged_status
                previous["proven_fused"] = merged_status == "proven_fused"
                previous["alignment_count"] += len(unique_alignments)
                previous["analysis_truncated"] |= alignment_truncated

    return list(boundaries.values()), plan_alignments, inferred_plans


class ProfiledExportFusionDetector:
    name = "profiled_export"

    def detect(self, context: FusionDetectionContext) -> FusionDetectionResult:
        result = FusionDetectionResult(detector=self.name)
        dataset_plan, skip_reason = _dataset_fusion_plan(context.sample_extra_info)
        if dataset_plan is None:
            result.skipped_reason = skip_reason
            return result
        result.reference_plans = [dataset_plan]
        try:
            reference_graph, export_metadata = _export_reference_lifted_graph(context)
            result.reference_graph = _compact_lifted_graph(
                reference_graph, context.max_graph_nodes
            )
            result.reference_analysis_mode = str(export_metadata.get("mode", "export"))
            warning = export_metadata.get("warning")
            result.reference_export_error = str(warning)[:2000] if warning else None
        except Exception as exc:  # noqa: BLE001
            result.error = (
                f"reference graph export failed: {type(exc).__name__}: {exc}"
            )
            return result
        try:
            generated_graph, export_metadata = _export_generated_lifted_graph(
                context
            )
            result.generated_graph = _compact_lifted_graph(
                generated_graph, context.max_graph_nodes
            )
            result.generated_export_mode = str(
                export_metadata.get("mode", "export")
            )
            warning = export_metadata.get("warning")
            result.generated_export_warning = (
                str(warning)[:2000] if warning else None
            )
            result.profiler_launches = _matched_profiler_launch_names(
                generated_graph, context.profiler_metadata
            )
            (
                result.boundaries,
                result.plan_alignments,
                result.generated_plans,
            ) = _classify_profiled_boundaries(
                result.reference_plans,
                generated_graph,
                result.profiler_launches,
                context.correctness,
                result.generated_export_mode == "export",
                result.reference_graph,
            )
            result.analysis_truncated = any(
                item.get("analysis_truncated", False)
                for item in result.boundaries
            )
        except Exception as exc:  # noqa: BLE001
            result.error = (
                f"profiled export detection failed: {type(exc).__name__}: {exc}"
            )
            result.generated_plans = []
            result.generated_graph = None
            result.profiler_launches = []
            result.plan_alignments = []
            result.boundaries = _unknown_boundaries(
                result.reference_plans, "export_failure"
            )
        return result


def _dataset_fusion_plan(
    sample_extra_info: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Build one compression target covering the complete reference program.

    Structured generation controls prompt construction only. Compression is
    measured across every reference operator for both structured and random
    samples, so neither ``is_structured`` nor ``fusion_chain`` limits reward
    eligibility or scope.
    """
    info = sample_extra_info if isinstance(sample_extra_info, dict) else {}
    operators = info.get("operators")
    if not isinstance(operators, (list, tuple)):
        return None, "missing_operator_chain"
    semantics = [
        str(item) for item in operators if item is not None and str(item)
    ]
    if len(semantics) < 2:
        return None, "operator_chain_too_short"

    program_graph = info.get("program_graph")
    graph_nodes = (
        program_graph.get("nodes", [])
        if isinstance(program_graph, dict)
        else []
    )
    node_ids = [
        str(node.get("node_id"))
        for node in graph_nodes
        if isinstance(node, dict) and node.get("node_id") is not None
    ]
    if len(node_ids) != len(semantics):
        node_ids = []

    return {
        "plan_id": "dataset_full_program",
        "action": "fuse",
        "semantics": semantics,
        "node_ids": node_ids,
        "target_nodes": node_ids,
    }, None


def _export_reference_lifted_graph(
    context: FusionDetectionContext,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Export/lift the reference graph without generating a fusion plan."""
    mods = _load_defrag_modules(context.defrag_framework_path)
    with tempfile.TemporaryDirectory(prefix="fusion_reference_export_") as temp_dir:
        reference_path = Path(temp_dir) / "reference.py"
        reference_path.write_text(context.reference_code)
        try:
            program, metadata = mods["export_file"](
                reference_path,
                model_class="Model",
                use_meta=True,
                strict=False,
                return_metadata=True,
            )
            graph = mods["lift_graph"](program.to_dict())
        except Exception as export_exc:  # noqa: BLE001
            try:
                program = mods["parse_program"](context.reference_code, mode="defrag")
                graph = mods["lift_graph"](program.to_dict())
            except Exception as ast_exc:  # noqa: BLE001
                raise RuntimeError(
                    "reference export and AST fallback failed: "
                    f"{type(export_exc).__name__}: {export_exc}; "
                    f"{type(ast_exc).__name__}: {ast_exc}"
                ) from ast_exc
            metadata = {
                "mode": "ast_fallback",
                "warning": f"{type(export_exc).__name__}: {export_exc}",
            }
    _compact_lifted_graph(graph, context.max_graph_nodes)
    return graph, metadata


def compute_fusion_reward(
    reference_code: str,
    kernel_code: str,
    *,
    fusion_weight: float = 0.5,
    speedup: float = 0.0,
    defrag_framework_path: Optional[str] = None,
    correctness: bool = False,
    profiler_metadata: Optional[Dict[str, Any]] = None,
    sample_extra_info: Optional[Dict[str, Any]] = None,
    detector: str = "profiled_export",
) -> Dict[str, Any]:
    """Compute the correctness-gated, speedup-weighted fusion reward.

    For a dataset fusion chain with N reference operators and G generated
    execution groups, gain=(N-G)/(N-1). Each aligned Python/Torch operation or
    Triton/CUDA call is one group. Thus no compression is 0 and complete fusion
    is exactly 1, independent of chain length.
    """
    zero = {
        "fusion_reward": 0.0,
        "compression_reference_ops": 0,
        "compression_execution_groups": None,
        "compression_ratio": None,
        "compression_gain": 0.0,
        "compression_speed_factor": 0.0,
        "fusion_error": None,
        "compression_skipped_reason": None,
        "fusion_detector": detector,
        "fusion_plan_artifact": None,
    }

    if not correctness:
        zero["compression_skipped_reason"] = "incorrect_kernel"
        return zero

    context = FusionDetectionContext(
        reference_code=reference_code or "",
        kernel_code=kernel_code or "",
        correctness=correctness,
        profiler_metadata=profiler_metadata or {},
        sample_extra_info=sample_extra_info or {},
        defrag_framework_path=defrag_framework_path,
    )
    selected_detector = ProfiledExportFusionDetector()
    detection = selected_detector.detect(context)
    zero["fusion_plan_artifact"] = detection.artifact()
    zero["fusion_error"] = detection.error
    plan = detection.reference_plans[0] if detection.reference_plans else None
    reference_ops = len(plan.get("semantics", [])) if plan else 0
    zero["compression_reference_ops"] = reference_ops
    if detection.skipped_reason:
        zero["compression_skipped_reason"] = detection.skipped_reason
        return zero
    if detection.error:
        zero["compression_skipped_reason"] = "graph_export_failed"
        return zero

    group_counts = [
        int(group_count)
        for alignment in detection.plan_alignments
        for group_count in alignment.get("execution_group_counts", [])
        if int(group_count) > 0
    ]
    if not group_counts:
        zero["compression_skipped_reason"] = "no_execution_group_alignment"
        return zero

    # Alignment ambiguity does not prevent measurement. If alternative valid
    # alignments span different group counts, use the conservative larger count
    # so ambiguity cannot inflate the fusion reward.
    execution_groups = max(group_counts)
    compression_ratio = execution_groups / reference_ops
    compression_gain = (reference_ops - execution_groups) / (reference_ops - 1)
    speed_factor = max(0.0, min((float(speedup) - 1.0) / 2.0, 1.0))
    fusion_reward = float(fusion_weight) * compression_gain * speed_factor
    zero.update(
        {
            "fusion_reward": fusion_reward,
            "compression_execution_groups": execution_groups,
            "compression_ratio": compression_ratio,
            "compression_gain": compression_gain,
            "compression_speed_factor": speed_factor,
            "compression_skipped_reason": None,
            "fusion_detector": detection.detector,
        }
    )
    zero["fusion_plan_artifact"]["reward"] = {
        "fusion_reward": fusion_reward,
        "fusion_weight": float(fusion_weight),
        "compression_reference_ops": reference_ops,
        "compression_execution_groups": execution_groups,
        "compression_ratio": compression_ratio,
        "compression_gain": compression_gain,
        "compression_speed_factor": speed_factor,
    }
    return zero
