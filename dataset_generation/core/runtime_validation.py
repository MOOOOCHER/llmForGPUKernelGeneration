from __future__ import annotations

import ast
import multiprocessing as mp
import re
import traceback
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeValidationResult:
    valid: bool
    reason: str = ""


class _ForwardFiniteCheckTransformer(ast.NodeTransformer):
    """Insert a finite-value check after assignments in Model.forward."""

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        if node.name != "Model":
            return node
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "forward":
                new_body: list[ast.stmt] = []
                for statement in item.body:
                    new_body.append(statement)
                    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                        target = statement.targets[0]
                        if isinstance(target, ast.Name):
                            check = ast.Expr(
                                value=ast.Call(
                                    func=ast.Name(id="_assert_finite", ctx=ast.Load()),
                                    args=[
                                        ast.Name(id=target.id, ctx=ast.Load()),
                                        ast.Constant(value=f"Model.forward:{target.id}"),
                                    ],
                                    keywords=[],
                                )
                            )
                            new_body.append(ast.copy_location(check, statement))
                item.body = new_body
        return node


def _rewrite_device(source: str, device: str) -> str:
    source = re.sub(
        r"device\s*=\s*(['\"])cuda(?::\d+)?\1",
        f"device='{device}'",
        source,
    )
    source = re.sub(
        r"torch\.device\(\s*(['\"])cuda(?::\d+)?\1\s*\)",
        f"torch.device('{device}')",
        source,
    )
    if device == "cpu":
        source = re.sub(r"\.cuda\(\s*(?:device\s*=\s*)?\d*\s*\)", ".cpu()", source)
    return source


def _instrument_forward(source: str) -> Any:
    tree = ast.parse(source)
    tree = _ForwardFiniteCheckTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return compile(tree, "<generated_sample>", "exec")


def _validation_worker(
    source: str,
    requested_device: str,
    repeats: int,
    check_intermediates: bool,
    connection: Any,
) -> None:
    try:
        import torch

        if requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = requested_device

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA validation requested but torch.cuda.is_available() is false")

        source = _rewrite_device(source, device)

        def _walk_tensors(value: Any):
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, dict):
                for nested in value.values():
                    yield from _walk_tensors(nested)
            elif isinstance(value, (list, tuple)):
                for nested in value:
                    yield from _walk_tensors(nested)

        def _assert_finite(value: Any, location: str) -> None:
            for tensor in _walk_tensors(value):
                if tensor.is_floating_point() or tensor.is_complex():
                    if not bool(torch.isfinite(tensor).all().item()):
                        raise RuntimeError(f"non-finite tensor at {location}")

        namespace: dict[str, Any] = {
            "__name__": "__generated_validation__",
            "_assert_finite": _assert_finite,
        }
        compiled = _instrument_forward(source) if check_intermediates else compile(
            source, "<generated_sample>", "exec"
        )
        exec(compiled, namespace)

        model_cls = namespace.get("Model")
        get_inputs = namespace.get("get_inputs")
        get_init_inputs = namespace.get("get_init_inputs")
        if model_cls is None or get_inputs is None:
            raise RuntimeError("generated program must define Model and get_inputs")

        init_inputs = get_init_inputs() if get_init_inputs is not None else []
        if init_inputs is None:
            init_inputs = []
        if not isinstance(init_inputs, (list, tuple)):
            init_inputs = [init_inputs]

        model = model_cls(*init_inputs).to(device)
        model.eval()
        inputs = get_inputs()
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        first_output = None
        with torch.inference_mode():
            for repeat in range(max(1, repeats)):
                output = model(*inputs)
                _assert_finite(output, "Model.forward:return")
                if repeat == 0:
                    first_output = output
                else:
                    first_tensors = list(_walk_tensors(first_output))
                    current_tensors = list(_walk_tensors(output))
                    if len(first_tensors) != len(current_tensors):
                        raise RuntimeError("output structure changed across repeated runs")
                    for previous, current in zip(first_tensors, current_tensors):
                        if previous.shape != current.shape or previous.dtype != current.dtype:
                            raise RuntimeError("output shape or dtype changed across repeated runs")
                        if previous.is_floating_point() or previous.is_complex():
                            if not torch.allclose(previous, current, rtol=1e-5, atol=1e-6, equal_nan=False):
                                raise RuntimeError("output values changed across repeated runs")
                        elif not torch.equal(previous, current):
                            raise RuntimeError("output values changed across repeated runs")

        del model, inputs, first_output
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        connection.send({"valid": True, "reason": ""})
    except BaseException as exc:
        try:
            connection.send(
                {
                    "valid": False,
                    "reason": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}",
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def validate_generated_program(
    source: str,
    *,
    device: str = "auto",
    timeout_seconds: float = 20.0,
    repeats: int = 2,
    check_intermediates: bool = True,
) -> RuntimeValidationResult:
    """Execute generated reference code in an isolated subprocess."""
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_validation_worker,
        args=(source, device, repeats, check_intermediates, child_connection),
    )
    process.start()
    child_connection.close()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(5)
        parent_connection.close()
        return RuntimeValidationResult(False, f"validation timed out after {timeout_seconds:g}s")

    try:
        if not parent_connection.poll(2.0):
            return RuntimeValidationResult(
                False,
                f"validation subprocess exited with code {process.exitcode} without returning a result",
            )
        payload = parent_connection.recv()
    except (EOFError, OSError) as exc:
        return RuntimeValidationResult(
            False,
            f"could not receive validation result after subprocess exit: {type(exc).__name__}: {exc}",
        )
    finally:
        parent_connection.close()

    return RuntimeValidationResult(bool(payload["valid"]), str(payload.get("reason", "")))
