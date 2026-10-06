from __future__ import annotations

from typing import Dict, List, Tuple

from .graph import OperatorNode, ProgramGraph
from .operators import OPERATORS


class PyTorchEmitter:
    def emit(self, graph: ProgramGraph, shapes: Dict[str, List[int]]) -> str:
        modules, module_calls = self._module_definitions(graph, shapes)
        input_edges = [e for e in graph.edges.values() if e.is_program_input]
        input_names = {edge.edge_id: f"input_{i}" for i, edge in enumerate(input_edges)}
        value_names = dict(input_names)

        lines = [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "",
            "",
            "class Model(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
        ]
        if modules:
            lines.extend(f"        {line}" for line in modules)
        else:
            pass

        args = ", ".join(input_names[e.edge_id] for e in input_edges)
        lines.extend(["", f"    def forward(self, {args}):"])

        for node in graph.topological_nodes():
            out_name = f"tensor_{node.output_edge[1:]}"
            value_names[node.output_edge] = out_name
            inputs = [value_names[e] for e in node.input_edges]
            expr = self._emit_expr(node, inputs, module_calls)
            lines.append(f"        {out_name} = {expr}")

        outputs = [value_names[e] for e in graph.output_edges]
        lines.append(f"        return {outputs[0] if len(outputs) == 1 else '[' + ', '.join(outputs) + ']'}")
        lines.extend(["", "", "def get_inputs():"])
        for edge in input_edges:
            shape = ", ".join(str(v) for v in shapes[edge.edge_id])
            if edge.create_op == "positive_rand":
                lines.append(
                    f"    {input_names[edge.edge_id]} = torch.rand({shape}, device='cuda', dtype=torch.float32) + 0.5"
                )
            else:
                fn = {"randn": "randn", "ones": "ones", "zeros": "zeros"}[edge.create_op]
                lines.append(f"    {input_names[edge.edge_id]} = torch.{fn}({shape}, device='cuda', dtype=torch.float32)")
        lines.append(f"    return [{', '.join(input_names[e.edge_id] for e in input_edges)}]")
        lines.extend(["", "", "def get_init_inputs():", "    return []", ""])
        return "\n".join(lines)

    def _module_definitions(self, graph: ProgramGraph, shapes: Dict[str, List[int]]) -> Tuple[List[str], Dict[str, str]]:
        lines: List[str] = []
        calls: Dict[str, str] = {}
        for node in graph.topological_nodes():
            op = node.op_name
            cat = OPERATORS[op].category
            name = f"op_{node.node_id}"
            x_shape = shapes[node.input_edges[0]]
            attrs = node.attributes
            if cat == "Linear":
                lines.append(f"self.{name} = nn.Linear({attrs['in_features']}, {attrs['out_features']}, bias={attrs['bias']})")
                calls[node.node_id] = f"self.{name}"
            elif cat == "LayerNorm":
                lines.append(f"self.{name} = nn.LayerNorm({x_shape[-1]})")
                calls[node.node_id] = f"self.{name}"
            elif cat == "BatchNorm":
                cls = {2: "BatchNorm1d", 3: "BatchNorm1d", 4: "BatchNorm2d", 5: "BatchNorm3d"}[len(x_shape)]
                lines.append(f"self.{name} = nn.{cls}({x_shape[1]})")
                calls[node.node_id] = f"self.{name}"
            elif cat == "GroupNorm":
                lines.append(f"self.{name} = nn.GroupNorm({attrs['num_groups']}, {x_shape[1]})")
                calls[node.node_id] = f"self.{name}"
            elif cat == "InstanceNorm":
                cls = {3: "InstanceNorm1d", 4: "InstanceNorm2d", 5: "InstanceNorm3d"}[len(x_shape)]
                lines.append(f"self.{name} = nn.{cls}({x_shape[1]}, affine=True)")
                calls[node.node_id] = f"self.{name}"
            elif cat.startswith("Conv"):
                dim = len(x_shape) - 2
                transposed = "Transpose" in cat
                cls = f"ConvTranspose{dim}d" if transposed else f"Conv{dim}d"
                lines.append(
                    f"self.{name} = nn.{cls}({x_shape[1]}, {attrs['out_channels']}, "
                    f"kernel_size={attrs['kernel_size']}, stride={attrs['stride']}, "
                    f"padding={attrs['padding']}, bias={attrs['bias']})"
                )
                calls[node.node_id] = f"self.{name}"
        return lines, calls

    def _emit_expr(self, node: OperatorNode, ins: List[str], module_calls: Dict[str, str]) -> str:
        op = node.op_name
        cat = OPERATORS[op].category
        a = node.attributes
        if node.node_id in module_calls:
            return f"{module_calls[node.node_id]}({ins[0]})"
        if cat == "UnaryPointwise":
            if op == "clamp": return f"torch.clamp({ins[0]}, min={a['min']}, max={a['max']})"
            if op in {"abs", "sin", "cos", "exp2", "sigmoid", "tanh"}: return f"torch.{op}({ins[0]})"
            return f"F.{op}({ins[0]})"
        if cat == "BinaryPointwise": return f"torch.{op}({ins[0]}, {ins[1]})"
        if cat == "Reduction":
            if op in {"max", "min"}: return f"torch.{op}({ins[0]}, dim={a['dim']}, keepdim={a['keepdim']}).values"
            if op == "norm": return f"torch.linalg.vector_norm({ins[0]}, dim={a['dim']}, keepdim={a['keepdim']})"
            return f"torch.{op}({ins[0]}, dim={a['dim']}, keepdim={a['keepdim']})"
        if cat == "Softmax": return f"F.{op}({ins[0]}, dim={a['dim']})"
        if cat == "Cumulative":
            suffix = ".values" if op in {"cummax", "cummin"} else ""
            return f"torch.{op}({ins[0]}, dim={a['dim']}){suffix}"
        if cat in {"Matmul", "BMM"}: return f"torch.{op}({ins[0]}, {ins[1]})"
        if cat.startswith("Pool"):
            return f"F.{op}({ins[0]}, kernel_size={a['kernel_size']}, stride={a['stride']})"
        raise NotImplementedError(f"cannot emit {op}/{cat}")
