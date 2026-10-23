#!/usr/bin/env python3
"""Rewrite ZipVoice Swoosh softplus subgraphs to avoid Log/Softplus.

ZipVoice SwooshL/R use:
  softplus(x - offset) - 0.08 * x - const

The exported ONNX represents softplus as Exp + Add + Log + Equal + Where.
Pulsar2 currently leaves those Log nodes as ONNX fallback subgraphs, and
Softplus replacement can fail as an unsupported lut_float op.  This script
replaces just the softplus subgraph with a no-log approximation made of
NPU-friendly primitive ops. The default is a piecewise-linear softplus:

  softplus(z) ~= b + m0 * z + sum_i delta_m_i * Relu(z - knot_i)

This preserves the surrounding Swoosh linear terms already present in the ONNX.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace Swoosh Log/Softplus pattern with hard-softplus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input ONNX model")
    parser.add_argument("--output", required=True, help="Output ONNX model")
    parser.add_argument(
        "--approx",
        choices=["pwl", "hard", "sigmoid-bump"],
        default="pwl",
        help="No-log softplus approximation",
    )
    parser.add_argument(
        "--include-substring",
        action="append",
        default=[],
        help="Only rewrite patterns whose Exp node name contains this substring. Can be repeated.",
    )
    parser.add_argument(
        "--hard-include-substring",
        action="append",
        default=[],
        help=(
            "Rewrite matching patterns with hard softplus while other rewritten "
            "patterns use --approx. Can be repeated."
        ),
    )
    return parser.parse_args()


def _producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }


def _consumer_map(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append(node)
    return consumers


def _single_consumer(
    consumers: dict[str, list[onnx.NodeProto]], tensor_name: str, op_type: str
) -> onnx.NodeProto | None:
    matches = [node for node in consumers.get(tensor_name, []) if node.op_type == op_type]
    if len(matches) != 1:
        return None
    return matches[0]


def _other_input(node: onnx.NodeProto, known_input: str) -> str | None:
    others = [input_name for input_name in node.input if input_name != known_input]
    if len(others) != 1:
        return None
    return others[0]


def _find_swoosh_patterns(model: onnx.ModelProto) -> list[dict[str, onnx.NodeProto | str]]:
    producers = _producer_map(model)
    consumers = _consumer_map(model)
    patterns: list[dict[str, onnx.NodeProto | str]] = []

    for exp_node in model.graph.node:
        if exp_node.op_type != "Exp" or not exp_node.name.endswith("/Exp"):
            continue
        if len(exp_node.input) != 1 or len(exp_node.output) != 1:
            continue

        z = exp_node.input[0]
        sub_node = producers.get(z)
        if sub_node is None or sub_node.op_type != "Sub":
            continue

        add_node = _single_consumer(consumers, exp_node.output[0], "Add")
        if add_node is None or len(add_node.output) != 1:
            continue

        log_node = _single_consumer(consumers, add_node.output[0], "Log")
        if log_node is None or len(log_node.output) != 1:
            continue

        equal_node = _single_consumer(consumers, log_node.output[0], "Equal")
        where_candidates = [
            node for node in consumers.get(log_node.output[0], []) if node.op_type == "Where"
        ]
        if equal_node is None or len(where_candidates) != 1:
            continue
        where_node = where_candidates[0]
        if len(where_node.input) != 3 or len(where_node.output) != 1:
            continue
        if where_node.input[0] != equal_node.output[0]:
            continue
        if where_node.input[1] != z or where_node.input[2] != log_node.output[0]:
            continue

        sub1_node = _single_consumer(consumers, where_node.output[0], "Sub")
        if sub1_node is None:
            continue

        if _other_input(add_node, exp_node.output[0]) is None:
            continue

        patterns.append(
            {
                "exp": exp_node,
                "add": add_node,
                "log": log_node,
                "equal": equal_node,
                "where": where_node,
                "sub1": sub1_node,
                "z": z,
            }
        )

    return patterns


def _find_stable_softplus_patterns(
    model: onnx.ModelProto,
) -> list[dict[str, onnx.NodeProto | str]]:
    """Find Max(z, 0) + Log(Exp(-Abs(z)) + 1) patterns."""
    producers = _producer_map(model)
    consumers = _consumer_map(model)
    patterns: list[dict[str, onnx.NodeProto | str]] = []

    for log_node in model.graph.node:
        if log_node.op_type != "Log" or len(log_node.input) != 1 or len(log_node.output) != 1:
            continue

        add_node = producers.get(log_node.input[0])
        if add_node is None or add_node.op_type != "Add":
            continue

        exp_node = None
        for input_name in add_node.input:
            producer = producers.get(input_name)
            if producer is not None and producer.op_type == "Exp":
                exp_node = producer
                break
        if exp_node is None or len(exp_node.input) != 1:
            continue

        neg_node = producers.get(exp_node.input[0])
        if neg_node is None or neg_node.op_type != "Neg" or len(neg_node.input) != 1:
            continue

        abs_node = producers.get(neg_node.input[0])
        if abs_node is None or abs_node.op_type != "Abs" or len(abs_node.input) != 1:
            continue

        final_add = _single_consumer(consumers, log_node.output[0], "Add")
        if final_add is None or len(final_add.output) != 1:
            continue

        max_input = _other_input(final_add, log_node.output[0])
        if max_input is None:
            continue
        max_node = producers.get(max_input)
        if max_node is None or max_node.op_type != "Max":
            continue

        # Use the Max non-constant input as z. This is the actual softplus input
        # in the stable form. It may be produced by a separate but equivalent Sub
        # from the Abs input after ONNX simplification.
        z_candidates = [
            input_name
            for input_name in max_node.input
            if producers.get(input_name) is not None
        ]
        if len(z_candidates) != 1:
            continue

        patterns.append(
            {
                "add": add_node,
                "exp": exp_node,
                "neg": neg_node,
                "abs": abs_node,
                "log": log_node,
                "max": max_node,
                "final_add": final_add,
                "z": z_candidates[0],
            }
        )

    return patterns


def _ensure_const(model: onnx.ModelProto, name: str, value: float) -> str:
    if not any(init.name == name for init in model.graph.initializer):
        model.graph.initializer.append(
            helper.make_tensor(name, TensorProto.FLOAT, dims=[], vals=[value])
        )
    return name


def _make_scalar_const(model: onnx.ModelProto, prefix: str, value: float) -> str:
    safe = str(value).replace("-", "neg").replace(".", "p")
    return _ensure_const(model, f"{prefix}_const_{safe}", value)


def rewrite_swoosh_nolog(
    model: onnx.ModelProto,
    approx: str = "pwl",
    include_substrings: list[str] | None = None,
    hard_include_substrings: list[str] | None = None,
) -> int:
    patterns = _find_swoosh_patterns(model)
    if include_substrings:
        patterns = [
            pattern
            for pattern in patterns
            if any(text in pattern["exp"].name for text in include_substrings)
        ]
    if not patterns:
        return 0

    c0 = _ensure_const(model, "hard_softplus_const_0", 0.0)
    c3 = _ensure_const(model, "hard_softplus_const_3", 3.0)
    c6 = _ensure_const(model, "hard_softplus_const_6", 6.0)
    c12 = _ensure_const(model, "hard_softplus_const_12", 12.0)

    remove_names = {
        node.name
        for pattern in patterns
        for node in (
            pattern["exp"],
            pattern["add"],
            pattern["log"],
            pattern["equal"],
            pattern["where"],
        )
    }
    insert_by_exp_name: dict[str, list[onnx.NodeProto]] = {}

    for index, pattern in enumerate(patterns):
        exp_node = pattern["exp"]
        where_node = pattern["where"]
        z = str(pattern["z"])
        prefix = exp_node.name.rsplit("/", 1)[0]
        hard_prefix = f"{prefix}/HardSoftplus_{index}"
        softplus_out = where_node.output[0]

        z_plus_3 = f"{hard_prefix}/Add3_output_0"
        h = f"{hard_prefix}/Clip_output_0"
        h2 = f"{hard_prefix}/Square_output_0"
        quad = f"{hard_prefix}/Div12_output_0"
        z_minus_3 = f"{hard_prefix}/Sub3_output_0"
        tail = f"{hard_prefix}/ReluTail_output_0"

        pattern_approx = approx
        if hard_include_substrings and any(
            text in exp_node.name for text in hard_include_substrings
        ):
            pattern_approx = "hard"

        if pattern_approx == "hard":
            insert_by_exp_name[exp_node.name] = _make_hard_softplus_nodes(
                z=z,
                output=softplus_out,
                prefix=hard_prefix,
                c0=c0,
                c3=c3,
                c6=c6,
                c12=c12,
            )
        elif pattern_approx == "sigmoid-bump":
            insert_by_exp_name[exp_node.name] = _make_sigmoid_bump_softplus_nodes(
                model=model,
                z=z,
                output=softplus_out,
                prefix=hard_prefix,
            )
        else:
            insert_by_exp_name[exp_node.name] = _make_pwl_softplus_nodes(
                model=model,
                z=z,
                output=softplus_out,
                prefix=hard_prefix,
            )

    new_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.name in insert_by_exp_name:
            new_nodes.extend(insert_by_exp_name[node.name])
        if node.name not in remove_names:
            new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return len(patterns)


def _make_hard_softplus_nodes(
    z: str,
    output: str,
    prefix: str,
    c0: str,
    c3: str,
    c6: str,
    c12: str,
) -> list[onnx.NodeProto]:
    z_plus_3 = f"{prefix}/Add3_output_0"
    h = f"{prefix}/Clip_output_0"
    h2 = f"{prefix}/Square_output_0"
    quad = f"{prefix}/Div12_output_0"
    z_minus_3 = f"{prefix}/Sub3_output_0"
    tail = f"{prefix}/ReluTail_output_0"

    return [
        helper.make_node("Add", [z, c3], [z_plus_3], name=f"{prefix}/Add3"),
        helper.make_node("Clip", [z_plus_3, c0, c6], [h], name=f"{prefix}/Clip"),
        helper.make_node("Mul", [h, h], [h2], name=f"{prefix}/Square"),
        helper.make_node("Div", [h2, c12], [quad], name=f"{prefix}/Div12"),
        helper.make_node("Sub", [z, c3], [z_minus_3], name=f"{prefix}/Sub3"),
        helper.make_node("Relu", [z_minus_3], [tail], name=f"{prefix}/ReluTail"),
        helper.make_node("Add", [quad, tail], [output], name=f"{prefix}/Add"),
    ]


def _make_pwl_softplus_nodes(
    model: onnx.ModelProto,
    z: str,
    output: str,
    prefix: str,
) -> list[onnx.NodeProto]:
    # Dense around the nonlinear region.  The tails are almost linear/zero, so
    # wide intervals there keep graph size bounded.
    xs = [-12.0, -10.0]
    xs.extend([i * 0.25 for i in range(-32, 33)])
    xs.extend([10.0, 12.0])
    ys = [math.log1p(math.exp(x)) for x in xs]
    slopes = [(ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]

    nodes: list[onnx.NodeProto] = []
    m0 = _make_scalar_const(model, f"{prefix}/Pwl", slopes[0])
    b0 = _make_scalar_const(model, f"{prefix}/Pwl", ys[0] - slopes[0] * xs[0])
    linear = f"{prefix}/PwlLinear_output_0"
    acc = f"{prefix}/PwlAcc0_output_0"
    nodes.append(helper.make_node("Mul", [z, m0], [linear], name=f"{prefix}/PwlMul0"))
    nodes.append(helper.make_node("Add", [linear, b0], [acc], name=f"{prefix}/PwlAddBias"))

    for i, knot in enumerate(xs[1:-1], start=1):
        delta = slopes[i] - slopes[i - 1]
        if abs(delta) < 1.0e-7:
            continue
        ck = _make_scalar_const(model, f"{prefix}/PwlKnot{i}", knot)
        cd = _make_scalar_const(model, f"{prefix}/PwlDelta{i}", delta)
        shifted = f"{prefix}/PwlShift{i}_output_0"
        relu = f"{prefix}/PwlRelu{i}_output_0"
        term = f"{prefix}/PwlTerm{i}_output_0"
        next_acc = output if i == len(xs) - 2 else f"{prefix}/PwlAcc{i}_output_0"
        nodes.append(helper.make_node("Sub", [z, ck], [shifted], name=f"{prefix}/PwlSub{i}"))
        nodes.append(helper.make_node("Relu", [shifted], [relu], name=f"{prefix}/PwlRelu{i}"))
        nodes.append(helper.make_node("Mul", [relu, cd], [term], name=f"{prefix}/PwlMul{i}"))
        nodes.append(helper.make_node("Add", [acc, term], [next_acc], name=f"{prefix}/PwlAdd{i}"))
        acc = next_acc

    if acc != output:
        czero = _make_scalar_const(model, f"{prefix}/Pwl", 0.0)
        nodes.append(helper.make_node("Add", [acc, czero], [output], name=f"{prefix}/PwlOutput"))
    return nodes


def _make_sigmoid_bump_softplus_nodes(
    model: onnx.ModelProto,
    z: str,
    output: str,
    prefix: str,
) -> list[onnx.NodeProto]:
    """Approximate softplus without Log/Exp/PWL chains.

    softplus(z) ~= relu(z)
                 + a1 * sigmoid(k1 * z) * (1 - sigmoid(k1 * z))
                 + a2 * sigmoid(k2 * z) * (1 - sigmoid(k2 * z))

    The constants are a least-squares fit over z in [-12, 12] chosen to keep
    the max absolute softplus error around 0.017 while avoiding PwlTerm nodes.
    """
    k1 = _make_scalar_const(model, f"{prefix}/SigBumpK1", 1.25)
    k2 = _make_scalar_const(model, f"{prefix}/SigBumpK2", 6.0)
    a1 = _make_scalar_const(model, f"{prefix}/SigBumpA1", 1.84093409)
    a2 = _make_scalar_const(model, f"{prefix}/SigBumpA2", 0.86202629)
    one = _make_scalar_const(model, f"{prefix}/SigBump", 1.0)

    relu_z = f"{prefix}/SigBumpReluZ_output_0"
    kz1 = f"{prefix}/SigBumpKz1_output_0"
    sig1 = f"{prefix}/SigBumpSigmoid1_output_0"
    inv1 = f"{prefix}/SigBumpInv1_output_0"
    bump1 = f"{prefix}/SigBumpMul1_output_0"
    term1 = f"{prefix}/SigBumpTerm1_output_0"
    kz2 = f"{prefix}/SigBumpKz2_output_0"
    sig2 = f"{prefix}/SigBumpSigmoid2_output_0"
    inv2 = f"{prefix}/SigBumpInv2_output_0"
    bump2 = f"{prefix}/SigBumpMul2_output_0"
    term2 = f"{prefix}/SigBumpTerm2_output_0"
    acc = f"{prefix}/SigBumpAcc_output_0"

    return [
        helper.make_node("Relu", [z], [relu_z], name=f"{prefix}/SigBumpReluZ"),
        helper.make_node("Mul", [z, k1], [kz1], name=f"{prefix}/SigBumpMulK1"),
        helper.make_node("Sigmoid", [kz1], [sig1], name=f"{prefix}/SigBumpSigmoid1"),
        helper.make_node("Sub", [one, sig1], [inv1], name=f"{prefix}/SigBumpInv1"),
        helper.make_node("Mul", [sig1, inv1], [bump1], name=f"{prefix}/SigBumpMul1"),
        helper.make_node("Mul", [bump1, a1], [term1], name=f"{prefix}/SigBumpTerm1"),
        helper.make_node("Mul", [z, k2], [kz2], name=f"{prefix}/SigBumpMulK2"),
        helper.make_node("Sigmoid", [kz2], [sig2], name=f"{prefix}/SigBumpSigmoid2"),
        helper.make_node("Sub", [one, sig2], [inv2], name=f"{prefix}/SigBumpInv2"),
        helper.make_node("Mul", [sig2, inv2], [bump2], name=f"{prefix}/SigBumpMul2"),
        helper.make_node("Mul", [bump2, a2], [term2], name=f"{prefix}/SigBumpTerm2"),
        helper.make_node("Add", [relu_z, term1], [acc], name=f"{prefix}/SigBumpAdd1"),
        helper.make_node("Add", [acc, term2], [output], name=f"{prefix}/SigBumpAdd2"),
    ]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path))
    count = rewrite_swoosh_nolog(
        model,
        args.approx,
        args.include_substring,
        args.hard_include_substring,
    )

    stable_patterns = _find_stable_softplus_patterns(model)
    if args.include_substring:
        stable_patterns = [
            pattern
            for pattern in stable_patterns
            if any(text in pattern["log"].name for text in args.include_substring)
        ]
    if stable_patterns:
        c0 = _ensure_const(model, "hard_softplus_const_0", 0.0)
        c3 = _ensure_const(model, "hard_softplus_const_3", 3.0)
        c6 = _ensure_const(model, "hard_softplus_const_6", 6.0)
        c12 = _ensure_const(model, "hard_softplus_const_12", 12.0)
        remove_names = {
            node.name
            for pattern in stable_patterns
            for node in (
                pattern["add"],
                pattern["exp"],
                pattern["neg"],
                pattern["abs"],
                pattern["log"],
                pattern["max"],
                pattern["final_add"],
            )
        }
        insert_by_log_name: dict[str, list[onnx.NodeProto]] = {}
        for index, pattern in enumerate(stable_patterns):
            log_node = pattern["log"]
            prefix = log_node.name.rsplit("/", 1)[0]
            hard_prefix = f"{prefix}/HardStableSoftplus_{index}"
            pattern_approx = args.approx
            if args.hard_include_substring and any(
                text in log_node.name for text in args.hard_include_substring
            ):
                pattern_approx = "hard"

            if pattern_approx == "hard":
                insert_by_log_name[log_node.name] = _make_hard_softplus_nodes(
                    z=str(pattern["z"]),
                    output=pattern["final_add"].output[0],
                    prefix=hard_prefix,
                    c0=c0,
                    c3=c3,
                    c6=c6,
                    c12=c12,
                )
            elif pattern_approx == "sigmoid-bump":
                insert_by_log_name[log_node.name] = _make_sigmoid_bump_softplus_nodes(
                    model=model,
                    z=str(pattern["z"]),
                    output=pattern["final_add"].output[0],
                    prefix=hard_prefix,
                )
            else:
                insert_by_log_name[log_node.name] = _make_pwl_softplus_nodes(
                    model=model,
                    z=str(pattern["z"]),
                    output=pattern["final_add"].output[0],
                    prefix=hard_prefix,
                )

        new_nodes: list[onnx.NodeProto] = []
        for node in model.graph.node:
            if node.name in insert_by_log_name:
                new_nodes.extend(insert_by_log_name[node.name])
            if node.name not in remove_names:
                new_nodes.append(node)
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    print(f"Rewrote {count} Swoosh softplus pattern(s) to {args.approx} softplus")
    print(f"Rewrote {len(stable_patterns)} stable softplus pattern(s) to {args.approx} softplus")
    print(output_path)


if __name__ == "__main__":
    main()
