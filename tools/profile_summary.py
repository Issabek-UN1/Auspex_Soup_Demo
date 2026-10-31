"""Summarise runs/solution_profile/profile.json (written by `harness/run_task.py --profile`).

    python tools/profile_summary.py runs/solution_profile/profile.json [--top 25]

Prints GPU time per training step grouped by kernel family, then the top kernels, so it is
clear where the step time goes (GEMMs, NF4 dequantization, attention, our Triton kernels...).
"""

import argparse
import json

FAMILIES = (
    ("our triton kernels", ("_chunked_cross_entropy", "_custom_swiglu", "_fused_rope", "_rms_norm")),
    ("nf4 dequant (bitsandbytes)", ("dequant", "kdequant", "bnb", "blockwise")),
    ("attention (sdpa)", ("flash", "efficient_attention", "fmha", "sdpa", "mem_eff", "attention")),
    ("gemm (cutlass/cublas)", ("gemm", "cutlass", "cublas", "sgemm", "hgemm", "turing_h", "ampere_h", "volta_h")),
    ("optimizer", ("adam", "multi_tensor", "foreach")),
    ("copy / cast / elementwise", ("copy", "cast", "elementwise", "vectorized", "fill", "mul", "add", "neg")),
    ("reduce / softmax", ("reduce", "softmax", "norm")),
)


def family(name):
    low = name.lower()
    for label, needles in FAMILIES:
        if any(n.lower() in low for n in needles):
            return label
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("profile")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    data = json.load(open(args.profile, encoding="utf-8"))
    steps = data["active_steps"]
    total = data["total_gpu_us"]
    print(f"GPU time per step: {total / steps / 1000:.1f} ms  ({steps} profiled steps)\n")

    groups = {}
    for row in data["kernels"]:
        groups[family(row["name"])] = groups.get(family(row["name"]), 0.0) + row["time_us"]
    for label, us in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"{label:32s} {us / steps / 1000:9.1f} ms/step  {100 * us / total:5.1f}%")

    print(f"\nTop {args.top} kernels:")
    for row in data["kernels"][: args.top]:
        print(f"{row['time_us'] / steps / 1000:9.2f} ms/step  {100 * row['time_us'] / total:5.1f}%  "
              f"x{row['count'] // steps:<5d} {row['name'][:110]}")


if __name__ == "__main__":
    main()
