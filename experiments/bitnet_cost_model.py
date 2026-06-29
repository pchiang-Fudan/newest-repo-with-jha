#!/usr/bin/env python3
"""Rough BitNet b1.58 short-context hardware feasibility model.

This is intentionally simple: it separates model/KV facts from hardware
assumptions so we can replace guesses with measurements as the project matures.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class BitNetShape:
    name: str = "BitNet-b1.58-2B-4T"
    params_billion: float = 2.74
    weight_bits: float = 1.58
    layers: int = 30
    hidden_size: int = 2560
    kv_heads: int = 5
    head_dim: int = 128

    @property
    def raw_weight_bytes(self) -> float:
        return self.params_billion * 1e9 * self.weight_bits / 8.0

    def kv_bytes_per_token(self, kv_bits: float) -> float:
        return 2 * self.layers * self.kv_heads * self.head_dim * kv_bits / 8.0


@dataclass(frozen=True)
class HardwareAssumptions:
    dram_bandwidth_gbps: float = 1024.0
    target_tokens_per_second: float = 100_000.0
    watts: float = 250.0
    chip_cost_usd: float = 1000.0
    amortized_tokens: float = 1e15

    @property
    def bytes_per_second(self) -> float:
        return self.dram_bandwidth_gbps * 1e9 / 8.0

    @property
    def energy_j_per_token(self) -> float:
        return self.watts / self.target_tokens_per_second

    @property
    def silicon_usd_per_token(self) -> float:
        return self.chip_cost_usd / self.amortized_tokens


def fmt_bytes(value: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:,.2f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--generated", type=int, default=128)
    parser.add_argument("--kv-bits", type=float, default=8.0)
    parser.add_argument("--bandwidth-gbps", type=float, default=1024.0)
    parser.add_argument("--target-tps", type=float, default=100_000.0)
    parser.add_argument("--watts", type=float, default=250.0)
    args = parser.parse_args()

    model = BitNetShape()
    hw = HardwareAssumptions(
        dram_bandwidth_gbps=args.bandwidth_gbps,
        target_tokens_per_second=args.target_tps,
        watts=args.watts,
    )

    kv_per_token = model.kv_bytes_per_token(args.kv_bits)
    kv_per_user = kv_per_token * args.context
    total_active_kv = kv_per_user * args.users
    decode_kv_read_per_user_token = kv_per_user
    decode_kv_read_all_users_step = decode_kv_read_per_user_token * args.users
    max_user_tokens_per_second_from_kv = hw.bytes_per_second / max(kv_per_user, 1.0)
    output_tokens = args.users * args.generated

    print(f"model: {model.name}")
    print(f"raw ternary weight payload: {fmt_bytes(model.raw_weight_bytes)}")
    print(f"kv precision: {args.kv_bits:g} bits")
    print(f"kv per token: {fmt_bytes(kv_per_token)}")
    print(f"kv per user at {args.context} tokens: {fmt_bytes(kv_per_user)}")
    print(f"active kv for {args.users:,} users: {fmt_bytes(total_active_kv)}")
    print(
        "kv read per decode step across all users: "
        f"{fmt_bytes(decode_kv_read_all_users_step)}"
    )
    print(
        "kv-bandwidth-limited generated tokens/sec estimate: "
        f"{max_user_tokens_per_second_from_kv:,.0f}"
    )
    print(f"energy at target throughput: {hw.energy_j_per_token * 1000:.3f} mJ/token")
    print(
        "energy for requested generated tokens: "
        f"{hw.energy_j_per_token * output_tokens:,.2f} J"
    )


if __name__ == "__main__":
    main()
