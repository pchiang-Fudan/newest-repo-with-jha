#!/usr/bin/env python3
"""First-order PPA simulator for a BitNet-style W2A8 inference ASIC.

This is an architectural model, not a signoff tool. It turns BitNet model
dimensions and 12nm-calibrated primitive assumptions into per-token estimates.
Replace the placeholder hardware assumptions with synthesized tile/SRAM/NoC
numbers from your PDK flow.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class BitNet2B:
    layers: int = 30
    dim: int = 2560
    ffn_dim: int = 6912
    heads: int = 20
    kv_heads: int = 5
    head_dim: int = 128
    vocab: int = 128256
    params: float = 2.74e9
    weight_bits: float = 2.0

    @property
    def qkv_out(self) -> int:
        return (self.heads + 2 * self.kv_heads) * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim


@dataclass(frozen=True)
class PrimitivePPA:
    # Replace these with 12nm synthesized values.
    freq_ghz: float = 1.0
    tiles: int = 512
    macs_per_tile_per_cycle: int = 256
    tile_area_mm2: float = 0.025
    tile_power_mw_at_util: float = 12.0
    sram_area_mm2_per_mib: float = 0.65
    sram_pj_per_byte: float = 1.2
    dram_pj_per_byte: float = 20.0
    noc_pj_per_byte: float = 2.0
    control_area_mm2: float = 20.0
    io_area_mm2: float = 30.0
    leakage_w: float = 15.0
    target_utilization: float = 0.65
    gpu_decode_tps: float = 120.0
    gpu_prefill_tps: float = 3500.0
    gpu_watts: float = 700.0

    @property
    def peak_ops_per_s(self) -> float:
        return (
            self.freq_ghz
            * 1e9
            * self.tiles
            * self.macs_per_tile_per_cycle
            * self.target_utilization
        )

    @property
    def tile_area_total_mm2(self) -> float:
        return self.tiles * self.tile_area_mm2

    @property
    def tile_power_total_w(self) -> float:
        return self.tiles * self.tile_power_mw_at_util / 1000.0


def fmt_bytes(value: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:,.2f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def linear_ops(in_dim: int, out_dim: int) -> int:
    return in_dim * out_dim


def layer_w2a8_ops(model: BitNet2B) -> int:
    # Decode path, one token. Linear layers dominate:
    # attention qkv, attention output, FFN w13, FFN w2.
    return (
        linear_ops(model.dim, model.qkv_out)
        + linear_ops(model.dim, model.dim)
        + linear_ops(model.dim, 2 * model.ffn_dim)
        + linear_ops(model.ffn_dim, model.dim)
    )


def kv_bytes_per_token(model: BitNet2B, kv_bits: float) -> float:
    return 2 * model.layers * model.kv_heads * model.head_dim * kv_bits / 8.0


def simulate(args: argparse.Namespace) -> None:
    model = BitNet2B()
    ppa = PrimitivePPA(
        freq_ghz=args.freq_ghz,
        tiles=args.tiles,
        macs_per_tile_per_cycle=args.macs_per_tile,
        tile_area_mm2=args.tile_area,
        tile_power_mw_at_util=args.tile_power_mw,
        sram_area_mm2_per_mib=args.sram_area_per_mib,
        sram_pj_per_byte=args.sram_pj_per_byte,
        dram_pj_per_byte=args.dram_pj_per_byte,
        noc_pj_per_byte=args.noc_pj_per_byte,
        control_area_mm2=args.control_area,
        io_area_mm2=args.io_area,
        leakage_w=args.leakage_w,
        target_utilization=args.util,
        gpu_decode_tps=args.gpu_decode_tps,
        gpu_prefill_tps=args.gpu_prefill_tps,
        gpu_watts=args.gpu_watts,
    )

    ops_per_layer = layer_w2a8_ops(model)
    ops_per_token = ops_per_layer * model.layers
    compute_s = ops_per_token / ppa.peak_ops_per_s
    compute_tps = 1.0 / compute_s

    # Prefill can process prompt tokens in parallel at the kernel level, but a
    # first-order bound is still total prompt-token work divided by throughput.
    # Replace with a dataflow/cycle model once tile scheduling is defined.
    asic_prefill_tps = compute_tps * args.prefill_parallelism
    asic_ttft_s = args.context / asic_prefill_tps + compute_s
    gpu_ttft_s = args.context / ppa.gpu_prefill_tps + 1.0 / ppa.gpu_decode_tps

    kv_token_bytes = kv_bytes_per_token(model, args.kv_bits)
    effective_context = args.context / args.kv_seq_compression
    kv_user_bytes = kv_token_bytes * effective_context
    kv_fleet_bytes = kv_user_bytes * args.users
    kv_read_per_generated_token = kv_user_bytes
    kv_write_per_generated_token = kv_token_bytes

    # This is compressed/sparse attention decode: each generated token reads the
    # retained/compressed KV entries rather than the entire raw context.
    dram_bytes_per_token = kv_read_per_generated_token + kv_write_per_generated_token
    dram_energy_j = dram_bytes_per_token * ppa.dram_pj_per_byte * 1e-12

    # Coarse activation traffic estimate: each layer moves hidden vectors and
    # intermediate vectors through SRAM/NoC. Replace with trace-derived counts.
    act_bytes_per_layer = (
        6 * model.dim * args.act_bits / 8.0
        + 3 * model.ffn_dim * args.act_bits / 8.0
    )
    sram_bytes_per_token = act_bytes_per_layer * model.layers
    sram_energy_j = sram_bytes_per_token * ppa.sram_pj_per_byte * 1e-12
    noc_energy_j = sram_bytes_per_token * ppa.noc_pj_per_byte * 1e-12

    dynamic_energy_j = dram_energy_j + sram_energy_j + noc_energy_j
    dynamic_power_w_at_compute_limit = dynamic_energy_j * compute_tps
    total_power_w = (
        ppa.tile_power_total_w
        + dynamic_power_w_at_compute_limit
        + ppa.leakage_w
    )
    asic_energy_per_decode_token_j = total_power_w / compute_tps
    asic_energy_ttft_j = total_power_w * asic_ttft_s
    gpu_energy_per_decode_token_j = ppa.gpu_watts / ppa.gpu_decode_tps
    gpu_energy_ttft_j = ppa.gpu_watts * gpu_ttft_s

    sram_area = args.onchip_sram_mib * ppa.sram_area_mm2_per_mib
    total_area = ppa.tile_area_total_mm2 + sram_area + ppa.control_area_mm2 + ppa.io_area_mm2

    print("BitNet W2A8 ASIC PPA sketch")
    print(f"layers: {model.layers}, dim: {model.dim}, ffn: {model.ffn_dim}")
    print(
        f"context: {args.context}, effective KV context: {effective_context:,.1f}, "
        f"users: {args.users}, kv: {args.kv_bits:g}b"
    )
    print()
    print("Compute")
    print(f"w2a8 ops/token: {ops_per_token:,.0f}")
    print(f"peak effective ops/s: {ppa.peak_ops_per_s:,.0f}")
    print(f"compute-limited t/s: {compute_tps:,.0f}")
    print(f"prefill parallelism factor: {args.prefill_parallelism:g}x")
    print(f"estimated ASIC TTFT: {asic_ttft_s * 1e3:,.2f} ms")
    print(f"estimated GPU TTFT: {gpu_ttft_s * 1e3:,.2f} ms")
    print()
    print("KV memory")
    print(f"kv/token: {fmt_bytes(kv_token_bytes)}")
    print(f"kv/user: {fmt_bytes(kv_user_bytes)}")
    print(f"sequence compression: {args.kv_seq_compression:g}x")
    print(f"active fleet KV: {fmt_bytes(kv_fleet_bytes)}")
    print(f"dram bytes/generated token: {fmt_bytes(dram_bytes_per_token)}")
    print()
    print("Energy")
    print(f"dram energy/token: {dram_energy_j * 1e3:.3f} mJ")
    print(f"sram energy/token: {sram_energy_j * 1e3:.3f} mJ")
    print(f"noc energy/token: {noc_energy_j * 1e3:.3f} mJ")
    print(f"dynamic energy/token: {dynamic_energy_j * 1e3:.3f} mJ")
    print(f"estimated total power: {total_power_w:,.2f} W")
    print(f"ASIC decode energy/token: {asic_energy_per_decode_token_j * 1e3:.3f} mJ")
    print(f"GPU decode energy/token: {gpu_energy_per_decode_token_j * 1e3:.3f} mJ")
    print(f"ASIC TTFT energy: {asic_energy_ttft_j:.3f} J")
    print(f"GPU TTFT energy: {gpu_energy_ttft_j:.3f} J")
    print(f"tokens/s/W: {compute_tps / max(total_power_w, 1e-9):,.1f}")
    print()
    print("Area")
    print(f"tiles: {ppa.tile_area_total_mm2:.2f} mm^2")
    print(f"on-chip SRAM: {sram_area:.2f} mm^2")
    print(f"control+IO: {ppa.control_area_mm2 + ppa.io_area_mm2:.2f} mm^2")
    print(f"total: {total_area:.2f} mm^2")
    print(f"tokens/s/mm^2: {compute_tps / max(total_area, 1e-9):,.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--kv-bits", type=float, default=16.0)
    parser.add_argument("--kv-seq-compression", type=float, default=1.0)
    parser.add_argument("--act-bits", type=float, default=8.0)
    parser.add_argument("--freq-ghz", type=float, default=1.0)
    parser.add_argument("--tiles", type=int, default=512)
    parser.add_argument("--macs-per-tile", type=int, default=256)
    parser.add_argument("--tile-area", type=float, default=0.025)
    parser.add_argument("--tile-power-mw", type=float, default=12.0)
    parser.add_argument("--onchip-sram-mib", type=float, default=64.0)
    parser.add_argument("--sram-area-per-mib", type=float, default=0.65)
    parser.add_argument("--sram-pj-per-byte", type=float, default=1.2)
    parser.add_argument("--dram-pj-per-byte", type=float, default=20.0)
    parser.add_argument("--noc-pj-per-byte", type=float, default=2.0)
    parser.add_argument("--control-area", type=float, default=20.0)
    parser.add_argument("--io-area", type=float, default=30.0)
    parser.add_argument("--leakage-w", type=float, default=15.0)
    parser.add_argument("--util", type=float, default=0.65)
    parser.add_argument("--prefill-parallelism", type=float, default=8.0)
    parser.add_argument("--gpu-decode-tps", type=float, default=120.0)
    parser.add_argument("--gpu-prefill-tps", type=float, default=3500.0)
    parser.add_argument("--gpu-watts", type=float, default=700.0)
    simulate(parser.parse_args())


if __name__ == "__main__":
    main()
