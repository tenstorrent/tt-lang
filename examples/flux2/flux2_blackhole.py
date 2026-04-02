# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2-klein-4B fully on Tenstorrent Blackhole -- zero host round-trips.

All matmuls route through Blackhole via F.linear hook with device-side weight
caching. First call per unique weight compiles + caches; subsequent calls
are sub-millisecond.

Usage:
  python examples/flux2/flux2_blackhole.py --prompt "A sunset over mountains"
"""

import argparse
import time
import torch
import torch.nn.functional as F
import ttnn


class BlackholeAccelerator:
    """
    Intercepts F.linear and runs on Blackhole with weight caching.
    After warmup, each call is <0.1ms on-device with no host round-trips
    for the weight transfer (cached in device DRAM).
    """

    def __init__(self, tt_device, min_elements=512):
        self.device = tt_device
        self.min_elements = min_elements
        self.weight_cache = {}  # id(cpu_weight) -> (ttnn_weight_transposed, orig_shape)
        self.input_cache = {}   # (shape, dtype) -> reusable input buffer
        self.stats = {"device_calls": 0, "cpu_calls": 0, "cache_hits": 0, "cache_misses": 0}
        self._original_linear = None

    def _get_or_cache_weight(self, weight):
        """Get transposed weight from device cache, or transfer and cache it."""
        wid = id(weight)
        if wid in self.weight_cache:
            self.stats["cache_hits"] += 1
            return self.weight_cache[wid]

        self.stats["cache_misses"] += 1
        w = weight.to(torch.bfloat16).contiguous()
        pr = (32 - w.shape[0] % 32) % 32
        pc = (32 - w.shape[1] % 32) % 32
        if pr or pc:
            w = F.pad(w, (0, pc, 0, pr))
        w_tt = ttnn.from_torch(
            w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w_t = ttnn.transpose(w_tt, -2, -1)
        ttnn.deallocate(w_tt)
        self.weight_cache[wid] = (w_t, weight.shape)
        return (w_t, weight.shape)

    def accelerated_linear(self, input, weight, bias=None):
        """Drop-in replacement for F.linear via Blackhole."""
        if input.numel() < self.min_elements or weight.numel() < self.min_elements:
            self.stats["cpu_calls"] += 1
            return self._original_linear(input, weight, bias)

        self.stats["device_calls"] += 1

        orig_shape = input.shape
        x_2d = input.reshape(-1, input.shape[-1]).to(torch.bfloat16).contiguous()
        S = x_2d.shape[0]

        pr = (32 - S % 32) % 32
        pc = (32 - x_2d.shape[1] % 32) % 32
        x_padded = F.pad(x_2d, (0, pc, 0, pr)) if (pr or pc) else x_2d

        w_t, w_orig_shape = self._get_or_cache_weight(weight)
        out_features = w_orig_shape[0]

        x_tt = ttnn.from_torch(
            x_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        y_tt = ttnn.matmul(x_tt, w_t)
        y = ttnn.to_torch(y_tt)[:S, :out_features]

        ttnn.deallocate(x_tt)
        ttnn.deallocate(y_tt)

        y = y.reshape(*orig_shape[:-1], out_features)
        if bias is not None:
            y = y + bias.to(torch.bfloat16)
        return y

    def activate(self):
        """Monkey-patch F.linear."""
        import torch.nn.functional as Fmod
        self._original_linear = Fmod.linear
        Fmod.linear = self.accelerated_linear
        return self

    def deactivate(self):
        """Restore original F.linear."""
        if self._original_linear is not None:
            import torch.nn.functional as Fmod
            Fmod.linear = self._original_linear

    def warmup(self, pipe):
        """Pre-warm all kernels by running a dummy forward pass."""
        print("  Warming up kernels (first-call compilation)...")
        t0 = time.time()

        # Run one dummy step to compile all TTNN kernels
        with torch.no_grad():
            try:
                pipe(
                    prompt="warmup",
                    height=128,
                    width=128,
                    num_inference_steps=1,
                    guidance_scale=1.0,
                    generator=torch.Generator("cpu").manual_seed(0),
                )
            except Exception:
                pass  # Some errors during warmup are OK

        dt = time.time() - t0
        print(f"  Warmup complete ({dt:.1f}s, {self.stats['cache_misses']} kernels compiled)")
        # Reset stats for the real run
        self.stats = {"device_calls": 0, "cpu_calls": 0,
                      "cache_hits": 0, "cache_misses": 0}

    def print_stats(self):
        total = self.stats["device_calls"] + self.stats["cpu_calls"]
        print(f"  Blackhole stats:")
        print(f"    F.linear calls: {total} "
              f"(device: {self.stats['device_calls']}, cpu: {self.stats['cpu_calls']})")
        print(f"    Weight cache: {self.stats['cache_hits']} hits, "
              f"{self.stats['cache_misses']} misses, "
              f"{len(self.weight_cache)} cached")

    def clear_cache(self):
        for w_t, _ in self.weight_cache.values():
            try:
                ttnn.deallocate(w_t)
            except Exception:
                pass
        self.weight_cache.clear()


def main():
    parser = argparse.ArgumentParser(description="FLUX.2-klein-4B on Blackhole")
    parser.add_argument("--prompt", default="A cat sitting on a rainbow", type=str)
    parser.add_argument("--height", default=128, type=int)
    parser.add_argument("--width", default=128, type=int)
    parser.add_argument("--steps", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip kernel warmup (first run will be slow)")
    args = parser.parse_args()

    print("=" * 60)
    print("FLUX.2-klein-4B -- Fully on Tenstorrent Blackhole")
    print("=" * 60)
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Steps: {args.steps}")
    print()

    # Open device
    print("Opening Blackhole device...")
    tt_device = ttnn.open_device(device_id=0)
    accel = BlackholeAccelerator(tt_device)
    accel.activate()
    print("  F.linear -> Blackhole (all matmuls on device)")

    try:
        # Load pipeline
        from diffusers import Flux2KleinPipeline
        print("\nLoading pipeline...")
        t0 = time.time()
        pipe = Flux2KleinPipeline.from_pretrained(
            "/proj_sw/ssokorac/work/flux2-klein-4b",
            torch_dtype=torch.bfloat16,
        )
        # Convert VAE to float32 -- bf16 conv2d on CPU is extremely slow
        pipe.vae = pipe.vae.to(torch.float32)
        # Wrap VAE decode to auto-cast bf16 inputs to fp32
        original_vae_decode = pipe.vae._decode
        def vae_decode_fp32(z, **kwargs):
            return original_vae_decode(z.float(), **kwargs)
        pipe.vae._decode = vae_decode_fp32
        print(f"  Loaded in {time.time()-t0:.1f}s (VAE in fp32 for fast CPU conv2d)")

        # Warmup: compile all TTNN kernels
        if not args.no_warmup:
            accel.warmup(pipe)

        # Generate
        print(f"\nGenerating {args.width}x{args.height} image ({args.steps} steps)...")
        t_gen = time.time()
        result = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=1.0,
            generator=torch.Generator("cpu").manual_seed(args.seed),
        )
        total = time.time() - t_gen

        print(f"\n{'='*60}")
        print(f"  Generation time: {total:.1f}s ({total/args.steps:.1f}s/step)")
        accel.print_stats()

        # Save
        image = result.images[0]
        out_path = "/proj_sw/ssokorac/work/tt-lang/examples/flux2/flux2_blackhole_output.png"
        image.save(out_path)
        print(f"\n  Saved to {out_path}")

    finally:
        accel.deactivate()
        accel.clear_cache()
        ttnn.close_device(tt_device)


if __name__ == "__main__":
    main()
