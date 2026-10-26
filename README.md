# Q12H — lossless storage for bf16 weights, and a per-device Vulkan dispatch fix

Two separate pieces of work on top of [llama.cpp](https://github.com/ggml-org/llama.cpp)
(upstream `6050b467c`). They are independent; the second is useful on its own.

1. **`GGML_TYPE_Q12H`** — a *lossless* GGUF type for bf16 weights. Every weight
   decodes to the exact bits it was stored from. 0.805× the size of bf16.
2. **A per-device `mul_mat_vec_max_cols`** — upstream fixes this at 8 for every
   device. On an Intel Arc B60 that costs **3.7× token-generation throughput**
   above 8 concurrent sequences. The correct value is *inverted* between vendors.

---

## 1. Q12H

### The format

bf16 is 1 sign bit, 8 exponent bits, 7 mantissa bits. Within a 256-weight block
the exponent takes few distinct values, so it is the compressible part:

| field | size | bits/weight |
|---|---|---|
| `sm[256]` — sign + mantissa, one byte each | 256 B | **8.000** |
| `nib[128]` — 4-bit index into the block's exponent table | 128 B | **4.000** |
| | | **12.000 payload** |
| `lut[16]` — the block's 16 distinct exponents | 16 B | |
| `epos[4]`, `eval[4]` — four exception slots | 12 B | 0.875 overhead |
| **block total** | **412 B / 256 weights** | **12.875 all-in** |

The payload is exactly 12 of bf16's 16 bits — each field a whole number of bits,
no packing waste. All-in is 12.875 bits/weight = **0.8047× bf16**; a real file
measured **0.8050×**, which is the same number to three decimals, so nothing is
lost to framing. (Named by payload, as `Q8_0` is 8.5 bits all-in and `Q4_0` 4.5.)

Blocks needing more than 16 distinct exponents use the four exception slots.
A second mode (`lut[0] == 0xFF`) drops the table entirely and derives the
exponent as `base + nibble`; 98.58% of blocks qualify. That marker is free
because `0xFF` is an inf/nan exponent a real table can never hold, so v1 files
decode correctly under a v2 build.

### Is it actually lossless?

Perplexity, wikitext, 16 chunks, matched settings, two GPU architectures plus a
CPU reference:

| | bf16 | Q12H | Q12H-v2 |
|---|---|---|---|
| Arc Pro B60 | 9.4444 | 9.4445 | 9.4445 |
| RTX 3070 | 9.4447 | **9.4447** | **9.4447** |
| CPU (`GGML_DISABLE_VULKAN=1`) | 9.4447 | 9.4447 | 9.4447 |

On Ampere the Q12H result equals bf16 **and** the CPU reference to four decimals.
A greedy 1024-token generation at `--temp 0` matched the CPU reference token for
token. Where it diverged from *Vulkan* bf16, a tie-check showed Vulkan bf16 was
the arm that had drifted from CPU — not Q12H.

For scale: the two B60s in one machine differ from **each other** by 2.4e-4 on
*identical* bf16 weights. That is larger than Q12H's deviation from bf16.

### Speed — and the part to read before quoting a number

| | Arc Pro B60 | RTX 3070 |
|---|---|---|
| prefill vs bf16 | **2.02×** | **0.80×** |
| decode vs bf16 | 1.04× | 1.16× |
| file size | 0.805× | 0.805× |

**Prefill is 2× faster than bf16 on Arc and ~20% slower on Ampere.** The Arc
result is not Q12H being fast; it is Arc's bf16 GEMM being slow. Anyone quoting
"2× faster than bf16" without naming the card is quoting an artifact.

### A bug worth knowing about if you implement this

Q12H's dequant path presents `src0_type = F16`, so
`ggml_vk_get_mul_mat_mat_f16acc()` returns true and the GEMM accumulates
thousands of products per output in **half precision**. bf16 escapes this only
because that function returns false for `BF16` *by name*. Measured cost: 1.7e-3
relative perplexity error on Arc, 1.9e-3 on Ampere. Forcing `GGML_PREC_F32`
removes it, at ~3% of prefill on Arc and ~22% on Ampere.

f32 accumulation is **mandatory wherever the dequant target is f16** — it is not
a tuning knob. The way to recover Ampere's 22% is to dequantize to **bf16** on
devices that support it (then the existing BF16 rule gives f32 accumulation for
free), not to drop the precision. That change is not in this tree: llama.cpp has
only one dequant pipeline table and every entry outputs f16.

### What it is and is not for

Q12H competes with **bf16**, not with Q4_K_M. It is for keeping a model
bit-exact while making it smaller, which matters when you are measuring
quantization error and need a baseline that contributes none of its own. It will
not make a large model fit on a small card — for that you want a *lossy* type:

| | bits/weight | size |
|---|---|---|
| bf16 | 16 | 1.000× |
| **Q12H** | 12.875 | 0.805× |
| Q8_0 | 8.5 | 0.532× |
| Q4_K_M | ~4.9 | 0.315× |

---

## 2. Per-device `mul_mat_vec_max_cols`

`mul_mat_vec_max_cols` decides where `mul_mat` stops using the gemv path and
switches to the tiled GEMM. Upstream it is one compile-time `8` for every
device. With N sequences decoding concurrently that count **is** N, so the
constant caps token-generation throughput as soon as more than 8 sequences are
in flight — the common case for a server holding several sessions.

### Where the 8 came from

It was never a tuned value. `git log -S` puts its introduction in
**#10991, 2024-12-30, "vulkan: optimize mul_mat for small values of N"** — the
commit that first made the mul_mat_vec shaders support N>1 at all, with
`NUM_COLS` as a spec constant. The 8 is how far that optimization was taken.
It has not been revisited since, including when hardware with cooperative-matrix
units arrived. A feature's upper bound became a tuning constant by sitting still.

Move a batch across the switch with a single sequence:

| device | `-ub 8` | `-ub 9` | `-ub 12` | `-ub 16` |
|---|---|---|---|---|
| Arc Pro B60 (KHR_coopmat) | **2151 t/s** | **579 t/s** | | |
| RTX 3070 (NV_coopmat2) | 235 t/s | 251 | 325 | **429** |

On Arc the GEMM path is **3.7× slower** than the gemv path it replaces. On the
NVIDIA part there is no cliff at all — crossing into the GEMM path *gains*
throughput, and forcing the limit to 16 instead regresses the 9–16 range hard
(0.42× for Q4_K, 0.34× for bf16). Sweeping on Arc puts the optimum at 32; 64 is
worse again. The same Arc silicon under SYCL shows no cliff, so this is a
dispatch heuristic, not a hardware limit.

The patch keeps `mul_mat_vec_max_cols` as the pipeline-array bound (raised to 32)
and adds `vk_device_struct::mmv_max_cols` for the value a device dispatches with.
**Pipelines are created only up to the per-device value**, so a device left at 8
compiles exactly the shaders it compiled before — raising the bound costs
pointers, not shader compilation. Only Intel parts with cooperative-matrix
support are raised, which is the configuration the 32 was measured on; every
other device keeps today's behaviour rather than inheriting an unmeasured number.

### What the constant actually is on each vendor

The same `8` is doing two completely different jobs:

- **On Arc it is an artificial ceiling.** The gemv path keeps scaling to 32
  columns; nothing about the hardware objects to 9. The same silicon under SYCL
  shows no cliff at all, which is what makes it a software boundary rather than
  a physical one.
- **On NVIDIA it is a correctly placed handoff.** Past 8, the GEMM path really
  is faster there, and forcing gemv further *hurts* — 0.42× at 16.

So there is no single value that can be right, and that is the argument: the fix
is not a bigger number (a bigger number is exactly what breaks NVIDIA), it is
that the number belongs to the device.

It also fails silently. Nothing crashes and nothing warns — throughput simply
drops by roughly 4× the moment a ninth concurrent request arrives, and everything
still returns correct results. That is how it survived two years.

### What this patch does not fix

A static per-device table can only encode hardware someone happened to own. This
one raises Intel parts with cooperative-matrix support because that is what was
measured; every other device keeps 8 forever, including devices that would do
better at 16 or 24 and whose owners will never find out. That is the same flaw as
the original constant, with two entries instead of one.

The principled fix is a startup micro-benchmark with a cached result — time a
gemv at {8, 16, 32} once per (device, driver, type) and reuse the winner, the way
FFTW keeps wisdom files. Roughly three measurements and tens of milliseconds, and
it adapts to hardware no contributor owns. That is a much larger change with real
costs to reproducibility and CI determinism, so it is noted here as the direction
rather than proposed as part of this patch.

---

## Layout

```
patches/   the series, applies to llama.cpp 6050b467c
bench/     the measurement scripts
results/   their raw output
```

## Hardware

2× Intel Arc Pro B60 (24 GB, Vulkan, KHR_coopmat) and an NVIDIA RTX 3070
(8 GB, NV_coopmat2, driver 581.95), Windows 11. CPU references via
`GGML_DISABLE_VULKAN=1`.

## Caveats

- Perplexity at 16 chunks carries roughly ±0.48 confidence; the claim is that
  Q12H matches bf16 and the CPU **to four decimals on the same run**, not that
  perplexity is known to that precision.
- The 2× prefill on Arc is Arc-specific. See the speed table.
- Q12H is implemented for Vulkan and CPU. Other backends fall back to CPU.
- The exception-slot path is exercised by Qwen2.5-Coder-7B (whose worst block
  needs all four) and by Qwen3-1.7B. It is not proven across every architecture.
