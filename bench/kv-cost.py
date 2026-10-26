"""kv-1m.input.py - Carson: "do you thinnk we can pull 1M context?"

KV cache cost is not a vibe, it is multiplication, so this reads the real
architecture out of each GGUF we own and does the arithmetic:

    KV bytes = 2 (K and V) * n_layers_that_keep_KV * n_kv_heads * head_dim
               * context * bytes_per_element

The term that decides everything is n_layers_that_keep_KV. A DENSE model keeps
KV on every layer. A hybrid / linear-attention model (Qwen3-Next, the "Flash"
and "Next" lines) replaces most layers with Gated DeltaNet, which carries a
fixed-size recurrent state instead of a growing cache - presets.ini already
notes this for qwen36-35b-a3b: "30 Gated DeltaNet + 10 full-attention layers.
Only the 10 attention layers keep KV, so a long window is cheap."

So the honest answer differs by model family by an order of magnitude, and the
only way to be right is to read each file's metadata rather than reason about
"a 27B".

Budget to compare against: 2x Arc Pro B60 = 48.8 GB of real VRAM, minus the
weights, and the iGPU does NOT add capacity (measured 2026-10-25: its memory IS
system RAM, which is where overflow already goes - igpu-cascade.output.txt).

    python <path>
"""
import os
import sys

sys.path.insert(0, '<llama.cpp>/gguf-py')
from gguf import GGUFReader                                      # noqa: E402

VRAM = 48.8                      # GB, two B60s
CANDIDATES = [
    ('qwen3.8-flash-next',  '<path>', 84.0),
    ('qwen3-coder-next',    '<models>', 47.0),
    ('qwen3-next-80b',      '<models>', 40.0),
    ('qwen36-35b-a3b',      '<models>', 17.0),
    ('qwen3.6-27b (dense)', '<path>', 18.0),
    ('granite42-30b',       '<models>', 17.0),
]


def first_gguf(p):
    if os.path.isdir(p):
        for root, _, files in os.walk(p):
            g = sorted(f for f in files if f.endswith('.gguf') and 'mmproj' not in f)
            if g:
                return os.path.join(root, g[0])
        return None
    return p if os.path.exists(p) else None


def scalar(f):
    try:
        v = f.contents()
    except Exception:
        return None
    if isinstance(v, (list, tuple)):
        return v if len(v) > 1 else (v[0] if v else None)
    return v


def read_arch(path):
    r = GGUFReader(path)
    d = {}
    for f in r.fields.values():
        d[f.name] = scalar(f)
    return d


def pick(d, *suffixes):
    for s in suffixes:
        for k, v in d.items():
            if k.endswith(s) and not isinstance(v, (list, tuple)):
                return v
    return None


print('%-22s %7s %6s %7s %6s %9s   %s'
      % ('model', 'layers', 'kvH', 'headdim', 'kvLyr', 'native ctx', 'KV at 1M ctx (f16 / q8 / q4)'))
print('-' * 118)

for label, path, wgb in CANDIDATES:
    g = first_gguf(path)
    if not g:
        print('%-22s  not on this box' % label)
        continue
    try:
        d = read_arch(g)
    except Exception as e:
        print('%-22s  could not read: %s' % (label, e))
        continue

    layers = pick(d, 'block_count')
    kvh    = pick(d, 'attention.head_count_kv')
    klen   = pick(d, 'attention.key_length')
    nhead  = pick(d, 'attention.head_count')
    emb    = pick(d, 'embedding_length')
    ctx    = pick(d, 'context_length')

    # How many layers actually keep a growing cache. The first version of this
    # looked for a per-layer head_count_kv array, found none, and fell back to
    # "all of them" - which contradicted presets.ini and overstated the Next /
    # Flash family by 4x. arch-dump.input.py showed what the key really is.
    #
    # In order of directness:
    #   attention.compress_ratios  per-layer array, non-zero on the full-attention
    #                              layers ([0,0,0,4, 0,0,0,4, ...] = 12 of 48)
    #   full_attention_interval    every Nth layer is full attention
    #   per-layer head_count_kv    zero on linear layers, if a model has it
    # Anything else is dense and keeps KV everywhere.
    kv_layers, basis = layers, 'dense'
    comp = next((list(v) for k, v in d.items()
                 if k.endswith('attention.compress_ratios') and isinstance(v, (list, tuple))), None)
    interval = pick(d, 'full_attention_interval')
    kv_arr = next((list(v) for k, v in d.items()
                   if k.endswith('attention.head_count_kv') and isinstance(v, (list, tuple))), None)
    if comp:
        kv_layers, basis = sum(1 for x in comp if x), 'compress_ratios'
    elif isinstance(interval, int) and interval > 1:
        kv_layers, basis = layers // interval, 'full_attn_interval=%d' % interval
    elif kv_arr:
        kv_layers, basis = sum(1 for x in kv_arr if x), 'per-layer kv'
        kvh = max(kv_arr)

    if klen is None and nhead and emb:
        klen = emb // nhead

    if not all(isinstance(x, (int, float)) for x in (kv_layers, kvh, klen)):
        print('%-22s  metadata incomplete (layers=%s kvh=%s klen=%s)' % (label, layers, kvh, klen))
        continue

    per_tok = 2 * kv_layers * kvh * klen          # bytes per token at 1 byte/elem
    f16 = per_tok * 2 * 1_000_000 / 1024 ** 3
    q8  = per_tok * 1 * 1_000_000 / 1024 ** 3
    q4  = per_tok * 0.5 * 1_000_000 / 1024 ** 3
    print('%-22s %7s %6s %7s %6d %9s   %6.1f / %6.1f / %6.1f GB   [%s]'
          % (label, layers, kvh, klen, kv_layers, ctx, f16, q8, q4, basis))

    room = VRAM - wgb
    verdict = []
    for name, need in (('f16', f16), ('q8', q8), ('q4', q4)):
        verdict.append('%s %s' % (name, 'FITS' if need <= room else 'no'))
    print('%-22s   weights %.0f GB -> %.1f GB left of %.1f:  %s'
          % ('', wgb, room, VRAM, ',  '.join(verdict)))
