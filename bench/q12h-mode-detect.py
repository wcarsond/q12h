"""q12h-mode.input.py - read a Q12H GGUF and say which MODE its blocks are in.

Why this exists: `q12h` vs `q12h-v2` is a build-time difference, not a flag, so a
file's name is only as trustworthy as whoever built it. My own 7B script ran the
same binary twice and labelled the second output "v2", which it cannot be. Rather
than reason about which build-v8 was current, read the bits.

The marker (ggml-common.h): each 412-byte block starts with lut[16], and
  lut[0] == 0xFF  -> shift-offset mode (v2)
  lut[0] != 0xFF  -> LUT mode (v1)
About 98.58% of blocks qualify for v2, so a v2 file should show ~98-99% markers
and a v1 file ~0%.

Finding block alignment without parsing the GGUF header: try all 412 phases over a
window in the middle of the file and report the best. The correct phase stands out
because 0xFF at a fixed stride is otherwise rare. A v1 file has NO strong phase -
that absence is the v1 signal, which is why the calibration run below matters.

CALIBRATION IS NOT OPTIONAL: the script is pointed at the known v1 and v2 1.7B
files first. If it cannot separate those two, its answer on anything else is void
and it says so instead of printing a number.

    python <path>
"""
import os

BLK = 412
WIN = 16 * 1024 * 1024          # 16 MiB window: ~40k blocks per phase, plenty of
                                # signal, and 412 phases over it stays a few seconds
GG = '<models>'

CAL = [('1.7B v1 (known)', GG + '/Qwen3-1.7B-q12h.gguf',    'v1'),
       ('1.7B v2 (known)', GG + '/Qwen3-1.7B-q12h-v2.gguf', 'v2')]
TEST = [('7B as-built',    GG + '/qwen25coder7b-q12h.gguf'),
        ('7B 2nd output',  GG + '/qwen25coder7b-q12h-v2.gguf')]


def best_phase(path):
    """Find block alignment, then report the lut[0]==0xFF fraction.

    Alignment anchor is NOT lut[0] - the first version of this probe assumed it
    was and read 100% on both known files. The bytes said why: epos[4] is set to
    FF FF FF FF when a block uses none of its four exception slots, which is the
    common case, so epos is a far stronger periodic 0xFF than lut[0] will ever be.
    That 4-byte run is a better anchor precisely because it is near-constant:
    find the run of 4 consecutive high-0xFF phases, and since the struct is
    lut[16] then epos[4], block start = run_start - 16 (mod BLK).
    """
    sz = os.path.getsize(path)
    start = sz // 2                      # middle of the file = tensor data
    n = min(WIN, sz - start)
    with open(path, 'rb') as f:
        f.seek(start)
        buf = f.read(n)

    fr = []
    for ph in range(BLK):
        c = t = 0
        for off in range(ph, len(buf), BLK):
            t += 1
            if buf[off] == 0xFF:
                c += 1
        fr.append(c / t if t else 0.0)

    # the epos sentinel: 4 consecutive phases all strongly 0xFF
    runs = [(min(fr[p], fr[(p + 1) % BLK], fr[(p + 2) % BLK], fr[(p + 3) % BLK]), p)
            for p in range(BLK)]
    strength, run_start = max(runs)
    blk0 = (run_start - 16) % BLK        # lut[0] sits 16 bytes before epos
    return fr[blk0], blk0, strength, run_start


def call(frac):
    if frac >= 0.90:
        return 'v2'
    if frac <= 0.10:
        return 'v1'
    return 'UNCLEAR'


print('%-18s %10s %9s %10s  %s'
      % ('file', 'lut[0]=FF', 'blk phase', 'epos conf', 'call'))
rows = []
for label, path, expect in CAL:
    if not os.path.exists(path):
        print('%-18s MISSING' % label)
        rows.append(None)
        continue
    fr, blk0, strength, _ = best_phase(path)
    c = call(fr)
    print('%-18s %9.2f%% %9d %9.2f%%  %s  (expected %s)%s'
          % (label, fr * 100, blk0, strength * 100, c, expect,
             '' if c == expect else '   <-- MISMATCH'))
    rows.append(c == expect)

if not all(r is True for r in rows):
    print()
    print('CALIBRATION FAILED - the probe cannot separate the two known files,')
    print('so it says nothing about the 7B. Fix the probe before trusting it.')
    raise SystemExit(1)

print()
print('calibration OK: the probe separates known v1 from known v2.')
print()
for label, path in TEST:
    if not os.path.exists(path):
        print('%-18s MISSING' % label)
        continue
    fr, blk0, strength, _ = best_phase(path)
    print('%-18s %9.2f%% %9d %9.2f%%  %s'
          % (label, fr * 100, blk0, strength * 100, call(fr)))
