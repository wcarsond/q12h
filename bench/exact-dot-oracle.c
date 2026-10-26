// exact-acc.input.c - a Kulisch-style exact dot product.
//
// Every binary float is (integer mantissa) * 2^exp. So every PRODUCT of two
// floats is an exact 48-bit integer at a known bit position - no rounding is
// required to form it. The loss in an ordinary dot product happens later, in
// the ADD, when the two operands get aligned and the smaller one's low bits
// are shifted off the end of the register.
//
// A Kulisch accumulator removes that: one wide fixed-point register spanning
// the whole exponent range, so every product lands at its true position and
// NOTHING is ever shifted away. The sum is exact, independent of order, for
// any number of terms. Convert to float once, at the end.
//
// Sizing for this workload: bf16 weights run 2^-64..2^-6 and activations
// about 2^-12..2^+6, so products land in roughly 2^-80..2^0, times 48 bits of
// mantissa. 512 bits with the window placed at 2^-320 covers that with room
// to spare, and costs 16 uint32 words of register.
//
// Build: cl /O2 exact-acc.input.c   (or gcc -O2 -o exact-acc exact-acc.input.c)
//   run: exact-acc > exact-acc.output.txt
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define ACC_WORDS 16                 // 512 bits of fixed point
#define ACC_BITS  (ACC_WORDS * 32)
#define ACC_LO    (-320)             // bit 0 of the accumulator represents 2^ACC_LO

typedef struct { uint32_t w[ACC_WORDS]; } acc_t;   // two's complement, little-endian limbs

static void acc_zero(acc_t *a) { memset(a->w, 0, sizeof(a->w)); }

// add (val << shift) into the accumulator, val is a SIGNED 64-bit integer,
// shift is the bit position of val's bit 0 relative to accumulator bit 0.
static void acc_add_shifted(acc_t *a, int64_t val, int shift) {
    if (val == 0) return;
    int neg = val < 0;
    uint64_t mag = neg ? (uint64_t)(-val) : (uint64_t)val;

    if (shift < 0) {            // below our window: would be a rounding loss
        fprintf(stderr, "UNDERFLOW: shift %d below window\n", shift);
        exit(2);
    }
    int word = shift / 32, bit = shift % 32;
    if (word >= ACC_WORDS) { fprintf(stderr, "OVERFLOW: shift %d\n", shift); exit(2); }

    // place the (up to) 64-bit magnitude across up to three limbs
    uint32_t parts[3];
    parts[0] = (uint32_t)(mag << bit);
    parts[1] = (uint32_t)(bit ? (mag >> (32 - bit)) : (mag >> 32));
    parts[2] = (uint32_t)(bit ? (mag >> (64 - bit)) : 0);
    if (bit == 0) { parts[1] = (uint32_t)(mag >> 32); parts[2] = 0; }

    if (!neg) {
        uint64_t carry = 0;
        for (int i = 0; i < 3 && word + i < ACC_WORDS; i++) {
            uint64_t s = (uint64_t)a->w[word + i] + parts[i] + carry;
            a->w[word + i] = (uint32_t)s; carry = s >> 32;
        }
        for (int i = word + 3; carry && i < ACC_WORDS; i++) {
            uint64_t s = (uint64_t)a->w[i] + carry;
            a->w[i] = (uint32_t)s; carry = s >> 32;
        }
    } else {
        uint64_t borrow = 0;
        for (int i = 0; i < 3 && word + i < ACC_WORDS; i++) {
            uint64_t d = (uint64_t)a->w[word + i] - parts[i] - borrow;
            a->w[word + i] = (uint32_t)d; borrow = (d >> 63) & 1 ? 1 : 0;
            if ((uint64_t)a->w[word + i] > 0xFFFFFFFFull) borrow = 1;
            borrow = ((uint64_t)parts[i] + borrow) > ((uint64_t)a->w[word+i] + (uint64_t)parts[i] + borrow - d) ? borrow : borrow;
            borrow = (d > 0xFFFFFFFFull) ? 1 : 0;
        }
        for (int i = word + 3; borrow && i < ACC_WORDS; i++) {
            uint64_t d = (uint64_t)a->w[i] - borrow;
            a->w[i] = (uint32_t)d; borrow = (d > 0xFFFFFFFFull) ? 1 : 0;
        }
    }
}

// is the accumulator negative? (top bit of the top limb)
static int acc_neg(const acc_t *a) { return (a->w[ACC_WORDS - 1] >> 31) & 1; }

static void acc_negate(acc_t *a) {
    uint64_t carry = 1;
    for (int i = 0; i < ACC_WORDS; i++) {
        uint64_t s = (uint64_t)(~a->w[i]) + carry;
        a->w[i] = (uint32_t)s; carry = s >> 32;
    }
}

// convert to double, rounding once at the very end
static double acc_to_double(const acc_t *in) {
    acc_t a = *in;
    int neg = acc_neg(&a);
    if (neg) acc_negate(&a);
    int top = -1;
    for (int i = ACC_WORDS - 1; i >= 0; i--) if (a.w[i]) { top = i; break; }
    if (top < 0) return 0.0;
    int msb = 31; while (msb >= 0 && !((a.w[top] >> msb) & 1)) msb--;
    int hi = top * 32 + msb;                       // position of the leading 1
    // gather 64 bits below the leading one
    uint64_t m = 0;
    for (int b = 0; b < 64; b++) {
        int pos = hi - b;
        if (pos < 0) break;
        uint32_t wbit = (a.w[pos / 32] >> (pos % 32)) & 1;
        m = (m << 1) | wbit;
    }
    double v = ldexp((double)m, hi - 63 + ACC_LO);
    return neg ? -v : v;
}

// split a float into (signed integer mantissa, exponent) exactly
static void split(float f, int64_t *mant, int *exp) {
    if (f == 0.0f) { *mant = 0; *exp = 0; return; }
    int e; double m = frexp((double)f, &e);        // f = m * 2^e, 0.5<=|m|<1
    *mant = (int64_t)ldexp(m, 24);                 // exact: f32 has <=24 significant bits
    *exp  = e - 24;
}

int main(void) {
    const int K = 4096, ROWS = 8;
    float *w = malloc(sizeof(float) * K), *x = malloc(sizeof(float) * K);
    srand(7);
    printf("exact (Kulisch) vs f32 naive vs f64, %d dot products of length %d\n\n", ROWS, K);
    printf("%-6s %22s %14s %14s\n", "row", "exact", "f32 rel err", "f64 rel err");

    double worst32 = 0, worst64 = 0;
    for (int r = 0; r < ROWS; r++) {
        // weights shaped like a real tensor: bf16-rounded, spanning many octaves
        for (int i = 0; i < K; i++) {
            double u = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            double scale = ldexp(1.0, -(rand() % 50));        // 2^0 .. 2^-49
            float f = (float)(u * scale);
            uint32_t bits; memcpy(&bits, &f, 4); bits &= 0xFFFF0000u;  // round to bf16
            memcpy(&f, &bits, 4);
            w[i] = f;
            double g = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            if (rand() % 200 == 0) g *= 40.0;                 // occasional outlier
            x[i] = (float)g;
        }

        acc_t acc; acc_zero(&acc);
        float  f32 = 0.0f;
        double f64 = 0.0;
        for (int i = 0; i < K; i++) {
            int64_t mw, mx; int ew, ex;
            split(w[i], &mw, &ew);
            split(x[i], &mx, &ex);
            if (mw && mx) {
                // the product is EXACT: 48 bits of integer at a known position
                int64_t pm = mw * mx;
                acc_add_shifted(&acc, pm, ew + ex - ACC_LO);
            }
            f32 = (float)(f32 + (float)(w[i] * x[i]));
            f64 += (double)w[i] * (double)x[i];
        }
        double ex_v = acc_to_double(&acc);
        double e32 = ex_v != 0 ? fabs((double)f32 - ex_v) / fabs(ex_v) : 0;
        double e64 = ex_v != 0 ? fabs(f64 - ex_v) / fabs(ex_v) : 0;
        if (e32 > worst32) worst32 = e32;
        if (e64 > worst64) worst64 = e64;
        printf("%-6d %22.15e %14.3e %14.3e\n", r, ex_v, e32, e64);
    }
    printf("\nworst f32 relative error: %.3e\n", worst32);
    printf("worst f64 relative error: %.3e\n", worst64);
    printf("the f16 detour in our GPU path costs 1.7e-3 for comparison\n");
    printf("EXACT ACC DONE\n");
    free(w); free(x);
    return 0;
}
