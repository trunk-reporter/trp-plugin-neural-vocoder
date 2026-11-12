/*
 * p25voc -- P25 IMBE frames to 16 kHz speech with the Vocos-IMBE neural vocoder.
 *
 * From the p25_vocoder research project's engine/ (where the model is trained
 * and exported); the src/ and tools/ paths below are that project's.
 *
 * No Python, no PyTorch: libimbe (parameter decode only), ONNX Runtime for the
 * network (tools/export_onnx.py), and the excitation-phase head's DSP and
 * ISTFT in C. Mirrors src/vocos_modules.py (ExcitationPhaseHead, head
 * "voice") and src/imbe.py (decode_call, harmonics_to_grid) exactly; see
 * tools/engine_parity.py for the check against PyTorch.
 *
 * Input: a trunk-recorder .dvcf capture (u[7] un-shifted as in
 * imbe.unshift_captured), or a raw .fv file of int16 frame vectors (N x 8,
 * decoder form).
 *
 *   p25voc -m model.onnx call.dvcf out.wav
 *   p25voc -m model.onnx -o outdir/ *.dvcf      (one model load for many calls)
 *
 *   p25voc -m model.onnx -l       (a worker: one model load, calls from stdin)
 *
 * Options: -t threads (ONNX Runtime intra-op, default 1), -s seed (noise
 * phase), -e max FEC errors per frame (default 10, as captures.read_dvcf),
 * -z zero noise phase (parity testing), -b print timings.
 *
 * -l reads "<in>\t<out>\n" lines and answers each with one line on stdout:
 * "ok <seconds of audio>" or "fail" (the reason on stderr); an empty line is
 * answered "ready" (the model is loaded). It exits at EOF.
 */

#define _GNU_SOURCE     /* sincosf */
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kiss_fftr.h"
#include "onnxruntime_c_api.h"

/* libimbe, vocoder/imbe_capi.cpp */
void *imbe_create(void);
void imbe_destroy(void *handle);
int imbe_decode_params(void *handle, int16_t *frame_vector, int16_t *snd, float *fund_freq_hz,
                       int16_t *num_harms, int16_t *num_bands, int16_t *v_uv, int16_t *sa);

#define SR 16000
#define FRAME_16K 320           /* samples per 20 ms IMBE frame at 16 kHz */
#define GRID 128                /* feature grid, 0-4 kHz */
#define MAX_HARM 56
#define IMBE_NYQUIST 4000.0
#define F_MAX 7600.0f           /* excitation harmonics up to here */
#define VOICED_MAX_HZ 4000.0f
#define GLOTTAL_HARM 640
#define TWO_PI 6.283185307179586

static const OrtApi *ort;

static void die(const char *msg) { fprintf(stderr, "p25voc: %s\n", msg); exit(1); }

static void ort_check(OrtStatus *st)
{
    if (st) {
        fprintf(stderr, "p25voc: onnxruntime: %s\n", ort->GetErrorMessage(st));
        exit(1);
    }
}

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void *xcalloc(size_t n, size_t size)
{
    void *p = calloc(n ? n : 1, size);
    if (!p) die("out of memory");
    return p;
}

/* ------------------------------------------------------------------------ */
/*  Input                                                                   */
/* ------------------------------------------------------------------------ */

typedef struct {
    int n;
    int16_t *fv;    /* n x 8, decoder form */
    int64_t *errs;  /* n */
} Call;

static unsigned char *read_file(const char *path, long *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *len = ftell(f);
    rewind(f);
    unsigned char *d = xcalloc(*len, 1);
    if (fread(d, 1, *len, f) != (size_t)*len) { fclose(f); free(d); return NULL; }
    fclose(f);
    return d;
}

static uint32_t rd32(const unsigned char *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }

/* captures.read_dvcf + imbe.unshift_captured */
static int read_dvcf(const char *path, int max_errors, Call *c)
{
    long len;
    unsigned char *d = read_file(path, &len);
    if (!d) return -1;
    int cap = 1024, n = 0;
    uint32_t *u = xcalloc(cap * 8, sizeof *u);
    int64_t *errs = xcalloc(cap, sizeof *errs);
    long pos = 0;
    while (pos + 8 <= len && d[pos] == 'S' && d[pos + 1] == 'Y') {
        int type = d[pos + 3];
        long plen = rd32(d + pos + 4);
        pos += 8;
        if (pos + plen > len) break;
        if (type == 0x01 && plen >= 24) {
            int codec = d[pos + 20], count = d[pos + 21], ne = d[pos + 22];
            if (codec == 0 && count == 8 && 24 + 32 <= plen && (max_errors < 0 || ne <= max_errors)) {
                uint32_t v[8];
                int distinct = 0;
                for (int j = 0; j < 8; j++) {
                    v[j] = rd32(d + pos + 24 + 4 * j);
                    if (v[j] != v[0]) distinct = 1;
                }
                if (distinct) {    /* all-equal frames are degenerate */
                    if (n == cap) {
                        cap *= 2;
                        u = realloc(u, cap * 8 * sizeof *u);
                        errs = realloc(errs, cap * sizeof *errs);
                        if (!u || !errs) die("out of memory");
                    }
                    memcpy(u + 8 * n, v, sizeof v);
                    errs[n++] = ne;
                }
            }
        }
        pos += plen;
    }
    free(d);

    /* u[7]: > 127 certainly shifted, odd certainly not, else the nearest
     * preceding label (the first label for leading frames); no odd values
     * at all means shifted throughout. */
    int *label = xcalloc(n, sizeof *label), any_unshifted = 0, first = 0;
    for (int i = 0; i < n; i++) {
        label[i] = u[8 * i + 7] > 127 ? 1 : (u[8 * i + 7] & 1) ? -1 : 0;
        if (label[i] < 0) any_unshifted = 1;
    }
    for (int i = 0; i < n && !first; i++) first = label[i];
    c->n = n;
    c->fv = xcalloc(n * 8, sizeof *c->fv);
    c->errs = errs;
    for (int i = 0, cur = first; i < n; i++) {
        if (label[i]) cur = label[i];
        int shifted = !any_unshifted || cur > 0;
        for (int j = 0; j < 8; j++) {
            uint32_t v = u[8 * i + j];
            if (j == 7 && shifted) v >>= 1;
            c->fv[8 * i + j] = (int16_t)v;
        }
    }
    free(label);
    free(u);
    return 0;
}

static int read_fv(const char *path, Call *c)
{
    long len;
    unsigned char *d = read_file(path, &len);
    if (!d) return -1;
    c->n = len / 16;
    c->fv = (int16_t *)d;
    c->errs = xcalloc(c->n, sizeof *c->errs);
    return 0;
}

/* ------------------------------------------------------------------------ */
/*  Features (imbe.decode_call)                                             */
/* ------------------------------------------------------------------------ */

typedef struct {
    int T;
    float *log_spec, *voicing;          /* T x 128 */
    int64_t *b0, *num_harms, *errs;     /* T */
} Features;

/* np.interp(FREQ_GRID, xs, fs, left=fs[0], right=fs[-1]) for increasing xs */
static void interp_grid(const double *xs, const double *fs, int n, double *out)
{
    int j = 0;
    for (int g = 0; g < GRID; g++) {
        double x = g * (IMBE_NYQUIST / (GRID - 1));
        if (x <= xs[0]) { out[g] = fs[0]; continue; }
        if (x >= xs[n - 1]) { out[g] = fs[n - 1]; continue; }
        while (xs[j + 1] <= x) j++;
        out[g] = fs[j] + (x - xs[j]) * (fs[j + 1] - fs[j]) / (xs[j + 1] - xs[j]);
    }
}

static void decode_features(const Call *c, Features *F)
{
    int T = c->n;
    F->T = T;
    F->log_spec = xcalloc((size_t)T * GRID, sizeof(float));
    F->voicing = xcalloc((size_t)T * GRID, sizeof(float));
    F->b0 = xcalloc(T, sizeof(int64_t));
    F->num_harms = xcalloc(T, sizeof(int64_t));
    F->errs = xcalloc(T, sizeof(int64_t));
    void *dec = imbe_create();
    int16_t fv[8], nh, nb, vuv[MAX_HARM], sa[MAX_HARM];
    float f0;
    double xs[MAX_HARM], ms[MAX_HARM], vs[MAX_HARM], env[GRID], voi[GRID];
    for (int i = 0; i < T; i++) {
        memcpy(fv, c->fv + 8 * i, sizeof fv);
        int b0 = imbe_decode_params(dec, fv, NULL, &f0, &nh, &nb, vuv, sa);
        float *ls = F->log_spec + (size_t)i * GRID, *vo = F->voicing + (size_t)i * GRID;
        int L = nh < MAX_HARM ? nh : MAX_HARM, n = 0;
        for (int h = 0; h < L && f0 > 0; h++) {
            double hf = (h + 1) * (double)f0;
            if (hf > IMBE_NYQUIST) continue;
            xs[n] = hf;
            ms[n] = sa[h] / 4.0;      /* Q14.2 -> linear */
            vs[n] = vuv[h];
            n++;
        }
        if (n == 0) {
            for (int g = 0; g < GRID; g++) { ls[g] = -10.0f; vo[g] = 0.0f; }
        } else {
            interp_grid(xs, ms, n, env);
            interp_grid(xs, vs, n, voi);
            for (int g = 0; g < GRID; g++) {
                ls[g] = (float)log(env[g] > 1e-6 ? env[g] : 1e-6);
                vo[g] = (float)voi[g];
            }
        }
        F->b0[i] = b0;
        F->num_harms[i] = L;
        F->errs[i] = c->errs[i];
    }
    imbe_destroy(dec);
}

static void free_features(Features *F)
{
    free(F->log_spec); free(F->voicing); free(F->b0); free(F->num_harms); free(F->errs);
}

/* ------------------------------------------------------------------------ */
/*  Network (ONNX Runtime)                                                  */
/* ------------------------------------------------------------------------ */

typedef struct {
    OrtEnv *env;
    OrtSession *sess;
    OrtMemoryInfo *mem;
    int n_fft, hop;
} Net;

static int meta_int(OrtSession *sess, const char *key)
{
    OrtModelMetadata *md;
    OrtAllocator *alloc;
    char *val = NULL;
    ort_check(ort->SessionGetModelMetadata(sess, &md));
    ort_check(ort->GetAllocatorWithDefaultOptions(&alloc));
    ort_check(ort->ModelMetadataLookupCustomMetadataMap(md, alloc, key, &val));
    if (!val) die("model metadata is missing n_fft/hop (export with tools/export_onnx.py)");
    int v = atoi(val);
    ort_check(ort->AllocatorFree(alloc, val));
    ort->ReleaseModelMetadata(md);
    return v;
}

static void net_open(Net *net, const char *path, int threads)
{
    ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ort_check(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "p25voc", &net->env));
    OrtSessionOptions *so;
    ort_check(ort->CreateSessionOptions(&so));
    ort_check(ort->SetIntraOpNumThreads(so, threads));
    ort_check(ort->SetInterOpNumThreads(so, 1));
    ort_check(ort->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL));
    ort_check(ort->CreateSession(net->env, path, so, &net->sess));
    ort->ReleaseSessionOptions(so);
    ort_check(ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &net->mem));
    net->n_fft = meta_int(net->sess, "n_fft");
    net->hop = meta_int(net->sess, "hop");
    if (FRAME_16K % net->hop || net->n_fft % 2) die("unsupported n_fft/hop in model metadata");
}

/* Returns the head output (NF x (n_fft + 2)), caller frees via ReleaseValue. */
static OrtValue *net_run(Net *net, const Features *F, float **out)
{
    int64_t T = F->T;
    int64_t s3[3] = {1, T, GRID}, s2[2] = {1, T};
    OrtValue *in[5] = {0}, *res = NULL;
    ort_check(ort->CreateTensorWithDataAsOrtValue(net->mem, F->log_spec, T * GRID * sizeof(float), s3, 3,
                                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[0]));
    ort_check(ort->CreateTensorWithDataAsOrtValue(net->mem, F->voicing, T * GRID * sizeof(float), s3, 3,
                                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[1]));
    int64_t *ints[3] = {F->b0, F->num_harms, F->errs};
    for (int k = 0; k < 3; k++)
        ort_check(ort->CreateTensorWithDataAsOrtValue(net->mem, ints[k], T * sizeof(int64_t), s2, 2,
                                                      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &in[2 + k]));
    const char *in_names[5] = {"log_spec", "voicing", "b0", "num_harms", "errs"};
    const char *out_names[1] = {"head"};
    ort_check(ort->Run(net->sess, NULL, in_names, (const OrtValue *const *)in, 5, out_names, 1, &res));
    for (int k = 0; k < 5; k++) ort->ReleaseValue(in[k]);
    ort_check(ort->GetTensorMutableData(res, (void **)out));
    return res;
}

/* ------------------------------------------------------------------------ */
/*  Head DSP (vocos_modules.ExcitationPhaseHead, mixed+minphase+glottal)    */
/* ------------------------------------------------------------------------ */

typedef struct {
    int n_fft, hop, bins, pad;
    float *window;             /* periodic Hann, n_fft */
    float *glottal;            /* GLOTTAL_HARM + 1 */
    kiss_fftr_cfg fwd, inv;
} Head;

/* vocos_modules.glottal_phase_table: unwrapped phase of each harmonic of a
 * Rosenberg glottal flow derivative (open quotient 0.6, speed quotient 2). */
static void glottal_table(float *table)
{
    const int m = 8192;
    const double tp = 0.6 * 2.0 / 3.0, tn = 0.6 / 3.0;
    double *g = xcalloc(m, sizeof *g), *d = xcalloc(m, sizeof *d), phase[GLOTTAL_HARM + 1];
    for (int i = 0; i < m; i++) {
        double t = (double)i / m;
        g[i] = t < tp ? 0.5 * (1 - cos(M_PI * t / tp)) : t < tp + tn ? cos(M_PI * (t - tp) / (2 * tn)) : 0.0;
    }
    int amin = 0;
    for (int i = 0; i < m; i++) {
        d[i] = g[i] - g[(i + m - 1) % m];
        if (d[i] < d[amin]) amin = i;
    }
    for (int k = 0; k <= GLOTTAL_HARM; k++) {    /* DFT bins 0..640 of d rolled by -amin */
        double re = 0, im = 0, cr = 1, ci = 0, sr = cos(TWO_PI * k / m), si = -sin(TWO_PI * k / m);
        for (int i = 0; i < m; i++) {            /* (cr, ci) = exp(-2 pi i k i / m), by rotation */
            double x = d[(i + amin) % m], t = cr * sr - ci * si;
            re += x * cr;
            im += x * ci;
            ci = cr * si + ci * sr;
            cr = t;
        }
        phase[k] = atan2(im, re);
    }
    double p1 = phase[1], acc = 0;
    for (int k = 0; k <= GLOTTAL_HARM; k++) phase[k] -= p1;
    table[0] = 0.0f;
    for (int k = 1; k <= GLOTTAL_HARM; k++) {
        double diff = k == 1 ? 0.0 : phase[k] - phase[k - 1];
        acc += -nearbyint(diff / TWO_PI);
        table[k] = (float)(phase[k] + TWO_PI * acc);
    }
    free(g);
    free(d);
}

static void head_init(Head *h, int n_fft, int hop)
{
    h->n_fft = n_fft;
    h->hop = hop;
    h->bins = n_fft / 2 + 1;
    h->pad = (n_fft - hop) / 2;
    h->window = xcalloc(n_fft, sizeof(float));
    for (int i = 0; i < n_fft; i++) h->window[i] = (float)(0.5 - 0.5 * cos(TWO_PI * i / n_fft));
    h->glottal = xcalloc(GLOTTAL_HARM + 1, sizeof(float));
    glottal_table(h->glottal);
    h->fwd = kiss_fftr_alloc(n_fft, 0, NULL, NULL);
    h->inv = kiss_fftr_alloc(n_fft, 1, NULL, NULL);
}

/* xorshift64*: uniform [0, 1) */
static double uniform(uint64_t *s)
{
    *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
    return (double)((*s * 2685821657736338717ULL) >> 11) * (1.0 / 9007199254740992.0);
}

/* Linear interpolation position on the IMBE frame grid (to_stft_frames). */
static double time_pos(double pos, int T, int *i0, int *i1)
{
    if (pos < 0) pos = 0;
    if (pos > T - 1) pos = T - 1;
    *i0 = (int)floor(pos);
    if (*i0 > T - 1) *i0 = T - 1;
    *i1 = *i0 + 1 < T ? *i0 + 1 : T - 1;
    return pos - *i0;
}

/* head output (NF x (n_fft+2)) + features -> audio (NF * hop samples) */
static void head_synth(const Head *h, const Features *F, const float *net, int NF, int zero_noise,
                       uint64_t seed, float *audio)
{
    const int T = F->T, n_fft = h->n_fft, hop = h->hop, bins = h->bins, pad = h->pad;

    /* f0_from_b0: invalid frames take the last valid frame's pitch */
    float *f0 = xcalloc(T, sizeof *f0);
    for (int i = 0, last = 0; i < T; i++) {
        if (F->b0[i] >= 0) last = i;
        int64_t b = F->b0[last] < 0 ? 0 : F->b0[last];
        f0[i] = 16000.0f / ((float)b + 39.5f);
    }

    /* harmonic_excitation over (NF - 1) * hop + n_fft samples starting at -pad */
    long n_ex = (long)(NF - 1) * hop + n_fft;
    float *ex = xcalloc(n_ex, sizeof *ex);
    double acc = 0;
    for (long s = 0; s < n_ex; s++) {
        int i0, i1;
        double w = time_pos((s - pad + 0.5) / FRAME_16K - 0.5, T, &i0, &i1);
        double f = f0[i0] * (1.0 - w) + f0[i1] * (double)w;   /* double: the phase sums ~10^6 of these */
        acc += TWO_PI * f / SR;
        double phi = fmod(acc, TWO_PI);
        double K = floor(F_MAX / f);
        if (K < 1) K = 1;
        double half = sin(phi / 2);
        ex[s] = (float)(fabs(half) > 1e-4 ? sin((K + 0.5) * phi) / (2 * half) - 0.5 : K);
    }

    /* bin -> feature-grid position (to_stft_grid), and bin frequency */
    int *g0 = xcalloc(bins, sizeof *g0), *g1 = xcalloc(bins, sizeof *g1);
    float *gu = xcalloc(bins, sizeof *gu), *fbin = xcalloc(bins, sizeof *fbin);
    for (int k = 0; k < bins; k++) {
        fbin[k] = (float)k * SR / n_fft;
        float g = fbin[k] / (4000.0f / (GRID - 1));
        if (g > GRID - 1) g = GRID - 1;
        g0[k] = (int)floorf(g);
        if (g0[k] > GRID - 1) g0[k] = GRID - 1;
        g1[k] = g0[k] + 1 < GRID ? g0[k] + 1 : GRID - 1;
        gu[k] = g - g0[k];
    }

    float *frame = xcalloc(n_fft, sizeof *frame), *ceps = xcalloc(n_fft, sizeof *ceps);
    float *logmag = xcalloc(bins, sizeof *logmag);
    kiss_fft_cpx *spec = xcalloc(bins, sizeof *spec), *tmp = xcalloc(bins, sizeof *tmp);
    float *ola = xcalloc(n_ex, sizeof *ola), *env = xcalloc(n_ex, sizeof *env);
    float lgrid[GRID], vgrid[GRID];
    uint64_t rng = seed * 0x9E3779B97F4A7C15ULL + 1;
    const char *dump_path = getenv("P25VOC_DUMP");      /* debugging: complex spectrum, NF x bins */
    FILE *dump = dump_path ? fopen(dump_path, "wb") : NULL;

    for (int j = 0; j < NF; j++) {
        const float *x = net + (size_t)j * (n_fft + 2);    /* log-magnitude, then phase offset */
        int i0, i1;
        float w = (float)time_pos(((double)j * hop + hop / 2.0 - FRAME_16K / 2.0) / FRAME_16K, T, &i0, &i1);
        float f0j = f0[i0] * (1 - w) + f0[i1] * w;
        float v0 = F->b0[i0] >= 0, v1 = F->b0[i1] >= 0;
        for (int g = 0; g < GRID; g++) {
            lgrid[g] = F->log_spec[(size_t)i0 * GRID + g] * (1 - w) + F->log_spec[(size_t)i1 * GRID + g] * w;
            vgrid[g] = F->voicing[(size_t)i0 * GRID + g] * v0 * (1 - w) + F->voicing[(size_t)i1 * GRID + g] * v1 * w;
        }

        /* excitation_phase: phase of the windowed pulse train */
        for (int i = 0; i < n_fft; i++) frame[i] = ex[(long)j * hop + i] * h->window[i];
        kiss_fftr(h->fwd, frame, spec);

        /* envelope_min_phase: real cepstrum of the envelope, folded */
        for (int k = 0; k < bins; k++) {
            logmag[k] = lgrid[g0[k]] * (1 - gu[k]) + lgrid[g1[k]] * gu[k];
            tmp[k].r = logmag[k];
            tmp[k].i = 0;
        }
        kiss_fftri(h->inv, tmp, ceps);
        for (int i = 0; i < n_fft; i++) {
            float fold = i == 0 || i == n_fft / 2 ? 1.0f : i < n_fft / 2 ? 2.0f : 0.0f;
            ceps[i] *= fold / n_fft;
        }
        kiss_fftr(h->fwd, ceps, tmp);

        for (int k = 0; k < bins; k++) {
            /* phase = excitation + min phase + glottal (voiced) or noise, + offset.
             * The excitation enters as its unit phasor, which skips an atan2. */
            float phase = x[bins + k], ur = 1.0f, ui = 0.0f;
            float v = vgrid[g0[k]] * (1 - gu[k]) + vgrid[g1[k]] * gu[k];
            if (v > 0.5f && fbin[k] <= VOICED_MAX_HZ) {
                float hn = fbin[k] / f0j;
                if (hn > GLOTTAL_HARM) hn = GLOTTAL_HARM;
                int h0 = (int)floorf(hn);
                if (h0 > GLOTTAL_HARM - 1) h0 = GLOTTAL_HARM - 1;
                float u = hn - h0, a = hypotf(spec[k].r, spec[k].i);
                phase += tmp[k].i + h->glottal[h0] * (1 - u) + h->glottal[h0 + 1] * u;
                if (a > 0) { ur = spec[k].r / a; ui = spec[k].i / a; }
            } else if (!zero_noise) {
                phase += (float)((uniform(&rng) * 2 - 1) * M_PI);
            }
            float mag = expf(x[k]), c, sn;
            if (mag > 100.0f) mag = 100.0f;
            sincosf(phase, &sn, &c);
            spec[k].r = mag * (ur * c - ui * sn);
            spec[k].i = mag * (ur * sn + ui * c);
        }

        if (dump) fwrite(spec, sizeof *spec, bins, dump);

        /* ISTFT frame: irfft (1/n), window, overlap-add */
        kiss_fftri(h->inv, spec, frame);
        float *o = ola + (long)j * hop, *e = env + (long)j * hop;
        for (int i = 0; i < n_fft; i++) {
            o[i] += frame[i] / n_fft * h->window[i];
            e[i] += h->window[i] * h->window[i];
        }
    }
    for (long s = 0; s < (long)NF * hop; s++) audio[s] = ola[s + pad] / env[s + pad];
    if (dump) fclose(dump);

    free(f0); free(ex); free(g0); free(g1); free(gu); free(fbin); free(frame); free(ceps);
    free(logmag); free(spec); free(tmp); free(ola); free(env);
}

/* ------------------------------------------------------------------------ */
/*  Output                                                                  */
/* ------------------------------------------------------------------------ */

static int write_wav(const char *path, const float *x, long n)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t data = n * 2, riff = 36 + data, fmt = 16, rate = SR, brate = SR * 2;
    uint16_t pcm = 1, ch = 1, align = 2, bits = 16;
    fwrite("RIFF", 1, 4, f); fwrite(&riff, 4, 1, f); fwrite("WAVEfmt ", 1, 8, f);
    fwrite(&fmt, 4, 1, f); fwrite(&pcm, 2, 1, f); fwrite(&ch, 2, 1, f); fwrite(&rate, 4, 1, f);
    fwrite(&brate, 4, 1, f); fwrite(&align, 2, 1, f); fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&data, 4, 1, f);
    for (long i = 0; i < n; i++) {
        float v = x[i] < -1 ? -1 : x[i] > 1 ? 1 : x[i];
        int16_t s = (int16_t)(v * 32767);
        fwrite(&s, 2, 1, f);
    }
    return fclose(f);
}

/* ------------------------------------------------------------------------ */

static int vocode(Net *net, Head *head, const char *in, const char *out, int max_errors, int zero_noise,
                  uint64_t seed, int bench, long *samples)
{
    Call c = {0};
    size_t len = strlen(in);
    int is_fv = len > 3 && !strcmp(in + len - 3, ".fv");
    if ((is_fv ? read_fv(in, &c) : read_dvcf(in, max_errors, &c)) < 0) {
        fprintf(stderr, "p25voc: can't read %s: %s\n", in, strerror(errno));
        return -1;
    }
    if (c.n < 8) {
        fprintf(stderr, "p25voc: %s: only %d frames, skipped\n", in, c.n);
        free(c.fv); free(c.errs);
        return -1;
    }
    double t0 = now();
    Features F;
    decode_features(&c, &F);
    double t1 = now();
    float *y;
    OrtValue *res = net_run(net, &F, &y);
    double t2 = now();
    int NF = F.T * (FRAME_16K / head->hop);
    long n = (long)NF * head->hop;
    float *audio = xcalloc(n, sizeof *audio);
    head_synth(head, &F, y, NF, zero_noise, seed, audio);
    double t3 = now();
    ort->ReleaseValue(res);
    int rc = write_wav(out, audio, n);
    if (samples) *samples = n;
    if (rc) fprintf(stderr, "p25voc: can't write %s\n", out);
    if (bench) {
        double dur = (double)n / SR, tot = t3 - t0;
        fprintf(stderr, "%s: %.1f s audio | decode+features %.1f ms, network %.1f ms, head DSP %.1f ms | "
                "total %.1f ms = %.0fx real time\n", in, dur, (t1 - t0) * 1e3, (t2 - t1) * 1e3,
                (t3 - t2) * 1e3, tot * 1e3, dur / tot);
    }
    free(audio); free_features(&F); free(c.fv); free(c.errs);
    return rc;
}

static void usage(void)
{
    fprintf(stderr, "usage: p25voc -m model.onnx [-t threads] [-s seed] [-e max_errors] [-z] [-b] in.dvcf|in.fv out.wav\n"
                    "       p25voc -m model.onnx [options] -o outdir in1.dvcf [in2.dvcf ...]\n"
                    "       p25voc -m model.onnx [options] -l    (\"in<TAB>out\" lines on stdin)\n");
    exit(2);
}

int main(int argc, char **argv)
{
    const char *model = NULL, *outdir = NULL;
    int threads = 1, max_errors = 10, zero_noise = 0, bench = 0, lines = 0, i;
    uint64_t seed = 0;
    for (i = 1; i < argc && argv[i][0] == '-' && argv[i][1]; i++) {
        char o = argv[i][1];
        if (strchr("mtseo", o) && i + 1 >= argc) usage();
        switch (o) {
        case 'm': model = argv[++i]; break;
        case 't': threads = atoi(argv[++i]); break;
        case 's': seed = strtoull(argv[++i], NULL, 10); break;
        case 'e': max_errors = atoi(argv[++i]); break;
        case 'o': outdir = argv[++i]; break;
        case 'z': zero_noise = 1; break;
        case 'b': bench = 1; break;
        case 'l': lines = 1; break;
        default: usage();
        }
    }
    if (!model || (lines ? i != argc : outdir ? i >= argc : argc - i != 2)) usage();

    double t = now();
    Net net;
    net_open(&net, model, threads);
    Head head;
    head_init(&head, net.n_fft, net.hop);
    if (bench) fprintf(stderr, "model load %.0f ms (n_fft %d, hop %d)\n", (now() - t) * 1e3, net.n_fft, net.hop);

    int failed = 0;
    if (lines) {
        char line[8192];
        while (fgets(line, sizeof line, stdin)) {
            line[strcspn(line, "\r\n")] = 0;
            if (!line[0]) { printf("ready\n"); fflush(stdout); continue; }
            char *tab = strchr(line, '\t');
            if (!tab) { printf("fail\n"); fprintf(stderr, "p25voc: expected <in>\\t<out>\n"); fflush(stdout); continue; }
            *tab = 0;
            long n = 0;
            if (vocode(&net, &head, line, tab + 1, max_errors, zero_noise, seed, bench, &n)) printf("fail\n");
            else printf("ok %.2f\n", (double)n / SR);
            fflush(stdout);
        }
        return 0;
    }
    if (!outdir) return vocode(&net, &head, argv[i], argv[i + 1], max_errors, zero_noise, seed, bench, NULL) ? 1 : 0;
    for (; i < argc; i++) {
        const char *base = strrchr(argv[i], '/');
        base = base ? base + 1 : argv[i];
        char out[4096];
        const char *dot = strrchr(base, '.');
        int stem = dot ? (int)(dot - base) : (int)strlen(base);
        snprintf(out, sizeof out, "%s/%.*s.wav", outdir, stem, base);
        failed += vocode(&net, &head, argv[i], out, max_errors, zero_noise, seed, bench, NULL) != 0;
    }
    return failed ? 1 : 0;
}
