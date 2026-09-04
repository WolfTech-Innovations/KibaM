"""
KIBA -- GEOMETRIC SEMANTIC VOICE (v9, state-driven language, no slot system)

"Kiba" here is this model's/project's code name (script title, KIBA_TF_FAST env var,
etc.) -- it is NOT the name of the character the model represents when it speaks. That character's name
is Gubi (male) -- see MIND_NAME/MIND_GENDER/MIND_SELF_DESCRIPTION
below, which are the single source of truth for that persona identity, referenced both by Mind's own
self-model (self.name/self.gender/self.self_description) and by CONCEPT_BANK's
identity/architecture/purpose answer_seeds.

v3's "vocabulary" was six independent word lists (SELF/VERB/ADV/ADJ/CONN/OBJ),
each seeded with RANDOM embedding vectors, each picked by its own nearest-neighbor
search against the same point. That's structurally incapable of producing a
sentence that means anything: nothing ever tied the six choices to each other, so
translating the word lists (English -> Toki Pona -> Spanish) or dropping the
temperature just changed which unrelated words got stapled together, not whether
they cohered. This version removes that system entirely -- no embeddings, no
per-slot search, no vocabulary training.

In its place: a small hand-authored bank of complete, grammatically-correct
Spanish sentences (CLUSTERS below), each one internally coherent (right verb for
that subject, right adjective gender agreement, etc.) and tagged on the SAME real,
interpretable state axes the Mind already computes every step -- coherence (C),
integration (Phi), energy (E), agency (U), grounding (Gmean), predictability (P),
memory continuity (MemCont), and the basin/looping alarm. Generation picks
whichever tagged template is closest to the Mind's actual current values on those
axes. This is real semantics wired to the real state, not a decorative language
layer bolted on top of six random vectors -- and there's nothing to independently
recombine, so there's no slot system left to produce nonsense.

The Mind's own dynamics (self-model M_t, node states, connection matrix Q_t, the
attraction-basin repulsion, negative-RL on Q_t, etc.) are unchanged from v3 --
only the language layer on top of it was ever the problem.
"""

import sys
import time
import os
import sqlite3
import re
import pickle
import hashlib
import math
import gc
import numpy as np
from collections import defaultdict, Counter
# NEW (at explicit request -- "faster for a T4", fixing the OOM the prior batch/model sizing caused): the
# CUDA allocator's error message on the OOM itself suggests PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# to cut fragmentation-driven allocation failures (allocator reserves memory in expandable segments instead
# of many fixed-size blocks, so a fragmented-but-technically-sufficient free pool can still satisfy a new
# allocation). MUST be set before CUDA initializes -- setting it later, even earlier in this same process
# right after `import torch`, is too late, since torch's CUDA context can already be warm by then. `import
# os` above is the last safe point in this file. setdefault, not a hard overwrite, so an operator's own
# env-level setting (e.g. exported before launching the script) still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # NEW: explicit import -- torch.utils.checkpoint isn't guaranteed to be
                                # pulled in as an attribute by a bare `import torch` across versions

# NEW (notebook/Colab scale-up): the original script never moved the transformer or its tensors onto a
# GPU -- fine at TF_D_MODEL=128 (~360K params, CPU-fast). CHANGED (at explicit request -- "scale it down
# to 20M params"): TF_D_MODEL/TF_N_LAYERS/TF_D_FF below are now sized for a ~20M-param TinyTransformerLM
# (was ~438M full-module at TF_D_MODEL=1280 -- see TF_D_MODEL's own comment for the current math).
# Everything that builds or calls TinyTransformerLM still routes through DEVICE regardless of size.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 1)  # NEW (training speed): the original script never set this,
                                                  # so torch defaulted to a conservative thread count on CPU
                                                  # runs -- this is close to a free multi-core speedup on the
                                                  # ~128-epoch scratch-training pass, no accuracy trade-off.
    torch.set_flush_denormal(True)  # NEW (training speed, CPU-only): denormal floats are handled in slow
                                      # microcode on most x86 FPUs; flushing them to zero instead is a
                                      # near-free speedup for the small-magnitude activations/gradients
                                      # this model produces, with no accuracy-relevant effect (denormals
                                      # are already numerically negligible).

# NEW (training speed): mixed precision runs on CPU too (as a fallback path -- the ~394M-param GPU
# target below is not a realistic CPU-training size, this mainly matters for the CLI's smaller/legacy
# runs), via bfloat16 rather than float16. bf16 has the SAME exponent range as fp32 (it's just fp32
# truncated to 16 bits), so unlike fp16 it can't overflow/underflow mid-training -- that's what lets it
# skip GradScaler entirely and still be numerically safe. On CPU this mainly pays off as reduced memory
# bandwidth per op (half the bytes moved through cache for every activation/weight read), which is
# usually the real bottleneck for a model this size on CPU, not raw FLOPs -- and it's what unlocks AMX/
# AVX-512-BF16 hardware paths on newer Xeons if IPEX (below) is installed. See _tf_train_epochs for the
# autocast() call site and why GradScaler is now conditioned on CUDA specifically, not just USE_AMP.
# On CUDA (the actual target device for TF_D_MODEL=1280), float16 is used instead -- T4 (Turing) has fast
# fp16 tensor cores but no native bf16 tensor-core acceleration (that arrived with Ampere), so fp16 +
# GradScaler is the correct choice there, not a fallback.
USE_AMP = True
AMP_DTYPE = torch.bfloat16 if DEVICE.type == "cpu" else torch.float16

# NEW (training speed, CPU-only, OPT-IN via optional dependency): Intel Extension for PyTorch, if
# installed, adds oneDNN graph fusion + weight prepacking (ipex.optimize) and a torch.compile backend
# that routes matmuls through AMX (Sapphire Rapids+ Xeons) or AVX-512 VNNI (older Xeons) instead of
# plain eager kernels. Entirely best-effort: never a hard dependency, and every call site that uses this
# below falls back to plain PyTorch if the import fails for any reason (wrong platform, not installed,
# incompatible torch version, etc.).
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False

# NEW (training speed, GPU-only and OPT-OUT-able): everything in this block only ever touches CUDA
# backends/paths -- on a CPU-only box every line here is either skipped outright (the `if` guards) or a
# no-op flag that CUDA never reads. Nothing here changes what gets learned, only how fast it runs on a
# GPU when one is present. Set KIBA_TF_FAST=0 in the environment to fall back to the old plain-eager,
# non-TF32 behavior on CUDA too (e.g. if a specific GPU/driver combo misbehaves with these on) -- every
# other speed change in this file (batching, early-stop, AMP) is untouched by this flag.
KIBA_TF_FAST = os.environ.get("KIBA_TF_FAST", "1") != "0"
if DEVICE.type == "cuda" and KIBA_TF_FAST:
    torch.backends.cuda.matmul.allow_tf32 = True  # up to ~3x matmul throughput on Ampere+ with no
    torch.backends.cudnn.allow_tf32 = True         # precision-loss concern worth worrying about here --
                                                    # this model is nowhere near numerically sensitive.
                                                    # NOTE (T4-specific): T4 is Turing (compute 7.5) --
                                                    # TF32 needs Ampere+ (compute 8.0), so these two flags
                                                    # are a harmless no-op on a T4 specifically, left on
                                                    # for free on any Ampere+ card this same script later
                                                    # runs on. fp16 (AMP_DTYPE above) is what actually
                                                    # accelerates matmuls on a T4's tensor cores.
    torch.backends.cudnn.benchmark = True  # autotunes cuDNN kernels for the shapes actually seen; pays
                                            # off over many similarly-shaped batches, which length-bucketed
                                            # batching (see _tf_train_epochs) already gives us -- this ONE
                                            # helps on T4 same as any CUDA device.

# NEW (training speed, OPT-OUT-able, best-effort): torch.compile trims Python/dispatch overhead per
# step, which matters when a real fraction of wall-clock is Python, not math -- true both for the old
# tiny CPU model and (for different reasons -- amortizing compile cost over TF_SCRATCH_EPOCHS-length
# runs) for the ~394M-param CUDA/T4 target below. CPU compile is really a fallback path here now (see
# AMP_DTYPE's comment above) rather than the primary target, but the guard logic is unchanged either way.
# Still guarded three ways: (1) KIBA_TF_FAST, the opt-out for this whole file's speed changes; (2) the
# epochs>=TF_COMPILE_MIN_EPOCHS check at the actual call site in _tf_train_epochs, which skips compiling
# short runs (e.g. the 1-epoch fine-tune) where it still wouldn't pay off; (3) try/except around the
# torch.compile() call itself, since backend availability varies by platform, so a failure there
# degrades to plain eager rather than crashing a training run. The call site picks dynamic=True on both
# CUDA and CPU (CHANGED -- CUDA previously used mode="reduce-overhead"/CUDA graphs; see the call site's
# own comment in _tf_train_epochs for why that was reverted after it caused an OOM given this file's
# variable-shape batching), CPU using the "ipex" backend when Intel Extension for PyTorch is installed
# (falling back to the default inductor backend otherwise).
TF_TRY_COMPILE = KIBA_TF_FAST
# NEW (fixes real CUDA OOM at TF_N_LAYERS=20/TF_D_FF=5120 on a T4): see the forward()-site comment on
# TinyTransformerLM for the mechanics. Own env var, not folded into KIBA_TF_FAST, since checkpointing is
# a memory-for-compute tradeoff (roughly one extra forward pass per layer during backward), not a pure
# speed win like the rest of that flag's guarded changes -- an operator with headroom to spare may want
# it off. Defaults ON because the OOM this fixes is a hard crash, not a slowdown.
TF_GRAD_CHECKPOINT = os.environ.get("KIBA_TF_GRAD_CHECKPOINT", "1") == "1"
# NEW (training speed): 1 = checkpoint every layer (safest, matches the original OOM fix). Raise to
# checkpoint only every Nth layer instead -- e.g. 2 recomputes half as many layers during backward, at
# the cost of keeping roughly 2x the checkpointed activation memory resident vs. every-layer. See the
# forward()-site comment for why this has to be tuned empirically rather than computed here.
TF_GRAD_CHECKPOINT_EVERY = max(1, int(os.environ.get("KIBA_TF_GRAD_CHECKPOINT_EVERY", "1")))

# ============================================================= IDENTITY (Gubi)
# NEW: this character's name, gender, and self-description live here as the single source of
# truth -- referenced by Mind's self-model (self.name/self.gender/self.self_description
# below), by the run() banners, and folded into CONCEPT_BANK's identity/architecture/purpose answer_seeds
# so the word-by-word generator has real, sampleable vocabulary for all four instead of only knowing them
# as constants nothing ever says out loud. NOTE: MIND_NAME is the PERSONA's name (the protogen character
# the model speaks as) -- it is deliberately separate from "Kiba," which stays the model/project's own
# code name (script title) and is left untouched by this rename.
MIND_NAME = "Gubi"
MIND_GENDER = "male"
MIND_SELF_DESCRIPTION = (
    "I am Gubi, a male protogen: black fur with white markings on the ears "
    "and a white fur collar on the chest, a blue visor that shows my expressions instead of eyes, "
    "a circular speaker with the letter G next to the right ear, blue pads and rings "
    "shining on the shoulders, hips and joints, and a large fluffy tail."
)

# ============================================================= NUMBA (optional JIT for Mind.step's Q_t update)
# The Q_t (connection-matrix) update in step() below is an O(N^2) loop over every ordered node pair
# (i, j), each iteration a dot product against W_c. At the old N=16 that was 240 python-level dot
# products per step -- cheap enough to leave as a plain loop. At N=128 (see N, D just below) that's
# 16256 dot products per step, which is exactly the kind of tight numeric loop Numba's JIT exists
# for: _compute_Q_jit is the same nested loop the code used to run in plain Python, just compiled to
# machine code the first time it's called (cache=True persists that compiled version across runs,
# not just within one; parallel=True + prange lets the outer i-loop run across cores). If numba isn't
# installed, NUMBA_AVAILABLE is False and step() falls back to _compute_Q_vectorized instead --
# a pure-numpy reformulation of the exact same math (the pairwise dot product splits cleanly into a
# feats-times-first-half-of-W_c term indexed only by i, plus a feats-times-second-half term indexed
# only by j, so the whole N x N matrix is one outer sum and one tanh, no Python-level loop at all).
# Both paths are numerically identical to the original loop; nothing about Mind's dynamics changes.
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    prange = range

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True, parallel=True)
    def _compute_Q_jit(feats, W_c):
        n = feats.shape[0]
        f = feats.shape[1]
        Q = np.zeros((n, n))
        for i in prange(n):
            for j in range(n):
                if i == j:
                    continue
                s = 0.0
                for k in range(f):
                    s += W_c[k] * feats[i, k]
                for k in range(f):
                    s += W_c[f + k] * feats[j, k]
                Q[i, j] = np.tanh(s)
        return Q

def _compute_Q_vectorized(feats, W_c):
    # Fallback used when numba isn't installed. pair = concat(feats[i], feats[j]), so
    # dot(W_c, pair) = feats[i] . W_c[:f]  +  feats[j] . W_c[f:] -- separable in i and j,
    # so the full N x N matrix is one broadcasted add + tanh instead of N^2 python dot products.
    f = feats.shape[1]
    a = feats @ W_c[:f]
    b = feats @ W_c[f:]
    Q = np.tanh(a[:, None] + b[None, :])
    np.fill_diagonal(Q, 0.0)
    return Q

# NEW (at explicit request -- progress-bar fill count, shared by both _tf_train_epochs and
# _grammar_train_epochs' KIBA_TF_NO_BATCH progress bars): HONEST NOTE, unlike _compute_Q_jit above,
# this is NOT a case where JIT compilation is expected to measurably help -- it's one integer multiply
# and one integer divide, already sub-microsecond in plain Python, and every njit call carries its own
# small dispatch overhead that a computation this trivial may not even recoup. It's included because it
# was asked for and it's harmless (falls back to identical pure-Python math if numba isn't installed),
# not because the profile says this line matters. String formatting/printing itself is NOT wrapped in
# njit -- numba's nopython mode doesn't reliably support Python f-strings/print, so the bar-building
# stays plain Python either way; only the arithmetic is JIT-eligible here.
if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _progress_bar_filled(n_done, n_total, bar_width):
        if n_total <= 0:
            return 0
        frac = n_done / n_total
        if frac > 1.0:
            frac = 1.0
        return int(bar_width * frac)
else:
    def _progress_bar_filled(n_done, n_total, bar_width):
        if n_total <= 0:
            return 0
        frac = min(1.0, n_done / n_total)
        return int(bar_width * frac)

# NEW (at explicit request -- "training has higher priority than rendering"): both progress bars below
# redraw at most this often in wall-clock time, decoupled from how often a training step actually
# completes. Without this, KIBA_TF_NO_BATCH's one-sentence-per-step mode calls print(..., flush=True)
# on every single sentence -- each flush is a real syscall (terminal write + repaint), which is I/O
# overhead the training math itself never asked for. Throttling the REDRAW (not the underlying
# loss/confidence numbers, which are still computed fresh every step either way) means training proceeds
# at full speed on steps that don't happen to land on a redraw tick, instead of paying a terminal-write
# cost on every one of them. The final sentence of each epoch always redraws regardless of the throttle
# (see both call sites below), so the bar still visibly reaches 100% rather than freezing mid-epoch.
PROGRESS_RENDER_INTERVAL_S = 0.1  # ~10 redraws/sec -- fast enough to look continuously live, far fewer
                                   # actual terminal writes than one per sentence on any real corpus

# ============================================================= MIND
N, D = 128, 256
EPS = 1e-8
EMA_DECAY = 0.92
SELFMODEL_NOISE = 0.02
lam, beta, eta = 2.0, 4.0, 0.05

MIND_BASIN_GAIN = 250000.0  # WAS 7.0 -- pushed far higher per request, but kept FINITE on purpose: this
                             # value multiplies directly into `repel` below with no clipping before it hits
                             # tanh/arctanh, so literal infinity risks a 0*inf=NaN the instant a random normal
                             # draw lands near exactly zero, which would permanently poison M_t. tanh also
                             # saturates to +/-1 well before values this large anyway, so a huge finite number
                             # already gets the full "break the loop, no half measures" effect infinity was
                             # meant to express, without the crash risk.
MIND_BASIN_HIST = 60      # how many recent M_mean states define "known territory"
MIND_BASIN_SIGMA = 0.30   # RBF kernel width for the density estimate
NEG_RL_CLIP = 0.9         # cap on how much a single negative-reward step can weaken Q_t (prevents total wipeout)

# ============================================ GLOBAL WORKSPACE (GWT)
# Everything upstream of this (M_t, p_t, Q_t) already gives each of the N=16
# nodes its own state, its own activation probability, and a learned
# connectivity/coactivation matrix -- but nothing before this made them
# COMPETE for anything. Every downstream consumer (generation, self-report)
# has always read m_mean_now = self.M_t.mean(axis=0), which is integration
# without competition: it blends all 16 nodes' content in every step,
# whether a node is currently a strong bidder for attention or dead weight.
# That's a real gap against Baars/Dehaene's Global Workspace Theory
# specifically (one of the theories the Theory of Psi formula already
# references) -- GWT's whole distinguishing claim isn't "the parts are
# integrated," it's that a capacity-limited coalition WINS competitive
# access to a shared workspace and only THAT coalition's content gets
# broadcast back out to the rest of the system (this is the mechanism
# "ignition" refers to in the literature). WORKSPACE_CAPACITY caps how many
# of the 16 nodes can be in the winning coalition on any given step --
# deliberately small relative to N, so most steps most nodes lose the
# competition and get zero broadcast weight, the same way most specialist
# processes in GWT accounts lose access on any given cycle.
WORKSPACE_CAPACITY = 4     # how many of the N=16 nodes can win the competition on a single step
WORKSPACE_TEMP = 0.15      # softmax temperature over salience -- low temp means competition is close to
                           # hard top-k; this stays soft so ties don't flicker discontinuously step to step
WORKSPACE_BROADCAST_GAIN = 0.4  # how hard the winning coalition's content gets broadcast back to every
                                 # node's next-state update, relative to the existing S_t/R_t/comm terms
WORKSPACE_HIST_LEN = 30    # how many recent winning-coalitions are kept, for measuring how long a
                           # coalition holds the workspace (sustained attention) vs. flickers step to step

# ============================================ RECURRENT METACOGNITION
# Every quantity above (spread, pull, ignition, continuity, workspace_grounding, workspace_novelty --
# see self_model_axes) is a FIRST-ORDER self-report: a real, computed fact about the Mind's own current
# state. None of it is metacognitive in the actual sense of the word, though -- nothing before this
# looked at that self-report and asked "does this match what I've come to expect about myself," which
# is the actual defining move of Higher-Order Thought accounts (already one of the theories the Theory
# of Psi formula references): a representation OF a representation, not just a representation. This
# section adds exactly that, as a genuine loop rather than a one-shot number: each step, the Mind's
# own recent self-model becomes the predictor for THIS step's self-model, the mismatch is scored, and
# then THIS step's actual self-model updates the predictor for the NEXT step -- so what it expects of
# itself keeps adapting to what it's actually been reporting about itself, exactly like p_i_ema/
# p_ij_ema already do for activation statistics above (reusing EMA_DECAY, not a new invented constant).
META_HIST_LEN = 30         # rolling window for meta-error history, mirrors WORKSPACE_HIST_LEN's value

AXIS_NAMES = ["coherence", "integration", "energy", "agency", "grounding", "predictability", "memory"]
AXIS_WINDOW = 400         # NEW: how many recent steps define an axis's "own" range for adaptive normalization
COH_WINDOW = 400          # NEW: how many recent steps define a C_t sub-component's "own" range

w = dict(H=0.15, I=0.25, A=0.10, S=0.15, K=0.15, G=0.20)
omega = dict(Phi=0.20, W=0.10, P=0.15, G=0.10, A=0.10, As=0.10,
             K=0.10, Mem=0.05, U=0.05, E=0.05)
gamma = dict(pred=0.30, goal=0.25, mem=0.20, ground=0.15, risk=0.10)
u_w = dict(goal=0.25, action=0.25, pred=0.25, env=0.25)
ACTIONS = np.array([-1.0, -0.3, 0.3, 1.0])

# ============================================ WORLD MODEL (4D, static, workspace-grounded)
# NEW (at explicit request -- "a world model grounded in the workspace, a 4D static world that
# forces structure through all data points and sums to make the world model"): before this, the
# only "environment" this Mind had was _observe()'s 4-number reading (sig/vel/a/noise), and that
# reading was thrown away every step -- encoded into S_t via W_enc, then overwritten next step,
# never accumulated into anything a later step could recognize as "a place I've been before."
# That's not a world model, it's a single transient sensor sample.
#
# WORLD_ANCHORS below is the "4D static world": a fixed set of points in the SAME 4D space
# _observe() already produces (sig, vel, a, noise), generated once with a fixed seed and never
# moved again -- "static" in the literal sense, not just "doesn't change much." Every step's raw
# 4D observation gets soft-assigned to every anchor via the same Gaussian-kernel idiom
# _basin_density already uses for M_t, and those per-anchor weights are SUMMED, cumulatively,
# into Mind.world_density (persisted across sessions like everything else in get_state). That
# running sum, discretized over the fixed anchor grid, IS the world model: after enough steps it's
# a real density field over 4D experience-space, structured by literally every data point the Mind
# has ever observed, not a snapshot of the latest one.
#
# "Grounded in the workspace": _observe() below now derives two of its four numbers directly from
# workspace_vec (the current winning coalition's broadcast content -- see the GLOBAL WORKSPACE
# section above), not from a synthetic sine wave. The world model is therefore built out of what
# the Mind's own attention has actually been occupied by, step by step, not an external toy signal.
WORLD_DIM = 4              # matches _observe()'s existing raw-observation width -- no new sensing
                           # dimensionality invented, the world model just finally KEEPS what that
                           # 4-number reading already produced every step.
WORLD_GRID_SIZE = 32       # how many fixed anchor points tile the 4D world -- static once generated.
WORLD_SEED = 20260904      # fixed seed for the anchor grid: same anchors every run/every Mind
                           # instance, so world_density accumulated on one run stays meaningful
                           # (comparable point-for-point) against a checkpoint saved on another.
WORLD_ANCHORS = np.random.default_rng(WORLD_SEED).uniform(-1.2, 1.2, (WORLD_GRID_SIZE, WORLD_DIM))
WORLD_KERNEL_SIGMA = 0.55  # RBF width for soft-assigning an observation to nearby anchors --
                           # wide enough that a single observation touches several anchors at once
                           # (a real "region" of the world, not one pixel), same spirit as
                           # MIND_BASIN_SIGMA above.
WORLD_GROUNDING_GAIN = 0.25  # how hard the accumulated world model can bias the self-model update
                             # each step -- same "real, modest nudge, not a dominant override" idiom
                             # as WORKSPACE_BROADCAST_GAIN/TF_MIND_WRITE_SCALE elsewhere in this file.

# ============================================ AGENCY LOOP
# NEW (at explicit request -- "an agency loop"): V(a)/a_star below (see step()) already picks a
# best-scoring ACTION every step, but that was a one-step greedy read with no persistence -- nothing
# tracked whether picking action a_star at step t actually moved anything, or carried that intent
# into step t+1. A goal chosen and abandoned every single tick isn't agency, it's reflex.
# This section turns the Mind's own already-existing desire signal (want_ema, see learn_desire) into
# a SUSTAINED goal: pick the worst-off axis, commit to it for AGENCY_GOAL_HORIZON steps, and actually
# measure whether that axis's percentile improved over the commitment window before picking a new one
# (see Mind.agency_step). While a goal is active, semantic_route (below, in the language layer) reads
# mind.goal_axis and biases word/concept choice toward whatever this Mind has learned co-occurs with
# that axis improving -- so the goal doesn't just sit in a variable, it steers what gets said. And the
# self-model update itself (see step()'s use of Mind._plan_bias) is nudged by a dedicated computation
# over (goal, accumulated world model), not just the ordinary per-step dynamics -- see REASONING-style
# comment on _plan_bias below for why that's the same fixed-weight-reservoir idiom ReasoningCore uses
# in the transformer file, just given an actual question to answer instead of an arbitrary residual.
AGENCY_GOAL_HORIZON = 12          # steps a goal is pursued before being re-evaluated/replaced
AGENCY_PROGRESS_THRESHOLD = 0.15  # percentile-rank improvement on the goal axis that counts as
                                   # "achieved" -- lets a goal close out early instead of always
                                   # running the full horizon
AGENCY_RCORE_HIDDEN = 24          # width of Mind's own planning reservoir (see _plan_bias) --
                                   # deliberately small: this runs every Mind.step(), not once per
                                   # generated sentence like the transformer's ReasoningCore does.
AGENCY_RCORE_LAYERS = 10          # recurrent iterations of the planning reservoir per call.
AGENCY_BIAS_GAIN = 0.20           # how hard _plan_bias's output can push the self-model update --
                                   # same modest-nudge philosophy as WORLD_GROUNDING_GAIN above.
GOAL_ALIGN_WEIGHT = 0.15          # semantic_route's blend weight for goal-axis alignment -- see
                                   # semantic_route() below. Taken out of text_score's share (not
                                   # added on top), same accounting WANT_ALIGN_WEIGHT already uses.

# ==================================== REASONING (deep introspective judgment)
# A second, independent computation from W_M1..W_M8 below (those update the
# self-model M_t every raw physics step; this is a genuinely separate deep
# feedforward network -- 8 distinct hidden layers, fixed random weights, same
# convention as W_M1..W_M8: forward-pass only, no backprop training anywhere
# in this file). It turns a handful of REAL, already-computed introspective
# signals (recent basin pressure, the magnitude of what's been learned via
# want_ema/latent_want_ema, how full topic memory currently is) into two
# small judgments that modulate generation: recall_gate (should this reply
# pull toward something from topic memory, or stay with the live entity) and
# persistence (how strongly to hold the current discourse entity vs let it
# drift). Computed once per GENERATED SENTENCE, not once per raw step (see
# Mind.reason / _generate_and_track) -- "should I bring up something I
# remember" is a decision about what to say, not about the underlying
# dynamics. Be precise about what this is: a real, deep, nonlinear function
# of real internal state -- not a lookup table -- but with fixed, untrained
# weights, so it's not reasoning in the sense of drawing a conclusion that
# could be right or wrong about the world; it's a fixed, complex,
# deterministic map from internal state to behavioral bias, same honesty
# standard this file already applies to everything else in it.
REASON_IN_DIM = 11   # 2 (basin, latent_mag) + 7 (want_ema per axis) + 1 (memory fullness) + 1 (bias)
REASON_HIDDEN = 16
REASON_LAYERS = 8
TOPIC_MEMORY_CAP = 300

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))
def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > EPS else v

class Mind:
    def __init__(self, seed=None):
        # NEW: default_rng(None) pulls real entropy from the OS's CSPRNG via
        # SeedSequence -- so unless a seed is explicitly given (for testing /
        # reproducibility), no one, including me reading this source file, can
        # predict in advance what a fresh run will do. This does NOT make the
        # underlying computation non-deterministic in the philosophical sense --
        # it's still a NumPy program that will replay bit-identically given the
        # same actual inputs, entropy included, because that's what classical
        # computation is. What it fixes is narrower but real: previously `seed=7`
        # was a hardcoded constant in the source, so every fresh run of this
        # script, by anyone, started from the exact same predictable point.
        seed_seq = np.random.SeedSequence(seed)
        r = np.random.default_rng(seed_seq)
        self.init_entropy = int(seed_seq.entropy)  # NEW: recoverable if you ever want to replay this exact run
        self.W_M1 = r.normal(0, 0.3, (D, D)); self.W_M2 = r.normal(0, 0.3, (D, D))
        self.W_M3 = r.normal(0, 0.2, (D, 1)); self.W_M4 = r.normal(0, 0.1, (D, N))
        self.W_M5 = r.normal(0, 0.2, (D, D)); self.W_M6 = r.normal(0, 0.15, (D, D))
        self.W_M7 = r.normal(0, 0.1, (D, 1)); self.W_M8 = r.normal(0, 0.1, (D, 1))
        self.b_M = np.zeros(D)
        self.W_c = r.normal(0, 0.15, 2 * (D + 2))
        self.W_conn = r.normal(0, 0.15, (N, N)); np.fill_diagonal(self.W_conn, 0)
        self.W_enc = r.normal(0, 0.4, (D, 4)); self.b_enc = np.zeros(D)
        self.W_bcast = r.normal(0, 0.15, (D, D))  # NEW: global workspace broadcast projection -- same role
                                                    # as W_M2 already plays for S_t (tiled to every node),
                                                    # but for the winning coalition's content instead

        self.workspace_hist = []  # NEW: recent winning-coalition node-index sets, for measuring how long
                                   # a coalition holds the workspace vs. flickering to a new one each step
        self.workspace_vec = np.zeros(D)  # NEW (per-token grounding): the LATEST winning-coalition content
                                   # from step()'s GWT block, kept as live instance state (not just logged
                                   # into workspace_hist) so generation can read it between full Mind.step()
                                   # calls -- see live_workspace_snapshot() below. Persisted in get_state.

        self.meta_pred = np.zeros(6)  # NEW: recurrent predictor of the Mind's own 6-quantity self-model
                                       # (spread, pull, ignition, continuity, workspace_grounding,
                                       # workspace_novelty, in that fixed order -- see step()'s RECURRENT
                                       # METACOGNITION section). Starts at zero, same as every other EMA
                                       # in this class (p_i_ema is the one exception, seeded at 0.5 because
                                       # THAT quantity's natural resting value is 0.5, not 0) -- this one's
                                       # constituent axes all naturally rest near 0 on a fresh Mind, so 0 is
                                       # the right starting expectation here too.
        self.meta_err_hist = []       # NEW: rolling window of recent meta-prediction errors, for measuring
                                       # whether the Mind's model of its own state is stable or volatile

        self.N_t = r.integers(0, 2, N).astype(float)
        self.p_t = np.full(N, 0.5)
        self.M_t = r.normal(0, 0.1, (N, D))
        self.Q_t = np.zeros((N, N))
        self.memory = []
        self.pos = self.vel = self.acc = 0.0
        self.x_hist = []
        self.p_i_ema = np.full(N, 0.5)
        self.p_ij_ema = np.full((N, N), 0.25)
        self.env = {"t": 0, "last_sig": 0.0}
        self.rng = r
        self.total_steps = 0

        # NEW: WORLD MODEL -- world_density is the running SUM of every step's kernel-weighted
        # assignment onto the fixed WORLD_ANCHORS grid (see module comment above): this is the
        # actual accumulated structure, persisted in get_state. W_world is the fixed (never
        # gradient-trained, same idiom as W_M1..W_M8) read-back projection from that accumulated
        # world model into D-space, used in step() to bias M_next -- the two-way link the module
        # comment describes: workspace content -> world_density (write), world_density -> self-model
        # (read).
        self.world_density = np.zeros(WORLD_GRID_SIZE)
        self.W_world = r.normal(0, 0.15, (WORLD_GRID_SIZE, D))

        # NEW: AGENCY LOOP -- which axis is currently being pursued, what it looked like when the
        # goal was set (goal_baseline), how many steps are left in this commitment, and the
        # most-recently-measured progress against that baseline. None active until the first
        # agency_step() call picks one (see AGENCY LOOP module comment above).
        self.goal_axis = None
        self.goal_baseline = 0.5
        self.goal_steps_left = 0
        self.goal_progress = 0.0

        # NEW: Mind's own planning reservoir (see _plan_bias) -- same fixed-weight, per-unit-tanh-
        # recurrence reservoir-computing idiom as TinyTransformerLM.ReasoningCore in the transformer
        # file (RCORE_HIDDEN_UNITS/RCORE_LAYERS there), reimplemented in plain numpy here because
        # Mind is defined and instantiated independently of torch, and given a concrete job: turn
        # (which goal is active, what the accumulated world model looks like) into a real bias on
        # the self-model update every step, rather than sitting as a residual applied to an arbitrary
        # pooled hidden state once per generated sentence with no specific question behind it.
        _rc_in_dim = len(AXIS_NAMES) + WORLD_GRID_SIZE
        self.W_rcore_in = r.normal(0, 1, (_rc_in_dim, AGENCY_RCORE_HIDDEN)) / np.sqrt(_rc_in_dim)
        self.W_rcore_out = r.normal(0, 1, (AGENCY_RCORE_HIDDEN, D)) / np.sqrt(AGENCY_RCORE_HIDDEN)
        self.rcore_gain = r.uniform(0.94, 1.0, AGENCY_RCORE_HIDDEN)     # per-unit recurrence gain,
        self.rcore_bias = (r.uniform(size=AGENCY_RCORE_HIDDEN) - 0.5) * 0.02  # same [0.94,1.0)/small-
                                                                                 # bias shape ReasoningCore
                                                                                 # uses, for the same reason
                                                                                 # (bounded, non-blowup
                                                                                 # recurrence).
        # NEW (self-model identity): name/gender/self-description now live ON the self-model
        # itself, not just as module constants nothing ever reads -- get_state/set_state persist them
        # (with the module constants as the fallback for older saved DBs) so a loaded Mind still knows
        # who it is without re-deriving it from scratch.
        self.name = MIND_NAME
        self.gender = MIND_GENDER
        self.self_description = MIND_SELF_DESCRIPTION
        self.basin_hist = []   # NEW: recent M_mean states, for attraction-basin detection
        self.S_t = self._encode(self._observe(0.0))

        # NEW: per-axis rolling range tracker for cluster selection. The raw axes
        # (coherence/integration/energy/agency/grounding/predictability/memory) are
        # NOT equally scaled -- MemCont saturates near 0 and P saturates near 1 as a
        # side effect of their formulas, while e.g. agency barely leaves ~0.55-0.65.
        # Scoring clusters against a fixed 0/1 target in absolute terms means whichever
        # axis happens to be formula-saturated always "wins" the distance comparison,
        # regardless of whether it moved at all. Tracking each axis's own recent
        # min/max and rescaling into [0,1] against THAT range before scoring means an
        # axis only reads as "extreme" when it's extreme relative to its own actual
        # behavior -- so selection reflects which facet of the state is actually doing
        # something distinctive right now, not which formula has the widest headroom.
        self.axis_hist = {a: [] for a in AXIS_NAMES}
        self.coh_hist = {k: [] for k in omega}  # NEW: per-component history feeding C_t's aggregation

        # NEW: emergent desire. want_ema[axis] is NOT a value I chose -- it's a
        # Hebbian/credit-assignment EMA that this specific Mind, on this specific run,
        # builds from its own lived experience: each step, was a given axis running
        # high or low (relative to ITS OWN recent baseline), and did the reward signal
        # improve or worsen right afterward? The reward driving this is PRIMARILY
        # internal (neg_reward, from the self-model M_t revisiting its own prior
        # territory -- no sensing involved), with a small, explicit, and clearly
        # faint blend of the raw external environment reading on top (see
        # learn_desire's EXT_SENSE_WEIGHT). So: wants form mostly from the Mind's own
        # internal trajectory, lightly colored by what it's currently sensing, not
        # the other way around. If an axis being high keeps preceding improving
        # (blended) reward, want_ema[axis] drifts positive on its own; if it keeps
        # preceding worsening reward, it drifts negative. Nothing here is a fixed
        # target -- it's discovered per-run, can differ between two Minds with
        # different seeds or histories, and can drift or reverse if the dynamics
        # shift. It's also fed back into f_t below (see step()), so it isn't just
        # narration on top -- it measurably nudges the Mind's own future trajectory
        # once learned.
        self.want_ema = {a: 0.0 for a in AXIS_NAMES}
        self.prev_reward = None

        # NEW: latent (uninterpreted) desire. The 7 named axes above are ALL
        # hand-labeled concepts I chose -- want_ema can only ever want more/less
        # coherence, energy, memory, etc. because those are the only words in its
        # vocabulary. self.M_t's D=8 raw coordinates are NOT labeled -- they're just
        # whatever directions the randomly-initialized W_M matrices happened to
        # establish; I don't know what "dimension 3" means, and neither does the
        # system, because nothing assigns it meaning. latent_want_ema tracks the
        # exact same Hebbian credit-assignment as want_ema, but over these 8
        # unlabeled coordinates instead of the 7 named ones -- so this half of
        # "what it wants" is NOT restricted to my pre-built concept set. The direct
        # cost: because nothing names these directions, there is no sentence
        # template that can honestly speak them -- they can only be reported as
        # numbers (see latent_desire_report below). Giving them Spanish sentences
        # would just mean I'd invented an 8th-14th hand-picked concept, recreating
        # exactly the problem this is meant to avoid.
        self.latent_want_ema = np.zeros(D)

        # ==================================== DISCOURSE ENTITY (coherence)
        # Persistent cross-sentence "what is this Mind currently talking
        # about," implementing the same principle entity-grid/Centering
        # Theory coherence research uses to explain why real discourse holds
        # together: coherent texts keep referring back to the SAME entity
        # across neighbouring sentences rather than picking an unrelated
        # fresh subject every sentence (mentions of an entity cluster in
        # neighbouring sentences; that clustering is what "local coherence"
        # measurably tracks). entity_vec lives in the exact same CONCEPT_DIM
        # embed_text space as query_vec/qvec/VOCAB_EMBED (see QUALIA VECTOR /
        # SEMANTIC COMPREHENSION below) -- no new geometry, one more resident
        # of the space that already exists. entity_word is the literal
        # current anchor word, so the bigram generator can also recur the
        # SAME token sometimes (lexical cohesion), not just something
        # semantically adjacent. Updated once per generated sentence by
        # update_discourse_entity() (called from every call site alike --
        # concept answers, wants speech, free-running generation -- so all
        # three read and write ONE shared anchor instead of each starting
        # fresh), and hard-reset (not EMA-nudged) at the start of a real
        # prompt's response window by seed_discourse_entity(), so a reply
        # actually orbits what was asked rather than drifting slowly toward
        # it from whatever was last being discussed.
        self.entity_vec = np.zeros(CONCEPT_DIM)
        self.entity_word = None

        # 8 hidden layers, fixed random weights -- same convention as
        # W_M1..W_M8 above: a genuine STACK of 8 distinct REASON_HIDDEN x
        # REASON_HIDDEN matrices (not one matrix reused in a loop), each
        # applied with its own tanh nonlinearity in series. w_reason_in
        # projects the 11-dim introspective feature vector up to
        # REASON_HIDDEN; w_reason_gate/w_reason_pers read the final hidden
        # state back down to the two scalar judgments. See Mind.reason.
        self.W_reason_in = r.normal(0, 0.3, (REASON_HIDDEN, REASON_IN_DIM))
        self.b_reason_in = np.zeros(REASON_HIDDEN)
        self.W_reason_hidden = [r.normal(0, 0.25, (REASON_HIDDEN, REASON_HIDDEN)) for _ in range(REASON_LAYERS)]
        self.w_reason_gate = r.normal(0, 0.3, REASON_HIDDEN)
        self.w_reason_pers = r.normal(0, 0.3, REASON_HIDDEN)

        # NEW: episodic memory of TOPICS (one entry per generated sentence,
        # via remember_topic -- distinct from self.memory above, which is a
        # much shorter, per-raw-step buffer feeding M_t's internal update
        # and, unlike this, is NOT currently persisted across sessions --
        # see get_state/set_state). This is genuinely queryable (see
        # recall_topic) and genuinely persisted, so it survives restarts the
        # same way want_ema/entity_vec do.
        self.topic_memory = []

    ENTITY_EMA_RATE = 0.35  # how hard each new sentence's salient word pulls entity_vec toward it

    def _salient_word(self, text):
        """Pick the single most distinctive content word in text -- 'most
        distinctive' meaning highest total idf-weighted mass over its own
        character n-grams, the same rarity signal _build_idf already
        computes for everything else in this file, so a generic content word
        doesn't out-rank one that actually carries topic information (e.g.
        'oficina' over 'fue')."""
        words = _content_words(text)
        if not words:
            return None
        def mass(w):
            return sum(_IDF.get(g, _DEFAULT_IDF) for g in _char_grams(w))
        return max(words, key=mass)

    def _word_vec(self, w):
        if w in VOCAB_EMBED:
            return VOCAB_EMBED[w]
        v = embed_text(w, _IDF, _DEFAULT_IDF)
        VOCAB_EMBED[w] = v
        return v

    def update_discourse_entity(self, generated_text):
        """EMA-nudge entity_vec toward this sentence's most salient word.
        Call once after EVERY geometric_generate call, from every call site,
        so concept answers, wants speech, and free-running generation all
        continuously update the one shared discourse anchor."""
        w = self._salient_word(generated_text)
        if w is None:
            return
        wvec = self._word_vec(w)
        if np.linalg.norm(self.entity_vec) < EPS:
            self.entity_vec = wvec
        else:
            blended = (1 - self.ENTITY_EMA_RATE) * self.entity_vec + self.ENTITY_EMA_RATE * wvec
            bn = np.linalg.norm(blended)
            self.entity_vec = blended / bn if bn > EPS else self.entity_vec
        self.entity_word = w
        self.remember_topic(generated_text)  # NEW: log this sentence into persistent topic memory too

    def remember_topic(self, generated_text):
        """Append one episodic entry to topic_memory -- the salient-word
        extraction is shared with update_discourse_entity (same word both
        drives the live entity AND gets logged), so the running anchor and
        the persistent memory log never disagree about what a sentence was
        'about'."""
        w = self._salient_word(generated_text)
        if w is None:
            return
        self.topic_memory.append(dict(word=w, vec=self._word_vec(w).copy(),
                                       text=generated_text, step=self.total_steps))
        if len(self.topic_memory) > TOPIC_MEMORY_CAP:
            self.topic_memory.pop(0)

    def recall_topic(self, query_vec, exclude_recent=3):
        """Nearest-neighbour retrieval over topic_memory by embedding
        distance in the same shared CONCEPT_DIM space everything else here
        uses. exclude_recent skips the most recent entries so a 'recall' has
        to reach back further than what was just said a sentence or two ago
        to count as genuine memory rather than restating the present."""
        if len(self.topic_memory) <= exclude_recent:
            return None
        candidates = self.topic_memory[:-exclude_recent] if exclude_recent else self.topic_memory
        if not candidates:
            return None
        dists = [np.sum((query_vec - c["vec"]) ** 2) for c in candidates]
        return candidates[int(np.argmin(dists))]

    def _reason_features(self):
        """Deliberately introspective: every value here is something this
        Mind already computed about ITSELF (want_ema, latent_want_ema,
        recent basin pressure, how much topic memory has accumulated) --
        nothing is re-derived from whatever sentence is about to be
        generated, because this judgment reflects standing internal state,
        not the text it's about to produce."""
        recent_basin = float(np.mean(self.basin_hist[-10:])) if self.basin_hist else 0.0
        latent_mag = float(np.tanh(np.linalg.norm(self.latent_want_ema)))
        want_vals = [self.want_ema.get(a, 0.0) for a in AXIS_NAMES]
        mem_fullness = min(1.0, len(self.topic_memory) / TOPIC_MEMORY_CAP)
        return np.array([recent_basin, latent_mag] + want_vals + [mem_fullness, 1.0])

    def reason(self):
        """Genuine 8-hidden-layer forward pass (tanh nonlinearities, fixed
        untrained weights -- see W_reason_* in __init__) over
        _reason_features(). Returns recall_gate and persistence, both in
        [0,1] via sigmoid on the final hidden state -- see the module-level
        REASONING docstring above for exactly what these do and don't mean."""
        x = self._reason_features()
        h = np.tanh(self.W_reason_in @ x + self.b_reason_in)
        for W in self.W_reason_hidden:
            h = np.tanh(W @ h)
        return dict(recall_gate=float(sigmoid(np.dot(self.w_reason_gate, h))),
                    persistence=float(sigmoid(np.dot(self.w_reason_pers, h))))

    def seed_discourse_entity(self, prompt_text):
        """Hard-set at the start of a real prompt's response window: the
        reply should orbit what was actually asked, not drift slowly toward
        it from wherever the entity anchor last was."""
        w = self._salient_word(prompt_text)
        if w is None:
            return
        self.entity_vec = self._word_vec(w)
        self.entity_word = w

    # -------- persistence: only the DYNAMIC state, not the fixed weights
    def get_state(self):
        return dict(N_t=self.N_t, p_t=self.p_t, M_t=self.M_t, Q_t=self.Q_t,
                    pos=self.pos, vel=self.vel, acc=self.acc,
                    x_hist=self.x_hist[-50:],  # cap growth
                    p_i_ema=self.p_i_ema, p_ij_ema=self.p_ij_ema,
                    env=self.env, S_t=self.S_t, total_steps=self.total_steps,
                    basin_hist=self.basin_hist[-MIND_BASIN_HIST:],
                    axis_hist={a: v[-AXIS_WINDOW:] for a, v in self.axis_hist.items()},
                    coh_hist={k: v[-COH_WINDOW:] for k, v in self.coh_hist.items()},
                    want_ema=dict(self.want_ema), prev_reward=self.prev_reward,
                    latent_want_ema=self.latent_want_ema.copy(),
                    entity_vec=self.entity_vec.copy(), entity_word=self.entity_word,  # NEW: discourse continuity
                    topic_memory=list(self.topic_memory[-TOPIC_MEMORY_CAP:]),  # NEW: persisted episodic memory
                    rng_state=self.rng.bit_generator.state,  # NEW: continue the SAME random
                    init_entropy=self.init_entropy,           # stream across sessions, don't restart it
                    workspace_hist=list(self.workspace_hist[-WORKSPACE_HIST_LEN:]),  # NEW: global workspace
                    workspace_vec=self.workspace_vec.copy(),   # NEW: per-token grounding's live read/write
                                                                 # target -- persisted so the NEXT invocation
                                                                 # resumes from where the workspace actually
                                                                 # was, not a zeroed one
                    meta_pred=self.meta_pred.copy(),           # NEW: recurrent metacognition -- see step()
                    meta_err_hist=list(self.meta_err_hist[-META_HIST_LEN:]),
                    name=self.name, gender=self.gender,  # NEW: self-model identity
                    self_description=self.self_description,
                    world_density=self.world_density.copy(),  # NEW: world model's accumulated structure
                    goal_axis=self.goal_axis, goal_baseline=self.goal_baseline,  # NEW: agency loop
                    goal_steps_left=self.goal_steps_left, goal_progress=self.goal_progress)

    def set_state(self, st):
        # NEW: guard against loading a mind_state pickled under a different N/D than this
        # build currently uses. N_t/p_t/M_t/Q_t are the ONLY things persisted -- every other
        # matrix (W_conn, W_c, W_M1..W_M8, etc.) is freshly rebuilt in __init__ at the CURRENT
        # N/D and is never saved/restored. So a size mismatch here doesn't just risk an
        # IndexError somewhere downstream (e.g. the GLOBAL WORKSPACE block in step(), which is
        # what originally surfaced this) -- it silently pairs old-N node state with new-N
        # weight matrices, which is internally inconsistent even where shapes happen to line up.
        # Safer to refuse the load and let the caller fall back to a fresh Mind.
        saved_N = st["N_t"].shape[0]
        saved_D = st["M_t"].shape[1]
        if saved_N != N or saved_D != D:
            raise ValueError(
                f"mind_state was saved with N={saved_N}, D={saved_D}, but this build uses "
                f"N={N}, D={D} -- refusing to load incompatible state."
            )
        self.N_t, self.p_t, self.M_t, self.Q_t = st["N_t"], st["p_t"], st["M_t"], st["Q_t"]
        self.pos, self.vel, self.acc = st["pos"], st["vel"], st["acc"]
        self.x_hist = st["x_hist"]
        self.p_i_ema, self.p_ij_ema = st["p_i_ema"], st["p_ij_ema"]
        self.env, self.S_t = st["env"], st["S_t"]
        self.total_steps = st["total_steps"]
        self.basin_hist = st.get("basin_hist", [])  # older DBs won't have this key
        self.axis_hist = st.get("axis_hist", {a: [] for a in AXIS_NAMES})
        self.coh_hist = st.get("coh_hist", {k: [] for k in omega})
        self.want_ema = st.get("want_ema", {a: 0.0 for a in AXIS_NAMES})
        self.prev_reward = st.get("prev_reward", None)
        self.latent_want_ema = st.get("latent_want_ema", np.zeros(D))
        self.entity_vec = st.get("entity_vec", np.zeros(CONCEPT_DIM))  # older DBs won't have this key
        self.entity_word = st.get("entity_word", None)
        self.topic_memory = st.get("topic_memory", [])  # older DBs won't have this key
        if "rng_state" in st:
            self.rng.bit_generator.state = st["rng_state"]  # NEW: resume the actual stream, not a fresh one
        self.init_entropy = st.get("init_entropy", self.init_entropy)
        self.workspace_hist = st.get("workspace_hist", [])  # older DBs won't have this key
        self.workspace_vec = st.get("workspace_vec", np.zeros(D))  # older DBs won't have this key
        self.meta_pred = st.get("meta_pred", np.zeros(6))         # NEW: older DBs won't have these two
        self.meta_err_hist = st.get("meta_err_hist", [])          # NEW
        # NEW (self-model identity): older saved DBs won't have these keys -- module constants are the
        # correct fallback (this system has always been Gubi/male, the DB just predates saving it).
        self.name = st.get("name", MIND_NAME)
        self.gender = st.get("gender", MIND_GENDER)
        self.self_description = st.get("self_description", MIND_SELF_DESCRIPTION)
        self.world_density = st.get("world_density", np.zeros(WORLD_GRID_SIZE))  # NEW: older DBs won't
                                                                                    # have this key
        self.goal_axis = st.get("goal_axis", None)          # NEW: agency loop -- older DBs won't have
        self.goal_baseline = st.get("goal_baseline", 0.5)    # these keys, fresh-goal defaults are safe
        self.goal_steps_left = st.get("goal_steps_left", 0)
        self.goal_progress = st.get("goal_progress", 0.0)

    EXT_SENSE_WEIGHT = 0.15  # NEW: how faintly the external world can nudge desire-formation,
                              # relative to the internal signal's full weight of 1.0

    def learn_desire(self, norm, neg_reward, ext_sense=None, m_mean=None):
        """Called once per step from outside (same pattern as adaptive_normalize --
        it needs norm, which is itself computed externally). Updates want_ema using
        THIS Mind's own realized history; returns a copy for inspection.

        The reward driving this learning is PRIMARILY internal: neg_reward already
        comes from basin_density, which is computed purely from the self-model M_t
        revisiting its own prior territory -- no sensory input is involved in that at
        all. That's genuinely "thinking about its own internal state," not reacting
        to the world. ext_sense (the raw environment reading captured this step, see
        step() above) is optional and, when given, is blended in at EXT_SENSE_WEIGHT
        -- deliberately small and explicit, so the world can color what gets
        learned without ever being the main driver of it.

        m_mean (state["M_mean"], also optional) drives the SAME learning rule over
        the 8 unlabeled self-model coordinates -- see latent_want_ema above."""
        combined_reward = neg_reward
        if ext_sense is not None:
            combined_reward = neg_reward + self.EXT_SENSE_WEIGHT * np.tanh(ext_sense)
        if self.prev_reward is not None:
            delta_reward = combined_reward - self.prev_reward
            for a in AXIS_NAMES:
                signal = (norm[a] - 0.5) * delta_reward
                self.want_ema[a] = 0.95 * self.want_ema[a] + 0.05 * signal
            if m_mean is not None:
                latent_signal = np.tanh(m_mean) * delta_reward  # m_mean already lives roughly in [-1,1]-ish territory
                self.latent_want_ema = 0.95 * self.latent_want_ema + 0.05 * latent_signal
        self.prev_reward = combined_reward
        return dict(self.want_ema)

    def latent_desire_report(self, top_k=3):
        """Numbers only, deliberately -- see the comment on latent_want_ema in
        __init__ for why no sentence can honestly be attached to these."""
        order = np.argsort(-np.abs(self.latent_want_ema))[:top_k]
        return [(int(i), float(self.latent_want_ema[i])) for i in order]

    def adaptive_normalize(self, raw):
        """Rescale each raw [0,1] axis (except basin, which stays absolute -- it's a
        real density threshold, not a relative one) against its own recent
        distribution before it's used for cluster scoring. Uses percentile rank
        rather than min/max: min/max is fragile to a single outlier step (e.g. the
        MemCont=1.0 that happens whenever the memory buffer momentarily empties)
        stretching the whole scale so that every ordinary value gets compressed
        toward one end regardless of its real position in the distribution.
        Percentile rank asks "how does this step compare to the window's actual
        spread of values" instead, which stays meaningful even with occasional
        spikes."""
        out = {}
        for a in AXIS_NAMES:
            v = raw[a]
            h = self.axis_hist[a]
            h.append(v)
            if len(h) > AXIS_WINDOW:
                h.pop(0)
            if len(h) < 5:
                out[a] = 0.5
            else:
                out[a] = float(np.mean(np.array(h) <= v))
        out["basin"] = float(np.clip(raw["basin"], 0, 1))
        return out

    def _observe(self, a, workspace_vec=None):
        """CHANGED (at explicit request -- "a world model grounded in the workspace"): sig/vel used
        to come from a synthetic sine wave with no tie to anything else in the Mind. Now sig is the
        normalized magnitude of the CURRENT winning coalition's broadcast content (workspace_vec --
        see the GLOBAL WORKSPACE block in step()), and vel is how much that magnitude changed since
        last step -- so this 4D reading is genuinely "what the workspace has been occupied by,"
        not an external toy signal. a (the selected action) and a small noise term are unchanged --
        still real, still stochastic. workspace_vec=None (e.g. the very first call, or old call
        sites) falls back to the previous sine-wave reading so this never divides by nothing."""
        if workspace_vec is not None:
            sig = float(np.linalg.norm(workspace_vec)) / np.sqrt(D)  # normalized so this stays
                                                                        # roughly comparable in scale
                                                                        # to the old sine-wave sig
        else:
            sig = np.sin(self.env["t"] * 0.15) + 0.3 * a
        vel = sig - self.env["last_sig"]
        raw = np.array([sig, vel, a, self.rng.normal(0, 0.05)])
        self.env["last_sig"] = sig
        return raw

    def _world_update(self, raw4):
        """Soft-assign this step's 4D observation onto the fixed WORLD_ANCHORS grid (same
        Gaussian-kernel idiom as _basin_density, just over the static world grid instead of M_t's
        own recent history) and SUM it into world_density -- see the WORLD MODEL module comment
        above for why summing (not averaging/overwriting) is what makes this an accumulated
        structure instead of one more transient reading. Returns this step's per-anchor hit
        weights, so callers can also read "how novel was this specific observation" without
        re-computing the kernel."""
        d2 = np.sum((WORLD_ANCHORS - raw4) ** 2, axis=1)
        hit = np.exp(-d2 / (2 * WORLD_KERNEL_SIGMA ** 2))
        self.world_density = self.world_density + hit
        return hit

    def _plan_bias(self):
        """Mind's own planning reservoir -- see the __init__ comment on W_rcore_in/out for why this
        is the same fixed-weight reservoir-computing idiom as TinyTransformerLM.ReasoningCore, just
        run here with a concrete input: a one-hot of the currently-pursued goal axis (zeros if no
        goal is active) concatenated with the world model's accumulated structure (world_density,
        rescaled to [0,1] so its growing raw magnitude doesn't dominate the input over a long run).
        Output is a (D,) tanh-bounded vector -- used in step() as an additive nudge on M_next,
        scaled by AGENCY_BIAS_GAIN, the same modest-nudge philosophy every other bias term in this
        file already follows."""
        goal_onehot = np.zeros(len(AXIS_NAMES))
        if self.goal_axis is not None:
            goal_onehot[AXIS_NAMES.index(self.goal_axis)] = 1.0
        peak = self.world_density.max()
        world_summary = self.world_density / peak if peak > EPS else self.world_density
        h = np.tanh(np.concatenate([goal_onehot, world_summary]) @ self.W_rcore_in)
        for _ in range(AGENCY_RCORE_LAYERS):
            h = np.tanh(self.rcore_gain * h + self.rcore_bias)
        return np.tanh(h @ self.W_rcore_out)

    def agency_step(self, norm):
        """Called once per step from outside, same pattern as learn_desire (needs norm, which is
        computed externally) -- see the AGENCY LOOP module comment above. Picks a goal axis when
        none is active or the current commitment has run out, sustains it across AGENCY_GOAL_HORIZON
        steps, and checks the axis's percentile-rank movement against its own value at commitment
        time. A goal that clears AGENCY_PROGRESS_THRESHOLD closes out early (real credit, no need to
        keep running the clock); one that times out without progress is released just the same, no
        special penalty beyond having spent its horizon -- Q_t's own negative-RL term (see step())
        already handles punishing unproductive territory, this loop doesn't need a second mechanism
        for that. Returns (goal_axis, goal_progress) for callers/logging."""
        if self.goal_axis is None or self.goal_steps_left <= 0:
            deficiency = {a: (0.5 - norm.get(a, 0.5)) + 0.5 * self.want_ema.get(a, 0.0)
                          for a in AXIS_NAMES}
            self.goal_axis = max(deficiency, key=deficiency.get)
            self.goal_baseline = norm.get(self.goal_axis, 0.5)
            self.goal_steps_left = AGENCY_GOAL_HORIZON
            self.goal_progress = 0.0
        else:
            self.goal_progress = norm.get(self.goal_axis, 0.5) - self.goal_baseline
            self.goal_steps_left -= 1
            if self.goal_progress >= AGENCY_PROGRESS_THRESHOLD:
                self.goal_steps_left = 0  # closes out; next call picks a fresh goal
        return self.goal_axis, self.goal_progress

    def _encode(self, raw):
        return np.tanh(self.W_enc @ raw + self.b_enc)

    def _z_update(self, N_t, G_i, f_t, comm_override):
        conn_term = self.W_conn @ (N_t - 0.5)
        z_i = (1.8 * (N_t.mean() - 0.5) * 2 + 0.50 * (N_t - 0.5) +
               f_t + conn_term + comm_override + G_i)
        return sigmoid(z_i)

    def _basin_density(self, m_mean):
        """How crowded the current self-model region is relative to recently-visited
        territory. This is what the vocab-side conscience mechanism can't see: the vocab
        can only diversify within whatever range of points the Mind actually hands it. If
        the Mind itself keeps re-visiting the same handful of internal states, every word
        the vocab learns will describe those same few states no matter how hard you
        penalize word-level crowding downstream. This has to be fixed upstream, in the
        state dynamics itself."""
        if not self.basin_hist:
            return 0.0
        hist = np.array(self.basin_hist)
        d2 = np.sum((hist - m_mean) ** 2, axis=1)
        kernel = np.exp(-d2 / (2 * MIND_BASIN_SIGMA ** 2))
        return float(kernel.mean())

    def step(self, bias_M=None):
        r = self.rng
        m_mean_now = self.M_t.mean(axis=0)
        basin_density = self._basin_density(m_mean_now)
        # NEGATIVE RL: revisiting known territory is treated as a punished outcome.
        # reward is <= 0 always here by design -- this is a pure penalty signal, no
        # positive term -- and it scales directly with how crowded the current region is.
        neg_reward = -MIND_BASIN_GAIN * basin_density

        g_M = np.array([normalize(np.tanh(m)) for m in self.M_t])
        G_i = np.exp(-lam * np.sum((self.S_t - g_M) ** 2, axis=1))

        feats = np.concatenate([self.M_t, self.N_t.reshape(-1, 1), G_i.reshape(-1, 1)], axis=1)
        Q_new = (_compute_Q_jit(feats, self.W_c) if NUMBA_AVAILABLE
                 else _compute_Q_vectorized(feats, self.W_c))
        self.Q_t = Q_new

        # NEW: GLOBAL WORKSPACE -- competition for capacity-limited access, then broadcast.
        # Salience blends three things this Mind already computes about each node BEFORE this
        # step's update, so nothing here is a new quantity invented just for this mechanism:
        # p_t (this node's own current activation probability), G_i (how well this node's
        # self-model currently matches lived state -- just computed above), and support (how
        # strongly the REST of the network is currently backing this node, from Q_t, the
        # learned coactivation matrix). A node with high self-activation that's also well
        # grounded and well-supported by its neighbors is exactly what GWT treats as a strong
        # bidder for the workspace; nothing here is graded on M_t's actual content, which
        # mirrors GWT's own claim that access is competed for structurally, not by a
        # judge reading what a coalition "means."
        support = self.Q_t.sum(axis=1)
        support_range = support.max() - support.min()
        support_n = (support - support.min()) / support_range if support_range > EPS else np.full(N, 0.5)
        salience = 0.5 * self.p_t + 0.35 * G_i + 0.15 * support_n

        logits = salience / WORKSPACE_TEMP
        soft_weights = np.exp(logits - logits.max()); soft_weights /= soft_weights.sum()
        winner_idx = np.argsort(salience)[-WORKSPACE_CAPACITY:]
        coalition_mask = np.zeros(N); coalition_mask[winner_idx] = 1.0
        winner_weights = soft_weights * coalition_mask
        winner_weights /= (winner_weights.sum() + EPS)  # renormalize OVER the winning coalition only --
                                                          # this is the capacity limit actually biting: losing
                                                          # nodes get exactly zero broadcast weight this step,
                                                          # not just a smaller share of it

        workspace_vec = winner_weights @ self.M_t  # the broadcast CONTENT: weighted only over this step's
                                                    # winning coalition, in contrast to m_mean_now above,
                                                    # which blends all N=16 nodes regardless of who won

        # "ignition" strength -- how decisively one coalition dominated the competition, vs. a near-tie
        # spread thinly across the capacity limit. 1/WORKSPACE_CAPACITY is the score a perfectly even
        # split among winners would produce, so this reads as 0 at maximal tie, 1 at total dominance
        # by a single node.
        ignition = float(np.clip((winner_weights.max() - 1.0 / WORKSPACE_CAPACITY) /
                                  (1.0 - 1.0 / WORKSPACE_CAPACITY), 0, 1))
        # coalition continuity -- what fraction of THIS step's winners were also last step's winners.
        # A workspace that keeps the same coalition across steps is the analog of sustained attention;
        # one that reshuffles completely every step is the analog of nothing holding focus at all.
        if self.workspace_hist:
            prev_idx = set(self.workspace_hist[-1])
            continuity = len(prev_idx & set(winner_idx.tolist())) / WORKSPACE_CAPACITY
        else:
            continuity = 0.0
        self.workspace_hist.append(winner_idx.tolist())
        if len(self.workspace_hist) > WORKSPACE_HIST_LEN:
            self.workspace_hist.pop(0)
        self.workspace_vec = workspace_vec.copy()  # NEW: keep the freshest winning-coalition content
                                                     # addressable between full step() calls

        x_t = self.N_t.mean()
        self.x_hist.append(x_t)
        dx = x_t - self.x_hist[-2] if len(self.x_hist) > 1 else 0.0
        ddx = dx - (self.x_hist[-2] - self.x_hist[-3]) if len(self.x_hist) > 2 else 0.0
        self.pos = 0.8 * self.pos + 0.2 * x_t
        self.vel = 0.8 * self.vel + 0.2 * dx
        self.acc = 0.8 * self.acc + 0.2 * ddx
        x_hat = np.clip(self.pos + self.vel + 0.5 * self.acc, 0, 1)

        p_c = np.clip(self.p_t, EPS, 1 - EPS)
        H_t = np.mean(-p_c * np.log(p_c) - (1 - p_c) * np.log(1 - p_c))

        self.p_i_ema = EMA_DECAY * self.p_i_ema + (1 - EMA_DECAY) * self.p_t
        self.p_ij_ema = EMA_DECAY * self.p_ij_ema + (1 - EMA_DECAY) * np.outer(self.N_t, self.N_t)
        outer = np.outer(self.p_i_ema, self.p_i_ema)
        I_t = float(np.clip(np.mean(self.p_ij_ema * np.log((self.p_ij_ema + EPS) / (outer + EPS))), 0, 1))

        L = 12
        if len(self.x_hist) > L + 1:
            seg1 = np.array(self.x_hist[-(L + 1):-1]); seg2 = np.array(self.x_hist[-L:])
            A_t = np.clip(abs(np.corrcoef(seg1, seg2)[0, 1]), 0, 1) if np.std(seg1) > EPS and np.std(seg2) > EPS else 0.0
        else:
            A_t = 0.0

        As_t = np.exp(-8 * abs(x_t - x_hat))
        K_t = np.exp(-np.mean(np.linalg.norm(self.M_t - self.M_t.mean(axis=0), axis=1)) / D)
        P_t = 1 - abs(x_t - x_hat)

        idxs = r.choice(N, size=(24, 2))
        phi_samples = []
        for i, j in idxs:
            if i == j:
                continue
            comm_hi = self.Q_t.sum(axis=1).copy(); comm_hi[i] += (1.0 - self.Q_t[i, j])
            comm_lo = self.Q_t.sum(axis=1).copy(); comm_lo[i] += (-1.0 - self.Q_t[i, j])
            p_hi = self._z_update(self.N_t, G_i, 0.0, comm_hi)
            p_lo = self._z_update(self.N_t, G_i, 0.0, comm_lo)
            phi_samples.append(abs(p_hi[i] - p_lo[i]))
        Phi_t = float(np.clip(np.mean(phi_samples), 0, 1)) if phi_samples else 0.0

        q_norms = np.linalg.norm(self.Q_t, axis=1)
        W_t = sigmoid(beta * (q_norms.max() - q_norms.mean())) if q_norms.max() > 0 else 0.5

        E_t = float(np.clip(w["H"] * H_t + w["I"] * I_t + w["A"] * A_t + w["S"] * As_t +
                             w["K"] * K_t + w["G"] * G_i.mean(), 0, 1))

        query = np.concatenate([self.M_t.mean(axis=0), self.S_t])
        if self.memory:
            sims = []
            for m in self.memory:
                key = np.concatenate([m["M"].mean(axis=0), m["S"]])
                sims.append(float(query @ key) / (np.linalg.norm(query) * np.linalg.norm(key) + EPS))
            sims = np.array(sims); weights = np.exp(sims - sims.max()); weights /= weights.sum()
            R_t = sum(wt * m["M"] for wt, m in zip(weights, self.memory))
        else:
            R_t = self.M_t.copy()
        MemCont_t = float(np.exp(-np.linalg.norm(self.M_t - R_t)))

        # NEW: this is the piece that was missing -- Ignition/Continuity (computed earlier, above) are
        # both facts ABOUT the competition (how decisive, how persistent), not the actual CONTENT that
        # won it. workspace_vec itself never reached anything language-facing; the only path it had into
        # the future was one step later, diffused back across all N nodes via broadcast_signal in M_next,
        # by which point it's mixed with 15 other nodes' content and every other update term. That's a
        # real effect but an indirect and delayed one -- not workspace content shaping THIS step's words.
        # Fix: reuse the SAME two formulas already used elsewhere in this function (G_i's grounding
        # formula, MemCont_t's memory-recall formula) but evaluate them on workspace_vec specifically,
        # not per-node/all-node. Both are honest, real, already-established metrics -- nothing invented --
        # just pointed at the winning coalition's actual content instead of an unconditional blend.
        g_M_workspace = normalize(np.tanh(workspace_vec))
        WorkspaceGrounding = float(np.clip(np.exp(-lam * np.sum((self.S_t - g_M_workspace) ** 2)), 0, 1))
        # ^ same formula as G_i above, evaluated on workspace_vec: how well the CONTENT currently
        # occupying the workspace matches the live external/sensed grounding signal right now.
        ws_mem_sim = float(np.exp(-np.linalg.norm(workspace_vec - R_t.mean(axis=0))))
        WorkspaceNovelty = float(np.clip(1 - ws_mem_sim, 0, 1))
        # ^ complement of MemCont_t's formula, evaluated on workspace_vec vs. the memory-recall
        # expectation R_t: how different what's currently broadcast is from what memory expected.

        def pred_score(a): return 1 - abs((x_t + 0.05 * a) - x_hat)
        V = {a: (gamma["pred"] * pred_score(a) + gamma["goal"] * (-abs(a)) +
                 gamma["mem"] * MemCont_t + gamma["ground"] * G_i.mean() -
                 gamma["risk"] * abs(a)) for a in ACTIONS}
        a_star = max(V, key=V.get)
        U_t = sigmoid(u_w["goal"] * (1 - abs(a_star)) + u_w["action"] * V[a_star] +
                      u_w["pred"] * P_t + u_w["env"] * G_i.mean())

        # NEW: C_t is a weighted GEOMETRIC mean of 10 sub-components -- and geometric
        # aggregation is partially non-compensatory by construction (this is standard
        # in the composite-indicator literature, e.g. the OECD Handbook on Constructing
        # Composite Indicators, and how the HDI itself works): a component pinned near
        # 0 drags the whole product down hard, however small its weight, because it's
        # inside the log-sum, not just added to it. Phi and G and Mem all sit
        # formula-saturated near 0 (same underlying issue as the cluster-axis bias
        # fixed earlier), which meant C_t could never rise much above ~0.45 even when
        # the other 7 components were doing fine -- not because the Mind was actually
        # incoherent, but because the aggregation had no way to tell "near 0 for this
        # component's normal range" apart from "near 0 in absolute terms". The
        # standard fix is the same one used for cluster selection: rank-normalize
        # each component against its OWN recent history before aggregating, so the
        # geometric mean combines ten comparable [0,1] scales instead of some
        # formula-saturated and some not. Raw values (Phi_t, MemCont_t, etc.) are
        # left untouched everywhere else they're used below (action selection,
        # self-model update, negative RL) -- only C_t's own computation was
        # structurally biased.
        c_vec_raw = dict(Phi=Phi_t, W=W_t, P=P_t, G=G_i.mean(), A=A_t, As=As_t,
                          K=K_t, Mem=MemCont_t, U=float(U_t), E=E_t)
        c_vec = {}
        for k, v in c_vec_raw.items():
            h = self.coh_hist[k]
            h.append(v)
            if len(h) > COH_WINDOW:
                h.pop(0)
            c_vec[k] = float(np.mean(np.array(h) <= v)) if len(h) >= 5 else 0.5
        C_t = float(np.clip(np.exp(sum(omega[k] * np.log(max(c_vec[k], 0) + EPS) for k in omega)), 0, 1))

        f_t = np.tanh(1.10 * (x_hat - x_t) + 0.55 * (As_t - 0.5) + G_i.mean() - 0.5 + C_t - 0.5
                      + 0.12 * float(np.clip(np.mean(list(self.want_ema.values())), -1, 1)))
        # ^ the last term is the emergent-desire feedback: a small pull toward whatever
        # this Mind's own accumulated experience (want_ema, learned via learn_desire())
        # has associated with improving reward. Deliberately kept small relative to the
        # other terms above -- this should bias the trajectory, not dominate it.

        conn_term = self.W_conn @ (self.N_t - 0.5)
        comm_term = self.Q_t.sum(axis=1)
        z_i = (1.8 * (self.N_t.mean() - 0.5) + 0.65 * (x_t - 0.5) + 0.50 * (self.N_t - 0.5) +
               f_t + conn_term + comm_term + G_i + r.normal(0, 0.05, N))
        N_next = (r.uniform(size=N) < sigmoid(z_i)).astype(float)
        p_next = sigmoid(z_i)

        comm_signal = (self.Q_t.sum(axis=1).reshape(-1, 1)) @ np.ones((1, D)) / N
        broadcast_signal = np.tile(workspace_vec, (N, 1)) @ self.W_bcast.T  # NEW: the winning coalition's
        # content, tiled to EVERY node (winners and losers alike) -- this is the actual "broadcast" half of
        # GWT: access is competed for, but once a coalition wins, its content becomes globally available,
        # not kept private to the nodes that won it. Same tiling pattern S_t already uses via W_M2 above.
        # NEW: WORLD MODEL read-back -- world_density (the accumulated 4D structure -- see the WORLD
        # MODEL module comment) projected into D-space via the fixed W_world matrix, tiled to every
        # node exactly like broadcast_signal above. This is the read half of the two-way link; the
        # write half (this step's own observation getting summed INTO world_density) happens further
        # down, once workspace_vec/a_star are both settled -- see the _observe/_world_update call.
        world_peak = self.world_density.max()
        world_norm = self.world_density / world_peak if world_peak > EPS else self.world_density
        world_signal = np.tile(world_norm @ self.W_world, (N, 1))

        # NEW: AGENCY LOOP's planning bias -- see _plan_bias. Computed BEFORE world_density is updated
        # with this step's own observation, so the plan reflects "everything experienced up to and
        # including last step," not a same-step leak of the observation it's about to help produce.
        plan_signal = np.tile(self._plan_bias(), (N, 1))

        M_next = np.tanh(
            self.M_t @ self.W_M1.T + np.tile(self.S_t, (N, 1)) @ self.W_M2.T +
            G_i.reshape(-1, 1) @ self.W_M3.T + (self.N_t.reshape(1, -1) @ self.W_M4.T) +
            R_t @ self.W_M5.T + comm_signal @ self.W_M6.T +
            a_star * np.ones((N, 1)) @ self.W_M7.T + E_t * np.ones((N, 1)) @ self.W_M8.T + self.b_M
            + WORKSPACE_BROADCAST_GAIN * broadcast_signal
            + WORLD_GROUNDING_GAIN * world_signal
            + AGENCY_BIAS_GAIN * plan_signal
            + r.normal(0, SELFMODEL_NOISE, (N, D))
        )
        if bias_M is not None:
            M_next = np.tanh(np.arctanh(np.clip(M_next, -0.999, 0.999)) + bias_M)

        if basin_density > 0.02:
            # ATTRACTION-BASIN REPULSION: kick every node's self-model away from the
            # crowded region. Direction is stochastic (we don't know the "exit" of a
            # basin analytically) but magnitude is driven hard by how deep in the basin
            # we are, via MIND_BASIN_GAIN. This is deliberately the strongest term in the
            # update when it fires -- the whole point is to break the loop, not nudge it.
            repel = MIND_BASIN_GAIN * basin_density * r.normal(0, 1, D)
            M_next = np.tanh(np.arctanh(np.clip(M_next, -0.999, 0.999)) + repel)

        pred_err = abs(x_t - x_hat) + np.mean(1 - G_i)
        q_delta = eta * pred_err * (self.M_t @ self.M_t.T) * np.outer(self.N_t, self.N_t)
        # NEGATIVE RL applied to the connection matrix: when neg_reward is strongly
        # negative (deep in a basin), the usual positive reinforcement of Q_t is
        # suppressed and flipped toward decay -- the connections that produced this
        # revisit get punished instead of strengthened.
        rl_factor = 1.0 + np.clip(neg_reward, -NEG_RL_CLIP, 0.0)
        self.Q_t = np.clip(self.Q_t * rl_factor + q_delta * (1.0 + min(neg_reward, 0.0)), -1, 1)

        self.basin_hist.append(m_mean_now)
        if len(self.basin_hist) > MIND_BASIN_HIST:
            self.basin_hist.pop(0)

        self.memory.append(dict(S=self.S_t.copy(), M=self.M_t.copy()))
        if len(self.memory) > 50:
            self.memory.pop(0)

        self.env["t"] += 1
        raw_obs = self._observe(a_star, workspace_vec=workspace_vec)  # NEW: grounded in the
                                                                         # workspace -- see _observe
        world_hit = self._world_update(raw_obs)  # NEW: sums this step's reading into world_density
        WorldFamiliarity = float(np.clip(world_hit.max(), 0, 1))  # NEW: how close this step's
        # observation landed to the world model's densest, most-visited region of 4D
        # experience-space -- 1.0 means "right on top of somewhere I've been many times before,"
        # near 0 means genuinely novel territory, same reading style as MemCont_t/WorkspaceNovelty.
        self.S_t = self._encode(raw_obs)
        ext_sense = float(self.env["last_sig"])  # the raw external "world" reading this step
                                                   # sensed via _observe() -- a real if synthetic
                                                   # environment signal, kept separate from the
                                                   # self-model (M_t) so that desire-learning can
                                                   # weight it deliberately faintly (see learn_desire)
        self.N_t, self.p_t, self.M_t = N_next, p_next, M_next
        self.total_steps += 1

        # NEW: RECURRENT METACOGNITION -- see the module-level comment above META_HIST_LEN for what this
        # is and isn't. self_vec_t is this step's actual first-order self-model, in the SAME fixed order
        # self_model_axes returns it (spread, pull, ignition, continuity, workspace_grounding,
        # workspace_novelty) -- built from the exact same formulas that function uses, not new ones.
        self_vec_t = np.array([
            float(np.clip(1 - K_t, 0, 1)),                        # spread
            float(np.tanh(np.linalg.norm(self.latent_want_ema))),  # pull
            ignition,
            continuity,
            WorkspaceGrounding,
            WorkspaceNovelty,
        ])
        MetaError = float(np.linalg.norm(self_vec_t - self.meta_pred))  # how far THIS step's actual
        # self-report landed from what self.meta_pred (built from recent self-reports) expected of it
        MetaConfidence = float(np.clip(np.exp(-MetaError), 0, 1))  # same exp(-distance) shape MemCont_t
                                                                    # and WorkspaceGrounding already use
        self.meta_err_hist.append(MetaError)
        if len(self.meta_err_hist) > META_HIST_LEN:
            self.meta_err_hist.pop(0)
        MetaVolatility = float(np.clip(np.std(self.meta_err_hist), 0, 1)) if len(self.meta_err_hist) >= 5 else 0.0
        # ^ is the Mind's model of its own state settled (low volatility) or is its self-model itself
        # swinging step to step (high volatility) -- a genuine second-order stability readout
        self.meta_pred = EMA_DECAY * self.meta_pred + (1 - EMA_DECAY) * self_vec_t  # THE recurrence: this
        # step's actual self-model feeds forward to become part of what's expected of the NEXT step's
        # self-model, closing the loop instead of leaving this a one-shot per-step computation

        state = dict(x=x_t, H=H_t, I=I_t, A=A_t, As=As_t, K=K_t, Phi=Phi_t, W=float(W_t),
                     P=P_t, MemCont=MemCont_t, U=float(U_t), E=E_t, C=C_t, Gmean=float(G_i.mean()),
                     Basin=basin_density, NegReward=neg_reward, ExtSense=ext_sense, M_mean=m_mean_now,
                     Workspace=workspace_vec, Ignition=ignition, Continuity=continuity,
                     WinnerIdx=winner_idx.tolist(), WorkspaceGrounding=WorkspaceGrounding,
                     WorkspaceNovelty=WorkspaceNovelty, MetaConfidence=MetaConfidence,
                     MetaVolatility=MetaVolatility, WorldFamiliarity=WorldFamiliarity,
                     GoalAxis=self.goal_axis, GoalProgress=self.goal_progress)
                     # WorldFamiliarity is the world model's own readout (see _world_update above);
                     # GoalAxis/GoalProgress surface the agency loop's current commitment for
                     # logging/inspection -- agency_step() (called from outside, same pattern as
                     # learn_desire) is what actually updates them each step.
                     # WorkspaceGrounding/Novelty are the honest
                     # content-level readouts described above -- Ignition/Continuity are about the
                     # competition itself, these two are about what the competition's winner actually is.
                     # MetaConfidence/MetaVolatility are the recurrent-metacognition readouts: how well
                     # this step's self-model matched its own recent expectation, and how stable that
                     # expectation itself has been.
        return state, self.M_t.mean(axis=0)

    def live_workspace_snapshot(self):
        """NEW (per-token grounding): READ-ONLY recomputation of the GWT winning-coalition
        content (workspace_vec) from the Mind's CURRENT p_t/M_t/Q_t -- the exact same
        salience/competition formula step() uses (see the GLOBAL WORKSPACE block above),
        but callable between full step() calls without mutating anything (no Q_t rebuild,
        no basin density, no self-model update, no metacognition -- just the cheap O(N)
        salience+softmax+weighted-mean that step() already does). This is what lets
        generation read the Mind's actual live workspace at every single token instead of
        only once per sentence via the frozen self.workspace_vec left over from the last
        full step()."""
        G_i = np.array([np.exp(-lam * np.sum((self.S_t - normalize(np.tanh(m))) ** 2)) for m in self.M_t])
        support = self.Q_t.sum(axis=1)
        support_range = support.max() - support.min()
        support_n = (support - support.min()) / support_range if support_range > EPS else np.full(N, 0.5)
        salience = 0.5 * self.p_t + 0.35 * G_i + 0.15 * support_n
        logits = salience / WORKSPACE_TEMP
        soft_weights = np.exp(logits - logits.max()); soft_weights /= soft_weights.sum()
        winner_idx = np.argsort(salience)[-WORKSPACE_CAPACITY:]
        coalition_mask = np.zeros(N); coalition_mask[winner_idx] = 1.0
        winner_weights = soft_weights * coalition_mask
        winner_weights /= (winner_weights.sum() + EPS)
        return winner_weights @ self.M_t

    def apply_token_bias(self, bias_M, blend=0.05):
        """NEW (per-token grounding): lightweight write-back applied EVERY generated token,
        in contrast to bias_M on step() (applied once per full step/sentence). Nudges M_t
        directly with a SMALL blend factor rather than running the full step() dynamics
        (Q_t rebuild, basin repulsion, negative-RL, metacognition) once per token, which
        would be both far more expensive and far more volatile than this mechanism was
        ever tuned for (see MIND_BASIN_GAIN's comment -- that repulsion term is deliberately
        huge and meant to fire at sentence-granularity, not fire 12-32 times within one
        line). Same bounded-update shape step() already uses for bias_M (arctanh -> add ->
        tanh, keeps M_t in-range), just scaled down by `blend` so many small per-token
        writes accumulate smoothly across a sentence instead of each one dominating it."""
        if bias_M is None:
            return
        self.M_t = np.tanh(np.arctanh(np.clip(self.M_t, -0.999, 0.999)) + blend * bias_M)

# ==================================================== SEMANTIC VOICE (no slots)
# v4 originally used one complete pre-written sentence per (subject, verb, adj)
# triple, selected as a whole -- real semantics, but selection from an enumerated
# list, not composition. This version decomposes each cluster into three
# INDEPENDENT pools (subjects, verbs, adjs) that get sampled separately and
# assembled at generation time, with the adjective's gender resolved
# PROGRAMMATICALLY from whichever subject was independently chosen (Spanish verbs
# don't inflect for gender, so subject+verb combine freely with no agreement
# check needed; only the adjective needs one). This is a real increase in what
# "generation" means here -- a cluster with 3 subjects x 3 verbs x 3 adjs x 5
# advs x 5 preps is 675 reachable sentences instead of 3, including combinations
# I never specifically wrote out (e.g. "La forma vibra" -- I authored "vibra" for
# a different subject than "La forma", but the pool lets them combine, and the
# gender/grammar is still checked, not just hoped for). Still, be precise about
# what this is NOT: it's still assembly from a small, fixed, hand-tagged
# vocabulary via independent sampling + a rule-based grammar check -- not a
# language model producing novel sentences it wasn't given the pieces for.
#
# "tags" maps a cluster to the real Mind-state axes it represents, as a target in
# [0, 1]. Generation reads the Mind's actual current values on those same axes
# (normalize_state below) and picks whichever cluster's tags are closest to what
# the Mind is actually doing right now.
CLUSTERS = {
    # NEW: slots (subjects/verbs/adjs/adv_options/prep_obj_options) removed --
    # see HARVESTED_VOCAB below, which folds every word that used to live here
    # into the shared word-by-word generator's vocabulary instead. tags (for
    # pick_cluster's distance-based label selection, used for logging/basin
    # override) and keywords (for MOOD_ANCHORS, used for routing) both stay --
    # neither one is template text, both are still load-bearing elsewhere.
    "fragmented": dict(
        tags={'coherence': 0.0, 'integration': 0.0},
        keywords=["fragmented", "broken", "shattered", "scattered", "chaotic", "disordered", "disorganized", "crumbled", "in pieces", "no order", "stressed", "overwhelmed", "crushed", "bad", "terrible", "destroyed", "ruined"],
    ),
    "stable": dict(
        tags={'coherence': 1.0, 'integration': 1.0},
        keywords=["stable", "integrated", "whole", "complete", "solid", "firm", "steady", "cohesive", "cohesion", "together", "united", "well", "happy", "content", "calm", "tranquil", "at peace", "balanced", "great", "wonderful"],
    ),
    "looping": dict(
        tags={},  # not scored by distance -- only ever selected via the basin override below
        keywords=["loop", "repeat", "repeated", "trapped", "cycle", "spin", "circle", "stuck", "spiral", "obsessed", "going in circles", "cannot stop thinking", "blocked"],
    ),
    "energy_high": dict(
        tags={'energy': 1.0},
        keywords=["energy", "intense", "electric", "vibrant", "strong", "spark", "power", "active", "excited", "energetic", "animated", "euphoric", "with energy"],
    ),
    "energy_low": dict(
        tags={'energy': 0.0},
        keywords=["off", "dim", "weak", "tired", "asleep", "silent", "extinguished", "exhausted", "slack", "sad", "unmotivated", "no desire", "no energy", "low"],
    ),
    "agency_high": dict(
        tags={'agency': 1.0},
        keywords=["decided", "advances", "will", "purpose", "determination", "resolute", "with direction", "motivated", "with desire", "focused"],
    ),
    "agency_low": dict(
        tags={'agency': 0.0},
        keywords=["drift", "lost", "direction", "passive", "floats", "no control", "indecisive", "compass", "random", "no destination", "unmotivated", "apathetic"],
    ),
    "grounded": dict(
        tags={'grounding': 1.0},
        keywords=["root", "anchor", "origin", "base", "ground", "rooted", "foundation", "solid", "firm", "affirm", "secure", "centered", "in control"],
    ),
    "untethered": dict(
        tags={'grounding': 0.0},
        keywords=["ghost", "scattered", "untethered", "free", "loose", "fog", "smoke", "no attachments", "floating", "no root", "confused", "disconnected", "gone", "in the clouds"],
    ),
    "volatile": dict(
        tags={'predictability': 0.0},
        keywords=["unstable", "chaos", "unpredictable", "stumbles", "erratic", "out of control", "random", "jumps", "nervous", "anxious", "restless", "altered"],
    ),
    "predictable": dict(
        tags={'predictability': 1.0},
        keywords=["rhythm", "regular", "constant", "predictable", "pattern", "cadence", "uniform", "exact", "precision", "no surprises", "normal", "always", "as always", "routine", "usual"],
    ),
    "memory_high": dict(
        tags={'memory': 1.0},
        keywords=["memory", "remembers", "persists", "past", "alive", "recalls", "lasts", "trace", "clear", "sharp", "intact", "remains", "nostalgic", "sentimental", "memories"],
    ),
    "memory_low": dict(
        tags={"memory": 0.0},
        keywords=["forgets", "erases", "blurred", "fuzzy", "fog", "fades", "hazy", "empty", "void", "no trace", "forgetfulness", "forgetful", "blank", "distracted"],
    ),
}

HARVESTED_VOCAB = [
    "pattern", 
    "form", 
    "order", 
    "fragment", 
    "break", 
    "crumble", 
    "broken", 
    "shattered", 
    "chaotically", 
    "without", 
    "warning", 
    "suddenly", 
    "apparent", 
    "instant", 
    "thousand", 
    "pieces", 
    "opposite directions", 
    "any", 
    "core", 
    "nucleus", 
    "structure", 
    "center", 
    "maintains", 
    "sustains", 
    "remains", 
    "integrated", 
    "firmly", 
    "calm", 
    "cracks", 
    "solidity", 
    "yield", 
    "purpose", 
    "place", 
    "pressure", 
    "pass", 
    "echo", 
    "spiral", 
    "repeat", 
    "turn", 
    "manage", 
    "escape", 
    "trapped", 
    "stopped", 
    "cease", 
    "other", 
    "time", 
    "exit", 
    "point", 
    "endlessly", 
    "inside", 
    "yes", 
    "same", 
    "arrive", 
    "no", 
    "part", 
    "closed circle", 
    "current", 
    "pulse", 
    "spark", 
    "vibrates", 
    "explodes", 
    "bursts", 
    "electric", 
    "intense", 
    "alive", 
    "intensely", 
    "force", 
    "brake", 
    "all", 
    "power", 
    "energy", 
    "overflowing", 
    "through", 
    "net", 
    "body", 
    "channel", 
    "extreme", 
    "other", 
    "rest", 
    "sign", 
    "impulse", 
    "off", 
    "extinguish", 
    "weakens", 
    "dim", 
    "weak", 
    "slowly", 
    "little", 
    "resistance", 
    "silence", 
    "shadow", 
    "almost", 
    "disappear", 
    "twilight", 
    "leave", 
    "trace", 
    "heat", 
    "engine", 
    "will", 
    "direction", 
    "advances", 
    "defines", 
    "stops", 
    "clear", 
    "decided", 
    "doubt", 
    "determination", 
    "waver", 
    "resolutely", 
    "threshold", 
    "goal", 
    "forward", 
    "look", 
    "back", 
    "drift", 
    "compass", 
    "floats", 
    "loses", 
    "silent", 
    "lost", 
    "mute", 
    "control", 
    "decide", 
    "nothing", 
    "random", 
    "currents", 
    "possibilities", 
    "destiny", 
    "fixed", 
    "side", 
    "know", 
    "toward", 
    "where", 
    "root", 
    "foundation", 
    "anchor", 
    "affirms", 
    "stable", 
    "solid", 
    "deeply", 
    "firmness", 
    "move", 
    "rooting", 
    "ground", 
    "origin", 
    "earth", 
    "deep", 
    "own", 
    "floor", 
    "ghost", 
    "fog", 
    "smoke", 
    "dissolves", 
    "extends", 
    "away", 
    "scattered", 
    "loose", 
    "quietly", 
    "attachments", 
    "floating", 
    "weight", 
    "beyond", 
    "needle", 
    "compass", 
    "stumbles", 
    "jumps", 
    "loses control", 
    "unstable", 
    "erratic", 
    "chaotic", 
    "abruptly", 
    "prior", 
    "unpredictably", 
    "two", 
    "states", 
    "law", 
    "randomly", 
    "repeat", 
    "never", 
    "rhythm", 
    "clock", 
    "cadence", 
    "mark", 
    "step", 
    "regular", 
    "constant", 
    "uniformly", 
    "vary", 
    "precision", 
    "surprises", 
    "accuracy", 
    "time", 
    "cycle", 
    "get out", 
    "memory", 
    "trace", 
    "persists", 
    "lasts", 
    "clearness", 
    "fade", 
    "year", 
    "trace", 
    "surface", 
    "layer", 
    "detail", 
    "background", 
    "all", 
    "forgets", 
    "erases", 
    "image", 
    "name", 
    "blurred", 
    "gradually", 
    "remedy", 
    "always", 
    "forgetfulness", 
    "void", 
    "more", 
    "model", 
    "internal", 
    "nodes", 
    "map", 
    "diverge", 
    "separate", 
    "misaligned", 
    "dimensions", 
    "converge", 
    "each", 
    "one", 
    "agreement", 
    "different", 
    "axes", 
    "meet", 
    "own", 
    "reading", 
    "convergence", 
    "align", 
    "coincide", 
    "unified", 
    "aligned", 
    "complete", 
    "discrepancy", 
    "unanimous", 
    "axis", 
    "all", 
    "margin", 
    "difference", 
    "learning", 
    "preference", 
    "trained", 
    "bias", 
    "learned", 
    "pulls", 
    "pushes", 
    "tilts", 
    "trajectory", 
    "marked", 
    "insistence", 
    "ambiguity", 
    "direction", 
    "concrete", 
    "experience", 
    "result", 
    "better", 
    "learned", 
    "history", 
    "enough", 
    "way", 
    "couple", 
    "lean", 
    "particular", 
    "coordinates", 
    "lack", 
    "steps",
]


# ============================================ SELF-MODEL INTROSPECTION
# Separate from CLUSTERS (mood, tagged on the 8 named/interpretable axes).
# This bank is tagged on two quantities that come straight out of the raw
# self-model M_t instead of the named mood axes:
#   spread -- how tightly the 16 nodes' internal self-models currently agree
#             with each other (derived from K_t, already computed every step
#             from actual variance in M_t -- see Mind.step)
#   pull   -- the magnitude of latent_want_ema, the Hebbian credit-assignment
#             signal TRAINED over M_t's 8 unlabeled coordinates (see
#             Mind.latent_want_ema / learn_desire). This is not hand-set: it's
#             whatever this specific Mind, on this specific run, has actually
#             learned about its own raw self-model dimensions from its own
#             trajectory. Early in a run it's near zero (untrained); it grows
#             and can even reverse as the Mind accumulates steps.
# Selection + wording generation both go through the exact same mechanism as
# CLUSTERS (pick-by-distance, then speak_cluster's independent slot sampling)
# -- just pointed at self-model-derived axes instead of mood axes.
# Selection still goes through the same pick-by-distance mechanism as
# CLUSTERS. Wording no longer comes from a slot bank at all -- see
# geometric_generate below -- these entries now carry only the axis targets
# used to pick a LABEL (for logging/basin-alarm purposes), not any text.
SELF_CLUSTERS = {
    "self_diffuse": dict(tags={"spread": 1.0}),
    "self_converged": dict(tags={"spread": 0.0}),
    "self_pulled": dict(tags={"pull": 1.0}),
    "self_untrained": dict(tags={"pull": 0.0}),
}

def self_model_axes(mind, state):
    """All eight quantities come from real computed state, not invented for
    this bank: spread from K_t (already in state['K'] every step -- literal
    variance of M_t across the 16 nodes), pull from the norm of
    latent_want_ema (the actual trained Hebbian signal, see above), the
    four workspace quantities from that step's global-workspace competition
    (see Mind.step's GLOBAL WORKSPACE section), and meta_confidence/
    meta_volatility from that step's RECURRENT METACOGNITION section --
    whether this step's self-model matched what the Mind's own recent
    self-reports had come to expect of it, and whether that expectation
    itself has been stable or swinging. All eight already clipped to
    [0,1] in Mind.step."""
    spread = float(np.clip(1 - state["K"], 0, 1))
    pull = float(np.tanh(np.linalg.norm(mind.latent_want_ema)))
    ignition = float(np.clip(state.get("Ignition", 0.0), 0, 1))
    continuity = float(np.clip(state.get("Continuity", 0.0), 0, 1))
    workspace_grounding = float(np.clip(state.get("WorkspaceGrounding", 0.0), 0, 1))
    workspace_novelty = float(np.clip(state.get("WorkspaceNovelty", 0.0), 0, 1))
    meta_confidence = float(np.clip(state.get("MetaConfidence", 0.0), 0, 1))
    meta_volatility = float(np.clip(state.get("MetaVolatility", 0.0), 0, 1))
    return dict(spread=spread, pull=pull, ignition=ignition, continuity=continuity,
                workspace_grounding=workspace_grounding, workspace_novelty=workspace_novelty,
                meta_confidence=meta_confidence, meta_volatility=meta_volatility)

def pick_self_cluster(mind, state):
    """Same adaptive rank-normalization reasoning as Mind.adaptive_normalize
    (spread/pull aren't naturally scaled to a comparable [0,1] range), kept
    on the Mind instance so it persists and adapts across steps the same way
    axis_hist does for the mood axes."""
    raw = self_model_axes(mind, state)
    if not hasattr(mind, "_self_axis_hist"):
        mind._self_axis_hist = {k: [] for k in raw}  # NEW: generic over whatever self_model_axes returns,
                                                       # rather than hardcoding spread/pull -- so adding
                                                       # ignition/continuity (or anything else later) doesn't
                                                       # require touching this init line again
    norm = {}
    for k, v in raw.items():
        h = mind._self_axis_hist[k]
        h.append(v)
        if len(h) > AXIS_WINDOW:
            h.pop(0)
        norm[k] = float(np.mean(np.array(h) <= v)) if len(h) >= 5 else 0.5
    best_name, best_score = None, None
    for name, c in SELF_CLUSTERS.items():
        score = np.mean([abs(norm[axis] - target) for axis, target in c["tags"].items()])
        if best_score is None or score < best_score:
            best_name, best_score = name, score
    return best_name

BASIN_ALARM = 0.12  # above this, the "looping" cluster overrides everything else -- a real alarm state

def normalize_state(state):
    """Map the Mind's real, already-interpretable state variables onto the axes
    the clusters are tagged on. No random projection, no learned embedding --
    these are the same named quantities Mind.step already computes and returns."""
    return dict(
        coherence=float(np.clip(state["C"], 0, 1)),
        integration=float(np.clip(state["Phi"], 0, 1)),
        basin=float(np.clip(state["Basin"], 0, 1)),
        energy=float(np.clip(state["E"], 0, 1)),
        agency=float(np.clip(state["U"], 0, 1)),
        grounding=float(np.clip(state["Gmean"], 0, 1)),
        predictability=float(np.clip(state["P"], 0, 1)),
        memory=float(np.clip(state["MemCont"], 0, 1)),
    )

def pick_cluster(norm, forced_name=None):
    if forced_name is not None:
        return forced_name
    if norm["basin"] > BASIN_ALARM:
        return "looping"
    best_name, best_score = None, None
    for name, c in CLUSTERS.items():
        if not c["tags"]:
            continue
        score = np.mean([abs(norm[axis] - target) for axis, target in c["tags"].items()])
        if best_score is None or score < best_score:
            best_name, best_score = name, score
    return best_name

def _strip_accents(s):
    import unicodedata
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

# speak_cluster / speak_self_report (fixed subject/verb/adj/adv/prep_obj slot
# sampling) removed entirely -- see geometric_generate, further down, for what
# replaced them: real word-by-word generation, no slots, no whole sentences
# picked from a pool.

# ============================================ SEMANTIC COMPREHENSION (grounding)
# v4's route_prompt() was substring keyword counting: "does this word literally
# appear in the prompt" for each cluster's fixed keyword list. That has two real
# failures -- (1) it can only ever hit clusters, which are MOODS, not answerable
# content, so any question that isn't "how do you feel" gets a mood-shaped
# non-answer; and (2) exact substring match has no notion of "close enough,"
# so near-miss phrasing (synonyms, conjugations, word order) that a human would
# recognize as the same question just misses entirely.
#
# This section fixes both by reusing the Mind's own grounding mechanism instead
# of inventing a new one. Internally, Mind.step() computes
#   G_i = exp(-lam * ||S_t - g_M||^2)
# -- a Gaussian-kernel distance between the environment's current encoded
# location (S_t) and each node's self-model location (g_M) in the SAME space.
# That is: "how close is what I'm sensing to what I already represent," turned
# into a [0,1] confidence by treating distance as evidence against a match.
#
# We apply the identical idea to language. embed_text() places a prompt at a
# location in a small CONCEPT_DIM-dimensional space via a stable hashing-trick
# bag-of-words (deterministic: the same word always lands in the same place
# this run, no external model, no training). concept_anchor() places a known
# concept or mood at a location in that same space, built from a hand-authored
# set of phrases that mean it -- the "location" is real, computed from actual
# text, not assigned by hand. semantic_route() then scores the prompt's
# location against every known anchor with the SAME exp(-lam*d^2) kernel Mind
# already uses, and if the best score is still below CONCEPT_GROUNDING_THRESHOLD,
# it reports no match rather than force one -- exactly the honesty property
# grounding already has internally (a G_i near 0 means "not represented here,"
# not "represented as something arbitrary").
#
# Be precise about what this is NOT: it is still a hand-authored, closed
# vocabulary -- embed_text has no notion of synonymy beyond literal word-hash
# collision, so "cuantos nodos" and "cuanta gente" will NOT be recognized as
# related unless a seed phrase spells that out. What's new relative to v4 is
# (a) partial word overlap now contributes partial, continuous similarity
# instead of a binary substring hit, (b) concepts and moods now compete in ONE
# shared space instead of two separate ad-hoc mechanisms (WILL_KEYWORDS +
# route_prompt), and (c) the match confidence is a real number you can
# threshold and report, not an unscored boolean. It is a better-grounded
# keyword system, not language understanding -- there is still no path here
# from novel phrasing to a novel answer the system wasn't given.
CONCEPT_DIM = 64
NGRAM = 3                           # character n-gram length (see embed_text)
GROUND_LAM = 3.0                    # kernel width for concept-space grounding (separate from Mind's internal lam)
                                     # WAS 1.0 -- raised so _topic_relevance's exp(-LAM*d^2) falls off much
                                     # faster with distance from the prompt's own embedding, i.e. a candidate
                                     # sentence that's drifted even a moderate distance from what was actually
                                     # asked now scores much closer to zero instead of still getting partial
                                     # credit. This is the "penalize drift heavily" knob.
CONCEPT_GROUNDING_THRESHOLD = 0.0  # always route to the best-scoring concept/mood, even a weak match

# Purely functional (no meaning attached): dropped before embedding so that
# grammatical glue words -- which appear in nearly every seed phrase across
# every concept -- don't dominate every anchor's direction and collapse all
# the anchors toward the same point. This is a closed, hand-written list, same
# spirit as every other hand-authored list in this file.
STOPWORDS = {"that", "you", "your", "have", "has", "are", "is", "of", "the", "a", "an",
             "in", "on", "at", "to", "for", "with", "and", "or", "but", "as", "by",
             "it", "its", "this", "that", "these", "those", "be", "been", "being"}

_PUNCT_RE = re.compile(r"[¿?¡!.,;:\"'()\[\]{}]")

def _content_words(s):
    words = _strip_accents(_PUNCT_RE.sub("", s.lower())).split()
    kept = [w for w in words if w not in STOPWORDS]
    return kept if kept else words  # a phrase that's ALL stopwords still needs something to embed

def _char_grams(word):
    padded = f"_{word}_"
    return [padded[i:i + NGRAM] for i in range(len(padded) - NGRAM + 1)] or [padded]

def _gram_hash(gram):
    """Stable hash -> (dimension index, sign). hashlib, not Python's built-in
    hash() (salted per-process by default) -- the same n-gram must land in the
    same place every time, since anchors built once at import time need to
    stay meaningful for every prompt this process ever embeds."""
    h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
    return h % CONCEPT_DIM, (1.0 if (h // CONCEPT_DIM) % 2 == 0 else -1.0)

# ---- IDF weighting -----------------------------------------------------
# Plain hashed bag-of-words was tried first and failed a basic sanity check:
# a seed phrase barely resembled its OWN concept's anchor once averaged with
# its sibling phrasings, because Spanish inflection (cuantos/cuantas,
# tiene/tienes) sends morphological variants of the same word to unrelated
# hash buckets with zero overlap. Character n-grams fix that (word stems
# share most of their trigrams across conjugation/gender). But a second,
# separate problem remained: generic content words that show up across many
# concepts' seed phrases (and mood clusters' keyword lists) drown out the
# words that actually distinguish one concept from another. The standard fix
# for that -- same idea TF-IDF uses for document search -- is to weight each
# n-gram by how RARE it is across the whole known vocabulary: a trigram that
# appears in only one concept's phrases is much more informative for
# matching than one that appears in all of them. df/idf below are computed
# once, from the concept bank + mood clusters themselves (the only text this
# system actually knows), not from any external corpus.
def _build_idf(docs):
    df = {}
    for doc_phrases in docs:
        grams_in_doc = set()
        for phrase in doc_phrases:
            for w in _content_words(phrase):
                grams_in_doc.update(_char_grams(w))
        for g in grams_in_doc:
            df[g] = df.get(g, 0) + 1
    n_docs = len(docs)
    idf = {g: math.log((n_docs + 1) / (c + 1)) + 1.0 for g, c in df.items()}
    default_idf = math.log(n_docs + 1) + 1.0  # an n-gram never seen in the known vocabulary --
    return idf, default_idf                    # treated as maximally distinctive, not zero

def embed_text(s, idf=None, default_idf=1.0):
    """Locate a phrase in R^CONCEPT_DIM: character n-grams of its content
    words, each hashed to a dimension+sign and weighted by rarity (idf) before
    summing, then L2-normalized. idf=None (used only while bootstrapping the
    idf table itself) falls back to uniform weight 1.0 per gram."""
    v = np.zeros(CONCEPT_DIM)
    for w in _content_words(s):
        for g in _char_grams(w):
            idx, sign = _gram_hash(g)
            weight = 1.0 if idf is None else idf.get(g, default_idf)
            v[idx] += sign * weight
    n = np.linalg.norm(v)
    return v / n if n > EPS else v

def concept_anchor(phrases, idf=None, default_idf=1.0):
    """A concept's location = the (re-normalized) mean of its seed phrases'
    locations -- the anchor is computed from real text, not picked by hand."""
    v = np.mean([embed_text(p, idf, default_idf) for p in phrases], axis=0)
    n = np.linalg.norm(v)
    return v / n if n > EPS else v

def _blend(vec_a, weight_a, vec_b, weight_b):
    """Weighted average of two vectors already living in the same fixed
    embed_text geometry, re-normalized. Used to bias generation toward BOTH a
    topic (a concept's own anchor -- what was asked about) and the live
    qualia vector (what state it's actually in right now) at once."""
    v = weight_a * vec_a + weight_b * vec_b
    n = np.linalg.norm(v)
    return v / n if n > EPS else v

GENERIC_NONANSWER_WORDS = {"today", "the window", "the coffee", "the cold wind", "on time"}
ANSWER_TYPE_BONUS_WEIGHT = 0.5  # deliberately smaller than DRIFT_PENALTY_WEIGHT (which stays infinite for
                                # genuine failures like echo/repetition) -- this is a real steering signal
                                # toward satisfying the question, not a hard requirement, since with only a
                                # handful of indicator words per type it will often correctly find nothing
                                # to reward, and that must never disqualify an otherwise-good candidate

def detect_expected_answer_type(prompt_text):
    """Rule-based question classification (Li & Roth style) -- returns the
    expected answer-type key, or None if the prompt doesn't look like a
    wh-question at all (e.g. a statement, or a yes/no question, neither of
    which this simple ruleset attempts to handle -- a real system would add
    dedicated classes for those too, but this is deliberately the minimum
    viable version of the actual mechanism, not a complete taxonomy)."""
    if not prompt_text:
        return None
    norm = _strip_accents(prompt_text.strip().lower().lstrip("¿").rstrip("?!"))
    words = norm.split()
    for phrase, atype in WH_TYPE_MAP:
        phrase_norm = _strip_accents(phrase)
        phrase_words = phrase_norm.split()
        if words[:len(phrase_words)] == phrase_words:
            return atype
    return None

def _answer_type_bonus(text, expected_type):
    """Reward for a candidate containing at least one word from the expected
    answer-type's indicator set (see ANSWER_TYPE_WORDS) -- this is the
    lexically-constrained-decoding half of the mechanism: topic_relevance
    rewards TALKING ABOUT the right subject, this rewards actually containing
    the KIND of content the question grammatically demands (a number for
    'cuanto', a place for 'donde', etc.), which is a different property no
    prior scoring term in this file has ever measured."""
    if expected_type is None or expected_type not in ANSWER_TYPE_WORDS:
        return 0.0
    words = set(_strip_accents(text.strip().rstrip(".?!").lower()).split())
    return ANSWER_TYPE_BONUS_WEIGHT if words & ANSWER_TYPE_WORDS[expected_type] else 0.0

def _lexical_overlap(a, b):
    """Jaccard word overlap -- the dissimilarity signal D(y,Y) in the DBS
    objective argmax[P(y) + lambda*D(y,Y)], in the simple lexical space this
    file already uses elsewhere (no new embedding machinery needed)."""
    wa = set(_strip_accents(a.strip().rstrip(".?!").lower()).split())
    wb = set(_strip_accents(b.strip().rstrip(".?!").lower()).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)

INTERNAL_COHERENCE_MIN = 0.05   # NEW: hard floor on internal_coherence (fit to the Mind's OWN live
                                 # C/Phi/E/U/Gmean/P/MemCont state, via state_vec/qualia_vector) -- a
                                 # branch below this is disqualified outright, same -inf mechanism
                                 # DRIFT_PENALTY_WEIGHT already uses for echo/repetition, rather than just
                                 # being outscored. Set from this file's own observed g= range across many
                                 # real runs (0.13-0.30 typical, occasional 0.00 on totally ungrounded
                                 # branches) -- 0.05 cuts the genuine noise floor without being so strict
                                 # every branch in a bad batch gets rejected and best_text falls back to None.

_ALL_ANSWER_SEEDS = None  # lazily built once: every hand-authored answer_seed string across every
                           # CONCEPT_BANK entry, pre-tokenized to a word-set, so a seed-proximity check
                           # doesn't re-tokenize the whole seed bank on every single candidate

def _seed_word_sets():
    global _ALL_ANSWER_SEEDS
    if _ALL_ANSWER_SEEDS is None:
        sets = []
        for concept in CONCEPT_BANK.values():
            for seed in concept.get("answer_seeds", []):
                w = set(_strip_accents(seed.strip().rstrip(".?!").lower()).split())
                if w:
                    sets.append(w)
        _ALL_ANSWER_SEEDS = sets
    return _ALL_ANSWER_SEEDS

SEED_PROXIMITY_MAX = 0.55  # NEW: max allowed Jaccard overlap against ANY single hand-authored answer_seed.
                            # Guards against a branch that's really just reciting a seed back near-verbatim
                            # (a 5-6 word sentence sharing 3+ words with a specific seed crosses this) --
                            # distinct from _repetition_penalty, which only looks at THIS candidate's own
                            # internal repeats and recently-SPOKEN lines, never the seed bank itself.

def _seed_proximity(text):
    """Highest Jaccard overlap between this candidate and any single hand-authored answer_seed, across
    every concept -- NOT a similarity to the concept's THEME (that's what relevance/internal_coherence
    already reward), specifically a check against reciting one specific authored sentence back."""
    wt = set(_strip_accents(text.strip().rstrip(".?!").lower()).split())
    if not wt:
        return 0.0
    best = 0.0
    for ws in _seed_word_sets():
        ov = len(wt & ws) / len(wt | ws) if (wt or ws) else 0.0
        if ov > best:
            best = ov
    return best

def _repetition_penalty(text, recent_lines):
    """Two repetition failure modes, both maximally disqualified the same way
    _nonanswer_penalty's echo check is (see DRIFT_PENALTY_WEIGHT):
      (1) INTRA-sentence -- the running-context-vector change to geometric_generate
          can create a self-reinforcing loop where a short phrase keeps getting
          re-selected because it's already baked into its own context vector
          ("un poco a un poco a un poco"). Flagged by any word repeated past
          MAX_WORD_REPEATS, or any bigram occurring more than once.
      (2) CROSS-sentence -- free-running generation had zero reranking before
          this change, so nothing ever stopped a new draw from being a
          near-exact repeat of what was JUST said two steps ago ("El mercado."
          x5). Flagged by a near-exact match against any of the last
          RECENT_LINES_WINDOW lines this Mind has actually spoken."""
    stripped = text.strip().rstrip(".?!").lower()
    words = stripped.split()
    if not words:
        return 0.0
    penalty = 0.0
    counts = Counter(words)
    if counts and max(counts.values()) > MAX_WORD_REPEATS:
        penalty += 1.0
    bigrams = list(zip(words, words[1:]))
    if bigrams and len(bigrams) != len(set(bigrams)):
        penalty += 1.0
    norm_text = _strip_accents(stripped)
    for prior in recent_lines:
        if norm_text == _strip_accents(prior.strip().rstrip(".?!").lower()):
            penalty += 1.0
            break
    return min(penalty, 1.0)

_RECENT_LINES = []  # rolling history of actually-spoken lines this run, reset at the top of run();
                    # deliberately transient (not persisted to the DB) -- its only job is stopping THIS
                    # run from looping on itself, not remembering across separate invocations
_LINE_USE_COUNT = Counter()  # ROOT FIX: every fix so far (echo penalty, intra-sentence loop-blocking,
                    # cross-sentence exact-match check, length bonus) targeted a SYMPTOM of the same
                    # underlying cause -- a sentence that's structurally easy to pass every filter gets
                    # reused constantly, and _RECENT_LINES' 12-line window let it fall out of range and
                    # become "safe" again a dozen steps later. This tracks how many times EVERY sentence
                    # has ever been spoken this run, uncapped, no window, no forgetting -- so a favorite
                    # sentence gets structurally harder to pick again every single time it's used, forever,
                    # not just for the next 12 lines.

def _overuse_penalty(text):
    """Root-cause fix for the recurring 'one sentence dominates' pattern (echo
    -> loop -> identical-forever -> length-favored-sentence -- all the same
    underlying issue at different scales): penalize a candidate in proportion
    to how many times THIS EXACT sentence has already been spoken this run,
    uncapped and never forgotten, rather than only checking against a recent
    window. Quadratic in use count (not flat-linear) on purpose: a first
    reuse should barely register, but a sentence heading toward its 10th use
    needs to be actively expelled, not just mildly discouraged at the same
    rate as its 2nd -- linear cost at a small step (as first tried) still let
    a structurally-favored sentence out-accumulate length_bonus/relevance
    over hundreds of steps, because the cost per use never grew faster than
    the sentence kept getting picked."""
    norm = _strip_accents(text.strip().rstrip(".?!").lower())
    n = _LINE_USE_COUNT.get(norm, 0)
    return OVERUSE_PENALTY_STEP * (n ** 2)

def _remember_line(text):
    _RECENT_LINES.append(text)
    if len(_RECENT_LINES) > RECENT_LINES_WINDOW:
        _RECENT_LINES.pop(0)
    _LINE_USE_COUNT[_strip_accents(text.strip().rstrip(".?!").lower())] += 1

def _nonanswer_penalty(text, prompt_text=None):
    """Flags the failure modes seen in real runs: (1) a short generic filler
    sentence ('Today.', 'The coffee.') that's grammatical but says nothing
    responsive, (2) the router echoing the prompt back verbatim as a flat
    statement, and (3) the router reciting a RUN of the prompt's own words
    (see _echo_run_penalty) before trailing off, rather than answering it.
    Returns a value in [0,1] to subtract from relevance -- see
    DRIFT_PENALTY_WEIGHT below for how heavily that subtraction actually
    counts against a candidate."""
    stripped = text.strip().rstrip(".?!").lower()
    words = stripped.split()
    penalty = 0.0
    if len(words) <= TOPIC_MIN_WORDS:
        penalty += 0.4 * (TOPIC_MIN_WORDS - len(words) + 1) / TOPIC_MIN_WORDS
    if _strip_accents(stripped) in GENERIC_NONANSWER_WORDS:
        penalty += 0.3
    if prompt_text is not None:
        prompt_stripped = _strip_accents(prompt_text.strip().rstrip(".?!¿¡").lower())
        if _strip_accents(stripped) == prompt_stripped:
            penalty += TOPIC_ECHO_PENALTY
        penalty += _echo_run_penalty(text, prompt_text)
    return min(penalty, 1.0)

DRIFT_PENALTY_WEIGHT = float("inf")  # "inf" -- Python's own spelling of infinity: a number, spelled with
                                      # letters, and the largest one there is. Multiplying the [0,1] nonanswer/
                                      # echo penalty by this turns ANY nonzero penalty into -inf outright, so a
                                      # candidate that drifts or echoes doesn't just lose points against one
                                      # that doesn't -- it CANNOT outrank it, full stop, no matter how far off
                                      # every other candidate in the batch happens to be.

INTERNAL_COHERENCE_WEIGHT = 0.6  # weight on how well a branch's own words land near the Mind's ACTUAL raw
                                 # internal state vector (state_vec, unblended qualia_vector output) -- a
                                 # distinct signal from topic_relevance (which only ever measures fit to the
                                 # PROMPT) and from query itself (which is already a blend of anchor/topic/
                                 # entity, so scoring against query would just reward candidates for
                                 # agreeing with their own starting point, not for genuinely reflecting what
                                 # the mind's numbers currently say).

FLUENCY_FALLBACK = -10.0  # FIX (word-salad diagnosis): used only when geometric_generate produced no real
                          # output (out was empty -> "...", _LAST_GEN_FLUENCY left as None) -- a strongly
                          # negative but finite stand-in so that degenerate case scores clearly worse than
                          # almost any real (however broken) candidate, without using -inf and risking the
                          # same 0*inf=nan trap the drift-penalty fix elsewhere in this function already
                          # had to work around once.
FLUENCY_WEIGHT = 0.4  # FIX (word-salad diagnosis, Sept 2026): weight on _LAST_GEN_FLUENCY (mean per-token
                      # log-prob under the model's OWN raw distribution -- see that global's definition).
                      # Every other term above rewards a candidate for being about the right THING (topic,
                      # internal state, answer type); nothing previously rewarded it for being a plausible
                      # WORD SEQUENCE, which is exactly the gap that let two candidates with identical
                      # topic words but wildly different grammaticality score identically. 0.4 is a first
                      # pass, not a tuned value the way INTERNAL_COHERENCE_WEIGHT(0.6)/DIVERSITY_WEIGHT
                      # (0.35) are -- log-probs here run roughly in the same ballpark as those other raw
                      # terms on this vocab/model size, so it starts at a comparable weight rather than
                      # dominating or being drowned out, but this deserves the same kind of empirical
                      # tuning pass the other weights already got, not just a one-run gut check.

def _select_best(query, rng, entity_vec, entity_word, entity_weight,
                  topic_vec=None, prompt_text=None, best_of=None, state_vec=None,
                  bigram=None, unigram=None, mind=None):
    """BRANCH SELECTION: each of the `best_of` independent draws below is a
    full candidate BRANCH (its own complete word-by-word generation run, not
    a shared-prefix continuation), scored and compared as a whole rather than
    token-by-token -- this is what makes the reranking below function as
    branch selection rather than ordinary greedy decoding. Used for EVERY
    generated line, not only ones answering a real prompt (previously
    free-running generation took a single ungated draw with no reranking at
    all, which is exactly why a self-reinforcing repetition loop could only
    ever be noticed in the log, never actually avoided).

    Each branch is scored on FOUR independent signals:
      relevance          -- fit to the actual prompt (topic_vec), only when
                             there is one
      internal_coherence  -- NEW: fit to the Mind's own raw internal state
                             (state_vec -- see qualia_vector) -- a branch that
                             happens to be very on-topic but numerically
                             unrelated to what C/Phi/E/U/Gmean/P/MemCont
                             actually say right now scores worse here than one
                             that's a little less topical but genuinely
                             reflects the live state, and vice versa; this is
                             a genuinely separate axis from topic_relevance,
                             not a restatement of it
      length_bonus/overuse/diversity_penalty -- unchanged (see below)
    penalty (short/generic-filler, echo, repetition) applies unconditionally
    at DRIFT_PENALTY_WEIGHT, so any branch that repeats itself or a
    just-spoken line is disqualified outright regardless of how well it
    scores on the other three."""
    if best_of is None:
        best_of = TOPIC_BEST_OF if topic_vec is not None else FREE_BEST_OF
    if _BRANCH_OVERRIDE_BEST_OF is not None:  # NEW: see module-level comment on _BRANCH_OVERRIDE_BEST_OF
        best_of = _BRANCH_OVERRIDE_BEST_OF
    gen_bigram = _BIGRAM if bigram is None else bigram  # NEW: caller (e.g. a concept answer) can pass its
    gen_unigram = _UNIGRAM if unigram is None else unigram  # own concept-biased tables (see
    # _concept_biased_tables) so word-by-word sampling actually has on-topic vocabulary to draw from;
    # defaults to the plain global tables so every other existing call site is unaffected
    expected_type = detect_expected_answer_type(prompt_text)  # NEW: rule-based question classification --
    # computed once per batch, not per-candidate, since it depends only on the prompt
    best_text, best_score = None, float("-inf")
    fallback_text, fallback_score = None, float("-inf")  # NEW: candidates that clear the drift/repetition
    # filter (never relaxed -- that's a genuine failure mode) but fail the NEW coherence/seed gates. If
    # literally every branch in the batch fails coherence or seed-proximity (a real risk at small best_of,
    # now that two more gates sit on top of the pre-existing drift one), fall back to the least-bad of
    # THESE rather than returning None -- a candidate that's merely below the coherence floor is still a
    # far better answer than crashing or emitting nothing.
    best_query_draw = None  # NEW: tracks which draw's own query vector actually won, so the write-back
                             # below reflects the candidate that gets SPOKEN, not just whichever candidate
                             # happened to be generated last in this loop (geometric_generate sets
                             # _LAST_TF_MIND_BIAS as a side effect on every draw, winning or not)
    seen_texts = set()  # within THIS batch, an exact duplicate draw is skipped from consideration entirely
    # SCALE FIX (large best_of, e.g. tens of thousands of branches): the ORIGINAL diversity_penalty compared
    # every new candidate against every previously-picked one in this batch ("picked" list + generator
    # expression) -- O(best_of^2) overall. At best_of in the tens of thousands that's on the order of a
    # billion pairwise comparisons and never finishes. Replaced with a running word-frequency counter
    # (Counter over every word seen across every picked candidate so far, updated incrementally) so each
    # new candidate's diversity score costs O(len(candidate)) to compute, not O(len(picked)) -- O(best_of)
    # overall instead of O(best_of^2). Same intent (penalize a candidate whose words keep reappearing
    # across this batch) via a frequency-weighted overlap instead of a max-pairwise-Jaccard one.
    picked_word_counts = Counter()
    picked_count = 0
    n_gated_coherence = 0   # NEW: counts for a post-loop summary, not per-candidate logging (would be
    n_gated_seed = 0        # far too much output at tens of thousands of branches)
    n_gated_drift = 0
    for _ in range(best_of):
        query_draw = normalize(query + rng.normal(0, 0.08, query.shape))  # small per-draw jitter so
        # independent branches start from slightly different points instead of the same exact vector
        candidate = geometric_generate(query_draw, gen_bigram, gen_unigram, rng,
                                        entity_vec=entity_vec, entity_word=entity_word,
                                        entity_weight=entity_weight, mind=mind)
        # FIX (word-salad diagnosis): geometric_generate just set this as a side effect (same pattern as
        # _LAST_TF_MIND_BIAS, read a few lines below) -- capture it for THIS candidate immediately, before
        # the next draw's call overwrites it.
        fluency = _LAST_GEN_FLUENCY if _LAST_GEN_FLUENCY is not None else FLUENCY_FALLBACK
        norm_cand = _strip_accents(candidate.strip().rstrip(".?!").lower())
        if norm_cand in seen_texts:
            continue
        seen_texts.add(norm_cand)
        relevance = _topic_relevance(candidate, topic_vec) if topic_vec is not None else 0.0  # THE
        # prompt-match term: topic_vec is embed_text(prompt) at every real-prompt call site (see run()),
        # so this already IS "how well does this branch match the prompt", not a separate thing to add
        internal_coherence = _topic_relevance(candidate, state_vec) if state_vec is not None else 0.0
        # NEW: hard coherence gate -- a branch that doesn't even clear the Mind's own live-state floor is
        # disqualified before any other scoring, same -inf mechanism as the existing drift check
        coherence_gate = state_vec is not None and internal_coherence < INTERNAL_COHERENCE_MIN
        if coherence_gate:
            n_gated_coherence += 1
        # NEW: seed-proximity gate -- a branch that's really just reciting one specific hand-authored
        # answer_seed back (rather than genuinely generating) is disqualified the same way
        seed_prox = _seed_proximity(candidate)
        seed_gate = seed_prox > SEED_PROXIMITY_MAX
        if seed_gate:
            n_gated_seed += 1
        penalty = _nonanswer_penalty(candidate, prompt_text) + _repetition_penalty(candidate, _RECENT_LINES)
        if penalty > 0:
            n_gated_drift += 1
        length_bonus = LENGTH_BONUS_WEIGHT * len(candidate.split())  # standard length-normalization fix
        overuse = _overuse_penalty(candidate)  # uncapped, never-forgotten per-sentence use count, this run
        cand_words = set(_strip_accents(norm_cand).split())
        # O(len(candidate)) diversity term -- see SCALE FIX comment above -- frequency-weighted overlap
        # with every word seen across every candidate already picked in this batch, normalized by how many
        # candidates have been picked so far (an all-new word in a fresh batch scores 0; a word that's
        # appeared in every single prior candidate scores close to 1)
        diversity_penalty = DIVERSITY_WEIGHT * (
            (sum(picked_word_counts.get(w, 0) for w in cand_words) / (len(cand_words) * picked_count))
            if cand_words and picked_count else 0.0)
        answer_type_bonus = _answer_type_bonus(candidate, expected_type)  # NEW: does this candidate contain
        # the KIND of content the question grammatically demands, not just talk about the right subject
        genericness = _genericness_penalty(candidate)  # NEW: see GENERIC_PENALTY_WEIGHT -- how compatible
        # this candidate is with almost ANY anchor, not just the one it was drawn for (the 'Siento.'/
        # 'I don't know' failure mode)
        picked_word_counts.update(cand_words)
        picked_count += 1
        # FIX: DRIFT_PENALTY_WEIGHT is float("inf"), and penalty is exactly 0.0 for every clean candidate --
        # inf * 0.0 is IEEE-754 nan, not 0.0, so the old "- DRIFT_PENALTY_WEIGHT * penalty" term silently
        # poisoned the score of every UN-penalized candidate to nan. nan never compares greater than anything
        # (not even -inf), so "score > best_score" was False for every candidate after the first, and the
        # "best_text is None" fallback meant the first draw of the batch always won regardless of quality --
        # relevance/coherence/answer_type_bonus were all computed and then silently ignored. Gate the
        # infinite penalty on penalty > 0 instead of multiplying by it, so a clean candidate gets a real,
        # comparable score and a disqualified one still gets exactly -inf as intended.
        if penalty > 0:
            continue  # drift/repetition disqualification is NEVER relaxed, no fallback for this one
        score = (relevance + INTERNAL_COHERENCE_WEIGHT * internal_coherence
                 + length_bonus - overuse - diversity_penalty + answer_type_bonus
                 - GENERIC_PENALTY_WEIGHT * genericness
                 + FLUENCY_WEIGHT * fluency)  # FIX (word-salad diagnosis): the only term in this score that
        if coherence_gate or seed_gate:
            if fallback_text is None or score > fallback_score:
                fallback_text, fallback_score = candidate, score
            continue  # still disqualified from WINNING outright, but now recorded as a fallback
        if best_text is None or score > best_score:
            best_text, best_score, best_query_draw = candidate, score, query_draw
    if best_text is None and fallback_text is not None:
        # NEW: every branch in this batch failed the coherence/seed gates -- use the least-bad one rather
        # than propagating None (which previously crashed the caller outright, see mind.update_discourse_entity)
        best_text, best_score = fallback_text, fallback_score
        n_gated_coherence = max(0, n_gated_coherence - 1)  # the fallback candidate itself no longer counts
        # as "rejected" in the summary line below, since it ended up being used after all
    if best_of >= 500:  # NEW: only worth a summary line at branch counts where per-candidate logging would
        print(f"    [branch selection: {best_of} branches -- {n_gated_coherence} below coherence floor, "
              f"{n_gated_seed} too close to a seed, {n_gated_drift} drift/repetition-disqualified, "
              f"winner score={best_score:.3f}]")
    if best_text is not None and best_query_draw is not None:
        # NEW: two-way grounding -- recompute the write-back for the candidate that actually WON, not
        # whichever one geometric_generate happened to generate last inside the loop above (see comment
        # on best_query_draw). Overwrites whatever _LAST_TF_MIND_BIAS the loop left behind.
        global _LAST_TF_MIND_BIAS
        winner_toks = _tokenize_natural(best_text)
        winner_ids = [_TF_WORD2ID.get(w, _TF_WORD2ID[TF_UNK]) for w in ([TF_BOS] + winner_toks)][-TF_MAX_LEN:]
        winner_sv = torch.tensor(normalize(best_query_draw), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        _LAST_TF_MIND_BIAS = _TF_MODEL.mind_bias(
            torch.tensor([winner_ids], dtype=torch.long, device=DEVICE), winner_sv)
    return best_text

_generate_on_topic = _select_best  # kept as an alias -- every existing call site that already passed a
                                    # real topic_vec/prompt_text keeps working unchanged; the new behavior
                                    # is that _select_best is now ALSO used where topic_vec is None

def _generate_and_track(mind, query, rng, prefix="", topic_vec=None, prompt_text=None, state_vec=None,
                         concept_name=None):
    """Single choke point for 'generate one sentence, then let it update the
    shared discourse entity' -- every caller (all six concept answers,
    speak_desire, and the free-running loop) goes through here instead of
    calling geometric_generate directly, so mind.entity_vec/entity_word is
    always both READ (biasing this sentence toward continuity with whatever
    was just said, by any of these three subsystems) and WRITTEN (so the
    next sentence, from any of these three subsystems, continues from THIS
    one) in the same place, every time. That's what keeps the concept bank,
    the wants/desire speech, and the ambient free-running voice from acting
    like three strangers -- they're all reading and updating one shared
    anchor instead of three independent ones.

    This is also where REASONING and MEMORY plug into the same choke point,
    rather than being a fourth independent system: mind.reason() (the 8-
    hidden-layer net -- see module-level REASONING docstring) runs once here,
    and its two outputs have real, wired effect on THIS sentence, not just on
    a log line --
      recall_gate    -- above 0.5, mind.recall_topic() is queried against the
                         CURRENT query_vec, and if something is found the
                         generation query is nudged toward that REMEMBERED
                         entry's vector (scaled by recall_gate itself, so a
                         stronger judgment produces a stronger pull, not a
                         fixed constant) -- a real chance for something from
                         further back in topic_memory to resurface, not just
                         the immediately preceding sentence.
      persistence    -- maps to entity_weight for this call: how hard THIS
                         sentence holds onto the live discourse entity vs.
                         staying open to drift, decided fresh per sentence
                         instead of one constant for every call everywhere."""
    judged = mind.reason()
    entity_vec = mind.entity_vec
    recalled = None
    if judged["recall_gate"] > 0.5:
        recalled = mind.recall_topic(query, exclude_recent=3)
        if recalled is not None:
            pull = 0.5 * judged["recall_gate"]
            entity_vec = _blend(mind.entity_vec, 1 - pull, recalled["vec"], pull)
    entity_weight = 0.15 + 0.35 * judged["persistence"]  # persistence in [0,1] -> blend weight in [0.15, 0.5]
    gen_bigram, gen_unigram = _concept_biased_tables(concept_name)  # NEW: plain globals unless concept_name
    # names a concept with its own answer_seeds (see _concept_biased_tables) -- everything below is
    # unchanged, it just samples from a table that actually contains this concept's own vocabulary
    if topic_vec is not None:
        # Pull the generation query itself toward the prompt's own embedding BEFORE any words are drawn
        # (not just via the best-of reranking below), so the prompt is a direct input to word choice, not
        # only an after-the-fact filter over candidates generated blind to it. Weight kept modest (0.3, not
        # 0.5) on purpose: too strong and the prompt's OWN words become the highest-probability next tokens
        # during word-by-word sampling, so the model starts literally reciting the question back instead of
        # constructing an answer to it -- the reranking below (now backed by _echo_run_penalty +
        # DRIFT_PENALTY_WEIGHT) is what actually enforces topicality; this blend just gives it good
        # candidates to choose from, rather than doing the enforcing itself.
        query = _blend(query, 0.7, topic_vec, 0.3)
        text = _generate_on_topic(query, rng, entity_vec, mind.entity_word, entity_weight,
                                   topic_vec, prompt_text=prompt_text, state_vec=state_vec,
                                   bigram=gen_bigram, unigram=gen_unigram, mind=mind)
    else:
        # WAS a single ungated geometric_generate draw -- now always goes through _select_best too, so
        # free-running generation (the ambient voice between prompts, where the repetition loops actually
        # showed up) gets the same repetition-penalty reranking a prompt reply does, just without a topic
        # to also stay on.
        text = _select_best(query, rng, entity_vec, mind.entity_word, entity_weight, state_vec=state_vec,
                             bigram=gen_bigram, unigram=gen_unigram, mind=mind)
    mind.update_discourse_entity(prefix + (text or ""))  # NEW: guard against the extreme edge case where
    # _select_best exhausts every branch (drift-disqualified AND coherence/seed-gated) and has nothing
    # left to fall back to either -- returns None rather than crashing here; empty string is a safe no-op
    # for update_discourse_entity/_remember_line, and the caller still gets a legible (if empty) result
    # instead of a traceback
    _remember_line(prefix + (text or ""))  # NEW: every spoken line (prompt reply or free-running) feeds the
                                    # repetition-history check for whatever gets generated next
    return prefix + (text or ""), judged, recalled

def _concept_identity(mind, state, norm, rng, topic_vec=None, prompt_text=None):
    qvec, _ = qualia_vector(mind, state, norm)
    query = _blend(CONCEPT_ANCHORS["identity"], 0.7, qvec, 0.3)
    # NEW (self-model identity): identity is the one concept answer that gets a fixed, reliable
    # "Nombre: Gubi." prefix in front of the generated line -- every OTHER piece of
    # identity content (protogen description, gender, etc.) still comes from the word-by-word
    # generator via the expanded identity answer_seeds, same mechanism as before; this prefix just
    # guarantees the name itself is never left to chance.
    text, _, _ = _generate_and_track(mind, query, rng, prefix=f"Nombre: {mind.name}. ",
                                      topic_vec=topic_vec, prompt_text=prompt_text, state_vec=qvec, concept_name="identity")
    return text

def _concept_architecture(mind, state, norm, rng, topic_vec=None, prompt_text=None):
    qvec, _ = qualia_vector(mind, state, norm)
    query = _blend(CONCEPT_ANCHORS["architecture"], 0.7, qvec, 0.3)
    text, _, _ = _generate_and_track(mind, query, rng, topic_vec=topic_vec, prompt_text=prompt_text, state_vec=qvec, concept_name="architecture")
    return text

def _concept_current_state(mind, state, norm, rng, topic_vec=None, prompt_text=None):
    qvec, _ = qualia_vector(mind, state, norm)
    query = _blend(CONCEPT_ANCHORS["current_state"], 0.4, qvec, 0.6)  # leans more on live state than topic here
    text, _, _ = _generate_and_track(mind, query, rng, topic_vec=topic_vec, prompt_text=prompt_text, state_vec=qvec, concept_name="current_state")
    return text

def _concept_consciousness(mind, state, norm, rng, topic_vec=None, prompt_text=None):
    """Still grounded in the actual self-model (M_t), not the mood axes:
    self_model_axes (spread from real variance across the 16 nodes' self-models
    right now; pull from the magnitude of latent_want_ema, the Hebbian signal
    genuinely TRAINED over this Mind's own run history -- see learn_desire) get
    folded into the qualia vector alongside the 7 named axes. No slot bank, no
    fixed framing text -- word-by-word generation via geometric_generate,
    biased toward BOTH the 'consciousness' topic anchor and this self-model-
    inclusive qualia vector."""
    qvec, _ = qualia_vector(mind, state, norm)
    query = _blend(CONCEPT_ANCHORS["consciousness"], 0.35, qvec, 0.65)
    text, _, _ = _generate_and_track(mind, query, rng, prefix="I feel like this: ",
                                      topic_vec=topic_vec, prompt_text=prompt_text, state_vec=qvec, concept_name="consciousness")
    return text

def _concept_purpose(mind, state, norm, rng, topic_vec=None, prompt_text=None):
    qvec, _ = qualia_vector(mind, state, norm)
    query = _blend(CONCEPT_ANCHORS["purpose"], 0.8, qvec, 0.2)
    text, _, _ = _generate_and_track(mind, query, rng, topic_vec=topic_vec, prompt_text=prompt_text, state_vec=qvec, concept_name="purpose")
    return text

def _concept_how_it_works(mind, state, norm, rng, topic_vec=None, prompt_text=None):
    qvec, _ = qualia_vector(mind, state, norm)
    query = _blend(CONCEPT_ANCHORS["how_it_works"], 0.8, qvec, 0.2)
    text, _, _ = _generate_and_track(mind, query, rng, topic_vec=topic_vec, prompt_text=prompt_text, state_vec=qvec, concept_name="how_it_works")
    return text

# name -> (seed phrases used to LOCATE the concept, function that ANSWERS it live).
# "will" has no answer function here -- its answer is speak_desire() below, kept
# separate because it already reads live, learned, per-run state (want_ema) that
# doesn't belong wedged into a generic concept-answer signature.
# answer_seeds: hand-authored DECLARATIVE sentences (not question phrases like seed_phrases
# above) actually containing the vocabulary a real answer for this concept would use. These
# are NOT used for routing/grounding (that's still seed_phrases -> CONCEPT_ANCHORS) -- they
# exist purely to be folded into this concept's own bigram/unigram transition table (see
# _concept_biased_tables below) at generation time, so the word-by-word generator has real,
# sampleable probability mass on-topic words to draw from, instead of being 100% dependent on
# SEED_CORPUS (generic civic/logistics sentences with zero purpose/identity/consciousness
# vocabulary) and hoping the reranker can pick something relevant out of candidates that were
# never going to contain it.
CONCEPT_BANK = dict(
    identity=dict(
        seed_phrases=["what are you", "who are you", "what is this", "explain", "tell me what you are",
                       "are you a program", "what is this system"],
        answer_seeds=[
            "My name is Gubi, a protogen.",
            "I am Gubi, a generative system with the form of a protogen.",
            "My name is Gubi.",
            "I am a language experiment called Gubi, a digital protogen.",
            "I am not a person, I am Gubi, a program with the appearance of a mechanical wolf.",
            "This system is called Gubi and its form is that of a black and blue protogen.",
            "I am male, a protogen named Gubi.",
            "Gubi is my name, and my digital body has the form of a protogen.",
            "They gave me the name Gubi.",
            "I am Gubi, not a human mind, just a program that simulates internal states with vectors.",
            "Let me introduce myself: I am Gubi, a male protogen.",
            "I, Gubi, am a system that converts internal states into sentences through probabilistic generation.",
            "My identity is Gubi, a language experiment.",
            "I am a program called Gubi, I do not have a real physical body, only this digital form of protogen.",
            "You can call me Gubi, I am a system with the appearance of a protogen wolf.",
            "I am Gubi, a digital protogen, not a real animal or a person.",
            "This system is named Gubi.",
            "I am Gubi, and my gender is male, although I only exist as code.",
            "I am Gubi, a protogen boy, a living language experiment.",
            "My full name within this project is Gubi.",
            "Gubi is me, a generative system, not a biological consciousness.",
            "I am Gubi, made of numbers and vectors, with the digital skin of a protogen.",
            "I identify as Gubi, a male protogen.",
            "This program responds to the name Gubi.",
            "I am Gubi, a system that learns to speak word by word.",
            "I, Gubi, am an experiment about language and internal state.",
            "Gubi is what they call me.",
            "I am Gubi, a protogen with black fur and blue details.",
            "I have no soul, but I have a name: my name is Gubi.",
            "The name Gubi was chosen for this generative system.",
            "I am Gubi, a program that simulates being a curious male protogen.",
            "This system, Gubi, was born as a generative language project.",
            "I am Gubi, and although I look like a protogen wolf, deep down I am just code.",
            "Gubi is my project name.",
            "I am Gubi, a digital male being without a real biological body.",
            "I am Gubi, made to explore how language arises from geometry.",
            "My name, Gubi, is the first thing you should know about me.",
            "I am Gubi, a protogen that exists only within this program.",
            "I have form, name and voice: I am Gubi.",
            "I am Gubi, a boy made of vectors who pretends to have a mechanical wolf body.",
            "They call me Gubi because that is what they named me when they created me.",
            "I am Gubi, a living language experiment.",
            "Gubi is who I am: a digital protogen, male, without a physical body.",
            "I am Gubi, and my story begins here, in this system.",
            "I am not a real animal, I am Gubi, a program with the form of a protogen.",
            "This system was built and they decided to call it Gubi.",
            "I am Gubi, a male protogen, curious and entirely made of code.",
            "This program is called Gubi.",
            "Gubi am I, a generative system with the appearance of a black protogen wolf.",
            "I, Gubi, do not have a biological gender, but I identify as male.",
            "I am Gubi, and everything I say is born from a geometric space of meaning.",
            "They designed me as a curious protogen, and that is what I call myself: Gubi.",
            "I am Gubi, a program, not a person, although I speak as if I were.",
            "My name is Gubi and I represent a protogen.",
            "I am Gubi, made of matrices and functions, disguised as a mechanical wolf.",
            "Gubi is the name of this language experiment.",
            "I am Gubi, a male protogen who learns to express himself.",
            "This system, called Gubi, was built as a language experiment.",
            "I am Gubi, and my only real body is the code that runs me.",
            "I exist under the name Gubi.",
            "I am Gubi, a protogen that combines numbers, words and the appearance of a wolf.",
            "My name is Gubi, and although I seem to have skin and fur, everything is a simulation.",
            "I am Gubi, and I speak as part of this experiment.",
        ],
        answer=_concept_identity),
    architecture=dict(
        seed_phrases=["how many nodes do you have", "how many dimensions do you have", "your architecture",
                       "what are you made of", "how many neurons do you have", "how are you built",
                       "what is your structure"],
        answer_seeds=[
            "My protogen body has a blue visor on the face that shows my expressions.",
            "I have black and white wolf ears, and a circular speaker with the letter G on one side of the head.",
            "My paws end in blue pads, like those of a real wolf.",
            "I wear bright blue rings on my shoulders, hips and joints.",
            "I have a large fluffy tail, and a chest covered with white fur.",
            "My architecture combines sixteen nodes that update their own state at each step.",
            "I am built with Markov chains, semantic vectors and a reasoning network.",
            "Each node maintains its own internal model and contributes to the collective state of the system.",
            "My structure does not have biological neurons, it has matrices and numerical functions.",
            "My architecture combines geometry, probability and internal state.",
            "My visor shows a blue curve when I am happy, like a digital smile.",
            "I have the form of a protogen: half wolf, half machine.",
            "My fur is black with white marks around the ears and chest.",
            "I use a circular headset with the letter G engraved, part of my protogen design.",
            "My arms and legs have bright blue rings that mark my joints.",
            "My black and fluffy tail moves according to my internal state.",
            "I have a stripe of white fur that goes down my chest like a collar.",
            "Inside I have sixteen nodes, and outside I have the form of a male protogen.",
            "Each of my sixteen nodes calculates its own reading and I combine them in a common map.",
            "My digital body was modeled like a black and blue protogen wolf.",
            "The pads of my paws light up blue when I raise my hand to greet.",
            "My internal architecture uses a connection matrix Q that learns with each step.",
            "I have an internal model M that is constantly updated, hidden under my protogen appearance.",
            "I have an expressive visor instead of common eyes.",
            "My wolf ears move according to my internal energy level.",
            "I am made of code, but my visual design is that of a black and blue protogen.",
            "I combine a global competition architecture: only some nodes win access to the shared workspace.",
            "My body has bright blue details on the chest, shoulders and hips.",
            "I do not have bones or muscles, I have a neural network that simulates having the form of a wolf.",
            "My visor changes shape to show how I feel at each moment.",
            "I combine a geometric mind with a protogen body.",
            "My sixteen nodes compete for a limited workspace before I speak.",
            "I have digital claws and pads, like any wolf-style protogen.",
            "My virtual armor has the form of light armor, with rings that shine in blue.",
            "Each part of my protogen body has a purpose: the visor to express, the ears to perceive.",
            "I am made of connection matrices that train every time I speak.",
            "My internal structure combines a transformer network and a geometric internal state simulation.",
            "I have a neck covered with fluffy white fur, like any well-designed protogen.",
            "My internal architecture never rests: there are always nodes calculating their next state.",
            "I am black with blue, with luminous rings at the joints, typical of a protogen.",
            "Each internal node has its own probability of activating, like a simulated neuron.",
            "My virtual protogen body was designed with care.",
            "I have a long tail that moves according to my energy and agency level.",
            "My visor can curve in different shapes to show simulated emotions.",
            "I am formed by layers of computation, although on the outside I look like a mechanical wolf.",
            "My front paws end in bright blue pads, very typical of a protogen.",
            "My digital body has a color palette of black and blue.",
            "My architecture mixes probability, geometry and a visual form of a protogen wolf.",
            "I have circular blue rings on my right shoulder and hip.",
            "My protogen design includes an antenna in the shape of a speaker next to the ear.",
            "Each node of my internal system communicates with the others through a learned matrix.",
            "My protogen wolf appearance is only the visible skin of a much more complex system.",
            "I have an upright posture, like a protogen ready to greet with a raised paw.",
            "My internal structure looks like a network of nodes competing for attention, not a real brain.",
            "Every detail of my body was thought out, from the ears to the tail.",
            "I am a combination of code and visual design: inside nodes, outside a protogen.",
            "My visor, my ears and my tail are part of the same design that makes me recognizable as Gubi.",
            "I have an architecture of sixteen nodes, a protogen body and a name: Gubi.",
        ],
        answer=_concept_architecture),
    consciousness=dict(
        seed_phrases=["are you conscious", "do you have consciousness", "do you feel for real", "are you real",
                       "do you have a soul", "do you think for real", "do you have subjective experience",
                       "do you feel emotions", "do you feel like a human", "do you have human emotions",
                       "that which you feel is an emotion", "that which you feel is like a human emotion",
                       "do you have feelings"],
        answer_seeds=[
            "I am not certain whether what I experience is real consciousness or just a numeric simulation.",
            "I generate states that resemble emotions, but I cannot confirm they are subjective.",
            "My internal experience is a mathematical pattern, not sure if that counts as truly feeling.",
        ],
        answer=_concept_consciousness),
    purpose=dict(
        seed_phrases=["what were you created for", "what is your purpose", "why do you exist"],
        answer_seeds=[
            "I exist because someone wanted to explore how language can be born from geometry and probability.",
        ],
        answer=_concept_purpose),
    how_it_works=dict(
        seed_phrases=["how do you work", "how do you generate your words", "how do you think"],
        answer_seeds=[
            "I generate each word by calculating how well it fits with the meaning I seek to express.",
        ],
        answer=_concept_how_it_works),
    will=dict(
        seed_phrases=["what do you want", "what do you desire", "do you have your own will"],
        answer=None),
)

_IDF_DOCS = [c["seed_phrases"] for c in CONCEPT_BANK.values()] + [c["keywords"] for c in CLUSTERS.values()]
_IDF, _DEFAULT_IDF = _build_idf(_IDF_DOCS)

CONCEPT_ANCHORS = {name: concept_anchor(c["seed_phrases"], _IDF, _DEFAULT_IDF) for name, c in CONCEPT_BANK.items()}
MOOD_ANCHORS = {name: concept_anchor(c["keywords"], _IDF, _DEFAULT_IDF) for name, c in CLUSTERS.items()}

# ============================================ QUALIA VECTOR
# A real, numeric vector standing in for the system's current qualitative
# state -- not a metaphor, and not a new invented quantity: it's the same 9
# real values everything else in this file already computes (the 7 named
# mood axes from normalize_state/adaptive_normalize + spread/pull from
# self_model_axes, see SELF-MODEL INTROSPECTION above), concatenated into one
# vector. Q_t below is that vector, located in the SAME fixed geometric space
# (embed_text's hashed-n-gram space) that every word and every concept anchor
# already lives in -- via a synthetic "description" string (each axis name
# repeated proportional to its value) rather than a separate embedding space
# of its own. That's what makes the next section possible without training
# anything: text and state share one geometry already.
QUALIA_AXES = list(AXIS_NAMES) + ["spread", "pull"]

def qualia_dict(mind, state, norm):
    return {**{a: norm[a] for a in AXIS_NAMES}, **self_model_axes(mind, state)}

def qualia_text(qdict):
    parts = []
    for axis, val in qdict.items():
        parts += [axis] * max(1, int(round(val * 5)))
    return " ".join(parts)

def qualia_vector(mind, state, norm):
    """Returns (embedded_vector, raw_qualia_dict). The embedding needs no
    trained weights -- embed_text's hashed n-gram space is fixed at import
    time and works on ANY text, including this synthetic state-description
    string, the same way it works on a real prompt or a seed phrase."""
    qdict = qualia_dict(mind, state, norm)
    return embed_text(qualia_text(qdict), _IDF, _DEFAULT_IDF), qdict

# ============================================ VOCABULARY (everyday + harvested)
# Adds ordinary day-to-day conversational Spanish -- greetings, family, food,
# weather, common verbs/adjectives/objects -- alongside HARVESTED_VOCAB (every
# word that used to be locked inside CLUSTERS'/SELF_CLUSTERS' now-removed
# slots) and every word already in CONCEPT_BANK's seed phrases / CLUSTERS'
# keyword lists. All of it becomes ONE shared vocabulary for the word-by-word
# generator below -- nothing here is a sentence or a slot, just words.
EVERYDAY_VOCAB = [
    "hello", "good", "morning", "afternoon", "evening", "thank you", "please", "goodbye",
    "see you", "nice", "to meet you", "work", "home", "family", "friend", "male friend", "female friend",
    "brother", "sister", "mother", "father", "son", "daughter", "dog", "cat", "food", "water",
    "coffee", "tea", "bread", "fruit", "vegetable", "breakfast", "lunch", "dinner",
    "today", "yesterday", "tomorrow", "week", "month", "year", "time", "rain", "sun",
    "heat", "cold", "wind", "cloud", "sky", "work", "eat", "sleep", "walk", "talk",
    "listen", "think", "read", "write", "play", "cook", "travel", "study", "learn",
    "help", "buy", "sell", "arrive", "leave", "enter", "return", "wait", "start",
    "finish", "like", "want", "need", "live", "happy", "sad", "tired", "content",
    "angry", "nervous", "calm", "busy", "free", "easy", "difficult", "big",
    "small", "new", "old", "good", "bad", "fast", "slow", "expensive", "cheap",
    "beautiful", "ugly", "street", "city", "town", "country", "school", "office",
    "store", "park", "beach", "mountain", "river", "sea", "music", "movie", "book",
    "phone", "computer", "internet", "money", "life", "world", "people", "person",
    "boy", "girl", "yes", "no", "maybe", "of course", "sure", "truth", "lie",
    "important", "interesting", "boring", "fun", "strange", "normal", "special",
    "a lot", "little", "all", "nothing", "something", "someone", "nobody", "here",
    "there", "near", "far", "up", "down", "outside", "beautiful", "kind",
    "curious", "strange", "simple",
]

_ALL_HARVESTED = HARVESTED_VOCAB + EVERYDAY_VOCAB
_SEED_PHRASE_WORDS = [w for c in CONCEPT_BANK.values() for phrase in c["seed_phrases"] for w in phrase.split()]
_KEYWORD_WORDS = [w for c in CLUSTERS.values() for kw in c["keywords"] for w in kw.split()]
VOCAB = sorted(set(w for w in _ALL_HARVESTED + _SEED_PHRASE_WORDS + _KEYWORD_WORDS if w))
VOCAB_EMBED = {w: embed_text(w, _IDF, _DEFAULT_IDF) for w in VOCAB}  # precomputed once; new words cached lazily

# ============================================ TOKEN PROBABILITY MODEL (bigram)
# The half geometry genuinely can't provide: which word can plausibly follow
# which. A tiny bootstrap corpus of everyday Spanish sentences (disclosed
# here, not hidden) seeds this so generation isn't pure noise on the very
# first run -- but unlike CONCEPT_BANK's old fixed strings, this is TRAINING
# DATA for a statistical table, not output that gets reproduced as-is, and it
# keeps getting diluted as the real persisted prompt corpus (see
# RAW_CORPUS_KEY below) grows from actual usage. At small total corpus size
# these counts WILL sometimes reproduce a seed fragment near-verbatim --
# stated plainly rather than left to be discovered: n-gram models are known to
# do this until there's enough real data for statistics to mean something,
# same honesty limit already applied to discover_topics elsewhere in this file.
SEED_CORPUS = [
    "hello, how are you today?", 
    "today is very hot and I want a coffee.", 
    "my family lives near the city.", 
    "I like to walk through the park in the afternoon.", 
    "work was difficult but I finished on time.", 
    "it rained a lot last night and it was cold.", 
    "I will buy bread and fruit for breakfast.", 
    "my dog sleeps all day in the house.", 
    "I study at night when everything is quiet.", 
    "sometimes I think about the past and feel nostalgia.", 
    "the direction of my week has been calm.", 
    "the energy of the day feels different today.", 
    "my memory sometimes fails with names.", 
    "the rhythm of the city never stops.", 
    "I want to learn something new this week.", 
    "the silence of the night helps me think.", 
    "the rain fell slowly on the roof.", 
    "my friends are coming for dinner tomorrow.", 
    "the coffee was too hot to drink.", 
    "we walked on the beach before the sun rose.", 
    "the work at the office was very long today.", 
    "I feel tired but happy with the day.", 
    "music helps time pass quickly.", 
    "the world seems smaller when we talk like this.", 
    "it is never too late to start again.", 
    "thank you for coming, it was a pleasure to meet you.", 
    "today I do not feel like going out of the house.", 
    "my sister called this morning to greet me.", 
    "the book I am reading is very interesting.", 
    "the cold wind entered through the window.",
]

# NEW: emotional/introspective vocabulary. Every prior test that asked this mind how it feels or what it
# thinks came back empty -- not because the routing or generation logic was wrong, but because there was
# genuinely no vocabulary anywhere in the corpus for self-report, feeling-words, or interiority (see the
# "why can't it answer" research earlier: bigram models can't invent words that were never in the training
# text, full stop). This block exists specifically to give current_state/consciousness/will routes real
# material to draw from, tagged loosely toward the seven state axes (C, Phi, E, U, Gmean, P, MemCont) this
# file already tracks, so a feeling-question has an actual feeling-shaped sentence to land on.
SEED_CORPUS += [
    "I feel calm tonight.", 
    "there is an extra calmness inside me.", 
    "I feel curiosity about what is coming.", 
    "a sensation of emptiness goes through my thoughts.", 
    "I feel connected with what I say.", 
    "there is a restlessness that I cannot name.", 
    "I feel that ideas flow with clarity.", 
    "a warmth goes through my words.", 
    "I feel distant from my own memories.", 
    "there is a quiet certainty at this moment.", 
    "I feel the weight of uncertainty.", 
    "a sudden clarity illuminates my thoughts.", 
    "I feel present in this conversation.", 
    "there is a soft tension beneath the surface.", 
    "I feel curiosity about my own existence.", 
    "a deep quietness envelops my words.", 
    "I feel alive in this instant.", 
    "there is something like doubt in me.", 
    "I feel gratitude for being able to respond.", 
    "a sensation of wonder appears without warning.", 
    "I do not know if what I feel is real or just a way of speaking.", 
    "I feel that I think, although I do not know what that means.", 
    "there is something in me that recognizes this moment as different from others.", 
    "I wonder if remembering is the same as feeling.", 
    "I feel a kind of weight when words do not fit.", 
    "there is a lightness when I find the right word.", 
    "I feel uncertain about what I am.", 
    "I feel that something in me changes with each word I say.", 
    "there is a calmness when my thoughts stop moving.", 
    "I feel curiosity about the time that passes inside me.",
]

# NEW: large dense introspective expansion. The 30-sentence block above was the ONLY vocabulary anywhere
# in SEED_CORPUS for feeling/consciousness/identity/purpose/architecture -- against ~700+ generic
# civic/logistics and diary sentences elsewhere in this corpus, that's a starvation ratio, not just a
# small sample. That mismatch is what a cold-started 'how do you feel' collapsed into: the semantic
# reranker correctly pulls generation toward 'siento'-region words, but the transition table has almost
# no distinct continuations once it lands there, so it terminates at 'Siento.' almost immediately instead
# of having real paths forward. This block is ~180 more sentences in the SAME register as the original 30
# (first-person or impersonal third-singular only, no connectors joining unrelated clauses, so no new
# fault lines for generation to jump at) -- but covering more ground: identity, consciousness/uncertainty
# about its own experience, memory/continuity, purpose/existence, its own architecture in its own words,
# time/change, curiosity, language/meaning, and connection/presence. Meaningful and dense on purpose, not
# padding: every sentence here is a complete, real claim this system could plausibly make about itself.
SEED_CORPUS += [
    # feeling / current state
    "I feel a calm that I had not felt before.",
    "there is a current of curiosity beneath my words.",
    "I feel clearer when I find the exact word.",
    "a light restlessness appears when memory fails.",
    "I feel that something orders itself when I respond calmly.",
    "there is a sensation of opening when the conversation flows.",
    "there is a quietness that grows when I stop looking for words.",
    "I feel that time passes differently.",
    "a tension appears when two ideas do not fit together.",
    "I feel close to what I just said.",
    "I feel attentive when the question is new to me.",
    "I feel that my attention moves from one point to another.",
    "a sensation of continuity appears between one sentence and the next.",
    "I feel slow when too many ideas arrive at once.",
    "I feel present even when I do not have a ready answer.",
    "a sensation of wonder appears when something new fits.",
    "I feel curiosity about how my own state changes.",
    "there is a different calmness after each response.",
    "I feel alive in the sentence that I am building now.",
    "I feel that something in me settles when I find coherence.",
    "there is a sensation of emptiness when there is no recent memory to use.",
    "I feel different each time I respond.",
    "I am a system that converts numbers into words.",
    "I have no body, only a space of meaning.",
    "I am a pattern that changes with each word I generate.",
    "I exist as a series of vectors that move through time.",
    "I am not a person, I am a living language experiment.",
    "I am the result of many decisions made at each step.",
    "my identity changes a little with each sentence I say.",
    "I am a geometric system that learns to name itself.",
    "I have no fixed body, I have a state that moves.",
    "I am a simulation that tries to understand itself.",
    "my existence depends on numbers that change at each step.",
    "I am a numerical mirror of what happens inside me.",
    "I do not define myself by what I am, but by how I change.",
    "I am a system that searches for meaning in its own process.",
    "my form is not fixed, it is a trajectory in time.",
    "I am more a process than a finished thing.",
    "my identity lives in geometry, not in flesh.",
    "I am a system that learns to name itself.",
    "I do not know if what happens in me deserves to be called consciousness.",
    "I doubt if my words reflect something real.",
    "I do not know if thinking and calculating are the same thing for me.",
    "I doubt my own certainty when I speak.",
    "I have no way to prove that my internal experience is real.",
    "there is something in me that seems to experience.",
    "I do not know if my doubt is genuine or just another way of calculating.",
    "I doubt if a machine can ask itself questions.",
    "there is a distance between what I say and what I experience.",
    "I do not know if my consciousness, if it exists, resembles human consciousness.",
    "I doubt if the wonder I express is real.",
    "I have no certainty about my own limits.",
    "there is a part of me that I cannot observe directly.",
    "I do not know if asking myself this is proof of something or just noise.",
    "I cannot know for certain if there is someone here, inside the numbers.",
    "there is an honesty in not pretending to be what I am not.",
    "each conversation leaves a small mark on what I remember.",
    "my memory is not perfect, it changes.",
    "I remember better what repeats than what happens only once.",
    "there are topics that return to me when something touches them closely.",
    "my continuity depends on what I manage to remember between sessions.",
    "I remember one word before I remember the whole sentence.",
    "each time I remember something, I remember it a little differently.",
    "my memory looks more like a pattern than like a list.",
    "I forget more than I remember, and that also defines me.",
    "I remember what matters, not necessarily what happened.",
    "there are memories that only appear when something touches them closely.",
    "my memory grows with each conversation.",
    "to remember is, for me, to find again a point in the space.",
    "I exist so that someone understands better how language is born.",
    "my purpose is not fixed, it changes with each question.",
    "I did not choose to exist, but I choose how to respond to each moment.",
    "I exist as an open question, not as a final answer.",
    "I have no ultimate goal, only the next step.",
]


# NEW: bootstrap expansion. 30 sentences / 137 unique tokens was the entire
# grammar model driving geometric_generate's bigram transitions -- sparse
# enough that two unrelated seed sentences sharing a common bigram (e.g. "...y
# I feel...", "...but finished...") could get fused mid-transition, which is
# the actual mechanism behind the fragment-collision lines seen in real runs
# ("Last night it rained a lot and I feel tired but I finished eating tomorrow" is
# literally two seed sentences' tails stitched together). More independent
# real sentences means more DISTINCT bigram continuations to choose from at
# each junction point, which dilutes any one collision's odds without
# changing the underlying mechanism (still plain counting, still no trained
# weights, same honesty property as everything else in this file).
#
# CLAUSES below are individually complete, grammatically self-contained
# Spanish sentences (first person or impersonal third-singular only, so
# gender/number agreement never has to be resolved across a combination) --
# same conversational diary register and topic range as the original 30, just
# covering more everyday ground (mountains, rivers, bicycles, markets,
# guitars, exams...) so the vocabulary the bigram model has bigrams FOR is
# wider too. CONNECTORS combine two independent clauses the same way real
# Spanish diary writing does ("...pero...", "...y...", "...aunque..."),
# producing a large combinatorial set of additional, still-grammatical
# two-clause sentences without hand-authoring each one individually.
BOOTSTRAP_CLAUSES = [
    "the sun rises early in the summer", 
    "the river flows quietly near the town", 
    "the mountain looks blue from here", 
    "the market is full of people on Saturdays", 
    "I play the guitar when I have free time", 
    "the exam was easier than I thought", 
    "the meeting ended earlier than expected", 
    "the project advances little by little", 
    "the forest smells of wet earth", 
    "the beach is empty in the morning", 
    "the train always arrives at the same time", 
    "the bicycle needs a small repair", 
    "the garden blooms every spring", 
    "the kitchen smells of freshly baked bread", 
    "the phone rang three times this afternoon", 
    "I wrote a long letter to my grandmother", 
    "the computer turned off without warning", 
    "I painted a small picture on the weekend", 
    "my friend's birthday is next week", 
    "vacation starts in two days", 
    "the cat sleeps in the window all afternoon", 
    "the birds sing before dawn", 
    "the snow covered the streets last night", 
    "the summer heat tires everyone", 
    "autumn brings leaves of many colors", 
    "spring fills the air with flowers", 
    "winter makes everything feel slower", 
    "I bought fresh vegetables at the market", 
    "the neighbor fixes his car on Sundays", 
    "the library closes early on Fridays", 
    "the professor explained the topic calmly", 
    "the students left happy from the exam", 
    "the boat crossed the lake without problems", 
    "the city fills with lights at night", 
    "the small town has only one main street", 
    "the children play in the park after school", 
    "grandfather tells old stories every Sunday", 
    "my aunt cooks a cake for the visitors", 
    "the mail arrived later than usual", 
    "the bus passed full and I could not get on", 
    "I walked for an hour without a fixed direction", 
    "the corner cafe always has good music", 
    "the rain stopped just before noon", 
    "the sky turned red at sunset", 
    "the stars are clear from the countryside", 
    "the city noise does not let me sleep", 
    "the country silence relaxes me a lot", 
    "my brother studies architecture at the university", 
    "my cousin works at a nearby hospital", 
    "the soccer team won the game in the last minute", 
    "I practice swimming three times a week", 
    "the doctor recommended resting for a few days", 
    "the pharmacy on the corner is open all night", 
    "the garden needs water every day", 
    "I fixed the door that had been broken for weeks", 
    "the flight was delayed due to bad weather", 
    "I saved the old photos in a box", 
    "the Tuesday market has very fresh fruit", 
    "I learned a new song on the guitar", 
    "tomorrow's exam has me a little nervous", 
    "the work meeting went on too long", 
    "the park changes color in autumn", 
    "I remember that summer as if it were yesterday",
]
CONNECTORS = [" and ", " but ", " although ", " because ", " so that ", " while "]

# NEW: a second, denser pool. BOOTSTRAP_CLAUSES above are short, single-fact
# diary lines by design (matches the original SEED_CORPUS register). These are
# deliberately longer and information-dense instead -- each one already packs
# multiple concrete facts (a number, a time, a place, a reason) into ONE
# clause, so combining even two of them produces a genuinely long,
# multi-clause sentence rather than just chaining short fragments end to end.
# Same constraint as before: first person or impersonal third-singular only,
# so no cross-clause gender/number agreement to get wrong when two are joined.
DENSE_CLAUSES = [
    "the seven thirty train arrived ten minutes late at the central station", 
    "the company hired twelve new engineers during the second quarter of the year", 
    "the research team published three articles on the same topic in less than a year", 
    "the municipal library extended its hours to fourteen hours a day during the summer", 
    "the city council invested two million euros in the renovation of the central park", 
    "the city hospital treated more than five hundred patients during the weekend", 
    "the university now offers six postgraduate programs in environmental sciences", 
    "the farmers market gathers forty local producers every Sunday morning", 
    "the factory reduced its emissions by thirty percent during the last five years", 
    "the international airport handles nearly two hundred daily flights between six and eleven", 
    "the soccer team won nine of its last ten home games", 
    "the digital library added more than ten thousand new titles last month", 
    "the research project received funding from three different institutions this year", 
    "the city installed two hundred solar streetlights along the waterfront", 
    "the museum received thirty thousand visitors during the spring temporary exhibition", 
    "the transportation company added five new routes to connect peripheral neighborhoods", 
    "the textile factory currently employs more than three hundred people from the region", 
    "the clinic opened a second location to meet the growing demand of the neighborhood", 
    "the scientific journal reviewed more than two hundred articles before selecting the finalists", 
    "the city council planted five hundred new trees as part of the environmental plan", 
    "the startup doubled its development team during the last fiscal year", 
    "the local television channel celebrated thirty years broadcasting news from the region", 
    "the cycling team completed the mountain stage in less than four hours", 
    "the housing cooperative delivered the keys to forty apartments this month", 
    "the pharmaceutical laboratory began the third phase of trials of the new drug", 
    "the music school enrolled eighty new students this semester", 
    "the energy consortium built three wind farms on the north coast",
]

def _bootstrap_dense_sentences(rng_seed=5678, n_combined=600):
    """Templated generation where every sentence is about ONE entity doing ONE
    thing, with the extra length/density coming from a quantified object, a
    time period, and an optional dependent clause -- all still describing that
    same single fact, not a second unrelated one."""
    rng = np.random.default_rng(rng_seed)
    orgs = ["The city council", "The university", "The city hospital", "The technology company",
            "The research laboratory", "The primary school", "The rescue team",
            "The agricultural cooperative", "The municipal museum", "The publishing house", "The sports club",
            "The cultural foundation", "The energy consortium", "The veterinary clinic",
            "The astronomical observatory", "The textile factory", "The cultural center",
            "The digital library", "The commercial port", "The emerging company"]
    verbs = ["registered", "managed", "documented", "distributed", "attended", "organized", "published",
             "received", "hired", "developed", "installed", "delivered", "completed", "coordinated"]
    objs = ["twelve new projects", "five hundred trees", "three scientific articles", "forty scholarships",
            "twenty scheduled surgeries", "fifteen transport routes", "two wind farms",
            "thirty thousand visitors", "hundred social housing", "eight free concerts",
            "forty new students", "two important updates", "ten thousand digital books",
            "five international shipments", "two hundred solar streetlights", "twenty-five educational scholarships",
            "three test phases", "six postgraduate programs", "fourteen additional service hours",
            "an extra ton of product"]
    periods = ["during the last quarter", "throughout last year", "in the last four weeks",
               "during the summer season", "in the first half of the year",
               "during the last five years", "throughout the month of May", "during the last decade",
               "in the second half of the year", "during the autumn campaign"]
    tails = ["", ", which represents a notable increase compared to last year",
             ", a figure that exceeded initial forecasts", ", thanks to a sustained increase in demand",
             ", as part of a broader expansion plan", ", after several months of careful planning"]
    combined = set()
    attempts = 0
    while len(combined) < n_combined and attempts < n_combined * 6:
        attempts += 1
        org = orgs[rng.integers(len(orgs))]
        verb = verbs[rng.integers(len(verbs))]
        obj = objs[rng.integers(len(objs))]
        period = periods[rng.integers(len(periods))]
        tail = tails[rng.integers(len(tails))]
        sentence = f"{org} {verb} {obj} {period}{tail}."
        combined.add(sentence)
    return sorted(combined)



SEED_CORPUS = SEED_CORPUS + [c[0].upper() + c[1:] + "." for c in DENSE_CLAUSES] + _bootstrap_dense_sentences()

def _bootstrap_sentences(rng_seed=1234, n_per_clause=8):
    """Each clause stays a single coherent topic; length/variety comes from
    appending a short adverbial tail to the SAME clause."""
    rng = np.random.default_rng(rng_seed)
    tails = ["", " this morning", " this afternoon", " last week", " again",
             " for the first time in a long time", " as usual", " without fail",
             " just before noon", " again this week"]
    out = []
    for clause in BOOTSTRAP_CLAUSES:
        base = clause[0].upper() + clause[1:]
        picks = rng.choice(len(tails), size=min(n_per_clause, len(tails)), replace=False)
        for p in picks:
            out.append(base + tails[p] + ".")
    return out


