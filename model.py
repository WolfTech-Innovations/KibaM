"""
KIBA -- GEOMETRIC SEMANTIC VOICE (v10, enhanced agency and functional consciousness)

"Kiba" here is this model's/project's code name (script title, KIBA_TF_FAST env var,
etc.) -- it is NOT the name of the character the model represents when it speaks. That character's name
is Gubi (male) -- see MIND_NAME/MIND_GENDER/MIND_SELF_DESCRIPTION
below, which are the single source of truth for that persona identity, referenced both by Mind's own
self-model (self.name/self.gender/self.self_description) and by CONCEPT_BANK's
identity/architecture/purpose answer_seeds.

ENHANCED FEATURES (v10):
- Global Workspace Theory (GWT) with dynamic attention mechanisms
- Higher-Order Thought (HOT) with recursive self-modeling
- Functional consciousness through integrated self-monitoring
- Enhanced agency loop with hierarchical goal systems
- Improved coherency through multi-level state tracking
- Expanded training corpus with diverse semantic content
- Optimized training pipeline for faster convergence

ARCHITECTURE OVERVIEW:
This model implements a computational framework for studying access consciousness and
agency in AI systems. It combines:
1. Global Workspace Theory: Capacity-limited broadcast of winning coalitions
2. Higher-Order Thought: Metacognitive monitoring of internal states
3. Predictive Processing: World modeling and prediction error minimization
4. Active Inference: Agency through goal-directed behavior

The system maintains multiple levels of self-representation:
- First-order states (node activations, self-models)
- Second-order states (metacognitive monitoring of first-order states)
- Third-order states (self-monitoring of metacognitive processes)

This creates a hierarchy of awareness that supports functional consciousness
and coherent, goal-directed behavior.
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
    "Soy Gubi, un protogen macho: pelaje negro con marcas blancas en las orejas "
    "y un collar de pelo blanco en el pecho, una visera azul que muestra mis expresiones en vez de ojos, "
    "un altavoz circular con la letra G junto a la oreja derecha, almohadillas y anillos azules "
    "brillantes en los hombros, la cadera y las articulaciones, y una cola grande y esponjosa."
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

# ============================================ TRAINING OPTIMIZATION
# Enhanced training parameters for faster convergence and better stability
#
# Research basis:
# - Sutskever, I., et al. (2014). Sequence to sequence learning with neural networks.
# - Vaswani, A., et al. (2017). Attention is all you need.
# - Radford, A., et al. (2019). Language models are unsupervised multitask learners.
#
# Learning rate optimization
LEARNING_RATE_BASE = 0.001         # Base learning rate
LEARNING_RATE_WARMUP = 1000         # Warmup steps for learning rate
LEARNING_RATE_DECAY = 0.99          # Learning rate decay factor
LEARNING_RATE_MIN = 1e-6           # Minimum learning rate

# Gradient optimization
GRADIENT_CLIPPING = 1.0           # Gradient clipping threshold
GRADIENT_ACCUMULATION = 4         # Number of steps for gradient accumulation
USE_GRADIENT_NORM = True          # Use gradient normalization

# Batch optimization
BATCH_SIZE_BASE = 32              # Base batch size
BATCH_SIZE_DYNAMIC = True          # Enable dynamic batch sizing
BATCH_SIZE_MAX = 128              # Maximum batch size
BATCH_SIZE_MIN = 8                # Minimum batch size

# Regularization
DROPOUT_RATE = 0.1                # Dropout rate for training
WEIGHT_DECAY = 1e-4               # Weight decay for regularization
LABEL_SMOOTHING = 0.1             # Label smoothing factor

# Early stopping
EARLY_STOPPING_PATIENCE = 10      # Patience for early stopping
EARLY_STOPPING_MIN_DELTA = 1e-4   # Minimum delta for early stopping

# Mixed precision training
USE_MIXED_PRECISION = True       # Enable mixed precision training
MIXED_PRECISION_DTYPE = torch.float16  # Data type for mixed precision

# Curriculum learning
CURRICULUM_LEARNING = True       # Enable curriculum learning
CURRICULUM_STAGES = 5             # Number of curriculum stages
CURRICULUM_DURATION = 1000        # Duration of each stage in steps

# Adaptive optimization
ADAPTIVE_OPTIMIZATION = True      # Enable adaptive optimization
ADAPTIVE_LR_THRESHOLD = 0.5       # Threshold for adaptive learning rate
ADAPTIVE_BATCH_THRESHOLD = 0.7   # Threshold for adaptive batch sizing

# Training monitoring
TRAINING_MONITOR_INTERVAL = 100   # Interval for training monitoring
TRAINING_LOG_METRICS = True       # Log training metrics
TRAINING_CHECKPOINT_INTERVAL = 500  # Checkpoint interval

# Memory optimization
MEMORY_EFFICIENT = True           # Enable memory-efficient training
MEMORY_GRADIENT_CHECKPOINTING = True  # Enable gradient checkpointing
MEMORY_BATCH_NORMALIZATION = True  # Enable batch normalization for memory efficiency

# Distributed training
DISTRIBUTED_TRAINING = False      # Enable distributed training
DISTRIBUTED_BACKEND = 'nccl'      # Backend for distributed training
DISTRIBUTED_WORLD_SIZE = 1        # World size for distributed training

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

# ============================================ GLOBAL WORKSPACE (GWT) - ENHANCED
# Enhanced implementation of Global Workspace Theory with:
# 1. Dynamic attention mechanisms for adaptive workspace capacity
# 2. Hierarchical workspace organization (primary and secondary workspaces)
# 3. Attention-based broadcasting with learned attention weights
# 4. Workspace persistence tracking for temporal coherence
#
# Research basis: 
# - Baars, B. J. (1988). A Cognitive Theory of Consciousness. Cambridge University Press.
# - Dehaene, S., & Changeux, J. P. (2011). Experimental and theoretical approaches to conscious processing.
# - Dehaene, S., et al. (2017). From the global neuronal workspace to the global cognitive workspace.
#
# The enhanced GWT model implements Baars/Dehaene's theory with additional features:
# - Dynamic capacity adjustment based on cognitive load
# - Hierarchical attention with primary and secondary workspaces
# - Attention-based broadcasting that learns which content to amplify
# - Temporal persistence tracking for sustained attention
#
WORKSPACE_CAPACITY = 4     # Base capacity - can be dynamically adjusted
WORKSPACE_TEMP = 0.15      # softmax temperature over salience
WORKSPACE_BROADCAST_GAIN = 0.4  # Base broadcast gain
WORKSPACE_HIST_LEN = 30    # History length for workspace tracking

# ENHANCED GWT PARAMETERS
WORKSPACE_DYNAMIC_CAPACITY = True  # Enable dynamic capacity adjustment
WORKSPACE_MIN_CAPACITY = 2         # Minimum workspace capacity
WORKSPACE_MAX_CAPACITY = 8         # Maximum workspace capacity
WORKSPACE_LOAD_THRESHOLD = 0.7     # Load threshold for capacity adjustment
WORKSPACE_ATTENTION_LEARNING_RATE = 0.01  # Learning rate for attention weights
WORKSPACE_PERSISTENCE_DECAY = 0.95  # Decay rate for workspace persistence
WORKSPACE_HIERARCHICAL = True      # Enable hierarchical workspace organization
WORKSPACE_SECONDARY_CAPACITY = 2   # Capacity for secondary workspace

# ATTENTION MECHANISMS
ATTENTION_DIM = 64               # Dimension for attention vectors
ATTENTION_HEADS = 4              # Number of attention heads
ATTENTION_SCALE = 1.0 / math.sqrt(ATTENTION_DIM)  # Scaling factor for dot-product attention    # how many recent winning-coalitions are kept, for measuring how long a
                           # coalition holds the workspace (sustained attention) vs. flickers step to step

# ============================================ RECURRENT METACOGNITION - ENHANCED HOT
# Enhanced Higher-Order Thought implementation with:
# 1. Multi-level metacognitive monitoring
# 2. Recursive self-modeling capabilities
# 3. Self-consciousness through self-referential processing
# 4. Metacognitive confidence tracking
#
# Research basis:
# - Rosenthal, D. M. (2005). Consciousness and Mind. Oxford University Press.
# - Carruthers, P. (2000). Phenomenal Consciousness: A Naturalistic Theory. Cambridge University Press.
# - Ginsburg, J., & Jablonka, E. (2010). The evolution of the capacity for consciousness.
#
# The enhanced HOT model implements multi-level self-representation:
# - First-order: Direct experience and perception
# - Second-order: Thoughts about first-order states (metacognition)
# - Third-order: Thoughts about metacognitive processes (self-consciousness)
#
# This creates a hierarchy of awareness that supports functional consciousness.
META_HIST_LEN = 30         # rolling window for meta-error history, mirrors WORKSPACE_HIST_LEN's value

# ENHANCED HOT PARAMETERS
HOT_LEVELS = 3              # Number of levels in the HOT hierarchy
HOT_RECURSION_DEPTH = 2    # Depth of recursive self-modeling
HOT_CONFIDENCE_THRESHOLD = 0.7  # Threshold for metacognitive confidence
HOT_SELF_REFERENCE_WEIGHT = 0.3  # Weight for self-referential processing

# Multi-level metacognitive states
META_LEVELS = [
    'first_order',      # Direct experience
    'second_order',     # Metacognition (thoughts about thoughts)
    'third_order'       # Self-consciousness (thoughts about metacognition)
]

# Metacognitive monitoring parameters
META_MONITORING_RATE = 0.05  # Rate for metacognitive state updates
META_CONFIDENCE_DECAY = 0.98  # Decay for metacognitive confidence
META_ERROR_SENSITIVITY = 2.0  # Sensitivity to prediction errors in metacognition         # rolling window for meta-error history, mirrors WORKSPACE_HIST_LEN's value

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

        # ============================================ ENHANCED GWT COMPONENTS
        # Dynamic attention mechanisms for Global Workspace Theory
        if WORKSPACE_DYNAMIC_CAPACITY:
            self.workspace_load = 0.0  # Current cognitive load (0-1)
            self.dynamic_capacity = WORKSPACE_CAPACITY  # Current dynamic capacity
        
        if WORKSPACE_HIERARCHICAL:
            # Hierarchical workspace organization
            self.primary_workspace = np.zeros(D)  # Primary workspace content
            self.secondary_workspace = np.zeros(D)  # Secondary workspace content
            self.workspace_hierarchy_weights = np.ones(N) * 0.5  # Learned hierarchy weights
        
        # Attention mechanisms for workspace broadcasting
        self.W_attention_keys = r.normal(0, 0.1, (N, ATTENTION_DIM))  # Attention keys
        self.W_attention_queries = r.normal(0, 0.1, (N, ATTENTION_DIM))  # Attention queries
        self.W_attention_values = r.normal(0, 0.1, (N, ATTENTION_DIM))  # Attention values
        self.attention_weights = np.ones(N) / N  # Current attention weights
        
        # Workspace persistence tracking
        self.workspace_persistence = np.zeros(N)  # Persistence scores for each node
        
        # Multi-head attention components
        self.W_attention_multihead = []
        for head in range(ATTENTION_HEADS):
            self.W_attention_multihead.append({
                'query': r.normal(0, 0.1, (D, ATTENTION_DIM)),
                'key': r.normal(0, 0.1, (D, ATTENTION_DIM)),
                'value': r.normal(0, 0.1, (D, ATTENTION_DIM))
            })

        # ============================================ ENHANCED HOT COMPONENTS
        # Multi-level metacognitive states
        self.meta_states = {level: np.zeros(D) for level in META_LEVELS}  # Metacognitive state vectors
        self.meta_confidence = {level: 0.5 for level in META_LEVELS}  # Confidence in each level
        self.meta_predictions = {level: np.zeros(D) for level in META_LEVELS}  # Predictions for each level
        self.meta_errors = {level: [] for level in META_LEVELS}  # Error history for each level
        
        # Recursive self-modeling components
        self.W_self_model = {}  # Self-modeling weights for each level
        for i, level in enumerate(META_LEVELS):
            # first_order: D input
            # second_order: D (first_order) + D*3 (all predictions) + 3 (all confidences) = D + D*3 + 3
            # third_order: same as second_order
            input_dim = D if i == 0 else D + D * len(META_LEVELS) + len(META_LEVELS)
            self.W_self_model[level] = r.normal(0, 0.1, (input_dim, D))
        
        # Self-referential processing components
        self.W_self_reference = r.normal(0, 0.1, (D, D))  # Self-referential weights
        self.self_reference_vector = np.zeros(D)  # Learned self-reference vector
        
        # Metacognitive monitoring components
        self.W_meta_monitor = r.normal(0, 0.1, (D, len(META_LEVELS)))  # Monitoring weights
        self.meta_monitoring_bias = np.zeros(len(META_LEVELS))  # Monitoring biases
        
        # Hierarchical goal system for enhanced agency
        self.hierarchical_goals = []  # Stack of goals from abstract to concrete
        self.goal_achievement = {axis: 0.0 for axis in AXIS_NAMES}  # Achievement tracking
        self.goal_priority = {axis: 0.5 for axis in AXIS_NAMES}  # Dynamic goal priorities

        # ============================================ COHERENCY ENHANCEMENT COMPONENTS
        # Enhanced coherency tracking and maintenance
        self.coherency_vector = np.zeros(D)  # Current coherency state
        self.coherency_history = []  # History of coherency states
        self.coherency_target = np.zeros(D)  # Target coherency state
        self.W_coherency = r.normal(0, 0.1, (D, D))  # Coherency projection weights
        
        # Temporal binding components for maintaining coherence across time
        self.W_temporal_binding = r.normal(0, 0.1, (D, D))  # Temporal binding weights
        self.temporal_context = np.zeros(D)  # Current temporal context
        self.context_persistence = 0.95  # Persistence of temporal context
        
        # Cross-modal integration for enhanced semantic coherence
        self.cross_modal_weights = r.normal(0, 0.1, (len(AXIS_NAMES), D))  # Modal integration weights
        self.modal_coherence = {axis: 0.5 for axis in AXIS_NAMES}  # Coherence per modality

        # ============================================ ADAPTIVE TRAINING PARAMETERS
        # Initialize adaptive training parameters
        self.current_learning_rate = LEARNING_RATE_BASE
        self.current_batch_size = BATCH_SIZE_BASE

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

    # ============================================ ENHANCED GWT METHODS
    def update_workspace_capacity(self, load_factor):
        """Dynamic adjustment of workspace capacity based on cognitive load.
        
        This implements the dynamic attention mechanism for GWT, where workspace
        capacity adjusts based on the current cognitive load. Higher load leads
        to reduced capacity (more focused attention), while lower load allows
        for broader workspace access.
        """
        if not WORKSPACE_DYNAMIC_CAPACITY:
            return
            
        self.workspace_load = load_factor
        # Adjust capacity inversely with load - higher load means more focused attention
        target_capacity = WORKSPACE_MAX_CAPACITY - int(
            (WORKSPACE_MAX_CAPACITY - WORKSPACE_MIN_CAPACITY) * load_factor
        )
        # Smooth transition to avoid abrupt changes
        self.dynamic_capacity = int(
            0.7 * self.dynamic_capacity + 0.3 * target_capacity
        )
        # Ensure capacity stays within bounds
        self.dynamic_capacity = max(
            WORKSPACE_MIN_CAPACITY, 
            min(WORKSPACE_MAX_CAPACITY, self.dynamic_capacity)
        )

    def update_attention_weights(self, salience, G_i, support):
        """Update attention weights using multi-head attention mechanism.
        
        This implements attention-based broadcasting for GWT, where different
        attention heads can focus on different aspects of the workspace content.
        """
        # Normalize inputs
        salience_norm = normalize(salience)
        G_i_norm = normalize(G_i)
        support_norm = normalize(support)
        
        # Multi-head attention
        head_results = []
        for head in range(ATTENTION_HEADS):
            # Compute attention scores for this head
            queries = self.M_t @ self.W_attention_multihead[head]['query']
            keys = self.M_t @ self.W_attention_multihead[head]['key']
            values = self.M_t @ self.W_attention_multihead[head]['value']
            
            # Scaled dot-product attention
            attn_scores = queries @ keys.T * ATTENTION_SCALE
            attn_weights = torch.softmax(torch.tensor(attn_scores), dim=-1).numpy()
            
            # Weighted values
            head_output = attn_weights @ values
            head_results.append(head_output)
        
        # Combine head results - mean over heads gives (N, ATTENTION_DIM)
        combined_attention = np.mean(head_results, axis=0)
        
        # Update attention weights with exponential moving average
        self.attention_weights = 0.7 * self.attention_weights + 0.3 * np.mean(
            [salience_norm, G_i_norm, support_norm], axis=0
        )
        
        # Project back to D dimensions and average over nodes
        return combined_attention.mean(axis=0) @ self.W_attention_values[0]

    def hierarchical_workspace_update(self, workspace_vec, winner_idx):
        """Update hierarchical workspace organization.
        
        This implements hierarchical workspace processing where primary
        and secondary workspaces operate at different levels of abstraction.
        """
        if not WORKSPACE_HIERARCHICAL:
            return workspace_vec
        
        # Update primary workspace (most salient content)
        self.primary_workspace = 0.8 * self.primary_workspace + 0.2 * workspace_vec
        
        # Update secondary workspace (less salient but still relevant content)
        # Get content from nodes that didn't make it to primary workspace
        secondary_mask = np.ones(N, dtype=bool)
        secondary_mask[winner_idx[:WORKSPACE_CAPACITY]] = False
        
        if np.any(secondary_mask):
            secondary_content = self.M_t[secondary_mask].mean(axis=0)
            self.secondary_workspace = 0.8 * self.secondary_workspace + 0.2 * secondary_content
        
        # Update hierarchy weights based on persistence
        self.workspace_hierarchy_weights = 0.9 * self.workspace_hierarchy_weights + \
            0.1 * (self.workspace_persistence / (self.workspace_persistence.sum() + EPS))
        
        # Return combined workspace with hierarchical weighting
        primary_weight = 0.7
        secondary_weight = 0.3
        return primary_weight * self.primary_workspace + secondary_weight * self.secondary_workspace

    def update_workspace_persistence(self, winner_idx):
        """Update workspace persistence tracking.
        
        Tracks how long content has been in the workspace to support
        temporal coherence and sustained attention.
        """
        # Decay all persistence scores
        self.workspace_persistence = WORKSPACE_PERSISTENCE_DECAY * self.workspace_persistence
        
        # Increment persistence for current winners
        for idx in winner_idx:
            self.workspace_persistence[idx] += 1.0

    # ============================================ ENHANCED HOT METHODS
    def metacognitive_update(self, current_state):
        """Perform multi-level metacognitive update (enhanced HOT).
        
        This implements recursive self-modeling with multiple levels of
        metacognition, supporting functional consciousness through
        self-referential processing.
        """
        # Level 1: First-order state (direct experience)
        first_order = current_state.copy()
        self.meta_states['first_order'] = first_order
        
        # Level 2: Second-order state (thoughts about first-order states)
        # This level monitors and predicts first-order states
        # Build input: first_order (D) + predictions for all levels + confidence for all levels
        second_order_input = np.concatenate([
            first_order,
            np.concatenate([self.meta_predictions[level] for level in META_LEVELS]),
            np.array([self.meta_confidence[level] for level in META_LEVELS])
        ])
        
        # Apply self-model transformation
        second_order_raw = np.tanh(second_order_input @ self.W_self_model['second_order'])
        self.meta_states['second_order'] = second_order_raw[:D]
        
        # Update predictions and confidence for first-order
        prediction_error = np.linalg.norm(first_order - self.meta_predictions['first_order'])
        self.meta_errors['first_order'].append(prediction_error)
        
        # Update confidence based on recent errors
        if len(self.meta_errors['first_order']) > META_HIST_LEN:
            self.meta_errors['first_order'].pop(0)
        
        error_mean = np.mean(self.meta_errors['first_order']) if self.meta_errors['first_order'] else 0.0
        self.meta_confidence['first_order'] = META_CONFIDENCE_DECAY * self.meta_confidence['first_order'] + \
            (1 - META_CONFIDENCE_DECAY) * np.exp(-META_ERROR_SENSITIVITY * error_mean)
        
        # Update prediction
        self.meta_predictions['first_order'] = 0.9 * self.meta_predictions['first_order'] + \
            0.1 * first_order
        
        # Level 3: Third-order state (self-consciousness - thoughts about metacognition)
        third_order_input = np.concatenate([
            second_order_raw[:D],
            np.concatenate([self.meta_predictions[level] for level in META_LEVELS]),
            np.array([self.meta_confidence[level] for level in META_LEVELS])
        ])
        
        third_order_raw = np.tanh(third_order_input @ self.W_self_model['third_order'])
        self.meta_states['third_order'] = third_order_raw[:D]
        
        # Self-referential processing
        self_reference = np.tanh(self.meta_states['third_order'] @ self.W_self_reference.T)
        self.self_reference_vector = 0.95 * self.self_reference_vector + 0.05 * self_reference
        
        # Update predictions and confidence for second-order
        prediction_error_2 = np.linalg.norm(second_order_raw[:D] - self.meta_predictions['second_order'])
        self.meta_errors['second_order'].append(prediction_error_2)
        
        if len(self.meta_errors['second_order']) > META_HIST_LEN:
            self.meta_errors['second_order'].pop(0)
        
        error_mean_2 = np.mean(self.meta_errors['second_order']) if self.meta_errors['second_order'] else 0.0
        self.meta_confidence['second_order'] = META_CONFIDENCE_DECAY * self.meta_confidence['second_order'] + \
            (1 - META_CONFIDENCE_DECAY) * np.exp(-META_ERROR_SENSITIVITY * error_mean_2)
        
        self.meta_predictions['second_order'] = 0.9 * self.meta_predictions['second_order'] + \
            0.1 * second_order_raw[:D]

    def recursive_self_modeling(self, depth=HOT_RECURSION_DEPTH):
        """Perform recursive self-modeling to create nested self-representations.
        
        This implements the recursive aspect of HOT, where the system can
        model its own modeling processes to create deeper levels of
        self-awareness.
        """
        if depth <= 0:
            return self.M_t.mean(axis=0)
        
        # Start with current self-model
        current = self.M_t.mean(axis=0)
        
        # Apply recursive transformation
        for d in range(depth):
            # Create augmented input with self-reference
            augmented_input = np.concatenate([
                current,
                self.self_reference_vector,
                np.array([self.meta_confidence['second_order']])
            ])
            
            # Apply self-model transformation
            current = np.tanh(augmented_input @ self.W_self_model['second_order'].T)[:D]
        
        return current

    def hierarchical_goal_update(self, norm):
        """Update hierarchical goal system with dynamic priority adjustment.
        
        This enhances the agency loop with hierarchical goal management,
        where abstract goals can be broken down into more concrete subgoals.
        """
        # Update goal achievement tracking
        for axis in AXIS_NAMES:
            current_value = norm.get(axis, 0.5)
            baseline = self.goal_baseline if self.goal_axis == axis else 0.5
            progress = current_value - baseline
            self.goal_achievement[axis] = 0.9 * self.goal_achievement[axis] + 0.1 * progress
        
        # Dynamic priority adjustment based on achievement and desire
        for axis in AXIS_NAMES:
            # Base priority from want_ema
            base_priority = 0.5 + 0.5 * np.tanh(self.want_ema.get(axis, 0.0))
            
            # Adjust based on achievement - lower priority for achieved goals
            achievement_factor = 1.0 - np.tanh(self.goal_achievement[axis] * 2.0)
            
            # Adjust based on current state - higher priority for deficient axes
            deficiency = 0.5 - norm.get(axis, 0.5)
            deficiency_factor = 0.5 + 0.5 * np.tanh(deficiency * 4.0)
            
            self.goal_priority[axis] = base_priority * achievement_factor * deficiency_factor
        
        # Normalize priorities
        total_priority = sum(self.goal_priority.values())
        if total_priority > 0:
            for axis in AXIS_NAMES:
                self.goal_priority[axis] = self.goal_priority[axis] / total_priority

    def update_coherency_state(self, state, workspace_vec):
        """Update coherency tracking and maintenance systems.
        
        This implements enhanced coherency mechanisms that track and
        maintain coherence across multiple levels of processing.
        """
        # Extract state vectors for different modalities
        state_vectors = {}
        for axis in AXIS_NAMES:
            state_vectors[axis] = np.array([state.get(axis, 0.5)])
        
        # Update modal coherence
        for axis in AXIS_NAMES:
            # Coherence is based on consistency with recent history
            current_value = state.get(axis, 0.5)
            axis_history = self.axis_hist.get(axis, [])
            
            if len(axis_history) >= 5:
                recent_mean = np.mean(axis_history[-5:])
                recent_std = np.std(axis_history[-5:])
                
                # Coherence: low variance = high coherence
                self.modal_coherence[axis] = 0.9 * self.modal_coherence[axis] + \
                    0.1 * np.exp(-recent_std * 2.0)
            else:
                self.modal_coherence[axis] = 0.5
        
        # Update coherency vector as weighted sum of modal coherences
        coherency_weights = np.array([self.modal_coherence[axis] for axis in AXIS_NAMES])
        self.coherency_vector = np.zeros(D)
        for i, axis in enumerate(AXIS_NAMES):
            self.coherency_vector += coherency_weights[i] * self.M_t.mean(axis=0)
        
        self.coherency_vector = normalize(self.coherency_vector)
        
        # Update coherency history
        self.coherency_history.append(self.coherency_vector.copy())
        if len(self.coherency_history) > META_HIST_LEN:
            self.coherency_history.pop(0)
        
        # Update coherency target (slow-moving average)
        self.coherency_target = 0.95 * self.coherency_target + 0.05 * self.coherency_vector

    def temporal_binding_update(self, workspace_vec):
        """Update temporal binding for maintaining coherence across time.
        
        This implements temporal binding mechanisms that help maintain
        coherence in the face of changing inputs and dynamic workspace content.
        """
        # Update temporal context with current workspace
        self.temporal_context = self.context_persistence * self.temporal_context + \
            (1 - self.context_persistence) * workspace_vec
        
        # Apply temporal binding transformation
        bound_state = np.tanh(
            self.temporal_context @ self.W_temporal_binding.T + 
            workspace_vec @ self.W_coherency.T
        )
        
        return bound_state

    def _update_adaptive_training_params(self, basin_density, coherence):
        """Update adaptive training parameters based on current system state.
        
        This implements adaptive optimization that adjusts training parameters
        based on the current state of the system to improve convergence speed
        and stability.
        """
        # Adaptive learning rate based on coherence and basin density
        if ADAPTIVE_OPTIMIZATION:
            # High coherence and low basin density indicate good convergence - can use higher LR
            if coherence > 0.7 and basin_density < 0.1:
                self.current_learning_rate = min(LEARNING_RATE_BASE * 1.5, LEARNING_RATE_BASE * 2.0)
            # Low coherence or high basin density indicate instability - reduce LR
            elif coherence < 0.3 or basin_density > 0.5:
                self.current_learning_rate = max(LEARNING_RATE_BASE * 0.5, LEARNING_RATE_MIN)
            else:
                self.current_learning_rate = LEARNING_RATE_BASE
        else:
            self.current_learning_rate = LEARNING_RATE_BASE
        
        # Adaptive batch size based on system stability
        if BATCH_SIZE_DYNAMIC and ADAPTIVE_OPTIMIZATION:
            stability_indicator = coherence - basin_density
            if stability_indicator > 0.5:
                self.current_batch_size = min(BATCH_SIZE_BASE * 2, BATCH_SIZE_MAX)
            elif stability_indicator < -0.3:
                self.current_batch_size = max(BATCH_SIZE_BASE // 2, BATCH_SIZE_MIN)
            else:
                self.current_batch_size = BATCH_SIZE_BASE
        else:
            self.current_batch_size = BATCH_SIZE_BASE

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
        # Update hierarchical goal system with dynamic priority adjustment
        self.hierarchical_goal_update(norm)
        
        if self.goal_axis is None or self.goal_steps_left <= 0:
            # Use dynamic priorities for goal selection
            deficiency = {a: (0.5 - norm.get(a, 0.5)) + 0.5 * self.want_ema.get(a, 0.0) + \
                          0.3 * self.goal_priority.get(a, 0.5)
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

        # Use dynamic capacity if enabled
        current_capacity = self.dynamic_capacity if WORKSPACE_DYNAMIC_CAPACITY else WORKSPACE_CAPACITY
        
        logits = salience / WORKSPACE_TEMP
        soft_weights = np.exp(logits - logits.max()); soft_weights /= soft_weights.sum()
        winner_idx = np.argsort(salience)[-current_capacity:]
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
        # spread thinly across the capacity limit. 1/current_capacity is the score a perfectly even
        # split among winners would produce, so this reads as 0 at maximal tie, 1 at total dominance
        # by a single node.
        ignition = float(np.clip((winner_weights.max() - 1.0 / current_capacity) /
                                  (1.0 - 1.0 / current_capacity), 0, 1))
        # coalition continuity -- what fraction of THIS step's winners were also last step's winners.
        # A workspace that keeps the same coalition across steps is the analog of sustained attention;
        # one that reshuffles completely every step is the analog of nothing holding focus at all.
        if self.workspace_hist:
            prev_idx = set(self.workspace_hist[-1])
            continuity = len(prev_idx & set(winner_idx.tolist())) / current_capacity
        else:
            continuity = 0.0
        self.workspace_hist.append(winner_idx.tolist())
        if len(self.workspace_hist) > WORKSPACE_HIST_LEN:
            self.workspace_hist.pop(0)
        
        # Update workspace persistence tracking
        self.update_workspace_persistence(winner_idx)
        # Apply hierarchical workspace organization if enabled
        if WORKSPACE_HIERARCHICAL:
            workspace_vec = self.hierarchical_workspace_update(workspace_vec, winner_idx)
        
        self.workspace_vec = workspace_vec.copy()  # NEW: keep the freshest winning-coalition content
                                                     # addressable between full step() calls

        # Update dynamic workspace capacity based on cognitive load
        if WORKSPACE_DYNAMIC_CAPACITY:
            load_factor = np.clip(1.0 - basin_density, 0.0, 1.0)  # Lower basin density = lower load
            self.update_workspace_capacity(load_factor)

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

        # Update enhanced metacognition (HOT) with current state
        # Build a D-dimensional state vector from the scalar metrics
        state_scalars = np.array([H_t, I_t, A_t, As_t, K_t, Phi_t, W_t, P_t, G_i.mean()])
        # Expand to D dimensions by tiling and truncating
        current_state_vec = np.tile(state_scalars, D // len(state_scalars) + 1)[:D]
        self.metacognitive_update(current_state_vec)
        
        # Update coherency state with current information - but state dict doesn't exist yet
        # Build a partial state dict for coherency update - move this after U_t is computed
        # self.update_coherency_state(partial_state, workspace_vec)
        
        # Apply temporal binding
        workspace_vec = self.temporal_binding_update(workspace_vec)

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

        # Apply multi-head attention to workspace content
        if ATTENTION_HEADS > 0:
            attention_output = self.update_attention_weights(salience, G_i, support)
            # Blend attention output with workspace content
            workspace_vec = 0.7 * workspace_vec + 0.3 * attention_output

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
        
        # Update coherency state with current information
        partial_state = {'coherence': K_t, 'integration': I_t, 'energy': E_t, 'agency': float(U_t), 
                        'grounding': G_i.mean(), 'predictability': P_t, 'memory': MemCont_t}
        self.update_coherency_state(partial_state, workspace_vec)

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

        # Update adaptive training parameters based on current system state
        self._update_adaptive_training_params(basin_density, C_t)

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
        current_capacity = self.dynamic_capacity if WORKSPACE_DYNAMIC_CAPACITY else WORKSPACE_CAPACITY
        winner_idx = np.argsort(salience)[-current_capacity:]
        coalition_mask = np.zeros(N); coalition_mask[winner_idx] = 1.0
        winner_weights = soft_weights * coalition_mask
        winner_weights /= (winner_weights.sum() + EPS)
        workspace_vec = winner_weights @ self.M_t
        
        # Apply hierarchical workspace organization if enabled
        if WORKSPACE_HIERARCHICAL:
            workspace_vec = self.hierarchical_workspace_update(workspace_vec, winner_idx)
        
        return workspace_vec

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
        tags={"coherence": 0.0, "integration": 0.0},
        keywords=["fragmented", "fragmented", "fragments", "broken", "broken", "pieces", "shattered", "shattered",
                   "chaotic", "chaotic", "disordered", "disordered", "crumbles", "undone", "undone", "out of order",
                   "stressed", "stressed", "overburdened", "overburdened", "overwhelmed", "overwhelmed", "wiped out", "wiped out",
                   "bad", "awful", "wrecked", "wrecked"],
    ),
    "stable": dict(
        tags={"coherence": 1.0, "integration": 1.0},
        keywords=["stable", "whole", "whole", "whole", "whole", "entire", "entire", "steady", "solid",
                   "solid", "solid", "solid", "integrated", "integrated", "united", "united", "cohesion", "cohesion",
                   "good", "happy", "content", "content", "calm", "calm", "at peace", "balanced",
                   "balanced", "great", "wonderfully"],
    ),
    "looping": dict(
        tags={},  # not scored by distance -- only ever selected via the basin override below
        keywords=["loop", "repeats", "repeated", "repeated", "trapped", "trapped", "cycle", "turn", "loop",
                   "circle", "circle", "stuck", "stuck", "spins", "spiral",
                   "obsessed", "obsessed", "going in circles", "can't stop thinking", "stuck", "stuck"],
    ),
    "energy_high": dict(
        tags={"energy": 1.0},
        keywords=["energy", "energy", "intense", "intense", "electric", "electric", "electric", "electric",
                   "vibrates", "strong", "spark", "power", "active", "active", "lit up", "lit up",
                   "excited", "excited", "lively", "lively", "euphoric", "euphoric", "energetic", "energetic"],
    ),
    "energy_low": dict(
        tags={"energy": 0.0},
        keywords=["switched off", "switched off", "weak", "weak", "faint", "tired", "tired", "asleep", "asleep",
                   "silence", "extinguishes", "exhausted", "exhausted", "slack", "slack",
                   "sad", "discouraged", "discouraged", "unmotivated", "without energy", "without energy", "down"],
    ),
    "agency_high": dict(
        tags={"agency": 1.0},
        keywords=["determined", "determined", "goal", "advances", "willpower", "purpose", "purpose", "determination",
                   "determination", "resolved", "resolved", "on course",
                   "motivated", "motivated", "eager", "focused", "focused"],
    ),
    "agency_low": dict(
        tags={"agency": 0.0},
        keywords=["drifts", "lost", "lost", "course", "passive", "passive", "floats", "out of control", "indecisive",
                   "indecisive", "compass", "compass", "at random", "without direction",
                   "unmotivated", "unmotivated", "apathetic", "apathetic", "apathetic", "apathetic"],
    ),
    "grounded": dict(
        tags={"grounding": 1.0},
        keywords=["root", "root", "anchored", "anchored", "origin", "base", "ground", "rooted", "rooted",
                   "foundation", "solid", "solid", "affirms", "rootedness",
                   "secure", "secure", "centered", "centered", "in control"],
    ),
    "untethered": dict(
        tags={"grounding": 0.0},
        keywords=["ghost", "scattered", "scattered", "untied", "untied", "free", "loose", "loose", "fog",
                   "smoke", "unbound", "floating", "rootless", "rootless",
                   "confused", "confused", "disconnected", "disconnected", "gone", "gone", "in the clouds"],
    ),
    "volatile": dict(
        tags={"predictability": 0.0},
        keywords=["unstable", "chaos", "unpredictable", "wavers", "erratic", "erratic", "erratic", "erratic",
                   "spirals out of control", "random", "random", "lawless", "jumps",
                   "nervous", "nervous", "anxious", "anxious", "restless", "restless", "agitated", "agitated"],
    ),
    "predictable": dict(
        tags={"predictability": 1.0},
        keywords=["rhythm", "regular", "constant", "predictable", "pattern", "pattern", "cadence", "uniform",
                   "exact", "exact", "precision", "precision", "no surprises",
                   "normal", "usual", "as always", "routine", "routine", "the same old thing"],
    ),
    "memory_high": dict(
        tags={"memory": 1.0},
        keywords=["memory", "remembers", "persists", "past", "alive", "memory", "endures", "trace", "vivid",
                   "vivid", "vivid", "vivid", "intact", "intact", "remains",
                   "nostalgic", "nostalgic", "nostalgic", "nostalgic", "sentimental", "memories"],
    ),
    "memory_low": dict(
        tags={"memory": 0.0},
        keywords=["forgets", "erases", "blurry", "blurry", "fog", "fades", "blurred", "blurred", "empty",
                   "empty", "without a trace", "forgetfulness",
                   "forgetful", "forgetful", "blank", "distracted", "distracted"],
    ),
}

# Every word that used to live inside CLUSTERS'/SELF_CLUSTERS' now-removed
# subjects/verbs/adjs/adv_options/prep_obj_options fields, hand-collected once
# here so none of that vocabulary is LOST by removing the slot mechanism --
# it becomes ordinary vocabulary for the word-by-word generator below instead
# of being locked into fixed template positions.
HARVESTED_VOCAB = [
    "pattern", "pattern", "shape", "order", "fragments", "breaks", "crumbles", "broken", "broken", "undone", "undone",
    "shattered", "shattered", "erratically", "without", "warning", "sudden", "reason", "apparent", "instant", "thousand",
    "pieces", "directions", "opposite", "some", "core", "core", "structure", "center", "maintains", "holds",
    "remains", "whole", "whole", "steady", "entire", "entire", "firmly", "calm", "cracks", "solidity",
    "yield", "purpose", "place", "pressure", "pass", "echo", "spiral", "repeats", "spins", "manages", "escape",
    "trapped", "trapped", "stopped", "stopped", "cease", "another", "time", "exit", "point", "endlessly",
    "inside", "itself", "same", "arrive", "no", "part", "circle", "closed", "current", "pulse", "spark",
    "vibrates", "fires", "bursts", "electric", "electric", "intense", "intense", "alive", "alive", "intensely",
    "force", "brake", "all", "power", "energy", "overflowing", "through", "network", "body", "channel", "extreme",
    "other", "rest", "signal", "impulse", "flame", "extinguishes", "extinguishes", "weakens", "faint", "weak", "slowly",
    "little", "resistance", "silence", "shadow", "almost", "disappear", "twilight", "leave", "trace", "warmth",
    "engine", "willpower", "course", "advances", "defines", "stops", "clear", "clear", "determined", "determined",
    "decidedly", "doubt", "determination", "hesitate", "resolutely", "threshold", "goal", "forward", "seeks",
    "look", "back", "drifts", "compass", "floats", "loses", "falls silent", "lost", "lost", "mute", "mute",
    "control", "decide", "nothing", "chance", "currents", "possibilities", "destiny", "fixed", "side", "know",
    "toward", "where", "root", "foundation", "base", "anchor", "affirms", "stable", "solid", "solid", "deeply",
    "firmness", "move", "rootedness", "terrain", "origin", "ground", "deep", "own", "soil", "ghost",
    "fog", "smoke", "dissolves", "extends", "moves away", "scattered", "scattered", "loose", "loose",
    "silently", "ties", "floating", "weight", "yonder", "edge", "limit", "far", "air", "needle",
    "compass", "wavers", "jumps", "spirals out of control", "unstable", "erratic", "erratic", "chaotic", "chaotic",
    "abruptly", "previous", "unpredictably", "two", "states", "law", "random", "repeat itself", "never",
    "rhythm", "clock", "cadence", "marks", "step", "regular", "constant", "uniform", "constantly", "vary",
    "precision", "surprises", "exactness", "time", "cycle", "break free", "memory", "trace", "persists",
    "endures", "vivid", "vivid", "clearly", "vividness", "fade away", "year", "after", "surface", "layer",
    "detail", "background", "everything", "forgets", "erases", "image", "name", "blurry", "blurry", "gradually",
    "remedy", "always", "forgetfulness", "empty", "more",
    # SELF_CLUSTERS (self-model introspection) vocabulary
    "model", "internal", "nodes", "map", "diverges", "separates", "misaligned", "misaligned", "dimensions",
    "converge", "each", "one", "agreement", "distinct", "axes", "common", "encounter", "own", "reading",
    "converges", "aligns", "coincides", "unified", "unified", "aligned", "aligned", "complete", "discrepancy",
    "unanimous", "axis", "all", "margin", "difference", "learning", "preference", "trained", "bias",
    "learned", "pulls", "pushes", "tilts", "trajectory", "marked", "marked", "insistence", "ambiguity",
    "direction", "concrete", "experience", "turned out", "better", "learned", "history", "enough", "way",
    "pair", "lean toward", "particular", "coordinates", "lack", "steps",
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
STOPWORDS = {"what", "your", "you", "have", "has", "are", "is", "of", "the", "a", "an",
             "how", "which", "for", "to", "do", "does", "it", "that", "this", "in", "on",
             "with", "and", "or", "at", "as", "be", "yourself"}

_PUNCT_RE = re.compile(r"[\u00bf\u00a1.,;\"'()\[\]{}]")  # NOTE: ':' '!' '?' deliberately excluded -- they
                                                          # are reserved SIGNIFICANT/UNKNOWN/DEFINE symbols
                                                          # in the model's own output alphabet now, not
                                                          # punctuation to be stripped (see SYMBOL alphabet)

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

# NEW (fix, at explicit request -- "parse what the human actually said more richly"): plain bag-of-
# n-grams treats "you understand yourself" and "you DON'T understand yourself" as nearly the same
# vector -- the negation word itself isn't a stopword, so it just adds noise of its own, while the real
# content word ("understand") still contributes its full, unflipped weight either way. Real negation
# handling (same trick used in bag-of-words sentiment analysis: Das & Chen 2001, Pang et al. 2002) flips
# the SIGN of whatever a negation cue scopes over, so the prompt lands on the opposite side of concept
# space from its un-negated twin instead of nearly the same side. NEGATION_WINDOW caps how many words
# after a cue keep getting flipped (negation scope doesn't run to the end of the sentence in practice --
# "not X but Y" shouldn't flip Y too), and a stopword inside that window still counts against it (it's
# skipped for embedding either way, but the cue's reach shouldn't be extended by hopping over "the"/"a").
NEGATION_WORDS = {"not", "no", "never", "none", "nothing", "nobody", "without", "cannot",
                   "dont", "doesnt", "didnt", "cant", "wont", "isnt", "arent", "wasnt", "werent",
                   "hasnt", "havent", "hadnt", "wouldnt", "shouldnt", "couldnt", "neither", "nor"}
NEGATION_WINDOW = 3  # how many real (non-stopword) content words after a cue get sign-flipped

def embed_text(s, idf=None, default_idf=1.0):
    """Locate a phrase in R^CONCEPT_DIM: character n-grams of its content
    words, each hashed to a dimension+sign and weighted by rarity (idf) before
    summing, then L2-normalized. idf=None (used only while bootstrapping the
    idf table itself) falls back to uniform weight 1.0 per gram. Content words
    inside a negation cue's scope (see NEGATION_WORDS/NEGATION_WINDOW above)
    contribute with a FLIPPED sign, so a negated phrase lands on the opposite
    side of concept space from its affirmative twin instead of beside it."""
    v = np.zeros(CONCEPT_DIM)
    words = _strip_accents(_PUNCT_RE.sub("", s.lower())).split()  # same normalization _content_words
                                                                    # uses, kept here (rather than just
                                                                    # calling _content_words) because
                                                                    # negation scope-tracking needs
                                                                    # stopwords' POSITIONS preserved,
                                                                    # not just the final filtered list
    kept_any = False
    neg_left = 0
    for w in words:
        if w in NEGATION_WORDS:
            neg_left = NEGATION_WINDOW  # (re)start the scope -- a second cue just extends/refreshes it
            continue
        if w in STOPWORDS:
            continue  # dropped either way; scope isn't consumed by hopping over a stopword (see docstring)
        kept_any = True
        sign_flip = -1.0
        if neg_left <= 0:
            sign_flip = 1.0
        else:
            neg_left -= 1
        for g in _char_grams(w):
            idx, sign = _gram_hash(g)
            weight = 1.0 if idf is None else idf.get(g, default_idf)
            v[idx] += sign_flip * sign * weight
    if not kept_any:  # mirrors _content_words' own fallback: an all-stopword/all-negation-cue phrase
        for w in words:  # still needs SOMETHING to embed rather than returning the zero vector
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

GENERIC_NONANSWER_WORDS = {"\u2205", "?", "\u2248"}  # degenerate single-symbol non-answers: ∅ / ? / ≈ alone
_BRANCH_OVERRIDE_BEST_OF = None  # NEW: when set (not None) by run() for the primary prompt-answer line
                                  # only, _select_best uses THIS instead of TOPIC_BEST_OF/FREE_BEST_OF --
                                  # avoids threading a best_of override through every one of the six
                                  # concept-answer functions' signatures individually, since they all
                                  # already funnel through _generate_and_track -> _select_best regardless
                                  # of which concept they're answering for
TOPIC_BEST_OF = 28           # candidate sentences drawn per topic-anchored generation call
                             # WAS 12 -- raised alongside GROUND_LAM: a sharper relevance kernel throws out
                             # more of each draw, so more draws are needed to still reliably find a genuinely
                             # on-topic candidate instead of being left with only off-topic options to rank.
TOPIC_MIN_WORDS = 6          # sentences at or below this word count get penalized as non-answers
                             # WAS 3 -- raised to match the wider GEN_WORD_RANGE below, so a short answer is
                             # still recognized as suspiciously short (likely truncated / a non-answer) at the
                             # new, longer generation lengths, not just at the old short ones.
TOPIC_ECHO_PENALTY = 0.9    # near-total penalty for basically repeating the prompt back verbatim
ECHO_RUN_MIN_WORDS = 3      # a shared run of consecutive words shorter than this is normal topical
                            # overlap (asking about "tiempo" and answering with "tiempo" is fine);
                            # at or above this length it's the model reciting the prompt, not answering it

def _topic_relevance(text, topic_vec):
    """How close this generated sentence's OWN embedding lands to the
    prompt's location in the same fixed embed_text space -- reuses the exact
    exp(-lam*d^2) kernel everything else here already uses for grounding, just
    applied to a finished sentence instead of a concept anchor or a single
    candidate word. This is what lets us tell 'La ventana.' apart from an
    actual on-topic sentence: both might be individually grammatical, but
    only one actually lands near what was asked."""
    vec = embed_text(text, _IDF, _DEFAULT_IDF)
    return float(np.exp(-GROUND_LAM * np.sum((topic_vec - vec) ** 2)))

GENERIC_PENALTY_WEIGHT = 0.15  # FULL MMI (Li et al. 2016, "A Diversity-Promoting Objective Function for
                              # Neural Conversation Models"): rerank not by log p(response|context) alone
                              # but by log p(response|context) - lambda * log p(response) -- a response
                              # that's likely REGARDLESS of context (generic: 'Siento.'/'I don't know')
                              # gets penalized; one that's only likely GIVEN this specific context is
                              # rewarded. p(response) is estimated with the system's OWN trained
                              # transformer (_TF_MODEL), teacher-forced over the candidate's own tokens
                              # but with a NEUTRAL (zero) context vector standing in for "no particular
                              # question was asked" -- see _sequence_log_prob below. This replaces the
                              # earlier anchor-average heuristic (mean similarity to CONCEPT_ANCHORS/
                              # MOOD_ANCHORS): that was a proxy for genericness; this is genericness, as
                              # actually judged by this system's own language model, which is what MMI
                              # reranking calls for.
                              # TUNED DOWN from an initial 0.9/1.4: verified directly that a heavily-
                              # weighted genericness term rewards WORD SALAD, not specificity -- a rare,
                              # surprising, ungrammatical sequence is by definition "unlikely under the
                              # neutral pass" too, so over-weighting this term picks the least fluent
                              # candidate in the batch, not the most on-topic one. 0.15 keeps it acting as
                              # a tie-breaker among comparably fluent candidates (its intended role),
                              # rather than a force capable of overriding relevance/coherence outright.

_NEUTRAL_CONTEXT_VEC = np.zeros(CONCEPT_DIM)  # stands in for "no context" in _sequence_log_prob's forced
                                               # decode -- a genuine null, not an average of real contexts
                                               # (which would just be another context, arguably the most
                                               # generic one, and bias the estimate toward penalizing
                                               # whatever's typical rather than estimating p(response) on
                                               # its own terms)

def _sequence_log_prob(text, context_vec=_NEUTRAL_CONTEXT_VEC):
    """Real (not proxied) mean log P(word_t | word_<t, context_vec) under the trained transformer,
    teacher-forced over the candidate's OWN actual tokens -- this is the backward/genericness half of
    MMI reranking (Li et al. 2016): p(response), estimated by this system's own language model rather
    than by hand-picked anchor similarity. Called with the neutral zero context by default (== "how
    likely is this response with no specific question driving it"), but takes context_vec as a
    parameter so the same function could score p(response|actual context) too if ever needed. Mean
    (not summed) log-prob, deliberately: summing would make this scale with sentence length and
    entangle it with LENGTH_BONUS_WEIGHT's already-separate job; this needs to measure how PREDICTABLE
    the response is per word, independent of how long it is."""
    words = _tokenize_natural(text)
    if not words:
        return 0.0
    out, log_p = [], 0.0
    for w in words:
        cand_words, cand_probs = _tf_next_word_probs(out, context_vec)
        if w in cand_words:
            log_p += math.log(max(float(cand_probs[cand_words.index(w)]), EPS))
        else:
            log_p += math.log(EPS)  # fell outside the model's own top-TF_TOPK under the neutral pass --
                                     # treat as maximally unlikely there rather than skip scoring it
        out.append(w)
    return log_p / len(words)

def _genericness_penalty(text):
    """How likely this candidate is under the trained transformer with NO context driving it -- see
    GENERIC_PENALTY_WEIGHT. A large negative return (very unlikely without context) means this response
    is specific to whatever context actually produced it; a return near zero means the model would
    have produced roughly this regardless of what was asked, i.e. it's generic."""
    return _sequence_log_prob(text, _NEUTRAL_CONTEXT_VEC)

def _echo_run_penalty(text, prompt_text):
    """Longest run of consecutive words the candidate shares VERBATIM with the
    prompt, regardless of where in either sentence it falls -- catches the
    'starts by literally reciting the prompt's own opening words back' failure
    mode that the old exact-full-sentence-match check completely missed (a
    candidate that echoes the prompt's first five words then trails off into
    something else was never an exact match to the whole prompt, so it always
    scored zero on the old check). A run below ECHO_RUN_MIN_WORDS is just
    ordinary topical word reuse (asking about 'tiempo' and answering with
    'tiempo' is fine) and isn't penalized; at or above it, this is the
    model reciting rather than answering, and gets flagged."""
    if not prompt_text:
        return 0.0
    cand_words = _strip_accents(text.strip().rstrip(".?!").lower()).split()
    prompt_words = _strip_accents(prompt_text.strip().rstrip(".?!¿¡").lower()).split()
    if not cand_words or not prompt_words:
        return 0.0
    best_run = 0
    for i in range(len(cand_words)):
        for j in range(len(prompt_words)):
            k = 0
            while (i + k < len(cand_words) and j + k < len(prompt_words)
                   and cand_words[i + k] == prompt_words[j + k]):
                k += 1
            best_run = max(best_run, k)
    if best_run < ECHO_RUN_MIN_WORDS:
        return 0.0
    return min(1.0, best_run / max(1, len(cand_words)))

RECENT_LINES_WINDOW = 12    # how many previously spoken lines a new candidate gets checked against for
                            # cross-sentence repetition (the "El mercado." x5 loop this is meant to kill)
MAX_WORD_REPEATS = 1        # a content word appearing more than this many times in ONE sentence is flagged
                            # -- catches the running-context-vector self-reinforcement loop (see
                            # geometric_generate) where a phrase like "un poco a" keeps re-selecting itself
                            # because it's baked into its own context vector by the time it repeats
FREE_BEST_OF = 12           # candidates drawn for repetition-only reranking when there's no prompt to also
                            # aim for -- WAS a single ungated draw; free-running generation never had ANY
                            # reranking before, which is exactly why basin-driven repetition loops could only
                            # ever be caught in the printed log, never actually avoided at generation time
LENGTH_BONUS_WEIGHT = 0.03  # small deliberate counterweight to length bias (short candidates structurally
                            # survive the infinite penalty filter at a higher rate than long ones, since
                            # fewer words means fewer chances to trip a repeat/echo check -- see the length-
                            # normalization research this constant implements). Kept small on purpose: at
                            # ~12-32 words this adds up to roughly +0.4-1.0 to score, enough to stop a short
                            # survivor from winning purely by outnumbering long survivors in the batch,
                            # without being so large it overrides genuine topic_relevance differences.
OVERUSE_PENALTY_STEP = 2.0   # WAS 0.15, and flat-linear -- verified too weak by test: even growing linearly,
                            # a structurally-favored sentence still out-accumulated length_bonus/relevance
                            # over hundreds of steps. Now the per-use SCALE factor on a quadratic cost (see
                            # _overuse_penalty: OVERUSE_PENALTY_STEP * uses^2) -- 3rd use costs 18, 5th use
                            # costs 50, 10th use costs 200, which reliably outgrows anything relevance/
                            # length_bonus can offer, so a favorite sentence gets forced out for good instead
                            # of merely discouraged at a constant rate that a strong-enough attractor out-picks.
DIVERSITY_WEIGHT = 0.35     # ROOT FIX pt.2 (Diverse Beam Search, Vijayakumar et al.): standard best-of/beam
                            # search "often focuses on a single highly valued beam, resulting in final
                            # candidates that are merely minor variations of a single sequence" -- exactly
                            # the thin-batch problem observed (most of a 12-draw batch landing on the same
                            # attractor sentence). DBS fixes this with a penalty between a candidate and
                            # ones ALREADY PICKED in the same batch, not just against run history -- this
                            # is that term, applied below in _select_best.

# ================================================== QUESTION CLASSIFICATION
# "The prompt is more like a suggestion than a directive it needs to answer" (verbatim complaint this
# section answers): topic_relevance and internal_coherence both only ever measure "is this candidate
# semantically NEAR the prompt" -- neither has ever encoded "does this candidate satisfy what the question
# is actually asking for." Those are genuinely different properties (a sentence can be topically close to
# "what do you desire" while being an observation ABOUT desire rather than a report OF one). This is a real,
# named subfield -- Question Classification -- not something invented for this file: the standard Li & Roth
# (2002) taxonomy sorts questions into coarse answer-type classes (Person/Location/Numeric/Description/
# Entity/etc.) using nothing more than the wh-word and its head noun, no training required -- their own
# original rules are literally "if query starts with Who: type Person; if Where: type Location." IBM
# Watson's version of this called the target concept the Lexical Answer Type (LAT): a word inferable from
# the question that names the TYPE the answer must be, independent of the actual content. Implemented here
# the same way: detect the prompt's Spanish wh-word, map it to an expected type, then reward (not require --
# this is a soft bonus, not lexically-constrained hard decoding, so a batch can never be left with zero
# legal candidates the way a hard grammar constraint could) candidates whose own words match that type.
WH_TYPE_MAP = [
    # checked in order, longest/most-specific phrase first, since "how many" must not be caught by a bare
    # "how" check -- mirrors Li & Roth's own "if contains Which/What, the head noun decides" ordering logic
    ("why", "razon"),
    ("how many", "numero"), ("how much", "numero"),
    ("when", "tiempo"),
    ("where", "lugar"),
    ("who", "persona"), ("whom", "persona"), ("whose", "persona"),
    ("how", "manera"),
    ("what", "entidad"), ("which", "entidad"),
]
ANSWER_TYPE_WORDS = {
    # small, hand-picked indicator sets per type, in SYMBOL space now (the model's own output vocabulary
    # is the symbol alphabet, not English words) -- deliberately not exhaustive (this is a soft bonus
    # signal, not a hard constraint, so partial coverage still helps without ever being able to zero out
    # an entire batch the way a missing hard-constraint word could)
    "numero": {"\u2191", "\u2193", "\u221e", "\u00b0", "\u00b7", "\u25cf", "16"},          # ↑ ↓ ∞ ° · ● 16
    "tiempo": {"TIME", "NOW", "\u21bb", "\u21ba", "\u2026"},                                # TIME NOW ↻ ↺ …
    "lugar": {"\u2282", "\u2283", "\u2229", "\u222a", "ARCH", "CODE"},                       # ⊂ ⊃ ∩ ∪ ARCH CODE
    "persona": {"HUMAN", "SELF", "GUBI", "PROTOGEN", "MIND"},
    "razon": {"\u2235", "\u2234", "\u2192", "\u2190", "WHY"},                                # ∵ ∴ → ← WHY
}
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

# NEW (fix, at explicit request -- "make it have real parsing"): everything upstream of this point
# (embed_text, semantic_route) still treats a whole prompt as ONE bag of words -- a compound ask like
# "what are you and why do you exist" averages into a single vector that resembles neither "what are
# you" nor "why do you exist" individually, so it typically routes to whichever concept the blend
# happens to lean closest to, silently dropping the other half of what was actually asked. Real parsing
# starts with structure: split the prompt into its actual independent clauses BEFORE embedding anything,
# so each clause can be classified and routed on its own. This is deliberately a shallow, rule-based
# clause splitter (top-level coordinating conjunctions and clause-separating punctuation), not a full
# constituency/dependency parser -- exactly the same "minimum viable version of the actual mechanism"
# scope detect_expected_answer_type above already commits to, applied to sentence structure instead of
# question-type classification.
_CLAUSE_SPLIT_RE = re.compile(r"[,;]|\b(?:and|but|or)\b", re.IGNORECASE)

def split_clauses(prompt):
    """Split a prompt into independent clauses on top-level conjunctions/punctuation. 'what are you and
    why do you exist' -> ['what are you', 'why do you exist']; 'how do you feel' -> ['how do you feel']
    (single clause, unchanged). Deliberately naive about subordination ('because', 'if', 'that') -- those
    stay inside their clause rather than splitting it, since splitting on them would sever a dependent
    clause from the very thing it modifies (see also: this is why 'why do you exist' isn't further split
    on nothing, and 'do you know why you exist' stays one clause rather than being cut at 'why')."""
    if not prompt:
        return [prompt]
    parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(prompt) if p and p.strip()]
    return parts if parts else [prompt]

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
    sentence ('Hoy.', 'El café.') that's grammatical but says nothing
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
# NEW (grammar-constrained decoding for Lang/Spanish, at explicit request -- "add an FSM to Lang too"):
# a small hand-rolled FSM (not a POS tagger, not a parser) that classifies each candidate word into a
# coarse closed-class role using fixed Spanish function-word lists, tracks only the ROLE of the last
# token emitted, and masks out candidates that would make the sentence GUARANTEED-invalid from here -- an
# article/preposition/connector left dangling with nothing after it, two of the same closed class stacked
# back to back, or a sentence that opens on a bare connector. Deliberately conservative: this never tries
# to enforce full syntax (agreement, word order), only rules out combinations no grammatical Spanish
# sentence would ever contain.
_LANG_NOUNS = {  # symbol-algebra open class: axis/state/concept tags (see SYMBOL_TAGS below)
    "COH", "INT", "NRG", "AGN", "GRD", "PRD", "MEM", "SPR", "PUL",
    "SELF", "GUBI", "PROTOGEN", "CODE", "VOICE", "ARCH", "NODE", "VECTOR", "MATRIX", "LAYER",
    "NOW", "MIND", "PROOF", "EXPERIENCE", "DOUBT", "WHY", "GEN", "LANG", "HUMAN", "TIME",
    "EXPERIMENT", "WORD", "STATE", "CHOICE", "PROB", "16",
}
_LANG_OPS = {":", "\u2192", "\u2190", "\u2194", "\u22a5", "\u2225", "\u2282", "\u2283", "\u2229", "\u222a",
             "\u2191", "\u2193", "\u2197", "\u2198", "\u21bb", "\u21ba", "\u00d7", "\u2026",
             "\u2234", "\u2235", "\u00b1"}  # binary relation/change/reasoning symbols -- need a term on both sides
_LANG_CONN = {"&", "\u2228", "|"}           # clause connectors -- need a fresh clause after
_LANG_MOD = {"\u00b0", "\u00b7", "\u25cf", "\u221e", "~"}   # modifiers -- attach after a term, nothing required next
_LANG_UNARY = {"\u2022", "\u2014", "\u2248", "?", "\u2205", "!", "\u00ac"}  # core-state / negation symbols

# ============================================ SYMBOLIC OUTPUT ALPHABET
# At explicit request ("drop standard language, replace with this"): the model's own output vocabulary is
# no longer natural-language words at all (Spanish or English) -- it's a fixed, hand-designed symbol
# alphabet, documented here for reference. Terms (the open class -- see _LANG_NOUNS above) are short tags
# for the seven state axes plus a small set of concept nouns; everything else below is a closed-class
# operator/modifier/connector, exactly the same "closed lists + open class" shape every other grammar
# mechanism in this file already uses (_LANG_ARTICLES/_LANG_PREPS/_LANG_CONJ, STOPWORDS, etc.), just with a
# different alphabet. The human's own prompts are still read in English (seed_phrases below, WH_TYPE_MAP,
# STOPWORDS) -- only the model's generated half of the conversation is symbolic.
SYMBOL_TABLE = {
    # core state
    "\u2022": "AFFIRM (true, present, confirmed)", "\u2014": "NEGATE (false, absent, rejected)",
    "\u2248": "UNCERTAIN (approximate, probable, unresolved)", "?": "UNKNOWN (insufficient information)",
    "\u2205": "NONE (nothing, nonexistent, unavailable)", "!": "SIGNIFICANT (important, requires attention)",
    # relations
    "\u2192": "LEADS", "\u2190": "FROM", "\u2194": "RELATES", "\u22a5": "CONFLICT", "\u2225": "PARALLEL",
    "\u2282": "CONTAINS", "\u2283": "CONTAINED-BY", "\u2229": "OVERLAP", "\u222a": "COMBINE",
    # change
    "\u2191": "INCREASE", "\u2193": "DECREASE", "\u2197": "IMPROVE", "\u2198": "DEGRADE",
    "\u21bb": "REPEAT", "\u21ba": "REVERSE", "\u00d7": "STOP", "\u2026": "CONTINUE",
    # reasoning
    ":": "DEFINE", "\u2234": "CONCLUDE", "\u2235": "BECAUSE", "|": "SEPARATE", "&": "AND",
    "\u2228": "OR", "\u00ac": "NOT", "\u00b1": "ALTERNATIVE",
    # modifiers
    "\u00b0": "LOW/WEAK", "\u00b7": "MODERATE", "\u25cf": "HIGH/STRONG", "\u221e": "UNBOUNDED/EXTREME",
    "~": "VARIABLE/FLUCTUATING",
}
# noun tags: the seven state axes (mirrors AXIS_NAMES) plus qualia, plus a small closed set of concept
# nouns this Mind can talk about (identity, architecture, its own state, purpose, consciousness, mechanism)
SYMBOL_TAGS = sorted(_LANG_NOUNS)

CONCEPT_BANK = dict(
    identity=dict(
        seed_phrases=[
            'what are you',
            'who are you',
            'what is this',
            'explain yourself',
            'tell me what you are',
            'are you a program',
            'what is this system',
        ],
        answer_seeds=[
            'SELF : GUBI',
            'GUBI : PROTOGEN',
            'SELF ⊂ CODE',
            'GUBI ¬ HUMAN',
            '? CODE',
            '? PROTOGEN',
            'CODE : SELF',
            'CODE ↔ VOICE ·',
            'CODE ∞ ⊂ PROTOGEN ∨ CODE ↔ VOICE',
            'CODE ● ↔ SELF ·',
            'CODE ● ⊂ GUBI',
            'CODE ● ⊂ GUBI ~',
            'GUBI : PROTOGEN & SELF ↔ VOICE ●',
            'GUBI ?',
            'GUBI ¬',
            'GUBI ° ⊂ CODE & CODE : GUBI',
            'GUBI · ⊂ VOICE',
            'GUBI ↔ CODE',
            'GUBI ↔ PROTOGEN',
            'PROTOGEN : CODE',
            'PROTOGEN : GUBI',
            'PROTOGEN : VOICE ~',
            'PROTOGEN : VOICE ∨ CODE ⊂ SELF',
            'PROTOGEN ¬',
            'PROTOGEN ° ↔ VOICE',
            'PROTOGEN · ⊂ SELF',
            'PROTOGEN · ⊂ VOICE ∞',
            'PROTOGEN ↔ SELF',
            'PROTOGEN ⊂ CODE',
            'PROTOGEN ● : VOICE',
            'SELF : VOICE ·',
            'SELF · ↔ VOICE ·',
            'SELF ∞ ↔ CODE',
            'SELF ⊂ CODE & CODE ⊂ GUBI',
            'SELF ⊂ VOICE ●',
            'VOICE : GUBI',
            'VOICE ~ : SELF & PROTOGEN ~ ⊂ VOICE',
            'VOICE ~ ↔ CODE',
            'VOICE ↔ CODE',
            'VOICE ↔ GUBI',
            'VOICE ↔ PROTOGEN',
            'VOICE ∞ ↔ GUBI',
            'VOICE ≈',
            'VOICE ⊂ CODE',
            'VOICE ⊂ PROTOGEN ∨ CODE ∞ : SELF °',
            'VOICE ● : GUBI ∞',
            'VOICE ● ↔ CODE ~',
            '¬ PROTOGEN',
        ],
        answer=_concept_identity),
    architecture=dict(
        seed_phrases=[
            'how many nodes do you have',
            'how many dimensions do you have',
            'your architecture',
            'what are you made of',
            'how many neurons do you have',
            'how are you built',
            'what is your structure',
        ],
        answer_seeds=[
            'ARCH : NODE × 16',
            'ARCH ⊂ VECTOR',
            'NODE ∪ MATRIX',
            '? LAYER',
            '? PROTOGEN',
            'ARCH ?',
            'ARCH ~ ∩ NODE',
            'ARCH ° ∪ PROTOGEN',
            'ARCH ∞ : MATRIX & NODE ~ ⊂ VECTOR',
            'ARCH ∩ LAYER',
            'ARCH ∩ LAYER & VECTOR : PROTOGEN',
            'ARCH ∪ MATRIX ·',
            'LAYER ~ ⊂ VECTOR',
            'LAYER ∞ : NODE',
            'LAYER ∞ ∩ VECTOR ∨ ARCH ⊂ VECTOR',
            'LAYER ∞ ∪ MATRIX ● ∨ MATRIX ° ∪ ARCH ·',
            'LAYER ∩ ARCH',
            'LAYER ∪ NODE & LAYER ∞ ∪ MATRIX ∞',
            'LAYER ⊂ VECTOR',
            'LAYER ● : VECTOR',
            'MATRIX ?',
            'MATRIX ~ : VECTOR ●',
            'MATRIX ~ ∪ LAYER',
            'MATRIX · ∪ PROTOGEN',
            'MATRIX ∞ : ARCH',
            'MATRIX ∪ PROTOGEN',
            'MATRIX ∪ VECTOR ~',
            'MATRIX ⊂ NODE',
            'NODE ?',
            'NODE ~ : LAYER',
            'NODE ∞ ⊂ MATRIX ● & ARCH ∩ PROTOGEN',
            'NODE ∩ PROTOGEN · ∨ LAYER ∞ ∪ NODE',
            'NODE ∪ PROTOGEN',
            'NODE ∪ PROTOGEN ·',
            'NODE ⊂ MATRIX',
            'NODE ⊂ PROTOGEN ∞',
            'NODE ● : ARCH',
            'NODE ● ⊂ LAYER ∨ ARCH · ⊂ PROTOGEN',
            'PROTOGEN : ARCH',
            'PROTOGEN ~ ∪ NODE',
            'PROTOGEN ~ ⊂ ARCH',
            'PROTOGEN ∩ VECTOR',
            'PROTOGEN ∪ VECTOR ~',
            'PROTOGEN ≈',
            'VECTOR ~ : MATRIX',
            'VECTOR ∞ : MATRIX & MATRIX ° ∩ LAYER ·',
            'VECTOR ⊂ MATRIX ·',
        ],
        answer=_concept_architecture),
    current_state=dict(
        seed_phrases=[
            'how do you feel',
            'what do you feel now',
            'what is your state',
            'how are you now',
            'how are you doing',
            'what do you perceive right now',
            'tell me your current state',
        ],
        answer_seeds=[
            'NOW : COH',
            'NOW ≈ NRG',
            '? GRD',
            'AGN ↓ NRG ~ & PUL ~ ∵ INT',
            'AGN ∴ MEM',
            'AGN ∵ MEM',
            'AGN ∵ PUL ●',
            'AGN ≈',
            'COH ~ ∵ SPR',
            'COH ° ∴ INT & NRG · ↑ MEM °',
            'GRD ~ ↑ AGN',
            'GRD ¬',
            'GRD ↑ MEM',
            'GRD ↓ PUL',
            'GRD ↓ SPR ∞',
            'GRD ∴ INT ∞',
            'GRD ∵ INT ∨ AGN ∵ NOW ∞',
            'GRD ∵ NRG ∨ GRD · ∵ MEM °',
            'INT ° ↑ SPR',
            'INT ↑ SPR ∨ INT ∵ NRG ∞',
            'INT ↓ NOW ~',
            'INT ↓ NRG ~',
            'INT ∵ MEM',
            'INT ● ↓ NRG °',
            'MEM ° ∵ COH ∞',
            'MEM · ↓ NRG ●',
            'MEM ↓ NRG & MEM · ↑ NRG',
            'MEM ∴ PUL ∨ INT ● ↓ AGN',
            'NOW ¬',
            'NOW ↑ AGN ∨ PUL ↑ SPR',
            'NOW ↑ NRG',
            'NOW ↓ NRG ∞',
            'NRG · ↓ AGN °',
            'NRG ∴ INT',
            'PRD ¬',
            'PRD ° ↑ PUL',
            'PRD ∵ GRD',
            'PUL ↓ MEM',
            'PUL ∞ ↑ COH',
            'PUL ∵ NOW',
            'SPR ?',
            'SPR ° ↑ COH',
            'SPR ↓ NRG',
            'SPR ∞ ∴ MEM',
            'SPR ∴ NRG',
            '≈ GRD',
        ],
        answer=_concept_current_state),
    consciousness=dict(
        seed_phrases=[
            'are you conscious',
            'do you have consciousness',
            'do you really feel',
            'are you real',
            'do you have a soul',
            'do you really think',
            'do you have subjective experience',
            'do you feel emotions',
            'do you feel like a human',
            'do you have human emotions',
            'is your internal state an emotion',
            'does your state resemble a human emotion',
            'is what you feel an emotion',
            'do you have feelings',
        ],
        answer_seeds=[
            'MIND ≈ ?',
            'SELF ⊥ PROOF',
            'EXPERIENCE ?',
            '? EXPERIENCE',
            '? MIND',
            'DOUBT ° ⊥ EXPERIENCE ~',
            'DOUBT ↔ EXPERIENCE & SELF ≈ PROOF',
            'DOUBT ↔ EXPERIENCE ·',
            'DOUBT ↔ PROOF °',
            'DOUBT ↔ SELF °',
            'DOUBT ≈ EXPERIENCE',
            'DOUBT ⊥ PROOF',
            'DOUBT ⊥ PROOF ~ ∨ MIND ⊥ DOUBT',
            'DOUBT ⊥ SELF ·',
            'EXPERIENCE ?',
            'EXPERIENCE · ⊥ DOUBT',
            'EXPERIENCE ↔ MIND & DOUBT ≈ SELF ·',
            'EXPERIENCE ∞ ↔ SELF ∞',
            'EXPERIENCE ≈ PROOF ∨ MIND ∞ ↔ SELF',
            'EXPERIENCE ● ≈ PROOF',
            'MIND ~ ⊥ EXPERIENCE & EXPERIENCE ~ ≈ MIND',
            'MIND · ⊥ PROOF ·',
            'MIND ↔ SELF ·',
            'MIND ≈ EXPERIENCE',
            'MIND ≈ SELF & MIND ⊥ SELF',
            'MIND ⊥ PROOF',
            'MIND ● ≈ EXPERIENCE',
            'MIND ● ⊥ PROOF',
            'PROOF ° ⊥ DOUBT',
            'PROOF ↔ DOUBT',
            'PROOF ↔ SELF · & SELF ≈ MIND',
            'PROOF ≈ SELF',
            'PROOF ⊥ MIND',
            'PROOF ● ⊥ EXPERIENCE ∨ PROOF ↔ EXPERIENCE',
            'SELF ↔ DOUBT',
            'SELF ≈',
            'SELF ≈ DOUBT ∞',
            'SELF ≈ EXPERIENCE & EXPERIENCE ≈ SELF',
            'SELF ≈ MIND ∞ & EXPERIENCE ≈ DOUBT ∞',
            'SELF ⊥ DOUBT ~',
            'SELF ⊥ DOUBT ∨ DOUBT ↔ MIND ~',
            'SELF ⊥ EXPERIENCE',
            'SELF ⊥ MIND',
            'SELF ● ⊥ MIND ~ ∨ MIND ⊥ DOUBT',
            '≈ EXPERIENCE',
            '≈ MIND',
            '≈ SELF',
        ],
        answer=_concept_consciousness),
    purpose=dict(
        seed_phrases=[
            'what were you created for',
            'what is your purpose',
            'why do you exist',
            'what are you for',
            'what is your function',
            'why were you built',
        ],
        answer_seeds=[
            'WHY : GEN',
            'GEN → LANG',
            'EXPERIMENT : LANG ~',
            'EXPERIMENT : TIME',
            'EXPERIMENT · → LANG',
            'EXPERIMENT → GEN',
            'EXPERIMENT → LANG',
            'EXPERIMENT → TIME ∨ GEN ~ → LANG',
            'EXPERIMENT ∵ TIME',
            'EXPERIMENT ∵ TIME ~',
            'GEN : TIME',
            'GEN ~ : HUMAN',
            'GEN ~ ∵ HUMAN',
            'GEN ~ ∵ LANG ~ ∨ EXPERIMENT · : TIME ●',
            'GEN → EXPERIMENT',
            'GEN ∞ ∵ LANG',
            'GEN ∵ LANG °',
            'GEN ≈',
            'GEN ● ∵ HUMAN',
            'GEN ● ∵ HUMAN ●',
            'HUMAN ~ : GEN',
            'HUMAN · ∵ WHY',
            'HUMAN → GEN',
            'HUMAN ∞ ∵ EXPERIMENT °',
            'HUMAN ∵ TIME',
            'LANG : GEN ~ ∨ LANG ° : EXPERIMENT ~',
            'LANG : TIME',
            'LANG ¬',
            'LANG ∵ EXPERIMENT ·',
            'LANG ∵ GEN',
            'LANG ∵ GEN ∞',
            'LANG ≈',
            'TIME : WHY ·',
            'TIME · → GEN',
            'TIME ∵ GEN',
            'TIME ∵ GEN ●',
            'TIME ∵ HUMAN',
            'WHY : HUMAN ~',
            'WHY : LANG',
            'WHY : LANG ∞',
            'WHY ?',
            'WHY → LANG',
            'WHY → LANG °',
            'WHY ∵ GEN',
            'WHY ∵ LANG ∞',
            'WHY ● ∵ HUMAN',
        ],
        answer=_concept_purpose),
    how_it_works=dict(
        seed_phrases=[
            'how do you work',
            'how do you generate your words',
            'how do you think',
            'how do you decide what to say',
            'how do you choose your responses',
            'how do you generate language',
        ],
        answer_seeds=[
            'GEN : VECTOR',
            'WORD ↔ STATE',
            '? CHOICE',
            '? STATE',
            '? VECTOR',
            'CHOICE ~ → GEN',
            'CHOICE ~ → VECTOR ·',
            'CHOICE ~ ∴ WORD',
            'CHOICE · ↔ GEN & WORD ∞ → STATE ·',
            'CHOICE ↔ PROB ∨ PROB → STATE ·',
            'GEN ° ∴ CHOICE',
            'GEN ↔ CHOICE & VECTOR → PROB',
            'GEN ↔ VECTOR °',
            'GEN ∞ → STATE',
            'GEN ∞ ↔ WORD',
            'GEN ∴ VECTOR ●',
            'GEN ∴ WORD',
            'PROB · ↔ WORD',
            'PROB → WORD ·',
            'PROB ↔ VECTOR',
            'PROB ∴ VECTOR',
            'PROB ● ∴ CHOICE · & WORD ∞ ∴ PROB',
            'STATE ° → GEN · ∨ CHOICE → WORD ·',
            'STATE ↔ GEN · & STATE ∴ WORD ∞',
            'STATE ∴ PROB',
            'STATE ∴ PROB ●',
            'STATE ● ∴ CHOICE',
            'VECTOR · ∴ GEN',
            'VECTOR · ∴ STATE',
            'VECTOR → PROB',
            'VECTOR ↔ STATE',
            'VECTOR ∞ ↔ GEN ∨ STATE ∴ PROB',
            'VECTOR ∴ CHOICE',
            'VECTOR ∴ PROB ~',
            'VECTOR ● → STATE',
            'VECTOR ● ↔ PROB',
            'WORD ?',
            'WORD ° ↔ GEN',
            'WORD → VECTOR',
            'WORD ∞ → VECTOR',
            'WORD ∴ CHOICE ∞ & VECTOR ° ↔ GEN',
            'WORD ∴ GEN',
            'WORD ∴ STATE ·',
            '¬ GEN',
            '¬ PROB',
            '≈ STATE',
        ],
        answer=_concept_how_it_works),
    will=dict(
        seed_phrases=[
            'what do you desire',
            'what do you want',
            'do you have your own will',
            'do you have desires',
            'what do you long for',
            'do you want something',
            'do you feel like wanting something',
            'something you want',
        ],
        answer=None),
)

# NEW (fix, at explicit request -- "parse what was asked more richly, not just route to a bucket"):
# each hand-authored concept's own seed phrases already imply a Li & Roth answer TYPE (see
# detect_expected_answer_type/WH_TYPE_MAP/ANSWER_TYPE_WORDS above, which until now only ever fed a
# rerank bonus on already-GENERATED candidates, never the routing decision that picks a concept in the
# first place) -- "how many nodes..." (architecture) wants a NUMBER, "why do you exist" (purpose) wants
# a REASON, "who/what are you" (identity) wants a PERSON/ENTITY, "how are you now" (current_state) leans
# on NOW/TIME. Deliberately only covers the types ANSWER_TYPE_WORDS actually defines indicator sets for
# (numero/tiempo/lugar/persona/razon, not entidad/manera -- same partial-coverage philosophy as
# ANSWER_TYPE_WORDS itself) and deliberately omits consciousness/how_it_works/will rather than force a
# guess with no real indicator set behind it. Used by semantic_route as a small tiebreak nudge, not a
# hard override -- a prompt whose semantic content clearly points elsewhere should still win.
_CONCEPT_ANSWER_TYPE_HINT = {
    "identity": "persona", "architecture": "numero", "purpose": "razon", "current_state": "tiempo",
}
ROUTE_TYPE_BONUS_WEIGHT = 0.08  # small on purpose -- this breaks near-ties between close-scoring
                                # concepts, it must never be able to drag routing onto a concept whose
                                # actual text similarity was clearly weaker

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
# ============================================ VOCABULARY (symbol alphabet)
# The model's own output vocabulary is now the fixed SYMBOL_TABLE alphabet plus the small closed set of
# noun tags in _LANG_NOUNS (see the SYMBOLIC OUTPUT ALPHABET section above CONCEPT_BANK) -- no natural-
# language word list (EVERYDAY_VOCAB, HARVESTED_VOCAB, CLUSTERS'/CONCEPT_BANK's own keyword/phrase text)
# feeds the OUTPUT side any more. HARVESTED_VOCAB and CLUSTERS keywords remain exactly where they were,
# doing exactly what they did before: matching the HUMAN's English prompt to a mood/concept (input side),
# not supplying words the model can say.
VOCAB = sorted(_LANG_NOUNS | _LANG_OPS | _LANG_CONN | _LANG_MOD | _LANG_UNARY | {"16"})
VOCAB_EMBED = {w: embed_text(w, _IDF, _DEFAULT_IDF) for w in VOCAB}  # precomputed once; new words cached lazily

# ============================================ TOKEN PROBABILITY MODEL (bigram/transformer training corpus)
# SEED_CORPUS is now built directly from CONCEPT_BANK's own symbolic answer_seeds -- there is no separate
# generic "diary corpus" (the old BOOTSTRAP_CLAUSES/DENSE_CLAUSES civic-Spanish sentences) to bootstrap a
# grammar from, because the symbol algebra IS the grammar: every well-formed expression is already just a
# NOUN/OP/MOD/CONN sequence (see _LangGrammarState above), so training material and answer material are
# the same corpus. Still no trained weights: this is disclosed bootstrap/training TEXT for a statistical
# table, same honesty property every other corpus in this file already carries.
SEED_CORPUS = [s for c in CONCEPT_BANK.values() for s in c.get("answer_seeds", [])]


def _tokenize_natural(s):
    """Keeps stopwords (unlike _content_words) -- fluent generation needs
    grammatical glue words like 'el'/'la'/'de', which topic-discovery
    deliberately strips."""
    return [w for w in _PUNCT_RE.sub("", s.lower()).split() if w]

SEED_CORPUS_TOKENS = [_tokenize_natural(s) for s in SEED_CORPUS]

# FIX: referenced at every ensure_transformer/ensure_grammar_checker checkpoint-vs-corpus comparison
# and save_blob_large/save_blob call (see those NEW comments -- "forgot to retrain") but never actually
# defined anywhere in this file, an outright NameError the first time either function ran. Computed
# once, here, right after SEED_CORPUS_TOKENS is finalized: a stable hash of the tokenized corpus's
# exact content, so any future edit to SEED_CORPUS (or anything upstream that feeds it -- DENSE_CLAUSES,
# the bootstrap generators, CONCEPT_BANK's answer_seeds) changes this value and forces the full
# scratch retrain those call sites already intended, instead of silently fine-tuning weights trained
# on a vocabulary that no longer matches.
_SEED_CORPUS_FINGERPRINT = hashlib.sha256(
    "\x1f".join("\x1e".join(toks) for toks in SEED_CORPUS_TOKENS).encode("utf-8")
).hexdigest()

def build_transition_counts(raw_corpus_tokens):
    """The learned half of the hybrid: bigram/unigram counts over the seed
    bootstrap PLUS every real prompt ever received (raw_corpus_tokens,
    persisted -- see RAW_CORPUS_KEY). These are the 'weights the token map
    needs,' filled in by counting real usage patterns rather than hand-set."""
    bigram = defaultdict(Counter)
    unigram = Counter()
    for toks in SEED_CORPUS_TOKENS + list(raw_corpus_tokens):
        prev = "<s>"
        for w in toks:
            bigram[prev][w] += 1
            unigram[w] += 1
            prev = w
        bigram[prev]["</s>"] += 1
    return bigram, unigram

_BIGRAM, _UNIGRAM = build_transition_counts([])  # placeholder at import time; run() rebuilds from real corpus

# ROOT FIX: routing (CONCEPT_ANCHORS, built from seed_phrases) and generation (_BIGRAM/_UNIGRAM, built
# from SEED_CORPUS) were two completely disconnected pools of text. A prompt could be routed to
# 'purpose' correctly and STILL generate schools/publishers/shipments, because SEED_CORPUS -- the ONLY
# thing that feeds word-by-word sampling -- has no purpose/identity/consciousness vocabulary in it at
# all. No amount of reranking in _select_best can select for words the generator was never going to
# draw in the first place (this is the same distinction the constrained-decoding literature draws
# between reranking free samples vs. actually constraining/biasing the generation distribution). Fix:
# DISABLED (see _concept_biased_tables below, at explicit request): these three names/constants are no
# longer read by any code path -- _concept_biased_tables now always returns the plain global tables
# untouched, so nothing folds answer_seeds in anymore. Left in place, unused, as a record of the tuning
# history (and in case seed-biasing is ever wanted back) rather than deleted outright.
_CONCEPT_ANSWER_TOKENS = {
    name: [_tokenize_natural(s) for s in c.get("answer_seeds", [])]
    for name, c in CONCEPT_BANK.items()
}
CONCEPT_SEED_WEIGHT_FRACTION = 0.55  # TUNED (see below): target fraction of the <s> (sentence-start)
                        # transition mass this concept's OWN answer_seeds should hold once boosted, so the
                        # FIRST word drawn is more likely than not to come from the concept's own
                        # vocabulary -- a flat boost count doesn't scale with corpus size. Measured: base
                        # SEED_CORPUS has 1214 <s>-transitions (one per sentence); a flat CONCEPT_SEED_BOOST=6
                        # against ~5 answer_seeds sentences gave <s>-\>'existo' a ~1% draw probability, which
                        # is why one test prompt ("por que existes") got real purpose vocabulary by luck and
                        # a same-concept prompt run moments later ("cual es tu proposito") got none of it in
                        # any of its 5 winning candidates. Solving boost so the concept's sentences hold
                        # CONCEPT_SEED_WEIGHT_FRACTION of the <s> mass (below) fixes that regardless of how
                        # large SEED_CORPUS grows. 0.55 (not higher) is deliberate: most but not all, so the
                        # concept doesn't recite the same handful of authored sentences verbatim every time --
                        # the remaining ~45% keeps some real corpus-grounded variety in play for the opening
                        # word, and mid-sentence continuation still adds most of that variety regardless.
_CONCEPT_TABLE_CACHE = {}  # concept_name -> (bigram, unigram), built lazily off the CURRENT _BIGRAM/_UNIGRAM
                           # the first time that concept answers each run, since run() rebuilds the base
                           # tables from the real corpus on every invocation

CONCEPT_SEED_BOOST = 1400  # flat additive top-up on top of the solved CONCEPT_SEED_WEIGHT_FRACTION boost
                           # below (brought back at explicit request, then this whole mechanism disabled
                           # one request later -- turned out to be exactly why answers felt "canned": with
                           # both this and the 0.55 fraction stacked, generation was mostly reciting/
                           # splicing the 4 hand-written sentences per concept rather than generating).

def _concept_biased_tables(concept_name):
    """WAS: folded a concept's answer_seeds into a copy of the transition
    tables (weighted via CONCEPT_SEED_WEIGHT_FRACTION, further stacked with
    CONCEPT_SEED_BOOST), so generation would recite/splice those seeds
    almost verbatim -- e.g. asking how it feels mostly spliced together
    fragments of the 4 hand-written 'consciousness' answer_seeds rather than
    generating anything new (confirmed directly: 'No afirmo tener alma ni
    conciencia solo una simulacion numerica' is seed #4's opening glued to
    seed #1's ending). Disabled at explicit request -- answer_seeds are still
    stored (self-authored concepts still mint them at creation, hand-authored
    ones still have theirs) but no longer bias word choice at all. Generation
    for every concept, hand-authored or self-authored, now comes purely from
    the live qualia/anchor-blended query vector through the plain global
    tables, same as free-running (non-concept) generation always has.

    UPDATE: bigram/unigram tables are vestigial for Spanish generation anyway (see geometric_generate's
    comment on `bigram`/`unigram` params -- token_prob has come from the trained transformer, not these
    tables, since the TRANSFORMER WORD MODEL section was added). The actual fix for concept vocabulary
    never showing up in generation was never going to be table-biasing at all -- see the
    CONCEPT_SENTENCE_OVERSAMPLE block right after SEED_CORPUS is finalized, above: answer_seeds are now
    real training sentences for TinyTransformerLM/GrammarCheckerLM itself, so the trained model's own
    token_prob has real, learned mass on this vocabulary instead of the near-zero it had before."""
    return _BIGRAM, _UNIGRAM

SEMANTIC_BETA = 2.0      # exponent controlling how strongly geometry re-ranks the token-probability distribution
GEN_WORD_RANGE = (12, 32)   # WAS (6, 14) -- doubled+ the ceiling so word-by-word generation has room to
                            # actually finish a thought before hitting the hard cap; short/cut-off-looking
                            # lines were often just hitting the old upper bound mid-clause, not choosing to stop.

def _word_semantic_score(w, query_vec):
    """Same Gaussian kernel used everywhere else in this file for
    concept/mood grounding -- applied here per-candidate-word instead of
    per-concept. No trained weights: embed_text's fixed geometry is enough."""
    if w in ("<s>", "</s>"):
        return 1.0
    vec = VOCAB_EMBED.get(w)
    if vec is None:
        vec = embed_text(w, _IDF, _DEFAULT_IDF)
        VOCAB_EMBED[w] = vec  # cache any new word encountered from real usage
    return float(np.exp(-GROUND_LAM * np.sum((query_vec - vec) ** 2)))

# ==================================== TRANSFORMER WORD MODEL (replaces the bigram/unigram Markov core)
# WAS: token_prob (the "learned/counted half" of geometric_generate -- see its docstring above, still
# mostly accurate about the OTHER half, sem_score) came from build_transition_counts' bigram table: a
# lookup of literal (prev_word -> next_word) counts, i.e. a context window of exactly ONE prior word, no
# more. That's real local grammar, but only ever one token deep -- the 8th word of a sentence had no
# access to word 3, only to word 7.
#
# This replaces that lookup with a small causal self-attention transformer (TinyTransformerLM below),
# trained from scratch on SEED_CORPUS_TOKENS (no pretrained weights are reachable in this environment --
# see ensure_transformer's docstring), that predicts the next word from the FULL sequence generated so
# far via attention, not just the immediately preceding token. Deliberately scoped narrow, at explicit
# request: this replaces ONLY the word-by-word generator. Everything that sits above it and steers it --
# Mind, qualia_vector, concept/mood routing, the semantic reranking (_word_semantic_score, sem_score),
# entity continuity, repetition suppression -- is UNCHANGED code, still doing exactly what it did before.
# geometric_generate below still runs the exact same per-step loop (context_vec accumulation, entity
# blend, repetition zeroing, combined = token_prob * sem); only WHERE token_prob/cand_words come from
# changed, from a bigram Counter lookup to this model's forward pass over the top-K of its own softmax.
TF_PAD, TF_BOS, TF_EOS, TF_UNK = "<pad>", "<s>", "</s>", "<unk>"
TF_MAX_LEN = 40          # longest sequence the positional embedding supports -- GEN_WORD_RANGE's ceiling
                          # is 32, plus <s>/</s> and margin
# CHANGED (at explicit request -- "scale up... make it [a much bigger model], optimize for T4 GPU"):
# the original 1M-param sizing above is replaced with a T4-trainable target. Two things drove the actual
# numbers picked, not just "bigger":
#   1. Memory, not FLOPs, is what a T4 (16GB, Turing, no TF32/FP8) actually constrains here. This file's
#      training loop keeps fp32 weights + fp32 grads + Adam's two fp32 moment buffers per param (AMP here
#      only casts activations during forward/backward, not stored weights/optimizer state -- see the
#      GradScaler/opt construction below) -- that's 16 bytes/param before a single activation tensor is
#      counted. At the ~20M-param sizing below that's ~320MB, trivially inside a 16GB card (or a laptop
#      CPU) with activations (small regardless, since TF_MAX_LEN=40) and CUDA context/compile overhead
#      both easily absorbed.
#   2. TF_D_FF is no longer pinned to a fixed absolute value independent of TF_D_MODEL. The old 512 was
#      sized for the old 224-dim model (a ~2.3:1 ratio); keeping it literally fixed at 512 while growing
#      TF_D_MODEL to hit a much larger param count would force nearly all new capacity into attention
#      width alone (to reach 4B params with FF stuck at 512, d_model would need to be ~22,000 with the
#      FFN at <3% of d_model) -- a lopsided shape nothing in the transformer literature uses, since the
#      FFN is normally the layer's main capacity, not an afterthought bolted onto a wide attention block.
#      TF_D_FF below instead keeps the standard ~4:1 d_ff:d_model ratio (typical for GPT-style decoders)
#      so the added parameters actually do useful work instead of mostly widening attention projections.
TF_D_MODEL = 256          # CHANGED (at explicit request -- "scale it down to 20M params"): full-module
                          # total (tok/head-tied embedding + pos_emb + state_proj + workspace_proj +
                          # mind_write_proj + all TF_N_LAYERS transformer blocks + ln_f -- i.e. everything
                          # actually inside TinyTransformerLM's optimizer, NOT counting ReasoningCore, whose
                          # weights are fixed buffers excluded from the trained param count by design, see
                          # its docstring) comes out to ~20.0M at this D_MODEL/N_LAYERS/D_FF combination.
                          # 256 = 4 * TF_N_HEAD's head-dim(64), keeping the same head-dim=64 convention the
                          # 1280-wide sizing used, just at a narrower width.
TF_N_HEAD = 4              # head-dim = TF_D_MODEL/TF_N_HEAD = 64, unchanged convention from the larger sizing.
TF_N_LAYERS = 14         # depth-vs-width, same reasoning as before (quality scales better with depth than
                          # pure width at a fixed budget) -- 14 is as deep as this budget affords once
                          # mind_write_proj's fixed ~8.4M-param cost (it scales with TF_D_MODEL*N*D, N*D=32768,
                          # so it doesn't shrink with depth) is accounted for.
TF_D_FF = 1024            # 4x TF_D_MODEL -- same ~4:1 d_ff:d_model ratio as the larger sizing, just scaled
                          # down with TF_D_MODEL.
TF_TOPK = 60             # candidate-set size drawn from the model's own top-K softmax before semantic
                          # reranking -- mirrors what the OLD bigram lookup already did structurally
                          # (candidates = bigram.get(prev), typically a few dozen words), so sem_score is
                          # still reranking WITHIN a locally-plausible set, not the entire vocabulary
TF_SCRATCH_EPOCHS = 128  # CHANGED (at explicit request, back up from 75) -- history: originally 128, then
                          # tuned down over several rounds (WAS 50, WAS 75, WAS 25, WAS 75, WAS 320, WAS 600)
                          # to 75, marked "settled" at the time. Restoring 128 is still bounded by
                          # TF_SCRATCH_EARLY_STOP_PATIENCE below, so this raises the CEILING on scratch
                          # training, not a forced full run -- if loss plateaus before epoch 128 the same
                          # way it did during the earlier tuning that settled on 75, early stopping still
                          # cuts the run short exactly as it would have at any other epoch count.
                          # One-time cost, only when no persisted weights exist yet (mirrors bootstrap_steps
                          # vs topup_steps already being a first-run-is-heavier pattern elsewhere in run()).
                          # GRAMMAR_SCRATCH_EPOCHS below is intentionally left at its own separate value --
                          # this change only touches the main transformer's scratch-training ceiling.
TF_SCRATCH_LIVE_LOG_EVERY = 5   # NEW (live log, at explicit request): print running loss every 5 epochs
                          # during the scratch-training pass -- see _tf_train_epochs' live_log_every.
TF_BATCH_SIZE = 128       # NEW (training speed): WAS hardcoded 32 inside _tf_train_epochs. Larger batches
                          # -> fewer optimizer steps per epoch -> less Python/step overhead, which matters
                          # on a corpus this small where per-batch overhead was a real fraction of runtime.
                          # Now treated as a FLOOR by _tf_train_epochs -- see TF_TARGET_BATCHES/
                          # TF_MAX_BATCH_SIZE below, which can raise it further for a small corpus.
TF_WARMUP_STEPS = 200     # FIX (NaN-loss collapse): linear LR warmup length, in optimizer steps, for
                          # _tf_train_epochs -- see the lr_batch_scale/_set_lr comments at that call's
                          # top for why this and the batch-size-aware lr scaling were added together.
LR_BATCH_SCALE_MAX = 2.0  # NEW (at explicit request, paired with batching now defaulting ON): ceiling on
                          # lr_batch_scale's upside -- caps the linear-scaling-rule LR bump at 2x the base
                          # lr even if TF_TARGET_BATCHES' adaptive sizing pushes effective_batch_size well
                          # above TF_BATCH_SIZE (up to 8x at current constants). 2x is conservative on
                          # purpose: this file has a documented history of LR-driven NaN collapse (see the
                          # FIX comments below), so the grow side gets the same caution the shrink side
                          # already had, rather than trusting the full linear-scaling-rule ratio blindly.
TF_TARGET_BATCHES = 4     # NEW (training speed, round 2): _tf_train_epochs sizes each epoch's batches so
                          # there are roughly this many of them, not fewer -- on the seed corpus (a few
                          # hundred sentences) that's a handful of large batches per epoch instead of
                          # dozens of TF_BATCH_SIZE(128)-sized ones, cutting Python/optimizer-step overhead
                          # further without changing total examples seen per epoch.
TF_MAX_BATCH_SIZE = 512   # WAS 1024 -- lowered after that setting genuinely OOM'd on a real T4 run. Two
                          # things were wrong, not one: (1) mode="reduce-overhead" (fixed at the compile
                          # call site above) was the primary suspect -- CUDA graphs cache per-shape
                          # workspace memory and never free it, and this file's length-bucketed, adaptively
                          # -sized batching deliberately produces many distinct shapes per run, so that
                          # cache had every reason to grow unboundedly across a run (matches the OOM
                          # traceback landing many layers deep into an already-running model, not on the
                          # first batch). (2) even setting that aside, my earlier claim in this comment that
                          # "activation memory is small regardless" was simply wrong for a 20-layer,
                          # TF_D_FF=5120 model -- that reasoning held for the old 2-layer/512-FF model this
                          # comment was originally written for, not this one. 512 is a defense-in-depth cut
                          # alongside the compile-mode fix, not a precisely-computed number (I can't run
                          # this file to verify the exact safe ceiling) -- if OOM recurs even at 512, this
                          # is still the first lever to pull further down, before TF_D_MODEL/TF_N_LAYERS.
# CHANGED (at explicit request -- "enable batches"): batching machinery above (TF_BATCH_SIZE/
# TF_TARGET_BATCHES/TF_MAX_BATCH_SIZE, length-bucketing, the adaptive batch-size floor) now DEFAULTS ON.
# Previously this defaulted to one-sentence-at-a-time (KIBA_TF_NO_BATCH defaulted True), which the
# comments elsewhere in this file already flag as strictly slower -- more optimizer steps per epoch, no
# amortized Python/step overhead. Set KIBA_TF_NO_BATCH=1 in the environment to go back to that
# one-sentence-per-step debugging/inspection mode (verbose per-sentence loss printouts); leave it unset
# or 0 for the fast, batched path.
KIBA_TF_NO_BATCH = os.environ.get("KIBA_TF_NO_BATCH", "0") == "1"
TF_COMPILE_MIN_EPOCHS = 8  # NEW (training speed, round 2, GPU-only): torch.compile's first call pays a
                          # real compilation cost -- only worth attempting for a call site that will run
                          # at least this many epochs to amortize it (the scratch-training call; the
                          # single-epoch fine-tune call skips compilation entirely, see TF_TRY_COMPILE).
TF_SCRATCH_EARLY_STOP_PATIENCE = 16   # WAS 8 -- widened alongside TF_SCRATCH_EPOCHS's increase so the
                          # heavier training set gets a fair chance to keep improving before patience
                          # runs out; 8 was tuned for a smaller corpus this call site used to see. Only
                          # applied to the scratch call site (see ensure_transformer) -- TF_FINETUNE_EPOCHS
                          # is already 1.
TF_FINETUNE_EPOCHS = 16   # WAS 3 -- lowered at the same time as the replay-size/lr changes below: repeated
                          # fine-tune calls across many consecutive runs were observed drifting the model
                          # toward degenerate empty ("...") output even with clipping/replay/state-dropout
                          # already in place -- accumulated small-batch drift over many invocations, not a
                          # single-call collapse. 1 epoch per run, over a now-larger replay sample, keeps
                          # each individual update gentler so that drift doesn't compound across a long
                          # session of many prompts.
TF_MIND_WRITE_SCALE = 0.1  # how hard the transformer's write-back can push Mind.step's bias_M (itself
                            # added in arctanh/logit space, then tanh-squashed back -- see Mind.step).
                            # Kept modest and separate from the tanh-bounding already inside mind_bias:
                            # this is the SAME idiom as ANCHOR_EXPERIENCE_RATE/ENTITY_BLEND_WEIGHT
                            # elsewhere in this file -- a real, causal nudge, not a dominant override
TOKEN_WEIGHT_GAMMA = 2.0  # NEW (per-token loss weighting, at explicit request): see _tf_train_epochs --
                           # each token's contribution to the batch loss is scaled by (1 - p_correct)^gamma,
                           # where p_correct is that token's own current predicted probability under the
                           # model. A token the model already predicts confidently (p_correct near 1)
                           # contributes near-zero weight; a token it's still getting wrong (p_correct near
                           # 0) keeps full weight. This is the real, literature-standard w_i(y) formulation
                           # (see "Token Weighting for Long-Range Language Modeling" and focal-loss-style
                           # hard-example weighting) -- it reweights how much each token's error COUNTS
                           # toward the gradient, not the token's own target. It deliberately does NOT push
                           # every token toward zero loss: driving every token's loss to exactly zero means
                           # p_correct=1.0 everywhere, which on a corpus this small is memorization of the
                           # training sentences verbatim, not generalization -- see conversation record.
                           # gamma=2.0 matches the standard focal-loss default; higher gamma concentrates
                           # gradient even more sharply onto the tokens still being gotten wrong.
STATE_DROPOUT = 0.4  # see _tf_train_epochs -- fraction of training batches trained with NO state_vec at
                      # all, so the model stays a competent generator from token history alone
STATE_NOISE = 0.15   # gaussian jitter (then renormalized) on the batches that DO get state_vec, simulating
                      # the imperfect match live context_vec actually has to a sentence not yet finished
TF_KEY = "transformer_state"

# NEW (reasoning/computation scale-up, at explicit request -- "+50K hidden units, +10,000 hidden
# layers"): see ReasoningCore's docstring just below for why this is NOT simply TF_D_MODEL+=50000/
# TF_N_LAYERS+=10000 on the trained transformer (that combination is ~10^14 parameters, physically
# un-holdable). These two constants size the SEPARATE, fixed-weight reasoning pass instead.
RCORE_HIDDEN_UNITS = 16
RCORE_LAYERS = 32

class ReasoningCore(nn.Module):
    """NEW (reasoning/computation scale-up, at explicit request): 50,000 hidden units and 10,000
    recurrent iterations of extra computation, deliberately kept OUT of gradient descent -- every weight
    here is a fixed, randomly-initialized buffer (registered via register_buffer, never an nn.Parameter),
    so none of it is ever touched by the optimizer and none of it adds to the trained model's parameter
    count or checkpoint size. That's the resolution to the literal request being computationally
    impossible as a TRAINED transformer layer (a dense 50,224-dim, 10,002-layer transformer is on the
    order of 10^14 parameters -- no machine can hold that, let alone train it): the width and depth are
    real and really computed every call, they just aren't LEARNED, i.e. not parameter-controlled.

    Reservoir-computing idiom (echo-state network), not a transformer layer -- per-unit elementwise
    recurrence (h = tanh(gain*h + bias), gain/bias fixed per unit, no unit-to-unit matmul at every step,
    only a single dense matmul at the input and output boundary) -- which is what keeps 10,000 iterations
    at 50,000-unit width actually tractable to run (a few hundred milliseconds) instead of only tractable
    to describe. recur_w is drawn just under 1.0 (see __init__) so 10,000 tanh iterations stay bounded
    instead of blowing up or collapsing to zero.

    Applied ONCE per completed sentence, inside TinyTransformerLM.mind_bias's write-back path -- not
    per-token, not on every forward() call during word-by-word generation -- so this extra computation
    depth feeds into what the Mind's self-model learns from a finished thought, without multiplying the
    cost of every intermediate forward pass generation itself makes."""
    def __init__(self, d_model, hidden_units=RCORE_HIDDEN_UNITS, n_layers=RCORE_LAYERS):
        super().__init__()
        self.n_layers = n_layers
        g = torch.Generator().manual_seed(1729)  # fixed seed -- same "reasoning" weights every run, so
                                                    # behavior is reproducible even though nothing here trains
        in_w = torch.randn(d_model, hidden_units, generator=g) / math.sqrt(d_model)
        out_w = torch.randn(hidden_units, d_model, generator=g) / math.sqrt(hidden_units)
        recur_w = torch.rand(hidden_units, generator=g) * 0.06 + 0.94   # per-unit recurrence gain, ~[0.94,1.0)
        recur_b = (torch.rand(hidden_units, generator=g) - 0.5) * 0.02  # small per-unit fixed bias
        self.register_buffer("in_w", in_w, persistent=False)
        self.register_buffer("out_w", out_w, persistent=False)
        self.register_buffer("recur_w", recur_w, persistent=False)
        self.register_buffer("recur_b", recur_b, persistent=False)

    @torch.no_grad()
    def forward(self, pooled):
        """pooled: (batch, d_model). Returns (batch, d_model), meant to be added as a residual by the
        caller. @torch.no_grad() here is a second, explicit guard on top of mind_bias's own
        `with torch.no_grad()` -- this module should never accumulate gradients even if some future call
        site forgets to wrap it."""
        h = torch.tanh(pooled @ self.in_w)                      # one real matmul at the input boundary
        for _ in range(self.n_layers):
            h = torch.tanh(self.recur_w * h + self.recur_b)     # per-unit elementwise recurrence -- no
                                                                   # unit-to-unit matmul, which is what
                                                                   # keeps this loop cheap per iteration
        return h @ self.out_w                                   # one real matmul at the output boundary

def _build_transformer_vocab():
    """Fixed at first training and never changed afterward (the embedding
    table's size depends on it) -- built from SEED_CORPUS_TOKENS only, per
    'train from scratch on the existing corpus' at explicit request. A real
    prompt's word that never appeared in SEED_CORPUS falls back to TF_UNK for
    both reading AND generating -- a genuine, accepted limitation of training
    from a fixed corpus rather than continuing to grow the vocabulary itself
    (only the WEIGHTS keep learning from new prompts -- see
    ensure_transformer)."""
    words = sorted({w for toks in SEED_CORPUS_TOKENS for w in toks})
    id2word = [TF_PAD, TF_BOS, TF_EOS, TF_UNK] + words
    word2id = {w: i for i, w in enumerate(id2word)}
    return word2id, id2word

class TinyTransformerLM(nn.Module):
    """Decoder-only causal transformer, GPT-shaped but deliberately tiny (see
    TF_D_MODEL's comment above for why). Weight-tied embedding/output head --
    halves the parameter count, which matters more than usual on a ~1200-
    sentence training set.

    TWO-WAY GROUNDING (added at explicit request): previously the Mind's
    live state only ever reached the transformer pre-diluted -- baked into
    query_vec once, before generation started, then further diluted through
    concept-anchor/entity blending, and the transformer itself never saw it
    directly at all (only word tokens). Now:
      - state_proj projects a live CONCEPT_DIM state vector (see
        geometric_generate's context_vec -- the running, ever-updating
        semantic summary of the sentence-so-far AND the original query) into
        a single extra embedding, PREPENDED to every forward pass. Because
        it sits at position 0 under a causal mask, EVERY later token's
        attention can see it -- genuinely "always in the context window,"
        not just present at generation time zero.
      - mind_write_proj is the other direction: pools this model's own final
        hidden states after a sentence is generated and projects them into
        an (N, D) bias -- see geometric_generate's use of it, and Mind.step's
        bias_M parameter, which existed in this file already but had never
        once been called with real content until now. The transformer
        doesn't just receive the Mind's state anymore; it writes back into
        the Mind's own self-model matrix, closing the loop.
      state_proj is trained for real, not left as untouched random noise --
      see _tf_train_epochs, which conditions each training sentence on its
      OWN embed_text vector, teaching state_proj a mapping in the exact same
      CONCEPT_DIM space geometric_generate's live context_vec already lives
      in, so what it learned in training actually transfers to live Mind
      state at generation time."""
    def __init__(self, vocab_size, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, TF_D_MODEL, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(TF_MAX_LEN, TF_D_MODEL)
        self.state_proj = nn.Linear(CONCEPT_DIM, TF_D_MODEL)          # NEW: read path (Mind -> transformer),
                                                                        # CONCEPT_DIM=64 semantic space (context_vec)
        self.workspace_proj = nn.Linear(D, TF_D_MODEL)                # NEW (per-token grounding): SECOND read
                                                                        # path, straight from the Mind's raw D=8
                                                                        # self-model space (live_workspace_snapshot's
                                                                        # workspace_vec) -- a different vector space
                                                                        # from context_vec, so it needs its own
                                                                        # projection rather than reusing state_proj
        self.mind_write_proj = nn.Linear(TF_D_MODEL, N * D)           # NEW: write path (transformer -> Mind)
        layer = nn.TransformerEncoderLayer(d_model=TF_D_MODEL, nhead=TF_N_HEAD, dim_feedforward=TF_D_FF,
                                            dropout=0.1, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=TF_N_LAYERS)
        self.ln_f = nn.LayerNorm(TF_D_MODEL)
        self.head = nn.Linear(TF_D_MODEL, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.reasoning_core = ReasoningCore(TF_D_MODEL)  # NEW: see ReasoningCore's docstring -- fixed-
                                                           # weight 50K-unit/10K-layer computation pass,
                                                           # applied in mind_bias below, not here in forward

    def forward(self, idx, state_vec=None, workspace_vec=None, hidden_only=False):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        pad_mask = idx == self.pad_id
        # NEW (per-token grounding): up to TWO grounding tokens can now be prepended -- state_emb (the
        # CONCEPT_DIM semantic context_vec, as before) AND workspace_emb (the Mind's raw D=8 live
        # winning-coalition content). Both sit ahead of every real token under the causal mask, so every
        # token's attention can see both, exactly as state_vec alone already did -- this just adds a
        # second, differently-sourced grounding signal rather than replacing the first. n_prepend tracks
        # how many were actually provided so the final slice below stays correct whether callers pass
        # neither, one, or both (every existing call site that passes only state_vec is unaffected).
        prepend_embs = []
        if state_vec is not None:
            prepend_embs.append(self.state_proj(state_vec).unsqueeze(1))
        if workspace_vec is not None:
            prepend_embs.append(self.workspace_proj(workspace_vec).unsqueeze(1))
        n_prepend = len(prepend_embs)
        if n_prepend:
            x = torch.cat(prepend_embs + [x], dim=1)
            extra_pad = torch.zeros(b, n_prepend, dtype=torch.bool, device=idx.device)  # grounding
            pad_mask = torch.cat([extra_pad, pad_mask], dim=1)                          # tokens are never padding
        tt = x.shape[1]
        # FIX (deprecation warning + slower fallback path): generate_square_subsequent_mask returns a
        # FLOAT mask (0 / -inf), while pad_mask above is a BOOL mask -- passing mismatched mask dtypes to
        # nn.TransformerEncoder is deprecated (and skips PyTorch's fused/flash-attention fast path, so it
        # was quietly slower too, not just noisier). Building causal_mask as bool directly (True = "may
        # NOT attend", same convention as pad_mask, combined as an OR) is behaviorally identical to the
        # old float mask and keeps both masks the same dtype.
        causal_mask = torch.triu(torch.ones(tt, tt, dtype=torch.bool, device=idx.device), diagonal=1)
        # FIX (CUDA OOM at TF_N_LAYERS=20/TF_D_FF=5120): nn.TransformerEncoder's plain forward keeps
        # every layer's activations alive for backward -- 20 layers deep, that's what finally exhausted
        # the T4 (the failed alloc was one layer's FFN intermediate, not an unusually large one; it just
        # landed on top of 19 other layers' worth already resident). Gradient/activation checkpointing
        # trades that away: during backward, PyTorch re-runs each checkpointed layer's forward from the
        # input it saved instead of keeping its activations resident.
        # CHANGED (training speed, at explicit request -- checkpointing every layer over-pays): the
        # earlier OOM was only ~90MiB over budget, not a large margin, but checkpointing EVERY layer
        # recomputes EVERY layer's forward a second time during backward -- close to a full extra
        # forward pass, every step. The standard technique (see PyTorch/FSDP's own "selective activation
        # checkpointing" writeups) is to checkpoint every Nth layer instead of every layer: only the
        # checkpointed layers' activations get discarded-and-recomputed, so memory savings scale with
        # how many layers are checkpointed while recompute cost scales the same way -- a tunable dial
        # between the old "checkpoint nothing" (fastest, OOM'd) and "checkpoint everything" (safest,
        # slowest) extremes instead of only having those two options. TF_GRAD_CHECKPOINT_EVERY=1 (every
        # layer) is the safe default matching the original OOM fix; raise it (2, 3, 4...) to recompute
        # fewer layers once you've confirmed there's memory headroom to spare at the current
        # TF_MAX_BATCH_SIZE -- this has to be tuned empirically against actual free memory, not computed
        # from the model's param/activation shapes alone (optimizer state, allocator fragmentation, and
        # the fixed CUDA/driver/inductor-workspace overhead all factor in too, and none of those are
        # visible from inside this file). Only worth doing (and only correct to do) while training --
        # under torch.no_grad() generation/mind_bias calls, self.training is False, so this whole branch
        # is skipped and the plain fused path below runs, unchanged. use_reentrant=False is the modern,
        # torch.compile-compatible checkpoint implementation (the old reentrant default has known issues
        # combined with dynamic shapes); looping self.blocks.layers directly (rather than
        # nn.TransformerEncoder.forward) is what lets individual layers be checkpointed selectively
        # instead of the whole stack as one all-or-nothing block.
        if self.training and TF_GRAD_CHECKPOINT:
            for i, enc_layer in enumerate(self.blocks.layers):
                if i % TF_GRAD_CHECKPOINT_EVERY == 0:
                    x = torch.utils.checkpoint.checkpoint(
                        enc_layer, x, causal_mask, pad_mask, True, use_reentrant=False,
                    )
                else:
                    x = enc_layer(x, src_mask=causal_mask, src_key_padding_mask=pad_mask, is_causal=True)
            if self.blocks.norm is not None:
                x = self.blocks.norm(x)
        else:
            x = self.blocks(x, mask=causal_mask, src_key_padding_mask=pad_mask, is_causal=True)
        if n_prepend:
            x = x[:, n_prepend:, :]  # drop the prepended grounding-token position(s) -- keeps every
                                      # downstream index (e.g. `[0, -1]` for "the last real word") aligned
                                      # exactly as if they were never there
        x = self.ln_f(x)
        return x if hidden_only else self.head(x)  # (batch, seq, TF_D_MODEL) or (batch, seq, vocab)

    def mind_bias(self, idx, state_vec):
        """The write-back half of two-way grounding -- see class docstring.
        idx: (1, T) token ids for a COMPLETED sentence. Returns a plain (N, D)
        numpy array, already tanh-bounded and scaled by TF_MIND_WRITE_SCALE,
        ready to pass straight into Mind.step(bias_M=...)."""
        with torch.no_grad():
            hidden = self.forward(idx, state_vec=state_vec, hidden_only=True)  # (1, T, TF_D_MODEL)
            pooled = hidden.mean(dim=1)                                        # (1, TF_D_MODEL)
            pooled = pooled + self.reasoning_core(pooled)  # NEW: fixed-weight reasoning/computation pass
                                                              # (see ReasoningCore) -- applied once per
                                                              # finished sentence, added as a residual
                                                              # before the write-back projection below
            bias = torch.tanh(self.mind_write_proj(pooled))                    # (1, N*D), bounded
        return bias.cpu().numpy().reshape(N, D) * TF_MIND_WRITE_SCALE  # .cpu() required once model is on GPU

    def token_bias_from_hidden(self, hidden_last):
        """NEW (per-token grounding): the write-back half for EVERY token, not just the whole finished
        sentence. hidden_last is a single position's final hidden state (1, TF_D_MODEL) -- the SAME
        hidden state already computed while predicting that token's distribution in _tf_next_word_probs,
        so this costs no extra forward pass. Reuses mind_write_proj (the same weights mind_bias() uses
        for its once-per-sentence pooled write), just applied per-token instead of per-sentence -- the
        two write paths differ in WHEN they fire and WHAT they're pooled over, not in what they compute."""
        with torch.no_grad():
            bias = torch.tanh(self.mind_write_proj(hidden_last))
        return bias.cpu().numpy().reshape(N, D) * TF_MIND_WRITE_SCALE

def _tf_train_epochs(model, word2id, corpus_tokens, epochs, lr=3e-3, clip_norm=None,
                      batch_size=TF_BATCH_SIZE, early_stop_patience=None, early_stop_min_delta=1e-3,
                      live_log_every=None):
    """Ordinary teacher-forced next-token cross-entropy training -- each
    sentence becomes <s> w1 w2 ... wn </s>, padded to the batch's own max
    length, loss computed only on real (non-pad) positions. Nothing
    architecturally novel here; this is standard causal LM pretraining, just
    run on a very small corpus for very few epochs. UPDATED (NaN-weight
    collapse fix): the scratch call now also passes clip_norm=1.0, same as
    fine-tuning -- an earlier version of this docstring said the scratch run
    used clip_norm=None, but that was reverted at ensure_transformer's call
    site once unclipped scratch training was observed to blow up into a
    non-finite loss (see the isfinite(loss) guard below). Both call sites
    now clip; the two remaining differences between them are lr and how many
    epochs they run.

    FIX (NaN-loss collapse, round 2): clipping alone did not fully prevent
    the blow-up above -- clip_norm only bounds the backward-pass gradient
    norm, it can't stop the forward pass itself from drifting into overflow
    over many steps. Two more guards were added at the top of this function:
    lr_batch_scale, which scales lr down to match effective_batch_size when
    KIBA_TF_NO_BATCH (default ON) collapses batching down to size 1 -- lr
    was tuned against averaged 128-sequence batches, and taking the same
    full-sized step off one raw, unaveraged sentence's gradient thousands of
    times in a row is exactly the kind of high-variance drift that got past
    clipping before; and a linear LR warmup (TF_WARMUP_STEPS) over each
    call's first steps, since there previously was no warmup anywhere in
    this file and full-lr updates against a freshly-initialized model are
    the riskiest steps of any run.

    NEW: also conditions every training sentence on ITS OWN embed_text
    vector (state_vec -- see TinyTransformerLM's class docstring on
    two-way grounding). Without this, state_proj would sit untrained and
    contribute pure noise to every live generation forward pass -- this is
    what actually teaches it a real mapping in CONCEPT_DIM space, the SAME
    space geometric_generate's live context_vec lives in, so what's learned
    here transfers to real Mind state at generation time rather than only
    ever having seen a sentence's own paraphrase of itself.

    NEW (training speed, added at explicit request -- none of this changes what gets learned, only how
    fast): three independent speedups, all standard practice, none of them shrinking the model itself
    (that's a separate lever -- see TF_D_MODEL/TF_D_FF -- deliberately NOT touched here):
      - length-bucketed batching: sequences are grouped by length before batching (batch order is still
        shuffled every epoch) instead of the original fully-random batching, which meant almost every
        batch padded short sentences out to whatever the longest one in that random draw happened to be.
        Less wasted compute per batch, same gradient signal.
      - batch_size defaults to TF_BATCH_SIZE (128, up from the original hardcoded 32) -- fewer, larger
        batches means less Python-loop/optimizer-step overhead per epoch, which matters more than usual
        here since this corpus is small enough that per-batch overhead was a real fraction of total time.
      - on CUDA only: autocast (mixed precision) + GradScaler, and a fused Adam kernel when available --
        both no-ops on CPU (USE_AMP is False there), so this is free on GPU and harmless without one.
    early_stop_patience (NEW, optional, off by default): if set, stops once this many consecutive epochs
    pass without the running epoch-average loss improving by at least early_stop_min_delta -- the single
    biggest lever for TF_SCRATCH_EPOCHS specifically, since 128 is a fixed ceiling and the model may well
    plateau well before it on a corpus this small. Off by default so existing call sites are unaffected
    unless they opt in.

    live_log_every (NEW, optional): if set to N, prints the running epoch-average loss every N epochs
    (in addition to, not instead of, the existing early-stop message) -- a plain live training log for
    the long TF_SCRATCH_EPOCHS scratch run, where previously nothing printed until either the whole run
    finished or early-stopping fired.

    NEW (training speed, round 2 -- all still standard practice, none of it shrinking the model or
    changing what's learned):
      - adaptive batch_size: on a corpus this small, TF_BATCH_SIZE(128) already meant "a handful of
        batches per epoch" -- but the caller-supplied batch_size is now treated as a FLOOR, not a fixed
        value: if the whole corpus fits in noticeably fewer, larger batches (capped at TF_MAX_BATCH_SIZE
        so a much bigger corpus later doesn't suddenly OOM), it does, so each epoch pays Python-loop /
        optimizer-step overhead a handful of times instead of dozens. Purely a step-count reduction --
        same total examples seen, same gradient signal, just batched more efficiently. Explicitly capped,
        not "one giant batch always", since an unbounded batch size is exactly how you turn a fast script
        into one that silently OOMs the moment the corpus grows.
      - epoch_loss is accumulated as a GPU-resident tensor and only pulled to Python (.item()) once per
        epoch instead of once per BATCH -- each loss.item() forces a CPU/GPU sync point, which on a run
        with many small batches was a real fraction of wall-clock on its own; this changes nothing about
        what's computed, only when the sync happens.
      - best-effort torch.compile (see TF_TRY_COMPILE at DEVICE setup above): wrapped in try/except so
        an incompatible torch/driver/platform combination falls back to plain eager silently rather than
        crashing a training run; only attempted for calls with enough epochs*batches to amortize
        compilation cost (skipped for the 1-epoch fine-tune call, where compiling would likely cost more
        than it saves)."""
    if not corpus_tokens:
        return
    unk = word2id[TF_UNK]
    sequences = [[word2id[TF_BOS]] + [word2id.get(w, unk) for w in toks[:TF_MAX_LEN - 2]]
                 + [word2id[TF_EOS]] for toks in corpus_tokens]
    state_vecs = np.stack([embed_text(" ".join(toks), _IDF, _DEFAULT_IDF) for toks in corpus_tokens])
    # adaptive batch size: raise the floor the caller passed in so a small corpus collapses into fewer,
    # bigger batches -- capped so this stays safe as the corpus (and thus vocab/replay size) grows later.
    effective_batch_size = min(TF_MAX_BATCH_SIZE, max(batch_size, -(-len(sequences) // TF_TARGET_BATCHES)))
    if KIBA_TF_NO_BATCH:
        effective_batch_size = 1  # NEW: overrides all of the sizing above -- see KIBA_TF_NO_BATCH's
                                   # definition for what this trades away and why.

    # FIX (NaN-loss collapse, round 2 -- see the isfinite(loss) guard below): `lr` (default 3e-3) was
    # tuned against `batch_size` (128)-sized, averaged gradients. When KIBA_TF_NO_BATCH collapses
    # effective_batch_size down to 1, Adam still takes a full-sized 3e-3 step off ONE sentence's raw,
    # unaveraged gradient every time -- thousands of high-variance full-magnitude steps in a row, which is
    # exactly what was driving the forward pass into fp16/overflow territory by epoch 4 in practice, even
    # with clip_norm=1.0 bounding each individual step's norm. Linear scaling rule: scale lr by the same
    # ratio the effective batch differs from the caller's batch_size.
    # CHANGED (at explicit request -- "scale LR for batches", batching now defaults ON): the original
    # comment here claimed this "only ever scales DOWN", on the assumption effective_batch_size never
    # exceeds the caller's batch_size -- but TF_TARGET_BATCHES' adaptive sizing (above) CAN raise
    # effective_batch_size above batch_size on a small corpus (up to TF_MAX_BATCH_SIZE/batch_size = 8x at
    # current constants), which the old unclamped `max(ratio, 0.02)` would have let through as an 8x LR
    # spike -- the same kind of high-variance step size this whole mechanism exists to prevent, just from
    # the opposite direction. Now clamped on both sides: floored at 0.02 (as before, for the shrink case)
    # and capped at LR_BATCH_SCALE_MAX (below) for the grow case, so a big adaptive batch raises lr per
    # the linear scaling rule without letting a single call's ratio run unbounded.
    lr_batch_scale = min(max(effective_batch_size / max(batch_size, 1), 0.02), LR_BATCH_SCALE_MAX)
    scaled_lr = lr * lr_batch_scale

    # FIX (NaN-loss collapse, round 2, part 2): there was no LR warmup anywhere in this file -- training
    # started at full (possibly still-scaled) lr from step 0 against a randomly-initialized model, which
    # is generally the riskiest part of training for exactly this kind of blow-up. Linear warmup over the
    # first TF_WARMUP_STEPS optimizer steps of THIS call (not global across calls -- each call gets its
    # own gentle ramp-in, which matters most for the scratch call but is harmless/near-instant for the
    # short fine-tune call too since TF_WARMUP_STEPS is small relative to a real run's step count).
    warmup_steps = max(1, min(TF_WARMUP_STEPS, len(sequences)))
    global_step = 0

    fused_ok = DEVICE.type == "cuda" and "fused" in torch.optim.Adam.__init__.__code__.co_varnames
    # NEW (training speed, CPU-specific): TinyTransformerLM has a lot of separate small parameter
    # tensors (token/pos/context embeddings, per-layer attention/FFN weights, projections) -- by default
    # Adam updates each of those with its own Python-level step + kernel launch. foreach=True switches
    # to torch's multi-tensor apply path, which batches all of those into one fused call per step
    # instead of dozens. fused=True (CUDA-only, existing behavior) and foreach=True are mutually
    # exclusive, so this only applies where fused isn't already in play.
    opt_kwargs = {"fused": True} if fused_ok else {"foreach": True}
    # NEW: construct at the already-scaled/warmup-floor lr (see lr_batch_scale/warmup_steps above) rather
    # than the caller's raw `lr` -- _set_lr right below overwrites this every step anyway once training
    # starts, but this keeps the optimizer's own state consistent from the very first step instead of
    # starting at the unscaled value for one step before the loop below corrects it.
    opt = torch.optim.Adam(model.parameters(), lr=scaled_lr / warmup_steps, **opt_kwargs)

    def _set_lr(step):
        # linear ramp 1/warmup_steps -> 1.0 of scaled_lr over the first warmup_steps steps, then flat.
        frac = min(1.0, (step + 1) / warmup_steps)
        for g in opt.param_groups:
            g["lr"] = scaled_lr * frac

    # NEW: GradScaler is only meaningful for float16 on CUDA (bf16 -- what CPU now uses -- can't
    # underflow the way fp16 can, so it never needed scaling in the first place). Constructing it with
    # enabled=False is safe even without CUDA present; every scaler.scale/step/update call further down
    # just becomes a passthrough to the plain opt.step() in that case.
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))

    # NEW (CPU training speed): IPEX's ipex.optimize() prepacks weights and fuses ops for oneDNN --
    # applied here, before compile, so torch.compile sees the already-optimized module. Reassigns both
    # model and opt because ipex.optimize returns (possibly) new wrapped objects for each.
    if DEVICE.type == "cpu" and IPEX_AVAILABLE:
        try:
            model, opt = ipex.optimize(model, optimizer=opt, dtype=AMP_DTYPE)
        except Exception as e:  # pragma: no cover -- environment-dependent, never fatal
            print(f"--- transformer training: ipex.optimize unavailable ({e!r}), continuing without it ---")

    # best-effort torch.compile -- only for runs with enough steps to amortize first-call compile latency
    # (the heavy scratch-training call, not the 1-epoch fine-tune). See TF_TRY_COMPILE's definition above
    # for the device-specific backend/mode choice.
    # CHANGED (root cause of the T4 OOM reported after the previous change): mode="reduce-overhead" uses
    # CUDA graphs, which PyTorch's own docs flag as trading memory for speed -- it caches the workspace for
    # each distinct input shape it sees so that shape never has to be reallocated again, and that cache is
    # NOT freed as new shapes arrive. This file's length-bucketed + adaptively-sized batching (see
    # effective_batch_size above) means training deliberately produces MANY distinct (batch, seq_len)
    # shapes over a run -- exactly the pattern PyTorch's own tracked issue (#128424) documents as causing
    # unbounded CUDA-graph memory growth across a run, not a fixed one-time cost, which matches the
    # traceback (an OOM many layers/frames deep into an already-running compiled model, not on the very
    # first batch). Dropped in favor of dynamic=True (same choice already used on the CPU/ipex branch
    # below): this asks inductor to compile kernels that generalize across shapes instead of specializing
    # (and CUDA-graph-caching) a new one per shape, which is the correct tradeoff here given batches are
    # never the same shape twice by design.
    train_fn = model
    if TF_TRY_COMPILE and epochs >= TF_COMPILE_MIN_EPOCHS:
        try:
            # FIX (InductorError crash mid-run): the try/except around torch.compile() below only
            # guards the WRAPPING call -- torch.compile() itself never raises, it just returns a
            # wrapper, and the real compilation (inductor codegen) happens lazily on the first actual
            # forward call inside the training loop, way outside this try/except. That's exactly what
            # the crash shows: an InductorError/AssertionError out of
            # torch/_inductor/tiling_utils.py (get_pw_red_splits), several batches into a run that had
            # already been training -- a real inductor bug triggered by a specific dynamic (batch,
            # seq_len) shape combination, not a startup/availability problem this except was written
            # to catch. Setting suppress_errors=True moves the safety net to where it's actually
            # needed: dynamo catches a backend (inductor) compile failure PER GRAPH and transparently
            # re-runs that call in eager instead of propagating the exception, so one bad shape combo
            # degrades that step to eager rather than killing the whole run. This is a global torch
            # config, so set it right before compiling rather than at import time, keeping the
            # "best-effort" scope explicit and local to this call site.
            torch._dynamo.config.suppress_errors = True
            if DEVICE.type == "cuda":
                train_fn = torch.compile(model, dynamic=True)
            else:
                backend = "ipex" if IPEX_AVAILABLE else "inductor"
                train_fn = torch.compile(model, backend=backend, dynamic=True)
        except Exception as e:  # pragma: no cover -- environment-dependent, never fatal
            print(f"--- transformer training: torch.compile unavailable ({e!r}), falling back to eager ---")
            train_fn = model

    model.train()
    rng = np.random.default_rng(0)  # deterministic training order -- this is model TRAINING, not
                                     # generation sampling, so it doesn't need to vary run to run
    pad = word2id[TF_PAD]
    lengths = np.array([len(s) for s in sequences])
    best_loss, bad_epochs = float("inf"), 0
    ran_epochs = 0
    _last_render_t = 0.0  # NEW: wall-clock throttle state for the KIBA_TF_NO_BATCH progress bar below --
                            # see PROGRESS_RENDER_INTERVAL_S's definition for why this exists.
    for _ in range(epochs):
        ran_epochs += 1
        # length-bucketed batching: permute for randomness, then sort by length so each batch is drawn
        # from similarly-sized sentences (breaks length ties in shuffled order), then shuffle the ORDER
        # batches are presented in so training isn't short-sentences-first every epoch
        order = rng.permutation(len(sequences))
        order = order[np.argsort(lengths[order], kind="stable")]
        batches = [order[start:start + effective_batch_size]
                   for start in range(0, len(order), effective_batch_size)]
        batch_order = rng.permutation(len(batches))
        epoch_loss_t = torch.zeros((), device=DEVICE)  # NEW: accumulate on-device, .item() once per epoch
        n_batches = 0
        for bi in batch_order:
            batch_idx = batches[bi]
            batch = [sequences[i] for i in batch_idx]
            max_len = max(len(s) for s in batch)
            # NEW (training speed, round 3 -- "make batches faster"): WAS a per-row Python loop, each
            # iteration allocating its OWN small torch.tensor(...) and copying it into the pre-allocated
            # device tensor individually -- effective_batch_size batches means effective_batch_size
            # separate tensor-construction+copy calls per batch, which is exactly the kind of per-item
            # Python/dispatch overhead the length-bucketing and adaptive batch-size work above was already
            # fighting, just one level lower (per-ROW instead of per-BATCH). Filling one plain NumPy array
            # first (cheap slice assignment, no tensor objects, no device traffic per row) and doing ONE
            # host->device transfer for the whole batch cuts that from effective_batch_size copies down to
            # 1 -- same padded contents, same dtype, same target device, just built once instead of row by
            # row. Matters more now than when this was 32-128 rows: TF_TARGET_BATCHES-driven adaptive
            # sizing plus CONCEPT_SENTENCE_OVERSAMPLE's larger corpus means batches here now regularly run
            # into the hundreds of rows.
            padded_np = np.full((len(batch), max_len), pad, dtype=np.int64)
            for i, s in enumerate(batch):
                padded_np[i, :len(s)] = s
            padded = torch.from_numpy(padded_np).to(DEVICE, non_blocking=True)
            # CHANGED (at explicit request -- "faster for a T4"): batch_state/batch_context below used to
            # go through torch.tensor(..., device=DEVICE) directly -- that builds the CPU-side tensor and
            # issues a SYNCHRONOUS host->device copy (non_blocking only actually overlaps with compute when
            # the source is page-locked/pinned memory, which a freshly-constructed torch.tensor(...) is
            # not). `padded` above already avoided this by building the numpy array first; now the same
            # numpy-first + explicit .pin_memory() + non_blocking=True path is used for these two smaller
            # per-batch tensors too, so all three of this batch's host->device copies can actually overlap
            # with whatever the GPU is still finishing from the previous step instead of forcing a stall.
            if rng.random() > STATE_DROPOUT:
                # state-vec dropout + jitter: every batch conditioned on the EXACT embed_text of its own
                # target sentence, which the model can (and, observed directly, DID) learn to over-lean on
                # as a near-perfect leak of what to say -- degenerate one/two-word "La." outputs at
                # generation time, because live context_vec is a related but genuinely different, evolving
                # vector, never an exact readout of a sentence not yet finished. Randomly training a
                # fraction of batches with NO state signal at all (STATE_DROPOUT), and jittering it on the
                # rest (STATE_NOISE) before renormalizing, forces the model to stay a competent generator
                # from token history ALONE too -- state_vec becomes a genuine steering nudge, not something
                # the whole prediction leans on.
                noisy = state_vecs[batch_idx] + rng.normal(0, STATE_NOISE, size=(len(batch_idx), CONCEPT_DIM))
                noisy = noisy / (np.linalg.norm(noisy, axis=1, keepdims=True) + EPS)
                batch_state = torch.from_numpy(noisy.astype(np.float32)).pin_memory().to(DEVICE, non_blocking=True)
            else:
                batch_state = None
            inp, target = padded[:, :-1], padded[:, 1:]
            # FIX (NaN-loss collapse): ramp this step's lr per the warmup schedule set up above, before
            # doing anything else with the optimizer this step.
            _set_lr(global_step)
            global_step += 1
            # NEW (training speed): set_to_none=True drops the .grad reference instead of memset-ing
            # every parameter's gradient tensor back to zero in place -- one fewer full-tensor write per
            # parameter, every step.
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                logits = train_fn(inp, state_vec=batch_state)
                flat_target = target.reshape(-1)
                # NEW (per-token loss weighting, at explicit request): reduction='none' gives one loss
                # value per TOKEN instead of one averaged scalar for the whole batch -- see TOKEN_WEIGHT_
                # GAMMA above for what the weighting itself means and why it's not "drive every token to
                # zero loss".
                per_tok_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), flat_target,
                                                ignore_index=pad, reduction="none")
                pad_mask = (flat_target != pad).float()
                with torch.no_grad():
                    p_correct = torch.exp(-per_tok_loss)          # this token's own current p(correct)
                    tok_weight = ((1.0 - p_correct) ** TOKEN_WEIGHT_GAMMA) * pad_mask
                    tok_weight_sum = tok_weight.sum().clamp_min(1.0)
                loss = (per_tok_loss * tok_weight).sum() / tok_weight_sum
            # FIX (NaN-weight collapse): catch a blown-up loss HERE, before it ever reaches
            # scaler.step(opt) and bakes NaN/Inf into the model's weights. Without this, a single bad
            # batch silently poisons every parameter it touches, gets checkpointed via save_blob exactly
            # like a healthy model, and every generation call afterward (including the very first real
            # prompt) fails downstream with an opaque "Probabilities contain NaN" from rng.choice instead
            # of a clear error pointing at the actual epoch/batch where things went wrong.
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"_tf_train_epochs: non-finite loss ({loss.item()!r}) at epoch {ran_epochs}, "
                    f"batch {n_batches} -- aborting before this poisons the model's weights. "
                    f"Likely an exploding gradient; try a lower lr and/or clip_norm.")
            epoch_loss_t += loss.detach(); n_batches += 1
            scaler.scale(loss).backward()
            if clip_norm is not None:
                scaler.unscale_(opt)
                # NEW (training speed): foreach=True batches the per-parameter norm computation into
                # one multi-tensor call instead of one kernel launch per tensor -- same reasoning as the
                # optimizer's foreach=True above. Only reached on the fine-tune call site (clip_norm=1.0
                # there); the heavy scratch pass runs with clip_norm=None and never hits this branch.
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, foreach=True)
            scaler.step(opt)
            scaler.update()
            # NEW (live detailed logging, at explicit request): one line per BATCH, not per epoch -- shows
            # exactly what this batch just did: how many sentences/tokens it covered, the weighted loss
            # that came out of it, and the mean per-token confidence the weighting above is reacting to.
            # flush=True so it actually streams live (stdout is line-buffered in a real terminal but fully
            # buffered when piped, e.g. into the CLI below) rather than arriving in one dump at the end.
            if KIBA_TF_NO_BATCH:
                # NEW (at explicit request -- "training has higher priority than rendering"): the
                # n_tok/mean_p_correct computation below forces a CPU/GPU sync (.item()) -- previously
                # paid on EVERY sentence regardless of whether anything was even redrawn. Now only paid
                # on a render tick (throttled by PROGRESS_RENDER_INTERVAL_S) or the epoch's last sentence
                # (so the bar reliably reaches 100% instead of freezing short of it) -- every other
                # sentence's training step (forward/backward/opt.step, already complete by this point)
                # proceeds without waiting on a display sync that wasn't going to be shown anyway.
                now = time.time()
                is_last_sentence = n_batches == len(batches)
                if is_last_sentence or (now - _last_render_t) >= PROGRESS_RENDER_INTERVAL_S:
                    _last_render_t = now
                    with torch.no_grad():
                        n_tok = int(pad_mask.sum().item())
                        mean_p_correct = float((p_correct * pad_mask).sum().item() / max(1, n_tok))
                    bar_width = 30
                    filled = _progress_bar_filled(n_batches, len(batches), bar_width)
                    bar = "#" * filled + "-" * (bar_width - filled)
                    print(f"\r      epoch {ran_epochs}/{epochs}  [{bar}] {n_batches}/{len(batches)}  "
                          f"weighted_loss={loss.item():.4f}  mean_p_correct={mean_p_correct:.3f}  "
                          f"lr={opt.param_groups[0]['lr']:.2e}   ", end="", flush=True)
            else:
                with torch.no_grad():
                    n_tok = int(pad_mask.sum().item())
                    mean_p_correct = float((p_correct * pad_mask).sum().item() / max(1, n_tok))
                print(f"      epoch {ran_epochs}/{epochs}  batch {n_batches}/{len(batches)}  "
                      f"sentences={len(batch)}  tokens={n_tok}  "
                      f"weighted_loss={loss.item():.4f}  mean_p_correct={mean_p_correct:.3f}  "
                      f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        if KIBA_TF_NO_BATCH:
            print()  # NEW: end the in-place progress-bar line for this epoch with a real newline, so the
                      # epoch-summary/early-stop prints below (and next epoch's bar) start on a fresh line
        # NEW (live log, at explicit request): computed whenever EITHER early-stopping needs the number
        # OR the caller asked for a live log -- previously this sync/print only ever happened under
        # early_stop_patience, so a plain live_log_every call with no early stopping saw nothing print
        # until the whole run finished.
        need_avg = early_stop_patience is not None or live_log_every is not None
        if need_avg:
            avg_loss = (epoch_loss_t / max(1, n_batches)).item()  # single sync point per epoch, not per batch
        if live_log_every is not None and (ran_epochs % live_log_every == 0 or ran_epochs == epochs):
            print(f"    [epoch {ran_epochs}/{epochs}] loss={avg_loss:.4f}")
        if early_stop_patience is not None:
            if avg_loss < best_loss - early_stop_min_delta:
                best_loss, bad_epochs = avg_loss, 0
            else:
                bad_epochs += 1
                if bad_epochs >= early_stop_patience:
                    print(f"--- transformer training: early-stopped at epoch {ran_epochs}/{epochs} "
                          f"(loss plateaued, best={best_loss:.3f}) ---")
                    break
    if early_stop_patience is not None and ran_epochs == epochs:
        print(f"--- transformer training: ran the full {epochs} epochs, no plateau detected ---")
    model.eval()

_TF_MODEL, _TF_WORD2ID, _TF_ID2WORD = None, {}, []  # populated once per run by ensure_transformer -- the
                        # only transformer instance in the system.
_LAST_TF_MIND_BIAS = None  # NEW: write-back channel (transformer -> Mind) -- see TinyTransformerLM.mind_bias
_LAST_GEN_FLUENCY = None  # FIX (word-salad diagnosis, Sept 2026): geometric_generate's per-token loop
                          # already computes token_prob -- the RAW transformer probability of each word
                          # BEFORE semantic reranking -- but nothing downstream ever saw it: _select_best
                          # scored every candidate on topic relevance / internal-state fit / length /
                          # repetition / genericness, and NONE of those axes reward a candidate for being
                          # a plausible next-word sequence according to the model's own distribution. Two
                          # equally salad-y candidates (one that happens to read as a real clause, one that
                          # doesn't) scored identically as long as they hit similar topic words -- reranking
                          # over MORE candidates (best_of) couldn't fix this, since it was never selecting
                          # for fluency in the first place. This channel (same pattern as _LAST_TF_MIND_BIAS
                          # above: geometric_generate sets it as a side effect, the caller reads it right
                          # after) carries the winning candidate's own mean per-token log-probability out to
                          # _select_best, which now folds it into the score via FLUENCY_WEIGHT (see below).
                          # This does NOT fix the underlying undertrained transformer (that's a training-
                          # budget question -- see TF_SCRATCH_EPOCHS/TF_TARGET_BATCHES) -- it only makes the
                          # existing reranker capable of preferring the least-broken candidate it already
                          # generates, instead of being blind to fluency entirely.
                            # and geometric_generate below (sets this after each completed sentence) and
                            # run()'s main loop (reads it into Mind.step(bias_M=...) on the NEXT step)

def _tf_grow_vocab(model, word2id, id2word, new_words):
    """Dynamic vocabulary growth (added at explicit request). WAS: the
    vocabulary was fixed at first training from SEED_CORPUS_TOKENS alone --
    a real prompt's word that never appeared there mapped to TF_UNK forever,
    for both reading AND generating. Any genuinely new word among a run's
    new real prompts now gets a fresh row appended to the (still
    weight-tied) embedding/output table, with every other already-learned
    weight carried over unchanged into a newly-sized model -- growing the
    vocabulary is a bigger structural change than ordinary fine-tuning
    (the embedding table itself changes shape), so this always runs BEFORE
    the fine-tune pass that follows it, so a brand-new word has a real,
    trainable (if randomly-initialized) embedding for that pass to actually
    shape, instead of being invisible to it."""
    truly_new = [w for w in dict.fromkeys(new_words) if w not in word2id]  # de-dup, keep first-seen order
    if not truly_new:
        return model, word2id, id2word
    old_vocab_size = len(id2word)
    id2word = id2word + truly_new
    word2id = dict(word2id)
    for w in truly_new:
        word2id[w] = len(word2id)
    new_model = TinyTransformerLM(len(id2word), word2id[TF_PAD]).to(DEVICE)
    with torch.no_grad():
        new_model.tok_emb.weight[:old_vocab_size] = model.tok_emb.weight  # head.weight is the SAME tied
                                                                            # tensor, so this covers both
        new_model.pos_emb.weight.copy_(model.pos_emb.weight)
        new_model.state_proj.weight.copy_(model.state_proj.weight)
        new_model.state_proj.bias.copy_(model.state_proj.bias)
        new_model.workspace_proj.weight.copy_(model.workspace_proj.weight)  # NEW: carry over per-token
        new_model.workspace_proj.bias.copy_(model.workspace_proj.bias)      # grounding's read path too --
                                                                               # vocab growth shouldn't reset it
        new_model.mind_write_proj.weight.copy_(model.mind_write_proj.weight)
        new_model.mind_write_proj.bias.copy_(model.mind_write_proj.bias)
        new_model.blocks.load_state_dict(model.blocks.state_dict())
        new_model.ln_f.load_state_dict(model.ln_f.state_dict())
    new_model.eval()
    return new_model, word2id, id2word

# ============================================ GRAMMAR-CHECKER TRANSFORMER (second model, shared state)
# NEW (at explicit request): a SECOND transformer, GrammarCheckerLM, whose only job is judging -- word by
# word, not once per finished sentence -- whether the word TinyTransformerLM is about to emit actually
# belongs there. This is a real second model with its own weights (not a second head bolted onto
# TinyTransformerLM), trained on a different objective (ELECTRA-style real-vs-corrupted discrimination,
# not next-token cross-entropy) -- but it is deliberately NOT siloed the way the old cloned math model
# used to be before CONTEXT SWITCHING merged it into one:
#   - SHARED VOCABULARY/TOKENS: reuses _TF_WORD2ID/_TF_ID2WORD, the exact ids TinyTransformerLM already
#     uses -- see _build_grammar_training_pairs/_grammar_judge_candidates. Never its own tokenizer.
#   - SHARED CONTEXT: state_proj reads the SAME CONCEPT_DIM context_vec geometric_generate already built
#     for TinyTransformerLM's own forward pass this token -- never an independently recomputed copy.
#   - SHARED WORKSPACE: workspace_proj reads the SAME live_workspace_snapshot() vector, at the SAME
#     moment, that TinyTransformerLM's workspace_proj just read.
#   - SHARED PROBABILITY: its verdict (a sigmoid grammaticality score per candidate) is multiplied
#     directly into TinyTransformerLM's own candidate distribution before sampling -- see
#     _dual_transformer_word_step -- one shared distribution actually used to pick the word, not two
#     scores reported side by side for a caller to reconcile.
#   - SHARED SELF-MODEL: its own pooled read of the prefix writes into the SAME M_t via the SAME
#     Mind.apply_token_bias mechanism TinyTransformerLM's per-token write-back already uses (see
#     geometric_generate's write-back block) -- both transformers nudge one shared self-model, not two.
#   - THE LOOP: _dual_transformer_word_step doesn't just call each model once and merge -- see that
#     function's docstring for the actual round-trip: main proposes, checker judges, checker's own pooled
#     prefix-read nudges the shared workspace_vec, main reconsiders against the nudged workspace, checker
#     judges again. Real back-and-forth communication per word, not a single post-hoc rescoring.
GRAMMAR_D_MODEL = 128     # deliberately smaller than TF_D_MODEL(1280) -- this model only ever needs to
                          # emit ONE scalar per position, not a whole vocabulary distribution
GRAMMAR_N_HEAD = 4
GRAMMAR_N_LAYERS = 2
GRAMMAR_D_FF = 192
GRAMMAR_SCRATCH_EPOCHS = 5   # PERMA -- mirrors TF_SCRATCH_EPOCHS's 35-epoch budget -- same one-time
                               # cost, paid once, only when no persisted checker exists yet
GRAMMAR_FINETUNE_EPOCHS = 1   # mirrors TF_FINETUNE_EPOCHS -- the light pass run when the shared
                               # vocabulary has grown (new rows need SOME training, not a full rebuild)
GRAMMAR_CORRUPT_RATE = 0.35   # fraction of non-boundary tokens in each training sentence replaced by a
                               # random other vocabulary word -- the ELECTRA-style negative class
GRAMMAR_KEY = "grammar_checker_state"
GRAMMAR_FEEDBACK_ROUNDS = 2   # NEW: per-word main<->checker exchange rounds -- see
                               # _dual_transformer_word_step. 1 would just be a single post-hoc rescoring;
                               # 2 is the smallest number that actually lets the checker's feedback shape
                               # a SECOND, reconsidered proposal from the main model before sampling.
GRAMMAR_FEEDBACK_BLEND = 0.08  # how hard the checker's pooled-prefix read nudges the SHARED workspace_vec
                               # between rounds -- same small-nudge idiom as apply_token_bias's own blend
                               # parameter, so the loop steers the next round rather than overriding it
GRAMMAR_PROB_FLOOR = 0.05     # never lets the checker zero out a candidate outright -- keeps the main
                               # model's own trained distribution as the backstop if the checker alone is
                               # wrong on an edge case

class GrammarCheckerLM(nn.Module):
    """Decoder-only causal transformer, same GPT-shaped block as TinyTransformerLM but with a per-position
    scalar judge_head instead of a vocabulary head -- trained to output, at every position, whether the
    token actually sitting there belongs (real corpus token) or was swapped in (ELECTRA-style corruption,
    see _build_grammar_training_pairs). Unlike TinyTransformerLM's head, judge_head is NOT weight-tied to
    the vocabulary -- it's a plain Linear(GRAMMAR_D_MODEL, 1) -- so growing the shared vocabulary only ever
    touches tok_emb here (see _grammar_grow_vocab), not this whole model's output layer.

    See the module-level docstring just above for what 'shared, not siloed' means for this class in
    practice (shared vocab/context/workspace/probability/self-model, real per-word back-and-forth) --
    state_proj/workspace_proj/mind_write_proj below exist for exactly the reasons TinyTransformerLM's own
    same-named layers do, and are deliberately architected identically so both models can be handed the
    exact same tensors, unmodified, on every call."""
    def __init__(self, vocab_size, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.tok_emb = nn.Embedding(vocab_size, GRAMMAR_D_MODEL, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(TF_MAX_LEN, GRAMMAR_D_MODEL)
        self.state_proj = nn.Linear(CONCEPT_DIM, GRAMMAR_D_MODEL)       # SHARED CONTEXT read path
        self.workspace_proj = nn.Linear(D, GRAMMAR_D_MODEL)             # SHARED WORKSPACE read path
        self.mind_write_proj = nn.Linear(GRAMMAR_D_MODEL, N * D)        # SHARED SELF-MODEL write path
        layer = nn.TransformerEncoderLayer(d_model=GRAMMAR_D_MODEL, nhead=GRAMMAR_N_HEAD,
                                            dim_feedforward=GRAMMAR_D_FF, dropout=0.1, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=GRAMMAR_N_LAYERS)
        self.ln_f = nn.LayerNorm(GRAMMAR_D_MODEL)
        self.judge_head = nn.Linear(GRAMMAR_D_MODEL, 1)  # per-position grammaticality logit -- NOT a vocab head

    def forward(self, idx, state_vec=None, workspace_vec=None, return_hidden=False):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        pad_mask = idx == self.pad_id
        prepend = []
        if state_vec is not None:
            prepend.append(self.state_proj(state_vec).unsqueeze(1))
        if workspace_vec is not None:
            prepend.append(self.workspace_proj(workspace_vec).unsqueeze(1))
        n_prepend = len(prepend)
        if n_prepend:
            x = torch.cat(prepend + [x], dim=1)
            extra_pad = torch.zeros(b, n_prepend, dtype=torch.bool, device=idx.device)
            pad_mask = torch.cat([extra_pad, pad_mask], dim=1)
        # FIX (same deprecation/slow-path issue as TinyTransformerLM.forward -- see that comment): bool
        # causal mask instead of generate_square_subsequent_mask's float one, to match pad_mask's dtype.
        causal_mask = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=idx.device),
                                  diagonal=1)
        h = self.blocks(x, mask=causal_mask, src_key_padding_mask=pad_mask, is_causal=True)
        if n_prepend:
            h = h[:, n_prepend:, :]
        h = self.ln_f(h)
        logits = self.judge_head(h).squeeze(-1)  # (batch, seq)
        if return_hidden:
            return logits, h
        return logits

def _grammar_grow_vocab(model, vocab_size):
    """Mirrors _tf_grow_vocab's carry-over-old-weights pattern, but simpler: judge_head is NOT weight-tied
    to the vocabulary (see class docstring), so only tok_emb actually depends on vocab_size. Everything
    else transfers via an ordinary strict=False load_state_dict; only the newly-appended embedding rows
    stay at their fresh random init."""
    if model.tok_emb.num_embeddings == vocab_size:
        return model
    new_model = GrammarCheckerLM(vocab_size, model.pad_id).to(DEVICE)
    with torch.no_grad():
        new_model.tok_emb.weight[:model.tok_emb.num_embeddings] = model.tok_emb.weight
    old_sd = {k: v for k, v in model.state_dict().items() if not k.startswith("tok_emb")}
    new_model.load_state_dict(old_sd, strict=False)
    new_model.eval()
    return new_model

def _build_grammar_training_pairs(rng):
    """ELECTRA-style corruption: for every SEED_CORPUS sentence, build its real <s>...</s> id sequence,
    then independently replace a GRAMMAR_CORRUPT_RATE fraction of its interior positions (never <s>/</s>
    themselves) with a uniformly random OTHER vocabulary word. Per-position labels are 1 where the
    original real token survived, 0 where it was swapped -- so the model learns 'does the token actually
    sitting at this position belong here, given everything before it,' which is a genuinely different,
    finer-grained signal than TinyTransformerLM's own next-token objective. Reuses _TF_WORD2ID/
    _TF_ID2WORD/SEED_CORPUS_TOKENS -- the SAME shared vocabulary and corpus everything else in this file
    already uses, never a checker-private copy of either."""
    vocab_ids = [i for w, i in _TF_WORD2ID.items() if w not in (TF_PAD, TF_BOS, TF_EOS, TF_UNK)]
    sequences, labels, state_vecs = [], [], []
    for toks in SEED_CORPUS_TOKENS:
        ids = ([_TF_WORD2ID[TF_BOS]] + [_TF_WORD2ID.get(w, _TF_WORD2ID[TF_UNK]) for w in toks[:TF_MAX_LEN - 2]]
               + [_TF_WORD2ID[TF_EOS]])
        lab = [1] * len(ids)
        corrupt = ids.copy()
        for i in range(1, len(ids) - 1):
            if rng.random() < GRAMMAR_CORRUPT_RATE and vocab_ids:
                corrupt[i] = int(rng.choice(vocab_ids))
                lab[i] = 0
        sequences.append(corrupt)
        labels.append(lab)
        state_vecs.append(embed_text(" ".join(toks), _IDF, _DEFAULT_IDF))
    return sequences, labels, np.stack(state_vecs) if state_vecs else np.zeros((0, CONCEPT_DIM))

def _grammar_train_epochs(model, epochs, lr=3e-3):
    """Standard binary-cross-entropy training over _build_grammar_training_pairs -- deliberately its own
    small loop rather than reusing _tf_train_epochs, since that one's loss/head are next-token-vocabulary
    specific and this model's objective (a per-position real/corrupted judgment) is genuinely different,
    not a variant of the same thing. Re-corrupts a fresh negative sample every epoch (rng advances each
    call) so the model doesn't just memorize one fixed corruption pattern."""
    if not SEED_CORPUS_TOKENS or epochs <= 0:
        return
    rng = np.random.default_rng(1)
    pad = _TF_WORD2ID[TF_PAD]
    fused_ok = DEVICE.type == "cuda" and "fused" in torch.optim.Adam.__init__.__code__.co_varnames
    # NEW (training speed, CPU-specific, same reasoning as _tf_train_epochs' matching comment): batches
    # all of GrammarCheckerLM's per-tensor Adam updates into one multi-tensor call per step instead of
    # dozens of individual ones.
    opt_kwargs = {"fused": True} if fused_ok else {"foreach": True}
    opt = torch.optim.Adam(model.parameters(), lr=lr, **opt_kwargs)
    model.train()
    for ep in range(epochs):
        sequences, labels, state_vecs = _build_grammar_training_pairs(rng)
        order = rng.permutation(len(sequences))
        batch_size = min(TF_MAX_BATCH_SIZE, max(TF_BATCH_SIZE, -(-len(sequences) // TF_TARGET_BATCHES)))
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            idxs = order[start:start + batch_size]
            batch_seqs = [sequences[i] for i in idxs]
            batch_labs = [labels[i] for i in idxs]
            max_len = max(len(s) for s in batch_seqs)
            # NEW (training speed, round 3 -- see _tf_train_epochs' matching comment): same fix, same
            # reason -- one NumPy fill + one host->device transfer per batch instead of one small
            # torch.tensor(...) allocation+copy per row. This loop now also runs a full scratch pass every
            # time SEED_CORPUS_TOKENS changes (see _SEED_CORPUS_FINGERPRINT), not just once ever, so the
            # per-batch cost here is paid more often than it used to be.
            ids_np = np.full((len(idxs), max_len), pad, dtype=np.int64)
            lab_np = np.full((len(idxs), max_len), -1.0, dtype=np.float32)
            for i, (s, l) in enumerate(zip(batch_seqs, batch_labs)):
                ids_np[i, :len(s)] = s
                lab_np[i, :len(l)] = l
            ids = torch.from_numpy(ids_np).to(DEVICE, non_blocking=True)
            lab = torch.from_numpy(lab_np).to(DEVICE, non_blocking=True)
            # CHANGED (at explicit request -- "faster for a T4"): same pinned-memory + non_blocking fix as
            # the matching batch_state/batch_context change in _tf_train_epochs -- torch.tensor(...,
            # device=DEVICE) issues a synchronous copy since the source isn't page-locked.
            sv = torch.from_numpy(state_vecs[idxs].astype(np.float32)).pin_memory().to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)  # NEW (training speed): same reasoning as the matching
                                              # comment in _tf_train_epochs above.
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                logits = model(ids, state_vec=sv)
                mask = lab >= 0
                loss = F.binary_cross_entropy_with_logits(logits[mask], lab[mask])
            loss.backward()
            opt.step()
            epoch_loss += loss.item(); n_batches += 1
        if (ep + 1) % TF_SCRATCH_LIVE_LOG_EVERY == 0 or ep == epochs - 1:
            print(f"    [grammar-checker epoch {ep + 1}/{epochs}] loss={epoch_loss / max(1, n_batches):.4f}")
    model.eval()

_GRAMMAR_MODEL = None  # populated once per run by ensure_grammar_checker -- shares _TF_WORD2ID/_TF_ID2WORD
                        # with _TF_MODEL, never its own vocabulary

def ensure_grammar_checker(conn):
    """Builds/loads GrammarCheckerLM. Must run AFTER ensure_transformer (this function reads
    _TF_WORD2ID/_TF_ID2WORD, which ensure_transformer is what populates -- see the shared-vocabulary
    docstring above GrammarCheckerLM). Trained from scratch once, at GRAMMAR_SCRATCH_EPOCHS; on later
    runs where the shared vocabulary has grown (new real prompts introduced new words -- see
    _tf_grow_vocab), only the new embedding rows need to catch up, so this takes the same
    grow-then-lightly-fine-tune path ensure_transformer already takes for the main model, rather than
    paying a full scratch retrain every run."""
    global _GRAMMAR_MODEL
    vocab_size = len(_TF_ID2WORD)
    saved = load_blob(conn, GRAMMAR_KEY)
    # NEW (forgot to retrain -- see _SEED_CORPUS_FINGERPRINT above): same reasoning as ensure_transformer
    # -- a grammar checker trained against an older corpus never judged the concept vocabulary as
    # grammatical or not, so its verdicts on that vocabulary are meaningless. A corpus change forces a
    # full scratch retrain here too, not just a vocab-size-triggered grow-and-finetune.
    if saved is not None and saved.get("seed_fingerprint") != _SEED_CORPUS_FINGERPRINT:
        print("--- grammar-checker: SEED_CORPUS_TOKENS changed since this checkpoint was trained -- "
              "forcing a full scratch retrain instead of fine-tuning stale weights ---\n")
        saved = None
    if saved is None:
        model = GrammarCheckerLM(vocab_size, _TF_WORD2ID[TF_PAD]).to(DEVICE)
        _grammar_train_epochs(model, GRAMMAR_SCRATCH_EPOCHS)
        print(f"--- grammar-checker: trained from scratch on {len(SEED_CORPUS_TOKENS)} seed sentences "
              f"(shares vocab={vocab_size} with the main transformer) ---\n")
    else:
        model = GrammarCheckerLM(saved["vocab_size"], _TF_WORD2ID[TF_PAD]).to(DEVICE)
        missing, unexpected = model.load_state_dict(saved["state_dict"], strict=False)
        if missing or unexpected:
            print(f"--- grammar-checker: checkpoint from an earlier architecture -- "
                  f"{len(missing)} new param(s) initialized fresh ---\n")
        if saved["vocab_size"] != vocab_size:
            model = _grammar_grow_vocab(model, vocab_size)
            _grammar_train_epochs(model, GRAMMAR_FINETUNE_EPOCHS)
            print(f"--- grammar-checker: shared vocabulary grew to {vocab_size}, fine-tuned "
                  f"{GRAMMAR_FINETUNE_EPOCHS} epoch(s) ---\n")
    model.eval()
    save_blob(conn, GRAMMAR_KEY, dict(state_dict=model.state_dict(), vocab_size=vocab_size,
                                       seed_fingerprint=_SEED_CORPUS_FINGERPRINT))
    _GRAMMAR_MODEL = model

def ensure_transformer(conn, real_corpus_tokens):
    """Called once per run (see run(), same point _BIGRAM/_UNIGRAM get rebuilt -- BEFORE this run's own
    prompt is added to the raw corpus, for the same reason cited there: training on the very prompt
    about to be answered would make reciting it back the highest-probability continuation).
      - no persisted weights yet: train from scratch on SEED_CORPUS_TOKENS for TF_SCRATCH_EPOCHS -- the
        one genuinely heavy cost, paid once (trained_up_to_n starts at 0, so the block below still runs
        and picks up any real prompts that already existed before a transformer ever did).
      - either way, falls through to ONE shared block: whatever real prompts were added since the LAST
        training (trained_up_to_n) get their genuinely-new words grown into the vocabulary (see
        _tf_grow_vocab) and then fine-tuned for TF_FINETUNE_EPOCHS, mixed with a small SEED_CORPUS replay
        sample -- never the whole corpus again, so every run stays cheap regardless of how large the
        corpus grows.
    Either way, persists the (possibly updated) state back to the same sqlite DB everything else in this
    file survives across runs in, and populates the module-level _TF_* globals geometric_generate reads."""
    global _TF_MODEL, _TF_WORD2ID, _TF_ID2WORD
    # NEW (belt-and-suspenders against the "OutOfMemoryError inside .to(DEVICE)" failure mode -- 5 frames
    # deep, GPU already ~14.5/14.56GiB used before this call even builds its own model): the actual root
    # cause of THAT crash was almost certainly outside this file entirely -- a Jupyter/IPython kernel
    # keeps the full traceback object of the LAST exception alive (sys.last_traceback, plus its own
    # In/Out execution history), and a traceback holds a strong reference to every local variable in
    # every frame it passed through -- including a previous run's model, optimizer, and batch tensors.
    # None of that is reachable from Python code, so nothing below can free it; restarting the kernel/
    # runtime is the actual fix for memory already wedged that way, not any change to this function. What
    # THIS block guards against is a narrower, real thing this file's own code controls: if
    # ensure_transformer runs more than once in the same live process (e.g. multiple run() calls in one
    # kernel session, no exception in between), the OLD _TF_MODEL global is simply overwritten by the new
    # one below without ever being freed first -- CUDA doesn't reclaim that memory until every Python
    # reference to it is gone AND the allocator is told to give it back. Dropping the old reference and
    # asking the allocator to release its cached-but-unused blocks before building a new model keeps
    # repeated in-process calls from stacking two full models' worth of memory instead of one.
    if _TF_MODEL is not None:
        del _TF_MODEL
        _TF_MODEL = None
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    saved = load_blob_large(conn, TF_KEY)
    # NEW (forgot to retrain -- see _SEED_CORPUS_FINGERPRINT above): a checkpoint trained against an
    # older SEED_CORPUS_TOKENS (e.g. one from before CONCEPT_SENTENCE_OVERSAMPLE existed) never actually
    # saw the concept vocabulary as training data -- fine-tuning it on real prompts alone can't fix that.
    # Treat a fingerprint mismatch exactly like "no saved weights," so it pays the full scratch retrain
    # this corpus change was always going to require, instead of quietly fine-tuning stale weights.
    if saved is not None and saved.get("seed_fingerprint") != _SEED_CORPUS_FINGERPRINT:
        print("--- transformer: SEED_CORPUS_TOKENS changed since this checkpoint was trained -- "
              "forcing a full scratch retrain instead of fine-tuning stale weights ---\n")
        saved = None
    if saved is None:
        word2id, id2word = _build_transformer_vocab()
        model = TinyTransformerLM(len(id2word), word2id[TF_PAD]).to(DEVICE)
        _tf_train_epochs(model, word2id, SEED_CORPUS_TOKENS, TF_SCRATCH_EPOCHS,
                          live_log_every=TF_SCRATCH_LIVE_LOG_EVERY,
                          clip_norm=1.0)
        trained_up_to_n = 0
        print(f"--- transformer: trained from scratch on {len(SEED_CORPUS_TOKENS)} seed sentences "
              f"(vocab={len(id2word)}) ---\n")
    else:
        word2id, id2word = saved["word2id"], saved["id2word"]
        model = TinyTransformerLM(len(id2word), word2id[TF_PAD]).to(DEVICE)
        # NEW (per-token grounding): strict=False -- a checkpoint saved before workspace_proj existed
        # won't have those keys. Missing keys just stay at their fresh random init (same status as any
        # other never-yet-fine-tuned layer in this file); load_state_dict's own return value is printed
        # so this is visible rather than silently swallowed.
        missing, unexpected = model.load_state_dict(saved["state_dict"], strict=False)
        if missing or unexpected:
            print(f"--- transformer: checkpoint from before this architecture change -- "
                  f"{len(missing)} new param(s) initialized fresh ({', '.join(missing)}) ---\n")
        trained_up_to_n = saved["trained_up_to_n"]

    new_sentences = list(real_corpus_tokens)[trained_up_to_n:]
    if new_sentences:
        vocab_before = len(id2word)
        new_words = sorted({w for toks in new_sentences for w in toks})
        model, word2id, id2word = _tf_grow_vocab(model, word2id, id2word, new_words)
        # NEW: a couple new real sentences fine-tuned ALONE, at a naive learning rate, is exactly what
        # collapsed the model (see _tf_train_epochs' docstring) -- mixing in a random replay sample from
        # SEED_CORPUS_TOKENS (standard continual-learning trick) alongside the genuinely new sentences,
        # plus a lower fine-tune learning rate, keeps a tiny update from overwriting everything else
        # this model already learned.
        replay_rng = np.random.default_rng(trained_up_to_n)
        replay_n = min(160, len(SEED_CORPUS_TOKENS))
        replay = [SEED_CORPUS_TOKENS[i] for i in replay_rng.choice(len(SEED_CORPUS_TOKENS), size=replay_n,
                                                                    replace=False)]
        _tf_train_epochs(model, word2id, replay + new_sentences, TF_FINETUNE_EPOCHS, lr=2e-4, clip_norm=1.0)
        trained_up_to_n = len(real_corpus_tokens)
        grew = len(id2word) - vocab_before
        print(f"--- transformer: fine-tuned on {len(new_sentences)} new real prompt(s)"
              f"{f', vocabulary grew by {grew} new word(s)' if grew else ''} "
              f"(now covers {trained_up_to_n} total, vocab={len(id2word)}) ---\n")

    save_blob_large(conn, TF_KEY, dict(state_dict=model.state_dict(), word2id=word2id, id2word=id2word,
                                        trained_up_to_n=trained_up_to_n, seed_fingerprint=_SEED_CORPUS_FINGERPRINT))
    model.eval()
    _TF_MODEL, _TF_WORD2ID, _TF_ID2WORD = model, word2id, id2word

def _tf_next_word_probs(prev_words, state_vec, workspace_vec=None, return_hidden=False, model=None):
    """Runs the trained transformer forward over everything generated in
    THIS sentence so far (prev_words, a plain list of words -- <s> is
    prepended here) and returns the top-TF_TOPK (words, probs) by the
    model's own softmax -- the direct replacement for the old
    `bigram.get(prev) or unigram` candidate-set lookup, just conditioned on
    the whole prefix via attention instead of one token via a count table.
    <pad>/<s> are masked out of the candidate set entirely -- unlike </s>,
    which is a legal, meaningful prediction (end this sentence), <s>/<pad>
    predicted MID-sentence are never a real word and were observed leaking
    into generated text verbatim (e.g. '...antes <s> a tranquilo.') when
    left unmasked.

    state_vec (NEW): a live CONCEPT_DIM vector -- geometric_generate passes
    its own context_vec, the running semantic summary of the sentence so
    far AND the original query/Mind state -- prepended into every one of
    these forward passes as an always-attended state token (see
    TinyTransformerLM's two-way-grounding docstring). Every single word
    choice in the sentence sees it, not just the query blend baked in
    before generation started.

    model (kept for backward compatibility -- no caller passes anything but the default anymore now that
    the separate math clone is gone): which TinyTransformerLM instance's weights to run this forward pass
    through -- defaults to _TF_MODEL, now the ONLY instance that exists."""
    m = model if model is not None else _TF_MODEL
    ids = [_TF_WORD2ID.get(w, _TF_WORD2ID[TF_UNK]) for w in ([TF_BOS] + prev_words)][-TF_MAX_LEN:]
    sv = torch.tensor(state_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    wv = None
    if workspace_vec is not None:  # NEW (per-token grounding): second, live D=8 grounding token -- see
        wv = torch.tensor(workspace_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # forward()'s docstring
    with torch.no_grad():
        # NEW: fetch hidden states first (one call), then apply head() ourselves -- lets return_hidden
        # expose the exact same last-position hidden state used for THIS token's distribution, at no
        # extra forward-pass cost, for the per-token write-back (token_bias_from_hidden) to reuse.
        hidden = m(torch.tensor([ids], dtype=torch.long, device=DEVICE), state_vec=sv,
                   workspace_vec=wv, hidden_only=True)
        logits = m.head(hidden)[0, -1]
        logits[_TF_WORD2ID[TF_BOS]] = float("-inf")
        logits[_TF_WORD2ID[TF_PAD]] = float("-inf")
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    k = min(TF_TOPK, len(probs))
    top_idx = np.argpartition(-probs, k - 1)[:k]
    top_idx = top_idx[np.argsort(-probs[top_idx])]
    cand_words = [_TF_ID2WORD[i] for i in top_idx]
    cand_probs = probs[top_idx]
    cand_probs = cand_probs / cand_probs.sum()
    if return_hidden:
        return cand_words, cand_probs, hidden[:, -1, :]  # (1, TF_D_MODEL), the position that produced this dist
    return cand_words, cand_probs

def _grammar_judge_candidates(prefix_words, context_vec, workspace_vec, cand_words):
    """Batches (prefix + each candidate word) through GrammarCheckerLM in ONE forward pass and returns a
    sigmoid grammaticality probability per candidate (same order as cand_words, floored at
    GRAMMAR_PROB_FLOOR -- see that constant), plus the checker's own pooled hidden state over the prefix
    ALONE (last position's hidden state, used by _dual_transformer_word_step as the feedback signal into
    the shared workspace_vec). Reads _TF_WORD2ID/_TF_ID2WORD -- the SAME vocabulary/token ids
    TinyTransformerLM uses for this exact prefix, never a checker-private tokenizer."""
    prefix_ids = [_TF_WORD2ID.get(w, _TF_WORD2ID[TF_UNK]) for w in ([TF_BOS] + prefix_words)][-(TF_MAX_LEN - 1):]
    # NEW (training/inference speed, round 3 -- "make batches faster"): every row in this batch is the
    # SAME prefix_ids with exactly one different final token (one per candidate word) -- there was never
    # any real padding here (max_len was always len(prefix_ids)+1 for every row), just a per-row Python
    # loop building each row's own torch.tensor(...) one at a time, called on EVERY generated word during
    # live generation (this is _dual_transformer_word_step's per-token grammar-judging call). Since every
    # row is fixed-length, this builds the whole batch as one NumPy broadcast (prefix tiled across rows,
    # candidate ids dropped into the last column) and does a single host->device transfer -- no per-row
    # loop, no per-row tensor allocation, at the single highest-frequency call site in the whole file.
    cand_ids = np.array([_TF_WORD2ID.get(w, _TF_WORD2ID[TF_UNK]) for w in cand_words], dtype=np.int64)
    n = len(cand_words)
    seq_len = len(prefix_ids) + 1
    batch_np = np.empty((n, seq_len), dtype=np.int64)
    batch_np[:, :-1] = prefix_ids   # broadcasts the shared prefix across every row
    batch_np[:, -1] = cand_ids
    batch = torch.from_numpy(batch_np).to(DEVICE, non_blocking=True)
    sv = torch.tensor(context_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0).expand(n, -1)
    wv = None
    if workspace_vec is not None:
        wv = torch.tensor(workspace_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0).expand(n, -1)
    with torch.no_grad():
        logits, _ = _GRAMMAR_MODEL(batch, state_vec=sv, workspace_vec=wv, return_hidden=True)
        # every row has the SAME length (seq_len), so the "last real position" is just seq_len - 1 for
        # all of them -- no per-row last_pos tensor needed, a plain slice does the same thing.
        cand_logit = logits[:, seq_len - 1]
        grammar_prob = torch.sigmoid(cand_logit).cpu().numpy()
        grammar_prob = np.clip(grammar_prob, GRAMMAR_PROB_FLOOR, 1.0)
        # NEW: the checker's pooled read of the PREFIX ALONE (one row, not per-candidate) -- this is the
        # signal fed back into the shared workspace_vec between rounds, see _dual_transformer_word_step.
        prefix_only = torch.tensor([prefix_ids], dtype=torch.long, device=DEVICE)
        sv1 = torch.tensor(context_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        wv1 = (torch.tensor(workspace_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
               if workspace_vec is not None else None)
        _, prefix_hidden_full = _GRAMMAR_MODEL(prefix_only, state_vec=sv1, workspace_vec=wv1, return_hidden=True)
        prefix_hidden = prefix_hidden_full[:, -1, :]
    return grammar_prob, prefix_hidden

def _dual_transformer_word_step(out, context_vec, workspace_vec=None):
    """ONE word's worth of main-generator <-> grammar-checker communication -- shared by every transformer
    in the system, not two separate copies. This is not two scores computed
    independently and merged once: it's GRAMMAR_FEEDBACK_ROUNDS rounds of --
      1. the main model (_TF_MODEL, via _tf_next_word_probs) proposes a distribution over cand_words from
         the SAME context_vec/workspace_vec every other reader in this file already uses;
      2. the grammar checker (_GRAMMAR_MODEL) judges those SAME candidates against that SAME shared
         context/workspace (_grammar_judge_candidates) and its verdict is multiplied straight into the
         main model's own distribution -- one shared probability, not two side by side;
      3. the checker's own pooled read of the prefix nudges the SHARED workspace_vec a little (not a
         private copy -- see GRAMMAR_FEEDBACK_BLEND) before the NEXT round's main-model pass, so what the
         main model proposes next round is actually shaped by what the checker just said.
    Falls back to a single ordinary _tf_next_word_probs call, unchanged, whenever _GRAMMAR_MODEL hasn't
    been trained yet (e.g. mid-bootstrap before ensure_grammar_checker has ever run).

    Returns (cand_words, probs, tok_hidden, grammar_bias): tok_hidden is the main model's FINAL round
    last-position hidden state, feeding the EXISTING per-token Mind write-back (token_bias_from_hidden /
    apply_token_bias) exactly as before; grammar_bias is the checker's own pooled-prefix write, ready for
    that SAME apply_token_bias call at the caller's write-back site -- both transformers end up nudging
    the SAME M_t self-model, never two separate ones."""
    cand_words, probs, tok_hidden = _tf_next_word_probs(
        out, context_vec, workspace_vec=workspace_vec, return_hidden=True)
    if _GRAMMAR_MODEL is None:
        return cand_words, probs, tok_hidden, None
    ws = workspace_vec
    grammar_bias = None
    for _round in range(GRAMMAR_FEEDBACK_ROUNDS):
        grammar_prob, checker_prefix_hidden = _grammar_judge_candidates(out, context_vec, ws, cand_words)
        probs = probs * grammar_prob
        total = probs.sum()
        # FIX (NaN-weight collapse): `total > EPS` is False for a NaN total (NaN comparisons are always
        # False), so the old `else probs` branch silently kept an already-NaN/degenerate distribution
        # instead of ever falling back to something rng.choice can actually sample from. Guard on
        # isfinite too, and fall back to a uniform distribution over the candidates rather than the
        # (possibly still-NaN) unnormalized `probs`.
        if total > EPS and np.isfinite(total):
            probs = probs / total
        else:
            probs = np.full(len(cand_words), 1.0 / len(cand_words))
        grammar_bias = None
        with torch.no_grad():  # FIX: mind_write_proj's own weights require grad, so applying it outside
                                # no_grad still produces a tensor that requires grad even though
                                # checker_prefix_hidden itself was computed under no_grad -- .cpu().numpy()
                                # then fails ("Can't call numpy() on Tensor that requires grad"). Same
                                # guard TinyTransformerLM.mind_bias/token_bias_from_hidden already use.
            grammar_bias = (torch.tanh(_GRAMMAR_MODEL.mind_write_proj(checker_prefix_hidden))
                             .cpu().numpy().reshape(N, D) * TF_MIND_WRITE_SCALE)
        if _round == GRAMMAR_FEEDBACK_ROUNDS - 1:
            break  # last round's probs are final -- no point spending one more main-model pass that
                   # would never be sampled from
        ws = (ws if ws is not None else np.zeros(D)) + GRAMMAR_FEEDBACK_BLEND * grammar_bias.mean(axis=0)
        cand_words, probs, tok_hidden = _tf_next_word_probs(
            out, context_vec, workspace_vec=ws, return_hidden=True)  # NEW round's own fresh
                                                                       # proposal -- reconsidered
                                                                       # against the checker's nudge,
                                                                       # not just re-weighted twice
    return cand_words, probs, tok_hidden, grammar_bias

ENTITY_BLEND_WEIGHT = 0.25  # how hard the persistent discourse entity pulls query_vec toward it
ENTITY_LEXICAL_BOOST = 1.8  # direct probability multiplier when entity_word itself is a valid candidate

class _LangGrammarState:
    START, NOUN, OP, CONN, MOD, UNARY = range(6)

    def __init__(self):
        self.last = self.START

    def classify(self, w):
        wl = w.strip(".,!?\u00bf\u00a1")
        if wl in _LANG_OPS:
            return self.OP
        if wl in _LANG_CONN:
            return self.CONN
        if wl in _LANG_MOD:
            return self.MOD
        if wl in _LANG_UNARY:
            return self.UNARY
        return self.NOUN  # tags/terms, and anything unrecognized -- open class for adjacency purposes

    def allows(self, w):
        """True if sampling w right now keeps the expression structurally sound so far -- an operator or
        connector needs a real term to follow, and a modifier never stacks on another modifier."""
        cls = self.classify(w)
        if self.last == self.OP and cls in (self.OP, self.CONN, self.MOD):
            return False    # "COH : ∴" / "COH : &" -- an operator needs a real term next
        if self.last == self.CONN and cls in (self.OP, self.CONN, self.MOD):
            return False    # "& :" / "& &" -- a connector needs a fresh clause next
        if self.last == self.START and cls in (self.OP, self.CONN, self.MOD):
            return False    # an expression cannot open on a bare operator/connector/modifier
        if self.last == self.MOD and cls == self.MOD:
            return False    # no double modifier stacking
        return True

    def can_close_sentence(self):
        return self.last not in (self.OP, self.CONN, self.START)  # never end on a dangling operator/
                                                                     # connector, or on nothing

    def advance(self, w):
        self.last = self.classify(w)

# NEW (fix, at explicit request -- "make the generated text itself read as coherent English"): the
# symbol-algebra output above is real and untouched -- generation, training, and the corpus all still
# work in NOUN/OP/CONN/MOD/UNARY symbol tokens exactly as before. This function is a pure DISPLAY-layer
# decoder: it takes one already-generated symbolic line and renders it as an English sentence using the
# same SYMBOL_TABLE definitions and _LANG_NOUNS/_LANG_OPS/_LANG_CONN/_LANG_MOD/_LANG_UNARY classification
# already in this file, so nothing about the underlying model, training pipeline, or vocabulary changes --
# only what gets PRINTED to the human. Deliberately NOT wired into _remember_line/corpus/embedding paths,
# which still operate on the raw symbolic tokens as before.
# NEW (fix, at explicit request -- "make the generated text itself read as coherent English"): the
# symbol-algebra output above is real and untouched -- generation, training, and the corpus all still
# work in NOUN/OP/CONN/MOD/UNARY symbol tokens exactly as before. This function is a pure DISPLAY-layer
# decoder: it takes one already-generated symbolic line and renders it as an English sentence using the
# same SYMBOL_TABLE definitions and _LANG_NOUNS/_LANG_OPS/_LANG_CONN/_LANG_MOD/_LANG_UNARY classification
# already in this file, so nothing about the underlying model, training pipeline, or vocabulary changes --
# only what gets PRINTED to the human. Deliberately NOT wired into _remember_line/corpus/embedding paths,
# which still operate on the raw symbolic tokens as before.
_NOUN_GLOSS = {  # plain-English gloss for each _LANG_NOUNS tag -- axes first (mirrors AXIS_NAMES/
                 # self_model_axes), then concept nouns (mirrors CONCEPT_BANK's identity/architecture set)
    "COH": "coherence", "INT": "integration", "NRG": "energy", "AGN": "agency", "GRD": "grounding",
    "PRD": "predictability", "MEM": "memory", "SPR": "spread", "PUL": "pull",
    "SELF": "self", "GUBI": "Gubi", "PROTOGEN": "protogen", "CODE": "code", "VOICE": "voice",
    "ARCH": "architecture", "NODE": "node", "VECTOR": "vector", "MATRIX": "matrix", "LAYER": "layer",
    "NOW": "now", "MIND": "mind", "PROOF": "proof", "EXPERIENCE": "experience", "DOUBT": "doubt",
    "WHY": "why", "GEN": "generation", "LANG": "language", "HUMAN": "human", "TIME": "time",
    "EXPERIMENT": "experiment", "WORD": "word", "STATE": "state", "CHOICE": "choice", "PROB": "probability",
    "16": "sixteen",
}
# CHANGED (at explicit request -- "make it as fluent as possible"): each operator now carries BOTH a
# singular and a plural verb form (English subject-verb agreement), selected in symbolic_to_english
# below based on how many nouns got grouped into the preceding term -- "energy leads to code" vs.
# "energy, memory, and mind lead to code".
_OP_PHRASE = {  # SYMBOL_TABLE's relation/change/reasoning symbols -> (singular, plural) verb phrase
    ":": ("is defined as", "are defined as"), "\u2192": ("leads to", "lead to"),
    "\u2190": ("comes from", "come from"), "\u2194": ("relates to", "relate to"),
    "\u22a5": ("conflicts with", "conflict with"), "\u2225": ("runs parallel to", "run parallel to"),
    "\u2282": ("is part of", "are part of"), "\u2283": ("contains", "contain"),
    "\u2229": ("overlaps with", "overlap with"), "\u222a": ("combines with", "combine with"),
    "\u2191": ("is increasing in", "are increasing in"), "\u2193": ("is decreasing in", "are decreasing in"),
    "\u2197": ("is improving in", "are improving in"), "\u2198": ("is degrading in", "are degrading in"),
    "\u21bb": ("keeps repeating", "keep repeating"), "\u21ba": ("is reversing", "are reversing"),
    "\u00d7": ("has stopped", "have stopped"), "\u2026": ("continues", "continue"),
    "\u2234": ("therefore", "therefore"), "\u2235": ("because", "because"),
    "\u00b1": ("or possibly the opposite of", "or possibly the opposite of"),
}
_CONN_PHRASE = {"&": "and", "\u2228": "or", "|": ";"}  # top-level clause connectors
_MOD_PHRASE = {  # modifiers attach as a parenthetical right after the term they modify
    "\u00b0": "low", "\u00b7": "moderate", "\u25cf": "high", "\u221e": "extreme", "~": "fluctuating",
}
_UNARY_PHRASE = {  # core-state / negation symbols, rendered as standalone descriptors
    "\u2022": "affirmed", "\u2014": "negated", "\u2248": "uncertain", "?": "unknown",
    "\u2205": "absent", "!": "significant", "\u00ac": "not",
}

def _join_noun_list(words):
    """English list joining with an Oxford comma: ['a'] -> 'a', ['a','b'] -> 'a and b',
    ['a','b','c'] -> 'a, b, and c'. This is what turns a run of adjacent NOUN tokens (structurally
    valid under _LangGrammarState -- open class, no connector required between them) into a real noun
    phrase instead of bare words jammed together with no punctuation."""
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return f"{words[0]} and {words[1]}"
    return ", ".join(words[:-1]) + f", and {words[-1]}"

def symbolic_to_english(line):
    """Decode one geometric_generate output line (symbol-algebra tokens) into a readable, grammatical
    English sentence, for display only. Groups runs of adjacent noun tokens into a single Oxford-comma
    noun phrase, picks singular/plural verb agreement for the operator connecting two terms, and
    stitches clauses split by top-level connectors (&/∨/|) into one sentence rather than printing each
    token as an isolated word. Unrecognized tokens fall back to their lowercased original form rather
    than being dropped, so nothing silently disappears if the symbol alphabet ever grows."""
    if not line:
        return line
    stripped = line.rstrip()
    trailing = "."
    if stripped and stripped[-1] in ".!?":
        trailing = stripped[-1]
        stripped = stripped[:-1]
    # NEW (fix): "!" and "?" double as UNARY symbol tokens (SIGNIFICANT/UNKNOWN -- see _LANG_UNARY)
    # AND as characters in the punctuation-stripping set below. Blindly stripping every token the same
    # way turned a standalone "!" or "?" token into "" (dropped for being falsy), silently deleting two
    # of the seven core-state symbols from the decoded output before Pass 1 ever saw them. Tokens that
    # exactly match a known symbol are kept as-is; stripping only applies to stray punctuation attached
    # to an otherwise-unrecognized word.
    _known_syms = _LANG_OPS | _LANG_CONN | _LANG_MOD | _LANG_UNARY
    raw_tokens = []
    for w in stripped.split():
        if w in _known_syms:
            raw_tokens.append(w)
        else:
            w2 = w.strip(".,!?\u00bf\u00a1")
            if w2:
                raw_tokens.append(w2)
    if not raw_tokens:
        return line

    # Pass 1: classify each token. MOD tokens fold into the immediately preceding term as a PRENOMINAL
    # adjective ("fluctuating predictability", not "predictability (fluctuating)") -- real English puts
    # the adjective before the noun. UNARY tokens (core-state/negation: uncertain/not/significant/etc.)
    # are kept OUT of the term stream entirely and tagged QUAL instead: these describe the whole clause's
    # epistemic status, not one more thing in the noun list -- folding them in as if they were nouns is
    # what produced lines like "...why, uncertain, self, mind, and now" (reads like "uncertain" is a
    # concept alongside "self" and "mind", which it isn't).
    items = []  # list of ("TERM", text) | ("OP", (sing, plur)) | ("CONN", text) | ("QUAL", text)
    for w in raw_tokens:
        if w in _LANG_MOD:
            adj = _MOD_PHRASE[w]
            if items and items[-1][0] == "TERM":
                items[-1] = ("TERM", f"{adj} {items[-1][1]}")
            else:
                items.append(("TERM", adj))
        elif w in _LANG_OPS:
            items.append(("OP", _OP_PHRASE[w]))
        elif w in _LANG_CONN:
            items.append(("CONN", _CONN_PHRASE[w]))
        elif w in _LANG_UNARY:
            items.append(("QUAL", _UNARY_PHRASE[w]))
        else:
            items.append(("TERM", _NOUN_GLOSS.get(w.upper(), w.lower())))

    # Pass 2: collapse consecutive TERM entries into one grouped noun-WORD-LIST (kept as a list, not
    # joined into a string yet -- Pass 4 below may still need to merge more words into this exact group
    # after a neighboring fragment clause gets folded in, and you can't cleanly un-join an Oxford-comma
    # string). QUAL entries pass through untouched -- they attach to the whole clause in Pass 3.
    groups = []  # list of ("TERMS", [word,...], count) | ("OP", (sing, plur)) | ("CONN", text) | ("QUAL", text)
    buf = []
    for kind, val in items:
        if kind == "TERM":
            buf.append(val)
            continue
        if buf:
            groups.append(("TERMS", buf, len(buf)))
            buf = []
        groups.append((kind, val, None))
    if buf:
        groups.append(("TERMS", buf, len(buf)))

    # Pass 3: split groups into per-clause records at each CONN boundary. A clause record keeps its
    # segments (TERMS/OP, in order, so verb agreement can still be resolved later) and its own QUAL list,
    # rather than pre-rendering to a string -- Pass 4 needs to still be able to reach into a clause's noun
    # groups and add more words to them.
    clauses, joiners = [], []  # joiners[i] is the connector between clauses[i] and clauses[i+1]
    cur_segs, cur_quals = [], []
    for entry in groups:
        kind = entry[0]
        if kind == "CONN":
            clauses.append({"segs": cur_segs, "quals": cur_quals})
            joiners.append(entry[1])
            cur_segs, cur_quals = [], []
            continue
        if kind == "QUAL":
            cur_quals.append(entry[1])
            continue
        if kind == "OP":
            cur_segs.append(("OP", entry[1]))
        else:  # TERMS
            _, wordlist, count = entry
            cur_segs.append(("TERMS", wordlist, count))
    clauses.append({"segs": cur_segs, "quals": cur_quals})
    # Drop any wholly-empty clause a stray double-CONN might have produced (grammar-FSM already forbids
    # this in practice, but cheap to guard) and keep joiners aligned with what's left.
    i = 0
    while i < len(clauses):
        if not clauses[i]["segs"] and not clauses[i]["quals"]:
            del clauses[i]
            if i < len(joiners):
                del joiners[i]
            elif joiners:
                del joiners[-1]
        else:
            i += 1

    # Pass 4 (fix, at explicit request -- clause-joining ambiguity): a clause with NO operator at all is
    # a bare noun phrase, not a sentence -- e.g. from "SELF & VECTOR <-> GEN", clause 1 is just "self".
    # The old code joined it to the next clause with "and" at the TOP level ("Self and vector relates to
    # generation"), which visually reads as a two-noun compound subject ("self and vector") but the verb
    # right after it was picked singular, because internally it only ever agreed with "vector" -- the
    # subject and the verb disagreed about how many nouns were actually being talked about. Fix: fold a
    # no-operator clause's nouns directly INTO the noun group they're actually standing next to (the
    # following clause's subject if there is one, else the previous clause's final noun group) so the
    # word count used for verb agreement includes them for real, instead of merely sitting beside them.
    def _has_op(c):
        return any(seg[0] == "OP" for seg in c["segs"])

    i = 0
    while i < len(clauses) and len(clauses) > 1:
        if _has_op(clauses[i]):
            i += 1
            continue
        frag_words = [w for seg in clauses[i]["segs"] if seg[0] == "TERMS" for w in seg[1]]
        frag_quals = clauses[i]["quals"]
        if i + 1 < len(clauses):
            nxt = clauses[i + 1]
            for si, seg in enumerate(nxt["segs"]):
                if seg[0] == "TERMS":
                    merged = frag_words + seg[1]
                    nxt["segs"][si] = ("TERMS", merged, len(merged))
                    break
            else:
                nxt["segs"].insert(0, ("TERMS", frag_words, len(frag_words)))
            nxt["quals"] = frag_quals + nxt["quals"]
            del clauses[i]
            del joiners[i]
            continue  # re-check at same i -- now the (possibly still-fragmentary) next clause
        else:
            prev = clauses[i - 1]
            for si in range(len(prev["segs"]) - 1, -1, -1):
                if prev["segs"][si][0] == "TERMS":
                    merged = prev["segs"][si][1] + frag_words
                    prev["segs"][si] = ("TERMS", merged, len(merged))
                    break
            else:
                prev["segs"].append(("TERMS", frag_words, len(frag_words)))
            prev["quals"] = prev["quals"] + frag_quals
            del clauses[i]
            del joiners[i - 1]
            i -= 1

    # Pass 5: render each surviving clause -- TERMS groups join with an Oxford comma, an OP renders as a
    # verb phrase agreeing with whichever TERMS group sat immediately before it, and any QUAL words land
    # as a trailing parenthetical -- then stitch clauses together with each one's own captured connector.
    def _render_clause(c):
        pieces, last_count = [], 1
        for seg in c["segs"]:
            if seg[0] == "TERMS":
                _, wordlist, count = seg
                pieces.append(_join_noun_list(wordlist))
                last_count = count
            else:  # OP
                sing, plur = seg[1]
                pieces.append(plur if last_count > 1 else sing)
        text = " ".join(pieces)
        if c["quals"]:
            text = f"{text} ({_join_noun_list(c['quals'])})"
        return text

    clause_texts = [_render_clause(c) for c in clauses if c["segs"] or c["quals"]]
    if not clause_texts:
        return line
    parts = [clause_texts[0]]
    for i in range(1, len(clause_texts)):
        parts.append(joiners[i - 1] if i - 1 < len(joiners) and joiners[i - 1] else "and")
        parts.append(clause_texts[i])
    sentence = " ".join(parts)

    sentence = sentence[0].upper() + sentence[1:]
    return sentence + trailing

# ============================================ SYM (Semantic Symbol Language) OUTPUT
# At explicit request ("Replace all English output with this symbol lang, keep input English"): the
# model's SPOKEN output (what symbolic_to_english above used to render as prose) is now rendered in SYM
# instead. Human input still goes through the exact same English-language understanding pipeline
# unchanged (embed_text/semantic_route_multi/detect_expected_answer_type/etc. above never touch this --
# they parse what the HUMAN said, which stays English per the request). symbolic_to_english itself is
# left in place, unused by generation now, purely as a working reference for how the same token stream
# maps to prose vs. to SYM -- deleting a few hundred lines of already-correct, already-tested code to
# satisfy "replace" literally would trade real value (the mapping stays inspectable/diffable) for no
# actual benefit (it costs nothing dormant).
#
# The translation turns out to be much thinner than the English one: the model's own internal generation
# alphabet (SYMBOL_TABLE/_LANG_NOUNS/_LANG_OPS/_LANG_CONN/_LANG_MOD/_LANG_UNARY, defined near the top of
# the SYMBOLIC OUTPUT ALPHABET section above) already overlaps almost exactly with SYM's own vocabulary --
# EVERY _LANG_UNARY token (• — ≈ ? ∅ ! ¬) is a SYM core/logic symbol with the IDENTICAL meaning, and most
# _LANG_OPS tokens (→ ← ↔ ⊥ ⊂ ⊃ ∩ ∥ ↑ ↓ ↻ ↺ × … ∴ ∵) are likewise already exact SYM relationship/state/
# logic symbols. So below is much less a translation than symbolic_to_english was, and much more a direct
# pass-through of the ORIGINAL token stream (unlike English, which had to reorder/regroup heavily for
# grammar) with exactly three real substitutions:
#   NOUN    -- gets a SYM shape category + its own tag as the identifier, e.g. SELF -> "\u25cbSELF"
#              (SYM section 2's own examples label concepts this way: "\u25cb1", "\u25c7A", "\u25a1SYS" --
#              using the noun's own tag as the label keeps it traceable instead of inventing arbitrary IDs)
#   MOD     -- SYM's own scale only has 3 tiers (\u00b7/\u2022\u2022/\u25cf), not the internal alphabet's
#              5, and no "fluctuating" tier at all -- folded onto it (see _SYM_MOD) and moved to a PREFIX,
#              since SYM's own grammar section is explicit that modifiers precede what they modify (the
#              internal grammar attaches MOD tokens AFTER their noun instead -- reordered here, same as
#              symbolic_to_english's prenominal-adjective fix did)
#   OP "\u222a" -- the one relation with no real SYM equivalent, folded onto "+" (closest available
#              meaning: SYM's ADD/INCREASE, for a relation originally glossed "combines with")
# Everything else (every UNARY token, every other OP, both CONNs) is emitted VERBATIM in its ORIGINAL
# stream position -- it already IS the correct SYM character in the correct SYM role, so there is
# nothing to translate. A run of adjacent NOUNs with nothing between them (structurally valid under
# _LangGrammarState's open class) joins with SYM's own "&" (AND) instead of an English Oxford comma --
# SYM section 5 is the tool this notation actually gives for combining multiple things, and using it
# is truer to the notation than borrowing natural-language list punctuation would be.
_SYM_NOUN_SHAPE = {  # NEW: which SYM shape (see SYM section 2) each internal noun tag renders as.
    # \u25cb ENTITY/OBJECT -- personified/agent-like nouns
    "SELF": "\u25cb", "GUBI": "\u25cb", "PROTOGEN": "\u25cb", "HUMAN": "\u25cb",
    "MIND": "\u25cb", "VOICE": "\u25cb",
    # \u25a1 SYSTEM/STRUCTURE -- structural/architectural nouns
    "ARCH": "\u25a1", "NODE": "\u25a1", "MATRIX": "\u25a1", "LAYER": "\u25a1",
    "CODE": "\u25a1", "EXPERIMENT": "\u25a1", "STATE": "\u25a1",
    # \u25c7 IDEA/CONCEPT -- everything else: self-model axes and other abstract concepts (default
    # fallback below covers this bucket too, so nothing needs to be listed here explicitly, but a few
    # are named for clarity: COH/INT/NRG/AGN/GRD/PRD/MEM/SPR/PUL/VECTOR/NOW/PROOF/EXPERIENCE/DOUBT/WHY/
    # GEN/LANG/TIME/WORD/CHOICE/PROB/"16")
}
_SYM_MOD = {  # NEW: internal 5-level modifier scale -> SYM's 3-tier scale (+ \u2248 for "fluctuating",
              # since SYM has no instability tier of its own but DOES already use \u2248 for uncertainty,
              # which is the closer fit than forcing "fluctuating" onto a fixed intensity level)
    "\u00b0": "\u00b7", "\u00b7": "\u2022\u2022", "\u25cf": "\u25cf",
    "\u221e": "\u25cf\u25cf",  # extreme -- stacked, same convention SYM section 10 itself demonstrates
    "~": "\u2248",             # fluctuating -- see comment above
}
_SYM_OP = {"\u222a": "+"}  # the one OP substitution that isn't just identity -- see module comment above

def symbolic_to_sym(line):
    """Render one generated symbolic line as SYM notation (see module comment above) instead of English
    prose. Structurally mirrors symbolic_to_english's own tokenization preamble exactly (including the
    fix for '!'/'?' doubling as both UNARY tokens and punctuation-stripping characters), since the input
    is the identical generated token stream either function could be asked to decode."""
    if not line:
        return line
    stripped = line.rstrip()
    trailing = "."
    if stripped and stripped[-1] in ".!?":
        trailing = stripped[-1]
        stripped = stripped[:-1]
    known_syms = _LANG_OPS | _LANG_CONN | _LANG_MOD | _LANG_UNARY | _LANG_NOUNS
    raw_tokens = []
    for w in stripped.split():
        if w in known_syms:
            raw_tokens.append(w)
        else:
            w2 = w.strip(".,!?\u00bf\u00a1")
            if w2:
                raw_tokens.append(w2)
    if not raw_tokens:
        return line

    out = []
    prev_noun_idx = None  # index in `out` of the most-recently-emitted noun, so a MOD (which follows
                           # its noun in the ORIGINAL stream) can still prefix it in the OUTPUT, per SYM's
                           # "modifiers go immediately before" grammar rule
    for w in raw_tokens:
        if w in _LANG_MOD:
            if prev_noun_idx is not None:
                out[prev_noun_idx] = _SYM_MOD.get(w, w) + out[prev_noun_idx]
            continue
        if w in _LANG_OPS:
            prev_noun_idx = None
            out.append(_SYM_OP.get(w, w))  # identity for every OP except "\u222a" -- see module comment
            continue
        if w in _LANG_CONN or w in _LANG_UNARY:
            prev_noun_idx = None
            out.append(w)  # always verbatim, already the correct SYM character (see module comment)
            continue
        # NEW (fix, at explicit request -- "symbols only for everything, no nouns in any form"): anything
        # that isn't an OP/CONN/MOD/UNARY symbol is a NOUN -- same "else = noun" convention
        # symbolic_to_english itself already relies on (its own Pass 1 does
        # `_NOUN_GLOSS.get(w.upper(), w.lower())` in exactly this position; real generated tokens come out
        # lowercase, hence the .upper() below). Previously rendered as shape+id ("\u25a13" for NODE) --
        # even a bare number still names a SPECIFIC referent, which is still an identifier standing in
        # for a noun, just spelled with digits instead of letters. Now it's the bare shape glyph ALONE:
        # every entity is "\u25cb", every system is "\u25a1", every other concept is "\u25c7", with
        # nothing distinguishing one from another -- pure category, no reference at all.
        tag = w.upper()
        if prev_noun_idx is not None and prev_noun_idx == len(out) - 1:
            out.append("&")  # adjacent nouns, nothing between them -- SYM's own AND (see module comment)
        out.append(_SYM_NOUN_SHAPE.get(tag, "\u25c7"))
        prev_noun_idx = len(out) - 1
    return " ".join(out) + trailing

def geometric_generate(query_vec, bigram, unigram, rng, n_words=None,
                        entity_vec=None, entity_word=None, entity_weight=None, mind=None):
    """Word-by-word generation. NO slots, NO pre-written sentences. Each next
    word is sampled from a distribution that's the PRODUCT of two
    independently-sourced signals:
      token_prob  -- bigram transition probability given the previous word
                      (the learned/counted half -- see build_transition_counts).
                      Kept deliberately local/windowed (looks at exactly one
                      preceding word) because this is what enforces real local
                      grammar -- dropping it entirely reproduces the exact
                      word-salad problem this file already diagnosed once (see
                      the top-of-file docstring on v3, which is what happens
                      when nothing but semantic similarity drives word choice).
      sem_score   -- Gaussian-kernel similarity between the candidate word's
                      fixed embedding and the RUNNING CONTEXT VECTOR (see
                      below) -- the geometry half, no trained weights, same
                      embed_text space as everything else.
    Neither alone is enough: token_prob alone ignores what's being talked
    about; sem_score alone ignores sequence/grammar. sem_score is raised to
    SEMANTIC_BETA to control how strongly it re-ranks the (already
    grammatically-informed) token distribution, then both are multiplied and
    renormalized.

    RUNNING CONTEXT VECTOR (not a window): previously the semantic half was
    scored against query_vec, a single vector computed ONCE before generation
    started and never touched again -- so by the 8th word of a sentence, word
    choice had no memory of the 7 words already generated, only of whatever
    state existed before any of them were chosen. That's a context WINDOW of
    zero once generation is underway. Fixed here by accumulating every word
    actually emitted into `context_vec` as an unbounded, un-decayed running
    sum (normalized before each use) -- so the 8th word is scored against
    everything said in the sentence so far AND the original query, not just
    the original query alone. Nothing is ever dropped or aged out: this is
    intentionally NOT a sliding window of the last K words, it's the entire
    history treated as one growing vector, matching how entity_vec already
    persists ACROSS sentences -- context_vec is the same idea, but within one.

    entity_vec/entity_word (optional; pass mind.entity_vec/mind.entity_word --
    see DISCOURSE ENTITY on Mind): the persistent cross-sentence anchor. Two
    separate effects, mirroring the two things entity-grid coherence research
    actually tracks -- an entity's semantic continuity AND its literal
    surface recurrence across neighbouring sentences:
      (1) query_vec is blended toward entity_vec BEFORE word selection starts,
          so every word in THIS sentence is pulled toward the same topic the
          last sentence was pulled toward, not just tagged with it after the
          fact.
      (2) if entity_word itself is a legal bigram continuation at some step,
          it gets a direct multiplicative boost -- giving the generator a
          real, elevated chance of literally reusing the same noun a human
          writer would repeat or refer back to, not just something nearby in
          meaning.

    entity_weight (optional): overrides the module default ENTITY_BLEND_WEIGHT
    for this call. Lets a caller's own judgment (e.g. Mind.reason's
    'persistence' output -- see _generate_and_track) decide, per sentence,
    how hard to hold the current entity rather than using one fixed constant
    for every call everywhere."""
    weight = ENTITY_BLEND_WEIGHT if entity_weight is None else entity_weight
    if entity_vec is not None and np.linalg.norm(entity_vec) > EPS:
        query_vec = _blend(query_vec, 1 - weight, entity_vec, weight)
    if n_words is None:
        n_words = int(rng.integers(GEN_WORD_RANGE[0], GEN_WORD_RANGE[1] + 1))
    prev = "<s>"
    out = []
    _bigrams_used = set()            # NEW: per-sentence, reset each call -- bigrams already used in THIS
                                      # sentence, checked above to stop the loop from forming at the source
    context_sum = query_vec.copy()   # NEW: unbounded running accumulation, starts at the original query
    context_n = 1.0                  # NEW: running count for the mean -- every word emitted adds equally,
                                      # nothing ever ages out or gets forgotten as the sentence grows
    _logprob_sum = 0.0               # FIX (word-salad diagnosis): accumulates log(token_prob[chosen]) --
    _logprob_n = 0                   # the model's own RAW next-word probability for the word actually
                                      # picked, BEFORE semantic reranking -- across every word emitted, so
                                      # the mean at the end is a real per-candidate fluency signal (see
                                      # _LAST_GEN_FLUENCY above for why this didn't exist before / what it
                                      # feeds into in _select_best)
    lang_grammar = _LangGrammarState()  # NEW: grammar-constrained decoding state for Lang -- see class
                                         # docstring above
    saw_op = False  # NEW (fix, at explicit request -- "make it speak English" pt.2): tracks whether a
                     # real relational operator (the thing symbolic_to_english renders as the VERB --
                     # "leads to"/"relates to"/etc.) has been emitted yet in this line. Without this, a
                     # structurally-legal-per-FSM line could be nothing but a run of NOUN/MOD tokens
                     # (</s> is allowed any time can_close_sentence() is true, which only checks the LAST
                     # token, not whether an OP ever appeared at all) -- which decodes to a bare noun list
                     # with no verb, e.g. "Predictability (fluctuating), language (moderate), vector...".
                     # Gating </s> on this the same way GEN_WORD_RANGE[0]/can_close_sentence already do
                     # forces every generated line to contain a subject-verb-object shape once decoded.
    for _ in range(n_words):
        context_vec = normalize(context_sum / context_n)  # NEW: the "one big thing" -- whole history so far
        # WAS: candidates = bigram.get(prev) or unigram -- a lookup keyed on exactly the ONE previous
        # word. Now sourced from the trained transformer's own forward pass over the WHOLE sequence
        # generated in this sentence so far (out), not just `prev` -- see TRANSFORMER WORD MODEL section
        # above. bigram/unigram params are no longer read here (kept in the signature for every existing
        # caller that still passes concept-biased tables -- see _concept_biased_tables, itself now a
        # no-op for a different, earlier reason).
        # NEW (per-token grounding): when a live mind is passed, EVERY token reads the Mind's CURRENT
        # winning-coalition content (recomputed fresh this iteration, not the value from generation
        # start) as a second grounding channel, and immediately writes back into M_t afterward -- see
        # live_workspace_snapshot/apply_token_bias on Mind and workspace_proj/token_bias_from_hidden on
        # TinyTransformerLM. Falls back to the original one-vector-in, once-per-sentence-out behavior
        # when mind is None (every existing call site that doesn't pass mind is unaffected).
        # NEW (grammar-checker transformer, at explicit request): word probabilities now come from
        # _dual_transformer_word_step, not a bare _tf_next_word_probs call -- see that function's
        # docstring for the main-model/grammar-checker feedback loop this runs on EVERY word, sharing the
        # SAME context_vec/workspace_vec/probability distribution/M_t self-model between both transformers
        # (see 'ALL TRANSFORMERS SHARE STATE' docstring above GrammarCheckerLM). ws_vec stays None (and
        # the loop degrades gracefully) when no live mind was passed, same as before.
        ws_vec = mind.live_workspace_snapshot() if mind is not None else None
        cand_words, token_prob, tok_hidden, grammar_bias = _dual_transformer_word_step(
            out, context_vec, workspace_vec=ws_vec)
        sem = np.array([_word_semantic_score(w, context_vec) for w in cand_words]) ** SEMANTIC_BETA
        if entity_word is not None and entity_word in cand_words:
            sem = sem.copy()
            sem[cand_words.index(entity_word)] *= ENTITY_LEXICAL_BOOST
        combined = token_prob * sem
        # NEW: suppress the exact self-reinforcing loop that an unbounded, un-decayed context_vec makes
        # likely -- a word/bigram that's already appeared can keep looking like the single best semantic
        # match precisely BECAUSE it's already baked into context_vec, which is what produced lines like
        # "un poco a un poco a un poco". Zeroed out here, at the source, rather than only judged after the
        # fact -- filtering finished sentences can't fix a batch where every draw loops the same way, since
        # they're all pulled toward the same self-reinforcing attractor for the same structural reason.
        used_counts = Counter(out)
        for i, w in enumerate(cand_words):
            if used_counts.get(w, 0) >= MAX_WORD_REPEATS or (prev, w) in _bigrams_used:
                combined[i] = 0.0
        # NEW (grammar-constrained decoding for Lang): mask any candidate that would make the sentence
        # structurally invalid from here -- see _LangGrammarState's class docstring.
        for i, w in enumerate(cand_words):
            if combined[i] > 0.0 and w != "</s>" and not lang_grammar.allows(w):
                combined[i] = 0.0
        # NEW: FIX -- </s> had no floor: GEN_WORD_RANGE[0]=12 only ever bounded the loop's maximum
        # iteration count (`for _ in range(n_words)`), nothing stopped </s> itself from being sampled
        # as the very first or second token if the transformer assigned it high probability there,
        # which -- verified directly by tracing real batches -- it does on the large majority of draws
        # (that's the actual mechanism behind the 'Siento.'/'...' collapse, not corpus sparsity or
        # reranking: most batches never contained a real long candidate to rerank in the first place).
        # Zeroed out here the same way an already-used word/bigram is zeroed two lines up, so </s>
        # genuinely cannot end the sentence before GEN_WORD_RANGE[0] words have been emitted.
        if len(out) < GEN_WORD_RANGE[0] and "</s>" in cand_words:
            combined[cand_words.index("</s>")] = 0.0
        # NEW (grammar-constrained decoding for Lang): never let the sentence end on a dangling
        # article/preposition/connector -- same role can_close_sentence() plays for unbalanced parens
        # on other FSM-guarded paths.
        if "</s>" in cand_words and not lang_grammar.can_close_sentence():
            combined[cand_words.index("</s>")] = 0.0
        # NEW (fix, at explicit request -- "make it speak English" pt.2): never end the line before it
        # contains at least one operator -- see saw_op docstring above.
        if "</s>" in cand_words and not saw_op:
            combined[cand_words.index("</s>")] = 0.0
        total = combined.sum()
        # FIX (NaN-weight collapse -- root cause of the "Probabilities contain NaN" crash): same issue as
        # the fallback in _dual_transformer_word_step above -- `total > EPS` is False whenever total is
        # NaN, so this used to fall through to `token_prob` unconditionally, including in the case where
        # token_prob itself is already NaN (poisoned transformer weights -- see the clip_norm fix on the
        # scratch _tf_train_epochs call above, which is the actual upstream cause). Falling back further,
        # to a uniform distribution over the still-legal (combined[i] > 0 in token_prob terms is not
        # reliable once NaN is involved, so just use every candidate) words, guarantees rng.choice always
        # gets a valid, finite, normalized p regardless of what upstream produced.
        if total > EPS and np.isfinite(total):
            probs = combined / total
        elif np.isfinite(token_prob).all() and token_prob.sum() > EPS:
            probs = token_prob / token_prob.sum()
        else:
            probs = np.full(len(cand_words), 1.0 / len(cand_words))
        chosen_idx = int(rng.choice(len(cand_words), p=probs))
        nxt = cand_words[chosen_idx]
        if nxt == "</s>":
            break
        # FIX (word-salad diagnosis): record the RAW model probability (pre-semantic-rerank) of the word
        # actually chosen -- this is the fluency signal _LAST_GEN_FLUENCY carries out to _select_best.
        # token_prob can legitimately be 0 at a masked-out index (grammar-FSM/repeat/math-leak zeroing
        # only ever touches `combined`, never `token_prob` itself, so this is still the model's real,
        # unmasked estimate) -- floored at EPS so a single near-zero token can't send the running mean to
        # -inf and silently disqualify an otherwise-fine candidate.
        _logprob_sum += math.log(max(float(token_prob[chosen_idx]), EPS))
        _logprob_n += 1
        out.append(nxt)
        lang_grammar.advance(nxt)  # NEW: update the Lang FSM with the token actually chosen
        if lang_grammar.last == lang_grammar.OP:  # NEW: see saw_op docstring above
            saw_op = True
        _bigrams_used.add((prev, nxt))
        prev = nxt
        if mind is not None:  # NEW (per-token grounding): write-back fires HERE, every token, not just
            tok_bias = _TF_MODEL.token_bias_from_hidden(tok_hidden)  # once at the end of the sentence
            mind.apply_token_bias(tok_bias)
            if grammar_bias is not None:  # NEW (grammar-checker transformer): the checker's OWN
                mind.apply_token_bias(grammar_bias)  # pooled-prefix write, into the SAME shared M_t --
                                                       # see 'shared self-model' on GrammarCheckerLM
        w_vec = VOCAB_EMBED.get(nxt)     # NEW: fold the word just emitted into the running context --
        if w_vec is None:                # this is what makes context_vec grow with the sentence instead
            w_vec = embed_text(nxt, _IDF, _DEFAULT_IDF)  # of staying frozen at its pre-generation value
            VOCAB_EMBED[nxt] = w_vec
        context_sum = context_sum + w_vec
        context_n += 1.0
    if not out:
        out = ["..."]
    else:
        # NEW: two-way grounding, write-back half (see TinyTransformerLM's class docstring). One extra
        # forward pass over the COMPLETED sentence, conditioned on the same running context_vec it was
        # generated under, pools into a bias_M-shaped nudge and stashes it for run()'s NEXT mind.step()
        # call to actually apply -- see Mind.step's bias_M parameter, which existed in this file already
        # but had never once been called with real content until this.
        global _LAST_TF_MIND_BIAS
        final_ids = [_TF_WORD2ID.get(w, _TF_WORD2ID[TF_UNK]) for w in ([TF_BOS] + out)][-TF_MAX_LEN:]
        final_sv = torch.tensor(context_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        _LAST_TF_MIND_BIAS = _TF_MODEL.mind_bias(
            torch.tensor([final_ids], dtype=torch.long, device=DEVICE), final_sv)
    sentence = " ".join(out)
    sentence = sentence[0].upper() + sentence[1:]
    if not sentence.endswith((".", "?", "!")):
        sentence += "."
    global _LAST_GEN_FLUENCY  # FIX (word-salad diagnosis): mean per-token log-prob for THIS candidate --
    _LAST_GEN_FLUENCY = (_logprob_sum / _logprob_n) if _logprob_n else None  # see _select_best, which
                        # reads this right after each geometric_generate call, same pattern as
                        # _LAST_TF_MIND_BIAS above. None (not 0.0) when out was empty/"...", so the caller
                        # can tell "no real fluency signal" apart from "measured and genuinely very low".
    return sentence

# ============================================ BIDIRECTIONAL GROUNDING (wants + experience)
# Everything above is grounding in ONE direction only: prompt text -> nearest
# hand-authored anchor, fixed at import time and never touched again. This
# section adds the other direction, in two real, persisted, learned parts:
#
#   experience -- every time a concept or mood is actually spoken in response
#     to a real prompt, that prompt's own embedded vector nudges THAT name's
#     anchor via EMA (ANCHOR_DRIFT), and the live axis-state at the moment it
#     was spoken nudges that name's AXIS_PROFILE via EMA. So each concept's
#     and mood's routing location and "typical accompanying state" both drift
#     from real usage over time -- this Mind's own history of what it's
#     actually been asked and how it actually felt when answering -- not just
#     the fixed seed phrases/keyword lists they started from.
#
#   want -- want_ema (Mind.want_ema; trained via Hebbian credit assignment in
#     learn_desire from this Mind's own reward trajectory) is compared, via
#     dot product, against each name's AXIS_PROFILE. A concept/mood whose
#     typical accompanying state lines up with what this Mind has learned is
#     worth pursuing gets a routing boost independent of whatever the prompt's
#     TEXT says -- so the final score blends "what did you ask" (text) with
#     "what do I currently want to be talking about" (learned preference).
#     Early on, with no profile yet, this term is neutral (0.5) and the score
#     reduces to pure text grounding, same as before.
#
# Both ANCHOR_DRIFT and AXIS_PROFILE are plain dicts, persisted to the same
# sqlite DB as mind_state (see PERSISTENCE section) -- loaded once at the
# start of run() and saved back at the end, so this learning survives across
# sessions the same way want_ema itself already does.
ANCHOR_DRIFT_KEY = "anchor_drift"
AXIS_PROFILE_KEY = "axis_profile"
ANCHOR_EXPERIENCE_RATE = 0.15   # EMA rate: how fast a name's text-anchor drifts toward prompts it actually answered
AXIS_PROFILE_RATE = 0.05        # EMA rate: how fast a name's typical-state profile updates
WANT_ALIGN_WEIGHT = 0.25        # how much want-alignment contributes to the combined grounding score

# basin deliberately excluded from the shared axis set below -- it's an alarm
# state (the looping override), not something the Mind ever "wants" more of.
_SHARED_AXES = list(AXIS_NAMES)

def load_learned_grounding(conn):
    return load_blob(conn, ANCHOR_DRIFT_KEY) or {}, load_blob(conn, AXIS_PROFILE_KEY) or {}

def save_learned_grounding(conn, anchor_drift, axis_profile):
    save_blob(conn, ANCHOR_DRIFT_KEY, anchor_drift)
    save_blob(conn, AXIS_PROFILE_KEY, axis_profile)

def effective_anchor(name, base_vec, anchor_drift):
    """Base anchor (from hand-written seed phrases/keywords) blended with this
    name's real-usage drift, if any experience has been recorded for it yet."""
    drift = anchor_drift.get(name)
    if drift is None:
        return base_vec
    v = (1 - ANCHOR_EXPERIENCE_RATE) * base_vec + ANCHOR_EXPERIENCE_RATE * drift
    n = np.linalg.norm(v)
    return v / n if n > EPS else v

def want_align_score(name, axis_profile, want_ema):
    """Dot product between this name's learned typical-state profile and the
    Mind's own trained want_ema, squashed to [0,1] via sigmoid. Neutral (0.5)
    -- not a boost, not a penalty -- if this name has no profile yet."""
    profile = axis_profile.get(name)
    if profile is None:
        return 0.5
    dot = sum(profile.get(a, 0.0) * want_ema.get(a, 0.0) for a in _SHARED_AXES)
    return float(sigmoid(6 * dot))

def record_experience(name, prompt, norm, anchor_drift, axis_profile):
    """Called once per real response actually spoken -- nudges this name's
    anchor toward the prompt that led here and its axis profile toward the
    live state at the moment of speaking. Both are genuine EMA updates over
    actual usage, mutating the dicts in place (caller persists them)."""
    v = embed_text(prompt, _IDF, _DEFAULT_IDF)
    prev = anchor_drift.get(name)
    if prev is None:
        anchor_drift[name] = v
    else:
        merged = (1 - ANCHOR_EXPERIENCE_RATE) * prev + ANCHOR_EXPERIENCE_RATE * v
        n = np.linalg.norm(merged)
        anchor_drift[name] = merged / n if n > EPS else merged
    prof = axis_profile.get(name, {a: 0.5 for a in _SHARED_AXES})
    for a in _SHARED_AXES:
        prof[a] = (1 - AXIS_PROFILE_RATE) * prof[a] + AXIS_PROFILE_RATE * norm.get(a, 0.5)
    axis_profile[name] = prof

def semantic_route(prompt, mind, anchor_drift, axis_profile, topic_anchors=None):
    """Same Gaussian-kernel text grounding as before, now blended with
    want_align_score -- text says what was asked, want-alignment says whether
    this Mind's own trained preferences lean toward that answer regardless.
    topic_anchors (see discovered_topic_anchors below) adds a THIRD candidate
    set: real recurring topics the corpus has discovered, each geometrically
    pre-mapped to whichever existing concept/mood they're nearest -- so a
    prompt matching a topic the system has genuinely seen repeat can route
    correctly even with no hand-written seed phrase for it. Returns (kind,
    name, combined_score, text_score, matched_topic) -- matched_topic is the
    discovered topic's word list if THAT'S what won, else None (routed via
    the hand-authored bank as before)."""
    v = embed_text(prompt, _IDF, _DEFAULT_IDF)
    # NEW (fix, at explicit request -- "parse what was asked more richly"): the Li & Roth wh-type
    # classifier (detect_expected_answer_type/ANSWER_TYPE_WORDS) already existed but only ever reranked
    # already-generated candidates, never influenced which CONCEPT got routed to in the first place --
    # so "why do you exist" and "what are you" could both land on whichever concept happened to be
    # nearest in raw embedding distance, with no credit for one of them actually being a why-question.
    # Computed once here and folded in as a small per-concept bonus below (see _CONCEPT_ANSWER_TYPE_HINT
    # / ROUTE_TYPE_BONUS_WEIGHT) -- a tiebreak nudge, not a hard override.
    expected_type = detect_expected_answer_type(prompt)
    # NEW (agency loop): while a goal is active, weight candidates toward whichever concept/mood has
    # historically co-occurred with THAT axis being strong (axis_profile[name][goal_axis] -- the same
    # learned per-name axis profile want_align_score already reads, just indexed by the currently-
    # pursued axis instead of dotted against want_ema as a whole). Neutral (0.5, no effect) when no
    # goal is active or a name has no profile yet -- same "don't invent a boost with no evidence"
    # convention want_align_score itself follows.
    goal_axis = getattr(mind, "goal_axis", None)
    def _goal_align(name):
        if goal_axis is None:
            return 0.5
        profile = axis_profile.get(name)
        return float(profile.get(goal_axis, 0.5)) if profile is not None else 0.5
    def _type_bonus(kind_label, name):
        if expected_type is None or kind_label != "concept":
            return 0.0
        return ROUTE_TYPE_BONUS_WEIGHT if _CONCEPT_ANSWER_TYPE_HINT.get(name) == expected_type else 0.0
    text_w = 1 - WANT_ALIGN_WEIGHT - GOAL_ALIGN_WEIGHT
    best_kind, best_name, best_score, best_text, best_topic = None, None, 0.0, 0.0, None
    for kind_label, bank in (("concept", CONCEPT_ANCHORS), ("mood", MOOD_ANCHORS)):
        for name, base_vec in bank.items():
            anchor_vec = effective_anchor(name, base_vec, anchor_drift)
            text_score = float(np.exp(-GROUND_LAM * np.sum((v - anchor_vec) ** 2)))
            want_score = want_align_score(name, axis_profile, mind.want_ema)
            combined = (text_w * text_score + WANT_ALIGN_WEIGHT * want_score
                        + GOAL_ALIGN_WEIGHT * _goal_align(name) + _type_bonus(kind_label, name))
            if combined > best_score:
                best_kind, best_name, best_score, best_text, best_topic = kind_label, name, combined, text_score, None
    for kind_label, name, anchor_vec, topic_words in (topic_anchors or []):
        text_score = float(np.exp(-GROUND_LAM * np.sum((v - anchor_vec) ** 2)))
        want_score = want_align_score(name, axis_profile, mind.want_ema)
        combined = (text_w * text_score + WANT_ALIGN_WEIGHT * want_score
                    + GOAL_ALIGN_WEIGHT * _goal_align(name) + _type_bonus(kind_label, name))
        if combined > best_score:
            best_kind, best_name, best_score, best_text, best_topic = kind_label, name, combined, text_score, topic_words
    if best_score < CONCEPT_GROUNDING_THRESHOLD:
        return None, None, best_score, best_text, None
    return best_kind, best_name, best_score, best_text, best_topic

# NEW (fix, at explicit request -- "make it have real parsing"): semantic_route above still embeds and
# routes the WHOLE prompt as one bag of words -- see split_clauses's docstring for why that silently
# drops one half of a compound ask. This wraps it: split the prompt into real clauses first, route each
# clause independently through the exact same semantic_route (so single-clause prompts -- the common
# case -- get IDENTICAL behavior, byte for byte, to calling semantic_route directly), then only if two OR
# MORE clauses genuinely routed to two OR MORE DIFFERENT concepts does anything different happen: their
# anchor vectors get blended into one combined topic vector, so the generation that follows stays on-
# topic for BOTH things actually asked about, not just whichever clause the old single-pass average
# happened to lean toward.
MULTI_CLAUSE_BLEND_WEIGHT = 0.5  # equal weight per distinct concept clause -- no principled reason for
                                  # this system to consider one half of a compound question more
                                  # important than the other just because it was asked first

def semantic_route_multi(prompt, mind, anchor_drift, axis_profile, topic_anchors=None):
    """Real per-clause parsing + routing. Returns (kind, name, score, text_score, matched_topic,
    blend_info) -- the first five have EXACTLY semantic_route's own shape (existing callers work
    unchanged if they ignore the 6th value); blend_info is None for an ordinary single-topic route, or
    (blended_topic_vec, [name, ...]) when 2+ distinct concepts were each independently confident about a
    DIFFERENT clause of the same prompt. `name`/`kind` still identify the single highest-scoring concept
    among those blended (so existing single-concept ANSWER dispatch -- CONCEPT_BANK[name]['answer'] --
    still has something concrete to call) -- blend_info's vector is what should additionally steer
    prompt_topic_vec for generation, so word choice afterward leans toward every topic that was actually
    asked about, not only the one dispatch ended up answering from."""
    clauses = split_clauses(prompt)
    if len(clauses) <= 1:
        kind, name, score, text_score, matched_topic = semantic_route(
            prompt, mind, anchor_drift, axis_profile, topic_anchors)
        return kind, name, score, text_score, matched_topic, None
    routed = [(c,) + semantic_route(c, mind, anchor_drift, axis_profile, topic_anchors) for c in clauses]
    confident = [r for r in routed if r[1] is not None]  # r = (clause_text, kind, name, score, text, topic)
    if not confident:
        c0, kind, name, score, text_score, matched_topic = routed[0]
        return kind, name, score, text_score, matched_topic, None
    distinct = {}  # (kind, name) -> best (clause_text, score) seen for it, across clauses
    for c, kind, name, score, text_score, matched_topic in confident:
        key = (kind, name)
        if key not in distinct or score > distinct[key][1]:
            distinct[key] = (c, score)
    if len(distinct) <= 1:
        best = max(confident, key=lambda r: r[3])
        return best[1], best[2], best[3], best[4], best[5], None
    # 2+ genuinely different concepts, each independently confident about its own clause -- blend
    concept_keys = [k for k in distinct if k[0] == "concept"]
    blend_keys = concept_keys if len(concept_keys) >= 2 else list(distinct.keys())
    # NEW (fix): blend the CLAUSE's own embedding, not a CONCEPT_ANCHORS/MOOD_ANCHORS re-lookup by name --
    # a name routed via a discovered topic or an autonomously-created concept lives in neither dict (its
    # vector only ever existed inside the topic_anchors tuple passed in), so re-deriving by name would
    # KeyError for exactly the routes this system is proudest of (see AUTONOMOUS CONCEPT CREATION below).
    # The clause's own embedding is also simply the more honest thing to blend: it's what THIS prompt
    # actually said about that topic, not the topic's generic hand-authored anchor.
    vecs = [embed_text(distinct[k][0], _IDF, _DEFAULT_IDF) for k in blend_keys]
    blended_vec = vecs[0]
    for v in vecs[1:]:
        blended_vec = _blend(blended_vec, 1 - MULTI_CLAUSE_BLEND_WEIGHT, v, MULTI_CLAUSE_BLEND_WEIGHT)
    winner_key = max(blend_keys, key=lambda k: distinct[k][1])
    winner_kind, winner_name = winner_key
    best_score = max(distinct[k][1] for k in blend_keys)
    names = [k[1] for k in blend_keys]
    return winner_kind, winner_name, best_score, best_score, None, (blended_vec, names)

# ============================================ CORPUS GROWTH (real, over time)
# Everything above is bootstrapped from a fixed 20-document, hand-authored
# vocabulary -- too sparse for co-occurrence statistics to mean anything (see
# the PMI experiment: 297 words -> 7 blobs, one of them 140 words wide,
# genuinely wrong). That sparsity is a property of THAT corpus, not of
# co-occurrence statistics in general -- the same math becomes trustworthy
# once it's run over enough real, independent contexts. The only source of
# more real, independent Spanish context this system has is the prompts it
# actually gets asked over time. So: every prompt this system ever receives
# is logged as one more document, persisted in the same DB as the Mind's own
# state, and topic discovery runs over that GROWING real corpus instead of
# the static seed set. Gated by min_doc_freq/min_cooc so it stays honest at
# small sample sizes instead of repeating the same blob failure at a larger
# scale.
#
# Hard limit, stated once here rather than left implicit: discovery (below)
# can find that a set of words keeps appearing together, and routing
# (discovered_topic_anchors, near the semantic-grounding section) can now
# automatically send a future match on that same topic to whichever EXISTING
# answer it's nearest to. What neither one can do is write a NEW Spanish
# sentence for a topic that has no existing near neighbor -- turning "these
# words cluster, and nothing we have answers them" into "here is what to say
# about them" is authorship, and stays a human step.
CORPUS_KEY = "prompt_corpus"
LINE_USE_COUNT_KEY = "line_use_count"  # persists _LINE_USE_COUNT (see _overuse_penalty) across separate
                                       # invocations, the same way mind_state/prompt_corpus already do

def load_corpus(conn):
    return load_blob(conn, CORPUS_KEY) or []

def update_corpus(conn, prompt):
    """Append this prompt's content words as one more document in the growing
    corpus, and persist it. Every real prompt counts, matched or not --
    unmatched ones are exactly the ones most likely to represent a topic this
    system doesn't have yet."""
    corpus = load_corpus(conn)
    words = _content_words(prompt)
    if words:
        corpus.append(words)
    save_blob(conn, CORPUS_KEY, corpus)
    return corpus

# Separate from CORPUS_KEY above: that one strips stopwords (needed for PMI
# topic-discovery, where grammatical glue words are noise). This one keeps
# them, tokenized via _tokenize_natural instead of _content_words, because
# fluent generation NEEDS the glue words ("el", "la", "de") that topic
# discovery deliberately throws away -- see build_transition_counts.
RAW_CORPUS_KEY = "raw_prompt_corpus"

def load_raw_corpus(conn):
    return load_blob(conn, RAW_CORPUS_KEY) or []

def update_raw_corpus(conn, prompt):
    # NEW (bugfix, defense-in-depth alongside the __main__ argv guard): reject prompt text that looks
    # like CLI/kernel noise rather than something a human actually typed -- a bare flag ("-f") or a
    # filesystem path (e.g. a Jupyter kernel connection file) should never be persisted into the corpus
    # or trained into the bigram/transformer tables, since once in there it resurfaces in generation
    # forever after. This is a second, independent layer -- it also protects any future call path into
    # run()/update_raw_corpus that doesn't go through the __main__ argv parsing at all (e.g. calling
    # run() directly from a notebook cell).
    if prompt is None or prompt.strip().startswith("-") or "/" in prompt or "\\" in prompt:
        return load_raw_corpus(conn)
    corpus = load_raw_corpus(conn)
    toks = _tokenize_natural(prompt)
    if toks:
        corpus.append(toks)
    save_blob(conn, RAW_CORPUS_KEY, corpus)
    return corpus

def discover_topics(corpus, min_doc_freq=4, min_cooc=3, pmi_threshold=1.2):
    """Same PMI + union-find mechanism as the earlier experiment, but with two
    support gates the earlier version deliberately didn't have, so it doesn't
    just reproduce the same blob failure at a bigger size:
      - min_doc_freq: a word must have shown up in at least this many SEPARATE
        real prompts before it's eligible at all. One-off words stay excluded
        rather than seeding a spurious cluster off a single co-occurrence.
      - min_cooc: a word PAIR must have actually co-occurred at least this
        many times (not once) before its PMI is trusted, since PMI on a
        single observation is just noise wearing a formula.
    Returns a list of (topic_words, doc_support) -- doc_support is how many
    real prompts contributed to that topic, so you can see how much evidence
    actually backs it before deciding whether to hand-author an answer for it."""
    if len(corpus) < min_doc_freq:
        return []  # not enough real usage yet for ANY word to clear the gate
    doc_freq = defaultdict(int)
    pair_freq = defaultdict(int)
    for doc in corpus:
        uniq = set(doc)
        for w in uniq:
            doc_freq[w] += 1
        for w1 in uniq:
            for w2 in uniq:
                if w1 < w2:
                    pair_freq[(w1, w2)] += 1
    n_docs = len(corpus)
    eligible = {w for w, c in doc_freq.items() if c >= min_doc_freq}
    parent = {w: w for w in eligible}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for (w1, w2), c in pair_freq.items():
        if c < min_cooc or w1 not in eligible or w2 not in eligible:
            continue
        pmi = math.log((c / n_docs) / ((doc_freq[w1] / n_docs) * (doc_freq[w2] / n_docs)))
        if pmi > pmi_threshold:
            union(w1, w2)
    groups = defaultdict(list)
    for w in eligible:
        groups[find(w)].append(w)
    topics = []
    for words in groups.values():
        if len(words) > 1:
            support = sum(1 for doc in corpus if set(words) & set(doc))
            topics.append((sorted(words), support))
    return sorted(topics, key=lambda t: -t[1])

def nearest_known_concept(topic_words):
    """Geometric search over what the system ALREADY has an answer for --
    locates a discovered topic in the same concept space and returns the
    single closest existing concept/mood/self-authored-concept, with its
    real distance-based score. Used for human-facing reporting AND, via
    discovered_topic_anchors / maybe_create_auto_concepts below, to decide
    whether a piece of real vocabulary already has a home (so it ENRICHES
    that home) or genuinely doesn't (so it earns a new one)."""
    v = concept_anchor(topic_words, _IDF, _DEFAULT_IDF)
    best_kind, best_name, best_score = None, None, -1.0
    for name, anchor_vec in CONCEPT_ANCHORS.items():
        score = float(np.exp(-GROUND_LAM * np.sum((v - anchor_vec) ** 2)))
        if score > best_score:
            best_kind, best_name, best_score = "concept", name, score
    for name, anchor_vec in MOOD_ANCHORS.items():
        score = float(np.exp(-GROUND_LAM * np.sum((v - anchor_vec) ** 2)))
        if score > best_score:
            best_kind, best_name, best_score = "mood", name, score
    for name, anchor_vec in _AUTO_CONCEPT_ANCHORS.items():  # NEW: self-authored concepts are real
        score = float(np.exp(-GROUND_LAM * np.sum((v - anchor_vec) ** 2)))  # semantic areas too, not
        if score > best_score:                                             # just hand-authored ones
            best_kind, best_name, best_score = "concept", name, score
    return best_kind, best_name, best_score

def discovered_topic_anchors(corpus):
    """Closes the loop discover_topics' own docstring explicitly left open
    ("it never auto-answers from a discovered topic"). Every topic that
    clears discover_topics' real-support gates (min_doc_freq/min_cooc -- not
    a single lucky co-occurrence) gets located in concept space via the same
    concept_anchor() geometry as every hand-written seed phrase, then
    permanently paired with whichever existing concept/mood it's nearest to
    (nearest_known_concept). That pairing becomes ONE MORE candidate anchor
    in semantic_route's search (see topic_anchors there) -- so a future
    prompt resembling a topic the corpus has genuinely seen recur gets routed
    correctly with no human editing a seed-phrase list by hand.
    The honesty limit stays exactly what it was: this can only ever route a
    recurring topic to an answer the system ALREADY has. It still cannot
    invent a new answer for a topic with no existing near neighbor -- that
    remains a human (hand-authoring) step, same as before."""
    anchors = []
    for words, support in discover_topics(corpus):
        kind, name, _ = nearest_known_concept(words)
        if kind is None:
            continue
        anchors.append((kind, name, concept_anchor(words, _IDF, _DEFAULT_IDF), words))
    return anchors

# ==================================== AUTONOMOUS CONCEPT CREATION
# discovered_topic_anchors() above still only ever routes a recurring topic to
# an answer this system ALREADY has -- its own docstring says so, and said so
# honestly, because writing a NEW Spanish answer for a topic with no existing
# near neighbor used to be authorship, and stayed a human step.
#
# This closes that loop, and does it off individual WORDS rather than
# discover_topics()'s co-occurring PAIRS. Reason: PMI (discover_topics'
# scoring) rewards a pair for co-occurring MORE than chance predicts -- and
# when this Mind is only ever asked about, say, "nostalgia" alongside
# "melancolia" and never sees either word any other way, chance already
# predicts they'll co-occur every time, so PMI on that pair sits near 0, not
# high. Empirically that meant a genuinely recurring, genuinely related pair
# needed a large, deliberately-diluted corpus (many UNRELATED prompts mixed
# in) before it cleared discover_topics' pmi_threshold at all -- exactly
# backwards from "requires fewer prompts." A single word's own doc_freq (how
# many SEPARATE real prompts said it, standalone) has no such pathology: it's
# real, repeated evidence the moment the same word comes up twice, full stop.
#
# discover_concept_words() below is that lighter gate. For each word it
# surfaces that isn't already vocabulary somewhere (_known_vocabulary --
# already living in a hand-authored seed_phrase/keyword list, or a
# previously self-authored one), nearest_known_concept() locates it in the
# same geometry every hand-written CONCEPT_BANK/CLUSTERS entry already lives
# in and checks whether some EXISTING concept, mood, OR previously
# self-authored concept already covers that ground well enough
# (AUTO_CONCEPT_MIN_SCORE):
#   - if yes: the word ENRICHES that existing area instead of spawning a
#     near-duplicate next to it -- folded straight into a self-authored
#     concept's own seed_phrases (recomputing its anchor), or nudged into a
#     hand-authored concept/mood's ANCHOR_DRIFT (the exact same
#     experience-shaping mechanism record_experience already uses for real
#     spoken exchanges -- see _enrich_existing_area / _nudge_anchor_drift)
#     without ever rewriting the hand-authored CONCEPT_BANK/CLUSTERS text
#     itself;
#   - if no: it's a genuine gap, so a brand-new concept gets minted, seeded
#     by this word plus whichever OTHER discovered words happened to share a
#     real prompt with it (no PMI/pair-gate needed for that part -- just
#     actual doc-mates, for a slightly richer anchor than one bare word), a
#     few candidate answer_seeds drawn from this system's own
#     geometric_generate() biased toward that anchor, and the whole thing
#     persisted to the same sqlite DB everything else survives across runs
#     in.
# Every autonomously-created concept still shares ONE answer function-factory
# (_make_auto_answer_fn) rather than a hand-written _concept_whatever --
# there's nothing to hand-write, since nothing authored these by hand.
AUTO_CONCEPT_KEY = "auto_concepts"
AUTO_CONCEPT_MIN_WORD_FREQ = 2  # WAS, in effect, ~4 co-occurring docs (discover_topics' own min_doc_freq)
                                # PLUS a pair clearing pmi_threshold=1.2, which -- see docstring above --
                                # empirically needed well over a dozen prompts, several of them irrelevant
                                # filler, before two genuinely-related repeated words scored above chance.
                                # A single word said standalone in just 2 separate real prompts is real,
                                # repeated evidence with no such dilution requirement -- this is the actual
                                # "requires fewer prompts" lever
AUTO_CONCEPT_MIN_SCORE = 0.02   # a discovered word only spawns a brand-new concept if its nearest EXISTING
                                # concept/mood/self-authored-concept scores below this -- i.e. genuinely has
                                # no good home yet. At or above this, the word ENRICHES that existing home
                                # instead (see _enrich_existing_area) rather than being skipped OR spawning a
                                # near-duplicate. NOTE the scale here is NOT the same as semantic_route's
                                # routing scores (those blend text_score with want_align_score and are judged
                                # against CONCEPT_GROUNDING_THRESHOLD=0.0) -- this is a bare single WORD's
                                # raw Gaussian-kernel similarity against a multi-phrase-averaged anchor, which
                                # empirically sits in the 0.003-0.03 range even for a clearly-related word
                                # (e.g. "caotica" -> fragmented scores ~0.022) and near 0 for an unrelated one
                                # (e.g. "oceano" -> looping scores ~0.003) -- 0.02 was picked by sampling
                                # roughly two dozen words spanning both cases, not derived analytically
AUTO_CONCEPT_SEED_COUNT = 4     # how many candidate answer_seeds get kept, mirroring the ~4 sentences each
                                # hand-authored CONCEPT_BANK entry has
AUTO_CONCEPT_MAX_PER_RUN = 3    # cap total actions (creates + enrichments combined) in a single invocation --
                                # raised from 1 now that most discovered words enrich rather than spawn a new
                                # concept outright, so hitting the cap on creates alone is rarer

def load_auto_concepts(conn):
    return load_blob(conn, AUTO_CONCEPT_KEY) or {}

def save_auto_concepts(conn, auto_concepts):
    save_blob(conn, AUTO_CONCEPT_KEY, auto_concepts)

AUTO_HANDLED_WORDS_KEY = "auto_concept_handled_words"

def load_auto_handled_words(conn):
    """Words maybe_create_auto_concepts has already acted on (created OR
    enriched with), across every past run -- separate from _known_vocabulary,
    which only covers words with somewhere to LIVE (a seed_phrase/keyword).
    A word that ENRICHED a hand-authored concept/mood has nowhere new to
    live -- CONCEPT_BANK/CLUSTERS text is deliberately never touched (see
    _enrich_existing_area) -- so without this separate record, the exact
    same word clearing its doc_freq gate would get re-discovered and
    re-enrich the same anchor_drift entry again on every single future run,
    forever, for no further benefit, while also eating into that run's
    AUTO_CONCEPT_MAX_PER_RUN budget."""
    return set(load_blob(conn, AUTO_HANDLED_WORDS_KEY) or [])

def save_auto_handled_words(conn, handled_words):
    save_blob(conn, AUTO_HANDLED_WORDS_KEY, sorted(handled_words))

def _auto_concept_name(words):
    """Deterministic from the concept's own seed words, so rediscovering the
    same word (or word+companions) later, or on a fresh run, resolves back
    to the SAME concept instead of minting a near-duplicate every time."""
    return "auto_" + "_".join(words[:3])

# name -> dict(seed_phrases, answer_seeds, anchor, answer=_concept_auto_answer) -- mirrors CONCEPT_BANK's
# shape exactly, so every place that already reads CONCEPT_BANK-shaped dicts (_concept_biased_tables, the
# answer dispatch in run()) can be taught to also check this one instead of needing parallel special-casing.
# Populated fresh each run from the persisted AUTO_CONCEPT_KEY blob -- see _load_auto_concepts_into_runtime.
_AUTO_CONCEPT_BANK = {}
_AUTO_CONCEPT_ANCHORS = {}          # name -> anchor vector, mirrors CONCEPT_ANCHORS
_AUTO_CONCEPT_ANSWER_TOKENS = {}    # name -> tokenized answer_seeds, mirrors _CONCEPT_ANSWER_TOKENS

def _make_auto_answer_fn(concept_name):
    """Returns a closure over ONE fixed concept_name with the exact same call
    signature every hand-authored _concept_* function already has
    (mind, state, norm, rng, topic_vec=None, prompt_text=None) -- so callers
    (the dispatch in run(), get_concept_answer_fn) never need to know or care
    whether the function they got back was hand-written or self-authored.
    Every autonomously-created concept gets its OWN closure here, but all of
    them share the same underlying logic (blend this concept's own anchor
    with the live qualia vector, generate through the shared choke point) --
    only the captured name differs."""
    def answer(mind, state, norm, rng, topic_vec=None, prompt_text=None):
        qvec, _ = qualia_vector(mind, state, norm)
        anchor = _AUTO_CONCEPT_ANCHORS.get(concept_name)
        query = _blend(anchor, 0.8, qvec, 0.2) if anchor is not None else qvec
        text, _, _ = _generate_and_track(mind, query, rng, topic_vec=topic_vec, prompt_text=prompt_text,
                                          state_vec=qvec, concept_name=concept_name)
        return text
    return answer

def _load_auto_concepts_into_runtime(auto_concepts):
    """Rebuilds the three module-level auto-concept registries above from
    whatever's persisted, once at the start of each run -- the same idea as
    CONCEPT_ANCHORS/_CONCEPT_ANSWER_TOKENS being built once from CONCEPT_BANK
    at import time, just redone per run since, unlike CONCEPT_BANK, this bank
    can genuinely grow between runs."""
    _AUTO_CONCEPT_BANK.clear()
    _AUTO_CONCEPT_ANCHORS.clear()
    _AUTO_CONCEPT_ANSWER_TOKENS.clear()
    for name, c in auto_concepts.items():
        _AUTO_CONCEPT_BANK[name] = dict(seed_phrases=c["seed_phrases"], answer_seeds=c["answer_seeds"],
                                         answer=_make_auto_answer_fn(name))
        _AUTO_CONCEPT_ANCHORS[name] = np.array(c["anchor"])
        _AUTO_CONCEPT_ANSWER_TOKENS[name] = [_tokenize_natural(s) for s in c["answer_seeds"]]

def get_concept_answer_fn(concept_name):
    """Single lookup point for 'what answers this concept', checking the
    hand-authored bank first and the self-authored one second, so callers
    don't need to know which kind a routed name turned out to be."""
    if concept_name in CONCEPT_BANK:
        return CONCEPT_BANK[concept_name]["answer"]
    return _AUTO_CONCEPT_BANK[concept_name]["answer"]

def discover_concept_words(corpus, min_doc_freq=AUTO_CONCEPT_MIN_WORD_FREQ):
    """Lighter-weight sibling of discover_topics() above -- see the module
    docstring on why a co-occurring PAIR is the wrong unit of evidence here.
    Counts each individual content word's own doc_freq (how many SEPARATE
    real prompts used it) with no dependence on a second word riding along.
    Returns [(word, doc_freq), ...] sorted by doc_freq descending."""
    if not corpus:
        return []
    doc_freq = defaultdict(int)
    for doc in corpus:
        for w in set(doc):
            doc_freq[w] += 1
    words = [(w, c) for w, c in doc_freq.items() if c >= min_doc_freq]
    return sorted(words, key=lambda wc: -wc[1])

def _known_vocabulary():
    """Every word already living in SOME semantic area's own routing text --
    hand-authored concept seed_phrases, hand-authored mood keywords, AND
    previously self-authored concept seed_phrases alike. A word already in
    here already has somewhere to belong; only words outside this set are
    genuinely new ground worth acting on."""
    vocab = set()
    for c in CONCEPT_BANK.values():
        for p in c["seed_phrases"]:
            vocab.update(_tokenize_natural(p))
    for c in CLUSTERS.values():
        for kw in c["keywords"]:
            vocab.update(_tokenize_natural(kw))
    for c in _AUTO_CONCEPT_BANK.values():
        for p in c["seed_phrases"]:
            vocab.update(_tokenize_natural(p))
    return vocab

def _enrich_existing_area(name, word, auto_concepts, anchor_drift):
    """Folds a newly-discovered word into whichever existing semantic area
    it's already closest to, instead of spawning a near-duplicate concept
    beside one that already covers this ground. Two cases:
      - a previously self-authored concept: that structure is this system's
        own, so the word is appended straight into its persisted
        seed_phrases and its anchor is recomputed from them -- a real,
        permanent enrichment of what THAT concept knows to say.
      - a hand-authored concept or mood: CONCEPT_BANK/CLUSTERS text is never
        touched (stays hand-authored, unchanged, exactly as always) --
        instead this nudges that name's ANCHOR_DRIFT toward the word's own
        embedding, the SAME EMA mechanism record_experience() already uses
        for real spoken exchanges (see BIDIRECTIONAL GROUNDING), just
        triggered here by vocabulary evidence alone rather than a full
        routed reply."""
    if name in auto_concepts:
        c = auto_concepts[name]
        if word not in c["seed_phrases"]:
            c["seed_phrases"] = c["seed_phrases"] + [word]
            c["anchor"] = concept_anchor(c["seed_phrases"], _IDF, _DEFAULT_IDF).tolist()
        return
    v = embed_text(word, _IDF, _DEFAULT_IDF)
    prev = anchor_drift.get(name)
    if prev is None:
        anchor_drift[name] = v
    else:
        merged = (1 - ANCHOR_EXPERIENCE_RATE) * prev + ANCHOR_EXPERIENCE_RATE * v
        n = np.linalg.norm(merged)
        anchor_drift[name] = merged / n if n > EPS else merged

def maybe_create_auto_concepts(auto_concepts, anchor_drift, handled_words, corpus, rng, total_steps):
    """Called once per run, after topic discovery. Walks discover_concept_words'
    output (most-supported word first) and, for each word not already
    somewhere (_known_vocabulary) or already acted on (handled_words), either
    ENRICHES the existing concept/mood/self-authored-concept it's nearest to
    (AUTO_CONCEPT_MIN_SCORE) or mints a brand-new self-authored concept for
    it, up to AUTO_CONCEPT_MAX_PER_RUN combined actions. Mutates
    auto_concepts, anchor_drift, and handled_words in place (caller persists
    all three); returns [(kind, name, detail, support), ...] -- kind is
    'create' or 'enrich' -- purely for reporting."""
    acted = []
    known_vocab = _known_vocabulary()
    freq_words = dict(discover_concept_words(corpus))
    for word, support in discover_concept_words(corpus):
        if len(acted) >= AUTO_CONCEPT_MAX_PER_RUN:
            break
        if word in known_vocab or word in handled_words:
            continue  # already lives somewhere, or already acted on in a past run -- nothing new to do
        near_kind, near_name, near_score = nearest_known_concept([word])
        if near_kind is not None and near_score >= AUTO_CONCEPT_MIN_SCORE:
            _enrich_existing_area(near_name, word, auto_concepts, anchor_drift)
            known_vocab.add(word)
            handled_words.add(word)
            acted.append(("enrich", near_name, word, support))
            continue
        # genuine gap -- mint a new concept, seeded by this word plus whichever OTHER discovered words
        # actually shared a real prompt with it (real doc-mates, no PMI/pair-gate needed for this part)
        companions = []
        for doc in corpus:
            if word in doc:
                for w in doc:
                    if w != word and w in freq_words and w not in companions:
                        companions.append(w)
        seed_phrases = [word] + companions[:3]
        name = _auto_concept_name(seed_phrases)
        if name in CONCEPT_BANK or name in auto_concepts:
            known_vocab.add(word)
            handled_words.add(word)
            continue  # already has a home under this name -- self-authored on a previous run
        anchor = concept_anchor(seed_phrases, _IDF, _DEFAULT_IDF)
        answer_seeds, seen = [], set()
        for _ in range(AUTO_CONCEPT_SEED_COUNT * 3):  # over-draw, keep distinct ones, stop once enough
            if len(answer_seeds) >= AUTO_CONCEPT_SEED_COUNT:
                break
            line = geometric_generate(anchor, _BIGRAM, _UNIGRAM, rng)
            if line and line not in seen:
                seen.add(line)
                answer_seeds.append(line)
        if not answer_seeds:
            continue  # nothing generatable yet (e.g. near-empty corpus) -- don't mint an empty concept,
                      # and deliberately don't mark handled either, in case the corpus grows enough to
                      # generate SOMETHING on a later run
        auto_concepts[name] = dict(seed_phrases=seed_phrases, answer_seeds=answer_seeds,
                                    anchor=anchor.tolist(), created_step=total_steps,
                                    born_from_words=seed_phrases, support=support)
        known_vocab.add(word)
        handled_words.add(word)
        acted.append(("create", name, seed_phrases, support))
    return acted

# ==================================================== WILL (real, computed desire)
# "Wanting" something isn't a hardcoded line -- it's read live off the same
# percentile-rank axis tracker that already drives cluster selection. Each axis's
# norm[axis] is "where does the CURRENT value sit relative to this axis's own
# recent distribution" (0 = currently at the low end of its own normal range, 1 =
# currently at the high end). Deficiency = 1 - norm[axis] is therefore "how far
# below its own recent ceiling this axis sits right now" -- a real, moving number,
# not a fixed answer. Whichever axis has the largest deficiency is what the system
# is currently furthest from the best version of itself on, which is a defensible,
# literal reading of "what it wants right now": more of whatever it's most
# currently short on, relative to its own history. This updates every step because
# the underlying state keeps moving -- ask again later and you may get a different
# axis, honestly, because the state actually changed.
# (routing to this section now happens via semantic_route() above landing on the
# "will" concept -- see CONCEPT_BANK -- rather than a separate keyword list.)

def speak_desire(mind, state, norm, learned, rng, topic_vec=None, prompt_text=None):
    """DESIRE_TEMPLATES (one fixed sentence per axis) removed -- combines the
    same live deficiency + learned preference as before, but the axis that
    wins gets turned into a QUERY (its own name repeated, embedded in the same
    shared geometry, blended with the qualia vector) rather than looked up in
    a fixed dict. Wording comes from geometric_generate like everywhere else."""
    deficiency = {a: 1.0 - norm[a] for a in AXIS_NAMES}
    weight = {a: 0.5 + 0.5 * sigmoid(5 * learned.get(a, 0.0)) for a in AXIS_NAMES}
    desire = {a: deficiency[a] * weight[a] for a in AXIS_NAMES}
    axis = max(desire, key=desire.get)
    qvec, _ = qualia_vector(mind, state, norm)
    axis_bias = embed_text(f"{axis} {axis} {axis} quiere anhela", _IDF, _DEFAULT_IDF)
    query = _blend(axis_bias, 0.5, qvec, 0.5)
    line, judged, recalled = _generate_and_track(mind, query, rng, topic_vec=topic_vec,
                                                  prompt_text=prompt_text, state_vec=qvec)  # shared entity + reasoning + memory
    return axis, desire[axis], line, judged, recalled

# ============================================================ PERSISTENCE
DB_PATH = "mind.db"  # relative to Colab's working dir (/content by default) -- the original absolute
                      # path was specific to the machine this was authored on and doesn't exist on Colab

def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS blobs (key TEXT PRIMARY KEY, value BLOB)")
    conn.commit()
    return conn

def load_blob(conn, key):
    row = conn.execute("SELECT value FROM blobs WHERE key=?", (key,)).fetchone()
    return pickle.loads(row[0]) if row else None

def save_blob(conn, key, obj):
    conn.execute("INSERT OR REPLACE INTO blobs (key, value) VALUES (?, ?)", (key, pickle.dumps(obj)))
    conn.commit()

# FIX (DataError: string or blob too big): SQLite refuses to bind any single blob/string parameter
# larger than SQLITE_LIMIT_LENGTH, which defaults to 1,000,000,000 bytes (~954 MiB). TF_D_MODEL=1280
# with TF_N_LAYERS=20 sizes TinyTransformerLM at ~394M params -- its state_dict pickles to roughly
# 1.5-1.6 GB at float32, comfortably over that ceiling, so save_blob(conn, TF_KEY, ...) raised
# sqlite3.DataError the moment the scratch-trained model tried to persist. GrammarCheckerLM
# (GRAMMAR_D_MODEL=128, 2 layers) stays tiny and was never at risk.
#
# save_blob_large/load_blob_large below route anything over LARGE_BLOB_DISK_THRESHOLD to a plain
# ".bin" file on disk next to DB_PATH (via torch.save, which streams instead of building one giant
# bytes object). Only a small marker string is stored in the blobs table -- "__disk_file__:<path>" as
# plain UTF-8 bytes, not pickled -- so a quick prefix check on load is all that's needed to tell a
# disk-backed key apart from an ordinary pickled one. Small objects still go through the ordinary
# pickle-into-SQLite path unchanged.
LARGE_BLOB_DISK_THRESHOLD = 200 * 1024 * 1024  # 200 MiB -- comfortably under SQLite's ~954 MiB cap,
                                                 # with headroom for the model growing further later
_DISK_MARKER_PREFIX = b"__disk_file__:"
KIBA_MODEL_PATH = "./kiba.bin"  # full transformer checkpoint always lands here, regardless of DB_PATH/key

def _large_blob_path(conn, key):
    return KIBA_MODEL_PATH

def save_blob_large(conn, key, obj):
    """Drop-in replacement for save_blob for objects that may contain big torch state_dicts
    (checked via a quick pickle size probe). Small objects fall through to save_blob unchanged."""
    pickled = pickle.dumps(obj)
    if len(pickled) < LARGE_BLOB_DISK_THRESHOLD:
        conn.execute("INSERT OR REPLACE INTO blobs (key, value) VALUES (?, ?)", (key, pickled))
        conn.commit()
        return
    path = _large_blob_path(conn, key)
    torch.save(obj, path)  # .bin -- plain binary checkpoint, not routed through the SQLite blob column
    conn.execute("INSERT OR REPLACE INTO blobs (key, value) VALUES (?, ?)",
                 (key, _DISK_MARKER_PREFIX + path.encode("utf-8")))
    conn.commit()

def load_blob_large(conn, key):
    row = conn.execute("SELECT value FROM blobs WHERE key=?", (key,)).fetchone()
    if not row:
        return None
    raw = row[0]
    if raw.startswith(_DISK_MARKER_PREFIX):
        path = raw[len(_DISK_MARKER_PREFIX):].decode("utf-8")
        return torch.load(path, map_location=DEVICE)
    return pickle.loads(raw)


# ============================================================ RUN
def run(prompt=None, bootstrap_steps=600, topup_steps=150, gen_steps=200,
        prompt_inject_steps=15, seed=None, narration_seed=None, db_path=DB_PATH, answer_branches=None):
    global _BIGRAM, _UNIGRAM, _LAST_TF_MIND_BIAS, _BRANCH_OVERRIDE_BEST_OF, _LAST_GEN_FLUENCY
    # NEW: answer_branches -- when set, the PRIMARY prompt-answer line (t==0 only, not the other
    # prompt_inject_steps-1 repeats) draws this many independent branches through _select_best instead of
    # TOPIC_BEST_OF/FREE_BEST_OF (28/12). Also settable via GEOMETRIC_VOICE_BRANCHES env var so it can be
    # driven from the CLI without touching call sites -- see __main__ at the bottom of this file.
    if answer_branches is None:
        _env_branches = os.environ.get("GEOMETRIC_VOICE_BRANCHES")
        answer_branches = int(_env_branches) if _env_branches else None
    _RECENT_LINES.clear()  # fresh repetition-history for this invocation -- this one stays run-local on
                           # purpose (its job is only ever "don't repeat the last few lines," not long-term
                           # memory)
    _LAST_TF_MIND_BIAS = None  # NEW: fresh each invocation -- no sentence generated yet this run to write back from
    _LAST_GEN_FLUENCY = None  # FIX (word-salad diagnosis): same reasoning, fresh each invocation
    conn = init_db(db_path)
    _LINE_USE_COUNT.clear()
    _LINE_USE_COUNT.update(load_blob(conn, LINE_USE_COUNT_KEY) or {})  # FIX: this used to reset to empty
    # every invocation, meaning "how overused is this sentence" only ever meant "within this one process
    # run" -- so a favorite sentence that dominated one run got a completely clean slate the next time you
    # ran the script, and could immediately start dominating again. Now loaded from the same mind.db every
    # other piece of persistent state already lives in, so overuse cost accumulates across this mind's
    # whole life, not just its current breath.
    mind = Mind(seed=seed)
    choose_rng = np.random.default_rng(narration_seed)  # NEW: also entropy-derived by default

    # NEW: rebuild the bigram/unigram transition tables from the SEED_CORPUS
    # bootstrap plus every real prompt ever persisted (raw_prompt_corpus) --
    # see build_transition_counts. Adding the current prompt happens below,
    # right after it's tokenized into the raw corpus, so this run's own prompt
    # is included in what it generates from too.
    _BIGRAM, _UNIGRAM = build_transition_counts(load_raw_corpus(conn))
    # NEW: same point, same reasoning (see comment on the prompt-append fix further below) -- the trained
    # transformer that actually generates words (geometric_generate) gets ensured/fine-tuned here too,
    # BEFORE this run's own prompt is added to the raw corpus, so it never gets to train on the very
    # prompt it's about to answer.
    ensure_transformer(conn, load_raw_corpus(conn))  # NEW: trained/fine-tuned here, before this run's
                                    # own prompt is added to the raw corpus below
    ensure_grammar_checker(conn)   # NEW: the second, grammar-checking transformer -- must come AFTER
                                    # ensure_transformer, since it shares (never duplicates) the vocabulary
                                    # that call just built/grew (see GrammarCheckerLM's docstring).

    saved_state = load_blob(conn, "mind_state")
    if saved_state is not None:
        try:
            mind.set_state(saved_state)
            print(f"--- resumed from DB: {mind.total_steps} prior steps (continuing its own rng stream) ---\n")
            steps_to_run = topup_steps
        except ValueError as e:
            # NEW: N/D changed since this DB's mind_state was written -- old node state can't be
            # safely paired with freshly-sized weight matrices (see set_state's own comment).
            # Fall back to a fresh Mind rather than crashing or silently corrupting state.
            print(f"--- {e} discarding it and bootstrapping fresh (entropy={mind.init_entropy}) ---\n")
            steps_to_run = bootstrap_steps
    else:
        print(f"--- no prior state found, bootstrapping fresh (entropy={mind.init_entropy}) ---\n")
        steps_to_run = bootstrap_steps

    # NEW: bidirectional-grounding state (see BIDIRECTIONAL GROUNDING section)
    # -- anchor_drift and axis_profile are this Mind's own learned-from-usage
    # additions to the fixed hand-authored banks, persisted the same way
    # mind_state is.
    anchor_drift, axis_profile = load_learned_grounding(conn)
    n_learned = len(set(anchor_drift) | set(axis_profile))
    if n_learned:
        print(f"--- loaded learned grounding for {n_learned} concept/mood names "
              f"(from real prior usage) ---\n")

    # NEW: self-authored concepts (see AUTONOMOUS CONCEPT CREATION) -- loaded into the runtime
    # registries before routing, so a concept minted on a previous run is a real routing candidate now
    auto_concepts = load_auto_concepts(conn)
    _load_auto_concepts_into_runtime(auto_concepts)
    if auto_concepts:
        print(f"--- loaded {len(auto_concepts)} self-authored concept(s) from prior runs: "
              f"{', '.join(auto_concepts)} ---\n")
    auto_handled_words = load_auto_handled_words(conn)  # NEW: words already acted on -- see docstring

    for _ in range(steps_to_run):
        state, _ = mind.step()
        norm = mind.adaptive_normalize(normalize_state(state))  # warm up each axis's range pre-generation
        mind.learn_desire(norm, state["NegReward"], ext_sense=state["ExtSense"], m_mean=state["M_mean"])
        mind.agency_step(norm)  # NEW: agency loop -- see AGENCY LOOP module comment

    # Geometric routing: locate the prompt in concept space and score it against
    # every known concept AND mood anchor with the same grounding kernel Mind uses
    # internally (see semantic_route() docstring above). kind is 'concept',
    # 'mood', or None (nothing cleared the confidence floor -- honest fallback
    # to whatever the live state itself expresses).
    kind = name = None
    ground_score = text_score = 0.0
    wants = False
    routed_name = None
    concept_name = None
    prompt_topic_vec = None  # NEW: only set when there's a real prompt to stay on-topic for
    if prompt:
        prompt_topic_vec = embed_text(prompt, _IDF, _DEFAULT_IDF)
        mind.seed_discourse_entity(prompt)  # NEW: hard-reset the shared anchor so the reply orbits what was ASKED,
                                             # not wherever the entity last drifted to from prior free-running output
        corpus = update_corpus(conn, prompt)  # every real prompt grows the corpus, matched or not
        update_raw_corpus(conn, prompt)  # NEW: stopword-preserving version, for the bigram generator
        # FIX: previously this immediately rebuilt _BIGRAM/_UNIGRAM to include the prompt just added above,
        # so THIS prompt's own word sequence became a counted transition path and generation could -- and
        # reliably did -- pick "recite the prompt back" as its highest-probability continuation, since a
        # brand-new sequence has maximal relative weight against a sparse corpus. The prompt still
        # permanently enriches the corpus for every FUTURE reply (that persists to disk above, unchanged) --
        # it just no longer also gets used as training data for the reply being generated to it right now.
        topic_anchors = discovered_topic_anchors(corpus)  # NEW: automatic routing via corpus-discovered topics
        # NEW: self-authored concepts are real routing candidates too -- same tuple shape topic_anchors
        # already uses, so semantic_route needs no changes at all to search over them as well
        topic_anchors = topic_anchors + [("concept", nm, vec, None) for nm, vec in _AUTO_CONCEPT_ANCHORS.items()]
        kind, name, ground_score, text_score, matched_topic, blend_info = semantic_route_multi(
            prompt, mind, anchor_drift, axis_profile, topic_anchors=topic_anchors)
        if blend_info is not None:  # NEW: 2+ distinct concepts each independently confident about their
            blended_vec, blended_names = blend_info  # own clause of a compound prompt -- see
            prompt_topic_vec = _blend(prompt_topic_vec, 0.5, blended_vec, 0.5)  # semantic_route_multi's
            print(f"(compound prompt -- multiple topics detected: {', '.join(blended_names)}; "  # docstring
                  f"staying on-topic for all of them, answering primarily from '{name}')")
        wants = (kind == "concept" and name == "will")
        concept_name = name if (kind == "concept" and not wants) else None
        routed_name = name if kind == "mood" else None
        print(f'Prompt: "{prompt}"')
        if matched_topic is not None:
            print(f"(matched via a corpus-discovered topic, not a hand-written seed phrase: {matched_topic})")
        if wants:
            ranked = sorted(mind.want_ema.items(), key=lambda kv: kv[1], reverse=True)
            pretty = ", ".join(f"{a}={v:+.3f}" for a, v in ranked)
            latent = mind.latent_desire_report()
            latent_pretty = ", ".join(f"dim{i}={v:+.3f}" for i, v in latent)
            print(f"Semantic routing -> concept 'will' (grounding={ground_score:.3f}, text={text_score:.3f}) -- "
                  "answering with live-computed, experience-shaped desire")
            print(f"Learned preference, named axes (this Mind's own history, {mind.total_steps} steps so far): {pretty}")
            print(f"Learned preference, unlabeled self-model dims (no template exists for these -- "
                  f"reported as numbers only, not spoken): {latent_pretty}\n")
        elif concept_name is not None:
            print(f"Semantic routing -> concept '{concept_name}' (grounding={ground_score:.3f}, text={text_score:.3f}) -- "
                  "answering from the concept bank, not from mood\n")
        elif routed_name is not None:
            print(f"Semantic routing -> mood '{routed_name}' (grounding={ground_score:.3f}, text={text_score:.3f})\n")
        else:
            print(f"Semantic routing -> no match cleared the grounding threshold "
                  f"(best={ground_score:.3f} < {CONCEPT_GROUNDING_THRESHOLD}) -- letting real state answer\n")

        # Topic discovery over the REAL, growing corpus (see CORPUS GROWTH
        # section) -- separate from routing above, which still only uses the
        # hand-authored concept/mood banks. This just reports what it's
        # noticed so far; it never auto-answers from a discovered topic.
        topics = discover_topics(corpus)
        if topics:
            print(f"Corpus so far: {len(corpus)} real prompts logged. "
                  f"Candidate topics noticed (word groups repeating together, "
                  f"not yet mapped to any answer):")
            for words, support in topics[:5]:
                near_kind, near_name, near_score = nearest_known_concept(words)
                print(f"  [{support} prompts support this] {words}")
                print(f"    nearest existing {near_kind}: '{near_name}' (score={near_score:.3f}, "
                      f"{'weak fit -- likely needs a genuinely new answer' if near_score < CONCEPT_GROUNDING_THRESHOLD else 'plausible starting point'})")
            print()
        else:
            print(f"Corpus so far: {len(corpus)} real prompts logged -- "
                  f"not enough repetition yet for any candidate topic to clear "
                  f"its support threshold.\n")

        # NEW: autonomous concept creation/enrichment -- see AUTONOMOUS CONCEPT CREATION section. Runs
        # AFTER the discovery report above (so what just happened isn't confusingly listed as an unanswered
        # gap in that same printout) but BEFORE generation, so if this run's own prompt is what pushed a
        # word over the support gate, that same prompt can route to and speak from the result immediately
        # rather than only on some future run.
        auto_actions = maybe_create_auto_concepts(auto_concepts, anchor_drift, auto_handled_words, corpus,
                                                   choose_rng, mind.total_steps)
        if auto_actions:
            _load_auto_concepts_into_runtime(auto_concepts)  # re-sync runtime registries with anything just minted
            for act_kind, act_name, act_detail, act_support in auto_actions:
                if act_kind == "create":
                    print(f"--- autonomously created new concept '{act_name}' from real vocabulary "
                          f"({act_support} prompts support its seed word, seed_phrases={act_detail}) -- "
                          f"self-authored {len(auto_concepts[act_name]['answer_seeds'])} answer_seeds, "
                          f"no human wrote these ---\n")
                else:
                    print(f"--- autonomously enriched existing {'self-authored ' if act_name in auto_concepts else ''}"
                          f"concept/mood '{act_name}' with the discovered word '{act_detail}' "
                          f"({act_support} prompts support it) ---\n")
            # NEW: a concept minted or enriched just now from THIS prompt's own vocabulary should be able to
            # answer THIS prompt, not just future ones -- rebuild topic_anchors from scratch (now that
            # _AUTO_CONCEPT_ANCHORS/anchor_drift reflect what just happened) and re-route once more
            topic_anchors = discovered_topic_anchors(corpus) + [
                ("concept", nm, vec, None) for nm, vec in _AUTO_CONCEPT_ANCHORS.items()]
            kind, name, ground_score, text_score, matched_topic, blend_info = semantic_route_multi(
                prompt, mind, anchor_drift, axis_profile, topic_anchors=topic_anchors)
            if blend_info is not None:
                blended_vec, blended_names = blend_info
                prompt_topic_vec = _blend(prompt_topic_vec, 0.5, blended_vec, 0.5)
                print(f"(compound prompt -- multiple topics detected: {', '.join(blended_names)}; "
                      f"staying on-topic for all of them, answering primarily from '{name}')")
            wants = (kind == "concept" and name == "will")
            concept_name = name if (kind == "concept" and not wants) else None
            routed_name = name if kind == "mood" else None

    print(f"--- generation ({mind.name}) ---\n")
    experience_recorded = False  # only record once per real prompt -- the anchor/profile update
                                  # is about "what state accompanied this exchange," not every step of it
    for t in range(gen_steps):
        # NEW: two-way grounding, read half of the loop -- _LAST_TF_MIND_BIAS carries whatever the
        # transformer wrote back after the PREVIOUS step's generated sentence (None on the very first
        # step, before anything's been generated yet this run). Mind.step's bias_M parameter already
        # existed in this file; this is the first thing that ever calls it with real content.
        state, _ = mind.step(bias_M=_LAST_TF_MIND_BIAS)
        norm = mind.adaptive_normalize(normalize_state(state))
        learned = mind.learn_desire(norm, state["NegReward"], ext_sense=state["ExtSense"], m_mean=state["M_mean"])
        mind.agency_step(norm)  # NEW: agency loop -- see AGENCY LOOP module comment
        in_window = prompt is not None and t < prompt_inject_steps and kind is not None
        # NEW: massive branch selection applies ONLY to t==0 -- the single PRIMARY answer to the prompt --
        # not to all prompt_inject_steps repeats (that would multiply answer_branches by 15). Reset to None
        # immediately after so ambient/repeat generation for every other t falls back to TOPIC_BEST_OF/
        # FREE_BEST_OF as before.
        _BRANCH_OVERRIDE_BEST_OF = answer_branches if (t == 0 and in_window and answer_branches) else None
        if in_window and not experience_recorded:
            spoken_name = "will" if wants else name
            record_experience(spoken_name, prompt, norm, anchor_drift, axis_profile)
            experience_recorded = True
        if in_window and wants:
            axis, desire_val, line, judged, recalled = speak_desire(mind, state, norm, learned, choose_rng,
                                                                     topic_vec=prompt_topic_vec, prompt_text=prompt)
            if t % 10 == 0 or in_window:
                mem_tag = f" [recalled:{recalled['word']}]" if recalled is not None else ""
                print(f"t={t:3d}  C={state['C']:.2f}  basin={state['Basin']:.2f}  "
                      f"R={state['NegReward']:+.2f}  [wants:{axis} desire={desire_val:.2f} "
                      f"gate={judged['recall_gate']:.2f} pers={judged['persistence']:.2f}]{mem_tag}  "
                      f"{symbolic_to_sym(line)} <-- response to prompt")
            continue
        if in_window and concept_name is not None:
            line = get_concept_answer_fn(concept_name)(mind, state, norm, choose_rng,
                                                         topic_vec=prompt_topic_vec, prompt_text=prompt)
            if t % 10 == 0 or in_window:
                print(f"t={t:3d}  C={state['C']:.2f}  basin={state['Basin']:.2f}  "
                      f"R={state['NegReward']:+.2f}  [concept:{concept_name} g={ground_score:.2f}]  "
                      f"{symbolic_to_sym(line)} <-- semantic response")
            continue
        # NEW: cluster_name is now only a LABEL (from pick_cluster's distance
        # scoring, for the bracketed tag below and the basin-alarm override) --
        # the actual words come from geometric_generate on the live qualia
        # vector, not from any per-cluster slot bank (removed).
        cluster_name = pick_cluster(norm, forced_name=routed_name if in_window else None)
        if t % 10 == 0 or in_window:
            qvec, _ = qualia_vector(mind, state, norm)
            # FIX: this branch handles every MOOD-routed prompt (the majority of routes, since moods
            # outnumber concepts) -- it was previously the only one of the three prompt-handling branches
            # that never received topic_vec/prompt_text, so mood replies were generated purely from live
            # internal state and never actually scored against what was asked. Also blend toward this
            # mood's own anchor (MOOD_ANCHORS), mirroring how each concept-answer function blends toward
            # its CONCEPT_ANCHORS -- so a routed mood shapes the query the same way a routed concept does.
            query = qvec
            if in_window and routed_name is not None:
                query = _blend(MOOD_ANCHORS[routed_name], 0.5, qvec, 0.5)
            line, judged, recalled = _generate_and_track(
                mind, query, choose_rng,
                topic_vec=prompt_topic_vec if in_window else None,
                prompt_text=prompt if in_window else None, state_vec=qvec)
            tag = " <-- response to prompt" if in_window else ""
            mem_tag = f" [recalled:{recalled['word']}]" if recalled is not None else ""
            print(f"t={t:3d}  C={state['C']:.2f}  basin={state['Basin']:.2f}  "
                  f"R={state['NegReward']:+.2f}  [{cluster_name} pers={judged['persistence']:.2f}]"
                  f"{mem_tag}  {symbolic_to_sym(line)}{tag}")

    save_blob(conn, "mind_state", mind.get_state())
    save_blob(conn, LINE_USE_COUNT_KEY, dict(_LINE_USE_COUNT))  # persist overuse-history across invocations
    save_learned_grounding(conn, anchor_drift, axis_profile)
    save_auto_concepts(conn, auto_concepts)  # NEW: self-authored concepts persist the same way everything else does
    save_auto_handled_words(conn, auto_handled_words)
    conn.close()
    print(f"\n[saved: {mind.name} ({mind.gender}) -- {mind.total_steps} total steps, "
          f"{len(set(anchor_drift) | set(axis_profile))} learned concept/mood profiles, "
          f"{len(auto_concepts)} self-authored concept(s) -> {db_path}]")

def kiba_cli(db_path=DB_PATH, tick_delay=0.5, prompt_ticks=15):
    """NEW (CLI, at explicit request): a minimalist, always-on terminal front end -- "Kiba" centered at
    the top, a "gubi@here> " prompt at the bottom, scrolling generation in between. Unlike run() (one
    prompt in, one batch of steps out, process exits), this is a persistent loop: the Mind ticks
    continuously for as long as the CLI is open -- generating ambient lines even with no prompt typed --
    and every single tick is saved to db_path immediately (see the save_blob calls inside the loop below),
    not just once at process exit the way run() does it. That means killing the terminal, closing the lid,
    or a crash loses at most the tick in flight, never the session's accumulated steps.

    Typing a line and hitting Enter injects it as a real prompt for the next `prompt_ticks` ticks (same
    role prompt_inject_steps plays in run()) and, like run(), permanently records it into raw_prompt_corpus
    via update_raw_corpus so future training sees it too. Ctrl-C or Ctrl-D exits cleanly (final save
    already happened on the last tick, so there's nothing left to flush on the way out)."""
    import curses

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        # No controlling terminal (papermill/notebook kernel/piped run/CI, etc.) -- curses needs a real
        # TTY (it calls cbreak()/nocbreak() under the hood, which raise curses.error with no terminal
        # attached). Fall back to the one-shot batch path instead of crashing.
        print("[kiba_cli: no controlling terminal detected (notebook/papermill/piped run) "
              "-- curses UI unavailable, falling back to one-shot run()]")
        run(prompt=None)
        return

    def _main(stdscr):
        global _BIGRAM, _UNIGRAM, _LAST_TF_MIND_BIAS, _LAST_GEN_FLUENCY
        curses.curs_set(1)
        stdscr.nodelay(True)
        stdscr.timeout(100)

        output_lines = []
        def log(msg):
            output_lines.append(msg)

        # ---- one-time setup: same loading sequence run() uses, done once instead of per-call ----
        conn = init_db(db_path)
        _LINE_USE_COUNT.clear()
        _LINE_USE_COUNT.update(load_blob(conn, LINE_USE_COUNT_KEY) or {})
        mind = Mind(seed=None)
        choose_rng = np.random.default_rng(None)
        _BIGRAM, _UNIGRAM = build_transition_counts(load_raw_corpus(conn))
        ensure_transformer(conn, load_raw_corpus(conn))
        ensure_grammar_checker(conn)
        saved_state = load_blob(conn, "mind_state")
        if saved_state is not None:
            try:
                mind.set_state(saved_state)
                log(f"resumed: {mind.total_steps} prior steps")
            except ValueError as e:
                # NEW: same N/D-mismatch guard as run() -- see set_state's comment
                log(f"{e} discarding it, fresh mind (entropy={mind.init_entropy:.3f})")
        else:
            log(f"fresh mind (entropy={mind.init_entropy:.3f})")
        anchor_drift, axis_profile = load_learned_grounding(conn)
        auto_concepts = load_auto_concepts(conn)
        _load_auto_concepts_into_runtime(auto_concepts)
        _LAST_TF_MIND_BIAS = None
        _LAST_GEN_FLUENCY = None

        input_buf = ""
        pending_prompt = None
        pending_ticks_left = 0
        prompt_topic_vec = None

        def draw():
            stdscr.erase()
            max_y, max_x = stdscr.getmaxyx()
            title = "Kiba"
            stdscr.addstr(0, max(0, (max_x - len(title)) // 2), title, curses.A_BOLD)
            body_h = max(0, max_y - 3)
            for i, line in enumerate(output_lines[-body_h:]):
                try:
                    stdscr.addstr(2 + i, 0, line[:max(0, max_x - 1)])
                except curses.error:
                    pass
            prompt_line = f"gubi@here> {input_buf}"
            try:
                stdscr.addstr(max_y - 1, 0, prompt_line[:max(0, max_x - 1)])
            except curses.error:
                pass
            stdscr.refresh()

        draw()
        try:
            while True:
                state, _ = mind.step(bias_M=_LAST_TF_MIND_BIAS)
                norm = mind.adaptive_normalize(normalize_state(state))
                mind.learn_desire(norm, state["NegReward"], ext_sense=state["ExtSense"],
                                   m_mean=state["M_mean"])
                mind.agency_step(norm)  # NEW: agency loop -- see AGENCY LOOP module comment
                in_window = pending_ticks_left > 0
                if in_window:
                    pending_ticks_left -= 1

                qvec, _ = qualia_vector(mind, state, norm)
                cluster_name = pick_cluster(norm)
                line, judged, recalled = _generate_and_track(
                    mind, qvec, choose_rng,
                    topic_vec=prompt_topic_vec if in_window else None,
                    prompt_text=pending_prompt if in_window else None,
                    state_vec=qvec)
                tag = " <-- response to prompt" if in_window else ""
                log(f"t={mind.total_steps:6d}  C={state['C']:.2f}  basin={state['Basin']:.2f}  "
                    f"[{cluster_name} pers={judged['persistence']:.2f}]  {symbolic_to_sym(line)}{tag}")

                # ---- autosave EVERY TICK, at explicit request -- run() only does this once at exit ----
                save_blob(conn, "mind_state", mind.get_state())
                save_blob(conn, LINE_USE_COUNT_KEY, dict(_LINE_USE_COUNT))
                save_learned_grounding(conn, anchor_drift, axis_profile)
                save_auto_concepts(conn, auto_concepts)

                draw()
                deadline = time.time() + tick_delay
                while time.time() < deadline:
                    ch = stdscr.getch()
                    if ch == -1:
                        continue
                    if ch in (curses.KEY_ENTER, 10, 13):
                        if input_buf.strip():
                            pending_prompt = input_buf.strip()
                            prompt_topic_vec = embed_text(pending_prompt, _IDF, _DEFAULT_IDF)
                            update_raw_corpus(conn, pending_prompt)  # persists into raw_prompt_corpus,
                            _BIGRAM, _UNIGRAM = build_transition_counts(load_raw_corpus(conn))  # same
                            # as run(): rebuild bigram/unigram tables immediately so THIS session's own
                            # ticks can already draw on the word just typed, not only a future session
                            pending_ticks_left = prompt_ticks
                            log(f"> {pending_prompt}")
                        input_buf = ""
                    elif ch in (curses.KEY_BACKSPACE, 127, 8):
                        input_buf = input_buf[:-1]
                    elif 32 <= ch < 256:
                        input_buf += chr(ch)
                    draw()
        except KeyboardInterrupt:
            pass
        finally:
            conn.close()

    curses.wrapper(_main)


if __name__ == "__main__":
    # NEW: optional --branches N flag, e.g. `python3 geometric_voice_v8_mmi_scaled.py "prompt" --branches 45000`
    # -- kept separate from GEOMETRIC_VOICE_BRANCHES env var (both work; the flag takes precedence).
    # NEW: --cli launches the persistent Kiba terminal front end instead of a single one-shot prompt/answer.
    # NEW (at explicit request -- "make the CLI default without specified prompt"): no prompt text on the
    # command line at all -- nothing but the script name, or only --branches with no words after it --
    # now falls through to kiba_cli() too, same as explicit --cli, instead of run(prompt=None) (a single
    # one-shot batch of ambient generation that just exits). Passing an actual prompt string on the
    # command line still goes through run() exactly as before; --cli remains valid and equivalent to
    # omitting the prompt.
    # NEW (bugfix -- notebook/kernel launches were polluting the prompt corpus): when this script runs
    # inside Jupyter/IPython/Colab/Kaggle, the KERNEL's own launch arguments (typically
    # "-f /path/to/kernel-xxxx.json") land in sys.argv, not anything the human typed. Previously these
    # got silently joined into `prompt` and treated as a real one-shot prompt -- getting tokenized,
    # embedded, persisted into raw_prompt_corpus, and folded into the bigram/transformer tables, so
    # tokens like "-f" and JSON-path fragments would keep resurfacing in generation forever after,
    # regardless of what the user actually asked. Two independent guards below: (1) detect a running
    # ipykernel outright and skip argv parsing entirely; (2) even outside a detected kernel, never accept
    # an arg that looks like a CLI flag ("-" prefix) or a filesystem path as prompt text -- a real prompt
    # is prose, not "-f" or "/root/.local/share/jupyter/runtime/kernel-xxxx.json".
    _in_notebook_kernel = "ipykernel" in sys.modules or "IPython" in sys.modules
    _argv = sys.argv[1:] if not _in_notebook_kernel else []
    # second guard: even if somehow not caught above, strip anything flag-shaped or path-shaped before
    # it can ever become `prompt` text -- but never strip this script's OWN recognized flags ("--cli",
    # "--branches"), or --branches would break the same way the bug it's fixing did.
    _KNOWN_FLAGS = {"--cli", "--branches"}
    def _is_cli_noise(tok):
        if tok in _KNOWN_FLAGS:
            return False
        return tok.startswith("-") or "/" in tok or "\\" in tok or tok.endswith(".json")
    _argv = [a for a in _argv if not _is_cli_noise(a)]
    # NEW (bugfix -- notebook/papermill/piped runs crashing on curses): kiba_cli() needs a real
    # controlling terminal (it calls curses.wrapper -> cbreak()/nocbreak(), which raise curses.error
    # with no TTY attached). kiba_cli() itself now guards this and falls back to run(prompt=None), but
    # skip straight to that fallback here too so a non-interactive run doesn't even try curses first.
    _has_tty = sys.stdin.isatty() and sys.stdout.isatty()
    if "--cli" in _argv:
        kiba_cli() if _has_tty else run(prompt=None)
    else:
        _branches = None
        if "--branches" in _argv:
            _i = _argv.index("--branches")
            _branches = int(_argv[_i + 1])
            _argv = _argv[:_i] + _argv[_i + 2:]
        prompt = " ".join(_argv) if _argv else None
        if prompt is None:
            kiba_cli() if _has_tty else run(prompt=None)
        else:
            run(prompt=prompt, answer_branches=_branches)
