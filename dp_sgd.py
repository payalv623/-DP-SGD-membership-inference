"""
dp_sgd.py  --  Differentially Private SGD for a CNN + membership inference (Assignment 3, Q3)
============================================================================================

What is implemented here (from scratch, in PyTorch):

1. DP-SGD (Abadi et al., 2016).  For every training step:
     a. Poisson-sample a batch: each training example is included independently
        with probability q = B / N  (B = expected batch size). Poisson sampling
        is what the privacy accountant below assumes.
     b. Per-example gradients g_i (torch.func.vmap over torch.func.grad).
     c. Clip each one:   g_i_bar = g_i * min(1, C / ||g_i||_2)
     d. Sum, add noise, average:
            g_tilde = (1/B) * ( sum_i g_i_bar + N(0, sigma^2 C^2 I) )
     e. Apply the (unchanged) SGD optimizer to g_tilde.
   The non-private baseline uses the same model, data, optimizer, batch size
   and number of epochs, but ordinary mini-batch gradients (no clip, no noise).

2. Privacy accounting: Renyi-DP (RDP) of the Poisson-subsampled Gaussian
   mechanism (Mironov, Talwar & Zhang, 2019) at integer orders, composed over
   T steps, converted to (epsilon, delta)-DP (Balle et al., 2020 conversion).
   Cross-checked against Opacus' RDPAccountant (`accountant-check` stage).

3. Membership inference attack (MIA): the loss-threshold attack (Yeom et al.,
   2018). Members = training examples the model saw; non-members = held-out
   test examples. The attacker predicts "member" when the per-example loss is
   low. We report ROC-AUC, membership advantage = max(TPR - FPR), and TPR at
   1% FPR (Carlini et al., 2022), and compare the advantage with the upper
   bound that (epsilon, delta)-DP implies:  adv <= (e^eps - 1 + 2 delta)/(e^eps + 1).

4. Experiments:
   A. noise multiplier sigma in {0.5, 1, 2, 4} x clipping norm C in {two values},
      + clip-only runs (sigma = 0) to separate the effect of clipping from noise,
      + the non-private baseline.
   B. training-set size N in {3k, 10k, 30k, 60k}: non-private vs DP.
   C. training duration: accuracy, epsilon and MIA are logged after every epoch.

What the reported epsilon does and does NOT cover (stated in the report):
  epsilon covers the released model weights produced by DP-SGD. It does NOT
  cover (i) the monitoring statistics logged during training (train loss,
  train accuracy, per-example gradient-norm statistics), (ii) the choice of
  the clipping norms, which were picked from per-example gradient norms
  measured on the private training data (grad_norms.csv), or (iii) the
  normalisation mean/std computed on the training set. A deployment would take
  these from public data or account for them separately.

Usage (project root, venv active):
  python q3_dpsgd/dp_sgd.py gradnorms          # per-example gradient-norm statistics (to choose C)
  python q3_dpsgd/dp_sgd.py accountant-check   # our accountant vs Opacus
  python q3_dpsgd/dp_sgd.py run [--quick]      # all experiments (resumable)
  python q3_dpsgd/dp_sgd.py plot               # figures from saved results
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

HERE = Path(__file__).resolve().parent
# shared with the other question when run inside the assignment folder; a
# standalone clone of this repository downloads Fashion-MNIST into ./data instead
DATA_DIR = HERE.parent / "data" if (HERE.parent / "data").exists() else HERE / "data"          # shared with Q2
RESULTS = HERE / "results"
FIG_DIR = RESULTS / "figures"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CFG = {
    "dataset": "Fashion-MNIST",
    "epochs": 20,
    "batch_size": 256,                 # B (expected batch size under Poisson sampling)
    "lr": 0.05, "momentum": 0.9,       # same SGD settings for private and non-private runs
    "delta": 1e-5,                     # < 1/N for N = 60,000
    "noise_multipliers": [0.5, 1.0, 2.0, 4.0],
    "clip_norms": [0.5, 2.0, 8.0],     # median per-example norm ~0.5 late in training, p90 ~16 (grad_norms.csv)
    "dataset_sizes": [3000, 10000, 30000, 60000],
    "size_exp_sigma": 1.0, "size_exp_clip": 0.5,
    "seeds": [0, 1, 2],
    # training-duration experiment: small N trained long enough to overfit
    "duration_n": 3000, "duration_epochs": 100,
    "mia_max_members": 10000,
}
# Fractional orders near 1 matter when epsilon is large (e.g. sigma = 0.5, where
# the best integer order was the boundary value 2). Same grid style as Opacus.
RDP_ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 257))


# ---------------------------------------------------------------------------
# 1. DATA AND MODEL (identical to Q2 so the two questions are comparable)
# ---------------------------------------------------------------------------
def load_fashion_mnist():
    from torchvision.datasets import FashionMNIST
    tr = FashionMNIST(DATA_DIR, train=True, download=True)
    te = FashionMNIST(DATA_DIR, train=False, download=True)
    xtr = tr.data.float().div(255.0).unsqueeze(1)
    xte = te.data.float().div(255.0).unsqueeze(1)
    mean, std = xtr.mean().item(), xtr.std().item()
    xtr, xte = (xtr - mean) / std, (xte - mean) / std
    return xtr.to(DEVICE), tr.targets.to(DEVICE), xte.to(DEVICE), te.targets.to(DEVICE)


class SmallCNN(nn.Module):
    """Same fixed architecture as Q2 (421,642 parameters). It has no BatchNorm,
    which is also a requirement for DP-SGD: BatchNorm mixes information across
    the examples of a batch, so per-example gradients would not be well defined."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# 2. PER-EXAMPLE GRADIENTS, CLIPPING AND NOISE  (the core of DP-SGD)
# ---------------------------------------------------------------------------
def per_example_grads(model, xb, yb):
    """
    Returns {param_name: tensor of shape (b, *param.shape)}: one gradient per
    example. torch.func.grad differentiates the loss of a SINGLE example and
    vmap vectorises that over the batch, so no example's gradient is mixed
    with another's.
    """
    params = {k: v.detach() for k, v in model.named_parameters()}

    def loss_one(p, x, y):
        out = functional_call(model, p, (x.unsqueeze(0),))
        return F.cross_entropy(out, y.unsqueeze(0))

    return vmap(grad(loss_one), in_dims=(None, 0, 0))(params, xb, yb)


def dp_gradient(model, xb, yb, clip, sigma, expected_batch, gen):
    """
    Implements, for one Poisson-sampled batch:
        g_bar_i = g_i * min(1, C / ||g_i||_2)                       (clip)
        g_tilde = (1/B) * ( sum_i g_bar_i + N(0, sigma^2 C^2 I) )   (noise + average)
    and writes g_tilde into p.grad so a standard optimizer can apply it.
    Returns statistics about the per-example norms (for logging).
    """
    names = [n for n, _ in model.named_parameters()]
    if len(yb) > 0:
        g = per_example_grads(model, xb, yb)
        # total L2 norm of each example's gradient over ALL parameters
        norms = torch.sqrt(sum(g[n].flatten(1).pow(2).sum(1) for n in names))
        factor = torch.clamp(clip / (norms + 1e-12), max=1.0)           # min(1, C/||g_i||)
        clipped_frac = (norms > clip).float().mean().item()
    for n, p in model.named_parameters():
        summed = (torch.einsum("b,b...->...", factor, g[n]) if len(yb) > 0
                  else torch.zeros_like(p))
        noise = torch.normal(0.0, sigma * clip, size=p.shape, device=p.device, generator=gen)
        p.grad = (summed + noise) / expected_batch
    if len(yb) == 0:
        return {"median_norm": float("nan"), "clipped_frac": float("nan")}
    return {"median_norm": norms.median().item(), "clipped_frac": clipped_frac}


# ---------------------------------------------------------------------------
# 3. PRIVACY ACCOUNTANT: RDP of the Poisson-subsampled Gaussian mechanism
# ---------------------------------------------------------------------------
def _log_add(a, b):
    """log(exp(a) + exp(b)) without overflow."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log1p(math.exp(min(a, b) - m))


def _log_sub(a, b):
    """log(exp(a) - exp(b)) for a >= b."""
    if b == -math.inf:
        return a
    if a == b:
        return -math.inf
    return math.log(math.expm1(a - b)) + b


def _log_erfc(x):
    from scipy.special import log_ndtr
    return math.log(2) + float(log_ndtr(-x * 2 ** 0.5))


def _log_a_frac(q, sigma, alpha):
    """
    log(A_alpha) for FRACTIONAL alpha (Mironov, Talwar & Zhang, 2019, Sec. 3.3):
    the binomial sum becomes an infinite series split at z0, evaluated until the
    terms are negligible (< e^-30).
    """
    log_a0, log_a1 = -math.inf, -math.inf
    i = 0
    z0 = sigma ** 2 * math.log(1 / q - 1) + 0.5
    from scipy.special import binom
    while True:
        coef = binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i
        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)
        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))
        log_s0 = log_t0 + (i * i - i) / (2 * sigma ** 2) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * sigma ** 2) + log_e1
        if coef > 0:
            log_a0, log_a1 = _log_add(log_a0, log_s0), _log_add(log_a1, log_s1)
        else:
            log_a0, log_a1 = _log_sub(log_a0, log_s0), _log_sub(log_a1, log_s1)
        i += 1
        if max(log_s0, log_s1) < -30:
            break
    return _log_add(log_a0, log_a1)


def rdp_subsampled_gaussian(q, sigma, alpha):
    """
    RDP of order alpha > 1 for ONE step of the sampled Gaussian mechanism with
    sampling rate q and noise multiplier sigma (Mironov et al., 2019, Sec. 3.3).
    Integer alpha uses the exact finite binomial expansion:
        A_alpha = sum_{k=0}^{alpha} C(alpha,k) (1-q)^(alpha-k) q^k exp((k^2 - k) / (2 sigma^2))
    fractional alpha uses the series in _log_a_frac. RDP(alpha) = log(A_alpha) / (alpha - 1).
    Computed in log-space for numerical stability.
    """
    if sigma == 0:
        return math.inf
    if q == 0:
        return 0.0
    if q == 1.0:
        return alpha / (2 * sigma ** 2)
    if float(alpha) != int(alpha):
        return _log_a_frac(q, sigma, alpha) / (alpha - 1)
    alpha = int(alpha)
    log_terms = []
    for k in range(alpha + 1):
        log_binom = math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
        log_terms.append(log_binom + k * math.log(q) + (alpha - k) * math.log1p(-q)
                         + (k * k - k) / (2 * sigma ** 2))
    m = max(log_terms)
    log_a = m + math.log(sum(math.exp(t - m) for t in log_terms))
    return log_a / (alpha - 1)


_RDP_CACHE = {}


def rdp_per_step(q, sigma, orders=RDP_ORDERS):
    """RDP of one step at every order (cached: it depends only on q and sigma)."""
    key = (q, sigma, tuple(orders))
    if key not in _RDP_CACHE:
        _RDP_CACHE[key] = np.array([rdp_subsampled_gaussian(q, sigma, a) for a in orders])
    return _RDP_CACHE[key]


def epsilon_from_rdp(q, sigma, steps, delta, orders=RDP_ORDERS):
    """
    Compose T steps (RDP adds up linearly) and convert to (eps, delta)-DP with
        eps = RDP_total(a) + log((a-1)/a) - (log(delta) + log(a)) / (a-1)
    (Balle et al., 2020; the same conversion Opacus uses), minimised over orders a.
    Returns (eps, best_order).
    """
    if sigma == 0:
        return math.inf, None
    a = np.array(orders, float)
    eps = steps * rdp_per_step(q, sigma, orders) + np.log((a - 1) / a) - (math.log(delta) + np.log(a)) / (a - 1)
    i = int(np.nanargmin(eps))
    return float(eps[i]), orders[i]


def accountant_check():
    """Compare our epsilon with Opacus' RDPAccountant on the settings actually used
    (20 epochs x 234 steps at N=60k; N=3000 runs: 12 steps/epoch x 20 or x 100
    epochs) plus a generic setting, using the same (fractional + integer) orders."""
    from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
    rows = []
    for q, sigma, steps in [(256 / 60000, 0.5, 4680), (256 / 60000, 1.0, 4680),
                            (256 / 60000, 2.0, 4680), (256 / 60000, 4.0, 4680),
                            (256 / 3000, 1.0, 240), (256 / 3000, 1.0, 1200), (0.01, 1.1, 10000)]:
        ours, a = epsilon_from_rdp(q, sigma, steps, CFG["delta"])
        rdp = compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=RDP_ORDERS)
        theirs, a2 = get_privacy_spent(orders=RDP_ORDERS, rdp=rdp, delta=CFG["delta"])
        rows.append({"q": q, "sigma": sigma, "steps": steps, "eps_ours": ours, "eps_opacus": theirs,
                     "abs_diff": abs(ours - theirs), "order_ours": a, "order_opacus": a2})
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "accountant_check.csv", index=False)
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# 4. MEMBERSHIP INFERENCE ATTACK (loss threshold)
# ---------------------------------------------------------------------------
@torch.no_grad()
def per_example_loss(model, x, y, bs=4000):
    model.eval()
    return torch.cat([F.cross_entropy(model(x[i:i + bs]), y[i:i + bs], reduction="none")
                      for i in range(0, len(y), bs)]).cpu().numpy()


def membership_inference(model, x_mem, y_mem, x_non, y_non):
    """
    Loss-threshold attack. Score = -loss (members are expected to have LOWER
    loss because the model was fitted to them).
      * AUC is threshold-free -> computed on all members / non-members, with a
        95% CI from the Hanley-McNeil standard error.
      * Advantage and TPR@1%FPR need a THRESHOLD. Picking it on the same data it
        is evaluated on is optimistically biased, so the attacker calibrates the
        threshold on one half (even indices) and is scored on the other half.
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    sm, sn = -per_example_loss(model, x_mem, y_mem), -per_example_loss(model, x_non, y_non)
    s = np.concatenate([sm, sn])
    lab = np.concatenate([np.ones(len(sm)), np.zeros(len(sn))])
    auc = roc_auc_score(lab, s)
    n1, n0 = len(sm), len(sn)
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    se = math.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0))
    # calibration half -> thresholds
    cm, cn, em, en = sm[0::2], sn[0::2], sm[1::2], sn[1::2]
    fpr, tpr, thr = roc_curve(np.r_[np.ones(len(cm)), np.zeros(len(cn))], np.r_[cm, cn])
    t_adv = thr[np.argmax(tpr - fpr)]
    ok = np.where(fpr <= 0.01)[0]
    t_lowfpr = thr[ok[-1]] if len(ok) else np.inf
    # evaluation half -> unbiased estimates at those thresholds
    def rates(t):
        return float(np.mean(em >= t)), float(np.mean(en >= t))
    tpr_a, fpr_a = rates(t_adv)
    tpr_l, fpr_l = rates(t_lowfpr)
    return {"mia_auc": auc, "mia_auc_ci_low": auc - 1.96 * se, "mia_auc_ci_high": auc + 1.96 * se,
            "mia_advantage": tpr_a - fpr_a, "mia_tpr_at_1fpr": tpr_l, "mia_fpr_at_1fpr_threshold": fpr_l}


def dp_advantage_bound(eps, delta):
    """Max membership advantage any attacker can achieve against an (eps,delta)-DP
    algorithm (from TPR <= e^eps FPR + delta and its symmetric counterpart)."""
    if not np.isfinite(eps):
        return 1.0
    return (math.exp(eps) - 1 + 2 * delta) / (math.exp(eps) + 1)


# ---------------------------------------------------------------------------
# 5. TRAINING LOOP (private and non-private share everything else)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, x, y, bs=4000):
    model.eval()
    correct, loss = 0, 0.0
    for i in range(0, len(y), bs):
        out = model(x[i:i + bs])
        loss += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(y), loss / len(y)


def train_run(data, run):
    """
    run = {"name", "private" (bool), "sigma", "clip", "n_train", "seed"}.
    Logs one row per epoch: test acc/loss, train loss, epsilon so far, MIA
    metrics, clipping statistics, wall-clock time.
    """
    xtr_all, ytr_all, xte, yte = data
    seed = run["seed"]
    g_cpu = torch.Generator().manual_seed(seed)
    # training subset (dataset-size experiment); same subset for private & non-private
    idx = torch.randperm(len(ytr_all), generator=g_cpu)[:run["n_train"]].to(DEVICE)
    xtr, ytr = xtr_all[idx], ytr_all[idx]
    N, B = len(ytr), CFG["batch_size"]
    q = B / N
    steps_per_epoch = max(1, round(N / B))

    # members / non-members for the MIA (equal sizes, disjoint by construction)
    m = min(CFG["mia_max_members"], N)
    mem = torch.randperm(N, generator=g_cpu)[:m].to(DEVICE)
    non = torch.randperm(len(yte), generator=g_cpu)[:m].to(DEVICE)

    torch.manual_seed(seed)
    model = SmallCNN().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=CFG["lr"], momentum=CFG["momentum"])
    g_dev = torch.Generator(device=DEVICE).manual_seed(seed)     # sampling + noise RNG
    log, t0, steps, train_s = [], time.time(), 0, 0.0
    sync = torch.cuda.synchronize if DEVICE == "cuda" else (lambda: None)
    for ep in range(1, run.get("epochs", CFG["epochs"]) + 1):
        model.train()
        tr_loss, seen, norm_stats = 0.0, 0, []
        if not run["private"]:
            perm = torch.randperm(N, device=DEVICE, generator=g_dev)
        for s in range(steps_per_epoch):
            sync()
            ts = time.time()                       # training-time clock (excludes monitoring)
            if run["private"]:
                # Poisson sampling: each example independently with prob q
                mask = torch.rand(N, device=DEVICE, generator=g_dev) < q
                xb, yb = xtr[mask], ytr[mask]
                opt.zero_grad(set_to_none=True)
                norm_stats.append(dp_gradient(model, xb, yb, run["clip"], run["sigma"], B, g_dev))
                sync()
                train_s += time.time() - ts
                if len(yb):                        # monitoring only (not part of DP-SGD)
                    with torch.no_grad():
                        tr_loss += F.cross_entropy(model(xb), yb, reduction="sum").item()
                        seen += len(yb)
                ts = time.time()
            else:
                bi = perm[s * B:(s + 1) * B]
                if len(bi) == 0:
                    continue
                xb, yb = xtr[bi], ytr[bi]
                loss = F.cross_entropy(model(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                tr_loss += loss.item() * len(yb)
                seen += len(yb)
            opt.step()
            sync()
            train_s += time.time() - ts
            steps += 1
        acc, te_loss = evaluate(model, xte, yte)
        eps = (epsilon_from_rdp(q, run["sigma"], steps, CFG["delta"])[0]
               if run["private"] else math.inf)
        mia = membership_inference(model, xtr[mem], ytr[mem], xte[non], yte[non])
        ns = pd.DataFrame(norm_stats) if norm_stats else pd.DataFrame({"median_norm": [np.nan],
                                                                       "clipped_frac": [np.nan]})
        log.append({**run, "epoch": ep, "steps": steps, "q": q, "test_acc": acc, "test_loss": te_loss,
                    "train_loss": tr_loss / max(seen, 1),
                    "train_acc": evaluate(model, xtr[mem], ytr[mem])[0],
                    "epsilon": eps, "eps_bound_adv": dp_advantage_bound(eps, CFG["delta"]),
                    **mia, "median_grad_norm": ns.median_norm.median(),
                    "clipped_frac": ns.clipped_frac.mean(), "train_s": train_s,
                    "wall_s": time.time() - t0})
    return log


def measure_grad_norms(data):
    """Per-example gradient norms during a short NON-private training run; used
    to pick clipping norms below / above the typical norm."""
    xtr, ytr, xte, yte = data
    torch.manual_seed(0)
    model = SmallCNN().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=CFG["lr"], momentum=CFG["momentum"])
    rows = []
    perm = torch.randperm(len(ytr), device=DEVICE)
    for s in range(0, 3 * 234):
        bi = perm[(s * 256) % 59904:(s * 256) % 59904 + 256]
        if s % 39 == 0:
            g = per_example_grads(model, xtr[bi], ytr[bi])
            norms = torch.sqrt(sum(v.flatten(1).pow(2).sum(1) for v in g.values())).cpu().numpy()
            rows.append({"step": s, "p10": np.percentile(norms, 10), "median": np.median(norms),
                         "p90": np.percentile(norms, 90)})
        loss = F.cross_entropy(model(xtr[bi]), ytr[bi])
        opt.zero_grad()
        loss.backward()
        opt.step()
    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "grad_norms.csv", index=False)
    print(df.round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# 6. EXPERIMENT GRID
# ---------------------------------------------------------------------------
def experiment_list():
    runs = []
    N = 60000
    for seed in CFG["seeds"]:
        runs.append({"exp": "grid", "name": "non-private", "private": False, "sigma": 0.0,
                     "clip": 0.0, "n_train": N, "seed": seed})
        for C in CFG["clip_norms"]:
            runs.append({"exp": "grid", "name": f"clip-only C={C}", "private": True, "sigma": 0.0,
                         "clip": C, "n_train": N, "seed": seed})
            for s in CFG["noise_multipliers"]:
                runs.append({"exp": "grid", "name": f"DP s={s} C={C}", "private": True, "sigma": s,
                             "clip": C, "n_train": N, "seed": seed})
        for n in CFG["dataset_sizes"]:
            if n == N:
                continue            # the N=60k runs are already in the grid
            runs.append({"exp": "size", "name": f"non-private N={n}", "private": False, "sigma": 0.0,
                         "clip": 0.0, "n_train": n, "seed": seed})
            runs.append({"exp": "size", "name": f"DP N={n}", "private": True,
                         "sigma": CFG["size_exp_sigma"], "clip": CFG["size_exp_clip"],
                         "n_train": n, "seed": seed})
        # C. training duration: long training on a small set (where memorisation happens)
        for priv in (False, True):
            runs.append({"exp": "duration", "name": f"{'DP' if priv else 'non-private'} long",
                         "private": priv, "sigma": CFG["size_exp_sigma"] if priv else 0.0,
                         "clip": CFG["size_exp_clip"] if priv else 0.0, "n_train": CFG["duration_n"],
                         "seed": seed, "epochs": CFG["duration_epochs"]})
    return runs


def run_all(quick=False):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    RESULTS.mkdir(parents=True, exist_ok=True)
    if quick:
        CFG.update({"epochs": 2, "noise_multipliers": [1.0], "clip_norms": [1.0],
                    "dataset_sizes": [3000, 60000], "seeds": [0], "duration_epochs": 3})
    out = RESULTS / ("epoch_logs_quick.csv" if quick else "epoch_logs.csv")
    logs = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(r["name"], int(r["n_train"]), int(r["seed"])) for r in logs}
    (RESULTS / "run_metadata.json").write_text(json.dumps(
        {"config": CFG, "torch": torch.__version__, "device": torch.cuda.get_device_name(0)}, indent=2))
    data = load_fashion_mnist()
    for run in experiment_list():
        if (run["name"], run["n_train"], run["seed"]) in done:     # resume support
            continue
        t = time.time()
        log = train_run(data, run)
        logs += log
        pd.DataFrame(logs).to_csv(out, index=False)                   # checkpoint after every run
        last = log[-1]
        print(f"{run['name']:<22} N={run['n_train']:<6} seed={run['seed']}  acc={last['test_acc']:.4f} "
              f"eps={last['epsilon']:.2f}  MIA-AUC={last['mia_auc']:.3f}  ({time.time() - t:.0f}s)",
              flush=True)
    summarize(out)


def final_rows(logs):
    """Last logged epoch of every run (runs may have different lengths)."""
    return logs.loc[logs.groupby(["exp", "name", "n_train", "seed"]).epoch.idxmax()]


def summarize(path=RESULTS / "epoch_logs.csv"):
    logs = pd.read_csv(path)
    final = final_rows(logs)
    summ = (final.groupby(["exp", "name", "sigma", "clip", "n_train"])
            .agg(test_acc=("test_acc", "mean"), test_acc_sd=("test_acc", "std"),
                 test_loss=("test_loss", "mean"), train_acc=("train_acc", "mean"),
                 epsilon=("epsilon", "mean"), mia_auc=("mia_auc", "mean"),
                 mia_advantage=("mia_advantage", "mean"), mia_tpr_at_1fpr=("mia_tpr_at_1fpr", "mean"),
                 eps_bound_adv=("eps_bound_adv", "mean"), clipped_frac=("clipped_frac", "mean"),
                 mia_auc_ci_low=("mia_auc_ci_low", "mean"), mia_auc_ci_high=("mia_auc_ci_high", "mean"),
                 train_s=("train_s", "mean"), wall_s=("wall_s", "mean"), n_seeds=("seed", "nunique"))
            .reset_index().sort_values(["exp", "n_train", "clip", "sigma"]))
    summ["gen_gap"] = summ.train_acc - summ.test_acc
    summ.to_csv(RESULTS / "summary.csv", index=False)
    with pd.option_context("display.width", 250):
        print(summ.round(4).to_string(index=False))


# ---------------------------------------------------------------------------
# 7. FIGURES
# ---------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=11)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # larger default text so figures stay readable when scaled to A4 width
    plt.rcParams.update({"font.size": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "axes.labelsize": 12, "legend.fontsize": 11})
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    logs = pd.read_csv(RESULTS / "epoch_logs.csv")
    grid = logs[logs.exp == "grid"]
    final = final_rows(grid)
    nonpriv = final[~final.private].test_acc.mean()
    clips = sorted(c for c in final["clip"].unique() if c > 0)

    # Fig 1: test accuracy vs achieved epsilon (final epoch), one line per clipping norm
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for j, C in enumerate(clips):
        d = final[(final["clip"] == C) & (final.sigma > 0)].groupby("sigma").agg(
            eps=("epsilon", "mean"), acc=("test_acc", "mean"), sd=("test_acc", "std")).reset_index()
        ax.errorbar(d.eps, d.acc, yerr=d.sd, color=SERIES[j], marker="o", ms=6, lw=2, capsize=0,
                    label=f"DP-SGD, C = {C}")
        if j == 0:          # epsilon depends only on sigma, so label each sigma once
            for _, r in d.iterrows():
                ax.annotate(f"σ={r.sigma:g}", (r.eps, r.acc), textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=11.2, color=INK2)
        co = final[(final["clip"] == C) & (final.sigma == 0)].test_acc.mean()
        ax.axhline(co, color=SERIES[j], ls=":", lw=1.2, label=f"clip only (σ=0, ε=∞), C = {C}")
    ax.axhline(nonpriv, color=INK, ls="--", lw=1.2, label="non-private baseline")
    ax.set_xscale("log")
    ax.set_xlabel("achieved privacy budget ε  (δ = 1e-5, log scale; smaller = more private)",
                  fontsize=12.8, color=INK2)
    ax.set_ylabel("test accuracy (mean ± sd)", fontsize=12.8, color=INK2)
    ax.set_title(f"Privacy–utility trade-off after {CFG['epochs']} epochs (Fashion-MNIST)",
                 fontsize=14, color=INK)
    ax.legend(fontsize=11.2, frameon=False, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_accuracy_vs_epsilon.png", dpi=200)
    plt.close(fig)

    # Fig 2: convergence (test accuracy per epoch) for each sigma, one panel per C
    fig, axes = plt.subplots(1, len(clips), figsize=(4.4 * len(clips), 4.6), sharey=True)
    for ax, C in zip(np.atleast_1d(axes), clips):
        np_curve = grid[~grid.private].groupby("epoch").test_acc.mean()
        ax.plot(np_curve.index, np_curve.values, color=INK, lw=2, ls="--", label="non-private")
        co = grid[(grid["clip"] == C) & (grid.sigma == 0) & grid.private].groupby("epoch").test_acc.mean()
        ax.plot(co.index, co.values, color=INK2, lw=1.5, ls=":", label="clip only (σ=0)")
        for j, s in enumerate(sorted(grid[grid.sigma > 0].sigma.unique())):
            c = grid[(grid["clip"] == C) & (grid.sigma == s)].groupby("epoch").test_acc.mean()
            ax.plot(c.index, c.values, color=SERIES[j], lw=2, label=f"σ = {s:g}")
        ax.set_title(f"Convergence, clipping norm C = {C}", fontsize=14, color=INK)
        ax.set_xlabel("epoch", fontsize=12.8, color=INK2)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("test accuracy", fontsize=12.8, color=INK2)
    np.atleast_1d(axes)[-1].legend(fontsize=11.2, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_convergence.png", dpi=200)
    plt.close(fig)

    # Fig 3: epsilon grows with training duration (theory) for each sigma
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for j, s in enumerate(sorted(grid[grid.sigma > 0].sigma.unique())):
        c = grid[(grid.sigma == s) & (grid["clip"] == clips[0])].groupby("epoch").epsilon.mean()
        ax.plot(c.index, c.values, color=SERIES[j], lw=2, label=f"σ = {s:g}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch (training duration)", fontsize=12.8, color=INK2)
    ax.set_ylabel("ε spent so far (δ = 1e-5, log scale)", fontsize=12.8, color=INK2)
    ax.set_title("Privacy budget consumed vs training duration (N = 60k, B = 256)", fontsize=14, color=INK)
    ax.legend(fontsize=11.2, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_epsilon_vs_epochs.png", dpi=200)
    plt.close(fig)

    # Fig 4: MIA advantage vs epsilon, with the theoretical DP upper bound
    fig, ax = plt.subplots(figsize=(7, 4.2))
    allfin = final_rows(logs[logs.exp != "duration"])
    e = np.logspace(-1, np.log10(max(200, allfin[np.isfinite(allfin.epsilon)].epsilon.max() * 1.2)), 200)
    ax.plot(e, [dp_advantage_bound(x, CFG["delta"]) for x in e], color=INK2, lw=1.5, ls="--",
            label="upper bound implied by (ε, δ)-DP")
    dpf = allfin[allfin.private & (allfin.sigma > 0)]
    for j, (lab, d) in enumerate([("DP-SGD, N = 60k", dpf[dpf.n_train == 60000]),
                                  ("DP-SGD, smaller N", dpf[dpf.n_train < 60000])]):
        ax.scatter(d.epsilon, d.mia_advantage, color=SERIES[j], s=36, edgecolor="white", lw=1, label=lab)
    np_f = allfin[~allfin.private]
    ax.scatter(np.full(len(np_f), e[-1]), np_f.mia_advantage, color=INK, marker="x", s=36,
               label="non-private (ε = ∞), all N")
    ax.set_xscale("log")
    ax.set_xlabel("ε (log scale)", fontsize=12.8, color=INK2)
    ax.set_ylabel("membership advantage (max TPR − FPR)", fontsize=12.8, color=INK2)
    ax.set_title("Loss-threshold membership inference vs privacy budget", fontsize=14, color=INK)
    ax.legend(fontsize=11.2, frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_mia_vs_epsilon.png", dpi=200)
    plt.close(fig)

    # Fig 5: dataset-size effect (accuracy, epsilon, MIA) private vs non-private
    size = logs[(logs.n_train.isin(CFG["dataset_sizes"])) & (logs.exp != "duration") &
                (logs.epoch == CFG["epochs"]) &
                ((logs.exp == "size") | ((logs.exp == "grid") &
                                         ((~logs.private) | ((logs.sigma == CFG["size_exp_sigma"]) &
                                                             (logs["clip"] == CFG["size_exp_clip"])))))]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for j, (priv, lab) in enumerate([(False, "non-private"),
                                     (True, f"DP-SGD σ={CFG['size_exp_sigma']:g}, C={CFG['size_exp_clip']:g}")]):
        d = size[size.private == priv].groupby("n_train").agg(acc=("test_acc", "mean"), eps=("epsilon", "mean"),
                                                             mia=("mia_auc", "mean")).reset_index()
        axes[0].plot(d.n_train, d.acc, marker="o", color=SERIES[j] if priv else INK, lw=2, label=lab)
        axes[2].plot(d.n_train, d.mia, marker="o", color=SERIES[j] if priv else INK, lw=2, label=lab)
        if priv:
            axes[1].plot(d.n_train, d.eps, marker="o", color=SERIES[j], lw=2, label=lab)
    for ax, t, yl in zip(axes, ["Test accuracy", "Privacy budget ε (DP run)", "MIA ROC-AUC (0.5 = chance)"],
                         ["test accuracy", "ε (δ = 1e-5)", "attack AUC"]):
        ax.set_xscale("log")
        ax.set_title(t, fontsize=14, color=INK)
        ax.set_xlabel("training-set size N (log scale)", fontsize=12.8, color=INK2)
        ax.set_ylabel(yl, fontsize=12.8, color=INK2)
        _style(ax)
    axes[2].axhline(0.5, color=INK2, ls=":", lw=1)
    axes[0].legend(fontsize=11.2, frameon=False)
    fig.suptitle(f"Effect of dataset size ({CFG['epochs']} epochs each)", fontsize=14, color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_dataset_size.png", dpi=200)
    plt.close(fig)

    # Fig 6: training duration -- generalisation gap, MIA and epsilon over epochs
    dur = logs[logs.exp == "duration"]
    if len(dur):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
        for j, priv in enumerate([False, True]):
            d = dur[dur.private == priv].groupby("epoch").agg(
                tr=("train_acc", "mean"), te=("test_acc", "mean"), auc=("mia_auc", "mean"),
                auc_sd=("mia_auc", "std"), eps=("epsilon", "mean"))
            col = SERIES[0] if priv else INK
            lab = (f"DP-SGD σ={CFG['size_exp_sigma']:g}, C={CFG['size_exp_clip']:g}" if priv
                   else "non-private")
            axes[0].plot(d.index, d.tr, color=col, lw=2, label=f"{lab}: train (members)")
            axes[0].plot(d.index, d.te, color=col, lw=2, ls="--", label=f"{lab}: test")
            axes[1].plot(d.index, d.auc, color=col, lw=2, label=lab)
            axes[1].fill_between(d.index, d.auc - d.auc_sd, d.auc + d.auc_sd, color=col, alpha=0.15)
            if priv:
                axes[2].plot(d.index, d.eps, color=col, lw=2, label=lab)
        axes[1].axhline(0.5, color=INK2, ls=":", lw=1)
        for ax, t, yl in zip(axes, ["Train vs test accuracy", "MIA ROC-AUC over training",
                                    "ε spent (DP run)"], ["accuracy", "attack AUC", "ε (δ = 1e-5)"]):
            ax.set_title(t, fontsize=14, color=INK)
            ax.set_xlabel("epoch", fontsize=12.8, color=INK2)
            ax.set_ylabel(yl, fontsize=12.8, color=INK2)
            _style(ax)
        axes[0].legend(fontsize=10.4, frameon=False)
        axes[1].legend(fontsize=11.2, frameon=False)
        fig.suptitle(f"Effect of training duration (N = {CFG['duration_n']}, "
                     f"{CFG['duration_epochs']} epochs)", fontsize=14, color=INK)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig6_training_duration.png", dpi=200)
        plt.close(fig)
    print(f"figures -> {FIG_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["gradnorms", "accountant-check", "run", "summarize", "plot"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.stage == "gradnorms":
        measure_grad_norms(load_fashion_mnist())
    elif a.stage == "accountant-check":
        accountant_check()
    elif a.stage == "run":
        run_all(quick=a.quick)
        if not a.quick:
            plot()
    elif a.stage == "summarize":
        summarize()
    else:
        plot()


if __name__ == "__main__":
    main()
