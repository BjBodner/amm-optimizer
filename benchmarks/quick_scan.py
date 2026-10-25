"""Lightweight COCO/bbob scan for quick A/B experiments (minutes, not hours).

    # run a solver at several budgets, 12 processes, resumable
    python benchmarks/quick_scan.py run --solver atm --seeds 4 --jobs 12 \
        --out benchmarks/results/scan/E08/base.json.gz
    # an ATMConfig override, dotted paths follow the config tree
    python benchmarks/quick_scan.py run --solver atm \
        --config "variant.amplitude_control=success_rate" --out x.json.gz
    # paired comparison with cluster bootstrap (the first file is the baseline)
    python benchmarks/quick_scan.py compare base.json.gz arm.json.gz

What it measures. For each (function, dimension, instance, seed, budget) the solver
runs once with ``budget = multiplier * D`` evaluations (restarted while budget
remains, as in ``run_bbob.py``, which supplies the box, penalty, budget cut-off and
restart loop) and the *precision* ``best f - f_opt`` is recorded. The score is the
COCO-style ECDF value: the fraction of (problem, target) pairs reached, over 51
targets 10^2 ... 10^-8 (five per decade). It is a fixed-budget cut of the ECDF,
cheap and good enough for screening; confirm survivors with ``run_bbob.py`` +
``cocopp``. The primary metric of ``docs/EXPERIMENTS.md`` is the mean over 30, 100
and 300 x D.

Pairing. Seeds come from the problem identity, never from the position in the
selected suite: ``SeedSequence([ns, seed, function, dimension, instance])`` spawns
one stream for the start points (including restart starts) and one for the
optimizer, so a variant run on any subset of a design is paired with a baseline
run on another subset. Budgets of one problem share their seed.

Solvers: ``atm`` (defaults plus ``--config``), ``atm_legacy`` (original
accounting), ``random`` (uniform), ``es11`` ((1+1)-ES with the 1/5 rule), ``cma``
(pycma), ``powell`` and ``nelder_mead`` (SciPy, restarted from random points). The
last three are reference anchors, not candidates. The anchors committed in
``benchmarks/results/quick_scan/`` were made by version 1 of this tool (seeds from
suite order): they are valid as reference levels, not as paired baselines.

Instances 11-15 are the locked Tier 3 set and are refused without ``--locked``.
"""

from __future__ import annotations

import os

# Forced, not defaulted: OpenBLAS with several threads splits its reductions
# differently and changes the last bits of SLSQP's steps, so an exported
# OPENBLAS_NUM_THREADS would silently change traces (as in the recorder).
BLAS_THREADS = "1"
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = BLAS_THREADS

import argparse  # noqa: E402
import contextlib  # noqa: E402
import difflib  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # this checkout, not an installed copy

import run_bbob  # noqa: E402
from process_pool import RetryingPool  # noqa: E402

FOPT_CACHE = HERE / "results" / "fopt_cache.json"
FOPT_DECIMALS = 2  # bbob optimal values are rounded to 0.01
# Until 2026-10-19 the cache was built as f(0) - (f(0) - f_opt) from a data line with
# 10 significant digits, which put f12's f_opt (f(0) of 1e8 to 1.6e9) on the wrong
# hundredth for 291 problems (D 10-40). The table maps each corrected id to [the
# value runs were scored against, the exact value] (``correct_fopt_precision``).
FOPT_CORRECTIONS = HERE / "results" / "fopt_corrections_2026-10-19.json"
FOPT_HEADER = re.compile(r"Fopt \(([-+0-9.eE]+)\)")
FOPT_BASIS = f"exact from the COCO data header, rounded to {FOPT_DECIMALS} decimals"
TARGETS = 10.0 ** np.arange(2, -8.01, -0.2)  # 51 targets, 100 ... 1e-8
GROUPS = {
    "separable f1-5": range(1, 6),
    "low-cond f6-9": range(6, 10),
    "high-cond f10-14": range(10, 15),
    "multimodal f15-19": range(15, 20),
    "weak-struct f20-24": range(20, 25),
}
PRIMARY = (30.0, 100.0, 300.0)  # primary metric: mean over these multipliers
GROUP_LOSS = 0.02  # acceptance rule (b): no group loses more than this
LOCKED_INSTANCES = range(11, 16)  # of the 2019 suite only
SEED_NAMESPACE = 0x41544D  # "ATM"
ID_PATTERN = re.compile(r"bbob_f(\d+)_i(\d+)_d(\d+)")
BOX = (-5.0, 5.0)


def parse_id(problem_id: str) -> tuple[int, int, int]:
    """``(function, instance, dimension)`` of a bbob problem id."""
    match = ID_PATTERN.fullmatch(problem_id)
    if match is None:
        raise ValueError(f"not a bbob problem id: {problem_id!r}")
    function, instance, dimension = map(int, match.groups())
    return function, instance, dimension


class ProblemProxy:
    """The few attributes ``run_bbob.solve_problem`` reads, with our own tracking.

    No observer is attached, so COCO does not keep the best value; the proxy does,
    and reports the final target (``f_opt + 1e-8``) as hit like COCO would. ``hits``
    holds the first evaluation that reached each of ``TARGETS`` (-1: not reached),
    so one run gives its score at every smaller budget (``score_by``), as
    ``mabbob.Proxy`` does; the values the solver sees are unchanged.
    """

    def __init__(self, problem, fopt: float, scale: float = 1.0) -> None:
        self._problem = problem
        self._fopt = fopt
        self._scale = scale
        self.dimension = problem.dimension
        self.lower_bounds = scale * np.asarray(problem.lower_bounds, dtype=float)
        self.upper_bounds = scale * np.asarray(problem.upper_bounds, dtype=float)
        self.evaluations = 0
        self.best_observed_fvalue1 = float("inf")
        self.hits = [-1] * len(TARGETS)
        self._next = 0  # the first target not reached yet (targets descend)

    @property
    def final_target_hit(self) -> bool:
        return self.best_observed_fvalue1 - self._fopt <= 1e-8

    def __call__(self, x):
        # ``--box-scale s``: the solver sees y = s x, the box [-5 s, 5 s].
        value = float(
            self._problem(x if self._scale == 1.0 else np.asarray(x) / self._scale)
        )
        self.evaluations += 1
        if value < self.best_observed_fvalue1:
            self.best_observed_fvalue1 = value
            precision = value - self._fopt
            while self._next < len(TARGETS) and precision <= TARGETS[self._next]:
                self.hits[self._next] = self.evaluations
                self._next += 1
        return value


def score_by(hits: list[int], evaluations: int) -> float:
    """The share of ``TARGETS`` a run with these ``hits`` had reached by its
    ``evaluations``-th evaluation (``record_score`` at that budget)."""
    return float(np.mean([0 < h <= evaluations for h in hits]))


# ---------------------------------------------------------------- solvers ----
class _Base:
    """Solver protocol used by ``run_bbob.solve_problem``.

    ``begin`` receives a ``SeedSequence`` and spawns two independent streams: the
    start points (``uniform``) and the optimizer (``rng``).
    """

    name = "base"
    # ``--box-scale``: the box the solver is told about is ``BOX`` times this.
    box_scale = 1.0
    # "uniform"; "center": the first start is the box centre; "coco2019": the
    # protocol of the 2019 GECCO submission (first start the centre, restarts
    # uniform in the middle 80% of the box).
    start = "uniform"

    def begin(self, seed) -> None:
        start_seed, optimizer_seed = seed.spawn(2)
        self.start_rng = np.random.RandomState(start_seed.generate_state(4))
        self.rng = np.random.RandomState(optimizer_seed.generate_state(4))
        self._first_start = True

    def uniform(self, dim: int) -> np.ndarray:
        first, self._first_start = self._first_start, False
        if first and self.start in ("center", "coco2019"):
            return np.full(dim, 0.5)
        draw = self.start_rng.rand(dim)
        if self.start == "coco2019":
            return 0.5 + 0.8 * (draw - 0.5)
        return draw


class RandomSolver(_Base):
    name = "random"

    def solve(self, fun, x0, budget):
        fun(x0)
        for _ in range(budget - 1):
            fun(self.rng.uniform(-5, 5, len(x0)))


class ES11Solver(_Base):
    """(1+1)-ES with the 1/5th success rule, sigma0 = 2 (a fifth of the box)."""

    name = "es11"

    def solve(self, fun, x0, budget):
        x, fx, sigma = x0.copy(), fun(x0), 2.0
        for _ in range(budget - 1):
            y = x + sigma * self.rng.randn(len(x))
            fy = fun(y)
            if fy <= fx:
                x, fx = y, fy
                sigma *= np.exp(0.8 / len(x))  # success: exp(1/d * 0.8)
            else:
                sigma *= np.exp(-0.2 / len(x))  # failure: exp(-1/d * 0.2)
            if sigma < 1e-12:
                return


class _Stagnation:
    """Stop rule of the restarted anchors: sigma collapsed, or no progress.

    True once the best value has improved by at most 1e-11 over the last
    ``100 + 50 * dim`` evaluations, or sigma fell below ``1e-12 * sigma0``.
    """

    def __init__(self, dim: int, sigma0: float) -> None:
        self.window = 100 + 50 * dim
        self.sigma_floor = 1e-12 * sigma0
        self.history: list[float] = []

    def stop(self, best: float, sigma: float) -> bool:
        self.history.append(best)
        if sigma < self.sigma_floor:
            return True
        if len(self.history) <= self.window:
            return False
        return self.history[-self.window - 1] - best <= 1e-11


class ES11RestartSolver(_Base):
    """(1+1)-ES with the 1/5th rule, restarted on stagnation (``_Stagnation``)."""

    name = "es11r"

    def solve(self, fun, x0, budget):
        x, fx, sigma = x0.copy(), fun(x0), 2.0
        stop = _Stagnation(len(x), sigma)
        for _ in range(budget - 1):
            y = x + sigma * self.rng.randn(len(x))
            fy = fun(y)
            if fy <= fx:
                x, fx = y, fy
                sigma *= np.exp(0.8 / len(x))
            else:
                sigma *= np.exp(-0.2 / len(x))
            if stop.stop(fx, sigma):
                return


class CMA11Solver(_Base):
    """(1+1)-Cholesky-CMA-ES (Igel, Suttorp and Hansen 2006), restarted on stagnation.

    The covariance is kept as its Cholesky factor ``A`` (and its inverse), updated in
    O(dim^2) per evaluation after every success.
    """

    name = "cma11r"

    def solve(self, fun, x0, budget):
        n = len(x0)
        d = 1 + n / 2
        p_target, c_p, c_c = 2 / 11, 1 / 12, 2 / (n + 2)
        c_cov, p_thresh = 2 / (n**2 + 6), 0.44
        x, fx, sigma = x0.copy(), fun(x0), 2.0
        a, a_inv = np.eye(n), np.eye(n)
        p_c, p_succ = np.zeros(n), p_target
        stop = _Stagnation(n, sigma)
        for _ in range(budget - 1):
            z = self.rng.randn(n)
            step = a @ z
            y = x + sigma * step
            fy = fun(y)
            success = fy <= fx
            p_succ = (1 - c_p) * p_succ + c_p * success
            sigma *= np.exp((p_succ - p_target) / (d * (1 - p_target)))
            if success:
                x, fx = y, fy
                if p_succ < p_thresh:
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * step
                    alpha = 1 - c_cov
                else:
                    p_c = (1 - c_c) * p_c
                    alpha = 1 - c_cov + c_cov * c_c * (2 - c_c)
                a, a_inv = _cholesky_update(a, a_inv, p_c, alpha, c_cov)
            if stop.stop(fx, sigma):
                return


def _cholesky_update(a, a_inv, v, alpha, beta):
    """Factors of ``alpha * A A^T + beta * v v^T`` (Igel et al. 2006, eq. 13-14)."""
    w = a_inv @ v
    norm2 = float(w @ w)
    if norm2 <= 0 or not np.isfinite(norm2):
        return a, a_inv
    root_a = np.sqrt(alpha)
    root = np.sqrt(1 + beta / alpha * norm2)
    a_new = root_a * a + root_a / norm2 * (root - 1) * np.outer(v, w)
    a_inv_new = a_inv / root_a - (1 - 1 / root) / (root_a * norm2) * np.outer(
        w, w @ a_inv
    )
    return a_new, a_inv_new


class CMASolver(_Base):
    name = "cma"

    def solve(self, fun, x0, budget):
        import cma

        opts = {
            "maxfevals": budget,
            "verbose": -9,
            "seed": int(self.rng.randint(1, 2**31 - 1)),
            "bounds": [-5, 5],
        }
        with contextlib.suppress(run_bbob.StopRunError):
            cma.fmin2(fun, x0, 2.0, opts)


class CMAIpopSolver(CMASolver):
    """pycma with IPOP: the population doubles at every restart of the harness.

    ``run_bbob.solve_problem`` restarts the solver from a new uniform point after
    each termination of pycma, so the first run uses the default population and
    every later one twice the previous.
    """

    name = "cma_ipop"

    def begin(self, seed) -> None:
        super().begin(seed)
        self._popsize = 0

    def solve(self, fun, x0, budget):
        import cma

        dim = len(x0)
        self._popsize = 2 * self._popsize or 4 + int(3 * np.log(dim))
        opts = {
            "maxfevals": budget,
            "verbose": -9,
            "seed": int(self.rng.randint(1, 2**31 - 1)),
            "bounds": [-5, 5],
            "popsize": self._popsize,
        }
        with contextlib.suppress(run_bbob.StopRunError):
            cma.fmin2(fun, x0, 2.0, opts)


class CMARestartSolver(CMASolver):
    """pycma 4.5.0 restart schemes run inside pycma (state-of-the-art references).

    ``mode="bipop"``: BIPOP-aCMA-ES, ``cma.fmin2(..., restarts=9, bipop=True)``;
    ``mode="lq"``: lq-CMA-ES, ``cma.fmin_lq_surr2(..., restarts=9)`` (IPOP restarts
    with the linear-quadratic surrogate). Active CMA is pycma's default. The first
    start is the harness's x0, later ones are uniform in [-4, 4]^D (COCO's
    convention); ``maxfevals`` counts across pycma's restarts.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.name = f"cma_{mode}"

    def solve(self, fun, x0, budget):
        import cma

        dim = len(x0)
        first = [True]

        def start():
            if first[0]:
                first[0] = False
                return np.array(x0, dtype=float)
            return self.rng.uniform(-4, 4, dim)

        opts = {
            "maxfevals": budget,
            "verbose": -9,
            "seed": int(self.rng.randint(1, 2**31 - 1)),
            "bounds": [-5, 5],
        }
        with contextlib.suppress(run_bbob.StopRunError):
            if self.mode == "bipop":
                cma.fmin2(fun, start, 2.0, opts, restarts=9, bipop=True)
            else:
                cma.fmin_lq_surr2(fun, start, 2.0, opts, restarts=9)


class PopCMAIpopSolver(_Base):
    """``atm.pop_cma``'s IPOP restarts on their own (the audit's Batch 11c check).

    The in-house (mu/mu_w, lambda)-CMA-ES that Batches 9-10 used, run like
    ``cma_ipop``: sigma0 = 2 (a fifth of the box), the first run from the
    harness's start point, later runs from uniform points with doubling lambda.
    ``active`` adds the negative update of Batch 12's ``pop_active``.
    """

    def __init__(self, active: bool = False) -> None:
        self.name = "popcma_ipop_active" if active else "popcma_ipop"
        self._active = active

    def solve(self, fun, x0, budget):
        from amm_optimizer.bounds import Box
        from amm_optimizer.objective import CountedObjective
        from amm_optimizer.pop_cma import run_population_restarts

        dim = len(x0)
        box = Box.make((-5.0, 5.0), dim)
        objective = CountedObjective(fun, budget)
        with contextlib.suppress(run_bbob.StopRunError):
            run_population_restarts(
                objective,
                self.rng,
                np.asarray(x0, dtype=float),
                float("inf"),
                2.0,
                box,
                budget,
                True,
                active=self._active,
            )


class ScipySolver(_Base):
    def __init__(self, method: str) -> None:
        self.name = method.lower().replace("-", "_")
        self._method = method

    def solve(self, fun, x0, budget):
        from scipy.optimize import minimize

        minimize(fun, x0, method=self._method, options={"maxfev": budget})


class SLSQPSolver(_Base):
    """SciPy's SLSQP inside the box (pip-installable reference, 2026-10-18).

    Gradients by SciPy's 2-point finite differences (dim + 1 evaluations each);
    ``ftol`` is tightened so a run goes for the final targets. The harness restarts
    it from uniform points until the budget is spent.
    """

    name = "slsqp"

    def solve(self, fun, x0, budget):
        from scipy.optimize import minimize

        dim = len(x0)
        minimize(
            fun,
            x0,
            method="SLSQP",
            bounds=[(-5.0, 5.0)] * dim,
            options={"maxiter": budget, "ftol": 1e-14},
        )


class COBYQASolver(_Base):
    """SciPy's COBYQA inside the box (pip-installable reference, 2026-10-19).

    Powell-style derivative-free trust-region SQP on quadratic models built from
    2 dim + 1 interpolation points (Ragonneau & Zhang; SciPy >= 1.14), the family
    HCMA uses (NEWUOA) for its first evaluations. Trust radius 1 to 1e-10 (the
    default stop at 1e-6 floors the final targets); restarted by the harness.
    """

    name = "cobyqa"

    def solve(self, fun, x0, budget):
        from scipy.optimize import minimize

        dim = len(x0)
        minimize(
            fun,
            x0,
            method="COBYQA",
            bounds=[(-5.0, 5.0)] * dim,
            options={
                "maxfev": budget,
                "maxiter": 10 * budget,
                "initial_tr_radius": 1.0,
                "final_tr_radius": 1e-10,
            },
        )


class COBYLASolver(_Base):
    """SciPy's COBYLA (PRIMA) inside the box, linear models from dim + 1 points
    (nevergrad's NGOpt picks Cobyla at small budgets). rhobeg 1, tol 1e-10."""

    name = "cobyla"

    def solve(self, fun, x0, budget):
        from scipy.optimize import minimize

        dim = len(x0)
        minimize(
            fun,
            x0,
            method="COBYLA",
            bounds=[(-5.0, 5.0)] * dim,
            tol=1e-10,
            options={"maxiter": budget, "rhobeg": 1.0},
        )


class SLSQPStepSolver(SLSQPSolver):
    """``slsqp`` with a forward-difference step of 1e-9 instead of SciPy's 1.49e-8
    (2026-10-19, detector report: the default step's truncation bias sets a floor near
    1e-5 on the discus f11)."""

    name = "slsqp_h9"

    def solve(self, fun, x0, budget):
        from scipy.optimize import minimize

        dim = len(x0)
        minimize(
            fun,
            x0,
            method="SLSQP",
            bounds=[(-5.0, 5.0)] * dim,
            options={"maxiter": budget, "ftol": 1e-14, "eps": 1e-9},
        )


class LBFGSBSolver(_Base):
    """SciPy's L-BFGS-B inside the box, finite-difference gradients, tight tolerances.

    A cycle 11 calibration reference next to ``slsqp``; restarted by the harness.
    """

    name = "lbfgsb"

    def solve(self, fun, x0, budget):
        from scipy.optimize import minimize

        dim = len(x0)
        minimize(
            fun,
            x0,
            method="L-BFGS-B",
            bounds=[(-5.0, 5.0)] * dim,
            options={"maxfun": budget, "maxiter": budget, "ftol": 1e-15, "gtol": 1e-12},
        )


class DESolver(_Base):
    """SciPy's ``differential_evolution`` with its defaults (pip-installable reference).

    best1bin, population 15 x dim, Latin hypercube start with the harness's start
    point included, L-BFGS-B polish; ``tol=0`` so a run lasts until the budget.
    """

    name = "de"

    def solve(self, fun, x0, budget):
        from scipy.optimize import differential_evolution

        dim = len(x0)
        differential_evolution(
            fun,
            [(-5.0, 5.0)] * dim,
            x0=x0,
            tol=0.0,
            atol=0.0,
            maxiter=budget // (15 * dim) + 1,
            seed=int(self.rng.randint(1, 2**31 - 1)),
        )


class NGOptSolver(_Base):
    """nevergrad's ``NGOpt`` wizard (pip-installable reference, nevergrad 1.0.12).

    An array parameter bounded to the box, initialised at the harness's start point,
    with the remaining budget as NGOpt's budget (it picks its algorithm from budget,
    dimension and bounds), one worker.
    """

    name = "ngopt"

    @staticmethod
    def _patch_metamodel() -> None:
        """Fix nevergrad 1.0.12's MetaModel for numpy >= 2.5 (in memory only).

        ``learn_on_k_best`` calls ``float()`` on a one-element 1-D prediction, which
        numpy 2.5 rejects (TypeError), so every NGOpt run that reaches its
        surrogate step crashed. The function is recompiled from its own source with
        ``float(np.ravel(...)[0])``; nothing else changes.
        """
        import inspect

        import nevergrad.optimization.metamodel as mm

        if getattr(mm, "_atm_patched", False):
            return
        source = inspect.getsource(mm.learn_on_k_best)
        old = (
            "return float(model.predict(trans("
            "np.asarray(x, dtype=X[0].dtype).flatten()[None, :])))"
        )
        if old in source:
            new = (
                "return float(np.ravel(model.predict(trans("
                "np.asarray(x, dtype=X[0].dtype).flatten()[None, :])))[0])"
            )
            exec(compile(source.replace(old, new), mm.__file__, "exec"), mm.__dict__)  # noqa: S102
            import nevergrad.optimization.optimizerlib as lib

            lib.learn_on_k_best = mm.learn_on_k_best  # its own imported reference
        mm._atm_patched = True

    def solve(self, fun, x0, budget):
        import nevergrad as ng

        self._patch_metamodel()
        param = ng.p.Array(init=np.asarray(x0, dtype=float)).set_bounds(-5.0, 5.0)
        param.random_state = np.random.RandomState(int(self.rng.randint(1, 2**31 - 1)))
        optimizer = ng.optimizers.NGOpt(
            parametrization=param, budget=budget, num_workers=1
        )
        optimizer.minimize(lambda x: float(fun(np.asarray(x, dtype=float))))


class TPESolver(_Base):
    """optuna's ``TPESampler`` at its defaults (pip-installable reference, optuna 5.0).

    One float per coordinate in [-5, 5]; the harness's start point is enqueued as the
    first trial. ``n_trials`` is the remaining budget.
    """

    name = "tpe"

    def solve(self, fun, x0, budget):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        dim = len(x0)
        sampler = optuna.samplers.TPESampler(seed=int(self.rng.randint(1, 2**31 - 1)))
        study = optuna.create_study(sampler=sampler)
        study.enqueue_trial({f"x{i}": float(x0[i]) for i in range(dim)})

        def objective(trial):
            x = np.array([trial.suggest_float(f"x{i}", -5.0, 5.0) for i in range(dim)])
            return float(fun(x))

        study.optimize(objective, n_trials=budget, catch=())


class NADESolver(_Base):
    """NeighborhoodAdaptiveDE (NADE), the GECCO 2025 MA-BBOB competition winner.

    Transcribed line by line from ``NeighborhoodAdaptiveDE.py`` of Zenodo record
    15074959 (van Stein, LLaMEA with Gemini-2; CC-BY-4.0), read on 2026-10-21 and
    not executed (it uses ``np.Inf``, removed in numpy 2). Changes: the harness's
    ``rng`` replaces the global ``np.random`` stream (same calls, same order), the
    budget is the harness's, and the run state is fresh at every call. Like the
    original it ignores ``x0``: its population is uniform in the box. Population 50,
    neighbourhood 5, F 0.5 and CR 0.9 adapted every 50 evaluations with rate 0.1,
    restart after 1000 evaluations without improvement.
    """

    name = "nade"
    pop_size, adapt_freq, stagnation_threshold = 50, 50, 1000
    learning_rate, neighborhood_size = 0.1, 5

    def solve(self, fun, x0, budget):
        rng, n = self.rng, self.pop_size
        dim = len(x0)
        lb, ub = np.full(dim, -5.0), np.full(dim, 5.0)
        f, cr = 0.5, 0.9
        success_f: list[float] = []
        success_cr: list[float] = []
        population = rng.uniform(lb, ub, size=(n, dim))
        fitness = np.array([fun(x) for x in population])
        evals = n
        last_improvement = 0
        f_opt = fitness[np.argmin(fitness)]
        while evals < budget:
            for i in range(n):
                neighbors = rng.choice(
                    np.arange(n), self.neighborhood_size, replace=False
                )
                best_neighbor = neighbors[np.argmin(fitness[neighbors])]
                x1, x2 = population[rng.choice(np.arange(n), 2, replace=False)]
                v = np.clip(population[best_neighbor] + f * (x1 - x2), lb, ub)
                u = np.copy(population[i])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < cr or j == j_rand:
                        u[j] = v[j]
                f_new = fun(u)
                evals += 1
                if f_new < fitness[i]:
                    success_f.append(f)
                    success_cr.append(cr)
                    fitness[i] = f_new
                    population[i] = u
                    if f_new < f_opt:
                        f_opt = f_new
                        last_improvement = evals
            if evals % self.adapt_freq == 0:
                if success_f:
                    rate = self.learning_rate
                    f = (1 - rate) * f + rate * np.mean(success_f)
                    cr = (1 - rate) * cr + rate * np.mean(success_cr)
                f = np.clip(f, 0.1, 0.9)
                cr = np.clip(cr, 0.1, 1.0)
                success_f, success_cr = [], []
            if evals - last_improvement > self.stagnation_threshold:
                population = rng.uniform(lb, ub, size=(n, dim))
                fitness = np.array([fun(x) for x in population])
                evals += n
                f_opt = fitness[np.argmin(fitness)]
                last_improvement = evals
            if evals >= budget:
                break


class ATMSolver(_Base):
    """ATM with the optimizer stream separate from the start points."""

    def __init__(self, name: str, config) -> None:
        import amm_optimizer as atm

        self.name = name
        self.config = config
        self._minimize = atm.api.minimize

    def solve(self, fun, x0, budget):
        box = (BOX[0] * self.box_scale, BOX[1] * self.box_scale)
        self._minimize(fun, x0, budget, rng=self.rng, config=self.config, bounds=box)


# ----------------------------------------------------------------- config ----
def _cast(raw: str, default, key: str):
    """``raw`` as the type of ``default`` (``;`` separates tuple items)."""
    try:
        if isinstance(default, bool):
            lowered = raw.lower()
            if lowered not in ("true", "false"):
                raise ValueError("expected true or false")
            return lowered == "true"
        if isinstance(default, int):
            number = float(raw)
            if not number.is_integer():
                raise ValueError("expected an integer")
            return int(number)
        if isinstance(default, float):
            return float(raw)
        if isinstance(default, (tuple, list)):
            sample = default[0] if default else 0.0
            return tuple(_cast(part, sample, key) for part in raw.split(";"))
    except ValueError as exc:
        raise SystemExit(f"--config {key}: cannot read {raw!r}: {exc}") from exc
    return raw


def parse_config(text: str):
    """``"variant.bounds=fold,conductor.mass=5"`` (or a JSON object) -> ATMConfig.

    The paths follow ``ATMConfig.to_dict()``; values are cast to the type of the
    default. Unknown keys exit with the valid ones and a close match. A first item
    ``preset:NAME`` starts from ``atm.presets.get(NAME)`` instead of the defaults
    (``"preset:c26,variant.cma_local_nonsep=qn6"``).
    """
    import amm_optimizer as atm
    from amm_optimizer import presets

    data = atm.ATMConfig().to_dict()
    text = text.strip()
    if text.startswith("{"):
        pairs = _flatten(json.loads(text))
    else:
        pairs = []
        items = [t.strip() for t in text.split(",") if t.strip()]
        for index, item in enumerate(items):
            if item.startswith("preset:"):
                if index:
                    raise SystemExit("--config: preset:NAME must be the first item")
                try:
                    data = presets.get(item[len("preset:") :].strip()).to_dict()
                except ValueError as exc:
                    raise SystemExit(f"--config: {exc}") from exc
                continue
            key, sep, raw = item.partition("=")
            if not sep:
                raise SystemExit(f"--config item {item!r} needs the form key=value")
            pairs.append((key.strip(), raw.strip()))
    for key, raw in pairs:
        node = data
        *path, leaf = key.split(".")
        for part in path:
            if not isinstance(node.get(part), dict):
                raise SystemExit(_unknown(key, part, node))
            node = node[part]
        if leaf not in node:
            raise SystemExit(_unknown(key, leaf, node))
        node[leaf] = (
            _cast(raw, node[leaf], key)
            if isinstance(raw, str)
            else (tuple(raw) if isinstance(raw, list) else raw)
        )
    try:
        return atm.ATMConfig.from_dict(data)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"--config: {exc}") from exc


def _flatten(tree: dict, prefix: str = "") -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for key, value in tree.items():
        if isinstance(value, dict):
            out.extend(_flatten(value, f"{prefix}{key}."))
        else:
            out.append((f"{prefix}{key}", value))
    return out


def _unknown(key: str, part: str, node) -> str:
    valid = sorted(node) if isinstance(node, dict) else []
    close = difflib.get_close_matches(part, valid, n=1)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    return f"--config {key}: unknown field {part!r}.{hint} Valid here: {valid}"


def make_solver(name: str, config: str = "", start: str = "uniform"):
    if name in ("atm", "atm_legacy"):
        text = config + (",strict_budget=false" if name == "atm_legacy" else "")
        solver: _Base = ATMSolver(name, parse_config(text))
    elif name == "random":
        solver = RandomSolver()
    elif name == "es11":
        solver = ES11Solver()
    elif name == "es11r":
        solver = ES11RestartSolver()
    elif name == "cma11r":
        solver = CMA11Solver()
    elif name == "cma":
        solver = CMASolver()
    elif name == "cma_ipop":
        solver = CMAIpopSolver()
    elif name in ("cma_bipop", "cma_lq"):
        solver = CMARestartSolver(name[len("cma_") :])
    elif name in ("popcma_ipop", "popcma_ipop_active"):
        solver = PopCMAIpopSolver(active=name.endswith("active"))
    elif name == "powell":
        solver = ScipySolver("Powell")
    elif name == "nelder_mead":
        solver = ScipySolver("Nelder-Mead")
    elif name == "slsqp":
        solver = SLSQPSolver()
    elif name == "slsqp_h9":
        solver = SLSQPStepSolver()
    elif name == "de":
        solver = DESolver()
    elif name == "lbfgsb":
        solver = LBFGSBSolver()
    elif name == "cobyqa":
        solver = COBYQASolver()
    elif name == "cobyla":
        solver = COBYLASolver()
    elif name == "ngopt":
        solver = NGOptSolver()
    elif name == "tpe":
        solver = TPESolver()
    elif name == "nade":
        solver = NADESolver()
    else:
        raise SystemExit(f"unknown solver {name!r}")
    solver.start = start
    return solver


# ------------------------------------------------------------- f_opt table ----
def fopt_table(
    suite_args: tuple[str, ...], suite_instance: str = "year: 2019"
) -> dict[str, float]:
    """Exact ``f_opt`` per problem id (bbob optima are multiples of 0.01).

    The observer log the cache is made from prints too few digits (errors of
    1e-8 to 4e-6, which floored every converged run at 2e-8 and hid the final
    targets until 2026-10-17), so the cached values are rounded on reading.
    """
    raw = raw_fopt_table(suite_args, suite_instance)
    return {k: exact_fopt(v) for k, v in raw.items()}


def exact_fopt(raw: float) -> float:
    return round(raw, FOPT_DECIMALS)


def raw_fopt_table(
    suite_args: tuple[str, ...], suite_instance: str = "year: 2019"
) -> dict[str, float]:
    """``f_opt`` per problem id as read from a throw-away observer log (cached)."""
    import cocoex

    cache = read_fopt_cache()
    suite = cocoex.Suite("bbob", suite_instance, *suite_args)
    missing = [p_id for p_id in suite.ids() if p_id not in cache]
    if missing:
        here, tmp = os.getcwd(), tempfile.mkdtemp()
        os.chdir(tmp)
        try:
            observer = cocoex.Observer("bbob", "result_folder: fopt")
            for p in suite:
                if p.id not in missing:
                    continue
                p.observe_with(observer)
                p(np.zeros(p.dimension))
                p_id, dim, fid = p.id, p.dimension, p.id_function
                p.free()
                dat = Path(tmp) / "exdata" / "fopt" / f"data_f{fid}"
                text = next(dat.glob(f"*DIM{dim}.dat")).read_text()
                cache[p_id] = header_fopt(text)
        finally:
            os.chdir(here)
            shutil.rmtree(tmp, ignore_errors=True)
        write_fopt_cache(cache)
    return cache


def header_fopt(text: str) -> float:
    """The exact ``f_opt`` of the last problem in a COCO ``.dat`` file's text.

    Each problem's block starts with a header line holding ``Fopt (<value>)``. The
    data lines carry only 10 significant digits, so ``f - (f - f_opt)`` read from them
    is off by more than 0.005 where ``f`` is about 1e8 or more (f12 at D 10-40).
    """
    found = FOPT_HEADER.findall(text)
    if not found:
        raise ValueError("no 'Fopt (...)' header in the COCO data file")
    return float(found[-1])


def read_fopt_corrections() -> dict[str, list[float]]:
    """{problem id: [f_opt the run was scored against, exact f_opt]} (may be empty)."""
    if not FOPT_CORRECTIONS.exists():
        return {}
    return dict(json.loads(FOPT_CORRECTIONS.read_text()))


def read_fopt_cache() -> dict[str, float]:
    """The cached ``f_opt`` values; empty if the file is missing or unreadable.

    Another run may be replacing the file (Windows refuses the read meanwhile), so a
    failed read is retried a few times before the cache counts as empty and the
    values are recomputed.
    """
    for attempt in range(5):
        try:
            if not FOPT_CACHE.exists():
                return {}
            return dict(json.loads(FOPT_CACHE.read_text()))
        except (OSError, ValueError):
            time.sleep(0.2 * (attempt + 1))
    return {}


def write_fopt_cache(cache: dict[str, float]) -> None:
    """Merge ``cache`` into the cache file, atomically (lesson 25).

    The merged table goes to a temporary file next to the cache and replaces it with
    ``os.replace``, so a concurrent reader never sees a partial file and concurrent
    first runs on new ids keep each other's values. A failure to write is reported
    and ignored: this run has its values in memory, and a later run recomputes them.
    """
    merged = {**read_fopt_cache(), **cache}
    tmp = FOPT_CACHE.with_name(f"{FOPT_CACHE.name}.{os.getpid()}.tmp")
    try:
        FOPT_CACHE.parent.mkdir(exist_ok=True)
        tmp.write_text(json.dumps(merged, indent=0, sort_keys=True))
        for attempt in range(5):
            try:
                os.replace(tmp, FOPT_CACHE)
                return
            except PermissionError:  # Windows: a reader holds the file open
                time.sleep(0.2 * (attempt + 1))
        raise OSError(f"{FOPT_CACHE} stayed locked")
    except OSError as exc:
        print(f"warning: f_opt cache not written ({exc})", file=sys.stderr, flush=True)
        with contextlib.suppress(OSError):
            tmp.unlink()


# ------------------------------------------------------------------- run ----
def parse_penalty(text: str) -> float | None:
    return None if text == "raw" else float(text)


def parse_ints(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def task_seed(seed_value: int, function: int, dimension: int, instance: int):
    """The ``SeedSequence`` of one problem and seed (budgets share it)."""
    return np.random.SeedSequence(
        [SEED_NAMESPACE, seed_value, function, dimension, instance]
    )


def run_one(
    suite, solver, fopts, problem_id, multiplier, seed_value, penalty, scale=1.0
):
    """Run one (problem, budget, seed) and return its record."""
    function, instance, dimension = parse_id(problem_id)
    problem = suite.get_problem(problem_id)
    proxy = ProblemProxy(problem, fopts[problem_id], scale)
    started = time.perf_counter()
    row = run_bbob.solve_problem(
        solver,
        proxy,
        multiplier,
        task_seed(seed_value, function, dimension, instance),
        penalty,
    )
    seconds = time.perf_counter() - started
    problem.free()
    return {
        "id": problem_id,
        "function": function,
        "dimension": dimension,
        "instance": instance,
        "multiplier": multiplier,
        "seed": seed_value,
        "evaluations": row["evaluations"],
        "precision": max(row["best_f"] - fopts[problem_id], 1e-300),
        "restarts": row["restarts"],
        "target_hit": row["target_hit"],
        "trace_sha256": row["trace_sha256"],
        "objective_seconds": row["objective_seconds"],
        "seconds": seconds,
        "hits": proxy.hits,
    }


_WORKER: dict = {}


def _init_worker(spec: dict) -> None:
    import cocoex
    import scipy.linalg  # noqa: F401  (loaded here, not inside the first run's timing)
    import scipy.optimize  # noqa: F401

    _WORKER["spec"] = spec
    _WORKER["suite"] = cocoex.Suite(
        "bbob", spec.get("suite_instance", "year: 2019"), *spec["suite_args"]
    )
    _WORKER["solver"] = make_solver(spec["solver"], spec["config"], spec["start"])
    _WORKER["solver"].box_scale = spec.get("box_scale", 1.0)


def _work(task: tuple[str, float, int]) -> dict:
    problem_id, multiplier, seed_value = task
    spec = _WORKER["spec"]
    return run_one(
        _WORKER["suite"],
        _WORKER["solver"],
        spec["fopts"],
        problem_id,
        multiplier,
        seed_value,
        spec["penalty"],
        spec.get("box_scale", 1.0),
    )


def sha256_of(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def git(*args: str) -> str:
    try:
        done = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return done.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_state() -> dict:
    """Commit, branch and a fingerprint of uncommitted changes to the code.

    ``benchmarks/results`` is left out: a batch's own result files (and the f_opt
    cache) are not code, and counted they marked every later arm ``dirty``.
    """
    results = ":(exclude)benchmarks/results"
    diff = git("diff", "HEAD", "--", "src", "benchmarks", results)
    untracked = git(
        "ls-files", "--others", "--exclude-standard", "src", "benchmarks", results
    )
    extra = hashlib.sha256()
    for name in untracked.splitlines():
        with contextlib.suppress(OSError):
            extra.update(name.encode() + (ROOT / name).read_bytes())
    return {
        "commit": git("rev-parse", "HEAD") or "unknown",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": bool(diff or untracked),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        "untracked_sha256": extra.hexdigest() if untracked else None,
    }


def versions() -> dict:
    """Package versions for the provenance; ``cma`` and ``scipy`` are read from their
    metadata (importing pycma pulls in matplotlib and cost about 6.6 s per arm)."""
    from importlib import metadata

    out = {"python": platform.python_version(), "numpy": np.__version__}
    for module in ("cocoex", "cma", "scipy", "amm_optimizer"):
        with contextlib.suppress(Exception):
            if module in ("cma", "scipy"):
                out[module] = metadata.version(module)
            else:
                out[module] = str(__import__(module).__version__)
    out["blas_threads"] = BLAS_THREADS
    return out


def build_provenance(args, design: dict, config, fopts: dict, ids: list[str]) -> dict:
    return {
        "format": 2,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "git": git_state(),
        "versions": versions(),
        "platform": platform.platform(),
        "jobs": args.jobs,
        "stage": args.stage,
        "solver": args.solver,
        "config_text": args.config,
        "config": config.to_dict() if config is not None else None,
        "config_hash": config.config_hash() if config is not None else None,
        "design": design,
        "design_hash": sha256_of(design),
        "targets_sha256": sha256_of(TARGETS.tolist()),
        "fopt_basis": FOPT_BASIS,
        "fopt_sha256": sha256_of({p: fopts[p] for p in sorted(ids)}),
        "quick_scan_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


_ABSENT = object()
# Fields removed by the 2026-10-19 merge (refactor 40222d9), with their defaults: a
# journal that stored one of them at its default still resumes.
REMOVED_DEFAULTS: dict[str, object] = {
    "variant.allocation": "conductor",
    "variant.credit_rate": 0.2,
    "variant.credit_floor": 0.05,
    "variant.credit_min_samples": 2,
    "variant.early_accept": False,
    "variant.early_min": 8,
    "variant.es_init_share": 1.0,
    # dead code 3 (2026-10-21): switches no config or arm ever set
    "variant.travel_max_dim": 0,
    "variant.travel_min_per_dim": 30.0,
    "variant.travel_max_per_dim": 300.0,
    "variant.travel_threshold": 0.15,
    "variant.cheap_probe_per_dim": 0.0,
    "variant.des_skip_step": False,
    # dead code 4: the race (Batch 20-21) and pop_first_share (Batch 22)
    "variant.race": "none",
    "variant.race_min_dim": 2,
    "variant.race_min_budget_per_dim": 100.0,
    "variant.race_share": 1 / 3,
    "variant.race_rule": "hcma",
    "variant.race_r2_max": 1.0,
    "variant.pop_first_share": 0.0,
    # dead code 8: the research controls
    "variant.separable_local": False,
    "variant.additive_control": False,
    "variant.basin_control": False,
    # dead code 6: the arm-only population surrogate
    "variant.pop_surrogate": "none",
    # dead code 5: the hand-over extras of Batches 12-27
    "variant.cma_model_gate": "none",
    "variant.cma_model_weights": False,
    "variant.cma_mirror": False,
    "variant.cma_tie_neutral": False,
    "variant.cma_pin_stick": 0.75,
    "variant.cma_handoff_sigma": "atm",
    "variant.cma_warm_max_dim": 1000000,
    "variant.pop_bounds": "clip",
    "variant.pop_switch_share": 1.0,
    "variant.pop_switch_plateau": 0.0,
    "variant.pop_sigma_shrink": 1.0,
    "variant.cma_trust_switch": 0.0,
    "variant.cma_decay_switch": 0.0,
    "variant.cma_decay_max_dim": 5,
    "variant.cma_tie_switch": 0.0,
    "variant.cma_model_max_budget_per_dim": 1000000000000.0,
    "variant.cma_warm_blend": 0.0,
}


def config_changes(config: dict | None) -> dict | None:
    """The leaves of a stored ``ATMConfig.to_dict()`` that differ from the defaults.

    A field the stored config has but today's ``ATMConfig`` lacks (since removed)
    counts as a change, unless it is in ``REMOVED_DEFAULTS`` at its old default.
    """
    if config is None:
        return None
    import amm_optimizer as atm

    normal = json.loads(json.dumps(atm.ATMConfig().to_dict()))  # tuples -> lists
    base = dict(_flatten(normal))
    stored = dict(_flatten(json.loads(json.dumps(config))))
    return {
        k: v
        for k, v in stored.items()
        if base.get(k, _ABSENT) != v
        and not (k not in base and REMOVED_DEFAULTS.get(k, _ABSENT) == v)
    }


def same_config(header: dict, prov: dict) -> bool:
    """Whether journal ``header`` belongs to this run's configuration (``prov``).

    Equal config hashes, or the same changes from the defaults: a ``Variant`` field
    added since the journal began, at its default, changes ``config_hash`` (it covers
    every default) but not the run, so the journal stays resumable (lesson 23).
    """
    if header.get("config_hash") == prov.get("config_hash"):
        return True
    old = config_changes(header.get("config"))
    return old is not None and old == config_changes(prov.get("config"))


def read_jsonl(path: Path) -> tuple[dict | None, list[dict]]:
    header, records = None, []
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if "provenance" in item:
                header = item["provenance"]
            else:
                records.append(item)
    return header, records


def write_run(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data)
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(text.encode(), mtime=0))
    else:
        path.write_text(text)


def cmd_run(args) -> int:
    import cocoex

    # ``--instance-ids``: raw bbob instance numbers (any positive integer, e.g.
    # fresh holdouts 16 and up) instead of indices into a suite year's 1-15;
    # ``--year``: the indices of another COCO suite year (2021: ids 91-100 at 6-15).
    raw_ids = args.instance_ids
    if raw_ids and args.year != 2019:
        raise SystemExit("--instance-ids and --year exclude each other")
    instances = parse_ints(raw_ids or args.instances)
    locked = (
        sorted(set(instances) & set(LOCKED_INSTANCES))
        if raw_ids or args.year == 2019
        else []
    )
    if locked and not args.locked:
        raise SystemExit(
            f"instances {locked} are the locked Tier 3 set; pass --locked to use them"
        )
    if raw_ids:
        suite_instance = f"instances: {raw_ids}"
        suite_args: tuple[str, ...] = (
            f"dimensions: {args.dims} function_indices: {args.functions}",
        )
    else:
        suite_instance = f"year: {args.year}"
        suite_args = (
            f"dimensions: {args.dims} instance_indices: {args.instances} "
            f"function_indices: {args.functions}",
        )
    fopts = fopt_table(suite_args, suite_instance)
    ids = list(cocoex.Suite("bbob", suite_instance, *suite_args).ids())
    multipliers = [float(m) for m in args.multipliers.split(",")]
    config = make_solver(args.solver, args.config).__dict__.get("config")
    design = {
        "dims": args.dims,
        "instances": args.instances,
        "functions": args.functions,
        "multipliers": multipliers,
        "seeds": args.seeds,
        "seed_offset": args.seed,
        "penalty": args.penalty,
        "start": args.start,
    }
    if args.box_scale != 1.0:
        design["box_scale"] = args.box_scale
    if raw_ids:
        design["instances"] = None
        design["instance_ids"] = raw_ids
    if args.year != 2019:  # (the design hash of 2019 runs is unchanged)
        design["year"] = args.year
    prov = build_provenance(args, design, config, fopts, ids)
    if args.stage == "3" and prov["git"]["dirty"]:
        raise SystemExit("stage 3 results must come from a clean tree; commit first")
    tasks = [
        (pid, m, args.seed + s)
        for s in range(args.seeds)
        for m in multipliers
        for pid in ids
    ]
    tasks.sort(key=lambda t: -t[1] * parse_id(t[0])[2])  # longest first
    evals = sum(m * parse_id(pid)[2] for pid, m, _ in tasks)
    print(
        f"{args.solver} {args.config or '(defaults)'}: {len(tasks)} runs, "
        f"~{evals / 1e6:.1f} M evaluations, jobs={args.jobs}",
        flush=True,
    )
    if args.dry_run:
        return 0

    out = Path(args.out)
    jsonl = out.with_name(out.name.split(".")[0] + ".jsonl")
    done_header, done = read_jsonl(jsonl)
    if done_header is not None and (
        done_header["design_hash"] != prov["design_hash"]
        or not same_config(done_header, prov)
    ):
        raise SystemExit(f"{jsonl} belongs to another design or config; delete it")
    if done and not args.resume:
        raise SystemExit(
            f"{jsonl} exists with {len(done)} runs; use --resume or delete"
        )
    finished = {(r["id"], r["multiplier"], r["seed"]) for r in done}
    todo = [t for t in tasks if t not in finished]
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = list(done)
    spec = {
        "suite_args": suite_args,
        "year": args.year,
        "solver": args.solver,
        "config": args.config,
        "start": args.start,
        "penalty": args.penalty,
        "fopts": fopts,
        "box_scale": args.box_scale,
        "suite_instance": suite_instance,
    }
    with jsonl.open("a") as sink:
        if done_header is None:
            sink.write(json.dumps({"provenance": prov}) + "\n")
        step = max(len(todo) // 10, 1)

        def collect(record: dict) -> None:
            records.append(record)
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            if len(records) % step == 0:
                print(
                    f"  {len(records)}/{len(tasks)} runs, "
                    f"{time.perf_counter() - started:.0f}s",
                    flush=True,
                )

        if args.jobs <= 1 or len(todo) < 2:
            _init_worker(spec)
            for task in todo:
                collect(_work(task))
        else:
            # a broken pool is replaced and reruns the runs without a record
            with RetryingPool(args.jobs, _init_worker, (spec,)) as pool:
                for record in pool.map(_work, todo):
                    collect(record)
    prov["wall_seconds"] = round(time.perf_counter() - started, 1)
    records.sort(key=lambda r: (r["id"], r["multiplier"], r["seed"]))
    data = {
        "solver": args.solver,
        "config": args.config,
        "penalty": args.penalty,
        "provenance": prov,
        "records": records,
    }
    write_run(out, data)
    jsonl.unlink()
    print(f"wrote {out} ({len(records)} runs, {prov['wall_seconds']}s)")
    print_scores({args.solver: data})
    return 0


# --------------------------------------------------------------- scoring ----
def record_score(record: dict) -> float:
    return float(np.mean(record["precision"] <= TARGETS))


def run_scores(records: list[dict]) -> dict[tuple, float]:
    """Per (problem id, multiplier, seed) ECDF value."""
    return {(r["id"], r["multiplier"], r["seed"]): record_score(r) for r in records}


def group_of(function: int) -> str:
    return next(g for g, fs in GROUPS.items() if function in fs)


def print_scores(runs: dict[str, dict]) -> None:
    for name, run in runs.items():
        records = run["records"]
        multipliers = sorted({r["multiplier"] for r in records})
        print(f"\n{name}: mean ECDF value (fraction of 51 targets reached)")
        print(f"{'':<22}" + "".join(f"{f'x{m:g} D':>9}" for m in multipliers))
        scored = [(function_of(r), r["multiplier"], record_score(r)) for r in records]
        for label in ["all", *GROUPS]:
            cells = []
            for m in multipliers:
                vals = [
                    s
                    for f, mm, s in scored
                    if mm == m and (label == "all" or group_of(f) == label)
                ]
                cells.append(vals)
            if not any(cells):
                continue  # group not in this design
            print(
                f"{label:<22}"
                + "".join(f"{np.mean(v) if v else float('nan'):>9.3f}" for v in cells)
            )


def function_of(record: dict) -> int:
    return record.get("function") or parse_id(record["id"])[0]


def load_run(path: Path) -> dict:
    """Read a result file (``.json`` or ``.json.gz``; version 1 or 2)."""
    raw = (
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
    )
    data = json.loads(raw)
    basis = (data.get("provenance") or {}).get("fopt_basis")
    if basis is None:
        correct_legacy_precision(data)
    elif basis != FOPT_BASIS:
        correct_fopt_precision(data)
    return data


def correct_legacy_precision(data: dict) -> None:
    """Re-express the precision of old result files against the exact ``f_opt``.

    Files written before 2026-10-17 measured ``best f - f_opt`` against a cached
    ``f_opt`` with errors of 1e-8 to 4e-6, so a converged run read at least 2e-8
    and never reached the last targets. The shift is exactly the cache error
    (``precision`` stays additive in ``best f``); a record clamped at 1e-300 had
    ``best f`` at or below the cached value and is taken as converged.
    """
    raw = json.loads(FOPT_CACHE.read_text())
    fixed = read_fopt_corrections()
    for record in data["records"]:
        # the raw value such a run was scored against (the cache before any fix)
        old = fixed[record["id"]][0] if record["id"] in fixed else raw[record["id"]]
        shift = old - exact_fopt(raw[record["id"]])
        if record["precision"] > 1e-300:
            record["precision"] = max(record["precision"] + shift, 1e-300)
        elif shift > 1e-8:
            _flag_false_hit(record, shift)
    data["precision_corrected"] = "exact f_opt (applied on loading)"


def correct_fopt_precision(data: dict) -> None:
    """Re-express files scored against the 2026-10-17 to 2026-10-19 cache.

    Their ``f_opt`` was the rounded cache value, wrong on the ids of
    ``FOPT_CORRECTIONS``. ``precision`` is ``best f - f_opt``, so it shifts by the
    error. A record clamped at 1e-300 whose ``f_opt`` was too high stopped on a false
    final-target hit: ``_flag_false_hit`` marks it for a re-run.
    """
    fixed = read_fopt_corrections()
    for record in data["records"]:
        if record["id"] not in fixed:
            continue
        used, exact = fixed[record["id"]]
        shift = exact_fopt(used) - exact_fopt(exact)
        if record["precision"] > 1e-300:
            record["precision"] = max(record["precision"] + shift, 1e-300)
        elif shift > 1e-8:
            _flag_false_hit(record, shift)
    data["precision_corrected"] = "f_opt from the COCO data header (applied on loading)"


def _flag_false_hit(record: dict, shift: float) -> None:
    """A run that stopped because ``best f`` fell below a too-high ``f_opt``.

    Its true precision lies in (0, shift]; the conservative end is kept and the record
    is flagged (``fopt_false_hit``) so the run can be repeated.
    """
    record["precision"] = shift
    record["target_hit"] = False
    record["fopt_false_hit"] = True


def cluster_of(record_or_key) -> tuple[int, int, int]:
    """(function, dimension, instance) of a record or of a score key."""
    pid = record_or_key["id"] if isinstance(record_or_key, dict) else record_or_key[0]
    function, instance, dimension = parse_id(pid)
    return function, dimension, instance


def cluster_bootstrap(values: np.ndarray, rng, n_boot: int = 2000):
    """Mean and 95% CI of per-cluster values, resampling whole clusters."""
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = values[rng.randint(0, len(values), (n_boot, len(values)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


class Pairs:
    """Paired scores of a variant against a baseline, aggregated per cluster."""

    def __init__(self, base: dict, other: dict, exclude: set[int] = frozenset()):
        base_rec = {(r["id"], r["multiplier"], r["seed"]): r for r in base["records"]}
        other_rec = {(r["id"], r["multiplier"], r["seed"]): r for r in other["records"]}
        keys = sorted(
            k for k in set(base_rec) & set(other_rec) if cluster_of(k)[0] not in exclude
        )
        self.keys = keys
        self.base = [base_rec[k] for k in keys]
        self.other = [other_rec[k] for k in keys]
        self.diff = np.array(
            [
                record_score(o) - record_score(b)
                for b, o in zip(self.base, self.other, strict=True)
            ]
        )
        self.multiplier = np.array([k[1] for k in keys])
        self.cluster = [cluster_of(k) for k in keys]
        self.function = np.array([c[0] for c in self.cluster])
        self.dimension = np.array([c[1] for c in self.cluster])

    def select(self, mask: np.ndarray) -> np.ndarray:
        return self.diff[mask]

    def per_cluster(self, mask: np.ndarray) -> np.ndarray:
        """Mean difference of every cluster that has runs under ``mask``."""
        sums: dict[tuple, list[float]] = {}
        for c, d, keep in zip(self.cluster, self.diff, mask, strict=True):
            if keep:
                sums.setdefault(c, []).append(d)
        return np.array([np.mean(v) for v in sums.values()])

    def primary_mask(self) -> np.ndarray:
        return np.isin(self.multiplier, PRIMARY)

    def identical(self, mask: np.ndarray) -> tuple[int, int]:
        same = [
            b.get("trace_sha256") is not None
            and b.get("trace_sha256") == o.get("trace_sha256")
            for b, o, keep in zip(self.base, self.other, mask, strict=True)
            if keep
        ]
        return sum(same), len(same)

    def median_log_precision(self, mask: np.ndarray) -> float:
        d = [
            np.log10(o["precision"]) - np.log10(b["precision"])
            for b, o, keep in zip(self.base, self.other, mask, strict=True)
            if keep
        ]
        return float(np.median(d)) if d else float("nan")


def check_designs(names: list[str], runs: dict[str, dict], allow: bool) -> None:
    provs = {n: runs[n].get("provenance") for n in names}
    missing = [n for n, p in provs.items() if p is None]
    if missing:
        print(
            f"note: {missing} have no provenance (version 1 files): designs and seeds "
            "are not verified, treat as unpaired reference levels"
        )
        return
    designs = {n: provs[n]["design_hash"] for n in names}
    if len(set(designs.values())) > 1 and not allow:
        detail = {n: provs[n]["design"] for n in names}
        raise SystemExit(
            f"different designs, comparison would not be paired:\n"
            f"{json.dumps(detail, indent=1)}\nuse --allow-mismatch to intersect anyway"
        )


def report_pair(name: str, base_name: str, pairs: Pairs, rng, label: str = "") -> dict:
    out: dict = {}
    n_clusters = len(set(pairs.cluster))
    print(
        f"\npaired{label}: {name} minus {base_name} (positive = {name} better); "
        f"{len(pairs.keys)} runs, {n_clusters} clusters"
    )
    head = (
        "budget",
        "mean diff",
        "95% CI (cluster bootstrap)",
        "win/loss/tie",
        "identical",
    )
    print(f"{head[0]:<10}{head[1]:>10}{head[2]:>30}{head[3]:>16}{head[4]:>12}")
    rows = [(f"x{m:g}", pairs.multiplier == m) for m in sorted(set(pairs.multiplier))]
    primary = pairs.primary_mask()
    if primary.any():
        rows.append(("primary", primary))
    for tag, mask in rows:
        mean, lo, hi = cluster_bootstrap(pairs.per_cluster(mask), rng)
        d = pairs.select(mask)
        wlt = f"{(d > 1e-12).sum()}/{(d < -1e-12).sum()}/{(abs(d) <= 1e-12).sum()}"
        same, total = pairs.identical(mask)
        ci = f"[{lo:+.3f}, {hi:+.3f}]"
        print(f"{tag:<10}{mean:>+10.3f}{ci:>30}{wlt:>16}{f'{same}/{total}':>12}")
        out[tag] = {"mean": mean, "lo": lo, "hi": hi, "identical": [same, total]}
    print(
        "median log10 precision change (negative = better): "
        + ", ".join(
            f"x{m:g}: {pairs.median_log_precision(pairs.multiplier == m):+.2f}"
            for m in sorted(set(pairs.multiplier))
        )
    )
    print("by group (mean diff; * = worse than -0.02 at 30-300 x D):")
    losers = []
    multipliers = sorted(set(pairs.multiplier))
    print(f"  {'':<22}" + "".join(f"{f'x{m:g}':>9}" for m in multipliers))
    for g, fs in GROUPS.items():
        in_group = np.isin(pairs.function, list(fs))
        if not in_group.any():
            continue
        cells = []
        for m in multipliers:
            mask = in_group & (pairs.multiplier == m)
            value = float(pairs.select(mask).mean()) if mask.any() else float("nan")
            flag = "*" if m in PRIMARY and value < -GROUP_LOSS else " "
            if flag == "*":
                losers.append(f"{g}@x{m:g}")
            cells.append(f"{value:>+8.3f}{flag}")
        print(f"  {g:<22}" + "".join(cells))
    out["groups_worse"] = losers
    dims = sorted(set(pairs.dimension))
    if len(dims) > 1 and primary.any():
        print(
            "by dimension, primary metric: "
            + ", ".join(
                f"D={dim}: "
                f"{pairs.select(primary & (pairs.dimension == dim)).mean():+.3f}"
                for dim in dims
            )
        )
    return out


def cmd_compare(args) -> int:
    runs = {}
    for path in args.files:
        runs[Path(path).name.split(".")[0]] = load_run(Path(path))
    if len(runs) < 2:
        raise SystemExit("need at least two result files")
    names = list(runs)
    baseline = args.baseline or names[0]
    if baseline not in runs:
        raise SystemExit(f"--baseline {baseline!r} is not one of {names}")
    check_designs(names, runs, args.allow_mismatch)
    print_scores(runs)
    rng = np.random.RandomState(0)
    summary: dict = {}
    exclude = set(parse_ints(args.exclude)) if args.exclude else set()
    for other in names:
        if other == baseline:
            continue
        pairs = Pairs(runs[baseline], runs[other])
        if not pairs.keys:
            print(f"\n{other}: no common (problem, budget, seed) with {baseline}")
            continue
        entry = {"all": report_pair(other, baseline, pairs, rng)}
        present = exclude & set(pairs.function.tolist())
        if present:
            tag = ",".join(f"f{f}" for f in sorted(present))
            entry["excluding"] = report_pair(
                other,
                baseline,
                Pairs(runs[baseline], runs[other], present),
                rng,
                f" excluding {tag}",
            )
        primary = entry["all"].get("primary")
        if primary:
            a = primary["lo"] > 0
            b = not entry["all"]["groups_worse"]
            d = (
                entry["excluding"].get("primary", {}).get("lo", float("nan")) > 0
                if "excluding" in entry
                else None
            )
            without = ",".join(f"f{f}" for f in sorted(present)) or "f5"
            print(
                f"acceptance (Tier 1 part): (a) CI above 0: {'yes' if a else 'NO'}; "
                f"(b) no group below -{GROUP_LOSS}: {'yes' if b else 'NO'}; "
                f"(d) holds without {without}: "
                f"{'n/a' if d is None else ('yes' if d else 'NO')}"
            )
            entry["accept"] = {"a": bool(a), "b": b, "d": d}
        summary[other] = entry
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--solver", required=True)
    run.add_argument("--dims", default="2,5,10")
    run.add_argument("--instances", default="1-3")
    run.add_argument("--functions", default="1-24")
    run.add_argument("--multipliers", default="10,30,100,300,1000")
    run.add_argument("--seeds", type=int, default=1)
    run.add_argument("--seed", type=int, default=0, help="first seed value")
    run.add_argument(
        "--config",
        default="",
        help='ATMConfig overrides, e.g. "variant.bounds=fold,variant.corner_clip=0.25"',
    )
    run.add_argument(
        "--penalty",
        type=parse_penalty,
        default=1.0,
        help='out-of-box penalty; "raw": evaluate unclipped, as plain COCO does',
    )
    run.add_argument(
        "--start", choices=["uniform", "center", "coco2019"], default="uniform"
    )
    run.add_argument(
        "--jobs", type=int, default=max(min((os.cpu_count() or 2) - 2, 12), 1)
    )
    run.add_argument(
        "--resume", action="store_true", help="continue an interrupted run"
    )
    run.add_argument(
        "--instance-ids",
        default="",
        help='raw bbob instance numbers, e.g. "16-30" (fresh holdouts); '
        "replaces --instances",
    )
    run.add_argument(
        "--box-scale",
        type=float,
        default=1.0,
        help="scale invariance check: the solver sees y = s x in the box [-5s, 5s]",
    )
    run.add_argument("--stage", default="", help="label stored in the provenance")
    run.add_argument("--locked", action="store_true", help="allow instances 11-15")
    run.add_argument(
        "--year", type=int, default=2019, help="COCO bbob suite year (instance set)"
    )
    run.add_argument("--dry-run", action="store_true", help="count the runs and stop")
    run.add_argument("--out", required=True, help="result file, .json or .json.gz")
    run.set_defaults(func=cmd_run)
    compare = sub.add_parser("compare")
    compare.add_argument("files", nargs="+")
    compare.add_argument(
        "--baseline", help="file stem of the baseline (default: first)"
    )
    compare.add_argument(
        "--exclude", default="5", help="also report without these functions"
    )
    compare.add_argument("--allow-mismatch", action="store_true")
    compare.add_argument("--json", help="write the numbers here")
    compare.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
