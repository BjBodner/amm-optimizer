"""Named configurations for ``minimize(..., config=...)``.

``presets.get(name)`` returns the :class:`~amm_optimizer.ATMConfig` of a preset, and
``minimize`` also takes the name itself (``config="c46"``). The presets on display
(``names()``):

* ``"default"``: AMM, the default preset (frozen as ``c46``), which
  ``minimize`` runs without ``config``.
* ``"c46"``: the same configuration under its research name, frozen for the paper
  on 2026-10-22 (config hash ``2153948ebed51150``).
* ``"atm"``: ``ATMConfig()``, the original ATM sampler alone; it needs no bounds.
* ``"composite"``: ``ATMConfig.composite()``, ATM's sampler with five confirmed
  options.

The research composites ``c23`` to ``c45``, the steps that led to ``c46``, stay
loadable so that earlier results reproduce; ``names(research=True)`` lists them and
``docs/HISTORY.md`` describes each one. Every composite (``c23`` to ``c46``) needs
``bounds`` and a deterministic objective.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .config import ATMConfig, Variant

_C23_VARIANT: dict[str, Any] = {
    "amplitude_control": "success_rate_targeted",
    "initial_design": "uniform",
    "bounds": "fold",
    "corner_clip": 0.25,
    "pool_rule": "budget",
    "pool_min": 12,
    "reevaluate_incumbent": False,
    "es_phase": "prefix",
    "es_fraction": 0.9,
    "es_max_budget_per_dim": 100.0,
    "covariance": "rank_mu",
    "cov_rate": 3.0,
    "cov_select": 0.5,
    "es_kind": "cma",
    "cma_handoff": 0.2,
    "cma_model": "quadratic",
    "cma_active": True,
    "cma_warm_start": True,
    "cma_restart_rule": "short",
    "cma_handoff_cap_per_dim": 60.0,
    "cma_local": "then_ipop",
    "cma_local_min_dim": 2,
    "pop_gate_per_dim": 100.0,
    "pop_active": True,
    "pop_lambda": "fixed2",
    "cma_model_high": "diagonal",
    "sep_probe": True,
    "nm_max_dim": 2,
    "nm_stage": "prefix",
    "pop_switch_per_dim": 150.0,
    "pop_switch_start": "incumbent",
    "pop_protect": True,
    "pop_switch_min_dim": 10,
}
# Each later composite as its changes to the one before.
_CHANGES: dict[str, tuple[str, dict[str, Any]]] = {
    "c24": (
        "c23",
        {
            "es_kind": "lq2",
            "es_lq_min_budget_per_dim": 100.0,
            "es_lq_nonsep": True,
            "cma_probe_at_per_dim": 30.0,
            "cma_probe_beta": 1.3,
            "cma_probe_min_dim": 5,
        },
    ),
    "c25": ("c24", {"es_kind": "slsqplq"}),
    "c26": ("c25", {"cma_local_nonsep": "qn3"}),
    "c27": (
        "c26",
        {
            "es_kind": "slsqplq10c",
            "nm_stage": "qnprefix30",
            "cma_local_nonsep": "qn7",
        },
    ),
    "c28": ("c27", {"cma_local_nonsep": "qn8"}),
    "c29": ("c28", {"es_kind": "slsqplq10h", "cma_local_nonsep": "qn9"}),
    "c30": ("c29", {"cma_local_nonsep": "qn8"}),
    "c31": ("c30", {"cma_local_nonsep": "qn12"}),
    "c32": ("c31", {"qn_fallback": "lbfgsb"}),
    "c33": ("c32", {"cma_local_nonsep": "qn17"}),
    "c34": ("c33", {"cma_local_nonsep": "qn19"}),
    "c35": ("c34", {"cma_local_nonsep": "qn20"}),
    "c36": (
        "c35",
        {
            "cma_local_nonsep": "qn29",
            "prefix_sep_probe": True,
            "prefix_full_per_dim": 100.0,
            "prefix_full_min_per_dim": 10.0,
        },
    ),
    "c37": ("c36", {"cma_local_nonsep": "qn38"}),
    "c38": ("c37", {"chain_route": True, "sep_route": True}),
    "c39": (
        "c38",
        {"basin_route": True, "sep_check": "stall", "sep_stall_min_per_dim": 1000.0},
    ),
    "c40": (
        "c39",
        {
            "es_kind": "slsqplq10hd",
            "prefix_full_min_per_dim": 0.0,
            "des_max_per_dim": 31.0,
        },
    ),
    "c41": ("c40", {"cma_local_nonsep": "qn44"}),
    "c42": ("c41", {"basin_descents": 1, "basin_keep_atm": True}),
    "c43": ("c42", {"small_engine": "cobyqa", "small_engine_max_dim": 3}),
    "c44": (
        "c43",
        {
            "prefix_plateau_pop": True,
            "prefix_plateau_min_dim": 5,
            "small_switch": True,
            "small_switch_trials": 6.0,
            "small_switch_keep_unit": True,
        },
    ),
    "c45": (
        "c44",
        {
            "rugged_try_rest": "cutkink",
            "prefix_nan_reading": "plateau",
            "prefix_ridge_after": 7,
            "prefix_ridge_max_per_dim": 30.0,
            "box_normalize": True,
            "prefix_descent_retries": 5,
            "rugged_direct_beta": 1.0,
            "rugged_direct_min_beta": 0.5,
        },
    ),
    "c46": ("c45", {"prefix_low_beta": 0.15, "prefix_low_beta_min_dim": 10}),
}


def _variants() -> dict[str, dict[str, Any]]:
    out = {"c23": dict(_C23_VARIANT)}
    for name, (base, changes) in _CHANGES.items():
        out[name] = {**out[base], **changes}
    return out


_VARIANTS = _variants()


#: The preset ``"default"`` is (the configuration ``minimize`` runs without ``config``).
DEFAULT = "c46"
#: The names :func:`names` lists.
SHOWN = ("default", DEFAULT, "atm", "composite")


def names(research: bool = False) -> tuple[str, ...]:
    """Return the preset names.

    Parameters
    ----------
    research : bool, optional
        ``False`` (the default): the presets on display, ``("default", "c46",
        "atm", "composite")``. ``True``: also the research composites ``c23`` to
        ``c45``, oldest first.

    Returns
    -------
    tuple of str
        Names that :func:`get` accepts.

    Examples
    --------
    >>> from amm_optimizer import presets
    >>> presets.names()
    ('default', 'c46', 'atm', 'composite')
    >>> len(presets.names(research=True))
    27
    """
    if not research:
        return SHOWN
    return SHOWN + tuple(name for name in _VARIANTS if name != DEFAULT)


def get(name: str) -> ATMConfig:
    """Return the configuration of preset ``name``.

    Parameters
    ----------
    name : str
        A name of ``names(research=True)``: ``"default"`` (the same as ``"c46"``),
        ``"atm"``, ``"composite"``, or a composite ``"c23"`` to ``"c46"``.

    Returns
    -------
    ATMConfig
        A new configuration; changing it (``dataclasses.replace``) does not change
        the preset.

    Raises
    ------
    ValueError
        If ``name`` is not a preset name.

    Examples
    --------
    >>> from amm_optimizer import presets
    >>> presets.get("default") == presets.get("c46")
    True
    >>> presets.get("c46").config_hash()
    '2153948ebed51150'
    """
    if name == "default":
        name = DEFAULT
    if name == "atm":
        return ATMConfig()
    if name == "composite":
        return ATMConfig.composite()
    if not isinstance(name, str) or name not in _VARIANTS:
        raise ValueError(
            f"unknown preset {name!r}; presets: {', '.join(SHOWN)}, and the research "
            "composites c23 to c45 (presets.names(research=True))"
        )
    return ATMConfig(
        iterations_per_hyperparameter_set=1_000_000,
        variant=Variant(**_VARIANTS[name]),
    )


def with_changes(name: str, **variant_changes: Any) -> ATMConfig:
    """Return preset ``name`` with some of its ``Variant`` fields changed.

    Parameters
    ----------
    name : str
        A preset name, as for :func:`get`.
    **variant_changes
        ``Variant`` fields and their new values.

    Returns
    -------
    ATMConfig
        ``get(name)`` with ``variant`` replaced by a copy carrying the changes.

    Examples
    --------
    >>> from amm_optimizer import presets
    >>> presets.with_changes("c46", anytime="replay").variant.anytime
    'replay'
    """
    config = get(name)
    return replace(config, variant=replace(config.variant, **variant_changes))
