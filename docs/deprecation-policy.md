# Deprecation policy

amm-optimizer follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the
version is 0.x, a minor release (0.6 to 0.7) may change behaviour, with notice in the
[changelog](changelog.md); a patch release (0.6.0 to 0.6.1) never changes results.

## How public API leaves

Public API that is deprecated in minor release N keeps working in N, with a warning, and is
removed in release N+1 at the earliest.

* **Deprecated in N:** the old name, argument or option still works and emits a warning that
  names the replacement, if there is one, and the release that removes it: a
  `DeprecationWarning`, or a `FutureWarning` where the people running the code, not only the
  authors of a library built on it, should see it. The changelog lists it under *Deprecated*.
* **Removed in N+1 or later:** the changelog lists it under *Removed*.

So code that runs without deprecation warnings on one minor release keeps running on the next.
To find deprecated calls in your code, turn these warnings into errors in your tests, for
example with `python -W error::DeprecationWarning -W error::FutureWarning` or pytest's
`filterwarnings = ["error::DeprecationWarning", "error::FutureWarning"]`.

## What counts as public API

The names exported by the package (`amm_optimizer.__all__`, documented in the
[API reference](api.md)), their documented arguments and attributes, and the named
[presets](presets.md). Modules, functions and attributes whose names start with an underscore,
and modules not documented here, are internal and can change in any release.

Results are a separate promise: the frozen default (`presets.get("c46")`, configuration hash
`2153948ebed51150`) runs bit for bit as recorded in the environment of its snapshots, and a
change to it is announced in the changelog as a change in behaviour.
