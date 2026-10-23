# AMM project site

The website for [AMM (Adaptive Multi Mode)](https://github.com/BjBodner/amm-optimizer), built with
[Astro](https://astro.build) and [Starlight](https://starlight.astro.build). It is published to GitHub Pages by
`.github/workflows/pages.yml`.

## Build it

```bash
cd site
npm ci
npm run dev        # local preview at http://localhost:4321
npm run build      # static site in dist/
```

Node 22.23.4 or newer is required (`.node-version` pins the version CI uses).

## Every number comes from one file

No result is typed into a page. Each number is a `<Claim id="..."/>` component backed by `data/claims.json`, and
each claim has a status:

| Status | Meaning |
|---|---|
| `final` | from a frozen run, or a plain fact |
| `dev` | development data on instances used while building AMM (shown with an "in-sample" badge) |
| `pending` | a placeholder that is not yet known (shown with a "pending" badge) |

The build writes a noindex preview by default. With `AMM_LAUNCH=1` the build fails if any rendered claim is
`pending`, or `dev` but marked `requiresFinal`, so unfinished numbers cannot be published as launched.
`AMM_SHOW_AUTHOR=1` shows the author blocks. `SITE_BASE` sets the base path (the workflow uses `/amm-optimizer`)
and `SITE_URL` the public address.

The deploy workflow reads the repository variables `AMM_LAUNCH` and `AMM_SHOW_AUTHOR`
(Settings, Secrets and variables, Actions, Variables). Change them and re-run the workflow; no code edit is needed.

## Data and checks

* `data/dev/` has the development comparison (instance ids 61-70) with its provenance README;
  `data/ft1/` has the final-test outputs and the preregistration (instance ids 101-110).
* `python tools/check_snippet.py` runs the home-page snippet against the repository's `src/` and checks that it prints
  what the page says. Run it from a Python environment with the package's dependencies installed.
* `python tools/check_base.py dist /amm-optimizer` checks a built site for links that ignore the base path.
* The cell-status rule used by the heatmaps and loss tables: ahead if the difference in share of targets reached is
  above +0.02, level within 0.02, behind below -0.02. This is a plotting threshold on one seed per instance. It is not
  the preregistered "credibly behind" rule, which the final-test pages state separately.

## Licence

Apache-2.0, as the rest of the repository.
