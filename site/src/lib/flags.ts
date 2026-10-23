// Build-time switches. Set in the environment of the build, never in the browser.
//   AMM_LAUNCH=1       public build: indexable, and the build fails on unfinished claims
//   AMM_SHOW_AUTHOR=1  show the named author block (hidden by default)
// @ts-ignore process exists at build time
const env: Record<string, string | undefined> = typeof process !== 'undefined' ? process.env : {};

export const LAUNCH = env.AMM_LAUNCH === '1';
export const SHOW_AUTHOR = env.AMM_SHOW_AUTHOR === '1';
