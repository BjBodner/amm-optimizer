# Security policy

## Supported versions

Only the latest released version receives security fixes.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Use GitHub's private vulnerability reporting: on the repository page open the
**Security** tab and choose **Report a vulnerability**. If that is not available to
you, open an issue that says only that you have a security report and ask for a private
channel; do not include details.

You can expect an acknowledgement within 7 days and a first assessment within 14 days.
Fixes are released as a new version with a GitHub security advisory (and a CVE where
appropriate); you will be credited unless you prefer otherwise.

## Scope

This is a numerical library with no network access, no file or process side effects at
import, and no deserialisation of untrusted data (configurations and results use JSON,
never pickle). In scope are, for example: arbitrary code execution through a crafted
configuration or objective-function return value, unbounded memory or CPU use that
ignores the evaluation budget, and problems in the build, release or CI configuration
that could let someone publish a modified package.

Not in scope: results that are numerically poor on a particular function (open a normal
issue), and vulnerabilities in NumPy or SciPy themselves (report those upstream).

## Verifying releases

Releases are built in GitHub Actions and published to PyPI with Trusted Publishing, so
no long-lived upload token exists. Each file on PyPI carries a PEP 740 attestation that
links it to the workflow run that built it.
