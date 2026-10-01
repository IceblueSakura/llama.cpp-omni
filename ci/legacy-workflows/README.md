# Archived runner recipes (not GitHub Actions checks)

These inherited workflows need project-owned hardware/runners that are not
provisioned on the validation fork. They are preserved as reference recipes,
not active checks, and have **not** been certified by the tier-1 CI result.
Moving them here does not remove backend source code or tests.

| Recipe | Required infrastructure |
| --- | --- |
| `build-ibm.yml` | s390x and ppc64le runners |
| `build-riscv.yml` | RISC-V runners and toolchain |
| `build-self-hosted.yml` | Dedicated CPU/GPU/Metal/OpenVINO runners and model fixtures |
| `build-sanitize.yml` | Persistent Linux CPU sanitizer runner |
| `ui-self-hosted.yml`, `ui-build-self-hosted.yml` | Persistent UI runners with matching Node/Playwright versions |

Before restoring a recipe to `.github/workflows/`:

1. Assign an owner and provision matching runner labels, SDKs, models and budget.
2. Review the inherited targets and test commands against current Omni contracts.
   Some recipes are stale: for example the RISC-V sanitizer disables tests before
   calling CTest and uses `continue-on-error`. These are not acceptable passing checks.
3. Remove swallowed failures; require CTest `--no-tests=error` when running tests.
4. Keep untrusted PR code off persistent runners. Use explicit trusted dispatch
   or an approved ephemeral-runner design.
5. Restore reusable workflows together and fix local `uses:` paths. Run real
   acceptance tests, then document the achieved coverage and remaining limitations.

Hosted Apple/Android and optional backend SDK workflows remain separate from
these archives. Their compile-only coverage does not establish GPU/device inference.
The obsolete disabled AI-issue, benchmark, CANN/SYCL placeholder, original
chat-server and upstream Winget automation files were removed; Git history is
available if a maintainer needs their original implementation.
