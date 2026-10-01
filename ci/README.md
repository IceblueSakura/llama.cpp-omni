# CI/CD for llama.cpp-omni

## Support levels

| Level | Platforms | Policy |
| --- | --- | --- |
| Tier 1 | macOS Apple Silicon, Linux, Windows | Required native build and tests; failures block release |
| Tier 2 | macOS Intel, non-macOS Apple platforms, Android, other platforms | Explicit compatibility runs; never block tier-1 release |

The required matrix currently covers macOS arm64, Linux x86_64 and Windows
x86_64. This is not a claim that all Linux/Windows architectures or GPU
backends have been tested. Additional architectures and hardware need their
own acceptance results.

## Upstream reference

The target project is [tc-mb/llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni).
CI conventions were refreshed against
[ggml-org/llama.cpp at ca2e2037b68ae7d821113d538110e6ce6903864c](https://github.com/ggml-org/llama.cpp/tree/ca2e2037b68ae7d821113d538110e6ce6903864c/.github/workflows):

- Checkout/Python/Node/artifact action generations and CPU compiler/cache setup.
- Apple SDK configuration, Android SDK action pin and NDK selection.
- Docker Buildx action pins and explicit release verification.
- HIP compiler selection in `ci/run.sh`.

This is a selective CI sync, **not** a runtime-source rebase. Upstream's newer
CMake versioning and model tests, HF cache credentials, AI issue automation,
Winget package ownership and original `llama-server` API are not transferable
contracts. Do not restore them by blindly copying upstream workflows.

## Required CI

`build.yml` runs on every PR and push to `master`, and supports manual runs.
There is deliberately no path filter on its `Required checks` job: configuring
that stable name in branch protection must not leave a PR waiting for a check
that never starts. Every dependency must succeed; cancellation and unexpected
skips are failures, not passes.

The pipeline builds the UI once, then all three native platforms. It runs:

- CI script unit/style/type checks and actionlint, including local workflow references.
- Full Python lint/type checks (including isolated Token2Wav), UI type/lint/browser
  tests, EditorConfig, model-factory naming, pinned vendor reproduction, operations
  documentation, compatibility requirements and CMake package consumption.
  These reusable checks share the release gate and do not launch duplicate PR runs.
- A full native tool build, including `llama-omni-cli`, `llama-omni-server`,
  `llama-tts-server` and `voxcpm2-cli`.
- CTest labels `main|python` (including existing tiny-model fixtures), with
  only `test-llama-archs` excluded from the required suite; see diagnostics below.
- `ci/smoke.py`: help/exit contracts, TTS missing-weight rejection, Omni
  HTTPS health endpoints, missing-model initialization and continued health.

The standalone TTS server requires weights before listening, so its HTTP
health/inference path is **not** covered by the model-free smoke test. The
inherited chat API suite in `tools/server/tests/` is not silently redirected
to Omni: it targets a different, removed server protocol (see issue #119).

TLS-enabled smoke runs create disposable localhost certificates. Both native
CI and archive checks use BoringSSL; Windows OpenMP is disabled to avoid an
unbundled runtime DLL. The current servers instantiate SSLServer when TLS is
compiled in, even without certificate arguments. CI supplies a certificate
rather than claiming plain HTTP works in that configuration.

Release archives use the tested runner ABI: Linux is built on Ubuntu 24.04,
macOS on macOS 14, and Windows with the MSVC-compatible runtime (`/MD`). Windows
users need a compatible Visual C++ Redistributable; archive smoke tests on a
hosted runner do not establish compatibility with every fresh/older OS install.

Model-weight inference, audio output quality, duplex behavior and GPU/ANE
execution require separate fixtures and hardware. A green build/help/health
check does not establish those capabilities.

## Optional checks

- `build-apple.yml`: manual macOS Intel tests or iOS/tvOS/visionOS library builds.
  Apple SDK cross-builds are compile-only, without code signing or device tests.
- `build-android.yml`: manual NDK arm64 Omni build; no on-device inference claim.
- Other backend/toolchain workflows: manual compatibility runs. Existing
  recipes are retained, but they are not certified by the tier-1 result.
- Dedicated-runner recipes have been moved to [legacy-workflows](legacy-workflows/README.md).
  They are not active GitHub checks or verified platform support.
- `server-sanitize.yml`: hosted, manual model-free server sanitizer checks.
- `models-check.yml`: full-architecture synthetic checks on the native matrix,
  manually or when that workflow changes. Existing CPU F32/F16 binary-op and
  macOS Meta split-state failures are reported as failed diagnostics, not passes.
  This workflow is intentionally not a release/required-check dependency; its
  isolation was approved for this CI repair instead of expanding into engine fixes.
- Python, naming, vendor, EditorConfig and operations-documentation checks use
  GitHub-hosted runners. UI's extended tests are part of the required gate.
  Main Python tools and the legacy Token2Wav service are type-checked in separate
  locked environments; neither failure is hidden with continue-on-error.

The six obsolete `.yml.disabled` workflows were removed (CANN/SYCL placeholders,
old benchmark/chat-server, upstream AI issue automation and Winget publication).
Their history remains in Git; restoring functionality requires a project-owned
implementation and the necessary infrastructure.

Vendor checks use `python scripts/sync_vendor.py --check`: immutable source
versions are downloaded to a temporary directory and compared byte-for-byte.
The existing header-only httplib patches are reproduced from
`scripts/vendor-cpp-httplib.patch`; this does not upgrade or modify vendored code.
The default command without `--check` remains an explicit snapshot-update operation.

Active workflows pass actionlint without diagnostic suppression. Runner labels
are declared in `.github/actionlint.yaml`; declaring a label does not provision
hardware. OpenVINO defaults to hosted CPU checks; its GPU dispatch choice requires
a project-owned runner. Snapdragon builds do not spend device-cloud quota unless
`run_device_tests=true`; requesting tests without credentials fails rather than
silently reporting success. These optional SDK/hardware paths remain outside the
required gate and must not be described as verified by a green tier-1 result.

EditorConfig follows the legacy UI's mixed Prettier-owned indentation instead of
forcing spaces onto tab-formatted source. Embedded Markdown fixtures retain their
intentional trailing spaces. Source whitespace cleanup does not change engine
behavior; C++ lexical tokens and Python executable ASTs were compared before applying
it (docstring trailing whitespace is normalized).

## Release and publishing

Run `release.yml` manually. Its default `create_release=false` verifies the
same commit through required CI, creates tier-1 archives and smoke-tests the
extracted binaries. It creates no tag and publishes nothing in this mode.

To publish, provide an **existing tag at the selected commit** and explicitly
set `create_release=true`. Publication waits for all required checks and
archive tests, targets the current repository and includes SHA256 checksums.
Tier-2 checks are not release dependencies. The UI artifact is reused, never
rebuilt/uploaded under the same name in the same run.

`docker.yml` is also manual and defaults to `push=false`. It requires an already
successful `build.yml` run for the exact commit (without repeating native CI),
builds the Linux x86_64 CPU Omni server image, and checks its actual entry
point and HTTP health before optionally pushing to the current repository's
GHCR. The CPU image uses plain HTTP (`LLAMA_OPENSSL=OFF`); terminate TLS at a
reverse proxy when deploying. It links the ordinary CPU backend: Omni currently
calls CPU symbols directly and does not support the inherited dynamic/all-CPU-
variants configuration. GPU image variants are not part of this gate.

PyPI publishing is manual and requires ownership of the package and a token.
The reusable HF UI publisher requires an explicit `owner/bucket` destination,
a token and an existing `ui-build` artifact. Neither is invoked by the default
release, and neither publishes into `ggml-org` implicitly.

## Python environments

The primary development/CI interpreter is Python 3.12 (`.python-version`).
`pyproject.toml` and `uv.lock` are authoritative; the stale root `poetry.lock`
was retired. The existing poetry-core packaging backend is retained. Install
uv 0.12.17, then use frozen dependency profiles rather than ad-hoc pip installs:

```bash
uv sync --frozen --only-group ci        # native CTest/Jinja and CI script tools
uv run --frozen --no-sync flake8 .
uv run --frozen --no-sync python -m unittest discover -s ci -v
uv sync --frozen --group typecheck --no-install-project
uv run --frozen --no-sync ty check --python .venv --extra-search-path . --exit-zero-on-warning
```

The CI group retains NumPy 1.26.4. The typecheck profile mirrors the existing
requirements-all tool coverage and includes librosa for APM conversion; it
uses CPU Torch wheels on Linux/Windows and PyPI's Apple Silicon wheels.
Compatibility `requirements*.txt` entry points remain available, but CI profiles
are resolved and pinned in the uv lockfile. Dependency updates must update the
matching lockfile; CI checks freshness before frozen installation.

Flake8 7 hosts the existing NP100 no-print plugin through an explicit uv dependency
override: its AST API is compatible, but its old Flake8 4 pin is not compatible
with Python 3.12. A regression test verifies both forbidden prints and existing
`# noqa: NP100` exceptions. This preserves the rule instead of disabling it.
The newer pinned ty avoids old dynamic-dictionary narrowing false positives.
The main check explicitly selects its venv and repository import root (including
scripts that mutate sys.path). Existing warning-level diagnostics remain visible;
`--exit-zero-on-warning` does not suppress errors or make them successful.

Token2Wav has its own `tools/omni/pyt2w/pyproject.toml` and `uv.lock`, because its
Torch 2.3/Transformers 4 stack conflicts with the main tools. It is excluded
from the root ty invocation **only because** a separate CI job checks it:

```bash
cd tools/omni/pyt2w
uv sync --frozen --group ci --extra cpu
uv run --frozen --no-sync ty check .
uv run --frozen --no-sync python -m unittest discover -s tests -v
```

The `cpu` extra is for CI. Without it, Linux/Windows use the standard PyPI Torch
build rather than forcing CPU wheels. CUDA deployment and real model inference
still require separate hardware/model acceptance; the tests cover JSON protocol,
missing-model rejection and deterministic PCM writing, not audio quality.

On NixOS, use `nix develop .#python-ci`; this declares Python 3.12 and exposes
only the compiler/zlib runtime libraries needed by the wheels inside that shell.
It sets `UV_PYTHON` and disables uv interpreter downloads. No global loader or
system Python configuration is changed.

## Local verification

```bash
uv run --frozen --no-sync python -m unittest discover -s ci -p 'test_*.py' -v
cmake -B build -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build --parallel
ctest --test-dir build -L 'main|python' --output-on-failure --timeout 900 --no-tests=error
# Use --tls for a TLS-enabled server (openssl CLI is required for the test cert).
uv run --frozen --no-sync python ci/smoke.py --bin-dir build/bin --tls
```

For a multi-config generator, build with `--config Release`, pass `-C Release`
to CTest and use `--bin-dir build/bin/Release`. Native Unix CI uses single-config
Ninja because the inherited tokenizer-repository script expects binaries in
`build/bin`, not its `Release` subdirectory. `ci/package.py` writes a fresh
archive in `dist/` and extracts it to `build-ci-package/`; it refuses to overwrite
an existing archive/extraction directory.

The existing heavy hardware harness remains available, for example:

```bash
bash ci/run.sh ./tmp/results ./tmp/mnt
GG_BUILD_CUDA=1 bash ci/run.sh ./tmp/results ./tmp/mnt
GG_BUILD_METAL=1 bash ci/run.sh ./tmp/results ./tmp/mnt
```

It can download substantial model data and rebuild disposable `build-ci-*`
directories. It is not the lightweight PR gate.
