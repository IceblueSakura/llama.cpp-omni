#!/usr/bin/env python3
"""Package the tested Omni tools and runtime libraries, then extract for smoke tests."""

import argparse
import logging
from pathlib import Path
import re
import zipfile


TOOLS = ("llama-omni-cli", "llama-omni-server", "llama-tts-server", "voxcpm2-cli", "llama", "llama-completion")


def package(bin_dir, name, staging=Path("build-ci-package"), output=Path("dist")):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError("Archive name must not contain paths or shell metacharacters")
    files = [p for p in bin_dir.iterdir() if p.is_file() and (
        p.stem in TOOLS or p.suffix in (".dll", ".dylib", ".metal") or ".so" in p.name
    )]
    for tool in TOOLS:
        if not any(p.name in (tool, tool + ".exe") for p in files):
            raise RuntimeError(f"Missing release tool: {tool}")
    if staging.exists():
        raise RuntimeError(f"Extraction directory already exists: {staging}; choose a fresh build directory")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / (name + ".zip")
    if archive.exists():
        raise RuntimeError(f"Archive already exists: {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for source in files:
            # Dereference runtime-library symlinks so the archive works on Windows too.
            zipped.write(source, source.name)
        zipped.write("LICENSE", "LICENSE")
        for source in Path("licenses").glob("LICENSE-*"):
            zipped.write(source, "licenses/" + source.name)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(staging)
        for info in zipped.infolist():
            permissions = info.external_attr >> 16
            if permissions:
                (staging / info.filename).chmod(permissions)
    logging.info("Packaged %s; extracted binaries in %s", archive, staging)
    return archive


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    package(args.bin_dir, args.name)
