#!/usr/bin/env python3
"""Reproduce checked-in vendor snapshots; --check never writes to the repository.

cpp-httplib is header-only in this project. Its existing Windows initialization
and TLS Expect/100-continue fixes are preserved by vendor-cpp-httplib.patch.
Updating these sources/patches is a separate, reviewed dependency upgrade.
"""

import argparse
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
HTTPLIB_VERSION = "008e107d0fcddee7cb96dc5ad3c3189fd090e40a"  # v0.46.0
STB_VERSION = "013ac3beddff3dbffafd5177e7972067cd2b5083"

VENDOR = {
    "https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp": "vendor/nlohmann/json.hpp",
    "https://github.com/nlohmann/json/releases/download/v3.12.0/json_fwd.hpp": "vendor/nlohmann/json_fwd.hpp",
    f"https://raw.githubusercontent.com/nothings/stb/{STB_VERSION}/stb_image.h": "vendor/stb/stb_image.h",
    "https://raw.githubusercontent.com/mackron/miniaudio/9634bedb5b5a2ca38c1ee7108a9358a4e233f14d/miniaudio.h":
        "vendor/miniaudio/miniaudio.h",
    f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/httplib.h": "vendor/cpp-httplib/httplib.h",
    f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/LICENSE": "vendor/cpp-httplib/LICENSE",
    "https://raw.githubusercontent.com/sheredom/subprocess.h/b49c56e9fe214488493021017bf3954b91c7c1f5/subprocess.h":
        "vendor/sheredom/subprocess.h",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare without replacing repository files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with tempfile.TemporaryDirectory(prefix="llama-vendor-") as directory:
        staging = Path(directory)
        for url, filename in VENDOR.items():
            logging.info("Fetching %s", filename)
            destination = staging / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as response:
                destination.write_bytes(response.read())
        subprocess.run(["git", "apply", "--unidiff-zero", str(ROOT / "scripts/vendor-cpp-httplib.patch")],
                       cwd=staging, check=True)
        mismatches = []
        for filename in VENDOR.values():
            expected, actual = staging / filename, ROOT / filename
            if args.check:
                if not actual.is_file() or actual.read_bytes() != expected.read_bytes():
                    mismatches.append(filename)
            else:
                shutil.copyfile(expected, actual)
        if mismatches:
            raise SystemExit("Vendor snapshots differ:\n" + "\n".join(mismatches))
        logging.info("Vendor snapshots %s", "verified" if args.check else "updated")


if __name__ == "__main__":
    main()
