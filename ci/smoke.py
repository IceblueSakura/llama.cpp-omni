#!/usr/bin/env python3
"""Model-free CLI and Omni protocol checks for native and packaged builds."""

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


HELP = {
    "llama-omni-cli": (0, "MiniCPM-o Omni CLI"),
    "voxcpm2-cli": (1, "VoxCPM2"),  # --help intentionally returns 1
    "llama-omni-server": (0, "--port"),
    "llama-tts-server": (0, "--voxcpm2-base-lm"),
}


def binary(directory, name):
    path = directory / (name + (".exe" if os.name == "nt" else ""))
    if not path.is_file():
        raise RuntimeError(f"Missing binary: {path}")
    return path.resolve()


def run_checked(command, code, banner):
    result = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=30)
    output = result.stdout + result.stderr
    if result.returncode != code or banner not in output:
        raise RuntimeError(f"{command[0]}: expected exit {code} and {banner!r}, got {result.returncode}\n{output}")


def validate_health(status, body):
    if status != 200 or body.get("status") != "ok" or body.get("engine") != "comni":
        raise RuntimeError(f"Invalid Omni health response: {status}, {body}")


def validate_missing_model(status, body):
    error = body.get("error", {})
    if status < 400 or "missing required model file (LLM)" not in error.get("message", ""):
        raise RuntimeError(f"Missing-model request did not exercise Omni validation: {status}, {body}")


def request_json(opener, url, data=None):
    request = urllib.request.Request(
        url, data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        response = opener.open(request, timeout=2)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


def smoke(directory, tls=False):
    for name, (code, banner) in HELP.items():
        run_checked([str(binary(directory, name)), "--help"], code, banner)
        logging.info("PASS: %s --help", name)

    run_checked([str(binary(directory, "llama-tts-server"))], 1, "No VoxCPM2 models specified")
    logging.info("PASS: TTS server rejects missing weights")

    logs = Path("build/omni-smoke-logs")
    logs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="omni-smoke-") as temporary:
        root = Path(temporary)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command = [str(binary(directory, "llama-omni-server")), "--host", "127.0.0.1", "--port", str(port),
                   "-m", str(root / "missing.gguf"), "--threads-http", "2"]
        handlers: list[urllib.request.BaseHandler] = [urllib.request.ProxyHandler({})]
        scheme = "http"
        if tls:
            # Disposable localhost certificate: never use repository/user credentials.
            openssl = shutil.which("openssl")
            if not openssl:
                raise RuntimeError("openssl is required to generate the smoke-test certificate")
            cert, key = root / "cert.pem", root / "key.pem"
            subprocess.run([
                openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                "-subj", "/CN=localhost", "-addext", "subjectAltName=IP:127.0.0.1",
                "-keyout", str(key), "-out", str(cert),
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            command += ["--ssl-cert-file", str(cert), "--ssl-key-file", str(key)]
            handlers.append(urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=str(cert))))
            scheme = "https"
        opener = urllib.request.build_opener(*handlers)
        base = f"{scheme}://127.0.0.1:{port}"
        with (logs / "omni-server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 30
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Omni server exited with {process.returncode}; see {logs}")
                    try:
                        status, body = request_json(opener, base + "/health")
                        break
                    except (urllib.error.URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError(f"Omni server startup timed out; see {logs}")
                        time.sleep(0.1)
                validate_health(status, body)
                validate_health(*request_json(opener, base + "/v1/health"))
                logging.info("PASS: Omni health endpoints")
                validate_missing_model(*request_json(opener, base + "/v1/stream/omni_init", {
                    "media_type": 1, "use_tts": False,
                }))
                validate_health(*request_json(opener, base + "/health"))
                logging.info("PASS: Omni rejects missing model and remains healthy")
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--tls", action="store_true", help="Verify TLS-enabled build with a disposable certificate")
    args = parser.parse_args()
    smoke(args.bin_dir, args.tls)
