"""No-model regression checks; these do not certify CUDA or audio inference."""

import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("token2wav_service", ROOT / "token2wav_service.py")
assert SPEC is not None and SPEC.loader is not None
SERVICE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVICE)


class ProtocolTests(unittest.TestCase):
    def test_missing_model_is_rejected_even_with_stale_flag(self):
        service = SERVICE.Token2WavService()
        for initialized in (False, True):
            service.initialized = initialized
            self.assertEqual(service.set_ref_audio("missing.wav")["status"], "error")
            self.assertEqual(service.process([1], True, "unused.wav")["status"], "error")

    def test_json_lines_protocol_and_quit(self):
        commands = ['not json', json.dumps({"cmd": "process", "tokens": [1]}),
                    json.dumps({"cmd": "unknown"}), json.dumps({"cmd": "quit"})]
        result = subprocess.run([sys.executable, str(ROOT / "token2wav_service.py")],
                                input="\n".join(commands) + "\n", capture_output=True,
                                text=True, check=True, timeout=30)
        responses = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([r["status"] for r in responses], ["ready", "error", "error", "error", "ok"])
        self.assertEqual(responses[-1]["message"], "Goodbye")

    def test_pcm_writer_clips_and_preserves_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            SERVICE.Token2WavService()._write_wav(str(path), np.array([0., 2., -2.]), 24000)
            with wave.open(str(path), "rb") as audio:
                self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, 24000))
                self.assertEqual(struct.unpack("<3h", audio.readframes(3)), (0, 32767, -32767))


if __name__ == "__main__":
    unittest.main()
