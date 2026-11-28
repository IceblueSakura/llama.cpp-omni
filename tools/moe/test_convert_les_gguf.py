"""Offline regression tests for lossless LES packing."""
import importlib.util
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("convert_les", Path(__file__).with_name("convert_les_gguf.py"))
les = importlib.util.module_from_spec(spec)
spec.loader.exec_module(les)


class TestLayout(unittest.TestCase):
    def source(self, fused=False):
        shapes = {"ffn_down_exps": (8, 16, 4)}
        shapes.update({"ffn_gate_up_exps": (16, 16, 4)} if fused else {
            "ffn_gate_exps": (16, 8, 4), "ffn_up_exps": (16, 8, 4)})
        tensors = []
        for index, (family, shape) in enumerate(shapes.items()):
            data = np.arange(np.prod(shape), dtype=np.float32) + index * 10000
            tensors.append(SimpleNamespace(name=f"blk.0.{family}.weight", shape=shape,
                n_bytes=data.nbytes, data=data, tensor_type=les.GGMLQuantizationType.F32))
        return SimpleNamespace(tensors=tensors, fields={
            "general.architecture": SimpleNamespace(contents=lambda: "qwen35moe"),
            "qwen35moe.expert_used_count": SimpleNamespace(contents=lambda: 2)})

    def test_split_and_fused_preserve_family_geometry_and_bytes(self):
        for fused in (False, True):
            with self.subTest(fused=fused):
                source = self.source(fused)
                expected = {t.name.split(".")[2]: t for t in source.tensors}
                experts, _, _, count, used = les.scan(source)
                layout = les.build_layout(experts, count, used)
                les.check_layout(layout, None)
                families = les.layer_layout(experts[0])
                packed = les.interleave(families, count)
                declared = layout["layers"][0]["groups"][0]["tensors"]
                self.assertCountEqual([entry["tensor_family"] for entry in declared], expected)
                for entry in declared:
                    tensor = expected[entry["tensor_family"]]
                    size = tensor.n_bytes // count
                    self.assertEqual(entry["bytes_per_expert"], size)
                    self.assertEqual(entry["ne"], list(tensor.shape[:2]))
                    original = tensor.data.view(np.uint8).reshape(count, size)
                    offset = entry["offset_in_expert"]
                    np.testing.assert_array_equal(packed[:, offset:offset+size], original)

    def test_missing_down_is_rejected(self):
        for fused in (False, True):
            with self.subTest(fused=fused):
                source = self.source(fused)
                source.tensors = [t for t in source.tensors if "down" not in t.name]
                with self.assertRaises(les.PackError):
                    les.scan(source)


    def test_invalid_layouts_are_rejected(self):
        experts, _, _, count, used = les.scan(self.source())
        layout = les.build_layout(experts, count, used)
        cases = [
            ("overlap", lambda j: j["layers"][0]["groups"][0]["tensors"][1].update(offset_in_expert=0)),
            ("negative", lambda j: j["layers"][0]["groups"][0]["tensors"][0].update(offset_in_expert=-32)),
            ("fractional count", lambda j: j["layers"][0].update(n_expert=4.5)),
            ("narrowed count", lambda j: j["layers"][0].update(n_expert=2**32 + count)),
            ("stride overflow", lambda j: j["layers"][0]["groups"][0].update(expert_stride=2**64)),
            ("zero size", lambda j: j["layers"][0]["groups"][0]["tensors"][0].update(bytes_per_expert=0)),
            ("unaligned", lambda j: j["layers"][0]["groups"][0]["tensors"][0].update(offset_in_expert=1)),
            ("version", lambda j: j.update(les_version=2)),
            ("alignment", lambda j: j.update(alignment=3000)),
            ("duplicate carrier", lambda j: j["layers"].append(dict(j["layers"][0], layer=1))),
        ]
        for name, mutate in cases:
            with self.subTest(name=name):
                invalid = copy.deepcopy(layout)
                mutate(invalid)
                with self.assertRaises(les.PackError):
                    les.check_layout(invalid, None)

    def test_carrier_type_size_and_prefix_are_checked(self):
        experts, _, _, count, used = les.scan(self.source())
        layout = les.build_layout(experts, count, used)
        group = layout["layers"][0]["groups"][0]
        carrier = SimpleNamespace(tensor_type=les.GGMLQuantizationType.I8,
            n_bytes=group["expert_stride"] * count, data_offset=4096)
        les.check_layout(layout, {group["tensor_name"]: carrier})
        for name, fields in [
            ("missing", {}),
            ("undersized", {"n_bytes": carrier.n_bytes - 32}),
            ("wrong type", {"tensor_type": les.GGMLQuantizationType.F32}),
            ("unaligned offset", {"data_offset": 4097}),
        ]:
            with self.subTest(name=name):
                invalid = copy.copy(carrier)
                invalid.__dict__.update(fields)
                tensors = {} if name == "missing" else {group["tensor_name"]: invalid}
                with self.assertRaises(les.PackError):
                    les.check_layout(layout, tensors)
        with self.assertRaises(les.PackError):
            les.check_layout(layout, {group["tensor_name"]: carrier,
                "resident": SimpleNamespace(data_offset=8192, n_bytes=32)})
        with self.assertRaises(les.PackError):
            les.check_layout(layout, {group["tensor_name"]: carrier}, 2048)
        legacy = copy.deepcopy(layout)
        legacy.pop("alignment")
        legacy_carrier = copy.copy(carrier)
        legacy_carrier.data_offset = 2048
        les.check_layout(legacy, {group["tensor_name"]: legacy_carrier}, 2048)


if __name__ == "__main__":
    unittest.main()
