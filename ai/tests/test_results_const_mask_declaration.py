import os
import tempfile
import unittest

import h5py
import numpy as np

from emulator.results import (
    _GRID2D_CLASS,
    _GRID2D_MASK_DECLARATION,
    _grid2d_const_mask_digest,
    _validate_grid2d_const_mask_declaration,
    save_emulator,
)


class Grid2DConstMaskDeclarationTests(unittest.TestCase):
    def test_caller_cannot_replace_writer_owned_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "forged-declaration")
            with self.assertRaisesRegex(KeyError, "reserved key"):
                save_emulator(
                    path_root=root,
                    model=None,
                    param_geometry=None,
                    geometry=None,
                    config={},
                    histories={},
                    attrs={_GRID2D_MASK_DECLARATION: "forged"},
                )
            self.assertFalse(os.path.exists(root + ".emul"))
            self.assertFalse(os.path.exists(root + ".h5"))

    def test_digest_covers_pin_order_not_only_true_count(self):
        first = np.asarray([1, 0, 0, 0], dtype=np.uint8)
        moved = np.asarray([0, 1, 0, 0], dtype=np.uint8)

        self.assertEqual(int(first.sum()), int(moved.sum()))
        self.assertNotEqual(
            _grid2d_const_mask_digest(first),
            _grid2d_const_mask_digest(moved),
        )

    def test_matching_declaration_passes_and_one_surface_tamper_refuses(self):
        with h5py.File("mask.h5", "w", driver="core", backing_store=False) as f:
            group = f.create_group("dv_geometry")
            group.attrs["cls"] = _GRID2D_CLASS
            mask = np.asarray([0, 0, 0], dtype=np.uint8)
            group.create_dataset("const_mask", data=mask)
            f.attrs[_GRID2D_MASK_DECLARATION] = (
                _grid2d_const_mask_digest(mask)
            )

            _validate_grid2d_const_mask_declaration(
                group=group,
                declaration_attrs=f.attrs,
                where="test dv_geometry",
            )
            group["const_mask"][0] = np.uint8(1)
            with self.assertRaisesRegex(ValueError, "const_mask.*declaration"):
                _validate_grid2d_const_mask_declaration(
                    group=group,
                    declaration_attrs=f.attrs,
                    where="test dv_geometry",
                )

    def test_non_grid2d_rejects_even_when_mask_and_digest_match(self):
        with h5py.File("foreign.h5", "w", driver="core",
                       backing_store=False) as f:
            group = f.create_group("dv_geometry")
            group.attrs["cls"] = "emulator.geometries.scalar.ScalarGeometry"
            mask = np.asarray([0], dtype=np.uint8)
            group.create_dataset("const_mask", data=mask)
            f.attrs[_GRID2D_MASK_DECLARATION] = (
                _grid2d_const_mask_digest(mask)
            )

            with self.assertRaisesRegex(ValueError, "not a Grid2DGeometry"):
                _validate_grid2d_const_mask_declaration(
                    group=group,
                    declaration_attrs=f.attrs,
                    where="test dv_geometry",
                )

    def test_older_schema_v3_grid2d_without_declaration_is_refused(self):
        with h5py.File("legacy.h5", "w", driver="core",
                       backing_store=False) as f:
            group = f.create_group("dv_geometry")
            group.attrs["cls"] = _GRID2D_CLASS
            group.create_dataset(
                "const_mask", data=np.asarray([0], dtype=np.uint8)
            )

            with self.assertRaisesRegex(
                    KeyError, "schema-v3 Grid2D artifact.*re-save"):
                _validate_grid2d_const_mask_declaration(
                    group=group,
                    declaration_attrs=f.attrs,
                    where="test dv_geometry",
                )


if __name__ == "__main__":
    unittest.main()
