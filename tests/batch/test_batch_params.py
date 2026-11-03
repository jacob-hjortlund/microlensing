"""Tests of the batch description expansion:
python -m unittest tests.batch.test_batch_params -v"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from microlensing.IPM.batch_params import SEED_BLOCK, SEED_MAX, derive_seeds, expand  # noqa: E402
from microlensing.IPM.ipm import _INPUT_NAMES  # noqa: E402


class ExpandTests(unittest.TestCase):
    def test_scalar_shared_and_sequence_paired(self):
        maps = expand({"kappa_tot": 0.4, "shear": [0.1, 0.2]})
        self.assertEqual(len(maps), 2)
        self.assertEqual([m["kappa_tot"] for m in maps], [0.4, 0.4])
        self.assertEqual([m["shear"] for m in maps], [0.1, 0.2])

    def test_num_maps_sets_count(self):
        self.assertEqual(len(expand({"kappa_tot": 0.4}, num_maps=5)), 5)

    def test_num_maps_must_match_sequences(self):
        with self.assertRaises(ValueError):
            expand({"shear": [0.1, 0.2]}, num_maps=3)

    def test_bad_num_maps_and_empty_sequence(self):
        for n in (0, -1):
            with self.assertRaises(ValueError):
                expand({}, num_maps=n)
        with self.assertRaises(ValueError):
            expand({"shear": []})

    def test_single_map_by_default(self):
        self.assertEqual(len(expand({})), 1)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError) as cm:
            expand({"shear": [1, 2], "kappa_tot": [1, 2, 3]})
        self.assertIn("shear: 2", str(cm.exception))
        self.assertIn("kappa_tot: 3", str(cm.exception))

    def test_strings_are_scalars(self):
        maps = expand({"mass_function": "salpeter", "outfile_prefix": "run/"}, num_maps=2)
        self.assertEqual([m["mass_function"] for m in maps], ["salpeter"] * 2)
        self.assertEqual([m["outfile_prefix"] for m in maps], ["run/0_", "run/1_"])

    def test_numpy_and_range_are_sequences(self):
        maps = expand({"shear": np.array([0.1, 0.2]), "num_pixels_y1": range(1, 3)})
        self.assertEqual([m["shear"] for m in maps], [0.1, 0.2])
        self.assertEqual([m["num_pixels_y1"] for m in maps], [1, 2])
        self.assertIs(type(maps[0]["shear"]), float)
        self.assertIs(type(maps[0]["num_pixels_y1"]), int)
        maps = expand({"kappa_tot": np.array(0.3)}, num_maps=2)
        self.assertEqual([m["kappa_tot"] for m in maps], [0.3, 0.3])

    def test_no_shared_state_with_caller(self):
        inputs = {"shear": [0.1, 0.2]}
        maps = expand(inputs)
        inputs["shear"][0] = 9
        self.assertEqual(maps[0]["shear"], 0.1)

    def test_seed_sequence_used_as_given(self):
        maps = expand({"random_seed": [7, 7, 9]})
        self.assertEqual([m["random_seed"] for m in maps], [7, 7, 9])

    def test_seed_start_derives_seeds(self):
        # map i gets derive_seeds(seed_start, num_maps)[i]: distinct ints in 1..SEED_MAX, the same whatever num_maps
        s = [m["random_seed"] for m in expand({}, num_maps=2000, seed_start=42)]
        self.assertEqual(s, derive_seeds(42, 2000))
        self.assertEqual(len(set(s)), 2000)
        self.assertTrue(all(type(x) is int and 1 <= x <= SEED_MAX for x in s))
        self.assertEqual(derive_seeds(42, 3000)[:2000], s)
        self.assertEqual([m["random_seed"] for m in expand({}, num_maps=7, seed_start=np.int64(42))], s[:7])
        self.assertNotEqual(expand({}, seed_start=42)[0]["random_seed"], 42)   # seed_start is not a map's seed

    def test_seed_start_must_be_an_integer(self):
        # a value that is not an integer is refused rather than truncated (2.7 would give the seeds of 2)
        for bad in (2.7, 42.0, np.float64(42.0), "42", True):
            with self.subTest(seed_start=repr(bad)):
                with self.assertRaisesRegex(TypeError, "seed_start must be an integer"):
                    expand({}, num_maps=2, seed_start=bad)
        self.assertEqual([m["random_seed"] for m in expand({}, num_maps=3, seed_start=np.array(42))],
                         derive_seeds(42, 3))

    def test_scalar_seed_one_map_is_that_seed(self):
        self.assertEqual(expand({"random_seed": 42})[0]["random_seed"], 42)
        self.assertEqual(expand({"random_seed": 42}, num_maps=1)[0]["random_seed"], 42)
        self.assertEqual(expand({"random_seed": np.int64(42)})[0]["random_seed"], 42)
        self.assertIs(type(expand({"random_seed": np.int64(42)})[0]["random_seed"]), int)

    def test_scalar_seed_many_maps_raises(self):
        with self.assertRaisesRegex(ValueError, "a single random_seed is one map's seed; give a sequence, or "
                                                "seed_start= to derive one seed per map"):
            expand({"random_seed": 42}, num_maps=2)
        with self.assertRaisesRegex(ValueError, "a single random_seed is one map's seed"):
            expand({"random_seed": 42, "shear": [0.1, 0.2, 0.3]})

    def test_scalar_seed_must_be_in_range(self):
        for bad in (0, -1, SEED_MAX + 1, 1.5, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                expand({"random_seed": bad})

    def test_seed_start_and_random_seed_conflict(self):
        for seed in (42, [1, 2]):
            with self.assertRaisesRegex(ValueError, "random_seed and seed_start cannot both be given"):
                expand({"random_seed": seed}, num_maps=2, seed_start=7)

    def test_scalar_seed_ignores_starfile_maps(self):
        maps = expand({"starfile": [None, "a.bin", "b.bin"], "random_seed": 42})
        self.assertEqual([m["random_seed"] for m in maps], [42, None, None])

    def test_no_seed_draws_distinct(self):
        s = [m["random_seed"] for m in expand({}, num_maps=500)]
        self.assertEqual(len(set(s)), 500)
        self.assertTrue(all(1 <= x <= SEED_MAX for x in s))

    def test_starfile_maps_get_no_seed(self):
        maps = expand({"starfile": ["a.bin", None]}, seed_start=3)
        self.assertIsNone(maps[0]["random_seed"])
        self.assertIsNotNone(maps[1]["random_seed"])

    def test_prefixes(self):
        maps = expand({"outfile_prefix": "run1/"}, num_maps=2)
        self.assertEqual([m["outfile_prefix"] for m in maps], ["run1/0_", "run1/1_"])
        self.assertEqual(expand({})[0]["outfile_prefix"], "./0_")
        with self.assertRaises(ValueError):
            expand({"outfile_prefix": ["a/", "a/"]})

    def test_duplicate_prefixes_are_listed_once_each(self):
        with self.assertRaises(ValueError) as cm:
            expand({"outfile_prefix": ["b/", "a/", "b/", "c/", "a/", "b/"]})
        self.assertEqual(str(cm.exception), "duplicate outfile_prefix values: ['a/', 'b/']")

    def test_many_seeds_draw_only_the_blocks_they_need(self):
        # derive_seeds draws whole blocks of SEED_BLOCK values: 10^5 seeds take ceil(10^5 / SEED_BLOCK) blocks, and
        # one more only if the draws repeat values
        n = 10**5
        real_default_rng = np.random.default_rng
        rngs = []

        def default_rng(seed=None):
            rngs.append(mock.Mock(wraps=real_default_rng(seed)))
            return rngs[-1]

        with mock.patch("numpy.random.default_rng", default_rng):
            maps = expand({"kappa_tot": 0.4, "outfile_prefix": "run/"}, num_maps=n, seed_start=42)
        self.assertEqual(len(rngs), 1)
        self.assertLessEqual(rngs[0].integers.call_count, -(-n // SEED_BLOCK) + 1)
        self.assertEqual(len({m["random_seed"] for m in maps}), n)
        self.assertEqual(len({m["outfile_prefix"] for m in maps}), n)

    def test_duplicate_prefix_check_is_linear(self):
        # each prefix is compared with a few others at most, not with all of them (which took 10 s for 30,000
        # maps): the comparisons are counted
        compared = 0

        class Prefix(str):
            def __eq__(self, other):
                nonlocal compared
                compared += 1
                return str.__eq__(self, other)

            __hash__ = str.__hash__

        n = 30000
        maps = expand({"outfile_prefix": [Prefix(f"run/{i}_") for i in range(n)]})
        self.assertEqual(len(maps), n)
        self.assertLess(compared, n)

    def test_bad_seed_entries_raise(self):
        for bad in (None, 0, -1, 2**31, 3.5, True):
            with self.assertRaises(ValueError) as cm:
                expand({"random_seed": [5, bad]})
            self.assertIn("map 1", str(cm.exception))
        maps = expand({"random_seed": [np.int64(5), SEED_MAX]})
        self.assertEqual([m["random_seed"] for m in maps], [5, SEED_MAX])
        self.assertTrue(all(type(m["random_seed"]) is int for m in maps))

    def test_starfile_map_accepts_none_seed_entry(self):
        maps = expand({"starfile": ["a.bin", None], "random_seed": [None, 4]})
        self.assertEqual([m["random_seed"] for m in maps], [None, 4])

    def test_none_prefix_entry_raises(self):
        with self.assertRaises(ValueError) as cm:
            expand({"outfile_prefix": ["a/", None]})
        self.assertIn("map 1", str(cm.exception))

    def test_unknown_argument_raises(self):
        with self.assertRaises(TypeError):
            expand({"kapa": 1})

    def test_every_input_present(self):
        self.assertEqual(set(expand({})[0]), set(_INPUT_NAMES))


if __name__ == "__main__":
    unittest.main()
