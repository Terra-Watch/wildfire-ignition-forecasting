"""Bundle golden test: needs a built bundle, given by RISK_TEST_BUNDLE (skipped otherwise)."""
import os, shutil, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "website", "forecast", "live"))
from bundle import Bundle  # noqa: E402

ROOT = os.environ.get("RISK_TEST_BUNDLE")


@unittest.skipUnless(ROOT and os.path.isdir(ROOT), "RISK_TEST_BUNDLE not set")
class BundleTests(unittest.TestCase):
    def test_golden_matches_stored_study_predictions(self):
        self.assertLess(Bundle(ROOT).golden(), 1e-6)

    def test_tampered_file_is_refused(self):
        d = tempfile.mkdtemp(); shutil.copytree(ROOT, d, dirs_exist_ok=True)
        with open(f"{d}/manifest.json", "a") as f: f.write(" ")
        with self.assertRaises(ValueError): Bundle(d)

    def test_signature_is_checked_against_the_trusted_key(self):
        import json
        sig = json.load(open(f"{ROOT}/SIGNATURE.json"))
        self.assertEqual(Bundle(ROOT, sig["public_key"]).signed_by, sig["key_id"])
        with self.assertRaises(ValueError): Bundle(ROOT, "AAAA" + sig["public_key"][4:])      # a different key is refused

    def test_licence_notices_are_listed(self):
        self.assertGreaterEqual(len(Bundle(ROOT).manifest["licence_notices"]), 8)

    def test_score_shape_and_range(self):
        import numpy as np
        b = Bundle(ROOT); z = np.load(f"{ROOT}/parity_cases.npz"); p = b.score(z["raw"])
        self.assertEqual(p.shape, (len(z["raw"]),)); self.assertTrue(((p >= 0) & (p <= 1)).all())


if __name__ == "__main__": unittest.main()
