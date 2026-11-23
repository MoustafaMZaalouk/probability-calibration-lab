import unittest
import tempfile
import json
from pathlib import Path
from demo import run

class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.output=Path(cls.temp.name)/'results.json'
        cls.r=run(cls.output)
    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
    def test_output_matches_returned_result(self):
        self.assertEqual(json.loads(self.output.read_text()),self.r)
    def test_ece_perfect(self):
        from demo import ece
        import numpy as np
        self.assertEqual(ece(np.array([0,1]),np.array([0.,1.])),0.)
    def test_split_sizes(self):
        self.assertEqual(self.r['train_rows']+self.r['calibration_rows']+self.r['test_rows'],3000)
    def test_prob_metrics(self):
        self.assertTrue(all(0<=m['brier']<=1 for m in self.r['models'].values()))

if __name__ == "__main__":
    unittest.main()
