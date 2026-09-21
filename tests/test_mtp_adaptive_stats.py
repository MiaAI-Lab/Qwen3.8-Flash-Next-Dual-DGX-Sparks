import json
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench/mtp_adaptive'))
from analyze import analyze, interval


class Tests(unittest.TestCase):
    def test_constant_pair_improvement(self):
        import math
        bounds = interval([math.log(1.1)] * 6, iterations=1000)
        self.assertAlmostEqual(bounds[0], 10)
        self.assertAlmostEqual(bounds[1], 10)

    def test_partial_data_not_pass(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'state.json').write_text('{"complete":false,"phase":"running"}')
            result = analyze(p)
            self.assertFalse(result['all_cells_positive_evidence'])
            self.assertEqual(result['complete_waves'], 0)


if __name__ == '__main__':
    unittest.main()
