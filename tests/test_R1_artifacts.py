"""Regression checks for honest interval precision and generated TeX."""

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments' / 'revision_R1_study'))
from build_artifacts import interval, table, wilson


class ArtifactTests(unittest.TestCase):
    def test_zero_events_do_not_round_upper_interval_to_zero(self):
        value = interval(0, 20000)
        self.assertIn('0.000192', value)
        self.assertEqual(interval(0, 0), 'NA')
        self.assertTrue(value.startswith('0 [0,'))
        self.assertEqual(wilson(0, 500)[0], 0)
        self.assertEqual(wilson(500, 500)[1], 1)
        self.assertNotIn('e-', interval(0, 500))

    def test_math_headers_keep_backslashes_and_no_control_characters(self):
        with tempfile.TemporaryDirectory() as directory:
            table(Path(directory), 'test.tex', 'tab:test', 'Test',
                  ['Input', r'$\rho$', r'$N_{A^\top}$'], [['a', 1, 2]])
            content = (Path(directory) / 'test.tex').read_bytes()
            self.assertTrue(all(byte >= 32 or byte == 10 for byte in content))
            self.assertIn(b'\\top', content)
            self.assertIn(b'\\small', content)
            self.assertIn(b'\\begin{adjustbox}{max width=\\linewidth}', content)
            self.assertIn(b'\\end{tabular}\n  \\end{adjustbox}', content)
            self.assertNotIn(b'\\resizebox', content)


if __name__ == '__main__':
    unittest.main()
