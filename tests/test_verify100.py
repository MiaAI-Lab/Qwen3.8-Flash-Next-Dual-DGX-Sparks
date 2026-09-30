"""A file reference that starts with d is not a data URI."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "html"))
import verify100


class ExternalRefTests(unittest.TestCase):
    def test_a_file_src_that_starts_with_d_is_reported(self):
        self.assertEqual(
            verify100.external_refs('<img src="draw.js">'),
            ['attr src="draw.js"'],
        )
        self.assertEqual(
            verify100.external_refs('<link href="docs/style.css">'),
            ['attr href="docs/style.css"'],
        )
        self.assertEqual(
            verify100.external_refs('<img src="Draw.js">'),
            ['attr src="Draw.js"'],
        )

    def test_data_uri_fragment_and_javascript_stay_quiet(self):
        self.assertEqual(
            verify100.external_refs('<img src="data:image/png;base64,aaaa">'),
            [],
        )
        self.assertEqual(
            verify100.external_refs('<img src="DATA:image/png;base64,aaaa">'),
            [],
        )
        self.assertEqual(verify100.external_refs('<a href="#top">'), [])
        self.assertEqual(
            verify100.external_refs('<a href="javascript:void(0)">'),
            [],
        )

    def test_other_file_src_and_an_svg_are_unchanged(self):
        self.assertEqual(
            verify100.external_refs('<img src="app.js">'),
            ['attr src="app.js"'],
        )
        self.assertEqual(verify100.external_refs('<img src="icon.svg">'), [])


if __name__ == "__main__":
    unittest.main()
