import pathlib
import shutil
import subprocess
import unittest


NODE_AVAILABLE = shutil.which('node') is not None


@unittest.skipUnless(NODE_AVAILABLE, 'Node.js is required for JavaScript utility tests')
class ColumnDetectionTests(unittest.TestCase):
    def test_complete_values_must_be_numeric(self):
        utils_uri = (
            pathlib.Path(__file__).parents[1] / 'static/js/utils.js'
        ).resolve().as_uri()
        script = f"""
globalThis.document = {{}};
globalThis.window = {{}};
const mod = await import('{utils_uri}');
const cases = [
  [[1, '2.5', ' 3e2 '], true],
  [['10 mg', '20 mg'], false],
  [['1st batch', '2nd batch'], false],
  [['', null, '  '], false],
  [['1', 'two', '3'], false]
];
for (const [values, expected] of cases) {{
  const actual = mod.detectColumnIsNumeric(values);
  if (actual !== expected) throw new Error(`${{JSON.stringify(values)}}: ${{actual}}`);
}}
"""
        completed = subprocess.run(
            ['node', '--input-type=module', '--eval', script],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
    unittest.main()
