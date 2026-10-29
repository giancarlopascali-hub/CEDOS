import pathlib
import shutil
import subprocess
import unittest


NODE_AVAILABLE = shutil.which('node') is not None


@unittest.skipUnless(NODE_AVAILABLE, 'Node.js is required for JavaScript polling tests')
class BackgroundPollingTests(unittest.TestCase):
    def test_gateway_error_during_submit_safely_retries_same_request(self):
        utils_uri = (
            pathlib.Path(__file__).parents[1] / 'static/js/utils.js'
        ).resolve().as_uri()
        script = f"""
globalThis.document = {{}};
globalThis.window = {{}};
const requestIds = [];
let submissions = 0;
globalThis.fetch = async (url, options = {{}}) => {{
  if (url === '/submit') {{
    submissions += 1;
    requestIds.push(options.headers['X-Analysis-Request-ID']);
    if (submissions === 1) return new Response('<html>gateway</html>', {{status: 502}});
    return new Response(JSON.stringify({{job_id: 'original-job', status: 'running'}}), {{status: 202}});
  }}
  if (String(url).startsWith('/job_status/')) {{
    return new Response(JSON.stringify({{status: 'completed', result: {{ok: true}}}}), {{status: 200}});
  }}
  throw new Error(`Unexpected URL: ${{url}}`);
}};
const mod = await import('{utils_uri}');
const completed = await mod.submitAndPollJob('/submit', {{}}, {{
  pollMilliseconds: 1,
  submitRetryBaseMilliseconds: 1,
  submitFailureGraceMilliseconds: 1000
}});
if (!completed.result.ok) throw new Error('Result was not returned');
if (submissions !== 2) throw new Error(`Expected 2 submissions, got ${{submissions}}`);
if (!requestIds[0] || requestIds[0] !== requestIds[1]) {{
  throw new Error('Submission retry did not preserve its idempotency key');
}}
"""
        completed = subprocess.run(
            ['node', '--input-type=module', '--eval', script],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_transient_gateway_error_keeps_polling_without_cancelling(self):
        utils_uri = (
            pathlib.Path(__file__).parents[1] / 'static/js/utils.js'
        ).resolve().as_uri()
        script = f"""
globalThis.document = {{}};
globalThis.window = {{}};
const calls = [];
let statusPolls = 0;
globalThis.fetch = async (url, options = {{}}) => {{
  calls.push([String(url), options.method || 'GET']);
  if (url === '/submit') {{
    return new Response(JSON.stringify({{job_id: 'job-1', status: 'queued'}}), {{status: 202}});
  }}
  if (String(url).startsWith('/job_status/')) {{
    statusPolls += 1;
    if (statusPolls === 1) return new Response('<html>gateway</html>', {{status: 502}});
    return new Response(JSON.stringify({{status: 'completed', result: {{ok: true}}}}), {{status: 200}});
  }}
  if (String(url).startsWith('/job_cancel/')) {{
    return new Response(JSON.stringify({{status: 'cancelling'}}), {{status: 200}});
  }}
  throw new Error(`Unexpected URL: ${{url}}`);
}};
const mod = await import('{utils_uri}');
const completed = await mod.submitAndPollJob('/submit', {{}}, {{
  pollMilliseconds: 1,
  pollFailureGraceMilliseconds: 1000
}});
if (!completed.result.ok) throw new Error('Result was not returned');
if (statusPolls !== 2) throw new Error(`Expected 2 status polls, got ${{statusPolls}}`);
if (calls.some(([url]) => url.startsWith('/job_cancel/'))) {{
  throw new Error('Transient 502 incorrectly cancelled the job');
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
