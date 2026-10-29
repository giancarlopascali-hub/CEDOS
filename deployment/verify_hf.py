import argparse
import json
import time
import urllib.request


parser = argparse.ArgumentParser()
parser.add_argument('--space', required=True)
parser.add_argument('--hf-commit', required=True)
parser.add_argument('--application-commit', required=True)
parser.add_argument('--timeout', type=int, default=900)
args = parser.parse_args()
deadline = time.time() + args.timeout
last = 'deployment not visible yet'
while time.time() < deadline:
    try:
        with urllib.request.urlopen(f'https://huggingface.co/api/spaces/{args.space}', timeout=30) as response:
            metadata = json.load(response)
        stage = (metadata.get('runtime') or {}).get('stage')
        sha = metadata.get('sha')
        host = metadata.get('host')
        last = f'stage={stage}, sha={sha}'
        if stage == 'RUNNING' and sha == args.hf_commit and host:
            with urllib.request.urlopen(f'{host}/health', timeout=30) as response:
                health = json.load(response)
            with urllib.request.urlopen(f'{host}/version', timeout=30) as response:
                version = json.load(response)
            if health.get('status') == 'ok' and version.get('application_commit') == args.application_commit:
                print(f'Verified {args.space} at {sha}: {version}')
                raise SystemExit(0)
            last = f'health={health}, version={version}'
    except Exception as exc:
        last = str(exc)
    print(f'Waiting for {args.space}: {last}')
    time.sleep(10)
raise SystemExit(f'Deployment verification timed out: {last}')
