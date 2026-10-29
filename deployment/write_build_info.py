import argparse
import datetime as dt
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--application', required=True)
parser.add_argument('--application-commit', required=True)
parser.add_argument('--foundation-commit', required=True)
args = parser.parse_args()
args.output.write_text(json.dumps({
    'application': args.application,
    'application_commit': args.application_commit,
    'edos_foundation_commit': args.foundation_commit,
    'deployed_at': dt.datetime.now(dt.timezone.utc).isoformat(),
}, indent=2) + '\n', encoding='utf-8')
