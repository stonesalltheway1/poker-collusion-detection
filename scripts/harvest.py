"""Harvest all S6E8 discussion threads (+ index) and extract the public leaderboard.

Handles two Kaggle CLI 2.2.4 quirks: a trailing "Next Page Token = X" line after the
JSON array, and leaderboard zip entries whose names contain ':' (illegal on Windows).
"""
import json
import re
import subprocess
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent   # scripts/ -> comp root
OUT = BASE / "research" / "discussions"
OUT.mkdir(parents=True, exist_ok=True)
COMP = "detect-suspicious-value-transfers-in-poker"   # e.g. playground-series-s6e9


def run(args):
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout


def parse_json_with_token(text):
    m = re.search(r"Next Page Token = (\S+)", text)
    token = m.group(1) if m else None
    end = text.rfind("]")
    return json.loads(text[: end + 1]), token


topics, token, page = [], None, 0
while True:
    args = ["kaggle", "competitions", "topics", "list", COMP, "--format", "json"]
    if token:
        args += ["--page-token", token]
    batch, token = parse_json_with_token(run(args))
    topics.extend(batch)
    page += 1
    if not token or page > 30:
        break

print(f"topics: {len(topics)}")
(OUT / "_topics_index.json").write_text(json.dumps(topics, indent=1), encoding="utf-8")

for t in topics:
    f = OUT / f"{t['id']}.txt"
    if f.exists():
        continue
    msgs = run(["kaggle", "competitions", "topic-messages", COMP, str(t["id"]),
                "-n", "-1", "--format", "json"])
    header = f"TITLE: {t.get('title')}\nVOTES: {t.get('votes')}  COMMENTS: {t.get('commentCount')}  DATE: {t.get('postDate')}\n=====\n"
    f.write_text(header + msgs, encoding="utf-8")

print(f"saved: {len(list(OUT.glob('*.txt')))} threads")

# leaderboard zip with ':' in member name
lb_zip = BASE / "research" / f"{COMP}.zip"
lb_dir = BASE / "research" / "leaderboard"
lb_dir.mkdir(exist_ok=True)
with zipfile.ZipFile(lb_zip) as z:
    for info in z.infolist():
        safe = re.sub(r"[:*?\"<>|]", "-", info.filename)
        (lb_dir / safe).write_bytes(z.read(info))
        print("extracted:", safe)
