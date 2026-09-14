import json
import os
from datetime import datetime
from pathlib import Path

base = Path(os.environ["USERPROFILE"]) / ".cursor/projects/c-Masters-Codes-Repo-Surrogate-Prop-hull/agent-transcripts"
files = sorted(base.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
files = [f for f in files if "subagents" not in str(f)]

for f in files:
    mtime = f.stat().st_mtime
    size_kb = f.stat().st_size / 1024
    first_user = None
    n_user = 0
    n_asst = 0
    with open(f, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = obj.get("role") or obj.get("type") or ""
            msg = obj.get("message") or obj
            if isinstance(msg, dict):
                role = msg.get("role") or role
                content = msg.get("content")
            else:
                content = obj.get("content")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, str):
                        parts.append(p)
                    elif isinstance(p, dict):
                        if p.get("type") == "text":
                            parts.append(p.get("text", ""))
                        elif "text" in p:
                            parts.append(p.get("text", ""))
                text = "\n".join(parts)
            role_l = str(role).lower()
            if role_l in ("user", "human") or obj.get("type") == "user":
                n_user += 1
                if first_user is None and text.strip():
                    first_user = " ".join(text.strip().split())[:220]
            elif role_l in ("assistant", "ai") or obj.get("type") == "assistant":
                n_asst += 1
    dt = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    print("===")
    print(f"id: {f.parent.name}")
    print(f"date: {dt}  size: {size_kb:.1f} KB  msgs: user={n_user} asst={n_asst}")
    print(f"topic: {first_user or '(empty)'}")
