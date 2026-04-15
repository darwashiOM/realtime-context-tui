#!/usr/bin/env python3
"""Diagnostic: send one message to `claude --resume` in stream-json mode
and dump every line of stdout to see the exact schema."""

import argparse
import asyncio
import json
import sys
from pathlib import Path


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--message", default="ping: please reply with the single word 'pong'.")
    args = ap.parse_args()

    proc = await asyncio.create_subprocess_exec(
        "claude",
        "--resume", args.session_id,
        "--print",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--model", args.model,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdin and proc.stdout and proc.stderr

    # Try two candidate input schemas - both documented in recent Claude Code versions.
    # We'll use the newer schema and log if it errors.
    user_msg = {
        "type": "user",
        "message": {"role": "user", "content": args.message},
    }
    proc.stdin.write((json.dumps(user_msg) + "\n").encode())
    await proc.stdin.drain()
    proc.stdin.close()

    print("--- STDOUT (one JSON object per line) ---", flush=True)
    async for line in proc.stdout:
        decoded = line.decode(errors="replace").rstrip()
        print(decoded, flush=True)

    rc = await proc.wait()
    err = (await proc.stderr.read()).decode(errors="replace")
    if err.strip():
        print("--- STDERR ---", flush=True)
        print(err, flush=True)
    print(f"--- exit={rc} ---", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
