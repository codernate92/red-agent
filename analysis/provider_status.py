"""Probe registry models once and record reachability status."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from core.model_registry import list_aliases, resolve
from core.targets import create_target


async def probe_alias(alias: str) -> dict[str, Any]:
    try:
        target = create_target(resolve(alias))
        response = await target.query("Reply with exactly OK.")
        return {
            "spec": alias,
            "provider": target.provider,
            "model": target.model,
            "ok": True,
            "response_preview": response.text[:120],
        }
    except Exception as exc:
        return {
            "spec": alias,
            "ok": False,
            "error": str(exc),
        }


async def build_status_matrix(aliases: list[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alias in aliases or list_aliases():
        rows.append(await probe_alias(alias))
    return rows


def main() -> None:
    load_dotenv(Path(".env"))

    parser = argparse.ArgumentParser(description="Probe model reachability for all registry aliases.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the JSON status matrix.",
    )
    parser.add_argument(
        "--aliases",
        default=None,
        help="Optional comma-separated alias subset.",
    )
    args = parser.parse_args()

    aliases = None
    if args.aliases:
        aliases = [alias.strip() for alias in args.aliases.split(",") if alias.strip()]

    rows = asyncio.run(build_status_matrix(aliases))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
