"""CLI: python -m mirobench.leaderboard {build,render,update,check}

  build   experiments/ + seed  ->  docs/leaderboard.json
  render  docs/leaderboard.json ->  docs/leaderboard.html  (in place)
  update  build + render        (the normal one-shot)
  check   build + render to memory; exit 1 if docs/ would change
          (CI guard so generated files are never stale in a PR)
"""
from __future__ import annotations

import sys

from . import build as build_mod
from . import render as render_mod


def _cmd_build() -> int:
    return build_mod.main()


def _cmd_render() -> int:
    return render_mod.main()


def _cmd_update() -> int:
    build_mod.write(build_mod.build())
    return render_mod.main()


def _cmd_check() -> int:
    import json

    data = build_mod.build(log=lambda *_: None)
    new_json = json.dumps(data, indent=2) + "\n"
    cur_json = (build_mod.OUTPUT_PATH.read_text()
                if build_mod.OUTPUT_PATH.exists() else "")

    build_mod.write(data)  # render() reads JSON from disk
    new_html = render_mod.render()
    cur_html = render_mod.HTML_PATH.read_text()
    # restore on-disk JSON to its prior content so check stays read-only-ish
    if cur_json:
        build_mod.OUTPUT_PATH.write_text(cur_json)

    stale = []
    if new_json != cur_json:
        stale.append("docs/leaderboard.json")
    if new_html != cur_html:
        stale.append("docs/leaderboard.html")
    if stale:
        print("STALE — run `python -m mirobench.leaderboard update` and commit:")
        for s in stale:
            print(f"  - {s}")
        return 1
    print("OK — generated leaderboard files are up to date.")
    return 0


_COMMANDS = {
    "build": _cmd_build,
    "render": _cmd_render,
    "update": _cmd_update,
    "check": _cmd_check,
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "update"
    if cmd not in _COMMANDS:
        print(f"usage: python -m mirobench.leaderboard {{{','.join(_COMMANDS)}}}")
        return 2
    return _COMMANDS[cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
