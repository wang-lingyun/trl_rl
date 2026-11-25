"""Wait until the vLLM server answers /health (stdlib only: the GPU box has no curl).

Exits 0 when healthy, 1 on timeout or if the server process (--pid) died first.
"""

import argparse
import os
import sys
import time
import urllib.error
import urllib.request


def alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--pid", type=int, default=None, help="server PID; stop waiting if it exits")
    args = p.parse_args()

    start = time.time()
    while time.time() - start < args.timeout:
        if not alive(args.pid):
            print(f"server process {args.pid} exited before becoming healthy", file=sys.stderr)
            return 1
        try:
            with urllib.request.urlopen(f"{args.url}/health", timeout=5) as r:
                if r.status == 200:
                    print(f"vLLM healthy after {time.time() - start:.0f}s")
                    return 0
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(2)
    print(f"timed out after {args.timeout:.0f}s waiting for {args.url}/health", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
