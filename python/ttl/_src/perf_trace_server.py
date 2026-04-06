# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Perfetto trace server for tt-lang profiler data.

Converts device profiler CSV to Chrome Trace Event format and serves
it over HTTP. An HTML landing page fetches the trace from the same
origin and pushes it into Perfetto UI via postMessage, avoiding
HTTPS/mixed-content issues.

Use TTLANG_PERF_SERV=1 alongside other profiler flags (e.g.
TTLANG_SIGNPOST_PROFILE=1) to automatically serve the trace after
profiling completes.

Usage:
    # With kernel execution (serves after profiling dumps):
    TTLANG_PERF_SERV=1 TTLANG_SIGNPOST_PROFILE=1 python my_kernel.py

    # Standalone:
    python -m ttl._src.perf_trace_server --path /path/to/profiler/.logs/
"""

import csv
import http.server
import json
import os
import re
import socket
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# FW/kernel wrapper zones that obscure the actual trace
_WRAPPER_ZONES = frozenset(
    {
        "BRISC-FW",
        "BRISC-KERNEL",
        "NCRISC-FW",
        "NCRISC-KERNEL",
        "TRISC-FW",
        "TRISC-KERNEL",
    }
)

# HTML page that fetches trace.json from same origin and pushes it
# into Perfetto via postMessage. No HTTPS required.
_LANDING_HTML = """\
<!DOCTYPE html>
<html>
<head><title>TTLang Trace</title></head>
<body>
<p>Loading trace into Perfetto...</p>
<script>
async function openTrace() {
  const resp = await fetch('/trace.json');
  const buf = await resp.arrayBuffer();
  const win = window.open('https://ui.perfetto.dev/');
  const timer = setInterval(() => win.postMessage('PING', 'https://ui.perfetto.dev'), 50);
  window.addEventListener('message', (evt) => {
    if (evt.data !== 'PONG') return;
    clearInterval(timer);
    win.postMessage({
      perfetto: {
        buffer: buf,
        title: 'TTLang Trace',
      }
    }, 'https://ui.perfetto.dev');
    document.querySelector('p').textContent = 'Trace loaded! You can close this tab.';
  });
}
openTrace();
</script>
</body>
</html>
"""


def _parse_chip_freq(header_line: str) -> Optional[float]:
    """Extract chip frequency in MHz from CSV header."""
    m = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", header_line)
    if m:
        return float(m.group(1))
    return None


def csv_to_trace_events(csv_path: Path) -> List[dict]:
    """Convert device profiler CSV to Chrome Trace Event format.

    Parses ZONE_START/ZONE_END pairs from the CSV and emits
    "X" (complete) events with duration. Filters out FW wrapper
    zones and normalizes timestamps to start at 0.
    """
    events = []

    with open(csv_path) as f:
        header_line = f.readline()
        chip_freq_mhz = _parse_chip_freq(header_line) or 1000.0

        f.readline()  # skip column header

        open_zones: Dict[Tuple[str, ...], float] = {}
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 14:
                continue

            (
                _pcie_slot,
                core_x,
                core_y,
                risc,
                timer_id,
                cycles_str,
                _data,
                _run_id,
                _trace_id,
                _trace_cnt,
                zone_name,
                zone_type,
                src_line,
                src_file,
            ) = row[:14]

            if zone_name in _WRAPPER_ZONES:
                continue

            cycles = int(cycles_str)
            us = cycles / chip_freq_mhz

            key = (core_x, core_y, risc, zone_name, timer_id)

            if zone_type == "ZONE_START":
                open_zones[key] = us
            elif zone_type == "ZONE_END" and key in open_zones:
                start_us = open_zones.pop(key)
                dur = us - start_us
                events.append(
                    {
                        "name": zone_name,
                        "cat": risc,
                        "ph": "X",
                        "ts": start_us,
                        "dur": dur,
                        "pid": f"Core ({core_x},{core_y})",
                        "tid": risc,
                        "args": {
                            "source": f"{src_file}:{src_line}",
                        },
                    }
                )

    if events:
        min_ts = min(e["ts"] for e in events)
        for e in events:
            e["ts"] -= min_ts

    return events


class _TraceHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that serves the landing page and trace JSON."""

    trace_json: bytes = b""

    def do_GET(self):
        if self.path == "/trace.json":
            self._serve_json()
        else:
            self._serve_landing()

    def _serve_landing(self):
        body = _LANDING_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.trace_json)))
        self.end_headers()
        self.wfile.write(self.trace_json)

    def log_message(self, format, *args):
        pass


def _find_free_port() -> int:
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _get_container_ip() -> Optional[str]:
    """Return this container's IP if running inside Docker, else None."""
    if not Path("/.dockerenv").exists():
        return None
    try:
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        return None


def serve_trace(csv_path: Path, port: Optional[int] = None):
    """Convert CSV to trace and start HTTP server.

    Serves an HTML landing page that auto-opens Perfetto and pushes
    the trace data via postMessage. Blocks until Enter is pressed.
    """
    events = csv_to_trace_events(csv_path)
    if not events:
        print("[perf_trace] No trace events found in CSV", file=sys.stderr)
        return

    trace_json = json.dumps({"traceEvents": events}).encode()

    if port is None:
        port = _find_free_port()

    handler = type("Handler", (_TraceHandler,), {"trace_json": trace_json})
    server = http.server.HTTPServer(("0.0.0.0", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    user = os.environ.get("USER", "user")
    container_ip = _get_container_ip()
    bind_addr = container_ip or "localhost"

    print()
    print("=" * 70)
    print("TTLANG PERFETTO TRACE SERVER")
    print("=" * 70)
    print(f"  {len(events)} trace events ready")
    print(f"  Serving on port {port}")
    print()
    print("  From your local machine, run:")
    print(f"    ssh -N -L {port}:{bind_addr}:{port} {user}@<server>")
    print()
    print("  Then open:")
    print(f"    http://localhost:{port}")
    print()
    print("  Press Enter to stop the server...")
    print("=" * 70)
    print()

    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        server.shutdown()


def _get_csv_path(logs_path: Optional[Path] = None) -> Path:
    """Resolve path to profile_log_device.csv."""
    if logs_path is not None:
        if logs_path.is_dir():
            return logs_path / "profile_log_device.csv"
        return logs_path

    tt_metal_home = os.environ.get("TT_METAL_HOME", "")
    if not tt_metal_home:
        raise ValueError("TT_METAL_HOME not set and no path provided")

    return (
        Path(tt_metal_home)
        / "generated"
        / "profiler"
        / ".logs"
        / "profile_log_device.csv"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Serve device profiler CSV as Perfetto trace"
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Path to profiler logs directory or CSV file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to serve on (auto-selected if not specified)",
    )
    args = parser.parse_args()

    csv_path = _get_csv_path(Path(args.path) if args.path else None)
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist", file=sys.stderr)
        sys.exit(1)

    serve_trace(csv_path, port=args.port)


if __name__ == "__main__":
    main()
