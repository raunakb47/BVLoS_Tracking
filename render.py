#!/usr/bin/env python3
"""
Module: render.py
Live view of Stage 4. Serves render.html and the Stage 4 log it draws from,
so the page redraws each device as soon as locate.py appends its fix.

  GET /                   the page
  GET /records?from=N     complete lines of <session>/stage4/locate.jsonl from
                          byte offset N: {"next": offset, "records": [...]}
  GET /segments?from=N    the same for <session>/stage4/segments.jsonl
                          (confidence.py): one line per segment state
  GET /meta               session name and the ranging settings in force

The page polls /records with the offset it last received, so each line is
sent once and a reload replays the session from its start. Nothing here
computes geometry: the page draws the fields locate.py wrote.

Binds to 127.0.0.1 unless RENDER_HOST says otherwise.
"""
import http.server
import json
import os
import sys
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))


def read_from(path, offset):
    """(records, next offset) for the complete lines of path after offset."""
    if not os.path.exists(path):
        return [], 0
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        if offset > size:          # the log was replaced: start again
            offset = 0
        handle.seek(offset)
        chunk = handle.read()
    end = chunk.rfind(b"\n") + 1   # a partly written last line waits
    records = [json.loads(line) for line in chunk[:end].splitlines() if line.strip()]
    return records, offset + end


def handler_for(session_dir, environ):
    logs = {"/records": os.path.join(session_dir, "stage4", "locate.jsonl"),
            "/segments": os.path.join(session_dir, "stage4", "segments.jsonl")}
    meta = {"session": os.path.basename(os.path.abspath(session_dir)),
            "path_loss_model": environ.get("PATH_LOSS_MODEL"),
            "range_sigma_k": environ.get("RANGE_SIGMA_K"),
            "anchor_mac": environ.get("ANCHOR_MAC") or None}

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, body, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            if url.path == "/":
                with open(os.path.join(HERE, "render.html"), "rb") as handle:
                    self._send(handle.read(), "text/html; charset=utf-8")
            elif url.path in logs:
                query = urllib.parse.parse_qs(url.query)
                try:
                    offset = max(0, int(query.get("from", ["0"])[0]))
                except ValueError:
                    offset = 0
                records, offset = read_from(logs[url.path], offset)
                self._send(json.dumps({"next": offset, "records": records}).encode(),
                           "application/json")
            elif url.path == "/meta":
                self._send(json.dumps(meta).encode(), "application/json")
            else:
                self.send_error(404)

        def log_message(self, *args):
            pass

    return Handler


def serve(session_dir, port, host="127.0.0.1", environ=None):
    environ = os.environ if environ is None else environ
    server = http.server.ThreadingHTTPServer((host, port), handler_for(session_dir, environ))
    print(f"[render] http://{host}:{server.server_address[1]}/  <- {session_dir}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: render.py <session_dir> <port>\n"
              "  env: RENDER_HOST, PATH_LOSS_MODEL, RANGE_SIGMA_K, ANCHOR_MAC",
              file=sys.stderr)
        raise SystemExit(2)
    serve(sys.argv[1], int(sys.argv[2]), os.environ.get("RENDER_HOST", "127.0.0.1"))
