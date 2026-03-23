"""Minimal HTTP server for reviewing and approving agent runs.

Serves a simple web UI and JSON API so a stakeholder who wasn't in the
room can review what the agent did, inspect captured writes, and
approve or reject them for replay.

Uses only the stdlib ``http.server`` — no Flask/FastAPI dependency.
"""

from __future__ import annotations

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any
from urllib.parse import urlparse, parse_qs

from agent_runner.sharing.run_store import RunStore, RunStatus

logger = logging.getLogger(__name__)


def create_approval_app(store: RunStore, host: str = "0.0.0.0", port: int = 8811) -> HTTPServer:
    """Create and return an HTTPServer (call ``.serve_forever()`` to start)."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/")

            if path == "" or path == "/runs":
                self._respond_json([r.to_dict() for r in store.list_pending()])

            elif path.startswith("/runs/"):
                run_id = path.split("/")[2]
                record = store.get(run_id)
                if record:
                    self._respond_json(record.to_dict())
                else:
                    self._respond_json({"error": "not found"}, 404)

            elif path.startswith("/review/"):
                run_id = path.split("/")[2]
                record = store.get(run_id)
                if record:
                    self._respond_html(self._render_review_page(record))
                else:
                    self._respond_html("<h1>Run not found</h1>", 404)

            else:
                self._respond_json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/")

            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len)) if content_len else {}

            if path.startswith("/runs/") and path.endswith("/approve"):
                run_id = path.split("/")[2]
                reviewer = body.get("reviewer", "anonymous")
                record = store.approve(run_id, reviewer)
                if record:
                    self._respond_json(record.to_dict())
                else:
                    self._respond_json({"error": "not found"}, 404)

            elif path.startswith("/runs/") and path.endswith("/reject"):
                run_id = path.split("/")[2]
                reviewer = body.get("reviewer", "anonymous")
                record = store.reject(run_id, reviewer)
                if record:
                    self._respond_json(record.to_dict())
                else:
                    self._respond_json({"error": "not found"}, 404)

            else:
                self._respond_json({"error": "not found"}, 404)

        def _respond_json(self, data: Any, status: int = 200) -> None:
            body = json.dumps(data, indent=2, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _respond_html(self, html: str, status: int = 200) -> None:
            body = html.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _render_review_page(self, record: Any) -> str:
            writes_html = ""
            for i, w in enumerate(record.captured_writes):
                inp = json.dumps(w["input"], indent=2, default=str)
                writes_html += f"""
                <div style="background:#1e1e1e;padding:12px;border-radius:6px;margin:8px 0">
                  <strong style="color:#569cd6">#{i+1} {w['tool']}</strong>
                  <pre style="color:#d4d4d4;margin:8px 0;white-space:pre-wrap">{inp}</pre>
                </div>"""

            calls_html = ""
            for c in record.tool_call_log:
                color = {"allow": "#4ec9b0", "deny": "#f44747", "mock": "#dcdcaa"}.get(c["action"], "#ccc")
                calls_html += f'<span style="color:{color}">[{c["action"]}] {c["tool"]}</span><br>'

            status_color = {
                "pending": "#dcdcaa",
                "approved": "#4ec9b0",
                "rejected": "#f44747",
            }.get(record.status.value, "#ccc")

            return f"""<!DOCTYPE html>
<html><head><title>Review Run {record.id}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #1a1a2e; color: #e0e0e0; }}
  h1 {{ color: #82aaff; }}
  .card {{ background: #16213e; padding: 20px; border-radius: 8px; margin: 16px 0; }}
  .btn {{ padding: 10px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; margin-right: 8px; }}
  .approve {{ background: #4ec9b0; color: #000; }}
  .reject {{ background: #f44747; color: #fff; }}
  pre {{ background: #0d1117; padding: 12px; border-radius: 6px; overflow-x: auto; }}
</style></head>
<body>
  <h1>Run {record.id}</h1>
  <p>Status: <strong style="color:{status_color}">{record.status.value.upper()}</strong></p>

  <div class="card">
    <h3>Prompt</h3>
    <p>{record.prompt}</p>
  </div>

  <div class="card">
    <h3>Agent Response</h3>
    <pre>{record.final_text}</pre>
  </div>

  <div class="card">
    <h3>Tool Calls ({len(record.tool_call_log)})</h3>
    {calls_html}
  </div>

  <div class="card">
    <h3>Captured Writes ({len(record.captured_writes)})</h3>
    {writes_html if writes_html else '<p style="color:#888">No writes captured.</p>'}
  </div>

  <div style="margin-top:24px">
    <button class="btn approve" onclick="decide('approve')">Approve &amp; Replay</button>
    <button class="btn reject" onclick="decide('reject')">Reject</button>
  </div>

  <script>
    async function decide(action) {{
      const reviewer = prompt('Your name (optional):', '') || 'anonymous';
      const res = await fetch('/runs/{record.id}/' + action, {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{reviewer}})
      }});
      const data = await res.json();
      alert('Status: ' + data.status);
      location.reload();
    }}
  </script>
</body></html>"""

        def log_message(self, format: str, *args: Any) -> None:
            logger.info(format, *args)

    server = HTTPServer((host, port), Handler)
    logger.info("Approval server ready at http://%s:%d", host, port)
    return server
