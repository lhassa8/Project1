# Plan: Make Agent Runner a Groundbreaking Enterprise Solution

## Honest Starting Point

The core loop and interceptor pattern are solid (8/10). Everything else ranges
from mediocre to demo-quality. This plan addresses the real gaps.

---

## Phase 1: Shadow Mode That Actually Works

**Problem**: Current shadow mode returns fake strings like `"[shadow] write_file
captured"`. When the agent then tries to read that file, it fails. Shadow mode
breaks multi-step agent plans.

**Solution**: Virtual filesystem overlay.

### 1.1 — Virtual State Layer

Add a `ShadowState` that maintains an in-memory overlay of what the world
*would* look like if writes had been executed:

```
agent_runner/shadow/
    state.py        — ShadowState: in-memory overlay (virtual FS, virtual shell output)
    executor.py     — ShadowExecutor: runs tools against virtual state
    diff.py         — Generates human-readable diffs of proposed changes
    replay.py       — Transaction-safe replay with rollback on failure
```

- `write_file("a.txt", "hello")` → stores in virtual FS, returns success
- `read_file("a.txt")` → checks virtual FS first, falls back to real FS
- `shell("cat a.txt")` → intercepts known patterns, delegates unknown to real shell
- `list_files(".")` → merges virtual FS entries with real directory listing

### 1.2 — Diff Preview

Before replay, generate a clear diff of all proposed changes:

```
Shadow Run Summary:
  CREATE  config/app.yaml       (142 bytes)
  MODIFY  src/main.py           (+12 / -3 lines)
  SHELL   npm install express   (would install 1 package)

Approve all? [y/N] Or review individually? [r]
```

### 1.3 — Transaction Replay

Replay with rollback semantics:
- Backup originals before writing
- Execute in order, stop on first failure
- On failure: rollback all changes, report what failed and why
- On success: commit and log

---

## Phase 2: Tool Sandboxing & Security

**Problem**: `shell(command)` runs with `shell=True` and no restrictions.
`read_file` / `write_file` have no path boundaries. An agent can read
`~/.ssh/id_rsa` or `rm -rf /`.

### 2.1 — Path Sandbox

```python
class PathSandbox:
    """Restrict file operations to allowed directories."""
    def __init__(self, allowed_roots: list[str], denied_patterns: list[str]):
        ...
    def validate(self, path: str) -> str:
        """Resolve and validate path. Raises SandboxViolation if outside bounds."""
```

- Resolve symlinks before checking (prevent symlink escapes)
- Default: current working directory only
- Configurable via `agent.json`: `"sandbox": {"roots": ["./src", "/tmp"], "deny": ["*.key", "*.pem"]}`

### 2.2 — Command Allowlist

```python
class ShellPolicy:
    """Control which shell commands the agent can execute."""
    allow: list[str]       # ["ls", "cat", "grep", "npm", "git"]
    deny: list[str]        # ["rm", "curl", "wget", "sudo"]
    max_runtime: float     # seconds
    max_output: int        # bytes
```

Parse the command, check the base executable against allow/deny lists.
This isn't foolproof (shell escapes exist) but catches 95% of accidents.

### 2.3 — Resource Limits

- Max output size per tool call (prevent 10GB file reads flooding context)
- Max total tool calls per run (cost/abuse protection)
- Max total tokens budget (stop the run if projected cost exceeds threshold)
- Per-tool timeout (not just shell — any tool can hang)

### 2.4 — PII-Safe Logging

`LoggingInterceptor` currently dumps raw tool inputs to logs. Add:
- Configurable field redaction: `redact_fields=["api_key", "password", "token"]`
- Output truncation: don't log 10KB of file contents
- Structured JSON logging option (for log aggregators)

---

## Phase 3: Approval Workflows That Work in Production

**Problem**: The approval server has no auth, uses filesystem JSON storage,
and only works on localhost.

### 3.1 — Authentication & RBAC

- Token-based auth for the approval API (Bearer tokens)
- Role model: `requester` (starts runs), `reviewer` (approves/denies), `admin`
- Configurable via `agent.json`:
  ```json
  {"approval": {"auth": "token", "reviewers": ["alice@co.com", "bob@co.com"]}}
  ```

### 3.2 — Async Approval Flow

Replace the synchronous stdin prompt with an async flow:
1. Agent run reaches an approval point → pauses, saves state
2. Notification sent (webhook, email, Slack)
3. Reviewer approves/denies via API or web UI
4. Agent run resumes from saved state

This requires:
- Run state serialization (save mid-run state to disk/DB)
- Resume capability (load state and continue the loop)
- Webhook/notification integration

### 3.3 — Approval Policies

Replace binary approve/deny with policy-based decisions:

```python
class ApprovalPolicy:
    """Declarative approval rules."""
    rules: list[ApprovalRule]

class ApprovalRule:
    tool: str | None           # None = all tools
    condition: str             # "always", "if_write", "if_matches"
    pattern: str | None        # regex for condition matching
    action: str                # "auto_approve", "require_review", "auto_deny"
    reviewers: list[str]       # who can approve this specific rule
```

### 3.4 — Immutable Audit Trail

- Every approval decision gets a signed record (tool, input, decision, reviewer, timestamp)
- Append-only log (not mutable JSON files)
- Export for compliance (CSV, JSON-lines)

---

## Phase 4: Robust MCP Integration

**Problem**: MCP client has no error recovery, no timeouts, blocks on slow
servers, and is only tested with mocks.

### 4.1 — Connection Resilience

- Read timeout on `stdout.readline()` (configurable, default 30s)
- Automatic reconnection on server crash (restart subprocess, re-initialize)
- Health checks (periodic ping to verify server is alive)
- Graceful degradation (if MCP server dies, mark its tools as unavailable
  instead of crashing the whole run)

### 4.2 — Async MCP Client

The current synchronous client blocks the entire agent while waiting for
an MCP tool call. Add an async variant:

```python
class AsyncMCPClient:
    async def call_tool(self, name: str, arguments: dict) -> Any:
        ...
```

This enables parallel tool calls when the agent requests multiple tools
in a single turn.

### 4.3 — MCP Server Enhancements

The MCP server (`mcp_server.py`) needs:
- Tool annotations (read-only vs. read-write hints per MCP spec)
- Proper error codes (not just -32601)
- Graceful shutdown on SIGTERM
- Health endpoint

### 4.4 — Integration Tests

Add real integration tests with a simple MCP server subprocess:
- Start a real MCP server
- Discover tools
- Call tools
- Verify responses
- Test crash recovery

---

## Phase 5: Conversation Intelligence

**Problem**: Conversation history grows unbounded. 100-turn conversations send
all 100 turns to the API every call, quadratic cost growth.

### 5.1 — Context Window Management

```python
class ContextManager:
    """Manage conversation history within token budgets."""
    max_context_tokens: int      # e.g., 100_000
    strategy: str                # "sliding_window", "summarize", "priority"
```

Strategies:
- **Sliding window**: Keep last N turns, drop oldest
- **Summarize**: Periodically compress old turns into a summary message
- **Priority**: Keep system prompt + first turn + last N turns + all tool results

### 5.2 — Cost Prediction & Budgets

Before starting a run, estimate the cost:
```python
runner.estimate_cost("What is 2+2?")
# → EstimatedCost(min=0.003, max=0.05, model="sonnet")
```

Add a hard budget cap:
```python
runner = AgentRunner(..., max_cost_usd=1.00)
# Stops the run if projected cost exceeds $1.00
```

### 5.3 — Checkpoint & Resume

Save conversation state to disk and resume later:
```python
conv = runner.conversation()
conv.ask("Start a long task...")
conv.save("checkpoint.json")

# Later, even in a different process:
conv = runner.conversation()
conv.load("checkpoint.json")
conv.ask("Continue where we left off")
```

---

## Phase 6: Observability & Operations

### 6.1 — Structured Event Stream

Replace scattered `logger.info()` calls with a structured event system:

```python
@dataclass
class AgentEvent:
    timestamp: float
    event_type: str        # "tool_call", "api_request", "approval", "error"
    data: dict[str, Any]
    run_id: str
    turn: int
```

Emit to: stdout (JSON-lines), file, webhook, or custom handler.

### 6.2 — Metrics

Track and expose:
- Tool call latency (p50, p95, p99 per tool)
- Token usage over time
- Error rates per tool
- Cost per run / per user
- Shadow mode capture rates

### 6.3 — Cost Dashboard

For enterprise use, provide a cost tracking interface:
- Per-run cost breakdown (which tools, how many tokens)
- Daily/weekly cost trends
- Budget alerts
- Per-team/per-user attribution

---

## Implementation Priority

| Phase | Impact | Effort | Priority |
|-------|--------|--------|----------|
| 1. Shadow virtual FS | Critical — current shadow mode is broken | Medium | **P0** |
| 2. Tool sandboxing | Critical — security blocker for production | Medium | **P0** |
| 3. Approval workflows | High — differentiator vs other frameworks | High | **P1** |
| 4. MCP robustness | High — required for marketplace | Medium | **P1** |
| 5. Conversation mgmt | Medium — cost/UX improvement | Medium | **P2** |
| 6. Observability | Medium — required for enterprise ops | Medium | **P2** |

**Start with Phase 1 + 2 (P0)**: These fix the two biggest lies in the
current codebase — shadow mode that breaks agents and tools with no
security boundaries. Everything else is enhancement; these are fixes.
