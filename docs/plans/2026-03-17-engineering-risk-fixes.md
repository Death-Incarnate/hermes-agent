# Engineering Risk Fixes — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Fix three identified engineering risks: process-global toolset state, model-unaware context truncation, and untyped MCP exception handling.

**Architecture:** Each fix is self-contained with no cross-dependencies. They can be implemented in parallel or sequentially. All changes are backward-compatible — no API surface changes.

**Tech Stack:** Python 3.11+, pytest, existing `agent/model_metadata.py`, `model_tools.py`, `agent/context_compressor.py`, `tools/mcp_tool.py`

---

## Risk ① — Process-global `_last_resolved_tool_names`

**Problem:** `model_tools.py:134` holds a module-level list mutated by `get_tool_definitions()`. `delegate_tool.py` save/restores it manually (lines 178, 375) — a race condition in concurrent gateway sessions.

**Fix:** Make `get_tool_definitions()` return the name list alongside the schemas. Pass it explicitly where needed. Remove the global mutation.

---

### Task 1: Add failing test for concurrent toolset isolation

**Objective:** Prove the race exists (or that isolation is required) before fixing it.

**Files:**
- Modify: `tests/tools/test_delegate.py`

**Step 1: Write the test**

```python
def test_tool_names_not_leaked_via_global():
    """get_tool_definitions should not mutate a shared global."""
    import model_tools
    from model_tools import get_tool_definitions

    # Simulate two calls with different toolsets
    get_tool_definitions(enabled_toolsets=["terminal"])
    names_a = list(model_tools._last_resolved_tool_names)

    get_tool_definitions(enabled_toolsets=["web"])
    names_b = list(model_tools._last_resolved_tool_names)

    # They must differ — proves the global is being mutated per-call
    assert names_a != names_b, "global is not being updated per call"

def test_get_tool_definitions_returns_name_list():
    """get_tool_definitions must return (schemas, name_list) tuple."""
    from model_tools import get_tool_definitions
    result = get_tool_definitions(enabled_toolsets=["terminal"])
    # After the fix, result is a tuple
    assert isinstance(result, tuple), "expected (schemas, names) tuple"
    schemas, names = result
    assert isinstance(schemas, list)
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)
```

**Step 2: Run to confirm current behavior**

```bash
cd /home/death/.hermes/hermes-agent && source .venv/bin/activate
python -m pytest tests/tools/test_delegate.py::test_get_tool_definitions_returns_name_list -xvs
```
Expected: FAIL — `get_tool_definitions` returns a list, not a tuple.

**Step 3: Commit the test**

```bash
git add tests/tools/test_delegate.py
git commit -m "test: add assertions for toolset name isolation and return type"
```

---

### Task 2: Return name list from `get_tool_definitions`

**Objective:** Make the function return `(schemas, names)` and assign the global from the return value only at call sites that need it for backward compat.

**Files:**
- Modify: `model_tools.py` lines 264–267

**Step 1: Change the return**

In `model_tools.py`, replace:
```python
    global _last_resolved_tool_names
    _last_resolved_tool_names = [t["function"]["name"] for t in filtered_tools]

    return filtered_tools
```

With:
```python
    resolved_names = [t["function"]["name"] for t in filtered_tools]
    # Keep global updated for any legacy callers not yet migrated
    global _last_resolved_tool_names
    _last_resolved_tool_names = resolved_names

    return filtered_tools, resolved_names
```

**Step 2: Run existing tests to catch breakage**

```bash
python -m pytest tests/ -q --tb=short 2>&1 | head -60
```

Expected: failures in callers of `get_tool_definitions` that unpack or use the return value directly. Note every failing file.

**Step 3: Fix callers**

Search for all call sites:
```bash
grep -rn "get_tool_definitions(" /home/death/.hermes/hermes-agent --include="*.py" | grep -v test | grep -v ".pyc"
```

For each call site that does `tools = get_tool_definitions(...)`, update to:
```python
tools, tool_names = get_tool_definitions(...)
```

If the caller only needs schemas, use: `tools, _ = get_tool_definitions(...)`.

**Step 4: Update `delegate_tool.py` to use returned names**

In `tools/delegate_tool.py` line ~178, the save/restore can be removed once `get_tool_definitions` is called inside the subagent scope. Document with a comment:

```python
# _last_resolved_tool_names global is still updated by get_tool_definitions
# for backward compat — no need to save/restore here anymore since we
# get the name list directly from the return value.
```

Leave the save/restore in place but mark it as `# TODO: remove after all callers migrated`.

**Step 5: Run full suite**

```bash
python -m pytest tests/ -q
```
Expected: all previously passing tests still pass + new tests pass.

**Step 6: Commit**

```bash
git add model_tools.py tools/delegate_tool.py
git commit -m "refactor: return resolved tool names from get_tool_definitions"
```

---

## Risk ② — Context compressor ignores model context for truncation

**Problem:** `context_compressor.py:107` hard-truncates tool output to `[:1000]` and `[-500:]` regardless of model context window. `summary_target_tokens * 2` has no ceiling. The compressor already has `self.context_length` — it's just not used for these calculations.

**Fix:** Derive the hard-truncation limit and the summary max_tokens ceiling from `self.context_length`.

---

### Task 3: Add failing tests for model-aware truncation

**Objective:** Assert that truncation limits scale with model context.

**Files:**
- Modify: `tests/test_context_compressor.py` (create if not present)

**Step 1: Locate existing compressor tests**

```bash
find /home/death/.hermes/hermes-agent/tests -name "*compress*" -o -name "*context*" | grep -v __pycache__
```

**Step 2: Write tests**

```python
from unittest.mock import patch, MagicMock
from agent.context_compressor import ContextCompressor

def make_compressor(context_length=8192):
    with patch("agent.context_compressor.get_model_context_length", return_value=context_length):
        return ContextCompressor(model="test-model", summary_target_tokens=2500)

def test_truncation_limit_scales_with_context():
    """Larger context → larger per-message truncation budget."""
    small = make_compressor(context_length=8_000)
    large = make_compressor(context_length=200_000)
    assert large._tool_output_truncation_limit > small._tool_output_truncation_limit

def test_summary_max_tokens_has_ceiling():
    """summary_target_tokens * 2 should not exceed a safe fraction of context_length."""
    comp = make_compressor(context_length=8_000)
    # max_tokens for summary call must not exceed context_length // 4
    assert comp._summary_max_tokens <= 8_000 // 4
```

**Step 3: Run to confirm failure**

```bash
python -m pytest tests/ -k "truncation_limit_scales or summary_max_tokens" -xvs
```
Expected: AttributeError — `_tool_output_truncation_limit` not defined yet.

**Step 4: Commit the tests**

```bash
git add tests/
git commit -m "test: assert compressor truncation scales with model context"
```

---

### Task 4: Implement model-aware truncation limits

**Objective:** Replace magic numbers with `context_length`-derived limits.

**Files:**
- Modify: `agent/context_compressor.py` lines 54–58, 107, 139

**Step 1: Add computed properties to `__init__`**

After line 58 (`self.context_length = ...`), add:

```python
# Tool output truncation: scale with context. Floor at 500, ceiling at 10_000.
# For an 8k model: ~1500 chars. For a 200k model: ~10_000 chars.
self._tool_output_truncation_limit: int = max(
    500,
    min(10_000, self.context_length // 5),
)
# Summary generation max_tokens: must fit within context with room for prompt
self._summary_max_tokens: int = min(
    self.summary_target_tokens * 2,
    max(512, self.context_length // 4),
)
```

**Step 2: Replace magic numbers in `_truncate_tool_output` (line 107)**

Replace:
```python
content = content[:1000] + "\n...[truncated]...\n" + content[-500:]
```
With:
```python
head = self._tool_output_truncation_limit
tail = head // 2
content = content[:head] + "\n...[truncated]...\n" + content[-tail:]
```

**Step 3: Replace `summary_target_tokens * 2` in `_generate_summary` (line 139)**

Replace:
```python
"max_tokens": self.summary_target_tokens * 2,
```
With:
```python
"max_tokens": self._summary_max_tokens,
```

**Step 4: Run tests**

```bash
python -m pytest tests/ -q
```
Expected: new tests pass, no regressions.

**Step 5: Commit**

```bash
git add agent/context_compressor.py
git commit -m "fix: derive compressor truncation limits from model context length"
```

---

## Risk ③ — MCP exception taxonomy

**Problem:** `tools/mcp_tool.py` has 8+ bare `except Exception as exc` sites. All errors surface identically — agent can't distinguish timeout (retry) from auth failure (surface to user) from config error (abort).

**Fix:** Define a small `MCPError` hierarchy. Replace the key catch-all sites with typed raises and typed catches.

---

### Task 5: Define `MCPError` hierarchy

**Objective:** Create the exception classes. No behavior change yet.

**Files:**
- Modify: `tools/mcp_tool.py` — add near the top, after imports

**Step 1: Write a test that the classes exist**

```python
from tools.mcp_tool import MCPError, MCPTimeoutError, MCPAuthError, MCPConfigError, MCPProtocolError

def test_mcp_error_hierarchy():
    assert issubclass(MCPTimeoutError, MCPError)
    assert issubclass(MCPAuthError, MCPError)
    assert issubclass(MCPConfigError, MCPError)
    assert issubclass(MCPProtocolError, MCPError)

def test_mcp_error_is_exception():
    assert issubclass(MCPError, Exception)
```

**Step 2: Run to confirm failure**

```bash
python -m pytest tests/tools/test_mcp_tool.py -k "mcp_error_hierarchy" -xvs
```
Expected: ImportError — classes don't exist.

**Step 3: Add the hierarchy to `mcp_tool.py`**

After the import block, add:

```python
# ---------------------------------------------------------------------------
# MCP Error hierarchy
# ---------------------------------------------------------------------------

class MCPError(Exception):
    """Base class for all MCP-related errors."""
    def __init__(self, message: str, server_name: str = "", retryable: bool = False):
        super().__init__(message)
        self.server_name = server_name
        self.retryable = retryable

class MCPTimeoutError(MCPError):
    """MCP server or tool call timed out. Retryable."""
    def __init__(self, message: str, server_name: str = ""):
        super().__init__(message, server_name=server_name, retryable=True)

class MCPAuthError(MCPError):
    """Authentication or permission failure. Not retryable."""
    def __init__(self, message: str, server_name: str = ""):
        super().__init__(message, server_name=server_name, retryable=False)

class MCPConfigError(MCPError):
    """Bad server configuration. Not retryable."""
    def __init__(self, message: str, server_name: str = ""):
        super().__init__(message, server_name=server_name, retryable=False)

class MCPProtocolError(MCPError):
    """Unexpected protocol response or parse failure. Not retryable."""
    def __init__(self, message: str, server_name: str = ""):
        super().__init__(message, server_name=server_name, retryable=False)
```

**Step 4: Run tests**

```bash
python -m pytest tests/tools/test_mcp_tool.py -q
```
Expected: hierarchy tests pass.

**Step 5: Commit**

```bash
git add tools/mcp_tool.py tests/tools/test_mcp_tool.py
git commit -m "feat: add MCPError exception hierarchy to mcp_tool"
```

---

### Task 6: Replace key catch-all sites with typed raises

**Objective:** Wire the hierarchy into the 5 most impactful exception sites.

**Files:**
- Modify: `tools/mcp_tool.py` lines 636, 642, 811, 865, 936

**Priority sites (by impact):**

| Line | Site | Correct type |
|------|------|-------------|
| 636 | `asyncio.TimeoutError` in sampling call | `MCPTimeoutError` |
| 642 | `Exception` in sampling call | `MCPProtocolError` |
| 811 | `Exception` in connection loop | detect auth/config vs protocol |
| 865 | `asyncio.TimeoutError` in stdio run | `MCPTimeoutError` |
| 936 | `Exception` in tool call dispatch | `MCPProtocolError` |

**Step 1: Replace line 636–646 (sampling timeout + generic)**

```python
        except asyncio.TimeoutError:
            self.metrics["errors"] += 1
            raise MCPTimeoutError(
                f"Sampling LLM call timed out after {self.timeout}s",
                server_name=self.server_name,
            )
        except MCPError:
            raise  # already typed, let it propagate
        except Exception as exc:
            self.metrics["errors"] += 1
            raise MCPProtocolError(
                f"Sampling LLM call failed: {_sanitize_error(str(exc))}",
                server_name=self.server_name,
            ) from exc
```

**Step 2: At the `_error()` call sites that catch these**, update to catch `MCPError` and inspect `.retryable`:

```python
except MCPTimeoutError as exc:
    return self._error(str(exc))  # caller could retry
except MCPError as exc:
    return self._error(str(exc))  # not retryable
```

**Step 3: Add a test that timeout produces `MCPTimeoutError`**

```python
import asyncio
from unittest.mock import patch, AsyncMock

async def test_sampling_timeout_raises_mcp_timeout():
    # ... mock setup ...
    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
        with pytest.raises(MCPTimeoutError) as exc_info:
            await server._call_sampling_llm(...)
        assert exc_info.value.retryable is True
```

**Step 4: Run full suite**

```bash
python -m pytest tests/ -q
```

**Step 5: Commit**

```bash
git add tools/mcp_tool.py tests/tools/test_mcp_tool.py
git commit -m "fix: replace bare Exception catches with typed MCPError raises in mcp_tool"
```

---

## Final verification

```bash
cd /home/death/.hermes/hermes-agent && source .venv/bin/activate
python -m pytest tests/ -q
```

All ~3000 tests should pass. If anything regresses, check:
1. Callers of `get_tool_definitions` that unpack as a list (Risk ①)
2. Compressor tests that mock `get_model_context_length` (Risk ②)
3. MCP tests that assert on error strings vs. exception types (Risk ③)

---

## Order of execution

Risk ② (Tasks 3–4) is the safest and most self-contained — do it first.
Risk ③ (Tasks 5–6) is additive — just new classes + reraise, low blast radius.
Risk ① (Tasks 1–2) touches the most call sites — do it last with the full suite handy.
