"""Keybinding chord support with timeout-based state machine.

Supports chord sequences like ``ctrl+x ctrl+k`` with a configurable
timeout (default 1s). Single-key bindings pass through immediately.

Usage::

    chord = ChordManager(timeout=1.0)
    chord.register("ctrl+x ctrl+k", "kill_agents")
    chord.register("ctrl+x ctrl+s", "save_session")
    chord.register("ctrl+c", "exit")

    # In key handler:
    result = chord.feed("ctrl+x")
    # result.state == ChordState.PENDING (waiting for next key)

    result = chord.feed("ctrl+k")
    # result.state == ChordState.MATCHED, result.action == "kill_agents"
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto


class ChordState(Enum):
    """Result of feeding a keypress to the chord manager."""

    PASSTHROUGH = auto()  # no chord prefix matched, pass key to default handler
    PENDING = auto()      # partial chord match, waiting for next key
    MATCHED = auto()      # full chord matched, action ready
    TIMEOUT = auto()      # chord timed out, reset


@dataclass(frozen=True)
class ChordResult:
    """Result from ChordManager.feed()."""

    state: ChordState
    action: str = ""
    consumed_keys: tuple[str, ...] = ()


@dataclass
class _ChordNode:
    """Trie node for chord prefix matching."""

    children: dict[str, "_ChordNode"] = field(default_factory=dict)
    action: str = ""  # non-empty if this node is a terminal


class ChordManager:
    """State machine for multi-key chord sequences.

    Builds a trie of registered chords. On each keypress, advances
    through the trie. Times out after ``timeout`` seconds of inactivity.
    """

    def __init__(self, timeout: float = 1.0) -> None:
        self._timeout = timeout
        self._root = _ChordNode()
        self._current = self._root
        self._pending_keys: list[str] = []
        self._last_key_time: float = 0.0
        self._single_key: dict[str, str] = {}  # key -> action for non-chords

    def register(self, keys: str, action: str) -> None:
        """Register a chord sequence mapped to an action name.

        Args:
            keys: Space-separated key sequence (e.g. "ctrl+x ctrl+k").
            action: Action name to return when the chord completes.
        """
        parts = keys.strip().split()
        if len(parts) == 1:
            self._single_key[parts[0]] = action
            return

        node = self._root
        for part in parts:
            if part not in node.children:
                node.children[part] = _ChordNode()
            node = node.children[part]
        node.action = action

    def feed(self, key: str) -> ChordResult:
        """Feed a keypress and return the chord state.

        Call this from the TUI's key handler. If the result is
        ``PENDING``, suppress the key event. If ``MATCHED``,
        execute the action. If ``PASSTHROUGH``, let the key
        propagate normally.
        """
        now = time.monotonic()

        # Check for timeout -- reset if too slow
        if self._pending_keys and (now - self._last_key_time) > self._timeout:
            self._reset()

        self._last_key_time = now

        # If we're in the middle of a chord, continue matching
        if self._pending_keys:
            if key in self._current.children:
                self._pending_keys.append(key)
                self._current = self._current.children[key]

                if self._current.action:
                    result = ChordResult(
                        state=ChordState.MATCHED,
                        action=self._current.action,
                        consumed_keys=tuple(self._pending_keys),
                    )
                    self._reset()
                    return result

                # Still more keys needed
                return ChordResult(
                    state=ChordState.PENDING,
                    consumed_keys=tuple(self._pending_keys),
                )

            # Key doesn't match any continuation -- timeout/reset
            consumed = tuple(self._pending_keys)
            self._reset()
            return ChordResult(state=ChordState.TIMEOUT, consumed_keys=consumed)

        # Not in a chord -- check if this key starts one
        if key in self._root.children:
            self._pending_keys = [key]
            self._current = self._root.children[key]

            if self._current.action:
                # Single-step chord (shouldn't happen, but handle it)
                result = ChordResult(
                    state=ChordState.MATCHED,
                    action=self._current.action,
                    consumed_keys=(key,),
                )
                self._reset()
                return result

            return ChordResult(state=ChordState.PENDING, consumed_keys=(key,))

        # Check single-key bindings
        if key in self._single_key:
            return ChordResult(
                state=ChordState.MATCHED,
                action=self._single_key[key],
                consumed_keys=(key,),
            )

        return ChordResult(state=ChordState.PASSTHROUGH)

    def reset(self) -> None:
        """Publicly reset chord state (e.g. on focus change)."""
        self._reset()

    def _reset(self) -> None:
        self._current = self._root
        self._pending_keys = []

    @property
    def is_pending(self) -> bool:
        """Whether we're in the middle of a chord sequence."""
        return len(self._pending_keys) > 0

    def describe(self) -> list[dict[str, str]]:
        """Return all registered bindings for display."""
        bindings = []
        for key, action in self._single_key.items():
            bindings.append({"keys": key, "action": action})
        self._collect_chords(self._root, [], bindings)
        return bindings

    def _collect_chords(
        self, node: _ChordNode, prefix: list[str], out: list[dict[str, str]],
    ) -> None:
        for key, child in node.children.items():
            path = [*prefix, key]
            if child.action:
                out.append({"keys": " ".join(path), "action": child.action})
            self._collect_chords(child, path, out)
