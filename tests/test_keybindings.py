"""Tests for chord keybindings and their MainScreen action wiring.

Verifies that the ChordManager built by MainScreen resolves the three
registered chords to their action names, and that the matching
``action_*`` methods exist on MainScreen so the chords are not no-ops.
"""

from cascade.keybindings import ChordManager, ChordState
from cascade.screens.main import MainScreen


# (chord keys, expected action, expected MainScreen method)
CHORDS = (
    (("ctrl+x", "ctrl+k"), "kill_workers", "action_kill_workers"),
    (("ctrl+x", "ctrl+e"), "export_session", "action_export_session"),
    (("ctrl+x", "ctrl+h"), "toggle_hooks", "action_toggle_hooks"),
)


class TestChordResolution:
    """The chord manager resolves the registered chords to action names."""

    def test_prefix_is_pending(self):
        cm = MainScreen._build_chord_manager()
        result = cm.feed("ctrl+x")
        assert result.state is ChordState.PENDING

    def test_each_chord_matches_its_action(self):
        for keys, action, _method in CHORDS:
            cm = MainScreen._build_chord_manager()
            assert cm.feed(keys[0]).state is ChordState.PENDING
            final = cm.feed(keys[1])
            assert final.state is ChordState.MATCHED
            assert final.action == action

    def test_unknown_continuation_times_out(self):
        cm = MainScreen._build_chord_manager()
        cm.feed("ctrl+x")
        result = cm.feed("ctrl+z")
        assert result.state is ChordState.TIMEOUT

    def test_unprefixed_key_passes_through(self):
        cm = MainScreen._build_chord_manager()
        result = cm.feed("a")
        assert result.state is ChordState.PASSTHROUGH


class TestActionMethods:
    """Each resolved action has a matching, callable MainScreen method."""

    def test_action_methods_exist_and_are_callable(self):
        for _keys, _action, method in CHORDS:
            assert hasattr(MainScreen, method), f"missing {method}"
            assert callable(getattr(MainScreen, method))

    def test_resolved_actions_map_to_methods(self):
        """The action name a chord resolves to must name a real action_* method."""
        cm = MainScreen._build_chord_manager()
        for keys, action, method in CHORDS:
            cm.reset()
            cm.feed(keys[0])
            resolved = cm.feed(keys[1]).action
            assert f"action_{resolved}" == method
            assert hasattr(MainScreen, f"action_{resolved}")
