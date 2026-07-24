"""PermissionScreen actually composes/renders (regression: PALETTE.text typo).

The screen referenced PALETTE.text -- a non-existent attribute -- so every
permission prompt crashed the app the moment it tried to render (an
AttributeError inside compose()). The unit suite never pushed the screen, so
it stayed green. This mounts it for real and drives each answer.
"""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from cascade.screens.permission import PermissionScreen


class _Harness(App):
    def compose(self) -> ComposeResult:
        yield Static("base")


@pytest.mark.asyncio
async def test_permission_screen_renders_for_write_file():
    app = _Harness()
    async with app.run_test() as pilot:
        result = []

        def _got(answer):
            result.append(answer)

        app.push_screen(
            PermissionScreen(
                "write_file",
                {"path": "src/config.ts"},
                "workspace write",
            ),
            _got,
        )
        await pilot.pause()
        # If compose() raised (the PALETTE.text bug), the screen wouldn't be up.
        assert isinstance(app.screen, PermissionScreen)
        await pilot.press("y")
        await pilot.pause()
        assert result == ["allow"]


@pytest.mark.asyncio
async def test_permission_screen_renders_for_shell_and_denies_on_escape():
    app = _Harness()
    async with app.run_test() as pilot:
        result = []
        app.push_screen(
            PermissionScreen(
                "run_command",
                {"command": "rm -rf /"},
                "dangerous shell construct",
            ),
            result.append,
        )
        await pilot.pause()
        assert isinstance(app.screen, PermissionScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert result == ["deny"]


@pytest.mark.asyncio
async def test_permission_screen_handles_long_value():
    app = _Harness()
    async with app.run_test() as pilot:
        app.push_screen(
            PermissionScreen("web_fetch", {"url": "https://x.test/" + "a" * 500}, "egress"),
            lambda _a: None,
        )
        await pilot.pause()
        assert isinstance(app.screen, PermissionScreen)
