"""The ROI editor page must actually run in a browser (HLD 8, ops tooling).

The page is a self-contained HTML+JS string embedded in a Python module, which makes one
failure mode very easy and completely silent: ``_ROI_HTML`` is a *non-raw* string, so a
``\\n`` written with a single backslash becomes a **real newline inside a JavaScript string
literal**. That is a SyntaxError, the whole script fails to parse, and every handler on the
page dies — including the one that populates the camera dropdown. The page still returns
200 and still looks fine to a server-side test; it is just inert.

That is not hypothetical: it shipped. These tests parse the JavaScript the page actually
serves, so a broken string literal fails here instead of in front of an operator staring at
an empty camera list.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

_SCRIPT = re.compile(r"<script>(.*?)</script>", re.S)


def _served_js(client: TestClient) -> str:
    response = client.get("/api/v1/tools/roi")
    assert response.status_code == 200
    scripts = _SCRIPT.findall(response.text)
    assert scripts, "the ROI page served no <script> block"
    return "\n".join(scripts)


def test_no_string_literal_is_broken_by_a_raw_newline(client: TestClient) -> None:
    """A ``\\n`` written with one backslash silently becomes a real newline and kills the page.

    Counting unescaped quotes per line catches it without needing a JS engine: a line that
    opens a string and never closes it has run off the end of a literal.
    """
    offenders: list[tuple[int, str]] = []
    for number, line in enumerate(_served_js(client).splitlines(), start=1):
        # Ignore escaped quotes; an odd count of the rest means an unterminated literal.
        singles = len(re.findall(r"(?<!\\)'", line))
        doubles = len(re.findall(r'(?<!\\)"', line))
        if singles % 2 or doubles % 2:
            offenders.append((number, line.strip()))

    assert not offenders, (
        "unterminated JS string literal(s) — a real newline inside a quote will "
        f"SyntaxError the whole page: {offenders}"
    )


def test_the_page_still_wires_up_the_camera_dropdown(client: TestClient) -> None:
    """The symptom of the break was an empty camera list, so assert the wiring exists."""
    js = _served_js(client)

    assert "loadCameras" in js
    assert "/cameras" in js
    # Called at load, not only on a token change — otherwise a stored token shows nothing.
    assert re.search(r"^loadCameras\(\);", js, re.M), (
        "loadCameras() is never invoked at page load"
    )


def test_the_editor_requires_a_space_before_saving_a_zone(client: TestClient) -> None:
    """Occupancy history is keyed by space; a zone without one is invisible to it.

    The API rejects it (see test_api_zones_lines), but the page must not offer to try.
    """
    js = _served_js(client)

    assert "dt_space_id is required" in js
    assert "confirm('No dt_space_id set" not in js, (
        "the space-less save is guarded by a click-through dialog again"
    )
