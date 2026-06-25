"""Unit tests for :class:`app.analytics.entry_exit.EntryExitCalculator`.

The calculator consumes ``Crossing`` events and maintains cumulative per-area and
per-line IN/OUT/net tallies, emitting a ``CrossingEvent`` for every crossing plus
one ``CountUpdate`` per affected area. These tests feed real ``Crossing`` values
and assert the returned events and the running totals.
"""

from __future__ import annotations

from app.analytics.entry_exit import EntryExitCalculator
from app.domain.models import CrossingDirection
from app.events.events import CountUpdate, CrossingEvent
from app.localisation.lines import Crossing
from tests.fixtures.specs import make_line_spec


def _crossing(direction: CrossingDirection, *, line_id=1, area_id="area-1", track_id=100):
    return Crossing(
        line_id=line_id,
        area_id=area_id,
        direction=direction,
        track_id=track_id,
        ts=1000.0,
        dt_space_id=None,
    )


def test_in_then_out_yields_running_in_out_net_per_area(fake_clock):
    calc = EntryExitCalculator(lines=[make_line_spec(id=1)], clock=fake_clock)

    calc.process([_crossing(CrossingDirection.IN, track_id=1)], ts=1000.0)
    calc.process([_crossing(CrossingDirection.OUT, track_id=1)], ts=1001.0)

    assert calc.area_counts("area-1") == (1, 1, 0)


def test_net_reflects_more_ins_than_outs(fake_clock):
    calc = EntryExitCalculator(lines=[make_line_spec(id=1)], clock=fake_clock)

    calc.process(
        [
            _crossing(CrossingDirection.IN, track_id=1),
            _crossing(CrossingDirection.IN, track_id=2),
        ],
        ts=1000.0,
    )
    calc.process([_crossing(CrossingDirection.OUT, track_id=1)], ts=1001.0)

    assert calc.area_counts("area-1") == (2, 1, 1)


def test_each_crossing_yields_a_crossing_event(fake_clock):
    calc = EntryExitCalculator(lines=[make_line_spec(id=1)], clock=fake_clock)

    events = calc.process(
        [
            _crossing(CrossingDirection.IN, track_id=1),
            _crossing(CrossingDirection.OUT, track_id=2),
        ],
        ts=1000.0,
    )

    crossing_events = [e for e in events if isinstance(e, CrossingEvent)]
    assert [(e.track_ref, e.direction) for e in crossing_events] == [
        (1, CrossingDirection.IN),
        (2, CrossingDirection.OUT),
    ]


def test_one_count_update_emitted_per_affected_area(fake_clock):
    calc = EntryExitCalculator(lines=[make_line_spec(id=1)], clock=fake_clock)

    events = calc.process(
        [
            _crossing(CrossingDirection.IN, area_id="area-1", track_id=1),
            _crossing(CrossingDirection.IN, area_id="area-2", line_id=2, track_id=2),
        ],
        ts=1000.0,
    )

    count_updates = [e for e in events if isinstance(e, CountUpdate)]
    assert {e.area_id for e in count_updates} == {"area-1", "area-2"}


def test_count_update_carries_cumulative_totals(fake_clock):
    calc = EntryExitCalculator(lines=[make_line_spec(id=1)], clock=fake_clock)

    calc.process([_crossing(CrossingDirection.IN, track_id=1)], ts=1000.0)
    events = calc.process([_crossing(CrossingDirection.OUT, track_id=1)], ts=1001.0)

    (update,) = [e for e in events if isinstance(e, CountUpdate)]
    assert (update.in_count, update.out_count, update.net) == (1, 1, 0)


def test_per_line_counts_tracked_independently(fake_clock):
    lines = [make_line_spec(id=1, area_id="area-1"), make_line_spec(id=2, area_id="area-1")]
    calc = EntryExitCalculator(lines=lines, clock=fake_clock)

    calc.process(
        [
            _crossing(CrossingDirection.IN, line_id=1, track_id=1),
            _crossing(CrossingDirection.IN, line_id=2, track_id=2),
            _crossing(CrossingDirection.OUT, line_id=2, track_id=2),
        ],
        ts=1000.0,
    )

    assert calc.line_counts(1) == (1, 0, 1)
    assert calc.line_counts(2) == (1, 1, 0)


def test_no_crossings_emits_nothing(fake_clock):
    calc = EntryExitCalculator(lines=[make_line_spec(id=1)], clock=fake_clock)

    assert calc.process([], ts=1000.0) == []
