"""Unit tests for :class:`app.analytics.safety.SafetyCalculator`.

The calculator raises ``AlertRaised`` events for intrusion (a confirmed member in
a restricted zone) and overcrowding (count strictly above a zone's ``safe_limit``)
once a condition holds for ``debounce_frames`` consecutive frames, with a per-zone
cooldown suppressing repeats. The deterministic ``FakeClock`` is the time source;
the per-frame ``ts`` drives the cooldown comparison.
"""

from __future__ import annotations

from app.analytics.safety import SafetyCalculator
from app.domain.models import AlertType, ZoneType
from app.events.events import AlertRaised
from tests.fixtures.specs import make_zone_spec


def test_intrusion_does_not_fire_before_debounce(fake_clock):
    zone = make_zone_spec(id=5, type=ZoneType.RESTRICTED)
    calc = SafetyCalculator(
        camera_id=2,
        zones=[zone],
        clock=fake_clock,
        debounce_frames=3,
        cooldown_seconds=60.0,
    )

    first = calc.process({5: {1}}, {5: 1}, ts=1000.0)
    second = calc.process({5: {1}}, {5: 1}, ts=1001.0)

    assert first == []
    assert second == []


def test_intrusion_fires_on_debounce_frame(fake_clock):
    zone = make_zone_spec(id=5, type=ZoneType.RESTRICTED)
    calc = SafetyCalculator(
        camera_id=2,
        zones=[zone],
        clock=fake_clock,
        debounce_frames=3,
        cooldown_seconds=60.0,
    )

    calc.process({5: {1}}, {5: 1}, ts=1000.0)
    calc.process({5: {1}}, {5: 1}, ts=1001.0)
    third = calc.process({5: {1}}, {5: 1}, ts=1002.0)

    (alert,) = third
    assert isinstance(alert, AlertRaised)
    assert (alert.alert_type, alert.zone_id) == (AlertType.INTRUSION, 5)


def test_intrusion_detail_describes_restricted_zone(fake_clock):
    zone = make_zone_spec(id=5, name="vault", type=ZoneType.RESTRICTED)
    calc = SafetyCalculator(
        camera_id=2,
        zones=[zone],
        clock=fake_clock,
        debounce_frames=1,
        cooldown_seconds=60.0,
    )

    (alert,) = calc.process({5: {1, 2}}, {5: 2}, ts=1000.0)

    assert "vault" in alert.detail
    assert "2 person(s)" in alert.detail


def test_intrusion_streak_resets_when_zone_clears(fake_clock):
    zone = make_zone_spec(id=5, type=ZoneType.RESTRICTED)
    calc = SafetyCalculator(
        camera_id=2,
        zones=[zone],
        clock=fake_clock,
        debounce_frames=3,
        cooldown_seconds=60.0,
    )

    calc.process({5: {1}}, {5: 1}, ts=1000.0)
    calc.process({5: {1}}, {5: 1}, ts=1001.0)
    calc.process({5: set()}, {5: 0}, ts=1002.0)  # clears -> streak resets
    again = calc.process({5: {1}}, {5: 1}, ts=1003.0)

    assert again == []


def test_overcrowding_fires_when_count_exceeds_safe_limit_after_debounce(fake_clock):
    zone = make_zone_spec(id=8, type=ZoneType.OCCUPANCY, safe_limit=2)
    calc = SafetyCalculator(
        camera_id=2,
        zones=[zone],
        clock=fake_clock,
        debounce_frames=2,
        cooldown_seconds=60.0,
    )

    calc.process({8: {1, 2, 3}}, {8: 3}, ts=1000.0)
    second = calc.process({8: {1, 2, 3}}, {8: 3}, ts=1001.0)

    (alert,) = second
    assert (alert.alert_type, alert.zone_id) == (AlertType.OVERCROWDING, 8)


def test_overcrowding_does_not_fire_at_or_below_safe_limit(fake_clock):
    zone = make_zone_spec(id=8, type=ZoneType.OCCUPANCY, safe_limit=2)
    calc = SafetyCalculator(
        camera_id=2,
        zones=[zone],
        clock=fake_clock,
        debounce_frames=1,
        cooldown_seconds=60.0,
    )

    at_limit = calc.process({8: {1, 2}}, {8: 2}, ts=1000.0)

    assert at_limit == []


def test_cooldown_suppresses_second_alert_within_window(fake_clock):
    zone = make_zone_spec(id=5, type=ZoneType.RESTRICTED)
    calc = SafetyCalculator(
        camera_id=2,
        zones=[zone],
        clock=fake_clock,
        debounce_frames=1,
        cooldown_seconds=30.0,
    )

    first = calc.process({5: {1}}, {5: 1}, ts=fake_clock.now())
    fake_clock.advance(10.0)  # still inside the 30s cooldown
    second = calc.process({5: {1}}, {5: 1}, ts=fake_clock.now())

    assert len(first) == 1
    assert second == []


def test_alert_fires_again_after_cooldown_elapses(fake_clock):
    zone = make_zone_spec(id=5, type=ZoneType.RESTRICTED)
    calc = SafetyCalculator(
        camera_id=2,
        zones=[zone],
        clock=fake_clock,
        debounce_frames=1,
        cooldown_seconds=30.0,
    )

    calc.process({5: {1}}, {5: 1}, ts=fake_clock.now())
    fake_clock.advance(31.0)  # past the cooldown window
    second = calc.process({5: {1}}, {5: 1}, ts=fake_clock.now())

    (alert,) = second
    assert alert.alert_type is AlertType.INTRUSION


# --------------------------------------------------------------------------- #
# Boundary tests (testing-guard boundary-value analysis)
#
# Overcrowding is active iff ``count > safe_limit`` (strict). The two boundary
# values around that threshold are ``count == safe_limit`` (must NOT fire) and
# ``count == safe_limit + 1`` (must fire once debounce is satisfied). Intrusion
# is active iff a restricted zone has >= 1 confirmed member; its boundary is the
# debounce frame itself — frame ``debounce_frames - 1`` must NOT fire, frame
# ``debounce_frames`` must.
# --------------------------------------------------------------------------- #
def test_overcrowding_does_not_fire_at_exactly_safe_limit(fake_clock):
    zone = make_zone_spec(id=8, type=ZoneType.OCCUPANCY, safe_limit=3)
    calc = SafetyCalculator(
        camera_id=2,
        zones=[zone],
        clock=fake_clock,
        debounce_frames=2,
        cooldown_seconds=60.0,
    )

    # count == safe_limit on every frame: condition never becomes active, so
    # even past the debounce window no alert is emitted.
    first = calc.process({8: {1, 2, 3}}, {8: 3}, ts=1000.0)
    second = calc.process({8: {1, 2, 3}}, {8: 3}, ts=1001.0)
    third = calc.process({8: {1, 2, 3}}, {8: 3}, ts=1002.0)

    assert first == []
    assert second == []
    assert third == []


def test_overcrowding_fires_at_safe_limit_plus_one_after_debounce(fake_clock):
    zone = make_zone_spec(id=8, type=ZoneType.OCCUPANCY, safe_limit=3)
    calc = SafetyCalculator(
        camera_id=2,
        zones=[zone],
        clock=fake_clock,
        debounce_frames=2,
        cooldown_seconds=60.0,
    )

    # count == safe_limit + 1: active. Debounce is 2, so frame 1 holds and frame
    # 2 fires exactly the overcrowding alert.
    first = calc.process({8: {1, 2, 3, 4}}, {8: 4}, ts=1000.0)
    second = calc.process({8: {1, 2, 3, 4}}, {8: 4}, ts=1001.0)

    assert first == []
    (alert,) = second
    assert (alert.alert_type, alert.zone_id) == (AlertType.OVERCROWDING, 8)


def test_intrusion_fires_exactly_at_debounce_frame_not_before(fake_clock):
    zone = make_zone_spec(id=5, type=ZoneType.RESTRICTED)
    calc = SafetyCalculator(
        camera_id=2,
        zones=[zone],
        clock=fake_clock,
        debounce_frames=3,
        cooldown_seconds=60.0,
    )

    # Frames 1 and 2 are below the debounce window; frame 3 is the boundary and
    # is the first frame that may fire.
    before = [
        calc.process({5: {1}}, {5: 1}, ts=1000.0),
        calc.process({5: {1}}, {5: 1}, ts=1001.0),
    ]
    on_debounce_frame = calc.process({5: {1}}, {5: 1}, ts=1002.0)

    assert before == [[], []]
    (alert,) = on_debounce_frame
    assert (alert.alert_type, alert.zone_id) == (AlertType.INTRUSION, 5)
