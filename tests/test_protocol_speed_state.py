from core.io.protocol import (
    build_packet,
    parse_packet,
    resolve_configured_speed_state,
    target_speed_to_speed_state,
    validate_drive_speed_state,
)
import pytest


def test_target_speed_is_quantized_to_four_states() -> None:
    assert target_speed_to_speed_state(0.0) == 0x00
    assert target_speed_to_speed_state(0.45) == 0x01
    assert target_speed_to_speed_state(1.6) == 0x02
    assert target_speed_to_speed_state(2.2) == 0x03


def test_packet_uses_only_low_two_bits_for_explicit_speed_state() -> None:
    packet = build_packet(
        {
            "lateral_error_px": -2,
            "steer_deg": 258,
            "speed_state": 0xFF,
        }
    )

    assert packet == bytes([0xAA, 0x55, 0xFF, 0xFE, 0x01, 0x02, 0x03])
    assert parse_packet(packet) == {
        "lateral_error_px": -2.0,
        "steer_deg": 258.0,
        "speed_state": 0x03,
    }


def test_packet_derives_speed_state_from_target_speed() -> None:
    expected_states = (0x00, 0x01, 0x02, 0x03)
    for target_speed, expected_state in zip((0.0, 0.45, 1.6, 2.2), expected_states):
        packet = build_packet({"target_speed": target_speed})
        assert len(packet) == 7
        assert packet[6] == expected_state
        assert packet[6] & 0xFC == 0


@pytest.mark.parametrize("configured_state", (0x01, 0x02, 0x03))
def test_configured_drive_state_is_used_only_while_moving(configured_state: int) -> None:
    assert resolve_configured_speed_state(1.6, configured_state) == configured_state
    assert resolve_configured_speed_state(0.0, configured_state) == 0x00


@pytest.mark.parametrize("invalid_state", (-1, 0, 4, 255))
def test_invalid_configured_drive_state_is_rejected(invalid_state: int) -> None:
    with pytest.raises(ValueError, match="drive_speed_state"):
        validate_drive_speed_state(invalid_state)
