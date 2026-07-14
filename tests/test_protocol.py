from core.protocol import build_packet, normalize_payload, parse_packet


def test_seven_byte_packet_sets_motion_flag_for_positive_speed() -> None:
    packet = build_packet({
        "lateral_error_px": -123,
        "steer_deg": 45,
        "target_speed": 0.25,
        "motion_flag": 1,
    })

    assert len(packet) == 7
    assert packet[:2] == bytes([0xAA, 0x55])
    assert packet[6] == 0x01
    assert parse_packet(packet) == {
        "lateral_error_px": -123.0,
        "steer_deg": 45.0,
        "motion_flag": 1,
    }


def test_seven_byte_packet_sets_zero_for_stop() -> None:
    packet = build_packet({
        "lateral_error_px": 32767,
        "steer_deg": -32768,
        "target_speed": 0.0,
        "motion_flag": 0,
    })

    assert len(packet) == 7
    assert packet[6] == 0x00
    assert parse_packet(packet)["motion_flag"] == 0


def test_motion_flag_uses_only_low_bit_and_can_derive_from_speed() -> None:
    assert normalize_payload({"motion_flag": 0xFF})["motion_flag"] == 1
    assert normalize_payload({"target_speed": 1.0})["motion_flag"] == 1
    assert normalize_payload({"target_speed": 0.0})["motion_flag"] == 0
    assert build_packet({"motion_flag": 0xFE})[6] == 0


def test_legacy_six_byte_packet_is_rejected() -> None:
    assert parse_packet(bytes([0xAA, 0x55, 0, 0, 0, 0])) == {}
