import pytest

from gto.cards import parse_card_sequence


def test_parse_card_sequence_compact_and_spaced() -> None:
    assert parse_card_sequence("Kh8s2c") == parse_card_sequence("Kh 8s 2c")


def test_parse_card_sequence_rejects_duplicates() -> None:
    with pytest.raises(ValueError):
        parse_card_sequence("KhKh2c")
