"""Test filter korpus — Fase 2."""

from src.corpus import MAX_PASANGAN_CHAIN, longest_pasangan_chain, valid_aksara

KANTHI = "ꦏꦤ꧀ꦛꦶ"                 # satu pasangan
MASJID_AKHIR = "ꦩꦱ꧀"             # pangkon di akhir kata, bukan pasangan
SERAPAN = "ꦲꦲꦸꦱ꧀ꦠ꧀ꦧ꧀ꦪ꧀ꦒ꧀ꦢ"      # kata dari baris yang menghentikan training 4b


def test_longest_pasangan_chain():
    assert longest_pasangan_chain("ꦧꦱ") == 0
    assert longest_pasangan_chain(MASJID_AKHIR) == 0
    assert longest_pasangan_chain(KANTHI) == 1
    assert longest_pasangan_chain(SERAPAN) == 5


def test_valid_aksara_rejects_long_pasangan_chains():
    assert MAX_PASANGAN_CHAIN == 3
    assert valid_aksara(KANTHI)
    assert not valid_aksara(SERAPAN)
