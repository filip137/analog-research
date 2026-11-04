"""Physical ReRAM preset identities (no numerical imports)."""

OM_PRESET = "reram_array_om"

HFO2_PRESET = "reram_array_hfo2"

SUPPORTED_PRESETS = (OM_PRESET, HFO2_PRESET)

PUBLISHED_CORRUPT_PROBABILITY = {
    OM_PRESET: 0.1348,
    HFO2_PRESET: 0.0977,
}
