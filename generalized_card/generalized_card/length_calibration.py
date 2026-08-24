"""Ask for the length that realizes the matched slot's length.

Every release through v97 rendered the matched slot's own word count into the
Writer's length cue and hoped for compliance. Measured over the 532 comments of
the v97 ten-thread run
(`generalized_card_camera_gpt54_v97_keyboard_n10_20260819_v1`), compliance is a
smooth, monotone regression toward the model's own preferred comment length:

    target words    n    median realized / target
      1-4          14        1.42
      5-7          30        1.33
      8-10         21        1.22
     11-15         44        1.17
     16-25        110        1.05
     26-40         93        1.06
     41-60         65        0.91
     61-90         62        0.91
     91-120        45        0.92
    121-180        25        0.87
    181-250        10        0.88
    251-400        10        0.71
    401+            3        0.89

The crossover sits near 40 words. Short slots come out long and long slots come
out short, so a thread's mean length survives while its spread collapses: pooled
`length_cv` was 0.857 against a real 0.947, below the matched real thread on 9 of
10 threads, Cliff -0.62. It also feeds `self_bertscore_mean_f1`, though not much:
reweighting the generated comment pairs onto the real pairs' length-ratio mix
closes 0.0033 of the 0.0163 gap, about a fifth.

Three releases tried to talk the Writer out of this. v96 added "do not trim
toward a medium-length answer", v97 added the measured paragraph layout, and the
250w+ realization ratio moved 0.61 -> 0.71. That is the return prompt wording
gets. The transfer function itself, however, is clean:

    log(realized) = 0.3835 + 0.8925 * log(asked)     n=532, R2=0.894

so it can be inverted. Asking for `exp((log(target) - a) / b)` words puts the
realized length on the target. The resulting multiplier is 0.71x at two words,
1.00x near 35, and 1.47x at 845 -- monotone, and inside the clamp across the
whole observed range.

This calibrates the number in the cue only. `real_word_count` stays the truth
everywhere else -- the layout profile, the development beats, the tone-length
band, and the substantive length floor all keep reading the matched slot's real
size, because those describe what a comment of that size actually looks like.

The fit is a property of this model and this prompt, not of the domain, so it
lives here as a recorded constant rather than in the domain profile. Both the
target and the asked value are written into every generation record, so the next
run's artifact refits it without rerunning anything.
"""

from __future__ import annotations

import math
from typing import Any


# log(realized_words) = INTERCEPT + SLOPE * log(asked_words), fitted by ordinary
# least squares over the 532 v97 slots. A slope below 1 is the regression toward
# the model's preferred length; inverting it is the whole mechanism.
WORD_TRANSFER_INTERCEPT = 0.3835
WORD_TRANSFER_SLOPE = 0.8925
# The fit's support runs 1-845 words. The clamp does not bind anywhere inside it
# and only bounds extrapolation past the largest slot ever observed.
MIN_ASK_MULTIPLIER = 0.60
MAX_ASK_MULTIPLIER = 1.60

# The v97 fit above regressed realized words on the *uncalibrated* ask, because
# v97 asked for the matched slot's own word count. Every release since asks the
# calibrated value, so the transfer function that matters now is realized-on-
# calibrated-ask -- a different object, and one this module never refitted.
# Refitting it over every artifact that records both numbers (1,436 slots: the
# v97 and v98 N=10 runs plus the v108 and v109 seed-8 gates, so 21 thread
# instances rather than the gate thread alone) gives:
#
#     log(realized) = 0.5580 + 0.8276 * log(asked)     n=1436, R2=0.879
#
# and the residual it leaves is large and one-directional, stable across all
# four runs: realized/asked runs 1.64x below 10 asked words and 0.68-0.80x above
# 80. That residual is the measured cause of the compression in
# `docs/DECISIONS.md` G43 -- the calibration has been under-correcting, not
# absent. Inverting the refitted line asks 167 words for a 121-word slot where
# the v97 constants ask 140, and 4 where they ask 5.
#
# `--length-transfer v97` keeps the constants above and reproduces v109
# byte-for-byte; `refit` selects these.
REFIT_TRANSFER_INTERCEPT = 0.5580
REFIT_TRANSFER_SLOPE = 0.8276
# The refitted line needs 0.51x at one word and 1.61x at 250, so the v97 clamp
# would bind inside the range the fit actually covers. Widened only for the
# refit arm.
REFIT_MIN_ASK_MULTIPLIER = 0.50
REFIT_MAX_ASK_MULTIPLIER = 1.70

LENGTH_TRANSFER_MODE = "v97"
# `off` reproduces every version through v97, which asked for the matched slot's
# own word count.
LENGTH_CALIBRATION_ENABLED = True


def set_length_calibration(mode: str) -> bool:
    """Select the length-calibration arm and return whether it is active."""

    global LENGTH_CALIBRATION_ENABLED
    LENGTH_CALIBRATION_ENABLED = (
        str(mode or "measured").strip().lower() != "off"
    )
    return LENGTH_CALIBRATION_ENABLED


def set_length_transfer(mode: str) -> str:
    """Select which fitted transfer function the calibration inverts."""

    global LENGTH_TRANSFER_MODE
    chosen = str(mode or "v97").strip().lower()
    LENGTH_TRANSFER_MODE = "refit" if chosen == "refit" else "v97"
    return LENGTH_TRANSFER_MODE


def active_transfer() -> tuple[float, float, float, float]:
    """Intercept, slope and clamp bounds for the selected arm."""

    if LENGTH_TRANSFER_MODE == "refit":
        return (
            REFIT_TRANSFER_INTERCEPT,
            REFIT_TRANSFER_SLOPE,
            REFIT_MIN_ASK_MULTIPLIER,
            REFIT_MAX_ASK_MULTIPLIER,
        )
    return (
        WORD_TRANSFER_INTERCEPT,
        WORD_TRANSFER_SLOPE,
        MIN_ASK_MULTIPLIER,
        MAX_ASK_MULTIPLIER,
    )


def ask_multiplier(target_words: Any) -> float:
    """Return the clamped ask/target ratio that realizes `target_words`."""

    target = _safe_int(target_words)
    intercept, slope, low, high = active_transfer()
    if target <= 0 or slope <= 0:
        return 1.0
    asked = math.exp((math.log(target) - intercept) / slope)
    return max(low, min(high, asked / target))


def calibrated_word_ask(target_words: Any) -> int:
    """Return the word count to ask for so `target_words` is what comes back."""

    target = _safe_int(target_words)
    if target <= 0:
        return 0
    if not LENGTH_CALIBRATION_ENABLED:
        return target
    return max(1, int(round(target * ask_multiplier(target))))


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
