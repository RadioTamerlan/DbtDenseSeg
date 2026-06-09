"""A fun coal-mining themed tqdm progress bar.

`mine(iterable, desc=...)` keeps tqdm's real bar / rate / ETA but renders the
filled portion as a dug tunnel and animates a little miner swinging a pickaxe.
On consoles that can't encode the emoji/box glyphs (e.g. a legacy Windows
cp1252 terminal) it automatically falls back to a plain ASCII bar instead of
crashing with UnicodeEncodeError.
"""
import sys
import itertools
from tqdm import tqdm


def _supports(s):
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        s.encode(enc)
        return True
    except Exception:
        return False


_FANCY = _supports("⛏🪨💎")

if _FANCY:
    _ASCII = " .:-=+*o0#"
    _TAIL = " 🪨"
    _SWING = ["(>'-')>⛏", "(>'-')>‾⛏", "(>'o')>⛏̲", "(>'-')>⛏"]
    _GOLD = "💎  {desc} — struck gold!"
else:                                   # ASCII-only fallback (Windows cmd, etc.)
    _ASCII = True                       # tqdm's built-in ascii bar
    _TAIL = ""
    _SWING = ["(>'-')>T", "(>'-')>-T", "(>'o')>=T", "(>'-')>T"]
    _GOLD = "[done] {desc}"

_BAR_FMT = ("{desc} |{bar}| {percentage:3.0f}% "
            "[{n_fmt}/{total_fmt} | {elapsed}<{remaining} | {rate_fmt}]" + _TAIL)


def mine(iterable, desc="mining coal", total=None, leave=True, **kw):
    try:
        total = total if total is not None else len(iterable)
    except TypeError:
        total = None
    bar = tqdm(iterable, total=total, leave=leave, dynamic_ncols=True,
               bar_format=_BAR_FMT, ascii=_ASCII, **kw)
    swing = itertools.cycle(_SWING)
    for x in bar:
        bar.set_description_str(f"{next(swing)}  {desc}")
        yield x
    bar.set_description_str(_GOLD.format(desc=desc))
    bar.refresh()
    bar.close()
