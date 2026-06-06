"""A fun coal-mining themed tqdm progress bar.

`mine(iterable, desc=...)` keeps tqdm's real bar / rate / ETA but renders the
filled portion as a dug tunnel and animates a little miner swinging a pickaxe.
Falls back gracefully to plain tqdm styling in non-UTF terminals.
"""
import itertools
from tqdm import tqdm

# rubble -> dug-out gradient for the {bar} fill
_ASCII = " .:-=+*o0#"
_BAR_FMT = ("{desc} |{bar}| {percentage:3.0f}% "
            "[{n_fmt}/{total_fmt} | {elapsed}<{remaining} | {rate_fmt}] 🪨")

# pickaxe swing frames (the miner digs into the coal face)
_SWING = ["(>'-')>⛏", "(>'-')>‾⛏", "(>'o')>⛏̲", "(>'-')>⛏"]


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
    # celebrate at the end
    bar.set_description_str(f"💎  {desc} — struck gold!")
    bar.refresh()
    bar.close()
