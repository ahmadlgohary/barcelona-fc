"""Load the raw ATD into dense numpy arrays (cached under data/cache, which is gitignored)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, DATA_DIR, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402

N_SLOTS = 46


def _parse(raw_path, n_frames):
    pos = np.full((n_frames, N_SLOTS, 2), np.nan, dtype=np.float32)
    spd = np.full((n_frames, N_SLOTS), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    nan = float("nan")

    def num(s):
        return nan if s[0] == "N" else float(s)

    with open(raw_path) as fh:
        for line in fh:
            p = line.split(":")
            if len(p) != 3:
                continue
            f = int(p[0]) - 1
            for k, g in enumerate(p[1].split(";")):
                if g[0] == "N" and g[1] == "a" and g.startswith("NaN,NaN"):
                    continue
                x, y, s = g.split(",")
                pos[f, k, 0] = num(x)
                pos[f, k, 1] = num(y)
                spd[f, k] = num(s)
            b = p[2].strip()
            if b[0] != "N":
                x, y, _ = b.split(",")
                ball[f, 0] = float(x)
                ball[f, 1] = float(y)
    return pos, spd, ball


def load(game=GAME):
    """Returns dict: pos (F,46,2) metres, spd (F,46), ball (F,2) metres, team (46,) 0/1."""
    cache = os.path.join(DATA_DIR, "cache")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, game + "_tactical_atd.npz")
    meta, raw = require(["atd_meta", "atd_raw"], game)
    if os.path.isfile(path) and os.path.getmtime(path) > os.path.getmtime(raw):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    n = source_frames("tactical", game)
    pos, spd, ball = _parse(raw, n)
    L, W = GAMES[game]["pitch"]
    pos[..., 0] *= L
    pos[..., 1] *= W
    ball[:, 0] *= L
    ball[:, 1] *= W
    team = team_of_each_track(meta)
    np.savez(path, pos=pos, spd=spd, ball=ball, team=team)
    return {"pos": pos, "spd": spd, "ball": ball, "team": team}
