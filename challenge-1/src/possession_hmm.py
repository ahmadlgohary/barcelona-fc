"""Possession labelling: per-frame evidence + Viterbi decoding over {H, A, D}.

    python src/possession_hmm.py --output submissions/possession_hmm.csv
    python evaluation/validate_submission.py submissions/possession_hmm.csv

No learning and no labels of any kind are used. Every constant below is a football prior chosen
before looking at scores, not fitted to the public window.

Pipeline
--------
1. Ball track: short gaps (<= GAP_FILL frames) are linearly interpolated; longer gaps stay missing.
2. Control evidence: nearest home / away player to the ball. A team "controls" the ball only when
   a player is close and the ball is not in flight, so evidence = closeness * tanh((d_away -
   d_home) / TEAM_SCALE). Contested balls and balls in the air give ~0 evidence, so the previous
   possessor keeps the ball (rule: loose balls stay with the last team that controlled them).
3. Dead-ball evidence: ball outside the pitch, ball stationary for seconds (set-piece set-up),
   and team-wide player activity (walking = stoppage, running = play, even with the ball lost).
4. Smart Tagging phases (Barcelona / Athletic tags) add a weak team prior.
5. Viterbi over H/A/D per half with switching costs. The H<->A cost implements "controlled
   possession": a hand-over needs about a second of clear control by the other side, so touches,
   deflections and lost duels do not split a possession.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, source_frames  # noqa: E402
import atd  # noqa: E402
import load_smart_tagging  # noqa: E402

FPS = 25
H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

# ---- parameters (priors) -------------------------------------------------------------------
GAP_FILL = 12            # frames of ball loss bridged by interpolation
CTRL_RADIUS = 3.0        # m, scale of the "ball is at a player's feet" weight
TEAM_SCALE = 2.0         # m, how much closer one team must be to be believed
FLIGHT_SPEED = 14.0      # m/s, above this the ball is in flight, control weight -> 0
OOB_MARGIN = 0.5         # m outside the lines before the ball counts as out
STILL_WIN = 40           # frames (+-1.6 s) used to decide that the ball is stationary
STILL_DIST = 1.0         # m of extent within that window below which it is "stationary"
ACT_WIN = 75             # frames (3 s) over which team-wide player speed is averaged
ACT_THR = 1.5            # m/s: walking pace. Live play is run at, stoppages are walked through
ACT_SCALE = 0.6          # m/s over which the activity cue saturates
W_OOB, W_STILL, W_ACT = 3.0, 0.6, 0.5
W_CTRL_AGAINST_DEAD = 0.5
W_TAG_TEAM = 0.10
COST_SWITCH = 20.0       # H<->A  (~1 s of clear control; chosen so mean possession ~14 s, see TECHNICAL.md)
COST_TO_DEAD = 15.0      # H/A -> D
COST_FROM_DEAD = 10.0    # D -> H/A


def fill_ball(ball):
    """Linear interpolation across gaps of at most GAP_FILL frames."""
    out = ball.copy()
    idx = np.flatnonzero(np.isfinite(ball[:, 0]))
    for a, b in zip(idx[:-1], idx[1:]):
        g = b - a
        if 1 < g <= GAP_FILL + 1:
            t = (np.arange(1, g) / g)[:, None]
            out[a + 1:b] = ball[a] + t * (ball[b] - ball[a])
    return out


def missing_run_length(have):
    """Per frame: length of the missing-ball run it belongs to (0 where the ball is present)."""
    n = len(have)
    out = np.zeros(n, dtype=np.int32)
    edges = np.flatnonzero(np.diff(np.concatenate(([1], have.astype(np.int8), [1]))))
    for s, e in zip(edges[::2], edges[1::2]):
        out[s:e] = e - s
    return out


def tag_prior(n):
    """Weak team prior (+1 home / -1 away / 0) from tagged phases, per frame."""
    prior = np.zeros(n, dtype=np.float32)
    try:
        events = load_smart_tagging.load()
    except IOError:
        return prior
    for e in events:
        if e["code"].upper() == "SET PIECES":
            continue
        team = (e["tags"].get("Team") or "").lower()
        if team.startswith("barcelona"):
            v = 1.0
        elif team.startswith(("atletic", "athletic")):
            v = -1.0
        else:
            continue
        prior[max(0, e["start_frame"]):min(n, e["end_frame"])] += v
    return np.clip(prior, -1, 1)


def frame_evidence(data):
    """Per-frame emission scores (n, 3) for H, A, D."""
    pos, ball_raw, team = data["pos"], data["ball"], data["team"]
    n = len(ball_raw)
    ball = fill_ball(ball_raw)
    have = np.isfinite(ball[:, 0])

    dH = np.full(n, np.inf, dtype=np.float32)
    dA = np.full(n, np.inf, dtype=np.float32)
    for s in range(0, n, 20000):
        e = min(n, s + 20000)
        d = np.hypot(pos[s:e, :, 0] - ball[s:e, None, 0], pos[s:e, :, 1] - ball[s:e, None, 1])
        d = np.where(np.isfinite(d), d, np.inf)
        dH[s:e] = d[:, team == 0].min(axis=1)
        dA[s:e] = d[:, team == 1].min(axis=1)

    k = 3
    sp = np.full(n, np.nan)
    sp[k:-k] = np.hypot(*(ball[2 * k:] - ball[:-2 * k]).T) / (2 * k) * FPS

    dmin = np.minimum(dH, dA)
    close = np.exp(-(np.minimum(dmin, 50) / CTRL_RADIUS) ** 2)
    flight = np.where(np.isfinite(sp), np.clip(2.0 - sp / FLIGHT_SPEED, 0.0, 1.0), 1.0)
    w_ctrl = np.where(have & np.isfinite(dmin), close * flight, 0.0)
    diff = np.clip(dA, 0, 60) - np.clip(dH, 0, 60)
    diff = np.where(np.isfinite(dH) | np.isfinite(dA), diff, 0.0)
    e_team = w_ctrl * np.tanh(diff / TEAM_SCALE)

    L, W = GAMES[GAME]["pitch"]
    oob = have & ((ball[:, 0] < -OOB_MARGIN) | (ball[:, 0] > L + OOB_MARGIN) |
                  (ball[:, 1] < -OOB_MARGIN) | (ball[:, 1] > W + OOB_MARGIN))

    r = pd.DataFrame(ball, columns=["x", "y"]).rolling(2 * STILL_WIN + 1, center=True,
                                                       min_periods=int(STILL_WIN * 1.6))
    spread = r.max() - r.min()
    ext = np.hypot(spread["x"].values, spread["y"].values)
    still = np.isfinite(ext) & (ext < STILL_DIST)

    # Activity: mean speed of all tracked players. It does not need the ball, which is exactly
    # why it is useful when the ball is lost: slow = stoppage, fast = play going on unseen.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # frames with no tracked player
        mean_speed = np.nanmean(data["spd"], axis=1)
    act = pd.Series(mean_speed).rolling(ACT_WIN, center=True, min_periods=ACT_WIN // 4).mean().values
    slow = np.where(np.isfinite(act), np.clip((ACT_THR - act) / ACT_SCALE, -1.0, 1.0), 0.0)

    dead = W_OOB * oob + W_STILL * still + W_ACT * slow
    tag = tag_prior(n)

    ev = np.zeros((n, 3), dtype=np.float32)
    ev[:, H] = e_team + W_TAG_TEAM * tag
    ev[:, A] = -e_team - W_TAG_TEAM * tag
    ev[:, D] = dead - W_CTRL_AGAINST_DEAD * w_ctrl
    return ev


def viterbi(ev):
    n = len(ev)
    trans = np.array([[0, COST_SWITCH, COST_TO_DEAD],
                      [COST_SWITCH, 0, COST_TO_DEAD],
                      [COST_FROM_DEAD, COST_FROM_DEAD, 0]], dtype=np.float64)
    score = ev[0].astype(np.float64)
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        cand = score[:, None] - trans
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + ev[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def predict(data=None):
    data = data or atd.load()
    n = source_frames("tactical", GAME)
    ev = frame_evidence(data)
    labels = np.full(n, "X", dtype="<U1")
    for _, first, last in half_video_blocks(GAME):
        labels[first:last + 1] = LABELS[viterbi(ev[first:last + 1])]
    return labels


def write_submission(path, labels):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))


def summary(labels):
    for _, first, last in half_video_blocks(GAME):
        seg = labels[first:last + 1]
        poss = int(np.sum((seg[1:] != seg[:-1]) & (seg[1:] != "D")))
        print("frames {}-{}: ~{} possession starts; H {} A {} D {}".format(
            first, last, poss, *(int(np.sum(seg == c)) for c in "HAD")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", default=os.path.join(ROOT, "submissions", "possession_hmm.csv"))
    args = ap.parse_args()
    lab = predict()
    write_submission(args.output, lab)
    summary(lab)
    print("wrote", args.output)
