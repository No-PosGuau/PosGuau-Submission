"""Possession pipeline: closest-player control events + hysteresis, instead of a plain majority vote.

Put this file in the repo root (next to games.py) and run:

    python possession.py --team YOURTEAM
    python evaluation/validate_submission.py YOURTEAM.csv

Idea
----
1. Interpolate the ball across short tracking gaps (<= --max-ball-gap frames).
2. A team "controls" a frame if its nearest player is within --radius metres of the ball.
3. Control only counts once it is held for >= --min-control frames. The possession owner changes
   only when the OTHER team has a confirmed control run; the change is back-dated to the start of
   that run. Touches, deflections and loose balls therefore never flip possession.
4. D = ball outside the pitch for >= --min-dead frames. Frames outside the two halves are X.
5. Frames where no control run has been confirmed yet are X (spellF1 forward-fills them).

The defaults are untuned guesses. Do NOT tune them on the 5-minute public ground truth
(see EVALUATION.md); only use it to check the file is aligned and well formed.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "starter"))
from games import GAME, GAMES, require, source_frames, half_video_blocks  # noqa: E402
from baseline import team_of_each_track, write_submission  # noqa: E402

HOME, AWAY = 0, 1


def load_tracking(raw_path, n_frames, n_slots, pitch):
    """Player and ball positions in metres, indexed by video frame (frame_id - 1)."""
    length, width = pitch
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan)
    by = np.full(n_frames, np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            if not 0 <= i < n_frames:
                continue
            for slot, group in enumerate(parts[1].split(";")):
                if slot >= n_slots:
                    break
                if group[0] == "N":
                    continue
                c = group.split(",")
                px[i, slot] = float(c[0]) * length
                py[i, slot] = float(c[1]) * width
            ball = parts[2]
            if ball[0] != "N":
                c = ball.split(",")
                bx[i] = float(c[0]) * length
                by[i] = float(c[1]) * width
    return px, py, bx, by


def fill_gaps(v, max_gap):
    """Linear interpolation across NaN gaps no longer than max_gap frames; longer gaps stay NaN."""
    n = len(v)
    idx = np.arange(n)
    ok = ~np.isnan(v)
    if ok.sum() < 2:
        return v.copy()
    out = np.interp(idx, idx[ok], v[ok])
    prev = np.maximum.accumulate(np.where(ok, idx, -1))
    nxt = np.minimum.accumulate(np.where(ok, idx, 2 * n)[::-1])[::-1]
    out[(prev < 0) | (nxt >= 2 * n) | (nxt - prev > max_gap + 1)] = np.nan
    return out


def runs(mask):
    """(starts, ends_exclusive) of the True runs in a boolean array."""
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def control_codes(px, py, bx, by, team, radius):
    """Per frame: 0 home / 1 away if that team's nearest player is within radius, else -1."""
    d = np.hypot(px - bx[:, None], py - by[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    dh = d[:, team == HOME].min(axis=1)
    da = d[:, team == AWAY].min(axis=1)
    ctrl = np.full(len(bx), -1, dtype=np.int8)
    near = np.minimum(dh, da) <= radius
    ctrl[near] = np.where(dh[near] <= da[near], HOME, AWAY)
    return ctrl


def possess_segment(ctrl, min_control, gap_fill):
    """Hysteresis over one live-ball stretch. Returns 0/1 per frame, -1 where undecided."""
    c = ctrl.copy()
    # Bridge short control dropouts inside one team's control.
    for s, e in zip(*runs(c < 0)):
        if 0 < s and e < len(c) and e - s <= gap_fill and c[s - 1] == c[e]:
            c[s:e] = c[s - 1]

    events = []
    for team in (HOME, AWAY):
        for s, e in zip(*runs(c == team)):
            if e - s >= min_control:
                events.append((s, team))
    events.sort()

    comp = []
    for start, team in events:
        if not comp or comp[-1][1] != team:
            comp.append((start, team))

    lab = np.full(len(c), -1, dtype=np.int8)
    if not comp:
        return lab
    comp[0] = (0, comp[0][1])  # first owner holds from the start of the stretch
    for k, (start, team) in enumerate(comp):
        end = comp[k + 1][0] if k + 1 < len(comp) else len(c)
        lab[start:end] = team
    return lab


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default="team", help="name used for the output CSV")
    parser.add_argument("--out", default=".", help="output folder")
    parser.add_argument("--radius", type=float, default=3.0, help="control radius, metres")
    parser.add_argument("--min-control", type=int, default=50, help="frames to confirm control")
    parser.add_argument("--gap-fill", type=int, default=12, help="bridge control dropouts, frames")
    parser.add_argument("--max-ball-gap", type=int, default=50, help="max ball gap to interpolate")
    parser.add_argument("--min-dead", type=int, default=25, help="frames outside pitch for D")
    args = parser.parse_args()

    n_frames = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    team = team_of_each_track(meta_path)

    print("reading tracking ...")
    px, py, bx, by = load_tracking(raw_path, n_frames, len(team), pitch)
    print("ball seen on {:.1f}% of all frames".format(100.0 * np.mean(~np.isnan(bx))))

    bx = fill_gaps(bx, args.max_ball_gap)
    by = fill_gaps(by, args.max_ball_gap)
    ctrl = control_codes(px, py, bx, by, team, args.radius)

    outside = (bx < 0) | (bx > pitch[0]) | (by < 0) | (by > pitch[1])  # NaN compares False
    dead = np.zeros(n_frames, dtype=bool)
    for s, e in zip(*runs(outside)):
        if e - s >= args.min_dead:
            dead[s:e] = True

    labels = np.full(n_frames, "X", dtype="<U1")
    for half, first, last in half_video_blocks(GAME):
        seg = labels[first:last + 1]  # view
        dead_h = dead[first:last + 1]
        ctrl_h = ctrl[first:last + 1]
        for a, b in zip(*runs(~dead_h)):
            lab = possess_segment(ctrl_h[a:b], args.min_control, args.gap_fill)
            seg[a:b] = np.where(lab == HOME, "H", np.where(lab == AWAY, "A", "X"))
        seg[dead_h] = "D"

    path = write_submission(os.path.join(args.out, "{}.csv".format(args.team)), labels)
    counts = {k: int(np.sum(labels == k)) for k in "HADX"}
    print("labels:", counts)
    print("wrote", path)
    print("next: python evaluation/validate_submission.py {}".format(path))


if __name__ == "__main__":
    main()
