"""Predict home, away and dead-ball possession on the tactical video timeline."""

import argparse
import csv
import math
import os
import sys
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import median
from typing import NamedTuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from games import GAME, GAMES, half_video_blocks, require, source_frames


HOME, AWAY, DEAD = range(3)
LABELS = "HAD"
STEP_FRAMES = 5
# Field-wide stoppage cue. Activity is the share of tracked players moving at >= 2 m/s. During play
# players are constantly repositioning; at a stoppage nearly everyone walks. Full evidence when at
# most 1 in 20 players runs, none once 1 in 7 does, averaged over a 3 s window so a single slow
# moment in open play does not count.
QUIET_WINDOW_S = 3.0
QUIET_FULL = 0.05
QUIET_ZERO = 0.15
QUIET_WEIGHT = 0.5             # per 0.2 s block; a 5 s stoppage outweighs entering and leaving D
QUIET_CONTACT_DISCOUNT = 0.6
# Laws of the Game, law 15-17: when the ball leaves the pitch, the team that did not touch it
# last restarts (throw-in, goal kick or corner). Look back up to 3 s for the last clear contact.
LAST_TOUCH_LOOKBACK_S = 3.0
LAST_TOUCH_MIN = 0.25
LAST_TOUCH_WEIGHT = 0.5
RESTART_EVIDENCE_S = 1.2       # how long restart-team evidence lasts after play resumes
# Law 8: after a goal the ball is dead until the conceding team kicks off.
GOAL_DEAD = 1.5                # same strength as the lead-in to a half's opening kick-off
GOAL_MAX_GAP_S = 180.0         # a kick-off later than this is not the one for that goal
# Boundary refinement: decoding works in blocks, so each change is re-placed at frame precision
# within one block either side, using the per-frame evidence for the two states involved.
MOVING_BALL = 2.4              # m/s, same threshold as the block-level "moving" cue
PHASE_WEIGHT = {
    "attacking transition": 0.32,
    "build up": 0.29,
    "creating chances": 0.24,
    "progression": 0.22,
    "long ball": 0.17,
    "goals": 0.20,
    "ball in final third": 0.10,
    "ball in the box": 0.10,
    "defensive transition": 0.17,
    "defending in defensive third": 0.15,
    "defending in middle third": 0.13,
    "defending in attacking third": 0.13,
}


class Observation(NamedTuple):
    ball: Optional[tuple]
    contact: tuple
    activity: float


class Block(NamedTuple):
    ball: Optional[tuple]
    speed: float
    contact: tuple
    activity: float
    outside: float


class Tag(NamedTuple):
    start: int
    end: int
    code: str
    kind: str
    team: Optional[int]


def club_key(name):
    key = unicodedata.normalize("NFKD", name or "").casefold()
    return "".join(character for character in key if character.isascii() and character.isalnum())


def read_teams(path):
    root = ET.parse(path).getroot()
    teams = list(root.iter("Team"))
    team_ids = [team.get("id") for team in teams]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if len(team_ids) != 2 or None in team_ids or len(set(team_ids)) != 2 or not slots:
        raise ValueError(f"{path}: expected two teams and player track slots")
    if any(slot not in team_ids for slot in slots):
        raise ValueError(f"{path}: player track slot has an unknown team")

    names = {club_key(team.findtext("Name")): side for side, team in enumerate(teams)}
    # SportsCode and the EPTS metadata use different spellings for the same clubs.
    names.update({name: HOME for name in ("barcelona", "fcbarcelona", "barcelonafc")})
    names.update({name: AWAY for name in ("atleticclub", "athleticclub", "athleticbilbao")})
    return [team_ids.index(slot) for slot in slots], names


def coordinates(group, pitch):
    fields = group.split(",")
    if fields[0].strip().lower() in ("nan", ""):
        return None
    x, y, speed = map(float, fields)
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x * pitch[0], y * pitch[1], speed


def is_outside_pitch(ball, pitch):
    x, y, _ = ball
    length, width = pitch
    return max(-x, x - length, -y, y - width) > 0.20


def contact_evidence(nearest, ball_speed):
    # Both teams can be nearby during a duel. Proximity only counts as control when one
    # team is close and in a better position than the other.
    result = []
    for team, distance in enumerate(nearest):
        closeness = max(0.0, 1 - distance / 3.0)
        advantage = min(1.0, max(0.0, (nearest[1 - team] - distance) / 1.5)) if closeness else 0.0
        fast_ball = 0.55 if ball_speed > 11.0 else 1.0
        result.append(closeness * advantage * fast_ball)
    return tuple(result)


def read_tracking(path, slots, pitch, frame_count):
    data = []
    with open(path, encoding="utf-8-sig") as stream:
        for frame_id, line in enumerate(stream, start=1):
            try:
                identifier, players, ball_text = line.strip().split(":")
                if int(identifier) != frame_id or frame_id > frame_count:
                    raise ValueError(f"expected tracking frame ID {frame_id}, found {identifier}")
                groups = players.rstrip(";").split(";")
                if len(groups) != len(slots):
                    raise ValueError(f"expected {len(slots)} track slots, found {len(groups)}")

                ball = coordinates(ball_text, pitch)
                ball_on_pitch = ball is not None and not is_outside_pitch(ball, pitch)
                nearest = [math.inf, math.inf]
                running = speed_samples = 0
                for team, group in zip(slots, groups):
                    player = coordinates(group, pitch)
                    if player is None:
                        continue
                    x, y, speed = player
                    if math.isfinite(speed) and speed >= 0.0:
                        speed_samples += 1
                        running += speed >= 2.0
                    if ball_on_pitch:
                        distance = math.hypot(x - ball[0], y - ball[1])
                        nearest[team] = min(nearest[team], distance)

                contact = contact_evidence(nearest, ball[2]) if ball_on_pitch else (0.0, 0.0)
                activity = running / speed_samples if speed_samples >= 6 else math.nan
                data.append(Observation(ball, contact, activity))
            except ValueError as error:
                raise ValueError(f"{path}, line {frame_id}: {error}") from error
    if len(data) != frame_count:
        raise ValueError(f"{path}: expected {frame_count} tactical records, found {len(data)}")
    return data


def read_tags(path, team_names, fps):
    tags = []
    for instance in ET.parse(path).iter("instance"):
        fields = {
            (label.findtext("group") or "").strip().casefold(): label.findtext("text") or ""
            for label in instance.findall("label")
        }
        tags.append(Tag(
            round(float(instance.findtext("start")) * fps),
            round(float(instance.findtext("end")) * fps),
            (instance.findtext("code") or "").strip().casefold(),
            fields.get("type", "").strip().casefold(),
            team_names.get(club_key(fields.get("team"))),
        ))
    return sorted(tags, key=lambda tag: (tag.start, tag.end))


def build_blocks(tracking, pitch, fps):
    blocks = []
    last_ball = None
    last_frame = None
    for start in range(0, len(tracking), STEP_FRAMES):
        observations = tracking[start:start + STEP_FRAMES]
        balls, speeds = [], []
        for frame, observation in enumerate(observations, start):
            ball = observation.ball
            if ball is None:
                continue
            balls.append(ball)
            if math.isfinite(ball[2]) and 0 <= ball[2] <= 45:
                speeds.append(ball[2])
            elif last_ball is not None and frame - last_frame <= STEP_FRAMES:
                speed = math.hypot(ball[0] - last_ball[0], ball[1] - last_ball[1])
                speed *= fps / (frame - last_frame)
                if speed <= 45:
                    speeds.append(speed)
            last_ball, last_frame = ball, frame

        contact = tuple(sum(item.contact[team] for item in observations) / len(observations)
                        for team in (HOME, AWAY))
        activity = [item.activity for item in observations if math.isfinite(item.activity)]
        outside = sum(is_outside_pitch(ball, pitch) for ball in balls)
        position = (median(ball[0] for ball in balls), median(ball[1] for ball in balls)) if balls else None
        blocks.append(Block(position, median(speeds) if speeds else math.nan, contact,
                            median(activity) if activity else math.nan, outside / len(observations)))
    return blocks


def restart_location(ball, kind, pitch):
    if ball is None:
        return 0.0
    x, y = ball
    length, width = pitch
    goal_line = min(abs(x), abs(x - length))
    touchline = min(abs(y), abs(y - width))
    if kind.startswith("kick off"):
        distance, radius = math.hypot(x - length / 2, y - width / 2), 8.0
    elif kind == "throw-in":
        distance, radius = touchline, 5.0
    elif kind == "corner kick":
        distance, radius = math.hypot(goal_line, touchline), 7.0
    elif kind == "goal kick":
        distance = math.hypot(max(0.0, goal_line - 6), max(0.0, abs(y - width / 2) - 10))
        radius = 9.0
    else:
        return 0.0
    return max(0.0, 1 - distance / radius)


def align_restart(blocks, start, end, tag, pitch, blocks_per_second):
    midpoint = (start + end - 1) // 2
    padding = round(blocks_per_second)
    first = max(0, start - padding)
    last = min(len(blocks), end + padding)
    best = min(len(blocks) - 1, max(0, midpoint))
    best_score, confidence = -math.inf, 0.0
    for i in range(first, last):
        ball = blocks[i].ball
        if ball is None:
            continue
        before = blocks[max(0, i - 3):i]
        after = blocks[i:i + 3]
        was_still = any(block.speed < 1.4 for block in before)
        is_moving = sum(block.speed > 3.0 for block in after) >= 2
        comes_back = len(before) == 3 and all(block.ball is None or block.outside >= 0.4
                                               for block in before)
        movement = 1.0 if was_still and is_moving else 0.0
        if comes_back and is_moving:
            movement += 0.55
        geometry = restart_location(ball, tag.kind, pitch)
        team_contact = blocks[i].contact[tag.team] if tag.team is not None else 0.0
        offset_penalty = 0.35 * abs(i - midpoint) / max(padding, end - start)
        score = movement + 0.85 * geometry + 0.25 * team_contact - offset_penalty
        if score > best_score:
            best = i
            best_score = score
            confidence = min(1.0, 0.25 + 0.45 * movement + 0.3 * geometry)
    return best, max(0.25, confidence)


def tag_evidence(blocks, first_frame, tags, pitch, blocks_per_second):
    phases = [[0.0, 0.0] for _ in blocks]
    restart_live = [0.0] * len(blocks)
    restart_dead = [0.0] * len(blocks)
    clip_end = first_frame + STEP_FRAMES * len(blocks)
    goals = [tag for tag in tags if tag.code == "goals"]
    for tag in tags:
        if tag.end <= first_frame or tag.start >= clip_end or tag.end <= tag.start:
            continue
        start = (tag.start - first_frame) // STEP_FRAMES
        end = (tag.end - first_frame + STEP_FRAMES - 1) // STEP_FRAMES
        if tag.code == "set pieces":
            goal = None
            if tag.kind == "kick off (after goal)":
                earlier = [g for g in goals if g.start < tag.start
                           and tag.start - g.end <= GOAL_MAX_GAP_S * STEP_FRAMES * blocks_per_second]
                goal = earlier[-1] if earlier else None
                if tag.team is None and goal is not None and goal.team is not None:
                    tag = tag._replace(team=1 - goal.team)  # the conceding side kicks off
            restart, confidence = align_restart(blocks, start, end, tag, pitch, blocks_per_second)
            if goal is not None:
                goal_end = max(0, (goal.end - first_frame) // STEP_FRAMES)
                for i in range(goal_end, restart):
                    restart_dead[i] = max(restart_dead[i], GOAL_DEAD)
            pre_window = round(24 * blocks_per_second)
            pre_start = max(0, restart - pre_window)
            if tag.kind == "kick off (start)" and restart < 180 * blocks_per_second:
                # A half can be on the graded timeline before the kickoff. Estimate the restart
                # from the automatic tag and the ball, then favour dead ball during the lead-in.
                for i in range(restart):
                    restart_dead[i] = max(restart_dead[i], 1.5)
            for i in range(pre_start, restart):
                decay = 1 - (restart - i) / pre_window
                restart_dead[i] = max(restart_dead[i], 0.75 * confidence * decay)
            for i in range(restart, min(len(blocks), restart + round(1.2 * blocks_per_second))):
                restart_live[i] = max(restart_live[i], confidence)
                if tag.team is not None:
                    phases[i][tag.team] = max(phases[i][tag.team], 0.58 * confidence)
        elif tag.team is not None and tag.code in PHASE_WEIGHT:
            defensive = tag.code.startswith("defend")
            team = 1 - tag.team if defensive else tag.team
            for i in range(max(start, 0), min(end, len(blocks))):
                phases[i][team] = max(phases[i][team], PHASE_WEIGHT[tag.code])
    return phases, restart_live, restart_dead


def smoothed_activity(blocks, blocks_per_second):
    """Centred mean of team activity over QUIET_WINDOW_S seconds, ignoring missing samples."""
    half = round(QUIET_WINDOW_S * blocks_per_second / 2)
    total = [0.0]
    count = [0]
    for block in blocks:
        valid = math.isfinite(block.activity)
        total.append(total[-1] + (block.activity if valid else 0.0))
        count.append(count[-1] + valid)
    result = []
    for i in range(len(blocks)):
        low, high = max(0, i - half), min(len(blocks), i + half + 1)
        samples = count[high] - count[low]
        result.append((total[high] - total[low]) / samples if samples else math.nan)
    return result


def quiet_evidence(activity):
    """1 when almost nobody runs, 0 at normal pace: the whole field stopping is a stoppage."""
    if not math.isfinite(activity):
        return 0.0
    return min(1.0, max(0.0, (QUIET_ZERO - activity) / (QUIET_ZERO - QUIET_FULL)))


def state_evidence(blocks, phases, restart_live, restart_dead, blocks_per_second):
    emissions = []
    still_blocks = missing_blocks = 0
    recent_out = 0.0
    field_activity = smoothed_activity(blocks, blocks_per_second)
    for i, block in enumerate(blocks):
        missing = block.ball is None
        moving = block.speed > 2.4 and block.outside < 0.4
        still = block.speed < 0.9 and block.outside < 0.4
        still_blocks = still_blocks + 1 if still else 0
        missing_blocks = missing_blocks + 1 if missing else 0
        contact = max(block.contact)
        phase = max(phases[i])
        live = 0.26 * moving + 0.10 * (contact > 0.3) + 0.16 * (block.activity > 0.40)
        live += 0.28 * phase + 0.95 * restart_live[i]
        dead = 0.0
        if block.outside >= 0.4:
            recent_out = 1.0
            dead += 1.6 * block.outside
        elif missing:
            dead += 0.50 * recent_out
        else:
            recent_out = 0.0
        recent_out = max(0.0, recent_out - 1 / (12 * blocks_per_second))

        quiet = block.activity < 0.22
        stationary = still_blocks >= round(0.8 * blocks_per_second)
        if stationary and quiet:
            dead += 0.36 if phase < 0.2 else 0.12
        if missing_blocks >= round(1.6 * blocks_per_second) and quiet and phase < 0.2:
            dead += 0.055
        if not moving or restart_dead[i] >= 1.0:
            dead += restart_dead[i] * (1.0 if still or missing or quiet else 0.45)
        # Field-wide stoppage: sustained walking pace with the ball not in motion. This needs no
        # ball position, which matters because the ball is often untracked during stoppages, and
        # most stoppages (fouls, many throw-ins) have no Smart Tagging set piece.
        field_quiet = 0.0 if moving else quiet_evidence(field_activity[i])
        dead += QUIET_WEIGHT * field_quiet
        dead -= 0.6 * restart_live[i]
        contact_multiplier = 1.20 * (1 - min(1.0, 2 * block.outside))
        # A player standing over the ball while nobody runs is setting up a restart, not
        # carrying a possession.
        contact_multiplier *= 1 - QUIET_CONTACT_DISCOUNT * field_quiet
        if stationary or restart_dead[i] >= 1.0:
            # The player standing over a ball before an automated restart cue is
            # weaker evidence of live possession than a moving player's contact.
            contact_multiplier *= 1 - min(0.85, 1.6 * restart_dead[i])
        team_evidence = [contact_multiplier * block.contact[side] + phases[i][side]
                         for side in (HOME, AWAY)]
        # Only a team's advantage should count as possession evidence. Overlapping equally
        # strong tactical phases still supply a weak live-ball cue via the common term above.
        shared = min(team_evidence)
        emissions.append((live + team_evidence[HOME] - shared,
                          live + team_evidence[AWAY] - shared, dead))
    return emissions


def team_switch_cost(contact, phases, receiving_team):
    control = contact[receiving_team]
    phase_advantage = max(0.0, phases[receiving_team] - phases[1 - receiving_team])
    if control == 0.0 and phase_advantage == 0.0:
        return math.inf
    return 5.0 - 1.7 * control - phase_advantage


def decode_states(blocks, emissions, phases, restart_live, restart_dead):
    """Viterbi decoding with switch costs and deterministic tie-breaking."""
    scores = (0.0, 0.0, 0.0)
    back = []
    for i, (home, away, dead) in enumerate(emissions):
        block = blocks[i]
        from_dead = max(0.8, 4.4 - 3.6 * restart_live[i])
        to_dead = 2.8 - min(1.3, restart_dead[i] + block.outside)
        # A direct handover needs evidence for the receiving side at the transition.
        switch_to_home = team_switch_cost(block.contact, phases[i], HOME)
        switch_to_away = team_switch_cost(block.contact, phases[i], AWAY)
        choices = (
            ((scores[HOME], HOME), (scores[AWAY] - switch_to_home, AWAY),
             (scores[DEAD] - from_dead, DEAD)),
            ((scores[AWAY], AWAY), (scores[HOME] - switch_to_away, HOME),
             (scores[DEAD] - from_dead, DEAD)),
            ((scores[DEAD], DEAD), (scores[HOME] - to_dead, HOME),
             (scores[AWAY] - to_dead, AWAY)),
        )
        best = [max(options, key=lambda option: option[0]) for options in choices]
        back.append(tuple(previous for _, previous in best))
        scores = (best[HOME][0] + home, best[AWAY][0] + away, best[DEAD][0] + dead)

    state = max(range(len(LABELS)), key=scores.__getitem__)
    path = []
    for previous in reversed(back):
        path.append(state)
        state = previous[state]
    return list(reversed(path))


def last_touch(blocks, out, blocks_per_second):
    """Team with the last clear contact before the ball left the pitch at block `out`, or None."""
    for i in range(out, max(-1, out - round(LAST_TOUCH_LOOKBACK_S * blocks_per_second)), -1):
        home, away = blocks[i].contact
        if max(home, away) >= LAST_TOUCH_MIN and abs(home - away) > 0.1:
            return HOME if home > away else AWAY
    return None


def out_of_play_restarts(blocks, path, blocks_per_second):
    """[(first live block, restarting team)] for stoppages that began with the ball going out."""
    restarts = []
    i = 0
    while i < len(path):
        if path[i] != DEAD:
            i += 1
            continue
        j = i
        while j < len(path) and path[j] == DEAD:
            j += 1
        if j < len(path):
            window = range(max(0, i - round(2 * blocks_per_second)),
                           min(j, i + round(5 * blocks_per_second)))
            out = next((k for k in window if blocks[k].outside >= 0.4), None)
            if out is not None:
                toucher = last_touch(blocks, out, blocks_per_second)
                if toucher is not None:
                    restarts.append((j, 1 - toucher))
        i = j
    return restarts


def frame_score(observation, state, pitch):
    """Per-frame evidence for one state, used only to place a boundary inside a block."""
    ball = observation.ball
    outside = ball is not None and is_outside_pitch(ball, pitch)
    if state == DEAD:
        return 1.0 if outside else 0.0
    moving = (ball is not None and not outside and math.isfinite(ball[2])
              and ball[2] > MOVING_BALL)
    return observation.contact[state] + (1.0 if moving else 0.0)


def refine_boundaries(path, tracking, pitch):
    """Per-frame states, with every block boundary moved to the best frame within one block.

    Only boundaries between two segments of at least two blocks are moved, so neighbouring
    refinement windows never overlap. Ties keep the boundary closest to where decoding put it.
    """
    states = [state for state in path for _ in range(STEP_FRAMES)][:len(tracking)]
    for b in range(2, len(path) - 1):
        source, target = path[b - 1], path[b]
        if source == target or path[b - 2] != source or path[b + 1] != target:
            continue
        boundary = b * STEP_FRAMES
        low, high = boundary - STEP_FRAMES, min(boundary + STEP_FRAMES, len(tracking))
        if boundary >= high:
            continue
        difference = [frame_score(tracking[f], target, pitch) - frame_score(tracking[f], source, pitch)
                      for f in range(low, high)]
        suffix = [0.0] * (len(difference) + 1)
        for k in range(len(difference) - 1, -1, -1):
            suffix[k] = suffix[k + 1] + difference[k]
        base = suffix[boundary - low]
        best, best_gain = boundary, 0.0
        for frame in range(low, high + 1):
            gain = suffix[frame - low] - base
            if gain > best_gain + 1e-9 or (abs(gain - best_gain) <= 1e-9
                                           and abs(frame - boundary) < abs(best - boundary)):
                best, best_gain = frame, gain
        for f in range(low, high):
            states[f] = source if f < best else target
    return states


def predict(data_dir=None):
    metadata, raw_tracking, smart_tagging = require(
        ("atd_meta", "atd_raw", "smart_tagging"), data_dir=data_dir
    )
    config = GAMES[GAME]
    fps, pitch = config["fps"], config["pitch"]
    blocks_per_second = fps / STEP_FRAMES
    frame_count = source_frames()
    slots, team_names = read_teams(metadata)
    tags = read_tags(smart_tagging, team_names, fps)
    tracking = read_tracking(raw_tracking, slots, pitch, frame_count)
    labels = ["X"] * frame_count
    for _, first, last in half_video_blocks(GAME):
        half = tracking[first:last + 1]
        blocks = build_blocks(half, pitch, fps)
        tag_scores = tag_evidence(blocks, first, tags, pitch, blocks_per_second)
        emissions = state_evidence(blocks, *tag_scores, blocks_per_second)
        path = decode_states(blocks, emissions, *tag_scores)

        # Second pass: a stoppage that began with the ball going out is restarted by the team
        # that did not touch it last. Add that evidence where play resumes and decode again.
        phases = tag_scores[0]
        restarts = out_of_play_restarts(blocks, path, blocks_per_second)
        span = round(RESTART_EVIDENCE_S * blocks_per_second)
        for first_live, team in restarts:
            for i in range(first_live, min(len(blocks), first_live + span)):
                phases[i][team] = max(phases[i][team], LAST_TOUCH_WEIGHT)
        if restarts:
            emissions = state_evidence(blocks, *tag_scores, blocks_per_second)
            path = decode_states(blocks, emissions, *tag_scores)

        states = refine_boundaries(path, half, pitch)
        labels[first:last + 1] = [LABELS[state] for state in states]
    return labels


def write_submission(path, labels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("frame", "label"))
            writer.writerows(enumerate(labels))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="match bundle root (default: MTAH_DATA_DIR or data/)")
    parser.add_argument("--output", type=Path, required=True, help="destination frame,label CSV")
    args = parser.parse_args()
    try:
        labels = predict(args.data_dir)
        write_submission(args.output, labels)
    except (OSError, ValueError, TypeError, ET.ParseError) as error:
        print(error, file=sys.stderr)
        return 1
    counts = ", ".join(f"{label}={labels.count(label)}" for label in "HADX")
    print(f"Wrote {len(labels):,} rows to {args.output} ({counts})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
