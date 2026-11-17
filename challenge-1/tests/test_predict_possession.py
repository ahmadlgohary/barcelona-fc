import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from games import GAME, half_video_blocks, source_frames
from src.main import (AWAY, DEAD, HOME, Block, Observation, Tag, decode_states, out_of_play_restarts,
                      read_tracking, refine_boundaries, state_evidence, tag_evidence)


ROOT = Path(__file__).resolve().parents[1]
FIRST, SECOND = [first for _, first, _ in half_video_blocks(GAME)]
FRAME_COUNT = source_frames()
HOME_PLAYERS = ";".join(["0.50,0.50,3"] + ["0.40,0.50,0.4"] * 3)
AWAY_PLAYERS = ";".join(["0.53,0.50,3"] + ["0.60,0.50,0.4"] * 3)
PLAYERS = f"{HOME_PLAYERS};{AWAY_PLAYERS}"


def instance(start, end, code, team, kind=""):
    return (
        f"<instance><start>{start / 25}</start><end>{end / 25}</end><code>{code}</code>"
        f"<label><group>Team</group><text>{team}</text></label>"
        f"<label><group>Type</group><text>{kind}</text></label></instance>"
    )


def make_bundle(root):
    tactical = root / "tactical"
    tactical.mkdir(parents=True)
    stem = tactical / "BAR-ATH_tactical"
    Path(f"{stem}_atd.xml").write_text(
        '<Metadata><Team id="B"><Name>Barcelona</Name></Team>'
        '<Team id="A"><Name>Athletic Club</Name></Team>'
        + '<Player teamId="B"/>' * 4 + '<Player teamId="A"/>' * 4 + '</Metadata>',
        encoding="utf-8",
    )
    Path(f"{stem}_smart_tagging.xml").write_text(
        "<root>" + instance(FIRST, FIRST + 70, "BUILD UP", "Barcelona")
        + instance(FIRST + 75, FIRST + 145, "PROGRESSION", "Atletic Club")
        + instance(FIRST + 194, FIRST + 226, "SET PIECES", "N/A", "Throw-in")
        + instance(SECOND + 55, SECOND + 95, "SET PIECES", "Atletic Club", "Kick Off (Start)")
        + "</root>",
        encoding="utf-8",
    )

    empty = ";".join(["NaN,NaN,NaN"] * 8)
    throw_in_players = PLAYERS.replace("0.50,0.50,3", "0.50,0.015,3", 1)
    second_half_players = PLAYERS.replace("0.53,0.50,3", "0.50,0.50,3").replace(
        "0.50,0.50,3", "0.47,0.50,3", 1
    )

    with Path(f"{stem}_atd.txt").open("w", encoding="utf-8") as stream:
        for frame in range(FRAME_COUNT):
            players, ball = empty, "NaN,NaN,NaN"
            first = frame - FIRST
            second = frame - SECOND
            if 0 <= first < 145:
                players = PLAYERS
                if 25 <= first < 45:
                    ball = "NaN,NaN,NaN"
                elif 50 <= first < 55 or first >= 75:
                    ball = "0.53,0.50,4"
                else:
                    ball = "0.50,0.50,4"
            elif 145 <= first < 155:
                players, ball = PLAYERS, "0.50,-0.012,5"
            elif 155 <= first < 210:
                players = PLAYERS.replace(",3", ",0.3")
            elif 210 <= first < 340:
                players, ball = throw_in_players, "0.50,0.018,5"
            elif 0 <= second < 75:
                players, ball = second_half_players.replace(",3", ",0.3"), "0.50,0.50,0"
            elif 75 <= second < 250:
                players, ball = second_half_players, "0.50,0.50,5"
            stream.write(f"{frame + 1}:{players}:{ball}\n")


class PossessionTest(unittest.TestCase):
    def test_conflicting_tags_do_not_advance_handover_during_occlusion(self):
        home_control = Block((52.0, 34.0), 4.0, (1.0, 0.0), 0.4, 0.0)
        hidden_ball = Block(None, float("nan"), (0.0, 0.0), float("nan"), 0.0)
        blocks = [home_control] * 7 + [hidden_ball] * 51
        phases = [(0.0, 0.0)] * 7 + [(0.2, 0.2)] + [(0.0, 0.1)] * 50
        restarts = [0.0] * len(blocks)
        emissions = state_evidence(blocks, phases, restarts, restarts, 5)
        path = decode_states(blocks, emissions, phases, restarts, restarts)
        self.assertEqual(path, [HOME] * 8 + [AWAY] * 50)

    def test_untagged_field_wide_stoppage_is_dead_ball(self):
        # A foul with no Smart Tagging set piece: the ball vanishes and everyone walks for 12 s.
        playing = Block((52.0, 34.0), 4.0, (1.0, 0.0), 0.5, 0.0)
        stopped = Block(None, float("nan"), (0.0, 0.0), 0.0, 0.0)
        blocks = [playing] * 25 + [stopped] * 60 + [playing] * 25
        none = [(0.0, 0.0)] * len(blocks)
        zeros = [0.0] * len(blocks)
        path = decode_states(blocks, state_evidence(blocks, none, zeros, zeros, 5), none, zeros, zeros)
        self.assertEqual(path[:25], [HOME] * 25)
        self.assertEqual(path[40:70], [DEAD] * 30)
        self.assertEqual(path[-10:], [HOME] * 10)

    def test_slow_circulation_with_a_moving_ball_stays_live(self):
        # Everybody jogs, but the ball is being passed around: open play, not a stoppage.
        circulating = Block((30.0, 34.0), 6.0, (0.6, 0.0), 0.02, 0.0)
        blocks = [circulating] * 80
        none = [(0.0, 0.0)] * len(blocks)
        zeros = [0.0] * len(blocks)
        path = decode_states(blocks, state_evidence(blocks, none, zeros, zeros, 5), none, zeros, zeros)
        self.assertEqual(path, [HOME] * 80)

    def test_ball_out_is_restarted_by_the_team_that_did_not_touch_it_last(self):
        away_control = Block((100.0, 34.0), 4.0, (0.0, 1.0), 0.5, 0.0)
        out = Block((106.0, 34.0), 5.0, (0.0, 0.0), 0.3, 1.0)
        stopped = Block(None, float("nan"), (0.0, 0.0), 0.0, 0.0)
        blocks = [away_control] * 10 + [out] * 5 + [stopped] * 30 + [away_control] * 20
        path = [AWAY] * 10 + [DEAD] * 35 + [HOME] * 20
        self.assertEqual(out_of_play_restarts(blocks, path, 5), [(45, HOME)])

    def test_goal_is_dead_until_the_conceding_team_kicks_off(self):
        centre = (105.3 / 2, 34.0)
        still = Block(centre, 0.0, (0.0, 0.0), 0.0, 0.0)
        moving = Block(centre, 5.0, (0.5, 0.5), 0.5, 0.0)
        blocks = [still] * 60 + [moving] * 40
        tags = [Tag(50, 100, "goals", "", HOME),
                Tag(275, 325, "set pieces", "kick off (after goal)", None)]
        phases, _, restart_dead = tag_evidence(blocks, 0, tags, (105.3, 68.0), 5)
        self.assertTrue(all(value >= 1.5 for value in restart_dead[20:55]))
        first_away = next(i for i, phase in enumerate(phases) if phase[AWAY] > 0.0)
        self.assertTrue(55 <= first_away <= 65)
        self.assertTrue(all(phase[HOME] == 0.0 for phase in phases[55:]))

    def test_boundaries_move_to_the_frame_where_control_changes(self):
        home = Observation((50.0, 34.0, 4.0), (1.0, 0.0), 0.5)
        away = Observation((50.0, 34.0, 4.0), (0.0, 1.0), 0.5)
        tracking = [home] * 13 + [away] * 17
        states = refine_boundaries([HOME] * 3 + [AWAY] * 3, tracking, (105.3, 68.0))
        self.assertEqual(states, [HOME] * 13 + [AWAY] * 17)

    def test_outside_ball_cannot_supply_player_control(self):
        with tempfile.TemporaryDirectory() as directory:
            tracking = Path(directory) / "tracking.txt"
            tracking.write_text(
                "1:0.50,-0.01,1;0.60,0.10,1:0.50,-0.01,1\n"
                "2:0.50,0.01,1;0.60,0.10,1:0.50,0.01,1\n",
                encoding="utf-8",
            )
            outside, inside = read_tracking(tracking, [HOME, AWAY], (105.3, 68.0), 2)
        self.assertIsNotNone(outside.ball)
        self.assertEqual(outside.contact, (0.0, 0.0))
        self.assertEqual(inside.contact, (1.0, 0.0))

    def test_full_length_reproduction_and_possession_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "data"
            make_bundle(bundle)
            first_csv, second_csv = root / "a.csv", root / "b.csv"
            command = [sys.executable, str(ROOT / "src" / "main.py")]
            subprocess.run(command + ["--data-dir", str(bundle), "--output", str(first_csv)],
                           check=True, capture_output=True, text=True)
            environment = dict(os.environ, MTAH_DATA_DIR=str(bundle))
            subprocess.run(command + ["--output", str(second_csv)], check=True,
                           capture_output=True, text=True, env=environment)
            self.assertEqual(first_csv.read_bytes(), second_csv.read_bytes())

            with first_csv.open(encoding="utf-8", newline="") as stream:
                reader = csv.reader(stream)
                self.assertEqual(next(reader), ["frame", "label"])
                labels = []
                for index, (frame, label) in enumerate(reader):
                    self.assertEqual(frame, str(index))
                    self.assertIn(label, "HADX")
                    labels.append(label)
            self.assertEqual(len(labels), 199319)
            for _, start, end in half_video_blocks(GAME):
                self.assertNotIn("X", labels[start:end + 1])
                self.assertEqual(labels[start - 1], "X")
                self.assertEqual(labels[end + 1], "X")
            self.assertEqual(labels[FIRST + 15:FIRST + 65], ["H"] * 50)
            self.assertEqual(labels[FIRST + 90:FIRST + 140], ["A"] * 50)
            self.assertEqual(labels[FIRST + 165:FIRST + 200], ["D"] * 35)
            self.assertEqual(labels[FIRST + 225:FIRST + 300], ["H"] * 75)
            self.assertEqual(labels[SECOND + 15:SECOND + 60], ["D"] * 45)
            self.assertEqual(labels[SECOND + 90:SECOND + 140], ["A"] * 50)

    def test_rejects_displaced_and_truncated_tracking(self):
        with tempfile.TemporaryDirectory() as directory:
            tracking = Path(directory) / "tracking.txt"
            tracking.write_text("1:0.5,0.5,1:0.5,0.5,1\n3:0.5,0.5,1:0.5,0.5,1\n",
                                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected tracking frame ID 2"):
                read_tracking(tracking, [0], (105.3, 68.0), 2)
            with self.assertRaisesRegex(ValueError, "expected 2 tactical records, found 1"):
                tracking.write_text("1:0.5,0.5,1:0.5,0.5,1\n", encoding="utf-8")
                read_tracking(tracking, [0], (105.3, 68.0), 2)


if __name__ == "__main__":
    unittest.main()
