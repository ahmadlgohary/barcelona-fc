# Technical note: Viterbi possession labeller (`src/possession_hmm.py`)

## Approach
A rule-based evidence model decoded with Viterbi over three states (H, A, D) per half. No machine
learning and no labels of any kind. Evidence per frame:

* **Control (H vs A):** distance from the ball to the nearest home / away player. Evidence is
  `closeness * tanh((d_away - d_home) / 2 m)`, where closeness decays over ~3 m. It is zeroed when
  the ball is in flight (> ~14 m/s), so a pass travelling past opponents is not a hand-over.
* **Dead ball (D):** ball outside the pitch (+0.5 m margin); ball stationary (< 1 m of extent over
  3.2 s: set-piece set-up); and team-wide player activity (mean tracked-player speed over 3 s,
  walking pace 1.5 m/s as the pivot). The activity cue needs no ball, which matters because the ball
  is missing on 41 % of graded frames: ball lost + players running means play continues unseen,
  ball lost + players walking means a stoppage.
* **Smart Tagging:** phase `Team` tags (Barcelona / Athletic, non set-piece) add a weak (0.1) team
  prior. `N/A` tags and set pieces are ignored.
* **Decoding:** switching costs (H<->A 20, to-dead 15, from-dead 10, in evidence units). The H<->A
  cost is the "controlled possession" rule: the other team needs about a second of clear control
  before possession changes, so touches, deflections and lost duels do not split a possession.
  Loose balls (no evidence) stay with the last possessor, matching the challenge's rules.

## Assets
Used: tactical ATD (players + ball), tactical Smart Tagging (weak prior). Not used: homography,
video, broadcast feed. Dropped idea: a ball-lost -> D rule (it labelled 46 % of the match dead, far
too many; the activity cue replaced it).

## Training data
None. No learned parameters.

## Constants
All constants are in the header of `src/possession_hmm.py` and are football priors (walking pace,
pitch lines, ~1 s of control). The only constant informed by published numbers is the H<->A cost
(20): it was chosen so that the mean live-possession length is about 14 s, the figure in the public
challenge documents, not by maximising any score. The public window was otherwise used only to check
the pipeline (frame alignment, class meaning). Reported public-window result for the submitted
setting: macroF1 0.716, spellF1 0.424, 16 predicted possessions vs 17 (17-possession samples cannot
rank anything, see EVALUATION.md).

## Limitations
* The ball is only seen 59 % of the time; a hand-over during a long gap is detected late.
* Stationary-ball and activity cues are team-wide and indirect; restarts (throw-ins, quick free
  kicks) are the hardest. Smart Tagging restart windows are not exploited yet.
* Dead-ball vs slow build-up play can be confused when everyone walks.
* Next steps: use set-piece windows to time restarts, use the homography/video to see the ball,
  use pass-completion logic (ball reaching a teammate) instead of nearest-player distance.

## Reproduction
```
python -m pip install -r requirements.txt          # numpy, pandas
# unpack the data bundle into data/ (see DATA.md)
python src/possession_hmm.py --output submissions/possession_hmm.csv
python evaluation/validate_submission.py submissions/possession_hmm.csv
```
The first run parses the tracking (about 25 s) and caches it in `data/cache/` (gitignored).
Output is deterministic (identical md5 across runs).
