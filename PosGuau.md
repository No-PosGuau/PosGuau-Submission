PosGuau: More than a Hack 2027, Phase 1 (Ball Possession)
Approach

A rule-based, fully algorithmic pipeline (no learning, no training data) built on the provided tactical-camera tracking. The idea is to treat possession as a state with hysteresis rather than a per-frame vote, because the ranking metric (spellF1) rewards recovering whole possessions.

Ball gap filling. The ball is missing on a large share of frames. Gaps of up to 50 frames (2 s) are linearly interpolated; longer gaps stay missing.
Control. For each frame, a team "controls" the ball if its nearest tracked player is within 4 m of the (interpolated) ball. Positions are scaled from normalised coordinates to metres using the 105.3 x 68 m pitch.
Hysteresis. Short control dropouts (up to 12 frames) inside one team's control are bridged. Control only counts once held for at least 25 frames (1 s). The possession owner changes only when the opposing team has such a confirmed control run, and the change is back-dated to the start of that run. Touches, deflections and loose balls therefore do not change possession, and in loose phases possession stays with the last team that had control.
Dead ball. D is assigned when the ball is outside the pitch for at least 25 frames. Frames outside the two playing periods (before kickoff, halftime, after the whistle) are X.
Abstention. Live-ball frames before any control run has been confirmed are X; the scorer forward-fills these, so it does not split a possession.
Assets used
BAR-ATH_tactical_atd.txt (player and ball tracking) and BAR-ATH_tactical_atd.xml (to map track slots to home/away; the first team in the metadata is home).
games.py for pitch size and the per-half play windows.
Not used: video (tactical and broadcast), homography, Smart Tagging, and the broadcast tracking.
Training data

None. No labelled data of any kind was used, and no hand labels, hand corrections, private ground truth or event data. The 5-minute public ground truth was used only to check that the output was aligned and well formed. The parameter values are defaults chosen from football reasoning (possessions lasting around ten to twenty seconds), not tuned against the public window.

Possession rules implemented
A change of possession needs controlled possession by the opposing team: its nearest player within 4 m of the ball for at least 1 s.
Loose balls, brief touches and deflections do not change possession.
Dead ball is a stoppage, never a possession; no dead-ball ownership is given.
Limitations
Dead ball is weak. Only "ball outside the pitch" is detected, so recall on D is very low (about 7% on the public window). Throw-ins, free kicks and other stoppages inside the pitch are mostly labelled as possession. Smart Tagging set-pieces would be the obvious next cue.
Nearest-player control is crude: it ignores ball speed and height, and pressing situations.
The tracking has identity switches and gaps; control confirmation is a fixed threshold, not adaptive.
Parameters are not tuned; only a 5-minute public window was available to check them.
Reproduction

Requirements: Python 3.8+ and numpy (see requirements.txt). Put the match data in ./data (or set MTAH_DATA_DIR), then:

python possession.py --team PosGuau --min-control 25 --radius 4
python evaluation/validate_submission.py PosGuau.csv

This regenerates the submitted PosGuau.csv. The other parameters use their defaults: --gap-fill 12, --max-ball-gap 50, --min-dead 25.