# PosGuau: More than a Hack 2027, Phase 1

See PosGuau.md for the technical document.

Setup: Python 3.8+ and numpy. Put the match data in ./data (or set MTAH_DATA_DIR), then run:

    python possession.py --team PosGuau --min-control 25 --radius 4
    python evaluation/validate_submission.py PosGuau.csv
