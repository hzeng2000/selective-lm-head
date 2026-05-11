#!/usr/bin/env python
"""Run end-to-end selective LM head constrained decoding."""

from run_baseline import main


if __name__ == "__main__":
    main(default_head="selective", default_record_allowed_ids=False)

