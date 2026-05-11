#!/usr/bin/env python
"""Run controlled full-head trace collection with allowed-id recording."""

from run_baseline import main


if __name__ == "__main__":
    main(default_head="full", default_record_allowed_ids=True)

