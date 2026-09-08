"""Entry point for the reusable 30-day mortality prediction pipeline.

The complete implementation currently lives in ``isds_option_a_pipeline.py`` for
backward compatibility with earlier repository revisions. This neutral entry point is
the public command-line interface used by the current documentation.
"""
from isds_option_a_pipeline import *  # noqa: F401,F403
from isds_option_a_pipeline import parse_args, run


if __name__ == "__main__":
    args = parse_args()
    run(args.data, args.output)
