# Intentionally empty.
#
# Its presence puts the repo root on sys.path under pytest's default import
# mode, so `from foolwatch import ...` resolves whether tests run as `pytest`
# (as CI does) or `python -m pytest` (as is easy to do locally). Without it
# the two invocations disagree and only CI fails.
