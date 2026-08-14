"""Allow ``python -m data_collection.wuji``."""

from .teleop import parse_args, run

run(parse_args())
