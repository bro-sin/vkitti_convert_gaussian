import sys
import typing
from typing import Dict
import tyro

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


def v2g_entry_point():
    discovered_entry_points = entry_points(
        group="vkitti_convert_gaussian.console_scripts"
    )
    # print(discovered_entry_points)
    methods: Dict[str, callable] = dict()
    for name in discovered_entry_points.names:
        spec = discovered_entry_points[name].load()
        spec = typing.cast(callable, spec)
        methods[name] = spec
    tyro.extras.set_accent_color("bright_yellow")
    tyro.extras.subcommand_cli_from_dict(subcommands=methods)


if __name__ == "__main__":
    v2g_entry_point()
