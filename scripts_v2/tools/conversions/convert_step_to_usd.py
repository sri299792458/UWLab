# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a CAD STEP/STP file to USD using Isaac Sim's HOOPS converter.

Example:

    CONDA_PREFIX=/data/kanth042/envs/uwlab-isaac51 \
        ./uwlab.sh -p scripts_v2/tools/conversions/convert_step_to_usd.py \
        --input /data/kanth042/downloads/VentionAssembly_511882_v57.STEP \
        --output /data/kanth042/converted_assets/lab_vention_preview/lab_vention_raw.usd
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Convert STEP/STP CAD assets to USD with the HOOPS converter.")
parser.add_argument("--input", required=True, help="Input STEP/STP file.")
parser.add_argument("--output", required=True, help="Output USD file.")
parser.add_argument("--tess-lod", type=int, default=1, choices=range(0, 5), help="HOOPS tessellation LOD, 0-4.")
parser.add_argument("--hidden", action="store_true", help="Convert hidden CAD elements too.")
parser.add_argument("--metadata", action="store_true", help="Convert CAD metadata to USD attributes.")
parser.add_argument("--curves", action="store_true", help="Convert CAD curves to USD BasisCurves.")
parser.add_argument("--no-instancing", action="store_true", help="Disable USD instancing/references.")
parser.add_argument("--no-materials", action="store_true", help="Do not author USD materials.")
parser.add_argument("--no-normals", action="store_true", help="Do not author mesh normals.")
parser.add_argument("--no-dedup", action="store_true", help="Do not weld/deduplicate mesh vertices.")
parser.add_argument(
    "--optimize",
    action="store_true",
    help="Run the converter's default scene optimization after conversion. Off by default for raw inspection.",
)
parser.add_argument(
    "--meters-per-unit",
    type=float,
    default=None,
    help="Optional stage metersPerUnit override after conversion.",
)
parser.add_argument(
    "--up-axis",
    choices=("default", "Y", "Z"),
    default="default",
    help="Optional stage up-axis override after conversion.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402


def _bool_arg(value: bool) -> str:
    return str(bool(value)).lower()


def _build_file_format_args() -> dict[str, str]:
    file_format_args = {
        "tessLOD": str(args_cli.tess_lod),
        "bInstancing": _bool_arg(not args_cli.no_instancing),
        "instancing": _bool_arg(not args_cli.no_instancing),
        "bConvertHidden": _bool_arg(args_cli.hidden),
        "convertHidden": _bool_arg(args_cli.hidden),
        "bConvertMetadata": _bool_arg(args_cli.metadata),
        "convertMetadata": _bool_arg(args_cli.metadata),
        "bConvertCurves": _bool_arg(args_cli.curves),
        "convertCurves": _bool_arg(args_cli.curves),
        "bOptimize": _bool_arg(args_cli.optimize),
        "materialType": "0" if args_cli.no_materials else "1",
        "useMaterials": _bool_arg(not args_cli.no_materials),
        "useNormals": _bool_arg(not args_cli.no_normals),
        "dedup": _bool_arg(not args_cli.no_dedup),
    }
    if args_cli.up_axis != "default":
        file_format_args["iUpAxis"] = {"Y": "1", "Z": "2"}[args_cli.up_axis]
    if args_cli.meters_per_unit is not None:
        file_format_args["dMetersPerUnit"] = str(args_cli.meters_per_unit)
    return file_format_args


def _tick_app(count: int = 10):
    for _ in range(count):
        simulation_app.update()


async def _convert():
    from omni.kit.converter.hoops_core import get_instance  # noqa: PLC0415

    converter = get_instance()
    if converter is None:
        raise RuntimeError("omni.kit.converter.hoops_core did not start correctly.")

    input_path = Path(args_cli.input).expanduser().resolve()
    output_path = Path(args_cli.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_format_args = _build_file_format_args()
    print(f"[INFO]: Input STEP: {input_path}", flush=True)
    print(f"[INFO]: Output USD: {output_path}", flush=True)
    print(f"[INFO]: HOOPS args: {file_format_args}", flush=True)

    output_url, status = await converter.create_converter_task(
        input_path.as_posix(), output_path.as_posix(), file_format_args
    )
    print(f"[INFO]: Converter output URL: {output_url}", flush=True)
    print(f"[INFO]: Converter status: code={status.error_code} message={status.error_msg}", flush=True)

    if status.error_code != 0:
        raise RuntimeError(status.error_msg)
    if not output_path.exists():
        raise RuntimeError(f"Converter reported success but output file is missing: {output_path}")

    print(f"[INFO]: Wrote {output_path} ({output_path.stat().st_size} bytes)", flush=True)


def main():
    enable_extension("omni.services.convert.cad")
    enable_extension("omni.kit.converter.hoops_core")
    enable_extension("omni.scene.optimizer.core")
    _tick_app()
    asyncio.get_event_loop().run_until_complete(_convert())


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
