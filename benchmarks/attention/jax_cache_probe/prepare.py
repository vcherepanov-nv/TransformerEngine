# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Build the native FFI counter and stage the plugin and recorded common test list."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess


def main():
    """Prepare a standalone probe directory using the active JAX installation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from jax import ffi  # pylint: disable=import-outside-toplevel

    source = Path(__file__).resolve().parent
    recorded = source / "results" / "2026-11-13"
    metadata = json.loads((recorded / "run-metadata.json").read_text())
    selection = gzip.decompress((recorded / "common-tests.args.gz").read_bytes())
    if hashlib.sha256(selection).hexdigest() != metadata["selection_sha256"]:
        raise ValueError("Recorded common test selection does not match its SHA256.")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-O2",
            f"-I{ffi.include_dir()}",
            str(source / "ffi_probe.cpp"),
            "-o",
            str(output / "ffi_probe.so"),
        ],
        check=True,
    )
    if output != source:
        shutil.copyfile(source / "ffi_probe_plugin.py", output / "ffi_probe_plugin.py")
    (output / "common-tests.args").write_bytes(selection)
    print(f"Prepared the native probe, pytest plugin, and common test list in {output}")


if __name__ == "__main__":
    main()
