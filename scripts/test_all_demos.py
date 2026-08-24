# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Run all training demos with both JAX and PyTorch backends.

This script iterates through all available demo environments and runs
training for each one using both training backends.
"""

import argparse
import subprocess
import sys
from typing import List

# List of all available demo environments
ALL_DEMOS = [
    "cartpole",
    "dm-cheetah",
    "dm-hopper-stand",
    "dm-hopper-hop",
    "dm-reacher",
    "dm-stander",
    "dm-walker",
    "dm-runner",
    "bounce_ball",
    "go1-flat-terrain-walk",
    "go1-rough-terrain-walk",
    "go1-stairs-terrain-walk",
    "anymal_c_navigation_flat",
    "franka-lift-cube",
    "franka-open-cabinet",
]

# Available training backends
BACKENDS = ["jax", "torch"]


def run_command(env: str, backend: str, extra_args: List[str]) -> int:
    """Run a single training command."""
    cmd = [
        "uv",
        "run",
        "scripts/train.py",
        "--env",
        env,
        "--train-backend",
        backend,
    ] + extra_args

    print(f"\n{'=' * 80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ Error: Training failed for {env} with {backend} backend")
        return result.returncode

    print(f"\n✅ Success: Training completed for {env} with {backend} backend")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run training demos with both JAX and PyTorch backends")
    parser.add_argument(
        "--demos",
        nargs="+",
        choices=ALL_DEMOS + ["all"],
        default=["all"],
        help="List of demos to run (default: all)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=BACKENDS + ["all"],
        default=["all"],
        help="Training backends to use (default: all)",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments to pass to train.py",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop execution if any training fails",
    )

    args = parser.parse_args()

    # Determine which demos to run
    if "all" in args.demos:
        demos_to_run = ALL_DEMOS
    else:
        demos_to_run = args.demos

    # Determine which backends to use
    if "all" in args.backends:
        backends_to_run = BACKENDS
    else:
        backends_to_run = args.backends

    print(f"Will run {len(demos_to_run)} demos with {len(backends_to_run)} backend(s)")
    print(f"Demos: {demos_to_run}")
    print(f"Backends: {backends_to_run}")
    print(f"Total training runs: {len(demos_to_run) * len(backends_to_run)}")

    failed_runs = []

    for demo in demos_to_run:
        for backend in backends_to_run:
            returncode = run_command(demo, backend, args.extra_args)

            if returncode != 0:
                failed_runs.append((demo, backend))
                if args.stop_on_error:
                    print("\n⚠️  Stopping due to error as requested")
                    sys.exit(1)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    total_runs = len(demos_to_run) * len(backends_to_run)
    successful_runs = total_runs - len(failed_runs)

    print(f"Total runs: {total_runs}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {len(failed_runs)}")

    if failed_runs:
        print("\nFailed runs:")
        for demo, backend in failed_runs:
            print(f"  - {demo} ({backend})")
        sys.exit(1)
    else:
        print("\n✅ All training runs completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
