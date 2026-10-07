"""Wrapper for the Picat SAT encoding from svancaj/MAPF-TU.

The wrapper converts the current TimeUncertaintyProblem into the Picat input
format using TimeUncertaintyProblem.print_picat_instance(), runs Picat under a
wall-clock timeout, and returns a small result object compatible with the CSV
fields used by experimentation_ajdin.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import re
import subprocess
import tempfile
import time
from typing import Optional


_COST_RE = re.compile(r"^\s*Cost\s+(-?\d+(?:\.\d+)?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class PicatSATResult:
    is_solved: bool
    success_rate: int
    cost: float
    time_to_solve: float
    iteration: None = None
    iteration_with_edge_conflicts: None = None
    timed_out: bool = False
    stdout: str = ""
    stderr: str = ""


class PicatSATPlanner:
    """Run the MAPF-TU Picat SAT encoding on an existing TU instance."""

    def __init__(
        self,
        problem,
        picat_dir: str | os.PathLike[str],
        objective: str = "mks",
        keep_instances: bool = False,
        instance_dir: Optional[str | os.PathLike[str]] = None,
    ) -> None:
        self.problem = problem
        self.picat_dir = Path(picat_dir).expanduser().resolve()
        self.objective = objective.lower()
        self.keep_instances = keep_instances
        self.instance_dir = Path(instance_dir).expanduser().resolve() if instance_dir else None

        if self.objective not in {"mks", "soc"}:
            raise ValueError("objective must be either 'mks' or 'soc'")

        self.picat_executable = self.picat_dir / "picat"
        self.encoding_file = self.picat_dir / f"{self.objective}.pi"
        self.aux_file = self.picat_dir / "_aux.pi"

        missing = [
            str(path)
            for path in (self.picat_executable, self.encoding_file, self.aux_file)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing Picat SAT files: " + ", ".join(missing)
            )

    @staticmethod
    def _parse_cost(output: str) -> Optional[float]:
        match = _COST_RE.search(output)
        if match is None:
            return None
        value = float(match.group(1))
        return int(value) if value.is_integer() else value

    def _create_instance_path(self, instance_name: str) -> tuple[Path, bool]:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", instance_name)

        if self.keep_instances:
            target_dir = self.instance_dir or (self.picat_dir / "generated_instances")
            target_dir.mkdir(parents=True, exist_ok=True)
            return target_dir / f"{safe_name}.pi", False

        fd, temporary_path = tempfile.mkstemp(prefix=f"{safe_name}_", suffix=".pi")
        os.close(fd)
        return Path(temporary_path), True

    def find_solution(self, time_lim: float, instance_name: str = "instance") -> PicatSATResult:
        """Generate a Picat instance and solve it within ``time_lim`` seconds."""
        if time_lim <= 0:
            return PicatSATResult(False, 0, math.inf, 0.0, timed_out=True)

        start = time.monotonic()
        instance_path, delete_after = self._create_instance_path(instance_name)

        try:
            self.problem.print_picat_instance(str(instance_path))
            elapsed = time.monotonic() - start
            remaining = max(0.0, float(time_lim) - elapsed)
            if remaining <= 0:
                return PicatSATResult(False, 0, math.inf, elapsed, timed_out=True)

            completed = subprocess.run(
                [str(self.picat_executable), self.encoding_file.name, str(instance_path)],
                cwd=str(self.picat_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=remaining,
                check=False,
            )
            total_time = min(time.monotonic() - start, float(time_lim))
            cost = self._parse_cost(completed.stdout)
            solved = completed.returncode == 0 and cost is not None

            return PicatSATResult(
                is_solved=solved,
                success_rate=1 if solved else 0,
                cost=cost if solved else math.inf,
                time_to_solve=total_time,
                timed_out=False,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            return PicatSATResult(
                is_solved=False,
                success_rate=0,
                cost=math.inf,
                time_to_solve=float(time_lim),
                timed_out=True,
                stdout=stdout,
                stderr=stderr,
            )
        except (OSError, ValueError) as exc:
            total_time = min(time.monotonic() - start, float(time_lim))
            return PicatSATResult(
                is_solved=False,
                success_rate=0,
                cost=math.inf,
                time_to_solve=total_time,
                timed_out=False,
                stderr=str(exc),
            )
        finally:
            if delete_after:
                try:
                    instance_path.unlink(missing_ok=True)
                except OSError:
                    pass