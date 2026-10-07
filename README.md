# MAPF-TU with Safe-Interval Reduction and k-Safe Policies

This repository contains the implementation and experimental material accompanying an anonymous submission on **Multi-Agent Path Finding with Time Uncertainty (MAPF-TU)**.

The repository includes implementations of the proposed safe-interval reduction and propagation methods, their bounded-suboptimal variants, the MAPF-TU baselines used in the paper, benchmark maps, and a simple script for generating random MAPF-TU instances and comparing the different methods.

## 1. Problem Description

Multi-Agent Path Finding (MAPF) considers a set of agents moving on a graph. Each agent has a start vertex and a goal vertex, and the objective is to find collision-free paths for all agents.

In **MAPF with Time Uncertainty (MAPF-TU)**, traversing an edge does not have a single deterministic duration. Instead, each edge is associated with an interval

\[
[\omega^-(e),\omega^+(e)],
\]

where \(\omega^-(e)\) and \(\omega^+(e)\) respectively represent the minimum and maximum traversal times.

Consequently, the exact arrival time of an agent at a vertex is not known during planning. A solution must therefore reason over possible arrival-time intervals rather than only over deterministic timesteps.

The methods implemented in this repository investigate how temporal conflicts can be resolved by restricting and propagating these arrival-time intervals instead of systematically introducing new CBS constraints and replanning agents.

The repository also contains the \(k\)-safe extension considered in the paper. Rather than requiring every possible temporal realization to be conflict-free, a \(k\)-safe policy may tolerate specific residual temporal conflicts when the estimated probability of safe execution remains above a user-defined threshold \(k\).

---

## 2. Implemented Methods

The testing code provides the following methods.

### CBS_TU

`CBS_TU` is the Conflict-Based Search baseline adapted to temporal uncertainty.

CBS maintains a Constraint Tree (CT). When a temporal conflict is detected between two agents, the conflict is resolved by branching and introducing constraints for the conflicting agents.

This implementation is based on the existing CBS framework for MAPF-TU.

### CBS_P

`CBS_P` extends CBS_TU with the **safe-interval reduction and propagation** mechanism proposed in the accompanying paper.

When a temporal conflict is detected, CBS_P first attempts to eliminate it by reducing the admissible arrival-time intervals of the involved agents and propagating these restrictions through their paths.

CBS branching is required only when the conflict cannot be resolved through interval reduction.

With `k_safe = 1.0`, the returned execution policy must satisfy the full safety requirement considered by CBS_P.

### CBS_k

`CBS_k` is the \(k\)-safe extension of CBS_P.

It uses the same search as CBS_P but may terminate when the execution policy reaches the requested safety threshold

\[
K(P) \geq k.
\]

Setting

```text
k = 1.0
```

recovers the fully safe CBS_P configuration.

Values below 1 allow the planner to accept policies satisfying the corresponding \(k\)-safe criterion.

### EECBS_TU

`EECBS_TU` extends Enhanced Conflict-Based Search (EECBS) to MAPF-TU.

Unlike optimal CBS, EECBS uses a FOCAL search to trade bounded solution quality for improved scalability.

The experiments use

```text
w = 1.2
```

for the high-level and low-level focal search.

### EECBS_P

`EECBS_P` combines bounded-suboptimal EECBS search with the safe-interval reduction and propagation mechanism used by CBS_P.

It therefore evaluates the effect of interval reduction in a bounded-suboptimal CBS framework.

### EECBS_k

`EECBS_k` is the \(k\)-safe version of EECBS_P.

As with CBS_k,

```text
k = 1.0
```

corresponds to the fully safe configuration, while lower values permit termination when the required \(k\)-safe threshold is reached.

### SAT

The repository also contains the SAT-based MAPF-TU baseline used in the experiments.

The SAT solver is implemented in **Picat** and uses the makespan (`mks`) formulation. For every generated Python `TimeUncertaintyProblem`, the testing code automatically converts the problem into the input representation expected by the Picat model before executing the SAT solver.

The Picat formulation originates from the previously published MAPF-TU policy-based solver and is included only to reproduce the experimental baseline.

---

## 3. Repository Structure

The relevant files are organized approximately as follows:

```text
.
├── maps/
│   └── benchmark/
│       ├── empty.map
│       ├── random.map
│       └── warehouse.map
│
├── pathfinding/
│   ├── planners/
│   │   ├── cbstu.py
│   │   ├── constraint_A_star.py
│   │   ├── picat_sat_runner.py
│   │   │
│   │   └── utils/
│   │       ├── constraint_node.py
│   │       ├── custom_heap.py
│   │       ├── maze.py
│   │       ├── time_error.py
│   │       ├── time_uncertainty_plan.py
│   │       ├── time_uncertainty_solution.py
│   │       ├── tu_problem.py
│   │       │
│   │       └── picat_files/
│   │           ├── picat
│   │           ├── mks.pi
│   │           └── aux.pi
│   │
│   └── testing/
│       └── test.py
│
└── README.md
```

The exact directory names should be preserved when running the supplied testing script because the Picat runner locates the SAT files relative to the repository structure.

---

## 4. Main Files

### `cbstu.py`

Contains the high-level CBS-based algorithms used in the experiments, including:

- CBS_TU;
- CBS_P;
- CBS_k;
- EECBS_TU;
- EECBS_P;
- EECBS_k;
- temporal conflict detection;
- Constraint Tree management;
- safe-interval reduction and propagation;
- \(k\)-safe policy evaluation.

### `constraint_A_star.py`

Contains the low-level path planner used by the CBS-based methods.

It computes individual agent paths while respecting the constraints generated by the high-level search.

### `tu_problem.py`

Represents a MAPF-TU instance.

It is responsible for:

- loading MovingAI maps;
- constructing the graph;
- generating uncertain edge traversal durations;
- generating start and goal positions;
- computing heuristic information;
- generating the representation required by the Picat SAT solver.

### `time_uncertainty_plan.py`

Represents an individual path under temporal uncertainty.

### `time_uncertainty_solution.py`

Represents a complete multi-agent solution and stores information such as solution cost, execution policy, computation time, and search statistics.

### `picat_sat_runner.py`

Provides the interface between the Python MAPF-TU implementation and the Picat SAT solver.

For a generated `TimeUncertaintyProblem`, it:

1. creates a temporary Picat representation of the instance;
2. invokes the Picat executable;
3. runs the `mks.pi` formulation;
4. applies the same wall-clock timeout used for the Python methods;
5. reads the solution cost returned by Picat;
6. reports the result to the Python testing script.

### `test.py`

Provides a simple entry point for testing all methods on the **same randomly generated MAPF-TU instances**.

For every random instance, the script executes:

```text
CBS_TU
SAT
CBS_P
CBS_k
EECBS_TU
EECBS_P
EECBS_k
```

This file is intended as the simplest way to verify the implementation and reproduce small-scale comparisons.

---

# 5. Requirements

The recommended environment is **Linux**.

The code requires:

- Python 3;
- `pip`;
- NetworkX;
- Picat for the SAT baseline.

A recent Python 3 version is recommended.

On Ubuntu/Debian, the basic Python environment can be installed with:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

Check the installation with:

```bash
python3 --version
pip3 --version
```

---

## 6. Python Environment

Using a virtual environment is recommended.

From the repository root:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Then upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install the Python dependencies:

```bash
pip install networkx
```

If a `requirements.txt` file is provided, the preferred installation is instead:

```bash
pip install -r requirements.txt
```

The MAPF-TU problem representation directly uses NetworkX, so `networkx` must be available in the Python environment.

You can verify the installation with:

```bash
python -c "import networkx; print(networkx.__version__)"
```

---

# 7. Picat SAT Solver

The SAT baseline is implemented using **Picat**.

The repository contains the files required by the experimental wrapper under:

```text
pathfinding/planners/utils/picat_files/
```

The directory should contain at least:

```text
picat
mks.pi
aux.pi
```

where:

- `picat` is the Picat executable;
- `mks.pi` contains the makespan formulation used by the SAT baseline;
- `aux.pi` contains auxiliary predicates used by the Picat encoding.

The testing script uses the `mks` formulation automatically. No Picat path or objective needs to be specified on the command line.

## 7.1 Giving Picat execution permission on Linux

After cloning or extracting the repository, the executable permission of `picat` may not be preserved.

From the repository root, run:

```bash
chmod +x pathfinding/planners/utils/picat_files/picat
```

You can verify the permission using:

```bash
ls -l pathfinding/planners/utils/picat_files/picat
```

The output should contain the executable flag (`x`), for example:

```text
-rwxr-xr-x ... picat
```

## 7.2 Testing Picat

Move to the Picat directory:

```bash
cd pathfinding/planners/utils/picat_files
```

Then test the executable:

```bash
./picat
```

If Picat starts correctly, the executable is available.

Return to the repository root before running the Python experiments:

```bash
cd ../../../..
```

Alternatively, from the repository root you can verify that the executable starts using:

```bash
./pathfinding/planners/utils/picat_files/picat
```

## 7.3 Common Linux Picat error

If the following error appears:

```text
Permission denied
```

run:

```bash
chmod +x pathfinding/planners/utils/picat_files/picat
```

and retry.

If the executable cannot be launched even after the permission is set, check its architecture using:

```bash
file pathfinding/planners/utils/picat_files/picat
```

The supplied executable must be compatible with the operating system and CPU architecture on which the experiments are executed.

---

# 8. Running a Test

All commands should preferably be executed from the **root of the repository**.

The testing module can be invoked as:

```bash
python3 -m pathfinding.testing.test \
    --map ./maps/benchmark/empty.map \
    --instances 1 \
    --agents 3 \
    --uncertainty 1 \
    --k 0.9 \
    --time-limit 300 \
    --seed 0
```

This command:

- uses the `empty` benchmark map;
- generates 1 random MAPF-TU instance;
- creates 3 agents;
- uses uncertainty parameter \(U=1\);
- uses \(k=0.9\) for CBS_k and EECBS_k;
- limits each method to 300 seconds;
- uses random seed 0.

The same generated instance is used for all methods so that their results are directly comparable.

---

# 9. Command-Line Parameters

The main parameters accepted by the testing script are:

```text
--map
```

Path to the MovingAI map used for the experiment.

```text
--instances
```

Number of independent random MAPF-TU instances to generate.

```text
--agents
```

Number of agents in each instance.

```text
--uncertainty
```

Maximum uncertainty parameter used when generating uncertain traversal durations.

```text
--k
```

Required safety threshold for CBS_k and EECBS_k.

The value must satisfy:

```text
0 <= k <= 1
```

`k = 1.0` corresponds to the fully safe configuration.

```text
--time-limit
```

Maximum computation time in seconds for each solver and instance.

```text
--seed
```

Initial random seed.

For multiple instances, the testing script derives the seed of each instance from this initial value so that experiments can be repeated.

```text
--output
```

Optional path of the CSV output file.

---

# 10. Example with Multiple Instances

For example, to generate 30 instances containing 20 agents with \(U=3\) and \(k=0.7\):

```bash
python3 -m pathfinding.testing.test \
    --map ./maps/benchmark/random.map \
    --instances 30 \
    --agents 20 \
    --uncertainty 3 \
    --k 0.7 \
    --time-limit 300 \
    --seed 0 \
    --output results.csv
```

---

# 11. Output

The testing script writes the results to a CSV file.

The output contains one row per method and instance, with fields corresponding to:

```text
name
method
solved
success_rate
cost
time
nb_iter
nb_iter_edge_conflicts
```

### `name`

Identifier of the generated instance.

### `method`

Algorithm used to solve the instance.

### `solved`

Indicates whether a solution was found within the time limit.

### `success_rate`

Safety value associated with the returned execution policy.

For the fully safe methods, this is expected to correspond to full safety when a solution is successfully returned.

### `cost`

Solution cost reported by the method. The experiments use the worst-case/makespan criterion when comparing the methods.

### `time`

Wall-clock solution time in seconds.

### `nb_iter`

Number of high-level search iterations/expanded Constraint Tree nodes for CBS-based methods.

This metric is not applicable to the Picat SAT baseline.

### `nb_iter_edge_conflicts`

Additional diagnostic information concerning CT iterations involving edge conflicts.

This metric is not applicable to the Picat SAT baseline.

---

# 12. Reproducing the Experimental Setting

The experiments reported in the accompanying paper use MovingAI benchmark maps representing environments with different levels of spatial constraint.

MAPF-TU instances are generated by assigning uncertain traversal-time intervals to graph edges and randomly generating agent start and goal positions.

The main experimental parameters include:

```text
U ∈ {1, 3}
```

for temporal uncertainty and different numbers of agents depending on the experiment.

For the bounded-suboptimal methods, the EECBS focal weight is fixed to:

```text
w = 1.2
```

The supplied testing script uses the same value.

For the \(k\)-safe experiments, the value passed through `--k` determines the requested safety threshold.

For example:

```bash
--k 0.5
```

tests a 0.5 safety threshold, whereas:

```bash
--k 1.0
```

runs the corresponding fully safe configuration.

---

# 13. Reproducibility Notes

For a fair comparison, all methods executed for a given instance use the same:

- map;
- uncertain edge traversal durations;
- agent start positions;
- agent goal positions;
- time limit.

The random seed is exposed through the command line so that generated instances can be reproduced.

For example:

```bash
--seed 42
```

will reproduce the same sequence of randomly generated instances when using the same code and environment.

Runtime measurements can naturally vary depending on hardware, operating system, Python version, and system load.

---

# 14. Quick Installation Summary for Ubuntu/Linux

From a fresh Linux environment:

```bash
# Clone/extract the anonymous supplementary repository and enter it
cd <repository-directory>

# Install Python support
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Create an isolated environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
python -m pip install --upgrade pip
pip install networkx

# Give the bundled Picat binary execution permission
chmod +x pathfinding/planners/utils/picat_files/picat

# Optional: verify Picat
./pathfinding/planners/utils/picat_files/picat

# Run a small experiment
python3 -m pathfinding.testing.test \
    --map ./maps/benchmark/empty.map \
    --instances 1 \
    --agents 3 \
    --uncertainty 1 \
    --k 0.9 \
    --time-limit 300 \
    --seed 0
```

The final command generates one MAPF-TU instance and evaluates all methods included in the testing script.

---

# 15. Troubleshooting

## `ModuleNotFoundError: No module named 'pathfinding'`

Run the testing script from the **repository root** as a Python module:

```bash
python3 -m pathfinding.testing.test ...
```

rather than entering `pathfinding/testing/` and executing:

```bash
python3 test.py
```

## `Permission denied` when executing Picat

Run:

```bash
chmod +x pathfinding/planners/utils/picat_files/picat
```

## `Missing Picat SAT files`

Verify that the Picat directory contains:

```text
picat
mks.pi
aux.pi
```

and that the files are located in the directory expected by `picat_sat_runner.py`.

## SAT works on Linux but not on Windows

The bundled `picat` executable is intended for a Unix/Linux environment. A Linux executable cannot be executed directly from standard Windows PowerShell.

For reproducibility, Linux is therefore recommended for experiments involving the SAT baseline. Windows users may alternatively use a compatible Picat distribution or a Linux environment such as WSL.

## Python methods work but SAT fails

First verify Picat independently:

```bash
./pathfinding/planners/utils/picat_files/picat
```

Then verify the files:

```bash
ls -l pathfinding/planners/utils/picat_files/
```

The directory should contain the executable and both Picat source files required by the SAT formulation.

---

# 16. External Components and Acknowledgements

Parts of this repository build upon previously published MAPF and MAPF-TU implementations.

In particular, the CBS_TU framework is based on the previously proposed conformant CBS approach for MAPF with temporal uncertainty.

The Picat SAT formulation is based on the previously published policy-based MAPF-TU solver. It is included to allow reproduction of the SAT baseline used in the experimental evaluation.

These baseline components are retained for experimental reproducibility and comparison with the methods introduced in the accompanying anonymous submission.

The safe-interval reduction and propagation methods, their integration into CBS/EECBS, and the \(k\)-safe extensions correspond to the methods evaluated in the accompanying submission.

---

# 17. Anonymity

This repository is provided as supplementary material for anonymous peer review.

Author names, affiliations, personal repository links, and other identifying information have intentionally been omitted. Identifying information and complete attribution for the new implementation will be added to