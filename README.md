# LatticeRun.jl

A structured wrapper around [`LatticeGPU.jl`](https://github.com/pietro-butti/LatticeGPU.jl) for running SU(3) pure-gauge lattice QCD simulations with periodic boundary conditions on NVIDIA GPUs.

LatticeRun handles everything *around* the physics: parameter management, structured logging, composable MC update scheduling, on-the-fly measurements, and configuration I/O — so you can stay focused on implementing new algorithms and observables.

---

## Table of contents

1. [Installation](#installation)
2. [Quickstart](#quickstart)
3. [Input file reference](#input-file-reference)
4. [Repository layout](#repository-layout)
5. [How the package works](#how-the-package-works)
   - [Parameters](#parameters-parametersjl)
   - [Logger](#logger-loggerjl)
   - [Runner](#runner-runnerjl)
6. [Log format and output routing](#log-format-and-output-routing)
7. [Extension guide](#extension-guide)

---

## Installation

LatticeRun depends on [LatticeGPU.jl](https://github.com/pietro-butti/LatticeGPU.jl), which must be added manually before installing this package.

```julia
] add https://github.com/pietro-butti/LatticeGPU.jl
] add https://github.com/pietro-butti/LatticeRun
```

To work on the source directly:

```bash
git clone https://github.com/pietro-butti/LatticeRun
cd LatticeRun
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

A CUDA-capable GPU and a working CUDA installation are required at runtime.

**Dependencies** (resolved automatically via `Project.toml`):
`LatticeGPU`, `BDIO`, `CSV`, `CUDA`, `Dates`, `Printf`, `TOML`

---

## Quickstart

### Minimal run script

```julia
using LatticeRun

# Load and validate parameters from a TOML file
p = SimParams("input.toml")
validate(p)

# Build a logger — stdout only if p.logfile is nothing, otherwise to file
lg = SimLogger(p)
log_banner!(lg, p)

# Initialise the simulation state (gauge field, HMC integrator, flow kernels)
state = setup_simulation(p, lg)

# One HMC trajectory per MC step
schedule = MCSchedule([(HMCUpdate(), 1)])

# Measurements to run every p.flow_each trajectories
meas = Measurement[FlowMeasurement()]

thermalize!(state, schedule, p, lg)
run!(state, schedule, meas, p, lg)

close_logger(lg)
```

### Redirect flow data to a dedicated file

By default all output goes to the main log. To send `TAG_FLOW` data to a
separate file while keeping the main log readable:

```julia
lg = SimLogger(p; redirects = Dict(TAG_FLOW => "$(p.ens_name)_flow.dat"))
```

The main log then receives a one-line stub per configuration instead of the raw data:

```
[ FLOW           ][ CONF 000205 ] - output redirected → B6.2_L16_flow.dat
```

### Store flow data in a CSV file

Set `flow_file` in the `[flow]` section of your TOML. Each configuration's flow
rows are appended to that CSV in a single write — nothing accumulates in memory
across configurations:

```toml
[flow]
flow_file = "B6.2_L16_flow.csv"
```

### Run without measurements

```julia
run!(state, schedule, Measurement[], p, lg)
```

### Suppress flow output entirely

```julia
lg = SimLogger(p; redirects = Dict(TAG_FLOW => "/dev/null"))
```

---

## Input file reference

Below is a fully commented input file. Every section except `[run]`,
`[geometry]`, and `[action]` is optional — omit a section entirely to
disable that feature.

```toml
# ──────────────────────────────────────────────────────────────────────────────
# [run]  — required
# ──────────────────────────────────────────────────────────────────────────────
[run]
device   = 0           # CUDA device index (0-based)
ens_name = "B6.2_L16"  # ensemble label; used in all output filenames


# ──────────────────────────────────────────────────────────────────────────────
# [geometry]  — required
# ──────────────────────────────────────────────────────────────────────────────
[geometry]
L  = [16, 16, 16, 16]  # full lattice extents; a scalar broadcasts (e.g. L = 16)
Lx = [8,  8,  8,  8]   # sub-lattice extents for domain decomposition; must divide L


# ──────────────────────────────────────────────────────────────────────────────
# [action]  — required
# ──────────────────────────────────────────────────────────────────────────────
[action]
beta = 6.2   # inverse coupling β
c0   = 1.0   # Symanzik coefficient (1.0 = Wilson plaquette action)


# ──────────────────────────────────────────────────────────────────────────────
# [hmc]  — optional; all four fields (ntherm/ntraj/delta/nleaps) must be
#          present together or all absent
# ──────────────────────────────────────────────────────────────────────────────
[hmc]
ntherm     = 200   # thermalization trajectories (no measurements performed)
ntraj      = 1000  # production trajectories
delta      = 0.1   # leapfrog step size δ
nleaps     = 10    # number of leapfrog steps per trajectory

# Where to start:
#   leave empty or "cold" → cold start (unit matrices)
#   "hot"                 → random start
#   "/path/to/file"       → load configuration from file
start_from = ""


# ──────────────────────────────────────────────────────────────────────────────
# [io]  — optional
# ──────────────────────────────────────────────────────────────────────────────
[io]
save_to    = "./"   # directory for saved configurations; required if save_each
                    # or save_final is set
save_final = true   # always save the last configuration
# save_each = 10    # also save every N trajectories (comment out to disable)
logfile    = ""     # path to log file; leave empty for stdout only


# ──────────────────────────────────────────────────────────────────────────────
# [flow]  — optional; flow_each/flow_type/epsilon/nflow must all be set together
# ──────────────────────────────────────────────────────────────────────────────
[flow]
flow_each = 5                     # measure every N production trajectories

# Kernels to apply: "wilson", "zeuthen", or both.
# Both kernels are run independently on the same unflowed configuration.
flow_type = ["wilson", "zeuthen"]

adaptive  = false   # true → adaptive step (requires Tflow); false → fixed step
# Tflow   = 2.0     # maximum flow time; only used when adaptive = true
epsilon   = 0.01    # flow step size ε
nflow     = 100     # number of flow steps (total flow time = nflow × epsilon)

# Optional CSV output. When set, each configuration's rows are appended to
# this file in a single write. When absent, flow data goes through the logger
# (main log or a redirect file set via SimLogger redirects).
# flow_file = "B6.2_L16_flow.csv"
```

### Parameter reference table

Fields typed `Union{T, Nothing}` are optional; `nothing` disables the feature.
Fields in the same group must be set together — `validate` will catch partial configurations.

| Section | Field | Type | Notes |
|---|---|---|---|
| `[run]` | `device` | `Int` | CUDA device index |
| | `ens_name` | `String` | Used in all output filenames |
| `[geometry]` | `L` | `NTuple{4,Int}` | Full lattice; scalar broadcasts to all 4 dirs |
| | `Lx` | `NTuple{4,Int}` | Sub-lattice; must divide `L` element-wise |
| `[action]` | `beta` | `Float64` | Inverse coupling β |
| | `c0` | `Float64` | Symanzik coefficient |
| `[hmc]` | `ntherm` | `Int?` | All four HMC fields required together |
| | `ntraj` | `Int?` | |
| | `delta` | `Float64?` | |
| | `nleaps` | `Int?` | |
| `[io]` | `start_from` | `String?` | `"cold"`, `"hot"`, or filepath |
| | `save_each` | `Int?` | `nothing` = never save periodically |
| | `save_final` | `Bool` | |
| | `save_to` | `String?` | Required if any saving is enabled |
| | `logfile` | `String?` | `nothing` = stdout only |
| `[flow]` | `flow_each` | `Int?` | All four flow fields required together |
| | `flow_type` | `Vector{String}?` | `"wilson"`, `"zeuthen"`, or both |
| | `epsilon` | `Float64?` | |
| | `nflow` | `Int?` | |
| | `adaptive` | `Bool?` | Requires `Tflow` if `true` |
| | `Tflow` | `Float64?` | Max flow time for adaptive mode |
| | `flow_file` | `String?` | CSV output path; `nothing` = use logger |

---

## Repository layout

```
LatticeRun/
├── Project.toml
├── main/
│   ├── simple_run.jl        # example run script
│   └── simple_run.toml      # example input file
└── src/
    ├── LatticeRun.jl              # package entry point and exports
    ├── Parameters.jl              # SimParams: constructors, validation, banner
    ├── Logger.jl                  # SimLogger: structured, multiplexed logging
    └── Runner/
        ├── Runner.jl              # SimState, setup, thermalize!, run!
        ├── RunnerUpdates.jl       # MCAlgorithm, HMCUpdate, MCSchedule, mc_sweep!
        └── RunnerMeasurements.jl  # Measurement, FlowMeasurement, run_measurements!
```

---

## How the package works

### Parameters (`Parameters.jl`)

All simulation parameters live in a single immutable struct `SimParams`. It is
constructed from a TOML file or a `Dict`, validated once, and then passed
read-only to every function in the run:

```julia
p = SimParams("input.toml")
validate(p)                    # throws ArgumentError on any inconsistency
```

`parameter_banner(p)` returns a formatted multi-line `String` summarising all
parameters, printed automatically by `log_banner!` at startup.

Fields that are not required for a given run are typed `Union{T, Nothing}`.
`validate` enforces that related fields are either all set or all unset.

---

### Logger (`Logger.jl`)

`SimLogger` is a thin multiplexer. Every message is written to any combination
of `stdout` and an on-disk log file simultaneously. Specific tags can be
**redirected** to dedicated data files — the main log then receives a one-line
stub instead of the raw data, keeping it readable.

```julia
lg = SimLogger(p)   # stdout if p.logfile is nothing, file otherwise

# per-tag redirect:
lg = SimLogger(p; redirects = Dict(TAG_FLOW => "$(p.ens_name)_flow.dat"))
```

Every log line follows a consistent format:

```
[ TAG            ][ CONF 000042 ] - <message>   # configuration-stamped
[ TAG            ] - <message>                  # run-level
```

**Logging functions:**

| Function | Use |
|---|---|
| `log_banner!(lg, p)` | Startup header (timestamp, host, device) + parameter banner |
| `log_conf(lg, tag, itraj, fmt, args...)` | Tagged, configuration-stamped line |
| `log_tag(lg, tag, fmt, args...)` | Tagged run-level line (no conf number) |
| `log_line(lg, msg)` | Raw line to main log |
| `flush_logger(lg)` | Flush all sinks |
| `close_logger(lg)` | Flush and close all open file handles |

Predefined tag constants: `TAG_INIT`, `TAG_THERM`, `TAG_HMC`, `TAG_FLOW`, `TAG_IO`.
Any plain string is also accepted as a tag.

---

### Runner (`Runner.jl`)

The runner is split into three layers.

**`SimState`** — mutable container for all GPU runtime objects:

```
SimState
├── U            # SU(3) gauge field (on GPU)
├── lp           # SpaceParm (lattice geometry)
├── gp           # GaugeParm (action parameters)
├── ymws         # YMworkspace (GPU workspace)
├── intsch       # HMC integrator schedule
├── flow_kernels # list of (label => kernel) pairs, built from p.flow_type
├── U_cpu        # CPU mirror of U for flow snapshots
└── itraj        # global trajectory counter
```

**`MCAlgorithm` / `MCSchedule`** (`RunnerUpdates.jl`) — the update layer. An
`MCSchedule` is an ordered list of `(algorithm, n_repeats)` pairs. One call to
`mc_sweep!` executes all of them in order and increments `itraj` by one.
Currently implemented: `HMCUpdate`.

**`Measurement`** (`RunnerMeasurements.jl`) — the measurement layer. Each
concrete subtype implements `run_measurement!` and emits output through the
logger. Currently implemented: `FlowMeasurement` (gradient flow observables for
every kernel in `p.flow_type`).

The top-level flow is:

```
setup_simulation(p, lg)  →  SimState
thermalize!(state, ...)  →  p.ntherm sweeps, no measurements
run!(state, ...)         →  p.ntraj sweeps + measurements + saves
```

---

## Log format and output routing

The main log looks like this:

```
# ════════════════════════════════════════════════════════════════════════
#  Run started  :  2026-03-05 14:22:01
#  Host         :  mynode.cluster
#  Device       :  GPU 0
# ════════════════════════════════════════════════════════════════════════
# ...full parameter table...

[ INIT           ] - CUDA device 0 selected
[ INIT           ] - Cold start: gauge field set to unit configuration
[ INIT           ] - initial plaquette = 1.0000000000000000e+00
[ THERM          ] - starting thermalization (200 trajectories)
[ HMC            ][ CONF 000001 ] - ΔH = +2.341e-05  [acc 1]  Plaq = 0.5503819347
[ HMC            ][ CONF 000002 ] - ΔH = -1.203e-04  [acc 1]  Plaq = 0.5491822013
...
[ THERM          ] - thermalization complete  (itraj = 200)
[ HMC            ] - starting production (1000 trajectories)
[ HMC            ][ CONF 000201 ] - ΔH = +4.112e-05  [acc 1]  Plaq = 0.5512034871
[ FLOW           ][ CONF 000205 ] - output redirected → B6.2_L16_flow.dat
[ IO             ][ CONF 000210 ] - configuration saved → ./B6.2_L16.cfg_n210
...
[ HMC            ] - production complete  (itraj = 1200)
```

When flow output goes to a data file (via redirect or `flow_file`), one line is
written per flow step per kernel. Column order: `kernel`, `t`, `Eplq`,
`t²Eplq`, `Eclv`, `t²Eclv`, `qtop`, `qrec`.

Both the main log and all redirect/data files are opened in **append mode**, so
a restarted run extends the existing files rather than overwriting them.

---

## Extension guide

### Adding a new measurement

All measurements live in `src/Runner/RunnerMeasurements.jl`.

**1. Define a struct** subtyping `Measurement`:

```julia
struct PolyakovMeasurement <: Measurement end
```

**2. Implement `run_measurement!`**:

```julia
function run_measurement!(::PolyakovMeasurement, state::SimState,
                           p::SimParams, lg::SimLogger)
    ploop = polyakov_loop(state.U, state.lp, state.gp, state.ymws)
    log_conf(lg, "POLY", state.itraj,
        "Re(P) = %.10f  Im(P) = %.10f", real(ploop), imag(ploop))
end
```

Then add it to your measurement list and optionally register a redirect:

```julia
meas = Measurement[FlowMeasurement(), PolyakovMeasurement()]

lg = SimLogger(p; redirects = Dict(
    TAG_FLOW => "$(p.ens_name)_flow.dat",
    "POLY"   => "$(p.ens_name)_poly.dat",
))
```

No other files need to change.

---

### Adding a new MC algorithm

All update algorithms live in `src/Runner/RunnerUpdates.jl`.

**1. Define a struct** subtyping `MCAlgorithm`:

```julia
struct ORUpdate <: MCAlgorithm
    n_hits :: Int
end
```

**2. Implement `run_update!`**. The method must **not** increment `state.itraj`
— that is done by `mc_sweep!` after all algorithms in the schedule have run:

```julia
function run_update!(alg::ORUpdate, state::SimState, p::SimParams, lg::SimLogger)
    for _ in 1:alg.n_hits
        OR!(state.U, state.lp, state.gp, state.ymws)
    end
    plq = plaquette(state.U, state.lp, state.gp, state.ymws)
    log_conf(lg, "OR", state.itraj, "Plaq = %.10f", plq)
end
```

Then include it in your schedule:

```julia
schedule = MCSchedule([
    (HMCUpdate(),  1),
    (ORUpdate(3),  3),
])
```

---

### Adding a new parameter

**1.** Add the field to `SimParams` in `src/Parameters.jl`.  
**2.** Wire it in both constructors (`SimParams(d::Dict)` and `SimParams(toml_path::String)`).  
**3.** Add a validation rule in `validate` if needed.  
**4.** Add it to `parameter_banner`.

---

## License

LatticeRun.jl is released under the [GNU General Public License v3.0](LICENSE).