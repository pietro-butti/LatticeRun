# LatticeRun.jl

A structured wrapper around [`LatticeGPU.jl`](https://github.com/pietro-butti/LatticeGPU.jl) for running SU(3) pure-gauge lattice QCD simulations with periodic boundary conditions on NVIDIA GPUs.

LatticeRun handles everything *around* the physics: parameter management, structured logging, composable MC update scheduling, on-the-fly measurements, and configuration I/O — so you can stay focused on implementing new algorithms and observables without touching the run infrastructure.

---

## Table of contents

1. [Installation](#installation)
2. [Running a simulation](#running-a-simulation)
3. [Repository layout](#repository-layout)
4. [How the package works](#how-the-package-works)
   - [Parameters](#parameters-parameterjl)
   - [Logger](#logger-loggerjl)
   - [Runner](#runner-runnerjl)
5. [Setting parameters](#setting-parameters)
   - [TOML input file (recommended)](#toml-input-file-recommended)
   - [Programmatic Dict](#programmatic-dict)
   - [Parameter reference](#parameter-reference)
6. [Log format and output routing](#log-format-and-output-routing)
7. [Extension guide](#extension-guide)
   - [Adding a new parameter](#adding-a-new-parameter)
   - [Adding a new measurement](#adding-a-new-measurement)
   - [Adding a new MC algorithm](#adding-a-new-mc-algorithm)

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
`LatticeGPU`, `BDIO`, `CUDA`, `Dates`, `Printf`, `TOML`

---

## Running a simulation

A minimal script using all defaults:

```julia
using LatticeRun

p  = SimParams("input.toml")
validate(p)

lg = SimLogger(p; redirects = Dict(TAG_FLOW => "$(p.ens_name)_flow.dat"))
log_banner!(lg, p)

state    = setup_simulation(p, lg)
schedule = MCSchedule([(HMCUpdate(), 1)])
meas     = Measurement[FlowMeasurement()]

thermalize!(state, schedule, p, lg)
run!(state, schedule, meas, p, lg)

close_logger(lg)
```

Omitting the `TAG_FLOW` redirect causes flow data to go to the main log instead of a separate file:

```julia
lg = SimLogger(p)   # no redirects → all output to main log / stdout
```

To run without any measurements:

```julia
run!(state, schedule, Measurement[], p, lg)
```

---

## Repository layout

```
LatticeRun/
├── Project.toml
└── src/
    ├── LatticeRun.jl            # package entry point and exports
    ├── Parameters.jl            # SimParams: constructors, validation, banner
    ├── Logger.jl                # SimLogger: structured, multiplexed logging
    └── Runner/
        ├── Runner.jl            # SimState, setup, thermalize!, run!
        ├── RunnerUpdates.jl     # MCAlgorithm, HMCUpdate, MCSchedule, mc_sweep!
        └── RunnerMeasurements.jl  # Measurement, FlowMeasurement, run_measurements!
```

| File | Responsibility |
|---|---|
| `Parameters.jl` | Single source of truth for all simulation parameters. Reads TOML or Dict, validates, formats. |
| `Logger.jl` | Multiplexed, tag-based structured logging to stdout and/or file, with optional per-tag data file redirection. |
| `Runner.jl` | Runtime state container, setup, thermalization and production loops. |
| `RunnerUpdates.jl` | MC update algorithms (`HMCUpdate`, …) and the composable `MCSchedule`. |
| `RunnerMeasurements.jl` | On-the-fly observables (`FlowMeasurement`, …) dispatched at configurable intervals. |

---

## How the package works

### Parameters (`Parameters.jl`)

All simulation parameters live in a single immutable struct `SimParams`. It is constructed from a TOML file or a `Dict`, validated once, and then passed read-only to every function in the run:

```julia
p = SimParams("input.toml")   # from file
validate(p)                    # throws ArgumentError on any inconsistency
```

`parameter_banner(p)` returns a formatted multi-line `String` summarising all parameters, suitable for logging or writing directly to a file.

Fields that are not required for a given run (e.g. flow parameters when no flow is requested) are typed `Union{T, Nothing}` and left as `nothing`. `validate` enforces that related fields are either all set or all unset (e.g. you cannot set `flow_each` without also setting `flow_type`, `epsilon`, and `nflow`).

---

### Logger (`Logger.jl`)

`SimLogger` is a thin multiplexer. Every message is written to any combination of `stdout` and an on-disk log file simultaneously. Specific tags can additionally be **redirected** to dedicated data files — the main log then receives a one-line stub instead of the raw data, keeping it readable.

```julia
lg = SimLogger(p)   # stdout if p.logfile is nothing, file otherwise

# with a per-tag redirect for flow data:
lg = SimLogger(p; redirects = Dict(TAG_FLOW => "$(p.ens_name)_flow.dat"))
```

Every log line follows the same format:

```
[ TAG            ][ CONF 000042 ] - <message>   # configuration-stamped
[ TAG            ] - <message>                  # run-level (no conf number)
```

Available logging functions:

| Function | Use |
|---|---|
| `log_banner!(lg, p)` | Write startup header (timestamp, host, device) + full parameter banner |
| `log_conf(lg, tag, itraj, fmt, args...)` | Tagged, configuration-stamped line (Printf format string) |
| `log_tag(lg, tag, fmt, args...)` | Tagged run-level line (no configuration number) |
| `log_line(lg, msg)` | Raw line to main log |
| `flush_logger(lg)` | Flush all sinks |
| `close_logger(lg)` | Flush and close all open file handles |

Predefined tag constants: `TAG_INIT`, `TAG_THERM`, `TAG_HMC`, `TAG_FLOW`, `TAG_IO`. Any plain string is also accepted as a tag.

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
└── itraj        # global trajectory counter (incremented by mc_sweep!)
```

**`MCAlgorithm` / `MCSchedule`** (`RunnerUpdates.jl`) — the update layer. An `MCSchedule` is an ordered list of `(algorithm, n_repeats)` pairs. One call to `mc_sweep!` executes all of them in order and increments `itraj` by one, regardless of how many sub-steps ran. Currently implemented: `HMCUpdate`.

**`Measurement`** (`RunnerMeasurements.jl`) — the measurement layer. Each concrete subtype implements `run_measurement!` and emits its output through the logger. `run_measurements!` iterates the list and calls each one. Currently implemented: `FlowMeasurement` (gradient flow observables for every kernel in `p.flow_type`).

The top-level flow is:

```
setup_simulation(p, lg)      → SimState
thermalize!(state, ...)      → p.ntherm sweeps, no measurements
run!(state, ...)             → p.ntraj sweeps + measurements + saves
```

---

## Setting parameters

### TOML input file (recommended)

Create an `input.toml` file. All sections except `[run]`, `[geometry]`, and `[action]` are optional — omit a section entirely to disable that feature.

```toml
[run]
device   = 0           # CUDA device index
ens_name = "B6.2_L16"  # ensemble identifier, used in all output filenames

[geometry]
L  = [16, 16, 16, 16]  # full lattice extents (or just L = 16 for uniform)
Lx = [8,  8,  8,  8]   # sub-lattice extents for domain decomposition

[action]
beta = 6.2
c0   = 1.0             # Symanzik coefficient (1.0 = Wilson plaquette action)

[hmc]
ntherm     = 200        # thermalization trajectories (no measurements)
ntraj      = 1000       # production trajectories
delta      = 0.02       # integrator step size δ
nleaps     = 20         # leapfrog steps per trajectory
start_from = ""         # "" or "cold" → cold start
                        # "hot"        → random (hot) start
                        # "/path/file" → load configuration from file

[io]
save_each  = 10         # save configuration every N trajectories
save_final = true       # always save the last configuration
save_to    = "./configs" # directory for saved configurations
logfile    = "./run.log" # omit or leave empty to write to stdout only

[flow]
flow_each = 5                       # measure every N trajectories
flow_type = ["wilson", "zeuthen"]   # one or both kernels
adaptive  = false
epsilon   = 0.01                    # flow step size
nflow     = 100                     # number of flow steps
# Tflow   = 2.0                     # required only if adaptive = true
```

Load it with:

```julia
using LatticeRun
p = SimParams("input.toml")
validate(p)
```

### Programmatic Dict

You can also build `SimParams` directly from a `Dict`, which is useful for scripting or parameter sweeps:

```julia
p = SimParams(Dict(
    "device"     => 0,
    "ens_name"   => "test",
    "L"          => [16, 16, 16, 16],
    "Lx"         => [8, 8, 8, 8],
    "beta"       => 6.2,
    "c0"         => 1.0,
    "start_from" => nothing,
    "save_each"  => nothing,    # nothing = disabled
    "save_final" => false,
    "save_to"    => nothing,
    "logfile"    => nothing,
    "ntherm"     => 100,
    "ntraj"      => 500,
    "delta"      => 0.02,
    "nleaps"     => 20,
    "flow_each"  => 5,
    "flow_type"  => ["wilson"],
    "adaptive"   => false,
    "Tflow"      => nothing,
    "epsilon"    => 0.01,
    "nflow"      => 100,
))
```

### Parameter reference

Fields typed `Union{T, Nothing}` are optional. Setting them to `nothing` (or omitting them from the TOML) disables the related feature. Fields in the same group must be set together — `validate` will catch partial configurations.

| Section | Field | Type | Default | Notes |
|---|---|---|---|---|
| `[run]` | `device` | `Int` | — | CUDA device index |
| | `ens_name` | `String` | — | Used in all output filenames |
| `[geometry]` | `L` | `NTuple{4,Int}` | — | Full lattice; scalar broadcasts to all 4 dirs |
| | `Lx` | `NTuple{4,Int}` | — | Sub-lattice; must divide `L` element-wise |
| `[action]` | `beta` | `Float64` | — | Inverse coupling β |
| | `c0` | `Float64` | `1.0` | Symanzik coefficient |
| `[hmc]` | `ntherm` | `Int?` | `nothing` | All four HMC fields required together |
| | `ntraj` | `Int?` | `nothing` | |
| | `delta` | `Float64?` | `nothing` | |
| | `nleaps` | `Int?` | `nothing` | |
| | `start_from` | `String?` | `nothing` | `"cold"`, `"hot"`, or filepath |
| `[io]` | `save_each` | `Int?` | `nothing` | `nothing` = never save periodically |
| | `save_final` | `Bool` | `false` | |
| | `save_to` | `String?` | `nothing` | `nothing` = don't save configurations |
| | `logfile` | `String?` | `nothing` | `nothing` = stdout only |
| `[flow]` | `flow_each` | `Int?` | `nothing` | All four flow fields required together |
| | `flow_type` | `Vector{String}?` | `nothing` | `"wilson"`, `"zeuthen"`, or both |
| | `epsilon` | `Float64?` | `nothing` | |
| | `nflow` | `Int?` | `nothing` | |
| | `adaptive` | `Bool?` | `nothing` | Requires `Tflow` if `true` |
| | `Tflow` | `Float64?` | `nothing` | Max flow time for adaptive mode |

---

## Log format and output routing

The main log (stdout or logfile) looks like this:

```
# ════════════════════════════════════════════════════════════════════════
#  Run started  :  2026-03-05 14:22:01
#  Host         :  mynode.cluster
#  Device       :  GPU 0
# ════════════════════════════════════════════════════════════════════════
# ════════...
#   Ensemble : B6.2_L16
#   ...full parameter table...

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
[ FLOW           ][ CONF 000205 ] - output appended → B6.2_L16_flow.dat
[ IO             ][ CONF 000210 ] - configuration saved → ./configs/B6.2_L16.cfg_n210
...
[ IO             ] - run complete
```

When flow is redirected, the data file `B6.2_L16_flow.dat` receives one line per flow-time step and per kernel:

```
[ FLOW           ][ CONF 000205 ] - kernel=Wilson   t=0.000000  Eplq=1.23456789e-02  t2Eplq=0.00000000e+00  ...
[ FLOW           ][ CONF 000205 ] - kernel=Wilson   t=0.010000  Eplq=1.23512345e-02  t2Eplq=1.23512345e-06  ...
...
```

Both the main log and all redirect files are opened in **append mode**, so a restarted run extends the existing files rather than overwriting them.

---

## Extension guide

### Adding a new parameter

**1. Add the field to `SimParams`** in `src/Parameters.jl`. Use `Union{T, Nothing}` if the parameter is optional:

```julia
struct SimParams
    ...
    my_param :: Union{Float64, Nothing}
end
```

**2. Wire it up in the Dict constructor:**

```julia
function SimParams(p::Dict)
    SimParams(
        ...
        p["my_param"],
    )
end
```

**3. Wire it up in the TOML constructor:**

```julia
"my_param" => _opt(mysection, "my_param", nothing),
```

**4. Add a validation rule** in `validate` if needed:

```julia
!isnothing(p.my_param) && p.my_param > 0 ||
    throw(ArgumentError("my_param must be positive"))
```

**5. Add it to `parameter_banner`** in the appropriate section:

```julia
p!("#   My param         :  $(_fmt(p.my_param))")
```

---

### Adding a new measurement

All measurements live in `src/Runner/RunnerMeasurements.jl`. Adding one requires two things only:

**1. Define a struct** that subtypes `Measurement`:

```julia
struct PolyakovMeasurement <: Measurement end
```

**2. Implement `run_measurement!`** for it. Use `log_conf` with whatever tag you choose — if you want the data in a separate file, register that tag as a redirect when building the logger:

```julia
function run_measurement!(::PolyakovMeasurement, state::SimState,
                           p::SimParams, lg::SimLogger)
    ploop = polyakov_loop(state.U, state.lp, state.gp, state.ymws)
    log_conf(lg, "POLY", state.itraj,
        "Re(P) = %.10f  Im(P) = %.10f", real(ploop), imag(ploop))
end
```

Then add it to your measurement list at the call site:

```julia
meas = Measurement[FlowMeasurement(), PolyakovMeasurement()]

# optionally redirect its output to a file:
lg = SimLogger(p; redirects = Dict(
    TAG_FLOW => "$(p.ens_name)_flow.dat",
    "POLY"   => "$(p.ens_name)_poly.dat",
))
```

No other files need to change.

---

### Adding a new MC algorithm

All update algorithms live in `src/Runner/RunnerUpdates.jl`. Adding one requires two things only:

**1. Define a struct** that subtypes `MCAlgorithm`. Put any algorithm-specific parameters (that are not in `SimParams`) as struct fields:

```julia
struct ORUpdate <: MCAlgorithm
    n_hits :: Int
end
```

**2. Implement `run_update!`** for it. The method must not increment `state.itraj` — that is handled by `mc_sweep!` after all algorithms in the schedule have run:

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
    (HMCUpdate(),  1),   # 1 HMC step
    (ORUpdate(3),  3),   # 3 OR sweeps, each with 3 hits
])
```

One call to `mc_sweep!` executes the full list in order and increments `itraj` by one — one schedule execution = one entry in the Markov chain.

---

## License

LatticeRun.jl is released under the [GNU General Public License v3.0](LICENSE).