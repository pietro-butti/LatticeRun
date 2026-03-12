## =============================================================================
## run.jl
##
## Command-line entry point for LatticeRun simulations.
##
## Usage:
##   julia run.jl --input input.toml [overrides...]
##
## Any option passed on the command line overrides the corresponding value
## in the TOML file.  If an override changes a value that was already set
## in the file, a WARNING is printed to stderr before proceeding.
##
## Structural parameters (L, Lx, beta, c0) additionally print a louder
## WARNING to make accidental physics changes harder to miss.
##
## Unrecognised arguments cause an error and exit.
##
## Examples:
##   julia run.jl --input B6.2.toml
##   julia run.jl --input B6.2.toml --saveto /scratch/B6.2 --ntraj 2000
##   julia run.jl --input B6.2.toml --device 1 --ens-name B6.2_run2 --logfile run2.log
##   julia run.jl --input B6.2.toml --start-from /data/B6.2.cfg_n500 --ntherm 0
##   julia run.jl --input B6.2.toml --L 16 16 16 32 --Lx 8 8 8 8 --beta 6.4
##   julia run.jl --input B6.2.toml --L 16 --Lx 4   # broadcasts to all 4 dirs
## =============================================================================

using ArgParse
using LatticeRun
using TimerOutputs


# ------------------------------------------------------------------------------
# CLI definition
# ------------------------------------------------------------------------------

function build_arg_table()
    s = ArgParseSettings(
        description     = "LatticeRun -- SU(3) pure-gauge lattice QCD simulation",
        exit_after_help = true,
    )

    @add_arg_table! s begin

        "--input"
            help     = "Path to TOML input file (required)"
            arg_type = String
            required = true

        # --- [run] ------------------------------------------------------------
        "--device"
            help     = "CUDA device index (overrides [run] device)"
            arg_type = Int
            default  = nothing

        "--ens-name"
            help     = "Ensemble name (overrides [run] ens_name)"
            arg_type = String
            default  = nothing

        # --- [geometry]  STRUCTURAL -------------------------------------------
        "--L"
            help     = "Full lattice extents: 1 or 4 integers " *
                       "(e.g. --L 16  or  --L 16 16 16 32) " *
                       "[STRUCTURAL: overrides [geometry] L]"
            arg_type = Int
            nargs    = '+'
            default  = nothing

        "--Lx"
            help     = "Sub-lattice extents: 1 or 4 integers " *
                       "(e.g. --Lx 4  or  --Lx 4 4 4 8) " *
                       "[STRUCTURAL: overrides [geometry] Lx]"
            arg_type = Int
            nargs    = '+'
            default  = nothing

        # --- [action]  STRUCTURAL ---------------------------------------------
        "--beta"
            help     = "Inverse coupling beta (overrides [action] beta) [STRUCTURAL]"
            arg_type = Float64
            default  = nothing

        "--c0"
            help     = "Symanzik coefficient c0 (overrides [action] c0) [STRUCTURAL]"
            arg_type = Float64
            default  = nothing

        # --- [io] -------------------------------------------------------------
        "--saveto"
            help     = "Directory to save configurations (overrides [io] save_to)"
            arg_type = String
            default  = nothing

        "--save-each"
            help     = "Save configuration every N trajectories (overrides [io] save_each)"
            arg_type = Int
            default  = nothing

        "--save-final"
            help     = "Save the final configuration: true/false (overrides [io] save_final)"
            arg_type = String        # String so we can detect whether it was explicitly provided
            default  = nothing

        "--logfile"
            help     = "Log file path; empty string means stdout only (overrides [io] logfile)"
            arg_type = String
            default  = nothing

        "--start-from"
            help     = "Starting config: empty/cold, hot, or a file path (overrides [hmc] start_from)"
            arg_type = String
            default  = nothing

        # --- [hmc] ------------------------------------------------------------
        "--ntherm"
            help     = "Number of thermalization trajectories (overrides [hmc] ntherm)"
            arg_type = Int
            default  = nothing

        "--ntraj"
            help     = "Number of production trajectories (overrides [hmc] ntraj)"
            arg_type = Int
            default  = nothing

        "--delta"
            help     = "HMC leapfrog step size (overrides [hmc] delta)"
            arg_type = Float64
            default  = nothing

        "--nleaps"
            help     = "Number of leapfrog steps per trajectory (overrides [hmc] nleaps)"
            arg_type = Int
            default  = nothing

        # --- [flow] -----------------------------------------------------------
        "--flow-each"
            help     = "Measure flow every N trajectories (overrides [flow] flow_each)"
            arg_type = Int
            default  = nothing

        "--epsilon"
            help     = "Flow step size (overrides [flow] epsilon)"
            arg_type = Float64
            default  = nothing

        "--nflow"
            help     = "Number of flow steps (overrides [flow] nflow)"
            arg_type = Int
            default  = nothing

        "--flow-file"
            help     = "Path for CSV flow output (overrides [flow] flow_file)"
            arg_type = String
            default  = nothing

    end

    return s
end


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

# Parameters that change the physics of the ensemble — warrant a louder warning.
const _STRUCTURAL_FIELDS = Set(["L", "Lx", "beta", "c0"])

# Maps normalised ArgParse key (hyphens -> underscores) -> SimParams Dict key.
# Only entries where the two names differ need to appear here.
const _CLI_REMAP = Dict(
    "saveto"     => "save_to",
    "ens_name"   => "ens_name",
    "start_from" => "start_from",
    "flow_each"  => "flow_each",
    "flow_file"  => "flow_file",
    "save_each"  => "save_each",
    "save_final" => "save_final",
)

"""
    _expand_vol(v) -> Vector{Int}

Normalise a volume value arriving from the CLI as a Vector{Int} (via nargs='+').
  - Single integer broadcasts: [16]          -> [16, 16, 16, 16]
  - Four integers used as-is:  [16, 16, 16, 32] -> [16, 16, 16, 32]
"""
function _expand_vol(v::Vector{Int})::Vector{Int}
    length(v) == 1 && return fill(v[1], 4)
    length(v) == 4 && return v
    error("--L/--Lx must be 1 or 4 integers (got $(length(v)))")
end


# ------------------------------------------------------------------------------
# Apply CLI overrides with conflict warnings
# ------------------------------------------------------------------------------

"""
    apply_overrides!(p_dict, args) -> nothing

For each CLI argument that was explicitly provided (not nothing), compare
it against the value already in p_dict.  If they differ:
  - structural parameters (L, Lx, beta, c0) print a prominent WARNING
  - all other parameters print a regular warning
Then update p_dict with the CLI value.
"""
function apply_overrides!(p_dict::Dict{String, Any}, args::Dict)
    for (raw_key, cli_val) in args
        raw_key == "input" && continue   # not a SimParams field
        isnothing(cli_val) && continue              # not provided on CLI
        cli_val isa Vector && isempty(cli_val) && continue   # nargs='+' default

        # L and Lx keep their original capitalisation as Dict keys.
        # Everything else: hyphens -> underscores, then remap if needed.
        norm_key  = raw_key in ("L", "Lx") ? raw_key : replace(raw_key, "-" => "_")
        field_key = get(_CLI_REMAP, norm_key, norm_key)

        # Special parsing for vector fields and bool
        parsed_val = if field_key in ("L", "Lx")
            _expand_vol(cli_val)   # cli_val is already Vector{Int} via nargs='+'
        elseif field_key == "save_final"
            lv = lowercase(string(cli_val))
            if lv in ("true", "yes", "1")
                true
            elseif lv in ("false", "no", "0")
                false
            else
                @warn "--save-final: unrecognised value \"$cli_val\" (expected true/false); ignoring"
                continue
            end
        else
            cli_val
        end

        toml_val = get(p_dict, field_key, nothing)

        if !isnothing(toml_val) && toml_val != parsed_val
            if field_key in _STRUCTURAL_FIELDS
                @warn "*** STRUCTURAL OVERRIDE *** '$field_key' changes the physics of the ensemble!\n" *
                      "      TOML : $(repr(toml_val))\n" *
                      "      CLI  : $(repr(parsed_val))\n" *
                      "    Make sure ens_name reflects the new ensemble."
            else
                @warn "CLI overrides TOML for '$field_key': " *
                      "$(repr(toml_val)) (TOML)  ->  $(repr(parsed_val)) (CLI)"
            end
        end

        p_dict[field_key] = parsed_val
    end
end


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

function main()
    args = parse_args(build_arg_table())

    # 1. Load TOML into a mutable Dict (single source of truth in Parameters.jl)
    p_dict = load_toml_dict(args["input"])

    # 2. Patch with CLI overrides (warns on conflicts)
    apply_overrides!(p_dict, args)

    # 3. Build SimParams and validate
    p = SimParams(p_dict)
    validate(p)

    # 4. Logger
    lg = SimLogger(p)
    log_banner!(lg, p)

    # 5. Run
    state    = setup_simulation(p, lg)
    schedule = MCSchedule([(HMCUpdate(), 1)])
    meas     = isnothing(p.flow_type) ? Measurement[] : Measurement[FlowMeasurement()]

    thermalize!(state, schedule, p, lg)
    run!(state, schedule, meas, p, lg)

    log_line(lg, sprint(print_timer))

    close_logger(lg)
end

main()