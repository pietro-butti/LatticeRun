## =============================================================================
## Parameters.jl
##
## Defines SimParams — the single, immutable source of truth for all simulation
## parameters.  Provides constructors from a TOML file and from a parsed Dict,
## plus validation and a pretty-printer.
##
## Public API:
##   load_toml_dict(path)    → Dict{String,Any}   (mutable; safe to patch)
##   SimParams(d::Dict)      → SimParams
##   SimParams(path::String) → SimParams           (convenience: load + construct)
##   validate(p)             → nothing / throws
##   parameter_banner(p)     → String
## =============================================================================
module Parameters
    using TOML

    # --------------------------------------------------------------------------
    # Struct
    # --------------------------------------------------------------------------

    """
        SimParams

    Immutable container for all simulation parameters.
    Constructed either from a TOML file (`SimParams(path::String)`) or from a
    flat `Dict` (`SimParams(d::Dict)`).  Use `load_toml_dict` to obtain a
    mutable Dict from a TOML file when you need to patch values before
    constructing (e.g. CLI overrides in run.jl).
    """
    struct SimParams
        # --- Run specifics ---
        device   :: Union{Int, Nothing}   # nothing → use whatever CUDA default is
        ens_name :: String

        # --- geometry ---
        L  :: NTuple{4, Int}   # full lattice extents
        Lx :: NTuple{4, Int}   # sub-lattice extents

        # --- action ---
        beta :: Float64
        c0   :: Float64

        # --- I/O ---
        start_from :: Union{String, Nothing}   # nothing → cold start
        save_each  :: Union{Int, Nothing}
        save_final :: Bool
        save_to    :: Union{String, Nothing}   # nothing → don't save configs
        logfile    :: Union{String, Nothing}   # nothing → stdout only

        # --- HMC ---
        ntherm :: Union{Int,     Nothing}
        ntraj  :: Union{Int,     Nothing}
        delta  :: Union{Float64, Nothing}
        nleaps :: Union{Int,     Nothing}

        # --- gradient flow ---
        flow_each :: Union{Int,            Nothing}
        flow_type :: Union{Vector{String}, Nothing}   # nothing → no flow
        adaptive  :: Union{Bool,           Nothing}
        Tflow     :: Union{Float64,        Nothing}
        epsilon   :: Union{Float64,        Nothing}
        nflow     :: Union{Int,            Nothing}
        flow_file :: Union{String,         Nothing} # nothing → use log/redirect; path → write CSV
    end


    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------

    function _to_vol(raw, name::String)::NTuple{4, Int}
        if raw isa Integer
            return ntuple(_ -> Int(raw), 4)
        elseif raw isa Union{AbstractVector, Tuple}
            length(raw) == 4 ||
                throw(ArgumentError("$name must have exactly 4 entries (got $(length(raw)))"))
            return NTuple{4, Int}(raw)
        else
            throw(ArgumentError("$name: unexpected type $(typeof(raw))"))
        end
    end

    _opt(d, k, def)        = let v = get(d, k, def); (isnothing(v) || v == "") ? def : v end
    _opt_string(d, k, def) = let v = get(d, k, def); (isnothing(v) || v == "") ? nothing : String(v) end


    # --------------------------------------------------------------------------
    # load_toml_dict  — public, mutable, patchable
    # --------------------------------------------------------------------------

    """
        load_toml_dict(toml_path::String) -> Dict{String, Any}

    Parse a TOML input file and return a flat, mutable `Dict{String,Any}` in
    the form expected by `SimParams(::Dict)`.

    This is the right entry point when you need to apply overrides before
    constructing `SimParams` (e.g. from a CLI script):

        d = load_toml_dict("input.toml")
        d["ntraj"] = 2000          # override
        p = SimParams(d)
        validate(p)

    For straightforward usage without overrides, `SimParams("input.toml")` is
    equivalent and more concise.
    """
    function load_toml_dict(toml_path::String)::Dict{String, Any}
        raw = TOML.parsefile(toml_path)

        run_ = get(raw, "run",      Dict())
        geo  = get(raw, "geometry", Dict())
        act  = get(raw, "action",   Dict())
        io   = get(raw, "io",       Dict())
        hmc  = get(raw, "hmc",      Dict())
        flw  = get(raw, "flow",     Dict())

        return Dict{String, Any}(
            "device"     => _opt(run_, "device", nothing),
            "ens_name"   => run_["ens_name"],
            "L"          => geo["L"],
            "Lx"         => geo["Lx"],
            "beta"       => act["beta"],
            "c0"         => act["c0"],
            "start_from" => _opt_string(io, "start_from", nothing),
            "save_each"  => _opt(io,  "save_each",  nothing),
            "save_final" => _opt(io,  "save_final", false),
            "logfile"    => _opt_string(io, "logfile",  nothing),
            "save_to"    => _opt_string(io, "save_to",  nothing),
            "ntherm"     => _opt(hmc, "ntherm", nothing),
            "ntraj"      => _opt(hmc, "ntraj",  nothing),
            "delta"      => _opt(hmc, "delta",  nothing),
            "nleaps"     => _opt(hmc, "nleaps", nothing),
            "flow_type"  => _opt(flw, "flow_type", nothing),
            "flow_each"  => _opt(flw, "flow_each", nothing),
            "adaptive"   => _opt(flw, "adaptive",  nothing),
            "Tflow"      => _opt(flw, "Tflow",     nothing),
            "epsilon"    => _opt(flw, "epsilon",   nothing),
            "nflow"      => _opt(flw, "nflow",     nothing),
            "flow_file"  => _opt_string(flw, "flow_file", nothing),
        )
    end


    # --------------------------------------------------------------------------
    # Constructors
    # --------------------------------------------------------------------------

    """
        SimParams(d::Dict)

    Construct a `SimParams` from a flat `Dict{String,Any}` as returned by
    `load_toml_dict` (possibly after patching with CLI overrides).
    """
    function SimParams(p::Dict)
        SimParams(
            p["device"],   p["ens_name"],
            _to_vol(p["L"], "L"), _to_vol(p["Lx"], "Lx"),
            p["beta"],     p["c0"],
            p["start_from"], p["save_each"], p["save_final"], p["save_to"], p["logfile"],
            p["ntherm"],   p["ntraj"],   p["delta"],  p["nleaps"],
            p["flow_each"], p["flow_type"], p["adaptive"], p["Tflow"], p["epsilon"], p["nflow"],p["flow_file"]
        )
    end

    """
        SimParams(toml_path::String)

    Convenience constructor: load a TOML file and construct `SimParams` in one
    step.  Equivalent to `SimParams(load_toml_dict(toml_path))`.
    """
    SimParams(toml_path::String) = SimParams(load_toml_dict(toml_path))


    # --------------------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------------------

    """
        validate(p::SimParams)

    Check parameter consistency.  Throws an `ArgumentError` on the first
    inconsistency found.
    """
    function validate(p::SimParams)

        all(p.L  .> 0)         || throw(ArgumentError("L must be positive"))
        all(p.Lx .> 0)         || throw(ArgumentError("Lx must be positive"))
        all(p.L .% p.Lx .== 0) ||
            throw(ArgumentError("L must be divisible by Lx (got L=$(p.L), Lx=$(p.Lx))"))

        p.beta > 0 || throw(ArgumentError("beta must be positive"))

        if p.save_final || !isnothing(p.save_each)
            !isnothing(p.save_to) || throw(ArgumentError(
                "If either `save_each` or `save_final` is specified," *
                " `save_to` must be specified too "
            ))
        end

        hmc_fields = (:ntherm, :ntraj, :delta, :nleaps)
        hmc_set    = [!isnothing(getfield(p, f)) for f in hmc_fields]
        if any(hmc_set)
            all(hmc_set) ||
                throw(ArgumentError(
                    "If any HMC parameter is set, all of ntherm/ntraj/delta/nleaps " *
                    "must be set (got ntherm=$(p.ntherm), ntraj=$(p.ntraj), " *
                    "delta=$(p.delta), nleaps=$(p.nleaps))"
                ))
            p.ntraj  > 0  || throw(ArgumentError("ntraj must be positive"))
            p.ntherm >= 0 || throw(ArgumentError("ntherm must be non-negative"))
            p.delta  > 0  || throw(ArgumentError("delta must be positive"))
            p.nleaps > 0  || throw(ArgumentError("nleaps must be positive"))
        end

        flow_fields = (:flow_each, :flow_type, :epsilon, :nflow)
        flow_set    = [!isnothing(getfield(p, f)) for f in flow_fields]
        if any(flow_set)
            all(flow_set) ||
                throw(ArgumentError(
                    "If any flow parameter is set, all of flow_each/flow_type/" *
                    "epsilon/nflow must be set (got flow_each=$(p.flow_each), " *
                    "flow_type=$(p.flow_type), epsilon=$(p.epsilon), nflow=$(p.nflow))"
                ))
            p.epsilon   > 0 || throw(ArgumentError("epsilon must be positive"))
            p.nflow     > 0 || throw(ArgumentError("nflow must be positive"))
            p.flow_each > 0 || throw(ArgumentError("flow_each must be positive"))
            for ft in p.flow_type
                ft in ("wilson", "zeuthen") ||
                    throw(ArgumentError(
                        "Unknown flow_type \"$ft\". Valid choices: wilson, zeuthen."
                    ))
            end
        end

        return nothing
    end


    # --------------------------------------------------------------------------
    # Pretty-printer
    # --------------------------------------------------------------------------

    """
        parameter_banner(p::SimParams) -> String

    Return a formatted summary of all simulation parameters as a single string.
    Fields that were not set (i.e. `nothing`) are shown as `[not set]`.
    """
    function parameter_banner(p::SimParams)::String
        _fmt(x)  = isnothing(x) ? "[not set]" : string(x)
        _bool(x) = isnothing(x) ? "[not set]" : (x ? "yes" : "no")
        _vol(t)  = join(t, " × ")   # NTuple{4,Int} → "16 × 16 × 16 × 16"

        line  = "# " * "─"^72
        thick = "# " * "═"^72

        buf = IOBuffer()
        p!(s) = println(buf, s)   # one-char helper to keep the block tidy

        p!(thick)
        p!("#")
        p!("#   Ensemble : $(p.ens_name)")
        p!("#   Device   : GPU $(p.device) " * (isnothing(p.device) ?  "[default]" : "[selected]"))
        p!("#")
        p!(thick)

        p!(line);  p!("#  GEOMETRY");  p!(line)
        p!("#   Lattice    L  =  $(_vol(p.L))")
        p!("#   Sub-latt   Lx =  $(_vol(p.Lx))")

        p!(line);  p!("#  ACTION");  p!(line)
        p!("#   β   =  $(p.beta)")
        p!("#   c₀  =  $(p.c0)")

        p!(line);  p!("#  HMC");  p!(line)
        p!("#   Thermalization   :  $(_fmt(p.ntherm)) trajectories")
        p!("#   Production       :  $(_fmt(p.ntraj)) trajectories")
        p!("#   Step size  δ     :  $(_fmt(p.delta))")
        p!("#   Leapfrog steps   :  $(_fmt(p.nleaps))")

        p!(line);  p!("#  GRADIENT FLOW");  p!(line)
        if isnothing(p.flow_each)
            p!("#   [disabled]")
        else
            p!("#   Measure every    :  $(p.flow_each) trajectories")
            p!("#   Kernel(s)        :  $(join(p.flow_type, " + "))")
            p!("#   Step size  ϵ     :  $(p.epsilon)  ($(p.nflow) steps)")
            if !isnothing(p.adaptive) && p.adaptive
                p!("#   Mode             :  adaptive  (T_max = $(_fmt(p.Tflow)))")
            else
                p!("#   Mode             :  fixed step")
            end
        end
        if !isnothing(p.flow_file)
            p!("#   CSV output       :  $(p.flow_file)")
        else
            p!("#   CSV output       :  [disabled, using log]")
        end

        p!(line);  p!("#  I/O");  p!(line)
        _start = isnothing(p.start_from) ? "unit (cold start)" : p.start_from
        p!("#   Starting config  :  $_start")
        p!("#   Save config to   :  $(_fmt(p.save_to))")
        p!("#   Save every       :  $(_fmt(p.save_each)) trajectories")
        p!("#   Save final       :  $(_bool(p.save_final))")
        p!("#   Log file         :  $(isnothing(p.logfile) ? "stdout only" : p.logfile)")

        p!(thick)

        return String(take!(buf))
    end


    export SimParams, load_toml_dict, validate, parameter_banner
end