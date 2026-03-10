## =============================================================================
## Parameters.jl
##
## Defines SimParams — the single, immutable source of truth for all simulation
## parameters.  Provides constructors from a TOML file and from a parsed CLI
## Dict, plus validation and a pretty-printer.
## =============================================================================
module Parameters
    using TOML

    # --------------------------------------------------------------------------
    # Struct
    # --------------------------------------------------------------------------

    """
        SimParams

    Immutable container for all simulation parameters.
    Constructed either from a TOML file (`SimParams(path::String)`) or from the
    Dict returned by `parse_commandline()` (`SimParams(args::Dict)`).
    """
    struct SimParams
        # --- Run specifics ---
        device   :: Int
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
        flow_each :: Union{Int,          Nothing}
        flow_type :: Union{Vector{String}, Nothing}   # nothing → no flow
        adaptive  :: Union{Bool,         Nothing}
        Tflow     :: Union{Float64,      Nothing}
        epsilon   :: Union{Float64,      Nothing}
        nflow     :: Union{Int,          Nothing}
    end


    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------

    # Convert whatever the user / TOML provided for L / Lx to NTuple{4,Int}.
    # Accepts a scalar (broadcast), a Vector, or an existing NTuple.
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
    # Constructor from Dict (CLI or pre-built)
    # --------------------------------------------------------------------------

    function SimParams(p::Dict)
        SimParams(
            p["device"],   p["ens_name"],
            _to_vol(p["L"], "L"), _to_vol(p["Lx"], "Lx"),
            p["beta"],     p["c0"],
            p["start_from"], p["save_each"], p["save_final"], p["save_to"], p["logfile"],
            p["ntherm"],   p["ntraj"],   p["delta"],  p["nleaps"],
            p["flow_each"], p["flow_type"], p["adaptive"], p["Tflow"], p["epsilon"], p["nflow"],
        )
    end


    # --------------------------------------------------------------------------
    # Constructor from TOML file
    # --------------------------------------------------------------------------

    """
        SimParams(toml_path::String)

    Load simulation parameters from a TOML file.

    Expected file structure:

    ```toml
    [run]
    device   = 0
    ens_name = "my_ensemble"

    [geometry]
    L  = [16, 16, 16, 16]   # or just L = 16 for a uniform lattice
    Lx = [8,  8,  8,  8]

    [action]
    beta = 6.2
    c0   = 1.0

    [hmc]
    ntherm     = 100
    ntraj      = 1000
    delta      = 0.02
    nleaps     = 20
    start_from = ""          # leave empty for cold start

    [io]
    save_each  = 10
    save_final = true
    save_to    = "./configs"
    logfile    = "./run.log"  # omit or leave empty for stdout only

    [flow]
    flow_each = 5
    flow_type = ["wilson", "zeuthen"]
    adaptive  = false
    epsilon   = 0.01
    nflow     = 100
    ```
    """
    function SimParams(toml_path::String)
        raw = TOML.parsefile(toml_path)

        run  = get(raw, "run",      Dict())
        geo  = get(raw, "geometry", Dict())
        act  = get(raw, "action",   Dict())
        io   = get(raw, "io",       Dict())
        hmc  = get(raw, "hmc",      Dict())
        flw  = get(raw, "flow",     Dict())

        d = Dict(
            "device"   => run["device"],
            "ens_name" => run["ens_name"],

            "L"  => geo["L"],
            "Lx" => geo["Lx"],

            "beta" => act["beta"],
            "c0"   => act["c0"],

            "start_from" => _opt_string(io, "start_from", nothing),
            "save_each"  => _opt(io,  "save_each",  nothing),
            "save_final" => _opt(io,"save_final", false),
            "logfile"    => _opt_string(io, "logfile", nothing),
            "save_to"    => _opt_string(io, "save_to", nothing),

            "ntherm" => _opt(hmc, "ntherm", nothing),
            "ntraj"  => _opt(hmc, "ntraj",  nothing),
            "delta"  => _opt(hmc, "delta",  nothing),
            "nleaps" => _opt(hmc, "nleaps", nothing),

            "flow_type" => _opt(flw, "flow_type", nothing),
            "flow_each" => _opt(flw, "flow_each", nothing),
            "adaptive"  => _opt(flw, "adaptive",  nothing),
            "Tflow"     => _opt(flw, "Tflow",     nothing),
            "epsilon"   => _opt(flw, "epsilon",   nothing),
            "nflow"     => _opt(flw, "nflow",     nothing),
        )

        return SimParams(d)
    end


    # --------------------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------------------

    """
        validate(p::SimParams)

    Check parameter consistency.  Throws an `ArgumentError` on the first
    inconsistency found.
    """
    function validate(p::SimParams)

        # --- geometry ---
        # (no length check needed: NTuple{4,Int} is always length 4 by construction)
        all(p.L  .> 0)         || throw(ArgumentError("L must be positive"))
        all(p.Lx .> 0)         || throw(ArgumentError("Lx must be positive"))
        all(p.L .% p.Lx .== 0) ||
            throw(ArgumentError("L must be divisible by Lx (got L=$(p.L), Lx=$(p.Lx))"))

        # --- action ---
        p.beta > 0 || throw(ArgumentError("beta must be positive"))

        # --- I/O ---
        if p.save_final || !isnothing(p.save_each)
            !isnothing(p.save_to) || throw(ArgumentError(
                "If either `save_each` or `save_final` is specified," *
                " `save_to` must be specified too "
            ))
        end

        # --- HMC: all four fields are required together -----------------------
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

        # --- flow: all required fields must be present together ---------------
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

    Typical usage:

        banner = parameter_banner(p)
        print(banner)                        # → stdout
        write(logfile_io, banner)            # → file
        print(banner); write(logfile_io, banner)   # → both
    """
    function parameter_banner(p::SimParams)::String
        # ---- formatting helpers ----------------------------------------------
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
        p!("#   Device   : GPU $(p.device)")
        p!("#")
        p!(thick)

        # --- Geometry ---------------------------------------------------------
        p!(line)
        p!("#  GEOMETRY")
        p!(line)
        p!("#   Lattice    L  =  $(_vol(p.L))")
        p!("#   Sub-latt   Lx =  $(_vol(p.Lx))")

        # --- Action -----------------------------------------------------------
        p!(line)
        p!("#  ACTION")
        p!(line)
        p!("#   β   =  $(p.beta)")
        p!("#   c₀  =  $(p.c0)")

        # --- HMC --------------------------------------------------------------
        p!(line)
        p!("#  HMC")
        p!(line)
        p!("#   Thermalization   :  $(_fmt(p.ntherm)) trajectories")
        p!("#   Production       :  $(_fmt(p.ntraj)) trajectories")
        p!("#   Step size  δ     :  $(_fmt(p.delta))")
        p!("#   Leapfrog steps   :  $(_fmt(p.nleaps))")

        # --- Gradient flow ----------------------------------------------------
        p!(line)
        p!("#  GRADIENT FLOW")
        p!(line)
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

        # --- I/O --------------------------------------------------------------
        p!(line)
        p!("#  I/O")
        p!(line)
        _start = isnothing(p.start_from) ? "unit (cold start)" : p.start_from
        p!("#   Starting config  :  $_start")
        p!("#   Save config to   :  $(_fmt(p.save_to))")
        p!("#   Save every       :  $(_fmt(p.save_each)) trajectories")
        p!("#   Save final       :  $(_bool(p.save_final))")
        p!("#   Log file         :  $(isnothing(p.logfile) ? "stdout only" : p.logfile)")

        p!(thick)

        return String(take!(buf))
    end


    export SimParams, validate, parameter_banner
end