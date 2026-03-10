# ==============================================================================
#  MEASUREMENTS
#  Add a new subtype + a run_measurement! method to introduce a new observable.
# ==============================================================================

"""
    Measurement

Abstract supertype for all on-the-fly observables.
Each concrete subtype carries the measurement-specific parameters and
implements:

    run_measurement!(meas, state, p, lg) -> nothing

which computes the observable on the current `state.U` and writes the
result through `lg` (to the main log or a redirect file depending on the
tag used inside the method).
"""
abstract type Measurement end

"""
    FlowMeasurement

Gradient-flow observables (plaquette and clover energy densities,
topological charge) measured for every kernel listed in `p.flow_type`.
"""
struct FlowMeasurement <: Measurement end

"""
    run_measurement!(::FlowMeasurement, state, p, lg)

Stash `state.U`, run each requested flow kernel, emit data, then restore
the unflowed configuration.

If `p.flow_file` is set, one conf's rows are collected into a local buffer,
flushed to CSV in a single call, then dropped — nothing accumulates in
memory across configurations. The main log receives a one-line stub only.

If `p.flow_file` is nothing, the existing log/redirect behaviour is used.
"""
function run_measurement!(::FlowMeasurement, state::SimState,
                           p::SimParams, lg::SimLogger)
    state.U_cpu .= Array(state.U)   # snapshot before flowing

    rows = NamedTuple[]

    log_tag(lg, TAG_FLOW, "kernel t Eplq t2Eplq Eclv t2Eclv qtop qrec")
    for (label, wflw) in state.flow_kernels
        copyto!(state.U, state.U_cpu)   # fresh start for each kernel

        # t = 0 -------------
        if isnothing(p.flow_file)
            _log_flow_row(state, p, lg, label, 0.0)
        else
            push!(rows, _flow_row_nt(state, p, label, 0.0))
        end

        # flow loop ----------
        for step in 1:p.nflow
            flw(state.U, wflw, 1, p.epsilon, state.gp, state.lp, state.ymws)

            if isnothing(p.flow_file)
                _log_flow_row(state, p, lg, label, step * p.epsilon)
            else
                push!(rows, _flow_row_nt(state, p, label, step * p.epsilon))
            end
        end

        # data flush ----------
        if !isnothing(p.flow_file)
            _append_flow_csv(rows, p.flow_file)
            log_conf(lg, TAG_FLOW, state.itraj, "flow appended → %s  (%i rows)",
                    p.flow_file, length(rows))    
        end

    end

    copyto!(state.U, state.U_cpu)   # restore for HMC to continue
end




# ------------------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------------------

# Compute all observables at the current flow time and pack into a NamedTuple.
function _flow_row_nt(state::SimState, p::SimParams, label::String, ft::Float64)
    Eplq = Eoft_plaq(state.U, state.gp, state.lp, state.ymws)
    Eclv = Eoft_clover(state.U, state.gp, state.lp, state.ymws)
    qtop = Qtop(state.U, state.gp, state.lp, state.ymws)
    qrec = Qtop_rect(state.U, state.gp, state.lp, state.ymws)
    return (conf   = state.itraj,
            kernel = label,
            t      = ft,
            Eplq   = Eplq,   t2Eplq = ft^2 * Eplq,
            Eclv   = Eclv,   t2Eclv = ft^2 * Eclv,
            qtop   = qtop,
            qrec   = qrec)
end

# Append all rows for one conf to the CSV file.
# Writes the header automatically on the very first call (empty or absent file).
function _append_flow_csv(rows, path::String)
    first_write = !isfile(path) || iszero(filesize(path))
    if first_write
        CSV.write(path, rows)
    else
        CSV.write(path, rows; append = true)
    end
end

# Compute all observables and emit a formatted line to the log / redirect file.
function _log_flow_row(state::SimState, p::SimParams, lg::SimLogger,
                        label::String, ft::Float64)
    Eplq = Eoft_plaq(state.U, state.gp, state.lp, state.ymws)
    Eclv = Eoft_clover(state.U, state.gp, state.lp, state.ymws)
    qtop = Qtop(state.U, state.gp, state.lp, state.ymws)
    qrec = Qtop_rect(state.U, state.gp, state.lp, state.ymws)
    log_conf(lg, TAG_FLOW, state.itraj,
        "[%-7s]  %.6f  %14.13e %14.13e %14.13e %14.13e %14.13e %14.13e",
        label, ft, Eplq, ft^2*Eplq, Eclv, ft^2*Eclv, qtop, qrec)
end

# ------------------------------------------------------------------------------
# Future measurements — add here, e.g.:
#
#   struct PolyakovMeasurement <: Measurement end
#
#   function run_measurement!(::PolyakovMeasurement, state, p, lg)
#       ploop = polyakov_loop(state.U, state.lp, state.gp, state.ymws)
#       log_conf(lg, "POLY", state.itraj, "Re(P) = %.10f  Im(P) = %.10f",
#                real(ploop), imag(ploop))
#   end
# ------------------------------------------------------------------------------


"""
    run_measurements!(state, measurements, p, lg)

Run every `Measurement` in `measurements` on the current configuration.
Called when `state.itraj % p.flow_each == 0`.
"""
function run_measurements!(state::SimState, measurements::Vector{<:Measurement},
                            p::SimParams, lg::SimLogger)
    for meas in measurements
        run_measurement!(meas, state, p, lg)
    end
end