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

Stash `state.U`, run each requested flow kernel, emit data through
`TAG_FLOW`, then restore the unflowed configuration.
"""
function run_measurement!(::FlowMeasurement, state::SimState,
                           p::SimParams, lg::SimLogger)
    state.U_cpu .= Array(state.U)   # snapshot before flowing

    for (label, wflw) in state.flow_kernels
        copyto!(state.U, state.U_cpu)   # fresh start for each kernel

        # t = 0
        _log_flow_row(state, p, lg, label, 0.0)

        for step in 1:p.nflow
            flw(state.U, wflw, 1, p.epsilon, state.gp, state.lp, state.ymws)
            _log_flow_row(state, p, lg, label, step * p.epsilon)
        end
    end

    copyto!(state.U, state.U_cpu)   # restore for HMC to continue
end

function _log_flow_row(state::SimState, p::SimParams, lg::SimLogger,
                        label::String, ft::Float64)
    Eplq = Eoft_plaq(state.U, state.gp, state.lp, state.ymws)
    Eclv = Eoft_clover(state.U, state.gp, state.lp, state.ymws)
    qtop = Qtop(state.U, state.gp, state.lp, state.ymws)
    qrec = Qtop_rect(state.U, state.gp, state.lp, state.ymws)
    log_conf(lg, TAG_FLOW, state.itraj,
        "kernel=%-7s  t=%.6f  Eplq=%.8e  t2Eplq=%.8e  Eclv=%.8e  t2Eclv=%.8e  qtop=%+.6f  qrec=%+.6f",
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