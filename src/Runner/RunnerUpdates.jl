# ==============================================================================
#  MC ALGORITHMS
#  Add a new subtype + a run_update! method to introduce a new algorithm.
# ==============================================================================

"""
    MCAlgorithm

Abstract supertype for all MC update algorithms. Each concrete subtype carries the algorithm-specific parameters and implements:

    run_update!(alg, state, p, lg) -> nothing

which advances `state.U` by one application of the algorithm and writes any relevant output through `lg`.
"""
abstract type MCAlgorithm end

"""
    HMCUpdate

One application of the Hybrid Monte Carlo algorithm. Parameters (step size, number of leaps) are read from `SimParams`.
"""
struct HMCUpdate <: MCAlgorithm end

"""
    run_update!(::HMCUpdate, state, p, lg)

Run one HMC trajectory, update `state.itraj`, and log the result.
"""
function run_update!(::HMCUpdate, state::SimState, p::SimParams, lg::SimLogger)
    dh, acc = HMC!(state.U, state.intsch, state.lp, state.gp, state.ymws)
    state.itraj += 1
    plq = plaquette(state.U, state.lp, state.gp, state.ymws)
    log_conf(lg, TAG_HMC, state.itraj,
        "ΔH = %+.6e  [acc %i]  Plaq = %.10f", dh, acc, plq)
end

# ------------------------------------------------------------------------------
# Future algorithms — add here, e.g.:
#
#   struct ORUpdate <: MCAlgorithm
#       n_hits :: Int
#   end
#
#   function run_update!(alg::ORUpdate, state, p, lg)
#       for _ in 1:alg.n_hits
#           OR!(state.U, state.lp, state.gp, state.ymws)
#       end
#       state.itraj += 1
#       plq = plaquette(...)
#       log_conf(lg, TAG_OR, state.itraj, "Plaq = %.10f", plq)
#   end
# ------------------------------------------------------------------------------


# ==============================================================================
#  MC SCHEDULE
#  Describes what one "step" of the Markov chain consists of.
# ==============================================================================

"""
    MCSchedule

Ordered list of `(algorithm, n_repeats)` pairs that together constitute one
step of the Markov chain (i.e. one new entry in the MC sequence).

Example — 1 HMC step followed by 3 OR sweeps (once OR is implemented):

    MCSchedule([(HMCUpdate(), 1), (ORUpdate(3), 3)])
"""
struct MCSchedule
    steps :: Vector{Pair{MCAlgorithm, Int}}
end

MCSchedule(pairs::Vector{<:Tuple}) = MCSchedule([alg => n for (alg, n) in pairs])




# ==============================================================================
#  MC SWEEP
# ==============================================================================

"""
    mc_sweep!(state, schedule, p, lg)

Execute one full step of the Markov chain as defined by `schedule`.
Each `(algorithm, n)` pair in the schedule is applied `n` times in order.
"""
function mc_sweep!(state::SimState, schedule::MCSchedule,
                   p::SimParams, lg::SimLogger)
    for (alg, n) in schedule.steps
        for _ in 1:n
            run_update!(alg, state, p, lg)
        end
    end
end


# ==============================================================================
#  THERMALIZATION
# ==============================================================================

"""
    thermalize!(state, schedule, p, lg)

Run `p.ntherm` MC sweeps to thermalise the gauge field.
No measurements are performed; `state.itraj` advances normally so the
trajectory counter is continuous going into production.
"""
function thermalize!(state::SimState, schedule::MCSchedule,
                    p::SimParams, lg::SimLogger)
    isnothing(p.ntherm) || p.ntherm == 0 && return

    log_tag(lg, TAG_THERM, "starting thermalization (%i trajectories)", p.ntherm)

    for _ in 1:p.ntherm
        mc_sweep!(state, schedule, p, lg)
    end

    log_tag(lg, TAG_THERM, "thermalization complete  (itraj = %i)", state.itraj)
end
