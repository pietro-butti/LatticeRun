## =============================================================================
## Runner.jl
##
## Orchestrates a full lattice simulation run.
##
## Responsibilities:
##   - set up runtime state from SimParams
##   - drive the MC update loop (composable algorithm schedule)
##   - dispatch measurements at the right trajectory intervals
##   - log and save configurations via SimLogger
##
## The *physics* lives elsewhere (LatticeGPU calls, flow kernels, …).
## This module is pure orchestration.
##
## Typical usage:
##
##   p      = SimParams("input.toml")
##   lg     = SimLogger(p; redirects = Dict(TAG_FLOW => "$(p.ens_name)_flow.dat"))
##   log_banner!(lg, p)
##
##   state    = setup_simulation(p, lg)
##   schedule = MCSchedule([(HMCUpdate(), 1)])
##   meas     = [FlowMeasurement()]
##
##   thermalize!(state, schedule, p, lg)
##   produce!(state, schedule, meas, p, lg)
##
##   close_logger(lg)
## =============================================================================

module Runner
    using CUDA, LatticeGPU
    using ..Parameters
    using ..Logger

    # include("RunnerUpdates.jl")
    # include("RunnerMeasurements.jl")

    # ==============================================================================
    #  SIMULATION STATE
    # ==============================================================================

    """
        SimState

    Mutable container for all runtime objects.

    `itraj` is the global trajectory counter — it starts at 0 (or at the
    counter embedded in a loaded configuration) and is incremented by every
    call to `run_update!`.
    """
    mutable struct SimState
        U            :: Any         # SU3 gauge field on GPU
        lp           :: Any         # SpaceParm
        gp           :: Any         # GaugeParm
        ymws         :: Any         # YMworkspace
        intsch       :: Any         # integrator schedule (HMC)
        flow_kernels :: Vector{Pair{String, Any}}   # [("Wilson", wfl), …]
        U_cpu        :: Array       # CPU mirror for flow snapshot/restore
        itraj        :: Int
    end



    """
        setup_simulation(p, lg) -> SimState

    Initialise all runtime objects from `p`:

    - Selects the CUDA device.
    - Builds `SpaceParm`, `GaugeParm`, `YMworkspace`, integrator.
    - Builds the requested flow kernels.
    - Initialises the gauge field (cold start or from file).
    - Logs the initial plaquette.

    Returns a `SimState` ready to be passed to `thermalize!` / `produce!`.
    """
    function setup_simulation(p::SimParams, lg::SimLogger)#::SimState
        # device!(p.device)  ############### CUDA ###############
        log_tag(lg, TAG_INIT, "CUDA device %i selected", p.device)

        # --- lattice and gauge objects ----------------------------------------
        lp    = SpaceParm{4}(p.L, p.Lx, BC_PERIODIC, (0,0,0,0,0,0))
        gp    = GaugeParm{Float64}(SU3{Float64}, p.beta, p.c0,
                                (1.0,1.0), (0.0,0.0), lp.iL)
        ymws  = YMworkspace(SU3, Float64, lp) ############### CUDA ###############
        U     = vector_field(SU3{Float64}, lp) ############### CUDA ###############

        # --- integrator -------------------------------------------------------
        intsch = omf4(Float64, p.delta, p.nleaps)

        # --- flow kernels (built once, reused every measurement) --------------
        flow_kernels = Pair{String, Any}[]
        if !isnothing(p.flow_type)
            "wilson"  in p.flow_type &&
                push!(flow_kernels, "Wilson"  => wfl_rk3(Float64, p.epsilon, 1e-7))
            "zeuthen" in p.flow_type &&
                push!(flow_kernels, "Zeuthen" => zfl_rk3(Float64, p.epsilon, 1e-7))
        end

        # --- gauge field initialisation ---------------------------------------
        itraj = init_gauge_field!(U, p, lp, gp, ymws, lg)

        plq = plaquette(U, lp, gp, ymws)
        log_tag(lg, TAG_INIT, "initial plaquette = %.16e", plq / 2)

        U_cpu = Array(U)

        return SimState(U, lp, gp, ymws, intsch, flow_kernels, U_cpu, itraj)
    end







    function init_gauge_field!(U, p::SimParams, lp, gp, ymws, lg::SimLogger)
        
        if isnothing(p.start_from) || p.start_from == "cold"
            fill!(U, one(eltype(U)))
            log_line(lg, "# [INIT] Cold start: gauge field set to unit configuration")
        elseif p.start_from == "hot"
            randomize!(ymws.mom, lp, ymws)
            U .= exp.(ymws.mom)
            log_line(lg, "# [INIT] Hot start: gauge field set to unit configuration")
        elseif isfile(p.start_cnfg)
            U .= LatticeGPU.read_cnfg(U; block=p.Lx)
            log_line(lg, "# [INIT] Loading explicit starting config: $(p.start_cnfg)")
        else
            error("Starting configuration not find. Select `cold`, `hot` or `<filename>`")
        end
        return 0
    end


    # """
    #     save_config!(state, p, lg)

    # Save the current gauge configuration to `p.save_to` and log the event.
    # """
    # function save_config!(state::SimState, p::SimParams, lg::SimLogger)
    #     isnothing(p.save_to) && return   # nowhere to save

    #     fname = joinpath(p.save_to,
    #                     "$(p.ens_name).cfg_n$(lpad(state.itraj, 6, '0'))")
    #     save_cnfg(fname, state.U, state.lp, state.gp)
    #     log_conf(lg, TAG_IO, state.itraj, "configuration saved → %s", fname)
    # end


    # """
    #     produce!(state, schedule, measurements, p, lg)

    # Run `p.ntraj` production MC sweeps.

    # After each sweep:
    # - measurements are run if `state.itraj % p.flow_each == 0`
    # - the configuration is saved if `state.itraj % p.save_each == 0`

    # If `p.save_final` is true the last configuration is always saved,
    # regardless of `save_each`.
    # """
    # function run!(state::SimState, schedule::MCSchedule,
    #                 measurements::Vector{<:Measurement},
    #                 p::SimParams, lg::SimLogger)

    #     log_tag(lg, TAG_HMC, "starting production (%i trajectories)", p.ntraj)

    #     for _ in 1:p.ntraj
    #         mc_sweep!(state, schedule, p, lg)

    #         # -- measurements --------------------------------------------------
    #         if !isnothing(p.flow_each) && state.itraj % p.flow_each == 0
    #             run_measurements!(state, measurements, p, lg)
    #         end

    #         # -- periodic save -------------------------------------------------
    #         if !isnothing(p.save_each) && state.itraj % p.save_each == 0
    #             save_config!(state, p, lg)
    #         end
    #     end

    #     # -- final save --------------------------------------------------------
    #     if p.save_final
    #         save_config!(state, p, lg)
    #     end

    #     log_tag(lg, TAG_HMC, "production complete  (itraj = %i)", state.itraj)
    # end


    # # """
    # #     run_simulation!(p, lg;
    # #                     schedule     = MCSchedule([(HMCUpdate(), 1)]),
    # #                     measurements = Measurement[FlowMeasurement()])

    # # Run a complete simulation (setup → thermalisation → production).

    # # The defaults give a single HMC step per trajectory with gradient-flow
    # # measurements.  Override either keyword to change the update strategy or
    # # the set of observables without touching this function.
    # # """
    # # function run_simulation!(p::SimParams, lg::SimLogger;
    # #     schedule     :: MCSchedule           = MCSchedule([(HMCUpdate(), 1)]),
    # #     measurements :: Vector{<:Measurement} = Measurement[FlowMeasurement()],
    # # )
    # #     state = setup_simulation(p, lg)
    # #     thermalize!(state, schedule, p, lg)
    # #     produce!(state, schedule, measurements, p, lg)
    # #     return state
    # # end



    export SimState, setup_simulation, init_gauge_field!, save_config!, run!
    export MCAlgorithm, HMCUpdate, run_update!, MCSchedule, mc_sweep!, thermalize!
    export Measurement, FlowMeasurement, run_measurement!, run_measurements!
end
