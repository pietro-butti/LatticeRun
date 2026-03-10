using LatticeRun

p  = SimParams("simple_run.toml")
validate(p)

lg = SimLogger(p; redirects = Dict(TAG_FLOW => "$(p.ens_name)_flow.dat"))
log_banner!(lg, p)

state    = setup_simulation(p, lg)
schedule = MCSchedule([(HMCUpdate(), 1)])
meas     = Measurement[FlowMeasurement()]

thermalize!(state, schedule, p, lg)
run!(state, schedule, meas, p, lg)

close_logger(lg)