using LatticeRun

# setup parameters
p  = SimParams("simple_run.toml")
validate(p)

# logger
lg = SimLogger(p)
# lg = SimLogger(p; redirects = Dict(TAG_FLOW => "$(p.ens_name)_flow.dat")) # if you want a specific file for FLOW data

log_banner!(lg, p)

state    = setup_simulation(p, lg)
schedule = MCSchedule([(HMCUpdate(), 1)])
meas     = Measurement[FlowMeasurement()]

thermalize!(state, schedule, p, lg)
run!(state, schedule, meas, p, lg)

close_logger(lg)