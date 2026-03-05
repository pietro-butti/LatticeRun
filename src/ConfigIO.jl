## =============================================================================
## ConfigIO.jl
##
## Everything related to gauge configuration files on disk.
##
## Responsibilities:
##   - Load a configuration from disk (wraps LatticeGPU).
##   - Save a configuration to disk (wraps LatticeGPU).
##   - Decide how to initialise the gauge field (cold / warm / explicit).
## =============================================================================

import LatticeGPU: read_cnfg, randomize!

"""
    init_gauge_field!(U, p::SimParams, lp, gp, lg::SimLogger) -> Int

Initialise the gauge field `U` and return the starting trajectory counter.

Decision logic:
1. If `p.start_cnfg` is explicitly set → load that file.
2. Else scan `p.saveto` for existing configs → load the latest (warm restart).
3. If none found → cold start (unit field).

The returned `Int` is the trajectory counter `CNFG` from which the run
should continue (0 for a cold start).
"""
function init_gauge_field!(U, p::SimParams, lp, gp, lg::SimLogger)
    if !isnothing(p.start_from)
        isfile(p.start_cnfg) || error("start_cnfg file not found: $(p.start_cnfg)")
        log_line(lg, "# [INIT] Loading explicit starting config: $(p.start_cnfg)")
        U .= LatticeGPU.read_cnfg(U, lp, gp)
    end

    if !isnothing(p.start_from) || p.start_from=="cold"
        fill!(U, one(eltype(U)))
        log_line(lg, "# [INIT] Cold start: gauge field set to unit configuration")
    end
    return 0
end
