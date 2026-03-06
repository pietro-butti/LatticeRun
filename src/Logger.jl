## =============================================================================
## Logger.jl
##
## SimLogger — a multiplexer that writes formatted output to stdout and/or an
## on-disk log file, with optional per-tag redirection to dedicated data files.
##
## Log-line format:
##   [ TAG            ][ CONF 000042 ] - free text
##
## Per-tag redirect: if a tag (e.g. TAG_FLOW) is registered to a file, its
## data lines go there and the main log receives a one-line stub instead:
##   [ FLOW           ][ CONF 000042 ] - output appended → ensemble_flow.dat
## =============================================================================

module Logger
    using Dates, Printf
    using ..Parameters


    # --------------------------------------------------------------------------
    # Tag constants
    # --------------------------------------------------------------------------

    const TAG_HMC   = "HMC"
    const TAG_FLOW  = "FLOW"
    const TAG_IO    = "IO"
    const TAG_INIT  = "INIT"
    const TAG_THERM = "THERM"


    # --------------------------------------------------------------------------
    # Struct
    # --------------------------------------------------------------------------

    """
        SimLogger

    Multiplexed logger. Writes every non-redirected message to any combination
    of stdout and an on-disk log file simultaneously.

    Per-tag redirects send a tag's data to a dedicated file; the main log
    receives a short stub so the log remains a complete narrative.

    Construct with `SimLogger(...)`, shut down with `close_logger(lg)`.
    """
    struct SimLogger
        screen             :: Union{IO, Nothing}
        file               :: Union{IO, Nothing}
        redirects          :: Dict{String, IO}   # tag → open file handle
        redirect_announced :: Set{String}        # tags whose stub has been printed once
    end


    # --------------------------------------------------------------------------
    # Constructors
    # --------------------------------------------------------------------------

    """
        SimLogger(; to_screen, logfile, redirects) -> SimLogger

    Low-level constructor.

    | Keyword    | Type                       | Default   | Effect                              |
    |------------|----------------------------|-----------|-------------------------------------|
    | `to_screen`| `Bool`                     | `true`    | mirror output to stdout             |
    | `logfile`  | `String` or `nothing`      | `nothing` | append main log to this path        |
    | `redirects`| `Dict{String,String}`      | `Dict()`  | `tag => path` pairs for data files  |

    All files are opened in **append** mode so restarts extend rather than
    overwrite existing logs.
    """
    function SimLogger(;
        to_screen :: Bool                    = true,
        logfile   :: Union{String, Nothing}  = nothing,
        redirects :: Dict{String, String}    = Dict{String, String}(),
    )
        screen = to_screen ? stdout : nothing
        fh     = isnothing(logfile) ? nothing : open(logfile, "a")

        redirect_ios = Dict{String, IO}(
            tag => open(path, "a") for (tag, path) in redirects
        )

        return SimLogger(screen, fh, redirect_ios, Set{String}())
    end

    """
        SimLogger(p::SimParams; redirects) -> SimLogger

    Convenience constructor that reads `to_screen` and `logfile` from a
    `SimParams`.  Pass `redirects` as a `Dict{String,String}` to register
    per-tag data files, e.g.:

        SimLogger(p; redirects = Dict(TAG_FLOW => "\$(p.ens_name)_flow.dat"))
    """
    function SimLogger(p::SimParams; redirects::Dict{String, String} = Dict{String, String}())
        return SimLogger(;
            to_screen = isnothing(p.logfile),
            logfile   = p.logfile,
            redirects = redirects,
        )
    end


    # --------------------------------------------------------------------------
    # Internal write primitive
    # --------------------------------------------------------------------------

    @inline function _write_main(lg::SimLogger, msg::String)
        isnothing(lg.screen) || print(lg.screen, msg)
        isnothing(lg.file)   || print(lg.file,   msg)
    end


    # --------------------------------------------------------------------------
    # Public API — untagged helpers
    # --------------------------------------------------------------------------

    """
        log_line(lg, msg)

    Write `msg` followed by a newline to all main sinks (screen + log file).
    """
    function log_line(lg::SimLogger, msg::String)
        _write_main(lg, msg * "\n")
    end

    """
        log_printf(lg, fmt, args...)

    Printf-style formatted output to all main sinks.
    Include `\\n` in the format string if a newline is needed.
    """
    function log_printf(lg::SimLogger, fmt::String, args...)
        _write_main(lg, Printf.format(Printf.Format(fmt), args...))
    end


    # --------------------------------------------------------------------------
    # Public API — tagged, configuration-stamped log lines
    # --------------------------------------------------------------------------

    """
        log_conf(lg, tag, cnfg, fmt, args...)

    Write a tagged, configuration-stamped log line to the appropriate sink:

        [ TAG            ][ CONF 000042 ] - <formatted message>

    If `tag` has a registered redirect the formatted message goes to the
    redirect file and the main log receives a short stub:

        [ TAG            ][ CONF 000042 ] - output appended → path

    `tag`  — short uppercase label, e.g. `TAG_HMC`, `TAG_FLOW`, `"MYNEWTAG"`.
    `cnfg` — current trajectory / configuration counter.
    `fmt`  — Printf format string for the message body.
    """
    function log_conf(lg::SimLogger, tag::String, cnfg::Int, fmt::String, args...)
        prefix = @sprintf("[ %-14s ][ CONF %06i ] - ", tag, cnfg)
        body   = Printf.format(Printf.Format(fmt), args...)
        line   = prefix * body * "\n"

        if haskey(lg.redirects, tag)
            redir_io = lg.redirects[tag]
            print(redir_io, line)
            flush(redir_io)

            if tag ∉ lg.redirect_announced
                redir_name = redir_io isa IOStream ? redir_io.name : repr(redir_io)
                _write_main(lg, prefix * "output redirected → $redir_name\n")
                push!(lg.redirect_announced, tag)
            end
        else
            _write_main(lg, line)
        end
    end

    """
        log_tag(lg, tag, fmt, args...)

    Like `log_conf` but without a configuration number — for messages that span
    the whole run rather than a single trajectory (e.g. startup, shutdown).

        [ TAG            ] - <formatted message>
    """
    function log_tag(lg::SimLogger, tag::String, fmt::String, args...)
        prefix = @sprintf("[ %-14s ] - ", tag)
        body   = Printf.format(Printf.Format(fmt), args...)
        _write_main(lg, prefix * body * "\n")
    end


    # --------------------------------------------------------------------------
    # Startup banner
    # --------------------------------------------------------------------------

    """
        log_banner!(lg, p)

    Write a startup header (timestamp, hostname, device) followed by the full
    parameter banner (from `Parameters.jl`) to the main log.
    Redirect files are not touched.
    """
    function log_banner!(lg::SimLogger, p::SimParams)
        thick = "# " * "═"^72
        header = join([
            thick,
            "#  Run started  :  $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))",
            "#  Host         :  $(gethostname())",
            "#  Device       :  GPU $(p.device)",
            thick,
            "",
        ], "\n")

        _write_main(lg, header * parameter_banner(p))
        flush_logger(lg)
    end


    # --------------------------------------------------------------------------
    # Flush and close
    # --------------------------------------------------------------------------

    """
        flush_logger(lg)

    Flush all active sinks (main + all redirect files).
    """
    function flush_logger(lg::SimLogger)
        isnothing(lg.screen) || flush(lg.screen)
        isnothing(lg.file)   || flush(lg.file)
        for io in values(lg.redirects)
            flush(io)
        end
    end

    """
        close_logger(lg)

    Flush and close the file sink and all redirect files.  Safe to call even if
    no files were opened.  Never closes `stdout`.
    """
    function close_logger(lg::SimLogger)
        flush_logger(lg)
        isnothing(lg.file) || close(lg.file)
        for io in values(lg.redirects)
            close(io)
        end
    end

    export TAG_INIT, TAG_IO, TAG_THERM, TAG_HMC, TAG_FLOW
    export SimLogger, log_banner!, log_tag, log_conf, close_logger
end