"""The derived branch's configuration arm: the override flag and its word.

The span arrangement itself lives in `sglang.srt.afd` and installs as the resident
"span" arm; what registers here is the configuration surface shift 1 adds -- the
DANGEROUS `--afd-query-shift-layers` override, its validation, its refusal on a host
(the pool owns the arrangement's flags), and the pushed word the host adopts. This arm
never installs anything on a model (`wanted` is False): the span serves both shifts,
and the read point it serves at is resolved through `afd.checkpoint` from what this
arm validated and moved.

Importing this module also claims the early read's inbound ops, which is the one side
effect a spawned scheduler needs before the first shifted frame.
"""

import sglang.srt.afd_query_shift  # noqa: F401 -- OP_STATE_EARLY / OP_STATE_APPLY

from sglang.srt.afd.arms import register


class QueryShiftConfig:
    name = "query-shift"

    def wanted(self) -> bool:
        # never the installing arm: the resident span arm serves both read points
        return False

    @staticmethod
    def check_args(server_args) -> None:
        """The override flag, validated -- and refused on a host outright.

        The pool owns the arrangement's flags; a host-side value could only agree with
        the pushed one (redundant) or disagree (a second source of truth), so a host
        that sets the override is refused by name before a model loads.
        """
        if (
            getattr(server_args, "afd_mode", "null") == "host"
            and getattr(server_args, "afd_query_shift_layers", None) is not None
        ):
            raise ValueError(
                "--afd-query-shift-layers configures the POOL, and --afd-mode=host. "
                "A host adopts the pool's pushed configuration at startup; set the "
                "override on the pool."
            )
        from sglang.srt.afd_query_shift.arg_checks import check

        check(server_args)

    @staticmethod
    def pushed_config(server_args) -> dict:
        """This arm's settings as the pool pushes them. Bodies in `pushed`, which is the
        one file allowed to move the raw request between the two ends' resolutions."""
        from sglang.srt.afd_query_shift.pushed import pushed_config

        return pushed_config(server_args)

    @staticmethod
    def adopt_config(cfg: dict, server_args) -> None:
        """Take the pool's word on the host. Bodies in `pushed`; see there for the rule."""
        from sglang.srt.afd_query_shift.pushed import adopt_config

        adopt_config(cfg, server_args)


register(QueryShiftConfig.name, QueryShiftConfig)
