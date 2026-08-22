"""The query-shift arm, derived from AFD and isolated from it.

AFD may land upstream before this does, so `sglang.srt.afd` must work with this directory deleted
-- not merely with the arm disabled. `test_afd_stands_alone.py` checks that as one grep: no file
under `srt/afd` may name this package.

The dependency runs one way. This package imports AFD; AFD never imports this. An arm announces
itself by importing `sglang.srt.afd.arms` and registering a factory, and whoever wants the arm
imports this package -- which is the only thing that puts it in the registry.
"""
