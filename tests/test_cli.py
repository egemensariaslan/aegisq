"""Command-line interface: argument routing and the cheap subcommands.

The demo and benchmark commands train circuits and are far too slow for a unit
suite, so their *routing* is checked with a stub while the fast, read-only
commands run for real.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from aegisq.cli import (
    COMMANDS,
    _int_list,
    build_parser,
    cmd_benchmark,
    cmd_demo,
    cmd_info,
    main,
    normalise_argv,
)


class TestArgvNormalisation:
    def test_bare_invocation_becomes_the_demo(self):
        assert normalise_argv([]) == ["demo"]

    def test_flags_without_a_subcommand_attach_to_the_demo(self):
        assert normalise_argv(["--quick"]) == ["demo", "--quick"]

    def test_an_explicit_subcommand_is_left_alone(self):
        for command in COMMANDS:
            assert normalise_argv([command]) == [command]
        assert normalise_argv(["benchmark", "--qubits", "5"]) == ["benchmark", "--qubits", "5"]

    @pytest.mark.parametrize("flag", ["-h", "--help", "--version"])
    def test_top_level_flags_reach_the_root_parser(self, flag):
        assert normalise_argv([flag]) == [flag]


class TestParser:
    def test_bare_invocation_resolves_to_the_demo_command(self):
        args = build_parser().parse_args(normalise_argv([]))
        assert args.func is cmd_demo
        assert args.quick is False

    def test_quick_flag_reaches_the_demo(self):
        args = build_parser().parse_args(normalise_argv(["--quick"]))
        assert args.func is cmd_demo and args.quick is True

    def test_every_subcommand_is_reachable(self):
        parser = build_parser()
        for command in ("demo", "benchmark", "plateau", "zne", "symmetry",
                        "choose-ansatz", "serve", "info"):
            args = parser.parse_args([command])
            assert callable(args.func)

    def test_benchmark_defaults_and_overrides(self):
        parser = build_parser()
        args = parser.parse_args(["benchmark", "--qubits", "5", "--epochs", "3", "--no-zne"])
        assert (args.qubits, args.epochs, args.no_zne) == (5, 3, True)
        assert args.dataset == "two_moons" and args.seeds == 2

    def test_benchmark_rejects_an_unknown_dataset(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["benchmark", "--dataset", "mnist"])

    def test_version_flag_exits_cleanly(self, capsys):
        import aegisq

        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])
        assert exit_info.value.code == 0
        assert aegisq.__version__ in capsys.readouterr().out


class TestQubitListOption:
    def test_parses_comma_separated_widths(self):
        assert _int_list("4,6,8") == [4, 6, 8]
        assert _int_list(" 4 , 8 ") == [4, 8]

    def test_rejects_non_integers(self):
        with pytest.raises(Exception, match="comma-separated integers"):
            _int_list("4,six")

    def test_requires_at_least_two_widths(self):
        with pytest.raises(Exception, match="at least two"):
            _int_list("4")


class TestFastCommands:
    def test_info_lists_the_catalog(self, capsys):
        assert cmd_info(build_parser().parse_args(["info"])) == 0
        out = capsys.readouterr().out
        for expected in ("local_entangler", "richardson", "hardware_like", "two_moons",
                         "particle_number"):
            assert expected in out

    def test_module_entry_point_runs(self):
        result = subprocess.run(
            [sys.executable, "-m", "aegisq", "--version"],
            capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 0
        assert "aegisq" in result.stdout

    def test_importing_main_module_does_not_launch_the_cli(self):
        """An unguarded __main__ would run the demo during test collection."""
        result = subprocess.run(
            [sys.executable, "-c", "import aegisq.__main__; print('imported')"],
            capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "imported"


class TestBenchmarkQuickMode:
    def test_quick_flag_parses(self):
        args = build_parser().parse_args(["benchmark", "--quick"])
        assert args.quick is True

    def test_quick_overrides_the_expensive_defaults(self, monkeypatch):
        """--quick must actually shrink the sweep, not just set a flag."""
        captured: dict = {}

        def stub(**kwargs):
            captured.update(kwargs)

            class _Result:
                def summary(self) -> str:
                    return "stub"

            return _Result()

        monkeypatch.setattr("aegisq.benchmark.standard_benchmark", stub)
        assert cmd_benchmark(build_parser().parse_args(["benchmark", "--quick"])) == 0
        assert captured["epochs"] == 5
        assert captured["seeds"] == (0,)
        assert set(captured["noise_levels"]) == {"noiseless", "depolarizing"}
        assert len(captured["dataset"]) + len(captured["dataset"].x_test) == 80

    def test_without_quick_the_full_noise_sweep_is_used(self, monkeypatch):
        captured: dict = {}

        def stub(**kwargs):
            captured.update(kwargs)

            class _Result:
                def summary(self) -> str:
                    return "stub"

            return _Result()

        monkeypatch.setattr("aegisq.benchmark.standard_benchmark", stub)
        assert cmd_benchmark(build_parser().parse_args(["benchmark"])) == 0
        assert len(captured["noise_levels"]) == 4
        assert captured["epochs"] == 10


class TestServeCommand:
    def test_serve_defaults_to_loopback(self):
        args = build_parser().parse_args(["serve"])
        assert args.host == "127.0.0.1"
        assert args.port == 8765
        assert args.no_browser is False

    def test_serve_accepts_overrides(self):
        args = build_parser().parse_args(["serve", "--port", "9000", "--no-browser"])
        assert (args.port, args.no_browser) == (9000, True)

    def test_serve_is_a_known_command_for_argv_normalisation(self):
        assert normalise_argv(["serve", "--port", "9000"]) == ["serve", "--port", "9000"]
