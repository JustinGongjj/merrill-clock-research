"""Tests for the clock-data command-line interface (new subcommands)."""
from __future__ import annotations

import argparse

import pytest

from clockdata import cli


def _ns(**kwargs) -> argparse.Namespace:
    base = {"db": ":memory:", "host": "127.0.0.1", "port": 8765,
            "output": None, "matrix": False, "verbose": False}
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_registry_subcommand_lists_every_source(capsys):
    assert cli.cmd_registry(_ns()) == 0
    out = capsys.readouterr().out
    for source_id in ("baostock", "akshare", "ccxt", "fred", "local_db"):
        assert source_id in out


def test_registry_matrix_subcommand_shows_capability_chains(capsys):
    assert cli.cmd_registry(_ns(matrix=True)) == 0
    out = capsys.readouterr().out
    for cap in ("a_share", "us_equity", "crypto", "macro", "factors"):
        assert cap in out
    # The chain must be ordered best-tier first.
    a_share = out.split("a_share  ")[1].split("\n\n")[0]
    tiers = [int(line.split("tier=")[1].split()[0]) for line in a_share.splitlines()[1:]
             if "tier=" in line]
    assert tiers == sorted(tiers)


def test_serve_subcommand_builds_a_service_without_binding_a_port(monkeypatch):
    """cmd_serve must hand a pre-built service to serve() without binding early."""
    captured = {}

    def fake_serve(host, port, state=None, **kwargs):
        captured["host"] = host
        captured["port"] = port
        captured["state"] = state

    monkeypatch.setattr("clockdata.server.serve", fake_serve)
    assert cli.cmd_serve(_ns(host="127.0.0.1", port=9999)) == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9999
    assert captured["state"] is not None
    assert captured["state"].db_path is not None


def test_registry_and_serve_are_registered_subcommands():
    """Both new subcommands must be wired into the parser with their flags."""
    parser = cli.build_parser()

    reg = parser.parse_args(["registry"])
    assert reg.command == "registry"
    assert reg.matrix is False
    assert reg.verbose is False
    assert parser.parse_args(["registry", "--matrix"]).matrix is True
    assert parser.parse_args(["registry", "--verbose"]).verbose is True

    srv = parser.parse_args(["serve", "--port", "9100", "--host", "0.0.0.0"])
    assert srv.command == "serve"
    assert srv.port == 9100
    assert srv.host == "0.0.0.0"
    assert callable(srv.func)


def test_existing_subcommands_still_parse():
    parser = cli.build_parser()
    for argv in (["init"], ["sources"], ["status"], ["check"], ["sync"], ["run"]):
        args = parser.parse_args(argv)
        assert callable(args.func), f"{argv} has no handler"
