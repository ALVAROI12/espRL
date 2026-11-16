"""ENV selection, success test, gates and the refusals in hil.py (no board needed)."""

import pytest

import hil
import train


def test_success_needs_last_100_upright():
    up = [1.0] * 500
    assert train.success(up, terminated=False)
    dip = list(up)
    dip[-50] = 0.9
    assert not train.success(dip, terminated=False)
    assert not train.success(up, terminated=True)
    assert not train.success([1.0] * 99, terminated=False)


def test_gate():
    assert train.gate("pendulum", [-200.0] * 10, [False] * 10)[0]
    assert not train.gate("pendulum", [-300.0] * 10, [True] * 10)[0]
    assert train.gate("cartpole_balance", [0.0] * 10, [True] * 9 + [False])[0]
    assert not train.gate("cartpole_swingup", [0.0] * 10, [True] * 8 + [False] * 2)[0]


def test_unknown_env():
    with pytest.raises(SystemExit) as e:
        train.check_env_name("cartpole")
    assert e.value.code == 2


def setup_files(tmp_path, monkeypatch, header_env, model_env):
    monkeypatch.setattr(train, "OUT", tmp_path)
    monkeypatch.setattr(train, "HEADER", tmp_path / "policy_weights.h")
    (tmp_path / "policy_weights.h").write_text(f'#pragma once\n#define POLICY_ENV "{header_env}"\n')
    if model_env:
        (tmp_path / f"{model_env}.zip").write_bytes(b"")


def test_hil_refuses_mismatched_header(tmp_path, monkeypatch, capsys):
    setup_files(tmp_path, monkeypatch, "pendulum", "cartpole_balance")
    monkeypatch.setattr("sys.argv", ["hil.py", "/dev/null-port", "cartpole_balance"])
    assert hil.main() == 2  # returns before opening the port
    assert "pendulum" in capsys.readouterr().err


def test_hil_missing_model(tmp_path, monkeypatch, capsys):
    setup_files(tmp_path, monkeypatch, "cartpole_swingup", None)
    monkeypatch.setattr("sys.argv", ["hil.py", "/dev/null-port", "cartpole_swingup"])
    assert hil.main() == 2
    assert "make train ENV=cartpole_swingup" in capsys.readouterr().err
