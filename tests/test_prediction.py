import subprocess
from pathlib import Path

import pandas as pd
import pytest

import prediction
from prediction import PredictionInputError, calculate_descriptors, parse_input_text


def test_parse_input_requires_two_columns_unique_names_and_limits():
    parsed = parse_input_text("CCO mol_a\nCCN mol_b\n", max_molecules=2)
    assert parsed.to_dict("records") == [
        {"smiles": "CCO", "molecule_name": "mol_a"},
        {"smiles": "CCN", "molecule_name": "mol_b"},
    ]
    for invalid in ("", "CCO only\nCCN only\n", "CCO\n", "CCO a\nCCN a\n"):
        with pytest.raises(PredictionInputError):
            parse_input_text(invalid)
    with pytest.raises(PredictionInputError):
        parse_input_text("CCO one\nCCN two\n", max_molecules=1)
    with pytest.raises(PredictionInputError):
        parse_input_text("CCO id\n", max_bytes=3)


def _fixture_app_dir(tmp_path):
    (tmp_path / "PaDEL-Descriptor").mkdir()
    (tmp_path / "PaDEL-Descriptor" / "PaDEL-Descriptor.jar").write_bytes(b"fixture only")
    (tmp_path / "PaDEL-Descriptor" / "PubchemFingerprinter.xml").write_text("<fixture/>")
    (tmp_path / "descriptor_list.csv").write_text("feat_a,feat_b\n1,2\n")
    return tmp_path


def test_descriptors_use_isolated_workspace_and_join_by_generated_id(monkeypatch, tmp_path):
    app_dir = _fixture_app_dir(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        out = Path(command[command.index("-file") + 1])
        input_dir = Path(command[command.index("-dir") + 1])
        input_lines = (input_dir / "molecules.smi").read_text().splitlines()
        ids = [line.split("\t")[1] for line in input_lines]
        pd.DataFrame({"Name": ids[::-1], "feat_a": [20, 10], "feat_b": [2, 1]}).to_csv(out, index=False)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(prediction.subprocess, "run", fake_run)
    rows = parse_input_text("CCO first\nCCN second\n")
    first = calculate_descriptors(rows, app_dir)
    second = calculate_descriptors(rows, app_dir)

    assert first["molecule_name"].tolist() == ["first", "second"]
    assert first[["feat_a", "feat_b"]].values.tolist() == [[10, 1], [20, 2]]
    assert second.equals(first)
    assert len(calls) == 2
    assert calls[0][0][calls[0][0].index("-file") + 1] != calls[1][0][calls[1][0].index("-file") + 1]
    assert calls[0][1]["check"] is True and calls[0][1]["timeout"] == prediction.DESCRIPTOR_TIMEOUT_SECONDS
    assert "-threads" in calls[0][0] and calls[0][0][calls[0][0].index("-threads") + 1] == "1"


def test_descriptor_process_errors_fail_closed(monkeypatch, tmp_path):
    app_dir = _fixture_app_dir(tmp_path)
    rows = parse_input_text("CCO mol_a\n")

    def failed(*_args, **_kwargs):
        raise subprocess.CalledProcessError(2, ["java"], stderr="fixture error")

    monkeypatch.setattr(prediction.subprocess, "run", failed)
    with pytest.raises(PredictionInputError, match="calculation failed"):
        calculate_descriptors(rows, app_dir)

    def timed_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["java"], 1)

    monkeypatch.setattr(prediction.subprocess, "run", timed_out)
    with pytest.raises(PredictionInputError, match="exceeded"):
        calculate_descriptors(rows, app_dir, timeout=1)


@pytest.mark.parametrize("output_kind, message", [
    ("missing", "without producing"),
    ("bad_ids", "IDs do not match"),
    ("duplicate_ids", "duplicate molecule IDs"),
    ("missing_feature", "missing 1 model features"),
    ("malformed", "valid CSV"),
])
def test_bad_or_stale_descriptor_output_fails_closed(monkeypatch, tmp_path, output_kind, message):
    app_dir = _fixture_app_dir(tmp_path)
    rows = parse_input_text("CCO mol_a\n")

    def fake_run(command, **kwargs):
        out = Path(command[command.index("-file") + 1])
        if output_kind == "missing":
            return subprocess.CompletedProcess(command, 0, "", "")
        if output_kind == "malformed":
            out.write_text("")
        elif output_kind == "bad_ids":
            pd.DataFrame({"Name": ["stale_id"], "feat_a": [1], "feat_b": [2]}).to_csv(out, index=False)
        elif output_kind == "duplicate_ids":
            pd.DataFrame({"Name": ["request_0000", "request_0000"], "feat_a": [1, 2], "feat_b": [2, 3]}).to_csv(out, index=False)
        elif output_kind == "missing_feature":
            pd.DataFrame({"Name": ["request_0000"], "feat_a": [1]}).to_csv(out, index=False)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(prediction.subprocess, "run", fake_run)
    with pytest.raises(PredictionInputError, match=message):
        calculate_descriptors(rows, app_dir)
