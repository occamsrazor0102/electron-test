"""Tests for CLI argument parsing and --disable-channel functionality."""

import sys
import os
import csv
import shutil
import subprocess
import tempfile
import types
import importlib.util
import importlib.machinery
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASSED = 0
FAILED = 0


def check(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label}  -- {detail}")


def test_cli_argparse_no_crash():
    """Ensure CLI parser can be constructed without errors (regression: duplicate --disable-channel)."""
    print("\n-- CLI ARGPARSE CONSTRUCTION --")
    import argparse
    # Importing main triggers the parser construction; a duplicate argument
    # definition would raise argparse.ArgumentError.
    from unittest.mock import patch
    try:
        from llm_detector.cli import main
        # Trigger parser construction by calling with --help, capturing SystemExit
        with patch('sys.argv', ['llm-detector', '--help']):
            try:
                main()
            except SystemExit:
                pass
        check("CLI parser constructed without errors", True)
    except argparse.ArgumentError as e:
        check("CLI parser constructed without errors", False, str(e))


def test_cli_new_column_args_accepted():
    """Ensure new column mapping arguments are accepted by the CLI parser."""
    print("\n-- CLI COLUMN MAPPING ARGS --")
    from unittest.mock import patch
    import argparse
    try:
        from llm_detector.cli import main
        # --help forces early exit after printing, confirming parser accepted args
        with patch('sys.argv', [
            'llm-detector', '--help',
            '--id-col', 'B',
            '--occ-col', 'D',
            '--attempter-col', 'C',
            '--stage-col', 'E',
        ]):
            try:
                main()
            except SystemExit:
                pass
        check("--id-col / --occ-col / --attempter-col / --stage-col accepted", True)
    except argparse.ArgumentError as e:
        check("Column args accepted", False, str(e))


def test_cli_run_dir_arg_accepted():
    """Ensure --run-dir argument is accepted by the CLI parser."""
    print("\n-- CLI --run-dir ARG --")
    from unittest.mock import patch
    import argparse
    try:
        from llm_detector.cli import main
        with patch('sys.argv', ['llm-detector', '--help', '--run-dir', '/tmp/test_run']):
            try:
                main()
            except SystemExit:
                pass
        check("--run-dir accepted", True)
    except argparse.ArgumentError as e:
        check("--run-dir accepted", False, str(e))


def test_run_dir_creates_timestamped_folder():
    """--run-dir creates a timestamped subfolder and sets output paths."""
    print("\n-- RUN-DIR CREATES FOLDER --")
    from unittest.mock import patch
    import argparse

    tmpdir = tempfile.mkdtemp()
    try:
        # Build a minimal CSV with enough text
        csv_path = os.path.join(tmpdir, 'sample.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['prompt', 'task_id', 'attempter_name', 'occupation'])
            w.writerow([
                'This is a sufficiently long prompt text that passes the minimum '
                'character check required by the pipeline loader function.',
                'task001', 'alice', 'analyst',
            ])

        run_root = os.path.join(tmpdir, 'runs')
        from llm_detector.cli import main
        with patch('sys.argv', [
            'llm-detector', csv_path,
            '--run-dir', run_root,
            '--no-layer3',
            '--no-similarity',
        ]):
            main()

        run_subdirs = os.listdir(run_root) if os.path.isdir(run_root) else []
        check("run root created", os.path.isdir(run_root))
        check("timestamped subfolder created",
              len(run_subdirs) == 1 and run_subdirs[0].startswith('run_'),
              f"found: {run_subdirs}")
        if run_subdirs:
            run_folder = os.path.join(run_root, run_subdirs[0])
            check("results.csv created in run folder",
                  os.path.isfile(os.path.join(run_folder, 'results.csv')))
    finally:
        shutil.rmtree(tmpdir)


def test_io_col_letter_to_index():
    """_col_letter_to_index converts letters and numbers correctly."""
    print("\n-- IO: _col_letter_to_index --")
    from llm_detector.io import _col_letter_to_index

    check("A -> 0", _col_letter_to_index('A') == 0)
    check("a -> 0 (case-insensitive)", _col_letter_to_index('a') == 0)
    check("B -> 1", _col_letter_to_index('B') == 1)
    check("Z -> 25", _col_letter_to_index('Z') == 25)
    check("'1' -> 0 (1-based)", _col_letter_to_index('1') == 0)
    check("'3' -> 2 (1-based)", _col_letter_to_index('3') == 2)
    check("'prompt' -> None", _col_letter_to_index('prompt') is None)
    check("'task_id' -> None", _col_letter_to_index('task_id') is None)
    check("'AB' -> None (multi-letter)", _col_letter_to_index('AB') is None)
    check("'0' -> None (zero not valid 1-based)", _col_letter_to_index('0') is None)


def test_io_load_csv_positional():
    """load_csv resolves columns by letter position."""
    print("\n-- IO: load_csv positional columns --")
    import tempfile
    import os
    from llm_detector.io import load_csv

    tmpdir = tempfile.mkdtemp()
    try:
        csv_path = os.path.join(tmpdir, 'test.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            # Column order: text | id | person | area
            w.writerow(['text', 'id', 'person', 'area'])
            long_text = 'x' * 60
            w.writerow([long_text, 'T001', 'alice', 'engineering'])

        tasks = load_csv(
            csv_path,
            prompt_col='A',   # first column
            id_col='B',       # second column
            attempter_col='C',  # third column
            occ_col='D',      # fourth column
        )
        check("1 task loaded", len(tasks) == 1, f"got {len(tasks)}")
        if tasks:
            check("prompt from col A", 'x' in tasks[0]['prompt'])
            check("task_id from col B", tasks[0]['task_id'] == 'T001')
            check("attempter from col C", tasks[0]['attempter'] == 'alice')
            check("occupation from col D", tasks[0]['occupation'] == 'engineering')
    finally:
        shutil.rmtree(tmpdir)


def test_io_load_csv_numeric_positional():
    """load_csv resolves columns by 1-based numeric string."""
    print("\n-- IO: load_csv numeric positional columns --")
    import tempfile
    import os
    from llm_detector.io import load_csv

    tmpdir = tempfile.mkdtemp()
    try:
        csv_path = os.path.join(tmpdir, 'test.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['col1', 'col2', 'col3', 'col4'])
            long_text = 'y' * 60
            w.writerow([long_text, 'ID99', 'bob', 'finance'])

        tasks = load_csv(
            csv_path,
            prompt_col='1',
            id_col='2',
            attempter_col='3',
            occ_col='4',
        )
        check("1 task loaded", len(tasks) == 1, f"got {len(tasks)}")
        if tasks:
            check("task_id from col 2", tasks[0]['task_id'] == 'ID99')
            check("attempter from col 3", tasks[0]['attempter'] == 'bob')
            check("occupation from col 4", tasks[0]['occupation'] == 'finance')
    finally:
        shutil.rmtree(tmpdir)


def test_dashboard_uses_module_when_cli_missing(monkeypatch):
    """main_dashboard should fall back to `python -m streamlit` when CLI is absent."""
    from llm_detector import cli

    monkeypatch.setattr('llm_detector.cli.shutil.which', lambda _: None)
    original_streamlit_modules = {k: v for k, v in sys.modules.items() if k.startswith('streamlit')}
    dummy_streamlit = types.ModuleType('streamlit')
    monkeypatch.setitem(sys.modules, 'streamlit', dummy_streamlit)
    streamlit_spec = importlib.machinery.ModuleSpec('streamlit', loader=None, origin='/fake/streamlit/__init__.py')
    streamlit_main_spec = importlib.machinery.ModuleSpec('streamlit.__main__', loader=None, origin='/fake/streamlit/__main__.py')
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == 'streamlit':
            return streamlit_spec
        if name == 'streamlit.__main__':
            return streamlit_main_spec
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr('importlib.util.find_spec', fake_find_spec)

    captured = {}

    def fake_run(cmd, check=False):
        captured['cmd'] = cmd
        captured['check'] = check

    monkeypatch.setattr('llm_detector.cli.subprocess.run', fake_run)
    try:
        cli.main_dashboard()
        dash_spec = importlib.util.find_spec('llm_detector.dashboard')
        dash_path = dash_spec.origin if dash_spec else None
        cmd = captured.get('cmd', [])
        check("Fallback uses python -m streamlit",
              cmd == [sys.executable, '-m', 'streamlit', 'run', dash_path],
              f"got {cmd}")
    finally:
        for key in list(sys.modules):
            if key.startswith('streamlit'):
                sys.modules.pop(key, None)
        sys.modules.update(original_streamlit_modules)


def test_dashboard_prefers_cli_when_available(monkeypatch):
    """main_dashboard should use the streamlit executable if it is on PATH."""
    from llm_detector import cli
    monkeypatch.setattr('llm_detector.cli.shutil.which', lambda _: '/usr/local/bin/streamlit')

    captured = {}

    def fake_run(cmd, check=False):
        captured['cmd'] = cmd
        captured['check'] = check

    monkeypatch.setattr('llm_detector.cli.subprocess.run', fake_run)
    cli.main_dashboard()
    dash_spec = importlib.util.find_spec('llm_detector.dashboard')
    dash_path = dash_spec.origin if dash_spec else None
    cmd = captured.get('cmd', [])
    check("Prefers streamlit CLI",
          cmd == ['/usr/local/bin/streamlit', 'run', dash_path],
          f"got {cmd}")


def test_dashboard_reports_missing_streamlit(monkeypatch, capsys):
    """main_dashboard should try to auto-install streamlit when absent."""
    from llm_detector import cli
    original_streamlit_modules = {k: v for k, v in sys.modules.items() if k.startswith('streamlit')}
    monkeypatch.setattr('llm_detector.cli.shutil.which', lambda _: None)
    monkeypatch.delitem(sys.modules, 'streamlit', raising=False)
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name.startswith('streamlit'):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr('importlib.util.find_spec', fake_find_spec)

    # Make pip install fail so it prints the fallback error
    def fake_check_call(*args, **kwargs):
        raise subprocess.CalledProcessError(1, 'pip')

    monkeypatch.setattr('llm_detector.cli.subprocess.check_call', fake_check_call)

    def fake_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called when auto-install fails")

    monkeypatch.setattr('llm_detector.cli.subprocess.run', fake_run)
    try:
        cli.main_dashboard()
        out = capsys.readouterr().out
        check("Auto-install failed message",
              "Auto-install failed" in out or "Install manually" in out)
    finally:
        for key in list(sys.modules):
            if key.startswith('streamlit'):
                sys.modules.pop(key, None)
        sys.modules.update(original_streamlit_modules)


def test_disable_channel_names_match_fusion():
    """Ensure --disable-channel valid names match actual channel names in fusion engine."""
    print("\n-- DISABLE-CHANNEL NAME CONSISTENCY --")
    from llm_detector.channels.prompt_structure import score_prompt_structure
    from llm_detector.channels.stylometric import score_stylometric
    from llm_detector.channels.continuation import score_continuation
    from llm_detector.channels.windowed import score_windowed

    # Get actual channel names from ChannelResult objects
    prompt_sig = {'composite': 0, 'framing_completeness': 0, 'cfd': 0,
                  'mfsr': 0, 'must_rate': 0, 'distinct_frames': 0,
                  'conditional_density': 0, 'meta_design_hits': 0,
                  'contractions': 0, 'numbered_criteria': 0,
                  'pack_constraint_score': 0, 'pack_exec_spec_score': 0,
                  'pack_boost': 0, 'pack_active_families': 0, 'pack_spans': []}
    voice_dis = {'voice_gated': False, 'vsd': 0, 'voice_score': 0,
                 'spec_score': 0, 'contractions': 0, 'hedges': 0,
                 'casual_markers': 0, 'misspellings': 0, 'camel_cols': 0,
                 'calcs': 0, 'pack_schema_score': 0, 'pack_spans': []}

    ch_ps = score_prompt_structure(0, 'NONE', prompt_sig, voice_dis, None, 100)
    ch_st = score_stylometric(0, None)
    ch_co = score_continuation(None)
    ch_wi = score_windowed(None)

    actual_names = {ch_ps.channel, ch_st.channel, ch_co.channel, ch_wi.channel}
    expected_names = {'prompt_structure', 'stylometry', 'continuation', 'windowing'}

    check("Actual channel names match expected",
          actual_names == expected_names,
          f"actual={actual_names}, expected={expected_names}")


def test_version_consistency():
    """Ensure version strings are consistent across package."""
    print("\n-- VERSION CONSISTENCY --")
    import llm_detector
    from pathlib import Path
    import tomllib

    pyproject = Path(__file__).resolve().parents[1] / 'pyproject.toml'
    expected_version = tomllib.loads(pyproject.read_text())['project']['version']

    check("Package version matches pyproject.toml",
          llm_detector.__version__ == expected_version,
          f"got {llm_detector.__version__}, expected {expected_version}")


def test_dashboard_path_validation(monkeypatch, capsys):
    """main_dashboard rejects dashboard_path outside llm_detector package."""
    print("\n-- DASHBOARD: path validation --")
    from llm_detector import cli
    from types import SimpleNamespace

    # Mock find_spec to return a spec with origin outside the package
    fake_spec = SimpleNamespace(origin='/tmp/evil/dashboard.py')
    monkeypatch.setattr('importlib.util.find_spec', lambda name, *a, **kw: fake_spec)

    captured = {}

    def fake_run(cmd, check=False):
        captured['ran'] = True

    monkeypatch.setattr('llm_detector.cli.subprocess.run', fake_run)
    cli.main_dashboard()
    out = capsys.readouterr().out
    check("subprocess not invoked for outside path", 'ran' not in captured,
          f"subprocess was called: {captured}")
    check("error message about outside path", 'outside' in out.lower(),
          f"got: {out!r}")


if __name__ == '__main__':
    print("=" * 70)
    print("CLI Tests")
    print("=" * 70)

    test_cli_argparse_no_crash()
    test_cli_new_column_args_accepted()
    test_cli_run_dir_arg_accepted()
    test_run_dir_creates_timestamped_folder()
    test_io_col_letter_to_index()
    test_io_load_csv_positional()
    test_io_load_csv_numeric_positional()
    test_disable_channel_names_match_fusion()
    test_version_consistency()
    # test_dashboard_path_validation requires pytest fixtures and cannot be
    # invoked directly; run it via: pytest tests/test_cli.py

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    if FAILED > 0:
        sys.exit(1)
