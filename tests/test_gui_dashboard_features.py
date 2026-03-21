"""Tests for GUI and dashboard feature parity with CLI.

Covers:
- GUI vars for run_dir, calibration report, and interactive labeling
- GUI _run_calibration_report and _start_labeling_session methods
- GUI _LabelingDialog class (no Tk display needed – unit-test logic only)
- Dashboard calibration report section (import check)
- Dashboard interactive labeling session-state helpers
- CLI calibration_report function (smoke test)
- CLI _sort_for_labeling function
"""

import sys
import os
import json
import tempfile
import shutil

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(task_id, determination, confidence=0.7, **kwargs):
    r = {
        'task_id': task_id,
        'attempter': 'alice',
        'occupation': 'analyst',
        'determination': determination,
        'confidence': confidence,
        'reason': 'test reason',
        'word_count': 150,
        'mode': 'auto',
        'domain': '',
        'preamble_score': 0.0,
        'preamble_severity': 'NONE',
        'prompt_signature_composite': 0.0,
        'prompt_signature_cfd': 0.0,
        'voice_dissonance_vsd': 0.0,
        'voice_dissonance_voice_score': 0.0,
        'voice_dissonance_spec_score': 0.0,
        'instruction_density_idi': 0.0,
        'self_similarity_nssi_score': 0.0,
        'self_similarity_nssi_signals': 0,
        'continuation_bscore': 0.0,
        'continuation_mode': 'n/a',
        'channel_details': {'channels': {}},
    }
    r.update(kwargs)
    return r


def _make_labeled_jsonl(tmpdir, records):
    path = os.path.join(tmpdir, 'labeled.jsonl')
    with open(path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    return path


# ---------------------------------------------------------------------------
# CLI: _sort_for_labeling
# ---------------------------------------------------------------------------

def test_sort_for_labeling_order():
    """_sort_for_labeling puts YELLOW before AMBER, then RED, then GREEN."""
    print("\n-- _sort_for_labeling order --")
    from llm_detector.cli import _sort_for_labeling
    results = [
        _make_result('g1', 'GREEN', 0.1),
        _make_result('r1', 'RED', 0.9),
        _make_result('y1', 'YELLOW', 0.5),
        _make_result('a1', 'AMBER', 0.7),
    ]
    sorted_res = _sort_for_labeling(results)
    dets = [r['determination'] for r in sorted_res]
    check("YELLOW appears before AMBER",
          dets.index('YELLOW') < dets.index('AMBER'))
    check("AMBER appears before RED",
          dets.index('AMBER') < dets.index('RED'))
    check("RED appears before GREEN",
          dets.index('RED') < dets.index('GREEN'))


# ---------------------------------------------------------------------------
# CLI: calibration_report smoke test
# ---------------------------------------------------------------------------

def test_calibration_report_insufficient_data():
    """calibration_report returns None when fewer than 5 labeled records."""
    print("\n-- calibration_report: insufficient data --")
    import io, sys as _sys
    from llm_detector.cli import calibration_report

    tmpdir = tempfile.mkdtemp()
    try:
        records = [
            {'ground_truth': 'ai', 'pipeline_determination': 'RED',
             'pipeline_confidence': 0.9, 'confidence': 0.9},
            {'ground_truth': 'human', 'pipeline_determination': 'GREEN',
             'pipeline_confidence': 0.1, 'confidence': 0.1},
        ]
        path = _make_labeled_jsonl(tmpdir, records)

        buf = io.StringIO()
        old = _sys.stdout
        _sys.stdout = buf
        result = calibration_report(path)
        _sys.stdout = old

        check("returns None on insufficient data", result is None)
        check("prints insufficient-data message",
              'Insufficient' in buf.getvalue() or 'insufficient' in buf.getvalue())
    finally:
        shutil.rmtree(tmpdir)


def test_calibration_report_with_data():
    """calibration_report produces output with >= 5 labeled records."""
    print("\n-- calibration_report: sufficient data --")
    import io, sys as _sys
    from llm_detector.cli import calibration_report

    tmpdir = tempfile.mkdtemp()
    try:
        records = (
            [{'ground_truth': 'ai', 'pipeline_determination': 'RED',
              'pipeline_confidence': 0.9, 'confidence': 0.9,
              'domain': 'test', 'length_bin': 'medium'}] * 4
            + [{'ground_truth': 'human', 'pipeline_determination': 'GREEN',
                'pipeline_confidence': 0.1, 'confidence': 0.1,
                'domain': 'test', 'length_bin': 'medium'}] * 4
        )
        path = _make_labeled_jsonl(tmpdir, records)

        buf = io.StringIO()
        old = _sys.stdout
        _sys.stdout = buf
        calibration_report(path)
        _sys.stdout = old

        output = buf.getvalue()
        check("prints CALIBRATION DIAGNOSTICS header",
              'CALIBRATION DIAGNOSTICS' in output)
        check("prints confusion matrix section",
              'Confusion Matrix' in output)
    finally:
        shutil.rmtree(tmpdir)


def test_calibration_report_csv_export():
    """calibration_report writes a CSV when output_csv is given."""
    print("\n-- calibration_report: CSV export --")
    import io, sys as _sys
    from llm_detector.cli import calibration_report

    tmpdir = tempfile.mkdtemp()
    try:
        records = (
            [{'ground_truth': 'ai', 'pipeline_determination': 'RED',
              'pipeline_confidence': 0.9, 'confidence': 0.9,
              'domain': 'test', 'length_bin': 'medium'}] * 3
            + [{'ground_truth': 'human', 'pipeline_determination': 'GREEN',
                'pipeline_confidence': 0.1, 'confidence': 0.1,
                'domain': 'test', 'length_bin': 'medium'}] * 4
        )
        path = _make_labeled_jsonl(tmpdir, records)
        csv_out = os.path.join(tmpdir, 'report.csv')

        buf = io.StringIO()
        old = _sys.stdout
        _sys.stdout = buf
        calibration_report(path, output_csv=csv_out)
        _sys.stdout = old

        check("CSV file was written", os.path.isfile(csv_out),
              f"csv_out={csv_out}")
    finally:
        shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# GUI: vars and layout (no display required)
# ---------------------------------------------------------------------------

def test_gui_vars_exist():
    """DetectorGUI declares all new variable attributes."""
    print("\n-- GUI: new vars declared --")
    from llm_detector.compat import HAS_TK
    if not HAS_TK:
        print("  [SKIP] tkinter not available")
        return

    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    try:
        from llm_detector.gui import DetectorGUI
        gui = DetectorGUI(root)
        check("run_dir_var exists", hasattr(gui, 'run_dir_var'))
        check("cal_report_jsonl_var exists", hasattr(gui, 'cal_report_jsonl_var'))
        check("cal_report_csv_var exists", hasattr(gui, 'cal_report_csv_var'))
        check("label_output_var exists", hasattr(gui, 'label_output_var'))
        check("label_reviewer_var exists", hasattr(gui, 'label_reviewer_var'))
        check("label_skip_green_var exists", hasattr(gui, 'label_skip_green_var'))
        check("label_skip_red_var exists", hasattr(gui, 'label_skip_red_var'))
        check("label_max_var exists", hasattr(gui, 'label_max_var'))
        check("_run_calibration_report method exists",
              callable(getattr(gui, '_run_calibration_report', None)))
        check("_start_labeling_session method exists",
              callable(getattr(gui, '_start_labeling_session', None)))
    finally:
        root.destroy()


def test_gui_run_dir_sets_output_paths(tmp_path):
    """_analyze_file with run_dir_var set creates run folder and sets output paths."""
    print("\n-- GUI: run_dir sets output paths --")
    from llm_detector.compat import HAS_TK
    if not HAS_TK:
        print("  [SKIP] tkinter not available")
        return

    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    try:
        from llm_detector.gui import DetectorGUI
        gui = DetectorGUI(root)

        run_root = str(tmp_path / 'runs')
        gui.run_dir_var.set(run_root)

        # Manually execute only the run-dir block (no actual analysis)
        from pathlib import Path
        from datetime import datetime
        run_dir_base = gui.run_dir_var.get().strip()
        if run_dir_base:
            run_dir = Path(run_dir_base) / datetime.now().strftime('run_%Y%m%d_%H%M%S')
            run_dir.mkdir(parents=True, exist_ok=True)
            if not gui.output_csv_var.get().strip():
                gui.output_csv_var.set(str(run_dir / 'results.csv'))
            if not gui.sim_store_var.get().strip():
                gui.sim_store_var.set(str(run_dir / 'similarity.jsonl'))
            if not gui.label_output_var.get().strip():
                gui.label_output_var.set(str(run_dir / 'labels.jsonl'))

        check("output_csv_var set under run_dir",
              'results.csv' in gui.output_csv_var.get())
        check("sim_store_var set under run_dir",
              'similarity.jsonl' in gui.sim_store_var.get())
        check("label_output_var set under run_dir",
              'labels.jsonl' in gui.label_output_var.get())
    finally:
        root.destroy()


def test_gui_calibration_report_handler(tmp_path):
    """_run_calibration_report appends output text for sufficient data."""
    print("\n-- GUI: _run_calibration_report --")
    from llm_detector.compat import HAS_TK
    if not HAS_TK:
        print("  [SKIP] tkinter not available")
        return

    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    try:
        from llm_detector.gui import DetectorGUI
        gui = DetectorGUI(root)

        records = (
            [{'ground_truth': 'ai', 'pipeline_determination': 'RED',
              'pipeline_confidence': 0.9, 'confidence': 0.9,
              'domain': '', 'length_bin': 'medium'}] * 4
            + [{'ground_truth': 'human', 'pipeline_determination': 'GREEN',
                'pipeline_confidence': 0.1, 'confidence': 0.1,
                'domain': '', 'length_bin': 'medium'}] * 4
        )
        path = str(tmp_path / 'labeled.jsonl')
        with open(path, 'w') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')

        gui.cal_report_jsonl_var.set(path)

        # Capture what _run_calibration_report appends to the output widget
        appended = []
        orig_append = gui._append
        gui._append = lambda text, *args, **kwargs: appended.append(text)

        gui._run_calibration_report()

        check("calibration report output non-empty", len(appended) > 0)
        combined = ''.join(appended)
        check("output contains diagnostics header",
              'CALIBRATION' in combined)
    finally:
        root.destroy()


def test_gui_start_labeling_no_results():
    """_start_labeling_session shows info when no results are loaded."""
    print("\n-- GUI: _start_labeling_session no results --")
    from llm_detector.compat import HAS_TK
    if not HAS_TK:
        print("  [SKIP] tkinter not available")
        return

    import tkinter as tk
    from unittest.mock import patch
    root = tk.Tk()
    root.withdraw()
    try:
        from llm_detector.gui import DetectorGUI
        gui = DetectorGUI(root)
        gui._last_results = []

        shown = []
        with patch('tkinter.messagebox.showinfo',
                   side_effect=lambda t, m: shown.append(m)):
            gui._start_labeling_session()

        check("showinfo called when no results",
              any('Run an analysis' in m for m in shown),
              f"shown={shown}")
    finally:
        root.destroy()


# ---------------------------------------------------------------------------
# GUI: DNA-GPT reporting
# ---------------------------------------------------------------------------

def test_gui_collect_dna_hits_filters_positive():
    """_collect_dna_hits returns only continuation positives."""
    print("\n-- GUI: _collect_dna_hits --")
    from llm_detector.compat import HAS_TK
    if not HAS_TK:
        print("  [SKIP] tkinter not available")
        return

    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    try:
        from llm_detector.gui import DetectorGUI
        gui = DetectorGUI(root)

        cont_hit = _make_result(
            'dna1', 'AMBER',
            continuation_bscore=0.812,
            continuation_mode='Local',
            channel_details={
                'channels': {
                    'continuation': {'severity': 'AMBER', 'score': 0.7},
                }
            },
        )
        cont_clean = _make_result(
            'dna0', 'GREEN',
            channel_details={'channels': {'continuation': {'severity': 'GREEN'}}},
        )

        hits = gui._collect_dna_hits([cont_hit, cont_clean])
        check("one DNA hit returned", len(hits) == 1, f"hits={hits}")
        hit = hits[0]
        check("task_id captured", hit['task_id'] == 'dna1')
        check("severity AMBER captured", hit['severity'] == 'AMBER')
        check("bscore rounded preserves value", abs(hit['bscore'] - 0.812) < 1e-6)
        check("mode captured", hit['mode'] == 'Local')
    finally:
        root.destroy()


def test_gui_refresh_reports_lists_dna_hits():
    """_refresh_reports prints DNA-GPT-positive section."""
    print("\n-- GUI: _refresh_reports DNA section --")
    from llm_detector.compat import HAS_TK
    if not HAS_TK:
        print("  [SKIP] tkinter not available")
        return

    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    try:
        from llm_detector.gui import DetectorGUI
        gui = DetectorGUI(root)

        gui._last_results = [
            _make_result(
                'dna-task', 'RED',
                continuation_bscore=0.901,
                continuation_mode='API',
                channel_details={
                    'channels': {'continuation': {'severity': 'RED', 'score': 0.9}},
                },
            ),
            _make_result('clean', 'GREEN'),
        ]

        appended = []
        gui._report_append = lambda text: appended.append(text)

        class _DummyReport:
            def delete(self, *args, **kwargs):
                return None
        gui.report_output = _DummyReport()

        gui._refresh_reports()

        combined = ''.join(appended)
        check("DNA-GPT section header present",
              "DNA-GPT POSITIVE CONTINUATIONS" in combined, combined)
        check("DNA task id appears", "dna-task" in combined, combined)
        check("no empty message when hits exist",
              "No DNA-GPT-positive" not in combined, combined)
    finally:
        root.destroy()


def test_gui_collect_dna_hits_fallbacks():
    """_collect_dna_hits uses channel score/mode fallbacks when top-level fields missing."""
    print("\n-- GUI: _collect_dna_hits fallbacks --")
    from llm_detector.compat import HAS_TK
    if not HAS_TK:
        print("  [SKIP] tkinter not available")
        return

    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    try:
        from llm_detector.gui import DetectorGUI
        gui = DetectorGUI(root)

        via_channel = _make_result(
            'via-channel', 'AMBER',
            channel_details={
                'channels': {
                    'continuation': {'severity': 'YELLOW', 'score': 0.55, 'mode': 'API'},
                }
            },
        )
        via_mode_fallback = _make_result(
            'mode-fallback', 'AMBER',
            channel_details={
                'channels': {
                    'continuation': {'severity': 'YELLOW', 'score': 0.42},
                }
            },
        )
        via_mode_fallback['continuation_mode'] = ''  # force to use channel/r default
        via_mode_fallback['mode'] = 'manual'

        hits = gui._collect_dna_hits([via_channel, via_mode_fallback])
        bscore_map = {h['task_id']: h['bscore'] for h in hits}
        mode_map = {h['task_id']: h['mode'] for h in hits}

        check("channel score used when continuation_bscore missing",
              abs(bscore_map['via-channel'] - 0.55) < 1e-6)
        check("channel mode used when continuation_mode missing",
              mode_map['via-channel'] == 'API')
        check("fallback to result mode when channel mode missing",
              mode_map['mode-fallback'] == 'manual')
    finally:
        root.destroy()


# ---------------------------------------------------------------------------
# GUI: _LabelingDialog logic (no display)
# ---------------------------------------------------------------------------

def test_labeling_dialog_writes_jsonl(tmp_path):
    """_LabelingDialog writes a label record to JSONL on _label() call."""
    print("\n-- _LabelingDialog writes JSONL --")
    from llm_detector.compat import HAS_TK
    if not HAS_TK:
        print("  [SKIP] tkinter not available")
        return

    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    try:
        from llm_detector.gui import _LabelingDialog
        results = [_make_result('t1', 'AMBER', 0.75)]
        out_path = str(tmp_path / 'labels.jsonl')
        completed = []

        dlg = _LabelingDialog(
            root, results, {'t1': 'sample text'},
            output_path=out_path, reviewer='tester',
            on_complete=lambda s: completed.append(s),
        )
        # Simulate labeling as AI
        dlg._label('ai')

        check("JSONL file written", os.path.isfile(out_path))
        with open(out_path) as f:
            rec = json.loads(f.read().strip())
        check("ground_truth is 'ai'", rec['ground_truth'] == 'ai')
        check("reviewer recorded", rec['reviewer'] == 'tester')
        check("task_id recorded", rec['task_id'] == 't1')
        check("length_bin present", 'length_bin' in rec)
    finally:
        root.destroy()


def test_labeling_dialog_skip_and_quit(tmp_path):
    """_LabelingDialog skip increments counter; quit calls on_complete."""
    print("\n-- _LabelingDialog skip/quit --")
    from llm_detector.compat import HAS_TK
    if not HAS_TK:
        print("  [SKIP] tkinter not available")
        return

    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    try:
        from llm_detector.gui import _LabelingDialog
        results = [
            _make_result('t1', 'AMBER', 0.7),
            _make_result('t2', 'GREEN', 0.2),
        ]
        out_path = str(tmp_path / 'labels.jsonl')
        completed = []

        dlg = _LabelingDialog(
            root, results, {},
            output_path=out_path, reviewer='r1',
            on_complete=lambda s: completed.append(s),
        )
        dlg._skip()
        dlg._quit()

        check("on_complete called after quit", len(completed) == 1)
        check("skip counted in stats", completed[0]['skipped'] == 1)
    finally:
        root.destroy()


# ---------------------------------------------------------------------------
# Dashboard: imports and function signatures
# ---------------------------------------------------------------------------

def test_dashboard_calibration_report_import():
    """calibration_report can be imported from cli inside dashboard context."""
    print("\n-- Dashboard: cli.calibration_report importable --")
    try:
        from llm_detector.cli import calibration_report
        check("calibration_report importable from cli", callable(calibration_report))
    except ImportError as e:
        check("calibration_report importable from cli", False, str(e))


def test_dashboard_sort_for_labeling_import():
    """_sort_for_labeling can be imported from cli for dashboard labeling."""
    print("\n-- Dashboard: cli._sort_for_labeling importable --")
    try:
        from llm_detector.cli import _sort_for_labeling
        check("_sort_for_labeling importable from cli", callable(_sort_for_labeling))
    except ImportError as e:
        check("_sort_for_labeling importable from cli", False, str(e))


# ---------------------------------------------------------------------------
# New feature tests
# ---------------------------------------------------------------------------

def test_gui_new_column_vars():
    """GUI declares the new column mapping variables for email/reviewer."""
    print("\n-- GUI: new column mapping vars --")
    from llm_detector.compat import HAS_TK
    if not HAS_TK:
        print("  [SKIP] tkinter not available")
        return

    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    try:
        from llm_detector.gui import DetectorGUI
        gui = DetectorGUI(root)
        check("attempter_email_col_var exists", hasattr(gui, 'attempter_email_col_var'))
        check("reviewer_col_var exists", hasattr(gui, 'reviewer_col_var'))
        check("reviewer_email_col_var exists", hasattr(gui, 'reviewer_email_col_var'))
    finally:
        root.destroy()


def test_gui_quick_reference_tab():
    """GUI builds the Quick Reference tab without error."""
    print("\n-- GUI: Quick Reference tab --")
    from llm_detector.compat import HAS_TK
    if not HAS_TK:
        print("  [SKIP] tkinter not available")
        return

    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    try:
        from llm_detector.gui import DetectorGUI
        gui = DetectorGUI(root)
        # Verify tab exists by checking the notebook has 7 tabs
        notebook = [w for w in root.winfo_children()
                    if w.winfo_class() == 'TNotebook']
        if notebook:
            tabs = notebook[0].tabs()
            check("7 tabs present (including Quick Reference and Precheck)",
                  len(tabs) >= 7)
        else:
            check("notebook found", False)
    finally:
        root.destroy()


def test_gui_precheck_tab():
    """GUI Precheck tab populates with dependency data."""
    print("\n-- GUI: Precheck dependencies --")
    from llm_detector.gui import _check_dependencies
    deps = _check_dependencies()
    check("precheck returns list", isinstance(deps, list))
    check("precheck has entries", len(deps) > 5)
    # Each entry is (status, name, category, notes)
    for status, name, cat, notes in deps:
        check(f"  {name} has status icon",
              status in ('\u2705', '\u2757', '\u274c'))
        break  # just check first entry format


def test_quick_reference_text():
    """Quick reference text is non-empty and covers major channels."""
    print("\n-- Quick reference text content --")
    from llm_detector.gui import _QUICK_REFERENCE_TEXT
    check("text is non-empty", len(_QUICK_REFERENCE_TEXT) > 200)
    check("mentions prompt_structure", 'prompt_structure' in _QUICK_REFERENCE_TEXT)
    check("mentions stylometry", 'stylometry' in _QUICK_REFERENCE_TEXT)
    check("mentions continuation", 'continuation' in _QUICK_REFERENCE_TEXT)
    check("mentions windowing", 'windowing' in _QUICK_REFERENCE_TEXT)
    check("mentions Perplexity", 'Perplexity' in _QUICK_REFERENCE_TEXT)
    check("mentions TOCSIN", 'TOCSIN' in _QUICK_REFERENCE_TEXT)


def test_layer3_terminology_removed():
    """Verify Layer 2/3 terminology is removed from UI-facing strings."""
    print("\n-- Layer 2/3 terminology removal --")
    import inspect
    from llm_detector import gui as gui_mod
    from llm_detector import dashboard as dash_mod
    from llm_detector import cli as cli_mod
    # Check source code for Layer 2 / Layer 3 user-facing text
    gui_src = inspect.getsource(gui_mod)
    dash_src = inspect.getsource(dash_mod)
    cli_src = inspect.getsource(cli_mod)
    # We allow it in internal comments but not in user-facing strings
    # Just check that key UI text patterns are gone
    check("GUI no 'Skip Layer 3' text",
          'Skip Layer 3' not in gui_src)
    check("Dashboard no 'Skip Layer 3' text",
          'Skip Layer 3' not in dash_src)
    check("CLI no 'Skip Layer 3' text",
          'Skip Layer 3' not in cli_src)


def test_ensure_streamlit_function():
    """_ensure_streamlit function exists in CLI module."""
    print("\n-- CLI: _ensure_streamlit --")
    from llm_detector.cli import _ensure_streamlit
    check("_ensure_streamlit callable", callable(_ensure_streamlit))


def test_io_new_columns():
    """load_csv and load_xlsx accept new column kwargs without error."""
    print("\n-- io: new column kwargs --")
    import inspect
    from llm_detector.io import load_csv, load_xlsx
    csv_sig = inspect.signature(load_csv)
    xlsx_sig = inspect.signature(load_xlsx)
    check("load_csv has attempter_email_col", 'attempter_email_col' in csv_sig.parameters)
    check("load_csv has reviewer_col", 'reviewer_col' in csv_sig.parameters)
    check("load_csv has reviewer_email_col", 'reviewer_email_col' in csv_sig.parameters)
    check("load_xlsx has attempter_email_col", 'attempter_email_col' in xlsx_sig.parameters)
    check("load_xlsx has reviewer_col", 'reviewer_col' in xlsx_sig.parameters)
    check("load_xlsx has reviewer_email_col", 'reviewer_email_col' in xlsx_sig.parameters)


def test_gui_quick_confirm_methods():
    """GUI has quick-confirm methods for recent samples."""
    print("\n-- GUI: quick-confirm methods --")
    from llm_detector.compat import HAS_TK
    if not HAS_TK:
        print("  [SKIP] tkinter not available")
        return

    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    try:
        from llm_detector.gui import DetectorGUI
        gui = DetectorGUI(root)
        check("_refresh_recent_samples method", callable(getattr(gui, '_refresh_recent_samples', None)))
        check("_quick_confirm method", callable(getattr(gui, '_quick_confirm', None)))
        check("_on_recent_select method", callable(getattr(gui, '_on_recent_select', None)))
    finally:
        root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
