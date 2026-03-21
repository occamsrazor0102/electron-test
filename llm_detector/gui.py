"""Desktop GUI for the LLM Detection Pipeline."""

import os
import json
import logging
import threading
import subprocess
import importlib.util
import shutil
import sys
from collections import Counter

logger = logging.getLogger(__name__)

from llm_detector.compat import HAS_TK
from llm_detector.pipeline import analyze_prompt
from llm_detector.io import load_xlsx, load_csv
from llm_detector._constants import (
    GROUND_TRUTH_LABELS as _GROUND_TRUTH_LABELS,
    STREAMLIT_MIN_VERSION as _STREAMLIT_MIN_VERSION,
)

if HAS_TK:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from tkinter import font as tkfont

def _real_python():
    """Return the real Python interpreter, even inside a frozen PyInstaller bundle."""
    if getattr(sys, 'frozen', False):
        # Frozen exe — find the system Python instead
        for name in ('python3', 'python'):
            path = shutil.which(name)
            if path:
                return path
        # Last resort: common locations
        for candidate in ('/usr/bin/python3', '/usr/local/bin/python3'):
            if os.path.isfile(candidate):
                return candidate
    return sys.executable


_DET_COLORS = {
    'RED': '#d32f2f',
    'AMBER': '#f57c00',
    'MIXED': '#7b1fa2',
    'YELLOW': '#fbc02d',
    'REVIEW': '#0288d1',
    'GREEN': '#388e3c',
}

_CHANNELS = ['prompt_structure', 'stylometry', 'continuation', 'windowing']

# Maximum number of preamble patterns printed in verbose output to avoid overflow.
_MAX_PREAMBLE_PATTERNS = 20

_DASHBOARD_THEME = {
    'bg': '#f4f7fb',
    'card': '#ffffff',
    'text': '#1f2937',
    'muted': '#6b7280',
    'accent': '#2563eb',
    'accent_light': '#dbeafe',
}

_TASK_ID_DISPLAY_LEN = 24

# ── Quick Reference content ──────────────────────────────────────────────────
_QUICK_REFERENCE_TEXT = """\
═══════════════════════════════════════════════════════════
  QUICK REFERENCE — Detection Pipeline Analyses & Signals
═══════════════════════════════════════════════════════════

CHANNELS (fusion combines these into a final determination)
──────────────────────────────────────────────────────────
  prompt_structure   Rule-based structural analysis: preamble detection,
                     fingerprint matching, prompt-signature scoring (CFD,
                     MFSR, framing), instruction density (IDI), and voice
                     dissonance (VSD).
  stylometry         Statistical stylometric features: function-word ratio,
                     sentence-length dispersion, TTR, avg word length,
                     short-word ratio, masked topical tokens.
  continuation       DNA-GPT continuation analysis: generates LLM
                     continuations and measures B-score, NCD, overlap,
                     conditional surprisal, repeat-4 rate, TTR. Requires
                     API key (Anthropic / OpenAI) or local model.
  windowing          Sliding-window analysis: max/mean window score,
                     variance, hot-span detection, FW trajectory CV,
                     comp trajectory, changepoint detection.

INDIVIDUAL ANALYZERS
──────────────────────────────────────────────────────────
  Preamble           Detects LLM boilerplate openings (e.g. "Sure, here's",
                     "Certainly!", "As an AI…"). Score, severity, hit count.
  Fingerprint        Pattern-matches known LLM output fingerprints.
  Prompt Signature   Composite metric: CFD (constraint-frame density), MFSR,
                     distinct frames, framing completeness, conditional
                     density, meta-design, contractions, must-rate, numbered
                     criteria.
  IDI                Instruction Density Index: imperatives, conditionals,
                     binary specs, missing references, flag count.
  VSD                Voice Dissonance Score: voice × spec score, casual
                     markers, misspellings, hedges, CamelCase columns,
                     calculations, gated voice detection.
  SSI / NSSI         Self-Similarity Index (Normalised): formulaic density,
                     power-adj density, demonstrative density, transition
                     density, scare-quote density, em-dash density,
                     this/the start rate, section depth, sentence-length CV,
                     compression ratio, hapax ratio, structural compression
                     delta.
  DNA-GPT            Continuation-based analysis: B-score, B-score max,
                     NCD, internal overlap, conditional surprisal, repeat-4,
                     TTR, composite, composite variance/stability.
  Perplexity         Token-level surprisal: mean perplexity, burstiness,
                     surprisal variance (first/second half), volatility
                     decay ratio, Binoculars score, compression ratio,
                     zlib-normalised PPL, comp/PPL ratio.
  TOCSIN             Token Cohesiveness: cohesiveness score and std.
  Semantic Resonance AI/human centroid similarity: AI score, human score,
                     delta, determination, confidence.
  Semantic Flow      Cross-sentence semantic coherence (cosine similarity).
  Lexicon Packs      Domain-specific lexicon matching: constraint, exec-spec,
                     schema scores, active families, prompt boost, IDI boost.

POST-PROCESSING
──────────────────────────────────────────────────────────
  Similarity         Within-batch Jaccard similarity to detect near-duplicates.
  Cross-batch        Persistent similarity store (JSONL) to compare across
                     analysis sessions.
  Shadow Model       ML-based classifier trained on confirmed labels;
                     detects disagreements with rule-based determination.
  Calibration        Conformal prediction tables for well-calibrated
                     confidence probabilities.
  Normalization      Unicode obfuscation detection: invisible chars,
                     homoglyphs, attack neutralisation.
  Language Gate      Non-English/non-Latin language support level check.
"""


def _check_dependencies():
    """Return a list of (status_icon, name, category, notes) tuples."""
    checks = []

    def _probe(module_name, display, category, required=True, note_ok='', note_miss=''):
        try:
            ok = importlib.util.find_spec(module_name) is not None
        except (ModuleNotFoundError, ValueError):
            ok = False
        except (ModuleNotFoundError, ValueError):
            ok = False
        if ok:
            checks.append(('\u2705', display, category, note_ok or 'Available'))
        elif required:
            checks.append(('\u274c', display, category, note_miss or 'Missing — required'))
        else:
            checks.append(('\u2757', display, category, note_miss or 'Missing — optional'))

    # Core (required)
    _probe('pandas', 'pandas', 'Core', required=True, note_ok='DataFrame processing')
    _probe('openpyxl', 'openpyxl', 'Core', required=True, note_ok='Excel I/O')

    # NLP (optional but recommended)
    _probe('spacy', 'spacy', 'NLP', required=False, note_miss='Optional — sentenciser will use regex fallback')
    _probe('ftfy', 'ftfy', 'NLP', required=False, note_miss='Optional — text normalisation')
    _probe('sentence_transformers', 'sentence-transformers', 'NLP', required=False,
           note_miss='Optional — semantic resonance and flow disabled')
    _probe('sklearn', 'scikit-learn', 'NLP', required=False,
           note_miss='Optional — ML fusion and semantic similarity disabled')

    # Perplexity (optional)
    _probe('transformers', 'transformers (HuggingFace)', 'Perplexity', required=False,
           note_miss='Optional — perplexity analyser disabled')
    _probe('torch', 'PyTorch', 'Perplexity', required=False,
           note_miss='Optional — perplexity analyser disabled')

    # API continuation (optional)
    _probe('anthropic', 'anthropic SDK', 'API', required=False,
           note_miss='Optional — Anthropic DNA-GPT continuation disabled')
    _probe('openai', 'openai SDK', 'API', required=False,
           note_miss='Optional — OpenAI DNA-GPT continuation disabled')

    # PDF (optional)
    _probe('pypdf', 'pypdf', 'PDF', required=False, note_miss='Optional — PDF ingestion disabled')

    # Web dashboard (optional)
    _probe('streamlit', 'streamlit', 'Web', required=False,
           note_miss='Optional — web dashboard unavailable (auto-install available)')

    # GUI
    _probe('tkinter', 'tkinter', 'GUI', required=False,
           note_miss='Optional — desktop GUI unavailable')

    # Build tools (optional)
    _probe('PyInstaller', 'PyInstaller', 'Build', required=False,
           note_miss='Optional — needed only for building executables')

    return checks


# Map display names from _check_dependencies() to pip install specifiers.
_PIP_INSTALL_MAP = {
    'spacy': 'llm-detector[nlp]',
    'ftfy': 'llm-detector[nlp]',
    'sentence-transformers': 'llm-detector[nlp]',
    'scikit-learn': 'llm-detector[nlp]',
    'transformers (HuggingFace)': 'llm-detector[perplexity]',
    'PyTorch': 'llm-detector[perplexity]',
    'anthropic SDK': 'llm-detector[api]',
    'openai SDK': 'llm-detector[api]',
    'pypdf': 'llm-detector[pdf]',
    'streamlit': 'llm-detector[web]',
    'PyInstaller': 'pyinstaller>=6.0',
}

# Descriptions shown when hovering each notebook tab header.
_TAB_TOOLTIPS = [
    None,  # Analysis tab — self-explanatory
    (
        'Configuration\n\n'
        'Set the API key for DNA-GPT continuation analysis, '
        'choose the LLM provider, tune similarity detection thresholds, '
        'and configure output paths for CSV and HTML reports.'
    ),
    (
        'Memory & Learning\n\n'
        'Load the BEET memory store that persists analysis history, '
        'attempter profiles, and learned models across sessions. '
        'Record ground-truth labels, view attempter history, and '
        'rebuild shadow models or centroids from a labeled corpus.'
    ),
    (
        'Calibration & Baselines\n\n'
        'Load or build a conformal calibration table to convert raw '
        'confidence scores into well-calibrated probabilities. '
        'Analyze a baselines JSONL to compute score distributions and '
        'tune detection thresholds for your specific domain.'
    ),
    None,  # Reports tab — self-explanatory
    (
        'Quick Reference\n\n'
        'Summary of every analysis, channel, and signal that runs in '
        'the detection pipeline. Use this as a quick lookup.'
    ),
    (
        'Precheck\n\n'
        'Shows all required and optional Python modules, models, and '
        'external programs needed by the pipeline. Green checkmarks '
        'indicate available items; exclamation marks indicate missing '
        'but non-critical items; red X marks indicate items whose '
        'absence will break parts of the analysis.'
    ),
]


if HAS_TK:
    class _NotebookToolTip:
        """Shows a brief tooltip when the mouse hovers over a notebook tab."""

        def __init__(self, notebook, tab_texts):
            """
            notebook  : ttk.Notebook instance
            tab_texts : list whose index matches the tab index; None = no tip
            """
            self._nb = notebook
            self._texts = tab_texts
            self._tip_win = None
            self._after_id = None
            notebook.bind('<Motion>', self._on_motion)
            notebook.bind('<Leave>', self._cancel)
            notebook.bind('<ButtonPress>', self._cancel)

        # ------------------------------------------------------------------
        def _on_motion(self, event):
            try:
                idx = self._nb.index('@%d,%d' % (event.x, event.y))
            except Exception:
                self._cancel()
                return
            text = (self._texts[idx]
                    if 0 <= idx < len(self._texts) else None)
            if not text:
                self._cancel()
                return
            if self._after_id is None:
                self._after_id = self._nb.after(
                    550,
                    lambda x=event.x_root, y=event.y_root: self._show(text, x, y),
                )

        def _cancel(self, event=None):
            if self._after_id is not None:
                self._nb.after_cancel(self._after_id)
                self._after_id = None
            self._hide()

        def _show(self, text, x, y):
            self._after_id = None
            self._hide()
            self._tip_win = tk.Toplevel(self._nb)
            self._tip_win.wm_overrideredirect(True)
            self._tip_win.wm_geometry('+%d+%d' % (x + 12, y + 20))
            tk.Label(
                self._tip_win,
                text=text,
                justify=tk.LEFT,
                background='#ffffcc',
                foreground='#333333',
                relief=tk.SOLID,
                borderwidth=1,
                font=('Segoe UI', 9),
                wraplength=320,
                padx=8,
                pady=6,
            ).pack()

        def _hide(self):
            if self._tip_win is not None:
                self._tip_win.destroy()
                self._tip_win = None


class DetectorGUI:
    """Full-featured desktop GUI exposing all pipeline capabilities."""

    def __init__(self, root):
        self.root = root
        self.root.title("LLM Authorship Signal Analyzer v0.67")
        self.root.geometry("1180x920")

        self._memory_store = None
        self._cal_table = None
        self._last_results = []
        self._last_text_map = {}

        self._init_vars()
        self._configure_theme()
        self._build_layout()

    def _init_vars(self):
        self.file_var = tk.StringVar()
        self.prompt_col_var = tk.StringVar(value='prompt')
        self.sheet_var = tk.StringVar()
        self.attempter_var = tk.StringVar()
        self.id_col_var = tk.StringVar(value='task_id')
        self.occ_col_var = tk.StringVar(value='occupation')
        self.attempter_col_var = tk.StringVar(value='attempter_name')
        self.stage_col_var = tk.StringVar(value='pipeline_stage_name')
        self.attempter_email_col_var = tk.StringVar()
        self.reviewer_col_var = tk.StringVar()
        self.reviewer_email_col_var = tk.StringVar()
        self.provider_var = tk.StringVar(value='anthropic')
        self.api_key_var = tk.StringVar()
        self.status_var = tk.StringVar(value='Ready')
        self.mode_var = tk.StringVar(value='auto')
        self.show_details_var = tk.BooleanVar(value=True)
        self.dna_model_var = tk.StringVar()
        self.dna_samples_var = tk.StringVar(value='3')
        self.workers_var = tk.StringVar(value='4')
        self.batch_var = tk.BooleanVar(value=False)
        self.no_layer3_var = tk.BooleanVar(value=False)
        self.ppl_model_var = tk.StringVar(value='Qwen/Qwen2.5-0.5B')
        self.verbose_var = tk.BooleanVar(value=False)
        self.output_csv_var = tk.StringVar()
        self.html_report_var = tk.StringVar()
        self.cost_var = tk.StringVar(value='400.0')
        self.no_similarity_var = tk.BooleanVar(value=False)
        self.sim_threshold_var = tk.StringVar(value='0.40')
        self.sim_store_var = tk.StringVar()
        self.instructions_var = tk.StringVar()
        self.memory_var = tk.StringVar()
        self.collect_var = tk.StringVar()
        self.cal_table_var = tk.StringVar()
        self.calibrate_var = tk.StringVar()
        self.baselines_jsonl_var = tk.StringVar()
        self.baselines_csv_var = tk.StringVar()
        self.labeled_corpus_var = tk.StringVar()
        self.confirm_task_var = tk.StringVar()
        self.confirm_label_var = tk.StringVar(value='ai')
        self.confirm_reviewer_var = tk.StringVar()
        self.quick_reviewer_var = tk.StringVar()
        self.attempter_history_var = tk.StringVar()
        self.run_dir_var = tk.StringVar()
        self.cal_report_jsonl_var = tk.StringVar()
        self.cal_report_csv_var = tk.StringVar()
        self.label_output_var = tk.StringVar()
        self.label_reviewer_var = tk.StringVar()
        self.label_skip_green_var = tk.BooleanVar(value=False)
        self.label_skip_red_var = tk.BooleanVar(value=False)
        self.label_max_var = tk.StringVar(value='')
        # KPI metric variables for Analysis tab dashboard cards.
        self.metric_total_var = tk.StringVar(value='0')
        self.metric_top_det_var = tk.StringVar(value='N/A')
        self.metric_avg_conf_var = tk.StringVar(value='0.00')
        self.metric_mode_var = tk.StringVar(value='auto')

        self.ablation_vars = {}
        for ch in _CHANNELS:
            self.ablation_vars[ch] = tk.BooleanVar(value=False)
        # Sync mode metric card whenever detection mode is changed.
        self.mode_var.trace_add('write', self._sync_mode_metric)
        # Memory store status label variable.
        self.memory_status_var = tk.StringVar(value='Not loaded')

    def _build_layout(self):
        header = ttk.Frame(self.root, style='DashboardHeader.TFrame', padding=(12, 10))
        header.pack(fill=tk.X, padx=6, pady=(6, 0))
        ttk.Label(header, text='LLM Authorship Signal Analyzer',
                  style='DashboardTitle.TLabel').pack(anchor='w', side=tk.LEFT)
        ttk.Button(
            header,
            text='🌐 Open Web Dashboard',
            command=self._launch_dashboard,
        ).pack(side=tk.RIGHT, padx=(0, 4))
        ttk.Button(
            header,
            text='🖥 New Desktop Window',
            command=self._launch_desktop_gui,
        ).pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Label(header, text='Analyst dashboard for prompt forensics, calibration, and reporting',
                  style='DashboardSubtitle.TLabel').pack(anchor='w')

        notebook = ttk.Notebook(self.root, style='Dashboard.TNotebook')
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Tab 1: Analysis (primary workflow)
        self._build_analysis_tab(notebook)
        # Tab 2: Configuration
        self._build_config_tab(notebook)
        # Tab 3: Memory & Learning
        self._build_memory_tab(notebook)
        # Tab 4: Calibration & Baselines
        self._build_calibration_tab(notebook)
        # Tab 5: Reports
        self._build_reports_tab(notebook)
        # Tab 6: Quick Reference
        self._build_quick_reference_tab(notebook)
        # Tab 7: Precheck
        self._build_precheck_tab(notebook)

        # Hover tooltips on tab headers
        _NotebookToolTip(notebook, _TAB_TOOLTIPS)

        # Status bar
        status = ttk.Frame(self.root, style='DashboardStatus.TFrame', padding=(10, 6))
        status.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Label(status, textvariable=self.status_var, style='DashboardStatus.TLabel').pack(anchor='w')

    # ── Tab 1: Analysis ──────────────────────────────────────────────

    def _build_analysis_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text='  Analysis  ')

        metrics = ttk.Frame(tab)
        metrics.pack(fill=tk.X, pady=(0, 8))
        self._build_metric_card(metrics, 'Total Results', self.metric_total_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self._build_metric_card(metrics, 'Top Determination', self.metric_top_det_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self._build_metric_card(metrics, 'Avg Confidence', self.metric_avg_conf_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self._build_metric_card(metrics, 'Mode', self.metric_mode_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        # File input
        file_row = ttk.Frame(tab)
        file_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(file_row, text='Input file:').pack(side=tk.LEFT)
        ttk.Entry(file_row, textvariable=self.file_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(file_row, text='Browse', command=self._browse_file).pack(side=tk.LEFT)

        # File options
        opts = ttk.Frame(tab)
        opts.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(opts, text='Prompt col').grid(row=0, column=0, sticky='w')
        ttk.Entry(opts, textvariable=self.prompt_col_var, width=14).grid(row=0, column=1, sticky='w', padx=4)
        ttk.Label(opts, text='Sheet').grid(row=0, column=2, sticky='w')
        ttk.Entry(opts, textvariable=self.sheet_var, width=14).grid(row=0, column=3, sticky='w', padx=4)
        ttk.Label(opts, text='Filter by attempter').grid(row=0, column=4, sticky='w')
        ttk.Entry(opts, textvariable=self.attempter_var, width=14).grid(row=0, column=5, sticky='w', padx=4)

        # Column mapping (advanced options)
        col_map = ttk.LabelFrame(
            tab,
            text='Column Mapping (optional - accepts column names, Excel letters A-Z, or 1-based numeric positions)',
        )
        col_map.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(col_map, text='Task ID col:').grid(row=0, column=0, sticky='w', padx=6, pady=3)
        ttk.Entry(col_map, textvariable=self.id_col_var, width=18).grid(row=0, column=1, sticky='w', padx=4, pady=3)
        ttk.Label(col_map, text='Occupation col:').grid(row=0, column=2, sticky='w', padx=(12, 6), pady=3)
        ttk.Entry(col_map, textvariable=self.occ_col_var, width=18).grid(row=0, column=3, sticky='w', padx=4, pady=3)
        ttk.Label(col_map, text='Attempter col:').grid(row=1, column=0, sticky='w', padx=6, pady=3)
        ttk.Entry(col_map, textvariable=self.attempter_col_var, width=18).grid(row=1, column=1, sticky='w', padx=4, pady=3)
        ttk.Label(col_map, text='Stage col:').grid(row=1, column=2, sticky='w', padx=(12, 6), pady=3)
        ttk.Entry(col_map, textvariable=self.stage_col_var, width=18).grid(row=1, column=3, sticky='w', padx=4, pady=3)
        ttk.Label(col_map, text='Attempter email col:').grid(row=2, column=0, sticky='w', padx=6, pady=3)
        ttk.Entry(col_map, textvariable=self.attempter_email_col_var, width=18).grid(row=2, column=1, sticky='w', padx=4, pady=3)
        ttk.Label(col_map, text='Reviewer col:').grid(row=2, column=2, sticky='w', padx=(12, 6), pady=3)
        ttk.Entry(col_map, textvariable=self.reviewer_col_var, width=18).grid(row=2, column=3, sticky='w', padx=4, pady=3)
        ttk.Label(col_map, text='Reviewer email col:').grid(row=3, column=0, sticky='w', padx=6, pady=3)
        ttk.Entry(col_map, textvariable=self.reviewer_email_col_var, width=18).grid(row=3, column=1, sticky='w', padx=4, pady=3)

        # Mode & detection options
        mode_row = ttk.Frame(tab)
        mode_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(mode_row, text='Mode:').pack(side=tk.LEFT)
        ttk.Combobox(mode_row, textvariable=self.mode_var,
                     values=['auto', 'task_prompt', 'generic_aigt'],
                     width=14, state='readonly').pack(side=tk.LEFT, padx=(4, 12))
        ttk.Checkbutton(mode_row, text='Show details', variable=self.show_details_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(mode_row, text='Verbose', variable=self.verbose_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(mode_row, text='Skip API Continuation', variable=self.no_layer3_var).pack(side=tk.LEFT)

        # Perplexity model selector
        ppl_row = ttk.Frame(tab)
        ppl_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(ppl_row, text='PPL Model:').pack(side=tk.LEFT)
        ttk.Combobox(ppl_row, textvariable=self.ppl_model_var,
                     values=['Qwen/Qwen2.5-0.5B', 'HuggingFaceTB/SmolLM2-360M',
                             'HuggingFaceTB/SmolLM2-135M', 'distilgpt2', 'gpt2'],
                     width=30, state='readonly').pack(side=tk.LEFT, padx=(4, 0))

        # Channel ablation — check a channel to *disable* it during analysis
        abl = ttk.LabelFrame(tab, text='Disable Channels (check to skip)')
        abl.pack(fill=tk.X, pady=(0, 6))
        for ch in _CHANNELS:
            ttk.Checkbutton(abl, text=ch, variable=self.ablation_vars[ch]).pack(side=tk.LEFT, padx=6, pady=3)

        # Text input
        ttk.Label(tab, text='Single text analysis (paste or type below):').pack(anchor='w')
        self.text_input = tk.Text(tab, height=7, wrap=tk.WORD)
        self.text_input.pack(fill=tk.BOTH, pady=(2, 6))
        self._add_paste_menu(self.text_input)

        # Action buttons
        actions = ttk.Frame(tab)
        actions.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(actions, text='▶ Analyze Text', style='DashboardPrimary.TButton',
                   command=lambda: self._run_async(self._analyze_text)).pack(side=tk.LEFT)
        ttk.Button(actions, text='📂 Analyze File', style='DashboardPrimary.TButton',
                   command=lambda: self._run_async(self._analyze_file)).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text='Clear', command=self._clear_output).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(actions, text='💾 Save CSV', command=self._save_results_csv).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(actions, text='Save HTML Reports', command=self._save_html_reports).pack(side=tk.LEFT)

        # Progress bar
        progress_frame = ttk.Frame(tab)
        progress_frame.pack(fill=tk.X, pady=(0, 6))
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                             maximum=100, mode='determinate',
                                             style='Dashboard.Horizontal.TProgressbar')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.progress_label = ttk.Label(progress_frame, text='', width=20)
        self.progress_label.pack(side=tk.LEFT, padx=(6, 0))

        # Results output
        output_frame = ttk.Frame(tab)
        output_frame.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(output_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output = tk.Text(output_frame, height=16, wrap=tk.WORD,
                              font=('Consolas', 10), yscrollcommand=scrollbar.set)
        self.output.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.output.yview)

        for det, color in _DET_COLORS.items():
            self.output.tag_configure(det, foreground=color)
        self.output.tag_configure('HEADER', foreground='#1565c0', font=('Consolas', 10, 'bold'))
        self.output.tag_configure('DIM', foreground='#757575')
        self.output.tag_configure('ALERT', foreground='#d32f2f', font=('Consolas', 10, 'bold'))

    # ── Tab 2: Configuration ─────────────────────────────────────────

    def _build_config_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text='  Configuration  ')

        # DNA-GPT / Continuation
        dna = ttk.LabelFrame(tab, text='Continuation Analysis (DNA-GPT)')
        dna.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(dna, text='Provider').grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Combobox(dna, textvariable=self.provider_var, values=['anthropic', 'openai'],
                     width=12, state='readonly').grid(row=0, column=1, sticky='w', pady=4)
        ttk.Label(dna, text='API Key').grid(row=0, column=2, sticky='w', padx=(12, 6), pady=4)
        api_entry = ttk.Entry(dna, textvariable=self.api_key_var, show='*')
        api_entry.grid(row=0, column=3, sticky='ew', padx=(0, 6), pady=4)
        self._add_paste_menu(api_entry)
        dna.columnconfigure(3, weight=1)
        ttk.Label(dna, text='Model (optional)').grid(row=1, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(dna, textvariable=self.dna_model_var, width=24).grid(row=1, column=1, columnspan=2, sticky='w', pady=4)
        ttk.Label(dna, text='Samples').grid(row=1, column=2, sticky='e', padx=(12, 6), pady=4)
        ttk.Spinbox(dna, textvariable=self.dna_samples_var, from_=1, to=10, width=4).grid(row=1, column=3, sticky='w', pady=4)
        ttk.Label(dna, text='Workers').grid(row=2, column=0, sticky='w', padx=6, pady=4)
        ttk.Spinbox(dna, textvariable=self.workers_var, from_=1, to=16, width=4).grid(row=2, column=1, sticky='w', pady=4)
        ttk.Checkbutton(dna, text='Batch API (50% cheaper)',
                        variable=self.batch_var).grid(row=2, column=2, columnspan=2, sticky='w', padx=(12, 0), pady=4)

        # Similarity
        sim = ttk.LabelFrame(tab, text='Similarity Analysis')
        sim.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(sim, text='Disable similarity', variable=self.no_similarity_var).grid(
            row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Label(sim, text='Threshold').grid(row=0, column=1, sticky='w', padx=(12, 6), pady=4)
        ttk.Entry(sim, textvariable=self.sim_threshold_var, width=6).grid(row=0, column=2, sticky='w', pady=4)
        ttk.Label(sim, text='Sim store (JSONL)').grid(row=1, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(sim, textvariable=self.sim_store_var).grid(row=1, column=1, columnspan=2, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(sim, text='...', width=3, command=lambda: self._browse_save(self.sim_store_var, [('JSONL', '*.jsonl')])).grid(
            row=1, column=3, sticky='w', padx=2, pady=4)
        ttk.Label(sim, text='Instructions file').grid(row=2, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(sim, textvariable=self.instructions_var).grid(row=2, column=1, columnspan=2, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(sim, text='...', width=3, command=lambda: self._browse_open(self.instructions_var)).grid(
            row=2, column=3, sticky='w', padx=2, pady=4)
        sim.columnconfigure(2, weight=1)

        # Output
        out = ttk.LabelFrame(tab, text='Output Options')
        out.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(out, text='Output CSV').grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(out, textvariable=self.output_csv_var).grid(row=0, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(out, text='...', width=3, command=lambda: self._browse_save(self.output_csv_var, [('CSV', '*.csv')])).grid(
            row=0, column=2, sticky='w', padx=2, pady=4)
        ttk.Label(out, text='HTML report').grid(row=1, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(out, textvariable=self.html_report_var).grid(row=1, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(out, text='...', width=3, command=lambda: self._browse_save(self.html_report_var, [('HTML', '*.html')])).grid(
            row=1, column=2, sticky='w', padx=2, pady=4)
        ttk.Label(out, text='Cost per prompt ($)').grid(row=2, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(out, textvariable=self.cost_var, width=8).grid(row=2, column=1, sticky='w', pady=4)
        ttk.Label(out, text='Run directory').grid(row=3, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(out, textvariable=self.run_dir_var).grid(row=3, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(out, text='...', width=3, command=lambda: self._browse_dir(self.run_dir_var)).grid(
            row=3, column=2, sticky='w', padx=2, pady=4)
        out.columnconfigure(1, weight=1)

        # Baseline collection
        bl = ttk.LabelFrame(tab, text='Baseline Collection')
        bl.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(bl, text='Collect to JSONL').grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(bl, textvariable=self.collect_var).grid(row=0, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(bl, text='...', width=3, command=lambda: self._browse_save(self.collect_var, [('JSONL', '*.jsonl')])).grid(
            row=0, column=2, sticky='w', padx=2, pady=4)
        bl.columnconfigure(1, weight=1)

    # ── Tab 3: Memory & Learning ─────────────────────────────────────

    def _build_memory_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text='  Memory & Learning  ')

        # Memory store directory
        mem = ttk.LabelFrame(tab, text='BEET Memory Store')
        mem.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(mem, text=(
            'The memory store persists analysis history across sessions.\n'
            'Use the default ".beet" folder or choose a custom location.'
        ), style='DashboardSubtitle.TLabel').grid(
            row=0, column=0, columnspan=5, sticky='w', padx=6, pady=(4, 2))

        ttk.Label(mem, text='Store directory').grid(row=1, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(mem, textvariable=self.memory_var).grid(row=1, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(mem, text='...', width=3, command=lambda: self._browse_dir(self.memory_var)).grid(
            row=1, column=2, sticky='w', padx=2, pady=4)
        ttk.Button(mem, text='Use Default (.beet)', command=self._use_default_memory).grid(
            row=1, column=3, sticky='w', padx=(4, 2), pady=4)
        ttk.Button(mem, text='Load', command=self._load_memory).grid(row=1, column=4, sticky='w', padx=6, pady=4)
        mem.columnconfigure(1, weight=1)

        # Status label
        status_frame = ttk.Frame(mem)
        status_frame.grid(row=2, column=0, columnspan=5, sticky='w', padx=6, pady=(0, 4))
        ttk.Label(status_frame, text='Status:').pack(side=tk.LEFT)
        self._memory_status_label = ttk.Label(
            status_frame, textvariable=self.memory_status_var,
            foreground=_DASHBOARD_THEME['muted'])
        self._memory_status_label.pack(side=tk.LEFT, padx=(4, 0))

        btn_row = ttk.Frame(mem)
        btn_row.grid(row=3, column=0, columnspan=5, sticky='w', padx=6, pady=4)
        ttk.Button(btn_row, text='Print Summary', command=lambda: self._run_async(self._memory_summary)).pack(side=tk.LEFT, padx=(0, 6))

        # Confirmations — manual entry
        conf = ttk.LabelFrame(tab, text='Record Ground Truth Confirmation')
        conf.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(conf, text='Task ID').grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(conf, textvariable=self.confirm_task_var, width=24).grid(row=0, column=1, sticky='w', padx=(0, 6), pady=4)
        ttk.Label(conf, text='Label').grid(row=0, column=2, sticky='w', padx=(12, 6), pady=4)
        ttk.Combobox(conf, textvariable=self.confirm_label_var, values=_GROUND_TRUTH_LABELS,
                     width=8, state='readonly').grid(row=0, column=3, sticky='w', pady=4)
        ttk.Label(conf, text='Reviewer').grid(row=0, column=4, sticky='w', padx=(12, 6), pady=4)
        ttk.Entry(conf, textvariable=self.confirm_reviewer_var, width=16).grid(row=0, column=5, sticky='w', padx=(0, 6), pady=4)
        ttk.Button(conf, text='Confirm', command=self._record_confirmation).grid(row=0, column=6, sticky='w', padx=6, pady=4)

        # Quick-confirm from recent results
        recent = ttk.LabelFrame(tab, text='Quick Confirm — Recent Scanned Samples')
        recent.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(recent, text=(
            'Select a recently scanned sample below and click Human / AI / Unsure to record the ground-truth label.'
        ), style='DashboardSubtitle.TLabel').grid(row=0, column=0, columnspan=6, sticky='w', padx=6, pady=(4, 2))

        # Listbox for recent results
        self._recent_frame = ttk.Frame(recent)
        self._recent_frame.grid(row=1, column=0, columnspan=6, sticky='nsew', padx=6, pady=4)
        recent.columnconfigure(0, weight=1)

        lb_scroll = ttk.Scrollbar(self._recent_frame, orient=tk.VERTICAL)
        lb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._recent_listbox = tk.Listbox(
            self._recent_frame, height=6, font=('Consolas', 9),
            yscrollcommand=lb_scroll.set, selectmode=tk.SINGLE)
        self._recent_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lb_scroll.config(command=self._recent_listbox.yview)
        self._recent_listbox.bind('<<ListboxSelect>>', self._on_recent_select)

        self._recent_preview = tk.Text(self._recent_frame, height=3, wrap=tk.WORD,
                                       font=('Consolas', 9), state='disabled')
        self._recent_preview.pack(fill=tk.X, pady=(4, 0))

        btn_row_confirm = ttk.Frame(recent)
        btn_row_confirm.grid(row=2, column=0, columnspan=6, sticky='w', padx=6, pady=4)
        ttk.Label(btn_row_confirm, text='Reviewer').pack(side=tk.LEFT, padx=(0, 4))
        ttk.Entry(btn_row_confirm, textvariable=self.quick_reviewer_var, width=16).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row_confirm, text='\U0001f9d1  Human',
                   command=lambda: self._quick_confirm('human')).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row_confirm, text='\U0001f916  AI',
                   command=lambda: self._quick_confirm('ai')).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row_confirm, text='?  Unsure',
                   command=lambda: self._quick_confirm('unsure')).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row_confirm, text='Refresh List',
                   command=self._refresh_recent_samples).pack(side=tk.LEFT, padx=(12, 0))

        # Attempter history
        hist = ttk.LabelFrame(tab, text='Attempter History')
        hist.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(hist, text='Attempter name').grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(hist, textvariable=self.attempter_history_var, width=24).grid(row=0, column=1, sticky='w', padx=(0, 6), pady=4)
        ttk.Button(hist, text='Show History', command=lambda: self._run_async(self._show_attempter_history)).grid(
            row=0, column=2, sticky='w', padx=6, pady=4)

        # Learning tools
        learn = ttk.LabelFrame(tab, text='Learning Tools (require memory store)')
        learn.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(learn, text='Labeled corpus (JSONL)').grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(learn, textvariable=self.labeled_corpus_var).grid(row=0, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(learn, text='...', width=3, command=lambda: self._browse_open(self.labeled_corpus_var, [('JSONL', '*.jsonl')])).grid(
            row=0, column=2, sticky='w', padx=2, pady=4)
        learn.columnconfigure(1, weight=1)

        btn_row2 = ttk.Frame(learn)
        btn_row2.grid(row=1, column=0, columnspan=3, sticky='w', padx=6, pady=4)
        ttk.Button(btn_row2, text='Rebuild Shadow Model', command=lambda: self._run_async(self._rebuild_shadow)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row2, text='Rebuild Centroids', command=lambda: self._run_async(self._rebuild_centroids)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row2, text='Discover Lexicon', command=lambda: self._run_async(self._discover_lexicon)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row2, text='Rebuild All', command=lambda: self._run_async(self._rebuild_all)).pack(side=tk.LEFT)

        # ML Fusion Readiness
        mlf = ttk.LabelFrame(tab, text='ML Fusion Readiness')
        mlf.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(mlf, text=(
            'ML fusion replaces heuristic rules with a trained classifier.\n'
            'Requires confirmed ground-truth labels in the memory store.'
        ), style='DashboardSubtitle.TLabel').grid(
            row=0, column=0, columnspan=4, sticky='w', padx=6, pady=(4, 2))

        # Progress row
        ttk.Label(mlf, text='Confirmed labels:').grid(row=1, column=0, sticky='w', padx=6, pady=4)
        self._ml_fusion_progress_var = tk.StringVar(value='0 / 200 (0%)')
        ttk.Label(mlf, textvariable=self._ml_fusion_progress_var).grid(
            row=1, column=1, sticky='w', padx=(0, 12), pady=4)

        ttk.Label(mlf, text='Class balance:').grid(row=1, column=2, sticky='w', padx=(12, 6), pady=4)
        self._ml_fusion_balance_var = tk.StringVar(value='AI: 0  |  Human: 0')
        ttk.Label(mlf, textvariable=self._ml_fusion_balance_var).grid(
            row=1, column=3, sticky='w', pady=4)

        # Progress bar
        self._ml_fusion_progress_bar = ttk.Progressbar(mlf, length=300, mode='determinate', maximum=100)
        self._ml_fusion_progress_bar.grid(row=2, column=0, columnspan=4, sticky='ew', padx=6, pady=(0, 4))

        # Model status
        ttk.Label(mlf, text='Model status:').grid(row=3, column=0, sticky='w', padx=6, pady=4)
        self._ml_fusion_model_var = tk.StringVar(value='No model trained')
        ttk.Label(mlf, textvariable=self._ml_fusion_model_var,
                  foreground=_DASHBOARD_THEME['muted']).grid(
            row=3, column=1, columnspan=3, sticky='w', pady=4)

        # Buttons
        btn_row_ml = ttk.Frame(mlf)
        btn_row_ml.grid(row=4, column=0, columnspan=4, sticky='w', padx=6, pady=4)
        ttk.Button(btn_row_ml, text='Refresh', command=self._refresh_fusion_readiness).pack(side=tk.LEFT, padx=(0, 6))
        self._train_fusion_btn = ttk.Button(btn_row_ml, text='Train ML Fusion', command=lambda: self._run_async(self._train_fusion_model), state='disabled')
        self._train_fusion_btn.pack(side=tk.LEFT, padx=(0, 6))

        self._ml_fusion_enabled_var = tk.BooleanVar(value=False)
        self._ml_fusion_checkbox = ttk.Checkbutton(
            btn_row_ml, text='Enable ML Fusion (use trained model)',
            variable=self._ml_fusion_enabled_var, state='disabled')
        self._ml_fusion_checkbox.pack(side=tk.LEFT, padx=(12, 0))

        mlf.columnconfigure(1, weight=1)

        # Interactive Labeling
        lbl = ttk.LabelFrame(tab, text='Interactive Labeling')
        lbl.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(lbl, text=(
            'Review pipeline results and assign ground-truth labels for calibration.\n'
            'Requires a completed analysis run on the Analysis tab.'
        ), style='DashboardSubtitle.TLabel').grid(
            row=0, column=0, columnspan=4, sticky='w', padx=6, pady=(4, 2))
        ttk.Label(lbl, text='Output JSONL').grid(row=1, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(lbl, textvariable=self.label_output_var).grid(row=1, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(lbl, text='...', width=3, command=lambda: self._browse_save(self.label_output_var, [('JSONL', '*.jsonl')])).grid(
            row=1, column=2, sticky='w', padx=2, pady=4)
        ttk.Label(lbl, text='Reviewer').grid(row=2, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(lbl, textvariable=self.label_reviewer_var, width=24).grid(row=2, column=1, sticky='w', pady=4)
        ttk.Label(lbl, text='Max labels').grid(row=3, column=0, sticky='w', padx=6, pady=4)
        ttk.Spinbox(lbl, textvariable=self.label_max_var, from_=0, to=9999, width=6).grid(row=3, column=1, sticky='w', pady=4)
        ttk.Checkbutton(lbl, text='Skip GREEN', variable=self.label_skip_green_var).grid(row=4, column=0, sticky='w', padx=6, pady=4)
        ttk.Checkbutton(lbl, text='Skip RED', variable=self.label_skip_red_var).grid(row=4, column=1, sticky='w', pady=4)
        ttk.Button(lbl, text='▶ Start Labeling Session', command=self._start_labeling_session,
                   style='DashboardPrimary.TButton').grid(row=5, column=0, columnspan=3, sticky='w', padx=6, pady=6)
        lbl.columnconfigure(1, weight=1)

    # ── Tab 4: Calibration & Baselines ───────────────────────────────

    def _build_calibration_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text='  Calibration & Baselines  ')

        # Calibration
        cal = ttk.LabelFrame(tab, text='Conformal Calibration')
        cal.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(cal, text='Cal table (JSON)').grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(cal, textvariable=self.cal_table_var).grid(row=0, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(cal, text='...', width=3, command=lambda: self._browse_open(self.cal_table_var, [('JSON', '*.json')])).grid(
            row=0, column=2, sticky='w', padx=2, pady=4)
        ttk.Button(cal, text='Load', command=self._load_cal_table).grid(row=0, column=3, sticky='w', padx=6, pady=4)
        cal.columnconfigure(1, weight=1)

        ttk.Label(cal, text='Build from JSONL').grid(row=1, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(cal, textvariable=self.calibrate_var).grid(row=1, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(cal, text='...', width=3, command=lambda: self._browse_open(self.calibrate_var, [('JSONL', '*.jsonl')])).grid(
            row=1, column=2, sticky='w', padx=2, pady=4)
        ttk.Button(cal, text='Build & Save', command=lambda: self._run_async(self._build_calibration)).grid(
            row=1, column=3, sticky='w', padx=6, pady=4)

        ttk.Button(cal, text='Rebuild from Memory', command=lambda: self._run_async(self._rebuild_calibration)).grid(
            row=2, column=0, columnspan=2, sticky='w', padx=6, pady=4)

        # Baseline analysis
        bl = ttk.LabelFrame(tab, text='Baseline Analysis')
        bl.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(bl, text='Baselines JSONL').grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(bl, textvariable=self.baselines_jsonl_var).grid(row=0, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(bl, text='...', width=3, command=lambda: self._browse_open(self.baselines_jsonl_var, [('JSONL', '*.jsonl')])).grid(
            row=0, column=2, sticky='w', padx=2, pady=4)
        ttk.Label(bl, text='Output CSV').grid(row=1, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(bl, textvariable=self.baselines_csv_var).grid(row=1, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(bl, text='...', width=3, command=lambda: self._browse_save(self.baselines_csv_var, [('CSV', '*.csv')])).grid(
            row=1, column=2, sticky='w', padx=2, pady=4)
        ttk.Button(bl, text='Analyze Baselines', command=lambda: self._run_async(self._analyze_baselines)).grid(
            row=2, column=0, columnspan=2, sticky='w', padx=6, pady=4)
        bl.columnconfigure(1, weight=1)

        # Calibration Diagnostics Report
        cr = ttk.LabelFrame(tab, text='Calibration Diagnostics Report')
        cr.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(cr, text='Labeled JSONL').grid(row=0, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(cr, textvariable=self.cal_report_jsonl_var).grid(row=0, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(cr, text='...', width=3, command=lambda: self._browse_open(self.cal_report_jsonl_var, [('JSONL', '*.jsonl')])).grid(
            row=0, column=2, sticky='w', padx=2, pady=4)
        ttk.Label(cr, text='Export CSV (optional)').grid(row=1, column=0, sticky='w', padx=6, pady=4)
        ttk.Entry(cr, textvariable=self.cal_report_csv_var).grid(row=1, column=1, sticky='ew', padx=(0, 6), pady=4)
        ttk.Button(cr, text='...', width=3, command=lambda: self._browse_save(self.cal_report_csv_var, [('CSV', '*.csv')])).grid(
            row=1, column=2, sticky='w', padx=2, pady=4)
        ttk.Button(cr, text='Generate Report', command=lambda: self._run_async(self._run_calibration_report)).grid(
            row=2, column=0, columnspan=2, sticky='w', padx=6, pady=4)
        cr.columnconfigure(1, weight=1)

    # ── Helpers ───────────────────────────────────────────────────────

    def _configure_theme(self):
        self.root.configure(bg=_DASHBOARD_THEME['bg'])
        style = ttk.Style(self.root)
        if 'clam' in style.theme_names():
            style.theme_use('clam')

        try:
            base_font = tkfont.nametofont('TkDefaultFont')
        except tk.TclError:
            try:
                base_font = tkfont.nametofont('TkTextFont')
            except tk.TclError:
                base_font = None

        self._title_font = None
        self._subtitle_font = None
        self._section_font = None
        if base_font is not None:
            self._title_font = base_font.copy()
            self._title_font.configure(size=14, weight='bold')
            self._subtitle_font = base_font.copy()
            self._subtitle_font.configure(size=10)
            self._section_font = base_font.copy()
            self._section_font.configure(size=10, weight='bold')

        style.configure('TFrame', background=_DASHBOARD_THEME['bg'])
        style.configure('TLabelframe', background=_DASHBOARD_THEME['card'])
        style.configure('TLabelframe.Label', background=_DASHBOARD_THEME['card'],
                        foreground=_DASHBOARD_THEME['text'])
        if self._section_font is not None:
            style.configure('TLabelframe.Label', font=self._section_font)
        style.configure('TLabel', background=_DASHBOARD_THEME['bg'], foreground=_DASHBOARD_THEME['text'])
        style.configure('TCheckbutton', background=_DASHBOARD_THEME['bg'], foreground=_DASHBOARD_THEME['text'])
        style.configure('TButton', padding=(10, 6))
        style.configure('TEntry', fieldbackground='white')
        style.configure('TCombobox', fieldbackground='white')
        style.configure('DashboardCard.TFrame', background=_DASHBOARD_THEME['card'])
        style.configure('DashboardCardLabel.TLabel', background=_DASHBOARD_THEME['card'],
                        foreground=_DASHBOARD_THEME['muted'])
        style.configure('DashboardCardValue.TLabel', background=_DASHBOARD_THEME['card'],
                        foreground=_DASHBOARD_THEME['text'])
        if self._section_font is not None:
            style.configure('DashboardCardValue.TLabel', font=self._section_font)

        style.configure('DashboardHeader.TFrame', background=_DASHBOARD_THEME['card'])
        style.configure('DashboardTitle.TLabel', background=_DASHBOARD_THEME['card'],
                        foreground=_DASHBOARD_THEME['text'])
        style.configure('DashboardSubtitle.TLabel', background=_DASHBOARD_THEME['card'],
                        foreground=_DASHBOARD_THEME['muted'])
        if self._title_font is not None:
            style.configure('DashboardTitle.TLabel', font=self._title_font)
        if self._subtitle_font is not None:
            style.configure('DashboardSubtitle.TLabel', font=self._subtitle_font)
        style.configure('DashboardStatus.TFrame', background=_DASHBOARD_THEME['card'])
        style.configure('DashboardStatus.TLabel', background=_DASHBOARD_THEME['card'],
                        foreground=_DASHBOARD_THEME['muted'])
        style.configure('DashboardPrimary.TButton', padding=(12, 6))
        style.map('DashboardPrimary.TButton',
                  background=[('!disabled', _DASHBOARD_THEME['accent']), ('active', '#1d4ed8')],
                  foreground=[('!disabled', '#ffffff')])

        style.configure('Dashboard.TNotebook', background=_DASHBOARD_THEME['bg'], borderwidth=0)
        style.configure('Dashboard.TNotebook.Tab', padding=(12, 8),
                        background=_DASHBOARD_THEME['accent_light'], foreground=_DASHBOARD_THEME['text'])
        style.map('Dashboard.TNotebook.Tab',
                  background=[('selected', _DASHBOARD_THEME['accent'])],
                  foreground=[('selected', '#ffffff')])
        try:
            style.configure('Dashboard.Horizontal.TProgressbar',
                            troughcolor='#e5e7eb', background=_DASHBOARD_THEME['accent'])
        except tk.TclError:
            style.configure('Dashboard.Horizontal.TProgressbar')

    def _browse_file(self):
        path = filedialog.askopenfilename(filetypes=[('Data files', '*.csv *.xlsx *.xlsm *.pdf'), ('All files', '*.*')])
        if path:
            self.file_var.set(path)

    def _browse_open(self, var, filetypes=None):
        filetypes = filetypes or [('All files', '*.*')]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def _browse_save(self, var, filetypes=None):
        filetypes = filetypes or [('All files', '*.*')]
        path = filedialog.asksaveasfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def _browse_dir(self, var):
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _get_disabled_channels(self):
        disabled = [ch for ch, var in self.ablation_vars.items() if var.get()]
        return disabled or None

    def _get_dna_samples(self):
        try:
            return int(self.dna_samples_var.get())
        except ValueError:
            return 3

    def _get_cost(self):
        try:
            return float(self.cost_var.get())
        except ValueError:
            return 400.0

    def _add_paste_menu(self, widget):
        """Add right-click context menu with Cut/Copy/Paste and ensure Ctrl+V works."""
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label='Cut', command=lambda: widget.event_generate('<<Cut>>'))
        menu.add_command(label='Copy', command=lambda: widget.event_generate('<<Copy>>'))
        menu.add_command(label='Paste', command=lambda: widget.event_generate('<<Paste>>'))
        menu.add_separator()
        menu.add_command(label='Select All', command=lambda: widget.event_generate('<<SelectAll>>'))

        def _show_menu(event):
            menu.tk_popup(event.x_root, event.y_root)

        widget.bind('<Button-3>', _show_menu)
        # macOS uses Button-2 for right-click
        widget.bind('<Button-2>', _show_menu)
        # Ensure Ctrl+V / Cmd+V paste works
        widget.bind('<Control-v>', lambda e: widget.event_generate('<<Paste>>'))
        widget.bind('<Control-V>', lambda e: widget.event_generate('<<Paste>>'))

    def _get_api_key(self):
        key = self.api_key_var.get().strip()
        if not key:
            env_var = ('ANTHROPIC_API_KEY' if self.provider_var.get() == 'anthropic'
                       else 'OPENAI_API_KEY')
            key = os.environ.get(env_var, '')
        return key or None

    def _get_sim_threshold(self):
        try:
            return float(self.sim_threshold_var.get())
        except ValueError:
            return 0.40

    def _browse_sim_store(self):
        path = filedialog.askopenfilename(
            filetypes=[('JSONL', '*.jsonl'), ('All', '*.*')])
        if path:
            self.sim_store_var.set(path)

    def _browse_instructions(self):
        path = filedialog.askopenfilename(
            filetypes=[('Text', '*.txt *.md'), ('All', '*.*')])
        if path:
            self.instructions_var.set(path)

    def _browse_corpus(self):
        path = filedialog.askopenfilename(
            filetypes=[('JSONL', '*.jsonl'), ('All', '*.*')])
        if path:
            self.corpus_path_var.set(path)

    def _update_progress(self, current, total):
        pct = current / max(total, 1) * 100
        self.root.after(0, lambda: self.progress_var.set(pct))
        self.root.after(0, lambda: self.progress_label.config(text=f'{current}/{total}'))

    def _reset_progress(self):
        self.root.after(0, lambda: self.progress_var.set(0))
        self.root.after(0, lambda: self.progress_label.config(text=''))

    def _build_metric_card(self, parent, label, value_var):
        card = ttk.Frame(parent, style='DashboardCard.TFrame', padding=(10, 8))
        ttk.Label(card, text=label, style='DashboardCardLabel.TLabel').pack(anchor='w')
        ttk.Label(card, textvariable=value_var, style='DashboardCardValue.TLabel').pack(anchor='w', pady=(2, 0))
        return card

    def _sync_mode_metric(self, *args):
        self.metric_mode_var.set(self.mode_var.get())

    def _update_dashboard_metrics(self, results):
        n_results = len(results)
        determinations = [r.get('determination') for r in results if r.get('determination')]
        counts = Counter(determinations)
        top_det = 'N/A'
        if counts:
            mc = counts.most_common(1)
            top_det = mc[0][0] if mc else 'N/A'
        avg_conf = 0.0
        if n_results > 0:
            avg_conf = sum(float(r.get('confidence') or 0) for r in results) / n_results

        def apply():
            self.metric_total_var.set(str(n_results))
            self.metric_top_det_var.set(top_det)
            self.metric_avg_conf_var.set(f'{avg_conf:.2f}')
            self.metric_mode_var.set(self.mode_var.get())
        self.root.after(0, apply)

    def _clear_output(self):
        self.output.delete('1.0', tk.END)
        self._reset_progress()
        self._update_dashboard_metrics([])
        self.status_var.set('Ready')

    def _run_async(self, fn):
        self.status_var.set('Running...')

        def runner():
            try:
                fn()
                self.root.after(0, lambda: self.status_var.set('Done'))
            except Exception as exc:
                self.root.after(0, lambda: self.status_var.set('Error'))
                self.root.after(0, lambda e=exc: messagebox.showerror('Error', str(e)))

        threading.Thread(target=runner, daemon=True).start()

    def _append(self, text, tag=None):
        def do_append():
            if tag:
                self.output.insert(tk.END, text, tag)
            else:
                self.output.insert(tk.END, text)
            self.output.see(tk.END)
        self.root.after(0, do_append)

    # ── Memory Store ──────────────────────────────────────────────────

    def _ensure_memory(self):
        if self._memory_store is not None:
            return True
        path = self.memory_var.get().strip()
        if not path:
            self.root.after(0, lambda: messagebox.showinfo('Memory required', 'Set a memory store directory first.'))
            return False
        self._load_memory()
        return self._memory_store is not None

    def _use_default_memory(self):
        """Set memory store to the default .beet directory and load it."""
        self.memory_var.set('.beet')
        self._load_memory()

    def _load_memory(self):
        path = self.memory_var.get().strip()
        if not path:
            messagebox.showinfo('Memory required', 'Set a memory store directory first.')
            return
        from llm_detector.memory import MemoryStore
        self._memory_store = MemoryStore(path)
        cfg = self._memory_store._config
        n_subs = cfg.get('total_submissions', 0)
        n_batches = cfg.get('total_batches', 0)
        self.memory_status_var.set(
            f'✓ Loaded: {path}  ({n_subs} submissions, {n_batches} batches)')
        if hasattr(self, '_memory_status_label'):
            self._memory_status_label.configure(foreground='#388e3c')
        self.status_var.set(f'Memory store loaded: {path}')

    def _load_cal_table(self):
        path = self.cal_table_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showinfo('Cal table', 'Select a valid calibration table JSON file.')
            return
        from llm_detector.calibration import load_calibration
        self._cal_table = load_calibration(path)
        self.status_var.set(f"Calibration loaded: {self._cal_table['n_calibration']} records, "
                            f"{len(self._cal_table.get('strata', {}))} strata")

    # ── Analysis Actions ──────────────────────────────────────────────

    def _build_analyze_kwargs(self):
        kwargs = {
            'run_l3': not self.no_layer3_var.get(),
            'api_key': self._get_api_key(),
            'dna_provider': self.provider_var.get(),
            'dna_model': self.dna_model_var.get().strip() or None,
            'dna_samples': self._get_dna_samples(),
            'mode': self.mode_var.get(),
            'disabled_channels': self._get_disabled_channels(),
            'cal_table': self._cal_table,
            'memory_store': self._memory_store,
            'ppl_model': self.ppl_model_var.get() or None,
        }
        return kwargs

    def _analyze_text(self):
        text = self.text_input.get('1.0', tk.END).strip()
        if not text:
            self.root.after(0, lambda: messagebox.showinfo('Input required', 'Enter text to analyze.'))
            return
        kwargs = self._build_analyze_kwargs()
        result = analyze_prompt(text, **kwargs)

        # Shadow model check
        if self._memory_store:
            disagreement = self._memory_store.check_shadow_disagreement(result)
            result['shadow_disagreement'] = disagreement
            result['shadow_ai_prob'] = (disagreement or {}).get('shadow_ai_prob')

        self._last_results = [result]
        self._last_text_map = {'_single': text}
        self._update_dashboard_metrics(self._last_results)
        self._display_result(result)

        # Collect baselines if configured
        collect_path = self.collect_var.get().strip()
        if collect_path:
            from llm_detector.baselines import collect_baselines
            collect_baselines([result], collect_path)
            self._append(f"  Baseline appended to {collect_path}\n", 'DIM')

    def _build_loader_column_kwargs(self):
        """
        Build keyword arguments for loader functions with consistent defaults
        for column names. This avoids duplication between CSV/XLSX branches.
        """
        kwargs = {
            'prompt_col': self.prompt_col_var.get().strip() or 'prompt',
            'id_col': self.id_col_var.get().strip() or 'task_id',
            'occ_col': self.occ_col_var.get().strip() or 'occupation',
            'attempter_col': self.attempter_col_var.get().strip() or 'attempter_name',
            'stage_col': self.stage_col_var.get().strip() or 'pipeline_stage_name',
        }
        ae = self.attempter_email_col_var.get().strip()
        if ae:
            kwargs['attempter_email_col'] = ae
        rv = self.reviewer_col_var.get().strip()
        if rv:
            kwargs['reviewer_col'] = rv
        rev_email = self.reviewer_email_col_var.get().strip()
        if rev_email:
            kwargs['reviewer_email_col'] = rev_email
        return kwargs

    def _analyze_file(self):
        path = self.file_var.get().strip()
        if not path:
            self.root.after(0, lambda: messagebox.showinfo('Input required', 'Choose a CSV/XLSX file.'))
            return

        # Apply run-directory defaults before starting analysis
        run_dir_base = self.run_dir_var.get().strip()
        if run_dir_base:
            from pathlib import Path
            from datetime import datetime
            run_dir = Path(run_dir_base) / datetime.now().strftime('run_%Y%m%d_%H%M%S')
            run_dir.mkdir(parents=True, exist_ok=True)
            self._append(f"Run directory: {run_dir}\n", 'HEADER')
            if not self.output_csv_var.get().strip():
                self.output_csv_var.set(str(run_dir / 'results.csv'))
            if not self.html_report_var.get().strip():
                self.html_report_var.set(str(run_dir / 'report.html'))
            if not self.memory_var.get().strip():
                self.memory_var.set(str(run_dir / 'memory'))
            if not self.sim_store_var.get().strip():
                self.sim_store_var.set(str(run_dir / 'similarity.jsonl'))
            if not self.label_output_var.get().strip():
                self.label_output_var.set(str(run_dir / 'labels.jsonl'))

        ext = os.path.splitext(path)[1].lower()
        if ext in ('.xlsx', '.xlsm'):
            column_kwargs = self._build_loader_column_kwargs()
            tasks = load_xlsx(
                path,
                sheet=self.sheet_var.get().strip() or None,
                **column_kwargs,
            )
        elif ext == '.csv':
            column_kwargs = self._build_loader_column_kwargs()
            tasks = load_csv(
                path,
                **column_kwargs,
            )
        elif ext == '.pdf':
            if not HAS_PYPDF:
                self.root.after(0, lambda: messagebox.showerror(
                    'Missing dependency', 'PDF support requires pypdf: pip install pypdf'))
                return
            from llm_detector.io import load_pdf
            tasks = load_pdf(path)
        else:
            self.root.after(0, lambda: messagebox.showerror('Unsupported file', f'Unsupported: {ext}'))
            return

        if self.attempter_var.get().strip():
            needle = self.attempter_var.get().strip().lower()
            tasks = [t for t in tasks if needle in t.get('attempter', '').lower()]
        if not tasks:
            self.root.after(0, lambda: messagebox.showinfo('No tasks', 'No qualifying prompts found.'))
            return

        kwargs = self._build_analyze_kwargs()
        results = []
        text_map = {}
        counts = Counter()
        n_tasks = len(tasks)
        n_workers = max(1, int(self.workers_var.get() or 1))
        use_batch = (self.batch_var.get()
                     and kwargs.get('api_key')
                     and self.provider_var.get() == 'anthropic'
                     and kwargs.get('run_l3', True))

        self._reset_progress()

        # Build text_map upfront
        for i, task in enumerate(tasks):
            tid = task.get('task_id', f'_row{i+1}')
            text_map[tid] = task['prompt']

        # ── Batch API pre-computation ──────────────────────────
        batch_cont_results = {}
        if use_batch:
            from llm_detector.analyzers.continuation_api import run_continuation_batch
            from llm_detector.normalize import normalize_text
            self._append("Submitting continuation batch to Anthropic...\n", 'HEADER')
            norm_texts = []
            norm_ids = []
            for task in tasks:
                nt, _ = normalize_text(task['prompt'])
                norm_texts.append(nt)
                norm_ids.append(task.get('task_id', ''))
            batch_cont_results = run_continuation_batch(
                norm_texts, norm_ids,
                api_key=kwargs['api_key'],
                model=kwargs.get('dna_model'),
                n_samples=kwargs.get('dna_samples', 3),
                progress_fn=lambda s: self._append(f"  {s}\n"),
            )
            self._append(f"Batch complete: {len(batch_cont_results)} results.\n\n", 'HEADER')

        def _run(idx_task):
            i, task = idx_task
            extra = {}
            if use_batch:
                extra['precomputed_continuation'] = batch_cont_results.get(i)
                extra['api_key'] = None  # skip per-submission API calls
            return analyze_prompt(
                task['prompt'],
                task_id=task.get('task_id', ''),
                occupation=task.get('occupation', ''),
                attempter=task.get('attempter', ''),
                stage=task.get('stage', ''),
                **{**kwargs, **extra},
            )

        if n_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            batch_size = n_workers
            done = 0
            for start in range(0, n_tasks, batch_size):
                chunk = list(enumerate(tasks[start:start + batch_size], start))
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    chunk_results = list(pool.map(_run, chunk))
                for r in chunk_results:
                    done += 1
                    results.append(r)
                    counts[r['determination']] += 1
                    self._update_progress(done, n_tasks)
                    self._append(f"[{done}/{n_tasks}] ")
                    self._display_result(r)
        else:
            for i, task in enumerate(tasks):
                r = _run((i, task))
                results.append(r)
                counts[r['determination']] += 1
                self._update_progress(i + 1, n_tasks)
                self._append(f"[{i+1}/{n_tasks}] ")
                self._display_result(r)

        # Shadow model checks
        if self._memory_store:
            shadow_count = 0
            for r in results:
                disagreement = self._memory_store.check_shadow_disagreement(r)
                r['shadow_disagreement'] = disagreement
                r['shadow_ai_prob'] = (disagreement or {}).get('shadow_ai_prob')
                if disagreement:
                    shadow_count += 1
            if shadow_count:
                self._append(f"\nShadow model: {shadow_count} disagreements\n", 'ALERT')

        self._last_results = results
        self._last_text_map = text_map
        self._update_dashboard_metrics(self._last_results)

        # Summary
        parts = []
        for det in ['RED', 'AMBER', 'MIXED', 'YELLOW', 'REVIEW', 'GREEN']:
            ct = counts.get(det, 0)
            if ct > 0 or det in ('RED', 'AMBER', 'YELLOW', 'GREEN'):
                parts.append(f"{det}={ct}")
        self._append(f"\nSummary: {' | '.join(parts)}\n", 'HEADER')

        # Load instruction text for similarity baseline
        instruction_text = None
        instr_path = self.instructions_var.get().strip()
        if instr_path and os.path.exists(instr_path):
            with open(instr_path, 'r') as f:
                instruction_text = f.read()

        # Similarity analysis
        if not self.no_similarity_var.get() and len(results) >= 2:
            self._run_similarity(results, text_map)

        # Cross-batch memory
        if self._memory_store:
            cross_flags = self._memory_store.cross_batch_similarity(results, text_map)
            if cross_flags:
                self._append(f"\nCross-batch memory: {len(cross_flags)} matches\n", 'HEADER')
                for cf in cross_flags[:5]:
                    self._append(f"  {cf['current_id'][:15]} <-> {cf['historical_id'][:15]} "
                                 f"(MH={cf['minhash_similarity']:.2f})\n", 'DIM')
            self._memory_store.record_batch(results, text_map)

        # Yellow alerts
        yellow = [r for r in results if r['determination'] == 'YELLOW']
        if yellow:
            self._append(f"\nYELLOW ({len(yellow)} minor signals):\n", 'YELLOW')
            for r in sorted(yellow, key=lambda x: x.get('confidence', 0), reverse=True)[:10]:
                self._append(f"  {r.get('task_id', '')[:12]:12s} {r.get('occupation', '')[:40]:40s} | "
                             f"{r.get('reason', '')[:50]}\n", 'DIM')

        # Attempter profiling & channel pattern summary
        if len(results) >= 5:
            try:
                from llm_detector.reporting import (
                    profile_attempters, channel_pattern_summary,
                )
                profiles = profile_attempters(results)
                if profiles:
                    self._append(f"\nAttempter profiles: {len(profiles)} attempters\n", 'HEADER')
                    for p in profiles[:5]:
                        self._append(f"  {p['attempter'][:20]:20s} submissions={p['n_submissions']} "
                                     f"flag_rate={p['flag_rate']:.0f}%\n")
            except Exception as exc:
                logger.debug("Attempter profiling display failed: %s", exc)

            try:
                import io, sys
                buf = io.StringIO()
                old_stdout = sys.stdout
                sys.stdout = buf
                try:
                    channel_pattern_summary(results)
                finally:
                    sys.stdout = old_stdout
                summary_text = buf.getvalue()
                if summary_text.strip():
                    self._append(f"\n{summary_text}", 'DIM')
            except Exception as exc:
                logger.debug("Channel pattern summary display failed: %s", exc)

        # Financial impact
        if len(results) >= 10:
            try:
                from llm_detector.reporting import financial_impact
                impact = financial_impact(results, cost_per_prompt=self._get_cost())
                self._append(f"\nFinancial impact: waste={impact['waste_estimate']:.0f} "
                             f"projected_annual={impact.get('projected_annual_waste', 0):.0f}\n", 'HEADER')
            except Exception as exc:
                logger.debug("Financial impact display failed: %s", exc)

        # Collect baselines
        collect_path = self.collect_var.get().strip()
        if collect_path:
            from llm_detector.baselines import collect_baselines
            collect_baselines(results, collect_path)
            self._append(f"Baselines appended to {collect_path}\n", 'DIM')

    def _run_similarity(self, results, text_map):
        try:
            from llm_detector.similarity import (
                analyze_similarity, apply_similarity_adjustments,
                save_similarity_store, cross_batch_similarity,
            )
            instruction_text = None
            instr_path = self.instructions_var.get().strip()
            if instr_path and os.path.exists(instr_path):
                with open(instr_path, 'r') as f:
                    instruction_text = f.read()

            sim_pairs = analyze_similarity(
                results, text_map,
                jaccard_threshold=self._get_sim_threshold(),
                instruction_text=instruction_text,
            )
            if sim_pairs:
                self._append(f"\nSimilarity: {len(sim_pairs)} pairs flagged\n", 'HEADER')
                results[:] = apply_similarity_adjustments(results, sim_pairs, text_map)
                upgrades = [r for r in results if 'similarity_upgrade' in r]
                if upgrades:
                    self._append(f"  {len(upgrades)} determinations upgraded by similarity\n", 'ALERT')

            sim_store = self.sim_store_var.get().strip()
            if sim_store:
                cross_flags = cross_batch_similarity(results, text_map, sim_store)
                if cross_flags:
                    self._append(f"  Cross-batch similarity: {len(cross_flags)} matches\n", 'DIM')
                save_similarity_store(results, text_map, sim_store)
        except Exception as exc:
            logger.debug("Similarity analysis failed: %s", exc)

    def _save_results_csv(self):
        if not self._last_results:
            messagebox.showinfo('No results', 'Run an analysis first.')
            return
        path = self.output_csv_var.get().strip()
        if not path:
            path = filedialog.asksaveasfilename(
                defaultextension='.csv', filetypes=[('CSV', '*.csv')])
        if not path:
            return

        try:
            import pandas as pd
            flat = []
            for r in self._last_results:
                row = {k: v for k, v in r.items() if k != 'preamble_details'}
                row['preamble_details'] = str(r.get('preamble_details', []))
                flat.append(row)
            pd.DataFrame(flat).to_csv(path, index=False)
            self.status_var.set(f'Results saved to {path}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to save CSV: {e}')

    def _save_html_reports(self):
        if not self._last_results:
            messagebox.showinfo('No results', 'Run an analysis first.')
            return

        report_path = self.html_report_var.get().strip()

        # Single-text analysis: generate a report regardless of determination
        if len(self._last_results) == 1:
            if not report_path:
                report_path = filedialog.asksaveasfilename(
                    title='Save HTML Report',
                    defaultextension='.html',
                    filetypes=[('HTML files', '*.html')],
                    initialfile='report.html',
                )
            if not report_path:
                return
            try:
                from llm_detector.html_report import generate_html_report
                r = self._last_results[0]
                text = (
                    self._last_text_map.get('_single')
                    or self._last_text_map.get(r.get('task_id', ''), '')
                )
                generate_html_report(text, r, report_path)
                self.status_var.set(f'HTML report written: {report_path}')
            except Exception as e:
                messagebox.showerror('Error', str(e))
            return

        # Batch analysis: report only flagged submissions
        flagged = [r for r in self._last_results if r['determination'] in ('RED', 'AMBER', 'MIXED')]
        if not flagged:
            messagebox.showinfo('No flagged', 'No flagged submissions to report.')
            return

        if not report_path:
            report_path = filedialog.asksaveasfilename(
                title='Save HTML Report',
                defaultextension='.html',
                filetypes=[('HTML files', '*.html')],
                initialfile='batch_report.html',
            )
        if not report_path:
            return

        try:
            from llm_detector.html_report import generate_batch_html_report
            generate_batch_html_report(flagged, self._last_text_map, report_path)
            self.status_var.set(f'HTML report written: {report_path} ({len(flagged)} submissions)')
        except Exception as e:
            messagebox.showerror('Error', str(e))

    # ── Display ───────────────────────────────────────────────────────

    def _display_result(self, result):
        det = result.get('determination', 'GREEN')
        conf = result.get('confidence', 0)
        wc = result.get('word_count', 0)
        reason = result.get('reason', '')

        self._append(f"  {det}", det)
        self._append(f" | conf={conf:.2f} | words={wc}\n")
        self._append(f"  {reason}\n", 'DIM')

        if not self.show_details_var.get():
            self._append('\n')
            return

        # Calibrated confidence
        cal_conf = result.get('calibrated_confidence')
        if cal_conf is not None:
            self._append(f"  Calibrated: {cal_conf:.2f}", 'HEADER')
            stratum = result.get('calibration_stratum', '')
            conformity = result.get('conformity_level', '')
            if stratum or conformity:
                self._append(f" ({stratum}/{conformity})", 'DIM')
            self._append('\n')

        # Channel details
        cd = result.get('channel_details', {})
        channels = cd.get('channels', {})
        mode = cd.get('mode', '?')
        self._append(f"  Mode: {mode}\n", 'HEADER')

        if channels:
            self._append("  Channels:\n", 'HEADER')
            for ch_name, info in channels.items():
                sev = info.get('severity', 'GREEN')
                score = info.get('score', 0)
                sufficient = info.get('data_sufficient', True)
                disabled = info.get('disabled', False)
                eligible = info.get('mode_eligible', True)
                role = info.get('role', '')

                status_parts = []
                if disabled:
                    status_parts.append('DISABLED')
                elif role:
                    status_parts.append(role)
                if not sufficient:
                    status_parts.append('no-data')
                if not eligible:
                    status_parts.append('mode-ineligible')
                status = f" [{', '.join(status_parts)}]" if status_parts else ''

                self._append(f"    {ch_name:20s} ", None)
                self._append(f"{sev:6s}", sev)
                self._append(f" score={score:.2f}{status}\n")

        triggering_rule = cd.get('triggering_rule', '')
        if triggering_rule:
            self._append(f"  Rule: {triggering_rule}\n", 'DIM')

        # Verbose details
        if self.verbose_var.get():
            self._display_verbose(result)

        # Attack types
        attack_types = result.get('norm_attack_types', [])
        if attack_types:
            self._append(f"  Attacks neutralized: {', '.join(attack_types)}\n", 'ALERT')

        # Binoculars
        bino = result.get('binoculars_score', 0)
        bino_det = result.get('binoculars_determination')
        if bino and bino > 0:
            self._append(f"  Binoculars: {bino:.4f}", None)
            if bino_det:
                self._append(f" ({bino_det})", bino_det)
            self._append('\n')

        # Shadow model
        shadow = result.get('shadow_disagreement')
        if shadow:
            self._append(f"  Shadow: {shadow.get('interpretation', 'disagrees')}\n", 'ALERT')
            self._append(f"    Rule={shadow.get('rule_determination', '?')}, "
                         f"Model={shadow.get('shadow_ai_prob', 0):.1%} AI\n", 'DIM')

        # Detection spans
        spans = result.get('detection_spans', [])
        if spans:
            span_sources = Counter(s.get('source', '?') for s in spans)
            span_str = ', '.join(f"{src}={ct}" for src, ct in span_sources.items())
            self._append(f"  Detection spans: {len(spans)} ({span_str})\n", 'DIM')

        self._append('\n')

    def _display_verbose(self, r):
        # ── Normalization & Language Gate ──────────────────────────────
        self._append("  ── Normalization & Language Gate ──\n", 'HEADER')
        self._append(
            f"  NORM: obfuscation_delta={r.get('norm_obfuscation_delta', 0):.1%}"
            f"  invisible={r.get('norm_invisible_chars', 0)}"
            f"  homoglyphs={r.get('norm_homoglyphs', 0)}\n", 'DIM')
        attacks = r.get('norm_attack_types', [])
        if attacks:
            self._append(f"  Attacks neutralized: {', '.join(attacks)}\n", 'ALERT')
        self._append(
            f"  GATE: {r.get('lang_support_level', 'SUPPORTED')}"
            f"  fw_coverage={r.get('lang_fw_coverage', 0):.3f}"
            f"  non_latin={r.get('lang_non_latin_ratio', 0):.3f}\n", 'DIM')

        # ── Preamble ───────────────────────────────────────────────────
        self._append("  ── Preamble ──\n", 'HEADER')
        self._append(
            f"  score={r.get('preamble_score', 0):.3f}"
            f"  severity={r.get('preamble_severity', '-')}"
            f"  matched_patterns={r.get('preamble_hits', 0)}\n", 'DIM')
        preamble_details = r.get('preamble_details', [])
        if preamble_details:
            self._append(f"  patterns: {', '.join(str(p) for p in preamble_details[:_MAX_PREAMBLE_PATTERNS])}\n", 'DIM')

        # ── Fingerprint ────────────────────────────────────────────────
        self._append("  ── Fingerprint ──\n", 'HEADER')
        self._append(
            f"  score={r.get('fingerprint_score', 0):.3f}"
            f"  hits={r.get('fingerprint_hits', 0)}\n", 'DIM')

        # ── Prompt Signature ───────────────────────────────────────────
        self._append("  ── Prompt Signature ──\n", 'HEADER')
        self._append(
            f"  composite={r.get('prompt_signature_composite', 0):.3f}"
            f"  CFD={r.get('prompt_signature_cfd', 0):.4f}"
            f"  MFSR={r.get('prompt_signature_mfsr', 0):.3f}"
            f"  frames={r.get('prompt_signature_distinct_frames', 0)}\n", 'DIM')
        self._append(
            f"  framing={r.get('prompt_signature_framing', 0)}/3"
            f"  cond_density={r.get('prompt_signature_conditional_density', 0):.4f}"
            f"  meta_design={r.get('prompt_signature_meta_design', 0)}\n", 'DIM')
        self._append(
            f"  contractions={r.get('prompt_signature_contractions', 0)}"
            f"  must_rate={r.get('prompt_signature_must_rate', 0):.4f}"
            f"  numbered_criteria={r.get('prompt_signature_numbered_criteria', 0)}\n", 'DIM')

        # ── Instruction Density (IDI) ──────────────────────────────────
        self._append("  ── Instruction Density (IDI) ──\n", 'HEADER')
        self._append(
            f"  IDI={r.get('instruction_density_idi', 0):.2f}"
            f"  imperatives={r.get('instruction_density_imperatives', 0)}"
            f"  conditionals={r.get('instruction_density_conditionals', 0)}"
            f"  binary_specs={r.get('instruction_density_binary_specs', 0)}"
            f"  missing_refs={r.get('instruction_density_missing_refs', 0)}"
            f"  flag_count={r.get('instruction_density_flag_count', 0)}\n", 'DIM')

        # ── Voice Dissonance (VSD) ─────────────────────────────────────
        self._append("  ── Voice Dissonance (VSD) ──\n", 'HEADER')
        self._append(
            f"  VSD={r.get('voice_dissonance_vsd', 0):.2f}"
            f"  voice_score={r.get('voice_dissonance_voice_score', 0):.2f}"
            f"  spec_score={r.get('voice_dissonance_spec_score', 0):.2f}"
            f"  voice_gated={r.get('voice_dissonance_voice_gated', False)}\n", 'DIM')
        self._append(
            f"  casual_markers={r.get('voice_dissonance_casual_markers', 0)}"
            f"  misspellings={r.get('voice_dissonance_misspellings', 0)}"
            f"  camel_cols={r.get('voice_dissonance_camel_cols', 0)}"
            f"  calcs={r.get('voice_dissonance_calcs', 0)}"
            f"  hedges={r.get('voice_dissonance_hedges', 0)}"
            f"  SSI={r.get('ssi_triggered', False)}\n", 'DIM')

        # ── Pack Diagnostics ───────────────────────────────────────────
        self._append("  ── Pack Diagnostics ──\n", 'HEADER')
        self._append(
            f"  constraint={r.get('pack_constraint_score', 0):.4f}"
            f"  exec_spec={r.get('pack_exec_spec_score', 0):.4f}"
            f"  schema={r.get('pack_schema_score', 0):.4f}"
            f"  families={r.get('pack_active_families', 0)}"
            f"  prompt_boost={r.get('pack_prompt_boost', 0):.4f}"
            f"  idi_boost={r.get('pack_idi_boost', 0):.4f}\n", 'DIM')

        # ── Stylometry ─────────────────────────────────────────────────
        self._append("  ── Stylometry ──\n", 'HEADER')
        self._append(
            f"  fw_ratio={r.get('stylo_fw_ratio', 0):.4f}"
            f"  sent_dispersion={r.get('stylo_sent_dispersion', 0):.4f}"
            f"  TTR={r.get('stylo_ttr', 0):.4f}"
            f"  avg_word_len={r.get('stylo_avg_word_len', 0):.3f}"
            f"  short_word_ratio={r.get('stylo_short_word_ratio', 0):.4f}"
            f"  mask_count={r.get('stylo_mask_count', 0)}\n", 'DIM')

        # ── Windowing ──────────────────────────────────────────────────
        self._append("  ── Windowing ──\n", 'HEADER')
        self._append(
            f"  max={r.get('window_max_score', 0):.4f}"
            f"  mean={r.get('window_mean_score', 0):.4f}"
            f"  var={r.get('window_variance', 0):.4f}"
            f"  hot_span={r.get('window_hot_span', 0)}"
            f"  n_windows={r.get('window_n_windows', 0)}"
            f"  mixed={r.get('window_mixed_signal', False)}\n", 'DIM')
        self._append(
            f"  fw_traj_cv={r.get('window_fw_trajectory_cv', 0):.4f}"
            f"  comp_traj_mean={r.get('window_comp_trajectory_mean', 0):.4f}"
            f"  comp_traj_cv={r.get('window_comp_trajectory_cv', 0):.4f}"
            f"  changepoint={r.get('window_changepoint') or 'none'}\n", 'DIM')

        # ── Self-Similarity (NSSI) ─────────────────────────────────────
        self._append("  ── Self-Similarity (NSSI) ──\n", 'HEADER')
        self._append(
            f"  NSSI={r.get('self_similarity_nssi_score', 0):.4f}"
            f"  signals={r.get('self_similarity_nssi_signals', 0)}"
            f"  det={r.get('self_similarity_determination') or 'n/a'}"
            f"  conf={r.get('self_similarity_confidence', 0):.3f}\n", 'DIM')
        self._append(
            f"  formulaic={r.get('self_similarity_formulaic_density', 0):.4f}"
            f"  power_adj={r.get('self_similarity_power_adj_density', 0):.4f}"
            f"  demonstrative={r.get('self_similarity_demonstrative_density', 0):.4f}"
            f"  transition={r.get('self_similarity_transition_density', 0):.4f}\n", 'DIM')
        self._append(
            f"  scare_quote={r.get('self_similarity_scare_quote_density', 0):.4f}"
            f"  emdash={r.get('self_similarity_emdash_density', 0):.4f}"
            f"  this_the_start={r.get('self_similarity_this_the_start_rate', 0):.4f}"
            f"  section_depth={r.get('self_similarity_section_depth', 0)}\n", 'DIM')
        self._append(
            f"  sent_len_cv={r.get('self_similarity_sent_length_cv', 0):.4f}"
            f"  comp_ratio={r.get('self_similarity_comp_ratio', 0):.4f}"
            f"  hapax_ratio={r.get('self_similarity_hapax_ratio', 0):.4f}"
            f"  hapax_count={r.get('self_similarity_hapax_count', 0)}"
            f"  unique_words={r.get('self_similarity_unique_words', 0)}\n", 'DIM')
        self._append(
            f"  shuffled_comp={r.get('self_similarity_shuffled_comp_ratio', 0):.4f}"
            f"  struct_delta={r.get('self_similarity_structural_compression_delta', 0):.4f}\n", 'DIM')

        # ── Continuation / DNA-GPT ─────────────────────────────────────
        self._append("  ── Continuation (DNA-GPT) ──\n", 'HEADER')
        self._append(
            f"  bscore={r.get('continuation_bscore', 0):.4f}"
            f"  bscore_max={r.get('continuation_bscore_max', 0):.4f}"
            f"  det={r.get('continuation_determination') or 'n/a'}"
            f"  conf={r.get('continuation_confidence', 0):.3f}"
            f"  n_samples={r.get('continuation_n_samples', 0)}"
            f"  mode={r.get('continuation_mode') or 'n/a'}\n", 'DIM')
        self._append(
            f"  NCD={r.get('continuation_ncd', 0):.4f}"
            f"  internal_overlap={r.get('continuation_internal_overlap', 0):.4f}"
            f"  cond_surprisal={r.get('continuation_cond_surprisal', 0):.4f}"
            f"  repeat4={r.get('continuation_repeat4', 0):.4f}"
            f"  TTR={r.get('continuation_ttr', 0):.4f}\n", 'DIM')
        self._append(
            f"  composite={r.get('continuation_composite', 0):.4f}"
            f"  comp_var={r.get('continuation_composite_variance', 0):.4f}"
            f"  comp_stab={r.get('continuation_composite_stability', 0):.4f}"
            f"  impr_rate={r.get('continuation_improvement_rate', 0):.4f}\n", 'DIM')
        self._append(
            f"  ncd_mat_mean={r.get('continuation_ncd_matrix_mean', 0):.4f}"
            f"  ncd_mat_var={r.get('continuation_ncd_matrix_variance', 0):.4f}"
            f"  ncd_mat_min={r.get('continuation_ncd_matrix_min', 0):.4f}\n", 'DIM')

        # ── Perplexity ─────────────────────────────────────────────────
        self._append("  ── Perplexity ──\n", 'HEADER')
        self._append(
            f"  ppl={r.get('perplexity_value', 0):.3f}"
            f"  det={r.get('perplexity_determination') or 'n/a'}"
            f"  conf={r.get('perplexity_confidence', 0):.3f}\n", 'DIM')
        self._append(
            f"  surp_var={r.get('surprisal_variance', 0):.4f}"
            f"  surp_var_1st={r.get('surprisal_first_half_var', 0):.4f}"
            f"  surp_var_2nd={r.get('surprisal_second_half_var', 0):.4f}"
            f"  volatility_decay={r.get('volatility_decay_ratio', 1):.4f}\n", 'DIM')
        self._append(
            f"  binoculars={r.get('binoculars_score', 0):.4f}"
            f"  bino_det={r.get('binoculars_determination') or 'n/a'}"
            f"  comp_ratio={r.get('perplexity_comp_ratio', 0):.4f}"
            f"  zlib_ppl={r.get('perplexity_zlib_normalized_ppl', 0):.4f}"
            f"  comp_ppl={r.get('perplexity_comp_ppl_ratio', 0):.4f}\n", 'DIM')

        # ── Surprisal Trajectory ───────────────────────────────────────
        self._append("  ── Surprisal Trajectory ──\n", 'HEADER')
        self._append(
            f"  traj_cv={r.get('surprisal_trajectory_cv', 0):.4f}"
            f"  var_of_var={r.get('surprisal_var_of_var', 0):.4f}"
            f"  stationarity={r.get('surprisal_stationarity', 0):.4f}\n", 'DIM')

        # ── TOCSIN ─────────────────────────────────────────────────────
        self._append("  ── TOCSIN (Token Cohesiveness) ──\n", 'HEADER')
        self._append(
            f"  cohesiveness={r.get('tocsin_cohesiveness', 0):.4f}"
            f"  std={r.get('tocsin_cohesiveness_std', 0):.4f}"
            f"  det={r.get('tocsin_determination') or 'n/a'}"
            f"  conf={r.get('tocsin_confidence', 0):.3f}\n", 'DIM')

        # ── Semantic Resonance ─────────────────────────────────────────
        self._append("  ── Semantic Resonance ──\n", 'HEADER')
        self._append(
            f"  ai_score={r.get('semantic_resonance_ai_score', 0):.4f}"
            f"  human_score={r.get('semantic_resonance_human_score', 0):.4f}"
            f"  ai_mean={r.get('semantic_resonance_ai_mean', 0):.4f}"
            f"  human_mean={r.get('semantic_resonance_human_mean', 0):.4f}\n", 'DIM')
        self._append(
            f"  delta={r.get('semantic_resonance_delta', 0):.4f}"
            f"  det={r.get('semantic_resonance_determination') or 'n/a'}"
            f"  conf={r.get('semantic_resonance_confidence', 0):.3f}\n", 'DIM')

    # ── Memory & Learning Actions ─────────────────────────────────────

    def _refresh_recent_samples(self):
        """Populate the recent-samples listbox from _last_results."""
        self._recent_listbox.delete(0, tk.END)
        for r in self._last_results:
            tid = r.get('task_id', '?')
            det = r.get('determination', '?')
            preview = (self._last_text_map.get(tid, '') or '')[:60].replace('\n', ' ')
            self._recent_listbox.insert(tk.END, f"[{det}] {tid}  — {preview}")

    def _on_recent_select(self, event=None):
        sel = self._recent_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self._last_results):
            return
        r = self._last_results[idx]
        tid = r.get('task_id', '?')
        text = (self._last_text_map.get(tid, '') or '')[:500]
        self._recent_preview.configure(state='normal')
        self._recent_preview.delete('1.0', tk.END)
        self._recent_preview.insert('1.0', text or '(no text available)')
        self._recent_preview.configure(state='disabled')

    def _quick_confirm(self, label):
        """Record a ground-truth confirmation for the currently selected recent sample."""
        sel = self._recent_listbox.curselection()
        if not sel:
            messagebox.showinfo('Select sample', 'Select a sample from the list first.')
            return
        idx = sel[0]
        if idx >= len(self._last_results):
            return
        reviewer = self.quick_reviewer_var.get().strip()
        if not reviewer:
            messagebox.showinfo('Reviewer required', 'Enter a reviewer name in the Quick Confirm section.')
            return
        if not self._ensure_memory():
            return
        r = self._last_results[idx]
        task_id = r.get('task_id', '')
        self._memory_store.record_confirmation(task_id, label, verified_by=reviewer)
        self.status_var.set(f'Confirmed: {task_id} = {label} by {reviewer}')
        # Visual feedback — remove confirmed item from list
        self._recent_listbox.delete(idx)
        # Refresh fusion readiness display
        self.root.after(0, self._refresh_fusion_readiness)

    def _memory_summary(self):
        if not self._ensure_memory():
            return
        import io
        import sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            self._memory_store.print_summary()
        finally:
            sys.stdout = old_stdout
        self._append(buf.getvalue())

    def _record_confirmation(self):
        if not self._ensure_memory():
            return
        task_id = self.confirm_task_var.get().strip()
        label = self.confirm_label_var.get()
        reviewer = self.confirm_reviewer_var.get().strip()
        if not task_id or not reviewer:
            messagebox.showinfo('Missing fields', 'Task ID and Reviewer are required.')
            return
        self._memory_store.record_confirmation(task_id, label, verified_by=reviewer)
        msg = f'Confirmed: {task_id} = {label} by {reviewer}'
        self.status_var.set(msg)
        self._append(f'{msg}\n', 'HEADER')
        # Clear task ID field for next entry
        self.confirm_task_var.set('')
        # Refresh fusion readiness display
        self.root.after(0, self._refresh_fusion_readiness)

    def _show_attempter_history(self):
        if not self._ensure_memory():
            return
        name = self.attempter_history_var.get().strip()
        if not name:
            self.root.after(0, lambda: messagebox.showinfo('Required', 'Enter an attempter name.'))
            return
        import io
        import sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            self._memory_store.print_attempter_history(name)
        finally:
            sys.stdout = old_stdout
        self._append(buf.getvalue())

    def _rebuild_shadow(self):
        if not self._ensure_memory():
            return
        pkg = self._memory_store.rebuild_shadow_model()
        if pkg:
            self._append(f"Shadow model rebuilt: AUC={pkg['cv_auc']:.3f}\n", 'HEADER')
        else:
            self._append("Shadow model: insufficient labeled data\n", 'ALERT')

    def _refresh_fusion_readiness(self):
        if not self._memory_store:
            self._ml_fusion_progress_var.set('0 / 200 (0%)')
            self._ml_fusion_balance_var.set('AI: 0  |  Human: 0')
            self._ml_fusion_model_var.set('No memory store loaded')
            self._ml_fusion_progress_bar['value'] = 0
            self._train_fusion_btn.configure(state='disabled')
            self._ml_fusion_checkbox.configure(state='disabled')
            return

        readiness = self._memory_store.get_fusion_readiness()
        total = readiness['total_confirmed']
        pct = readiness['progress_pct']
        self._ml_fusion_progress_var.set(
            f"{total} / {readiness['min_required']} ({pct:.0f}%)")
        self._ml_fusion_balance_var.set(
            f"AI: {readiness['n_ai']}  |  Human: {readiness['n_human']}")
        self._ml_fusion_progress_bar['value'] = pct

        if readiness['ready']:
            self._train_fusion_btn.configure(state='normal')
        else:
            self._train_fusion_btn.configure(state='disabled')

        info = readiness.get('model_info')
        if info:
            self._ml_fusion_model_var.set(
                f"Trained {info['trained_at'][:10]}  |  "
                f"AUC={info['cv_auc']:.3f}  |  "
                f"n={info['n_samples']}  |  "
                f"{info['algorithm']}")
            self._ml_fusion_checkbox.configure(state='normal')
        else:
            self._ml_fusion_model_var.set('No model trained')
            self._ml_fusion_checkbox.configure(state='disabled')
            self._ml_fusion_enabled_var.set(False)

    def _train_fusion_model(self):
        if not self._ensure_memory():
            return
        from llm_detector.ml_fusion import train_fusion_model
        result = train_fusion_model(self._memory_store)
        if result and 'error' not in result:
            self._append(
                f"ML Fusion model trained: AUC={result['cv_auc']:.3f}, "
                f"n={result['n_samples']}, features={result['n_features']}\n", 'HEADER')
            if result.get('top_features'):
                self._append("  Top features:\n", 'HEADER')
                for feat, imp in result['top_features'][:10]:
                    self._append(f"    {feat:<45} {imp:.4f}\n", 'DIM')
        else:
            msg = result.get('error', 'Unknown error') if result else 'Training failed'
            self._append(f"ML Fusion training: {msg}\n", 'ALERT')
        self.root.after(0, self._refresh_fusion_readiness)

    def _rebuild_centroids(self):
        if not self._ensure_memory():
            return
        corpus = self.labeled_corpus_var.get().strip()
        if not corpus:
            self.root.after(0, lambda: messagebox.showinfo('Required', 'Set a labeled corpus JSONL path.'))
            return
        result = self._memory_store.rebuild_semantic_centroids(corpus)
        if result:
            self._append(f"Centroids rebuilt: separation={result['separation']:.4f}\n", 'HEADER')
        else:
            self._append("Centroids: insufficient labeled text\n", 'ALERT')

    def _discover_lexicon(self):
        if not self._ensure_memory():
            return
        corpus = self.labeled_corpus_var.get().strip()
        if not corpus:
            self.root.after(0, lambda: messagebox.showinfo('Required', 'Set a labeled corpus JSONL path.'))
            return
        candidates = self._memory_store.discover_lexicon_candidates(corpus)
        n_new = sum(1 for c in candidates
                    if not c.get('already_in_fingerprints') and not c.get('already_in_packs'))
        self._append(f"Lexicon discovery: {len(candidates)} candidates ({n_new} new)\n", 'HEADER')

    def _rebuild_all(self):
        if not self._ensure_memory():
            return
        self._append("Rebuilding all learned artifacts...\n", 'HEADER')

        cal = self._memory_store.rebuild_calibration()
        if cal:
            self._append(f"  Calibration: {cal['n_calibration']} samples\n")
            self._cal_table = cal
        else:
            self._append("  Calibration: insufficient data\n", 'ALERT')

        shadow = self._memory_store.rebuild_shadow_model()
        if shadow:
            self._append(f"  Shadow model: AUC={shadow['cv_auc']:.3f}\n")
        else:
            self._append("  Shadow model: insufficient data\n", 'ALERT')

        corpus = self.labeled_corpus_var.get().strip()
        if corpus:
            centroids = self._memory_store.rebuild_semantic_centroids(corpus)
            if centroids:
                self._append(f"  Centroids: separation={centroids['separation']:.4f}\n")
            else:
                self._append("  Centroids: insufficient text\n", 'ALERT')
            candidates = self._memory_store.discover_lexicon_candidates(corpus)
            n_new = sum(1 for c in candidates
                        if not c.get('already_in_fingerprints') and not c.get('already_in_packs'))
            self._append(f"  Lexicon: {len(candidates)} candidates ({n_new} new)\n")
        else:
            self._append("  Centroids/Lexicon: skipped (no labeled corpus)\n", 'DIM')

        self._append("Rebuild complete.\n", 'HEADER')

    # ── Calibration & Baselines Actions ───────────────────────────────

    def _build_calibration(self):
        jsonl_path = self.calibrate_var.get().strip()
        if not jsonl_path or not os.path.exists(jsonl_path):
            self.root.after(0, lambda: messagebox.showinfo('Required', 'Select a valid baselines JSONL file.'))
            return
        from llm_detector.calibration import calibrate_from_baselines, save_calibration
        cal = calibrate_from_baselines(jsonl_path)
        if cal is None:
            self._append("Calibration failed: need >= 20 labeled human samples\n", 'ALERT')
            return
        cal_path = self.cal_table_var.get().strip()
        if not cal_path:
            cal_path = jsonl_path.replace('.jsonl', '_calibration.json')
            self.cal_table_var.set(cal_path)
        save_calibration(cal, cal_path)
        self._cal_table = cal
        self._append(f"Calibration built: {cal['n_calibration']} records, "
                     f"{len(cal.get('strata', {}))} strata\n", 'HEADER')
        self._append(f"  Global quantiles: {cal['global']}\n", 'DIM')
        self._append(f"  Saved to: {cal_path}\n", 'DIM')

    def _rebuild_calibration(self):
        if not self._ensure_memory():
            return
        cal = self._memory_store.rebuild_calibration()
        if cal:
            self._cal_table = cal
            self._append(f"Calibration rebuilt from memory: {cal['n_calibration']} samples\n", 'HEADER')
        else:
            self._append("Calibration rebuild: insufficient data\n", 'ALERT')

    def _analyze_baselines(self):
        jsonl_path = self.baselines_jsonl_var.get().strip()
        if not jsonl_path or not os.path.exists(jsonl_path):
            self.root.after(0, lambda: messagebox.showinfo('Required', 'Select a valid baselines JSONL file.'))
            return
        csv_path = self.baselines_csv_var.get().strip() or None

        import io
        import sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            from llm_detector.baselines import analyze_baselines
            analyze_baselines(jsonl_path, output_csv=csv_path)
        finally:
            sys.stdout = old_stdout
        self._append(buf.getvalue())

    def _run_calibration_report(self):
        jsonl_path = self.cal_report_jsonl_var.get().strip()
        if not jsonl_path or not os.path.exists(jsonl_path):
            self.root.after(0, lambda: messagebox.showinfo('Required', 'Select a valid labeled JSONL file.'))
            return
        csv_path = self.cal_report_csv_var.get().strip() or None

        import io
        import sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            from llm_detector.cli import calibration_report
            calibration_report(
                jsonl_path,
                cal_table=self._cal_table,
                output_csv=csv_path,
            )
        finally:
            sys.stdout = old_stdout
        self._append(buf.getvalue())

    def _start_labeling_session(self):
        if not self._last_results:
            messagebox.showinfo('No results', 'Run an analysis first.')
            return
        reviewer = self.label_reviewer_var.get().strip()
        if not reviewer:
            messagebox.showinfo('Required', 'Enter a reviewer name.')
            return

        from llm_detector.cli import _sort_for_labeling
        results = _sort_for_labeling(self._last_results)

        if self.label_skip_green_var.get():
            results = [r for r in results if r['determination'] != 'GREEN']
        if self.label_skip_red_var.get():
            results = [r for r in results if r['determination'] != 'RED']

        max_val = self.label_max_var.get().strip()
        if max_val.isdigit() and int(max_val) > 0:
            results = results[:int(max_val)]

        if not results:
            messagebox.showinfo('No items', 'No items to label with current settings.')
            return

        from datetime import datetime
        output_path = self.label_output_var.get().strip()
        if not output_path:
            output_path = f"beet_labels_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
            self.label_output_var.set(output_path)

        def on_complete(stats):
            self._append(
                f"Labeling complete: {stats.get('labeled_ai', 0)} AI, "
                f"{stats.get('labeled_human', 0)} human, "
                f"{stats.get('labeled_unsure', 0)} unsure, "
                f"{stats.get('skipped', 0)} skipped\n", 'HEADER'
            )

        _LabelingDialog(self.root, results, self._last_text_map,
                        output_path=output_path, reviewer=reviewer,
                        store=self._memory_store, on_complete=on_complete)


    # ── Tab 5: Reports ─────────────────────────────────────────────────

    def _build_reports_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text='  Reports  ')

        actions = ttk.Frame(tab)
        actions.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(actions, text='Refresh Reports',
                   command=self._refresh_reports).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text='Export Baselines',
                   command=self._export_baselines).pack(side=tk.LEFT, padx=4)

        report_frame = ttk.Frame(tab)
        report_frame.pack(fill=tk.BOTH, expand=True)
        self.report_output = tk.Text(report_frame, wrap=tk.WORD,
                                     font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(report_frame, orient=tk.VERTICAL,
                                  command=self.report_output.yview)
        self.report_output.configure(yscrollcommand=scrollbar.set)
        self.report_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _report_append(self, text):
        self.root.after(0, lambda: (
            self.report_output.insert(tk.END, text),
            self.report_output.see(tk.END)))

    # ── Tab 6: Quick Reference ─────────────────────────────────────────

    def _build_quick_reference_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text='  Quick Reference  ')

        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        ref_text = tk.Text(tab, wrap=tk.WORD, font=('Consolas', 9),
                           state='disabled', yscrollcommand=scrollbar.set)
        ref_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=ref_text.yview)

        content = _QUICK_REFERENCE_TEXT
        ref_text.configure(state='normal')
        ref_text.insert('1.0', content)
        ref_text.configure(state='disabled')

    # ── Tab 7: Precheck ────────────────────────────────────────────────

    def _build_precheck_tab(self, notebook):
        tab = ttk.Frame(notebook, padding=8)
        notebook.add(tab, text='  Precheck  ')

        ttk.Label(tab, text=(
            'Required and optional dependencies for the detection pipeline.\n'
            '\u2705 = available   \u2757 = missing (optional, pipeline still works)   \u274c = missing (may break analysis)'
        ), style='DashboardSubtitle.TLabel').pack(anchor='w', pady=(0, 8))

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(anchor='w', pady=(0, 6))
        ttk.Button(btn_frame, text='Refresh',
                   command=lambda: self._refresh_precheck(tree)).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_frame, text='Install Selected',
                   command=lambda: self._install_selected_deps(tree)).pack(side=tk.LEFT)

        columns = ('Status', 'Component', 'Category', 'Notes')
        tree = ttk.Treeview(tab, columns=columns, show='headings', height=20,
                            selectmode='extended')
        for col in columns:
            tree.heading(col, text=col)
        tree.column('Status', width=60, anchor='center')
        tree.column('Component', width=220)
        tree.column('Category', width=120)
        tree.column('Notes', width=400)

        tree_scroll = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=tree_scroll.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._refresh_precheck(tree)

    def _refresh_precheck(self, tree):
        for item in tree.get_children():
            tree.delete(item)
        for status, name, category, notes in _check_dependencies():
            tree.insert('', tk.END, values=(status, name, category, notes))

    def _install_selected_deps(self, tree):
        """Install selected missing optional dependencies via pip."""
        selected = tree.selection()
        if not selected:
            messagebox.showinfo('Install', 'Select one or more missing (\u2757) dependencies to install.')
            return

        packages = []
        for item_id in selected:
            values = tree.item(item_id, 'values')
            status, name = values[0], values[1]
            if status != '\u2757':
                continue
            spec = _PIP_INSTALL_MAP.get(name)
            if spec:
                packages.append(spec)

        if not packages:
            messagebox.showinfo('Install', 'No installable missing dependencies selected.\n'
                                'Select rows marked with \u2757 to install them.')
            return

        # Deduplicate (e.g. multiple nlp deps map to same extra)
        packages = sorted(set(packages))
        self.status_var.set(f'Installing {", ".join(packages)}\u2026')

        def _do_install():
            try:
                subprocess.check_call(
                    [_real_python(), '-m', 'pip', 'install'] + packages,
                    stdout=subprocess.DEVNULL,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.root.after(0, lambda: (
                    self.status_var.set('Installation failed.'),
                    messagebox.showerror(
                        'Install Failed',
                        'pip install failed. Try manually:\n    pip install '
                        + ' '.join(packages),
                    ),
                ))
                return
            self.root.after(0, lambda: (
                self.status_var.set('Installation complete.'),
                self._refresh_precheck(tree),
                messagebox.showinfo('Install', 'Dependencies installed successfully.'),
            ))

        threading.Thread(target=_do_install, daemon=True).start()

    def _collect_dna_hits(self, results):
        """Return DNA-GPT-positive results where continuation severity is RED/AMBER/YELLOW."""
        hits = []
        for r in results:
            channels = (r.get('channel_details') or {}).get('channels', {})
            cont = channels.get('continuation') or {}
            severity = cont.get('severity') or r.get('continuation_determination')
            if severity not in ('RED', 'AMBER', 'YELLOW'):
                continue

            bscore = r.get('continuation_bscore')
            if bscore is None:
                bscore = cont.get('score')
            mode = (
                r.get('continuation_mode')
                or cont.get('mode')
                or r.get('mode')
                or 'n/a'
            )
            hits.append({
                'task_id': r.get('task_id', '') or '(unknown task)',
                'overall': r.get('determination', '?'),
                'severity': severity,
                'bscore': bscore,
                'mode': mode,
            })
        return hits

    def _refresh_reports(self):
        self.report_output.delete('1.0', tk.END)
        if not self._last_results:
            self._report_append('No results available. Run a batch analysis first.\n')
            return

        results = self._last_results
        counts = Counter(r['determination'] for r in results)
        self._report_append(f"{'=' * 60}\n")
        self._report_append(f"  BATCH SUMMARY (n={len(results)})\n")
        self._report_append(f"{'=' * 60}\n")
        for det in ['RED', 'AMBER', 'MIXED', 'YELLOW', 'REVIEW', 'GREEN']:
            ct = counts.get(det, 0)
            if ct > 0:
                pct = ct / len(results) * 100
                self._report_append(f"  {det:>8}: {ct:>4} ({pct:.1f}%)\n")

        # Attempter profiling
        if len(results) >= 5:
            try:
                from llm_detector.reporting import profile_attempters
                profiles = profile_attempters(results)
                if profiles:
                    self._report_append(f"\n{'=' * 60}\n")
                    self._report_append("  ATTEMPTER RISK PROFILES\n")
                    self._report_append(f"{'=' * 60}\n")
                    for p in profiles[:20]:
                        self._report_append(
                            f"  {p['attempter'][:20]:20s} "
                            f"n={p['n_submissions']:>3} "
                            f"flag={p['flag_rate']:.0f}% "
                            f"conf={p.get('mean_confidence', 0):.2f}\n")
            except Exception as exc:
                logger.debug("Report attempter profiling failed: %s", exc)

        # Channel pattern summary
        flagged = [r for r in results
                   if r['determination'] in ('RED', 'AMBER', 'MIXED')]
        if flagged:
            self._report_append(f"\n{'=' * 60}\n")
            self._report_append("  CHANNEL PATTERNS (flagged submissions)\n")
            self._report_append(f"{'=' * 60}\n")
            channel_counts = Counter()
            for r in flagged:
                cd = r.get('channel_details', {}).get('channels', {})
                for ch_name, info in cd.items():
                    if info.get('severity') not in ('GREEN', None):
                        channel_counts[ch_name] += 1
            for ch, ct in channel_counts.most_common():
                self._report_append(f"  {ch:20s}: {ct} flags\n")

        # DNA-GPT positives (API or offline local)
        dna_hits = self._collect_dna_hits(results)
        if dna_hits:
            self._report_append(f"\n{'=' * 60}\n")
            self._report_append("  DNA-GPT POSITIVE CONTINUATIONS (offline/local-friendly)\n")
            self._report_append(f"{'=' * 60}\n")
            for hit in dna_hits:
                bscore = hit.get('bscore')
                bscore_str = f"{bscore:.3f}" if bscore is not None else 'n/a'
                self._report_append(
                    f"  {hit['task_id'][:_TASK_ID_DISPLAY_LEN]:{_TASK_ID_DISPLAY_LEN}s} "
                    f"cont={hit['severity']:<6} "
                    f"bscore={bscore_str} "
                    f"mode={hit['mode']} "
                    f"overall={hit['overall']}\n"
                )
        else:
            self._report_append(
                "\nNo DNA-GPT-positive continuations found in current results.\n"
            )

        # Financial impact
        if len(results) >= 10:
            try:
                from llm_detector.reporting import financial_impact
                impact = financial_impact(results, cost_per_prompt=self._get_cost())
                self._report_append(f"\n{'=' * 60}\n")
                self._report_append("  FINANCIAL IMPACT ESTIMATE\n")
                self._report_append(f"{'=' * 60}\n")
                self._report_append(
                    f"  Total submissions:     {impact['total_submissions']}\n")
                self._report_append(
                    f"  Flag rate:             {impact['flag_rate']:.1%}\n")
                self._report_append(
                    f"  Waste estimate:        ${impact['waste_estimate']:,.0f}\n")
                self._report_append(
                    f"  Projected annual:      ${impact.get('projected_annual_waste', 0):,.0f}\n")
            except Exception as exc:
                logger.debug("Report financial impact failed: %s", exc)

    def _export_baselines(self):
        if not self._last_results:
            messagebox.showinfo('Export', 'No results to export. Run analysis first.')
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.jsonl',
            filetypes=[('JSONL', '*.jsonl'), ('All', '*.*')])
        if not path:
            return
        try:
            from llm_detector.baselines import collect_baselines
            collect_baselines(self._last_results, path)
            messagebox.showinfo('Baselines', f'Results appended to {path}')
        except Exception as e:
            messagebox.showerror('Baselines Error', str(e))

    def _launch_dashboard(self):
        """Launch the Streamlit web dashboard in a background process."""
        spec = importlib.util.find_spec('llm_detector.dashboard')
        if spec is None or spec.origin is None:
            messagebox.showerror(
                'Dashboard Not Found',
                'llm_detector.dashboard module could not be located.\n'
                'Ensure the package is properly installed.',
            )
            return
        dashboard_path = os.path.realpath(spec.origin)
        pkg_dir = os.path.realpath(os.path.dirname(__file__))
        if not dashboard_path.startswith(pkg_dir + os.sep):
            messagebox.showerror(
                'Security Error',
                'Dashboard path is outside the llm_detector package.',
            )
            return
        streamlit_exe = shutil.which('streamlit')
        if streamlit_exe:
            self._start_dashboard_process([streamlit_exe, 'run', dashboard_path])
        elif importlib.util.find_spec('streamlit') is not None:
            # streamlit installed but not on PATH; use python -m
            self._start_dashboard_process(
                [_real_python(), '-m', 'streamlit', 'run', dashboard_path]
            )
        else:
            # Auto-install streamlit in a background thread to avoid blocking
            self.status_var.set('Installing Streamlit…')
            threading.Thread(
                target=self._install_and_launch_dashboard,
                args=(dashboard_path,),
                daemon=True,
            ).start()

    def _install_and_launch_dashboard(self, dashboard_path):
        """Install Streamlit and launch the dashboard (runs in a background thread)."""
        try:
            subprocess.check_call(
                [_real_python(), '-m', 'pip', 'install', _STREAMLIT_MIN_VERSION],
                stdout=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            def _show_error():
                self.status_var.set('Streamlit installation failed.')
                messagebox.showerror(
                    'Streamlit Not Found',
                    'streamlit could not be installed automatically.\n'
                    'Install it with:\n    pip install "llm-detector[web]"',
                )
            self.root.after(0, _show_error)
            return
        streamlit_exe = shutil.which('streamlit')
        if streamlit_exe:
            cmd = [streamlit_exe, 'run', dashboard_path]
        else:
            cmd = [_real_python(), '-m', 'streamlit', 'run', dashboard_path]
        self.root.after(0, lambda: self._start_dashboard_process(cmd))

    def _start_dashboard_process(self, cmd):
        """Spawn the dashboard subprocess."""
        kwargs = dict(
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if sys.platform == 'win32':
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs['start_new_session'] = True
        try:
            subprocess.Popen(cmd, **kwargs)
        except OSError as exc:
            messagebox.showerror('Launch Error', f'Could not start Streamlit:\n{exc}')
            return
        self.status_var.set('Web dashboard launched — opening in browser…')

    def _launch_desktop_gui(self):
        """Spawn another instance of the desktop GUI."""
        python = _real_python()
        # Quiet stdout to avoid duplicate console chatter; keep stderr for debugging startup failures.
        kwargs = {'stdout': subprocess.DEVNULL, 'stderr': sys.stderr}
        if sys.platform == 'win32':
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs['start_new_session'] = True

        try:
            subprocess.Popen([python, '-m', 'llm_detector', '--gui'], **kwargs)
            self.status_var.set('Opened a new desktop GUI window.')
        except OSError as exc:
            messagebox.showerror('Launch Error', f'Could not open desktop GUI:\n{exc}')


if HAS_TK:
    class _LabelingDialog:
        """Modal dialog for reviewing pipeline results and assigning ground-truth labels."""

        _ICONS = {
            'RED': '\U0001f534', 'AMBER': '\U0001f7e0', 'YELLOW': '\U0001f7e1',
            'GREEN': '\U0001f7e2', 'MIXED': '\U0001f535', 'REVIEW': '\u26aa',
        }

        def __init__(self, parent, results, text_map, output_path, reviewer,
                     store=None, on_complete=None):
            self._results = list(results)
            self._text_map = text_map or {}
            self._output_path = output_path
            self._reviewer = reviewer
            self._store = store
            self._on_complete = on_complete
            self._idx = 0
            self._stats = {
                'total_presented': 0,
                'labeled_ai': 0,
                'labeled_human': 0,
                'labeled_unsure': 0,
                'skipped': 0,
            }

            self.win = tk.Toplevel(parent)
            self.win.title('Interactive Labeling')
            self.win.geometry('900x680')
            self.win.grab_set()
            self.win.protocol('WM_DELETE_WINDOW', self._quit)

            self._build()
            self._show_current()

        def _build(self):
            # Progress bar
            top = ttk.Frame(self.win, padding=(8, 6))
            top.pack(fill=tk.X)
            self._progress_var = tk.StringVar()
            ttk.Label(top, textvariable=self._progress_var, style='DashboardSubtitle.TLabel').pack(side=tk.LEFT)

            # Result display
            display_frame = ttk.LabelFrame(self.win, text='Result', padding=6)
            display_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

            scrollbar = ttk.Scrollbar(display_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self._display_text = tk.Text(
                display_frame, wrap=tk.WORD, font=('Consolas', 9),
                state='disabled', yscrollcommand=scrollbar.set)
            self._display_text.pack(fill=tk.BOTH, expand=True)
            scrollbar.config(command=self._display_text.yview)

            # Notes
            notes_frame = ttk.Frame(self.win, padding=(8, 2))
            notes_frame.pack(fill=tk.X)
            ttk.Label(notes_frame, text='Notes (optional):').pack(side=tk.LEFT)
            self._notes_var = tk.StringVar()
            ttk.Entry(notes_frame, textvariable=self._notes_var).pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

            # Action buttons
            btn_frame = ttk.Frame(self.win, padding=(8, 6))
            btn_frame.pack(fill=tk.X)
            ttk.Button(btn_frame, text='\U0001f916  AI',
                       command=lambda: self._label('ai')).pack(side=tk.LEFT, padx=(0, 4))
            ttk.Button(btn_frame, text='\U0001f9d1  Human',
                       command=lambda: self._label('human')).pack(side=tk.LEFT, padx=(0, 4))
            ttk.Button(btn_frame, text='?  Unsure',
                       command=lambda: self._label('unsure')).pack(side=tk.LEFT, padx=(0, 4))
            ttk.Button(btn_frame, text='Skip',
                       command=self._skip).pack(side=tk.LEFT, padx=(0, 12))
            ttk.Separator(btn_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)
            ttk.Button(btn_frame, text='Quit Session',
                       command=self._quit).pack(side=tk.LEFT, padx=(4, 0))

        def _show_current(self):
            if self._idx >= len(self._results):
                self._finish()
                return

            r = self._results[self._idx]
            n = len(self._results)
            icon = self._ICONS.get(r.get('determination', ''), '?')
            self._progress_var.set(
                f'[{self._idx + 1}/{n}]  Reviewer: {self._reviewer}  |  '
                f'Labeled so far: {self._stats["labeled_ai"]} AI / '
                f'{self._stats["labeled_human"]} human / '
                f'{self._stats["labeled_unsure"]} unsure'
            )

            det = r.get('determination', '?')
            lines = [
                f"{icon} [{det}]  conf={r.get('confidence', 0):.2f}  mode={r.get('mode', '?')}",
                f"Task:      {r.get('task_id', '?')}",
                f"Attempter: {r.get('attempter', '(unknown)')}",
                f"Occupation: {r.get('occupation', '(unknown)')}",
                f"Words:     {r.get('word_count', 0)}",
                f"Reason:    {r.get('reason', '')[:120]}",
                '',
                '--- Key Signals ---',
                f"Preamble:    {r.get('preamble_score', 0):.2f} ({r.get('preamble_severity', 'NONE')})",
                f"Prompt Sig:  {r.get('prompt_signature_composite', 0):.2f}  "
                f"(CFD={r.get('prompt_signature_cfd', 0):.3f})",
                f"VSD:         {r.get('voice_dissonance_vsd', 0):.1f}  "
                f"(voice={r.get('voice_dissonance_voice_score', 0):.1f} x "
                f"spec={r.get('voice_dissonance_spec_score', 0):.1f})",
                f"IDI:         {r.get('instruction_density_idi', 0):.1f}",
                f"NSSI:        {r.get('self_similarity_nssi_score', 0):.3f}  "
                f"({r.get('self_similarity_nssi_signals', 0)} signals)",
                f"DNA-GPT:     {r.get('continuation_bscore', 0):.4f}  "
                f"({r.get('continuation_mode', 'n/a')})",
            ]

            cd = (r.get('channel_details') or {})
            channels_cd = cd.get('channels', {})
            if channels_cd:
                lines.append('')
                lines.append('--- Channels ---')
                for ch_name in ['prompt_structure', 'stylometry', 'continuation', 'windowing']:
                    info = channels_cd.get(ch_name, {})
                    sev = info.get('severity', 'GREEN')
                    if sev != 'GREEN':
                        role = info.get('role', '')
                        role_tag = f' [{role}]' if role else ''
                        lines.append(f'  {ch_name:20s} {sev:6s}{role_tag}  {info.get("explanation", "")[:60]}')
                triggering_rule = cd.get('triggering_rule', '')
                if triggering_rule:
                    lines.append(f'  Rule: {triggering_rule}')

            tid = r.get('task_id', '')
            if tid in self._text_map:
                text = self._text_map[tid]
                preview = text[:300] + (f'... [{len(text) - 300} more chars]' if len(text) > 300 else '')
                lines.append('')
                lines.append('--- Text Preview ---')
                lines.append(preview)

            self._display_text.configure(state='normal')
            self._display_text.delete('1.0', tk.END)
            self._display_text.insert('1.0', '\n'.join(lines))
            self._display_text.configure(state='disabled')
            self._notes_var.set('')

        def _label(self, ground_truth):
            try:
                import json
                from datetime import datetime
                r = self._results[self._idx]
                notes = self._notes_var.get().strip()

                wc = r.get('word_count', 0)
                length_bin = (
                    'short' if wc < 100 else
                    'medium' if wc < 300 else
                    'long' if wc < 800 else
                    'very_long'
                )
                record = {
                    'task_id': r.get('task_id', ''),
                    'attempter': r.get('attempter', ''),
                    'occupation': r.get('occupation', ''),
                    'ground_truth': ground_truth,
                    'pipeline_determination': r.get('determination', ''),
                    'pipeline_confidence': r.get('confidence', 0),
                    'reviewer': self._reviewer,
                    'notes': notes,
                    'timestamp': datetime.now().isoformat(),
                    'pipeline_version': 'v0.66',
                    'confidence': r.get('confidence', 0),
                    'word_count': wc,
                    'domain': r.get('domain', ''),
                    'mode': r.get('mode', ''),
                    'length_bin': length_bin,
                }

                with open(self._output_path, 'a') as f:
                    f.write(json.dumps(record) + '\n')

                if self._store and ground_truth in ('ai', 'human'):
                    self._store.record_confirmation(
                        r.get('task_id', ''), ground_truth,
                        verified_by=self._reviewer, notes=notes,
                    )

                if ground_truth == 'ai':
                    self._stats['labeled_ai'] += 1
                elif ground_truth == 'human':
                    self._stats['labeled_human'] += 1
                else:
                    self._stats['labeled_unsure'] += 1
                self._stats['total_presented'] += 1

                self._idx += 1
                self._show_current()
            except Exception as e:
                messagebox.showerror('Labeling Error', f'Failed to record label: {e}')

        def _skip(self):
            try:
                self._stats['skipped'] += 1
                self._stats['total_presented'] += 1
                self._idx += 1
                self._show_current()
            except Exception as e:
                messagebox.showerror('Skip Error', f'Failed to skip: {e}')

        def _quit(self):
            self._finish()

        def _finish(self):
            if self._on_complete:
                self._on_complete(self._stats)
            self.win.destroy()


def launch_gui():
    """Launch Tkinter GUI mode."""
    if not HAS_TK:
        print('ERROR: tkinter is not available in this Python environment.')
        return
    root = tk.Tk()
    DetectorGUI(root)
    root.mainloop()
