"""Streamlit-based React-style dashboard for the LLM Detection Pipeline.

Launch with:
    streamlit run llm_detector/dashboard.py
    # or
    llm-detector-dashboard
"""

import os
import io
import subprocess
import sys
import json
import html as _html
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd

from llm_detector.pipeline import analyze_prompt
from llm_detector.io import load_xlsx, load_csv

# ── Theme constants ──────────────────────────────────────────────────────────

_DET_COLORS = {
    "RED": "#d32f2f",
    "AMBER": "#f57c00",
    "MIXED": "#7b1fa2",
    "YELLOW": "#fbc02d",
    "REVIEW": "#0288d1",
    "GREEN": "#388e3c",
}

_DET_EMOJI = {
    "RED": "\U0001f534",
    "AMBER": "\U0001f7e0",
    "MIXED": "\U0001f7e3",
    "YELLOW": "\U0001f7e1",
    "REVIEW": "\U0001f535",
    "GREEN": "\U0001f7e2",
}

_DEFAULT_SIMILARITY_THRESHOLD = 0.40
_SIMILARITY_EXTRAS_MSG = (
    "Install similarity extras (pip install llm-detector[similarity]) "
    "to enable similarity analysis and store handling."
)

_CHANNELS = ["prompt_structure", "stylometry", "continuation", "windowing"]

# Maximum number of preamble patterns shown in verbose output to avoid overflow.
_MAX_PREAMBLE_PATTERNS = 20

def _det_badge(det: str) -> str:
    """Return a colored markdown badge for a determination."""
    color = _DET_COLORS.get(det, "#6b7280")
    emoji = _DET_EMOJI.get(det, "")
    return f":{color[1:]}[{emoji} **{det}**]"


def _rerun():
    """Compatibility wrapper for Streamlit rerun across versions."""
    rerun_fn = getattr(st, "rerun", None)
    if not callable(rerun_fn):
        rerun_fn = getattr(st, "experimental_rerun", None)
    if callable(rerun_fn):
        rerun_fn()
    else:
        raise RuntimeError("Streamlit rerun unavailable; please upgrade Streamlit.")


# ── Page configuration ───────────────────────────────────────────────────────

def _configure_page():
    st.set_page_config(
        page_title="LLM Detector Dashboard",
        page_icon="\u2728",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Inject custom CSS for React-style look
    st.markdown("""
    <style>
    /* Card-like containers */
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 0.75rem;
    }
    /* KPI metric cards */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #1e293b;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #f1f5f9;
    }
    section[data-testid="stSidebar"] .stRadio label span {
        color: #cbd5e1 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #334155;
    }
    /* Button styling */
    .stButton > button[kind="primary"] {
        background: #2563eb;
        border: none;
        border-radius: 6px;
    }
    .stButton > button[kind="primary"]:hover {
        background: #1d4ed8;
    }
    /* Top header area */
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Session state initialization ─────────────────────────────────────────────

def _init_state():
    defaults = {
        "results": [],
        "text_map": {},
        "memory_store": None,
        "cal_table": None,
        "run_count": 0,
        "sim_threshold": _DEFAULT_SIMILARITY_THRESHOLD,
        "no_similarity": False,
        "sim_store": "",
        "instructions_file": "",
        "collect_path": "",
        "output_dir": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Sidebar ──────────────────────────────────────────────────────────────────

def _render_sidebar():
    with st.sidebar:
        st.markdown("## \u2728 LLM Detector")
        st.caption("Authorship Signal Analyzer")
        st.divider()

        page = st.radio(
            "Navigation",
            [
                "\u25b6\ufe0f  Analysis",
                "\u2699\ufe0f  Configuration",
                "\U0001f9e0  Memory & Learning",
                "\u2696\ufe0f  Calibration",
                "\U0001f4ca  Reports",
                "\U0001f4d6  Quick Reference",
                "\u2705  Precheck",
            ],
            label_visibility="collapsed",
            help=(
                "\u2699\ufe0f Configuration — Set API keys for DNA-GPT continuation "
                "analysis, tune similarity thresholds, and choose output paths.\n\n"
                "\U0001f9e0 Memory & Learning — Load the BEET memory store that "
                "persists analysis history across sessions. Record ground-truth labels "
                "and rebuild shadow / centroid models.\n\n"
                "\u2696\ufe0f Calibration — Load or build conformal calibration tables "
                "to improve confidence estimates. Analyze baseline datasets to tune "
                "detection thresholds."
            ),
        )

        # Brief contextual description for non-obvious pages
        _page_hints = {
            "\u2699\ufe0f  Configuration": (
                "\u2139\ufe0f API keys, similarity settings, and output options."
            ),
            "\U0001f9e0  Memory & Learning": (
                "\u2139\ufe0f Persists history, attempter profiles, and learned models "
                "across sessions."
            ),
            "\u2696\ufe0f  Calibration": (
                "\u2139\ufe0f Conformal calibration tables and baseline analysis to "
                "sharpen confidence scores."
            ),
        }
        hint = _page_hints.get(page)
        if hint:
            st.caption(hint)

        st.divider()

        # Quick status — results
        n = len(st.session_state.get("results", []))
        if n > 0:
            st.success(f"{n} results in session")
        else:
            st.info("No results yet")

        # Memory store status
        mem = st.session_state.get("memory_store")
        if mem is not None:
            cfg = getattr(mem, "_config", {})
            n_subs = cfg.get("total_submissions", 0)
            st.success(f"\U0001f9e0 Memory: {n_subs} submissions")
        else:
            st.warning("\U0001f9e0 Memory store not loaded")

        st.caption("v0.68.0")

    return page


# ── Page: Analysis ───────────────────────────────────────────────────────────

def _page_analysis():
    st.markdown("### \U0001f50d Analysis")
    st.caption("Run detection on individual texts or batch files")

    # KPI Metrics Row
    results = st.session_state.get("results", [])
    col1, col2, col3, col4 = st.columns(4)
    n_results = len(results)
    if n_results > 0:
        dets = [r.get("determination", "") for r in results]
        counts = Counter(dets)
        mc = counts.most_common(1) if counts else []
        top_det = mc[0][0] if mc else "N/A"
        avg_conf = sum(float(r.get("confidence", 0)) for r in results) / n_results
    else:
        top_det = "N/A"
        avg_conf = 0.0

    with col1:
        st.metric("Total Results", n_results)
    with col2:
        st.metric("Top Determination", top_det)
    with col3:
        st.metric("Avg Confidence", f"{avg_conf:.2f}")
    with col4:
        st.metric("Mode", st.session_state.get("mode", "auto"))

    st.markdown("---")

    # ── Data Source Section ────────────
    with st.expander("\U0001f4c1 Data Source", expanded=True):
        tab_text, tab_file = st.tabs(["Single Text", "File Upload"])

        with tab_text:
            text_input = st.text_area(
                "Enter text to analyze",
                height=150,
                placeholder="Paste text here for analysis...",
            )
            analyze_text_btn = st.button(
                "\u25b6 Analyze Text", type="primary", key="analyze_text"
            )

        with tab_file:
            uploaded = st.file_uploader(
                "Upload CSV or XLSX", type=["csv", "xlsx", "xlsm"]
            )
            st.caption(
                "Column names **or** letters (A, B, C…) are accepted. "
                "Leave as default to auto-detect."
            )
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                prompt_col = st.text_input(
                    "Prompt column", value="prompt",
                    help="Column containing the text to analyze. Use a name or a letter, e.g. A",
                )
            with c2:
                id_col_input = st.text_input(
                    "Task ID column", value="task_id",
                    help="Column for the task/submission ID. Use a name or a letter, e.g. B",
                )
            with c3:
                attempter_col_input = st.text_input(
                    "Attempter column", value="attempter_name",
                    help="Column for the person's name/ID. Use a name or a letter, e.g. C",
                )
            with c4:
                occ_col_input = st.text_input(
                    "Occupation column", value="occupation",
                    help="Column for occupation/area. Use a name or a letter, e.g. D",
                )
            c5, c6, c7 = st.columns(3)
            with c5:
                sheet_name = st.text_input("Sheet name (XLSX)", value="")
            with c6:
                attempter_filter = st.text_input("Attempter filter", value="")
            with c7:
                stage_col_input = st.text_input(
                    "Stage column", value="pipeline_stage_name",
                    help="Column for pipeline stage (optional). Use a name or a letter, e.g. E",
                )
            c8, c9, c10 = st.columns(3)
            with c8:
                attempter_email_col_input = st.text_input(
                    "Attempter email col (optional)", value="",
                    help="Column for attempter email address",
                )
            with c9:
                reviewer_col_input = st.text_input(
                    "Reviewer col (optional)", value="",
                    help="Column for reviewer name",
                )
            with c10:
                reviewer_email_col_input = st.text_input(
                    "Reviewer email col (optional)", value="",
                    help="Column for reviewer email address",
                )
            analyze_file_btn = st.button(
                "\U0001f4c1 Analyze File", type="primary", key="analyze_file"
            )

    # ── Detection Settings ────────────
    with st.expander("\u2699\ufe0f Detection Settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            mode = st.selectbox("Mode", ["auto", "task_prompt", "generic_aigt"])
            st.session_state["mode"] = mode
        with c2:
            ppl_model = st.selectbox(
                "PPL Model",
                [
                    "Qwen/Qwen2.5-0.5B",
                    "HuggingFaceTB/SmolLM2-360M",
                    "HuggingFaceTB/SmolLM2-135M",
                    "distilgpt2",
                    "gpt2",
                ],
            )
        with c3:
            workers = st.number_input("Workers", min_value=1, max_value=16, value=4)

        c1, c2, c3 = st.columns(3)
        with c1:
            show_details = st.checkbox("Show details", value=True)
        with c2:
            verbose = st.checkbox("Verbose", value=False)
        with c3:
            no_layer3 = st.checkbox("Skip API Continuation", value=False)

        st.markdown("**Channel Ablation**")
        abl_cols = st.columns(len(_CHANNELS))
        disabled_channels = []
        for i, ch in enumerate(_CHANNELS):
            with abl_cols[i]:
                if st.checkbox(f"Disable {ch}", key=f"abl_{ch}"):
                    disabled_channels.append(ch)

    # ── Execute analysis ────────────
    def _build_kwargs():
        api_key = st.session_state.get("api_key", "").strip() or None
        return {
            "run_l3": not no_layer3,
            "api_key": api_key,
            "dna_provider": st.session_state.get("dna_provider", "anthropic"),
            "dna_model": st.session_state.get("dna_model", "").strip() or None,
            "dna_samples": st.session_state.get("dna_samples", 3),
            "mode": mode,
            "disabled_channels": disabled_channels or None,
            "cal_table": st.session_state.get("cal_table"),
            "memory_store": st.session_state.get("memory_store"),
            "ppl_model": ppl_model or None,
        }

    def _postprocess_results(results, text_map):
        """Apply CLI-parity post-processing (similarity, memory, baselines).

        Args:
            results: List of pipeline result dicts. Mutated in-place (e.g., similarity upgrades).
            text_map: Dict mapping task_id to original text content.

        Returns:
            List of human-readable status messages describing post-processing actions.
        """
        messages = []

        # Similarity analysis and cross-batch similarity store
        instruction_text = None
        instr_path = (st.session_state.get("instructions_file") or "").strip()
        if instr_path and os.path.exists(instr_path):
            try:
                with open(instr_path, "r") as f:
                    instruction_text = f.read()
            except OSError as e:
                messages.append(f"Instructions file error: {e}")

        # Respect the disable flag; otherwise require two or more results for pairwise metrics.
        if not st.session_state.get("no_similarity") and len(results) >= 2:
            try:
                from llm_detector.similarity import (
                    analyze_similarity,
                    apply_similarity_adjustments,
                )

                sim_pairs = analyze_similarity(
                    results,
                    text_map,
                    jaccard_threshold=st.session_state.get(
                        "sim_threshold", _DEFAULT_SIMILARITY_THRESHOLD
                    ),
                    instruction_text=instruction_text,
                )
                if sim_pairs:
                    results[:] = apply_similarity_adjustments(
                        results, sim_pairs, text_map
                    )
                    messages.append(f"Similarity: {len(sim_pairs)} pairs flagged")
            except ImportError:
                messages.append(_SIMILARITY_EXTRAS_MSG)
            except (OSError, ValueError) as e:
                messages.append(f"Similarity analysis failed: {e}")

        sim_store = (st.session_state.get("sim_store") or "").strip()
        if sim_store:
            try:
                from llm_detector.similarity import (
                    cross_batch_similarity,
                    save_similarity_store,
                )

                cross_flags = cross_batch_similarity(results, text_map, sim_store)
                if cross_flags:
                    messages.append(
                        f"Cross-batch similarity: {len(cross_flags)} matches to history"
                    )
                save_similarity_store(results, text_map, sim_store)
            except ImportError:
                messages.append(_SIMILARITY_EXTRAS_MSG)
            except (OSError, ValueError) as e:
                messages.append(f"Similarity store update failed: {e}")

        # Baseline accumulation (collect path)
        try:
            collect_path = (st.session_state.get("collect_path") or "").strip()
            if collect_path:
                from llm_detector.baselines import collect_baselines

                collect_baselines(results, collect_path)
                messages.append(f"Baselines appended to {collect_path}")
        except (OSError, ValueError) as e:
            messages.append(f"Baseline collection failed: {e}")

        # Memory store updates
        try:
            mem = st.session_state.get("memory_store")
            if mem:
                cross_flags = mem.cross_batch_similarity(results, text_map)
                if cross_flags:
                    messages.append(
                        f"Memory cross-batch: {len(cross_flags)} matches detected"
                    )
                mem.record_batch(results, text_map)
        except (OSError, ValueError, AttributeError) as e:
            messages.append(f"Memory store update failed: {e}")

        return messages

    if analyze_text_btn and text_input.strip():
        with st.spinner("Analyzing text..."):
            kwargs = _build_kwargs()
            result = analyze_prompt(text_input, **kwargs)

            mem = st.session_state.get("memory_store")
            if mem:
                dis = mem.check_shadow_disagreement(result)
                result["shadow_disagreement"] = dis
                result["shadow_ai_prob"] = (dis or {}).get("shadow_ai_prob")

            results = [result]
            text_map = {"_single": text_input}
            st.session_state["analysis_messages"] = _postprocess_results(
                results, text_map
            )

            st.session_state["results"] = results
            st.session_state["text_map"] = text_map
            st.session_state["run_count"] += 1
        _rerun()

    if analyze_file_btn and uploaded is not None:
        with st.spinner("Analyzing file..."):
            # Save upload to temp file
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            try:
                if suffix in (".xlsx", ".xlsm"):
                    tasks = load_xlsx(
                        tmp_path,
                        sheet=sheet_name.strip() or None,
                        prompt_col=prompt_col.strip() or "prompt",
                        id_col=id_col_input.strip() or "task_id",
                        occ_col=occ_col_input.strip() or "occupation",
                        attempter_col=attempter_col_input.strip() or "attempter_name",
                        stage_col=stage_col_input.strip() or "pipeline_stage_name",
                        attempter_email_col=attempter_email_col_input.strip(),
                        reviewer_col=reviewer_col_input.strip(),
                        reviewer_email_col=reviewer_email_col_input.strip(),
                    )
                else:
                    tasks = load_csv(
                        tmp_path,
                        prompt_col=prompt_col.strip() or "prompt",
                        id_col=id_col_input.strip() or "task_id",
                        occ_col=occ_col_input.strip() or "occupation",
                        attempter_col=attempter_col_input.strip() or "attempter_name",
                        stage_col=stage_col_input.strip() or "pipeline_stage_name",
                        attempter_email_col=attempter_email_col_input.strip(),
                        reviewer_col=reviewer_col_input.strip(),
                        reviewer_email_col=reviewer_email_col_input.strip(),
                    )
            finally:
                os.unlink(tmp_path)

            if attempter_filter.strip():
                needle = attempter_filter.strip().lower()
                tasks = [
                    t
                    for t in tasks
                    if needle in t.get("attempter", "").lower()
                ]

            if not tasks:
                st.warning("No qualifying prompts found.")
            else:
                kwargs = _build_kwargs()
                all_results = []
                text_map = {}
                progress = st.progress(0, text="Processing...")

                for i, task in enumerate(tasks):
                    tid = task.get("task_id", f"_row{i+1}")
                    text_map[tid] = task["prompt"]
                    r = analyze_prompt(
                        task["prompt"],
                        task_id=task.get("task_id", ""),
                        occupation=task.get("occupation", ""),
                        attempter=task.get("attempter", ""),
                        stage=task.get("stage", ""),
                        **kwargs,
                    )
                    all_results.append(r)
                    progress.progress(
                        (i + 1) / len(tasks),
                        text=f"Processing {i+1}/{len(tasks)}...",
                    )

                st.session_state["results"] = all_results
                st.session_state["text_map"] = text_map
                st.session_state["run_count"] += 1
                progress.empty()

                # ── Auto-save to output directory if configured ──
                output_dir = st.session_state.get("output_dir", "").strip()
                if output_dir:
                    try:
                        run_folder = (
                            Path(output_dir)
                            / datetime.now().strftime("run_%Y%m%d_%H%M%S")
                        )
                        run_folder.mkdir(parents=True, exist_ok=True)

                        # Auto-set sim_store and other paths
                        if not st.session_state.get("sim_store", "").strip():
                            st.session_state["sim_store"] = str(
                                run_folder / "similarity.jsonl"
                            )

                        # Save results CSV
                        flat = []
                        for r in all_results:
                            row = {k: v for k, v in r.items()
                                   if k != "preamble_details"}
                            row["preamble_details"] = str(
                                r.get("preamble_details", [])
                            )
                            flat.append(row)
                        csv_path = run_folder / "results.csv"
                        pd.DataFrame(flat).to_csv(csv_path, index=False)

                        # Save HTML report for flagged results
                        flagged = [
                            r for r in all_results
                            if r["determination"] in ("RED", "AMBER", "MIXED")
                        ]
                        if flagged:
                            from llm_detector.html_report import (
                                generate_batch_html_report,
                            )
                            html_path = run_folder / "report.html"
                            generate_batch_html_report(
                                flagged, text_map, str(html_path)
                            )

                        # Save labels JSONL placeholder
                        labels_path = run_folder / "labels.jsonl"
                        labels_path.touch(exist_ok=True)

                        # Auto-create memory store in the run folder if not
                        # already configured
                        if st.session_state.get("memory_store") is None:
                            from llm_detector.memory import MemoryStore
                            mem_path = run_folder / "memory"
                            st.session_state["memory_store"] = MemoryStore(
                                str(mem_path)
                            )

                        st.session_state["last_run_folder"] = str(run_folder)
                    except Exception as _auto_err:
                        st.session_state["last_run_folder_error"] = str(
                            _auto_err
                        )

                # Post-process after any auto-created artifacts are available
                st.session_state["analysis_messages"] = _postprocess_results(
                    all_results, text_map
                )

                _rerun()

    # ── Display Results ────────────
    if results:
        st.markdown("---")
        st.markdown("### \U0001f4cb Results")

        # Auto-save notifications
        last_run_folder = st.session_state.pop("last_run_folder", None)
        last_run_folder_error = st.session_state.pop("last_run_folder_error", None)
        if last_run_folder:
            st.success(f"\U0001f4be Results auto-saved to: `{last_run_folder}`")
        if last_run_folder_error:
            st.warning(f"Auto-save failed: {last_run_folder_error}")

        # CLI-parity post-processing messages
        for msg in st.session_state.get("analysis_messages", []):
            st.info(msg)

        # Summary bar
        counts = Counter(r["determination"] for r in results)
        summary_cols = st.columns(len(_DET_COLORS))
        for i, det in enumerate(["RED", "AMBER", "MIXED", "YELLOW", "REVIEW", "GREEN"]):
            ct = counts.get(det, 0)
            with summary_cols[i]:
                color = _DET_COLORS[det]
                emoji = _DET_EMOJI[det]
                st.markdown(
                    f"<div style='text-align:center; padding:8px; "
                    f"background:{color}15; border-left:3px solid {color}; "
                    f"border-radius:4px'>"
                    f"<span style='font-size:0.8rem; color:{color}'>"
                    f"{emoji} {det}</span><br>"
                    f"<strong style='font-size:1.2rem'>{ct}</strong></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("")

        # Results table
        for idx, r in enumerate(results):
            det = r.get("determination", "GREEN")
            conf = r.get("confidence", 0)
            task_id = r.get("task_id", f"#{idx+1}")
            reason = r.get("reason", "")
            color = _DET_COLORS.get(det, "#6b7280")
            emoji = _DET_EMOJI.get(det, "")

            with st.expander(
                f"{emoji} **{det}** — {task_id}  |  conf={conf:.2f}",
                expanded=(len(results) == 1),
            ):
                # Top-line info
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.metric("Determination", det)
                with mc2:
                    st.metric("Confidence", f"{conf:.2f}")
                with mc3:
                    st.metric("Word Count", r.get("word_count", 0))

                st.markdown(f"**Reason:** {reason}")

                # Calibrated confidence
                cal_conf = r.get("calibrated_confidence")
                if cal_conf is not None:
                    st.info(
                        f"Calibrated: {cal_conf:.2f}  "
                        f"({r.get('calibration_stratum', '')} / "
                        f"{r.get('conformity_level', '')})"
                    )

                # Channel details
                if show_details:
                    cd = r.get("channel_details", {})
                    channels = cd.get("channels", {})
                    if channels:
                        st.markdown("**Channel Details:**")
                        ch_data = []
                        for ch_name, info in channels.items():
                            ch_data.append({
                                "Channel": ch_name,
                                "Severity": info.get("severity", "GREEN"),
                                "Score": f"{info.get('score', 0):.2f}",
                                "Role": info.get("role", "—"),
                                "Data Sufficient": (
                                    "\u2705" if info.get("data_sufficient", True)
                                    else "\u274c"
                                ),
                                "Status": (
                                    "disabled" if info.get("disabled") else
                                    "ineligible" if not info.get("mode_eligible", True)
                                    else "active"
                                ),
                            })
                        st.dataframe(
                            pd.DataFrame(ch_data),
                            use_container_width=True,
                            hide_index=True,
                        )
                        triggering_rule = cd.get("triggering_rule", "")
                        if triggering_rule:
                            st.caption(f"Triggering rule: `{triggering_rule}`")
                        fc = cd.get("fusion_counts", {})
                        if fc:
                            fc_parts = [
                                f"primary RED={fc.get('n_primary_red', 0)}",
                                f"primary AMBER+={fc.get('n_primary_amber', 0)}",
                                f"all YELLOW+={fc.get('n_yellow_plus', 0)}",
                            ]
                            st.caption("Fusion counts: " + "  ·  ".join(fc_parts))

                # Verbose details — full sub-signal breakdown
                if verbose:
                    with st.expander("\U0001f9ea Full Sub-Signal Breakdown", expanded=True):
                        # ── Normalization & Language Gate ──────────────
                        st.markdown("**🔤 Normalization & Language Gate**")
                        norm_data = [
                            {"Signal": "Obfuscation delta", "Value": f"{r.get('norm_obfuscation_delta', 0):.1%}"},
                            {"Signal": "Invisible chars", "Value": str(r.get('norm_invisible_chars', 0))},
                            {"Signal": "Homoglyphs", "Value": str(r.get('norm_homoglyphs', 0))},
                            {"Signal": "Language support", "Value": r.get('lang_support_level', 'SUPPORTED')},
                            {"Signal": "Function-word coverage", "Value": f"{r.get('lang_fw_coverage', 0):.3f}"},
                            {"Signal": "Non-Latin ratio", "Value": f"{r.get('lang_non_latin_ratio', 0):.3f}"},
                        ]
                        attacks = r.get("norm_attack_types", [])
                        if attacks:
                            norm_data.append({"Signal": "Attack types neutralized", "Value": ", ".join(attacks)})
                        st.dataframe(pd.DataFrame(norm_data), use_container_width=True, hide_index=True)

                        # ── Preamble ───────────────────────────────────
                        st.markdown("**📋 Preamble**")
                        pre_data = [
                            {"Signal": "Score", "Value": f"{r.get('preamble_score', 0):.3f}"},
                            {"Signal": "Severity", "Value": r.get('preamble_severity', '-')},
                            {"Signal": "Matched patterns", "Value": str(r.get('preamble_hits', 0))},
                        ]
                        st.dataframe(pd.DataFrame(pre_data), use_container_width=True, hide_index=True)
                        preamble_details = r.get("preamble_details", [])
                        if preamble_details:
                            st.caption("Matched preamble patterns: " + ", ".join(str(p) for p in preamble_details[:_MAX_PREAMBLE_PATTERNS]))

                        # ── Fingerprint ────────────────────────────────
                        st.markdown("**🔍 Fingerprint**")
                        fp_data = [
                            {"Signal": "Score", "Value": f"{r.get('fingerprint_score', 0):.3f}"},
                            {"Signal": "Hits", "Value": str(r.get('fingerprint_hits', 0))},
                        ]
                        st.dataframe(pd.DataFrame(fp_data), use_container_width=True, hide_index=True)

                        # ── Prompt Signature ───────────────────────────
                        st.markdown("**✍️ Prompt Signature**")
                        ps_data = [
                            {"Signal": "Composite", "Value": f"{r.get('prompt_signature_composite', 0):.3f}"},
                            {"Signal": "CFD (constraint-frame density)", "Value": f"{r.get('prompt_signature_cfd', 0):.4f}"},
                            {"Signal": "MFSR (meta-frame score ratio)", "Value": f"{r.get('prompt_signature_mfsr', 0):.3f}"},
                            {"Signal": "Distinct frames", "Value": str(r.get('prompt_signature_distinct_frames', 0))},
                            {"Signal": "Framing completeness (0-3)", "Value": str(r.get('prompt_signature_framing', 0))},
                            {"Signal": "Conditional density", "Value": f"{r.get('prompt_signature_conditional_density', 0):.4f}"},
                            {"Signal": "Meta-design hits", "Value": str(r.get('prompt_signature_meta_design', 0))},
                            {"Signal": "Contractions", "Value": str(r.get('prompt_signature_contractions', 0))},
                            {"Signal": "Must-rate", "Value": f"{r.get('prompt_signature_must_rate', 0):.4f}"},
                            {"Signal": "Numbered criteria", "Value": str(r.get('prompt_signature_numbered_criteria', 0))},
                        ]
                        st.dataframe(pd.DataFrame(ps_data), use_container_width=True, hide_index=True)

                        # ── Instruction Density (IDI) ──────────────────
                        st.markdown("**📐 Instruction Density (IDI)**")
                        idi_data = [
                            {"Signal": "IDI score", "Value": f"{r.get('instruction_density_idi', 0):.2f}"},
                            {"Signal": "Imperatives", "Value": str(r.get('instruction_density_imperatives', 0))},
                            {"Signal": "Conditionals", "Value": str(r.get('instruction_density_conditionals', 0))},
                            {"Signal": "Binary specs (Y/N)", "Value": str(r.get('instruction_density_binary_specs', 0))},
                            {"Signal": "Missing refs", "Value": str(r.get('instruction_density_missing_refs', 0))},
                            {"Signal": "Flag count", "Value": str(r.get('instruction_density_flag_count', 0))},
                        ]
                        st.dataframe(pd.DataFrame(idi_data), use_container_width=True, hide_index=True)

                        # ── Voice Dissonance (VSD) ─────────────────────
                        st.markdown("**🎭 Voice Dissonance (VSD)**")
                        vsd_data = [
                            {"Signal": "VSD score", "Value": f"{r.get('voice_dissonance_vsd', 0):.2f}"},
                            {"Signal": "Voice score", "Value": f"{r.get('voice_dissonance_voice_score', 0):.2f}"},
                            {"Signal": "Spec score", "Value": f"{r.get('voice_dissonance_spec_score', 0):.2f}"},
                            {"Signal": "Voice gated", "Value": str(r.get('voice_dissonance_voice_gated', False))},
                            {"Signal": "Casual markers", "Value": str(r.get('voice_dissonance_casual_markers', 0))},
                            {"Signal": "Misspellings", "Value": str(r.get('voice_dissonance_misspellings', 0))},
                            {"Signal": "CamelCase cols", "Value": str(r.get('voice_dissonance_camel_cols', 0))},
                            {"Signal": "Calculations", "Value": str(r.get('voice_dissonance_calcs', 0))},
                            {"Signal": "Hedges", "Value": str(r.get('voice_dissonance_hedges', 0))},
                            {"Signal": "SSI triggered", "Value": str(r.get('ssi_triggered', False))},
                        ]
                        st.dataframe(pd.DataFrame(vsd_data), use_container_width=True, hide_index=True)

                        # ── Pack Diagnostics ───────────────────────────
                        st.markdown("**📦 Pack Diagnostics**")
                        pack_data = [
                            {"Signal": "Constraint score", "Value": f"{r.get('pack_constraint_score', 0):.4f}"},
                            {"Signal": "Exec-spec score", "Value": f"{r.get('pack_exec_spec_score', 0):.4f}"},
                            {"Signal": "Schema score", "Value": f"{r.get('pack_schema_score', 0):.4f}"},
                            {"Signal": "Active families", "Value": str(r.get('pack_active_families', 0))},
                            {"Signal": "Prompt boost", "Value": f"{r.get('pack_prompt_boost', 0):.4f}"},
                            {"Signal": "IDI boost", "Value": f"{r.get('pack_idi_boost', 0):.4f}"},
                        ]
                        st.dataframe(pd.DataFrame(pack_data), use_container_width=True, hide_index=True)

                        # ── Stylometry ─────────────────────────────────
                        st.markdown("**🖊️ Stylometry**")
                        stylo_data = [
                            {"Signal": "Function-word ratio", "Value": f"{r.get('stylo_fw_ratio', 0):.4f}"},
                            {"Signal": "Sentence length dispersion", "Value": f"{r.get('stylo_sent_dispersion', 0):.4f}"},
                            {"Signal": "Type-token ratio (TTR)", "Value": f"{r.get('stylo_ttr', 0):.4f}"},
                            {"Signal": "Avg word length", "Value": f"{r.get('stylo_avg_word_len', 0):.3f}"},
                            {"Signal": "Short-word ratio", "Value": f"{r.get('stylo_short_word_ratio', 0):.4f}"},
                            {"Signal": "Masked topical tokens", "Value": str(r.get('stylo_mask_count', 0))},
                        ]
                        st.dataframe(pd.DataFrame(stylo_data), use_container_width=True, hide_index=True)

                        # ── Windowing ──────────────────────────────────
                        st.markdown("**🪟 Windowing**")
                        win_data = [
                            {"Signal": "Max window score", "Value": f"{r.get('window_max_score', 0):.4f}"},
                            {"Signal": "Mean window score", "Value": f"{r.get('window_mean_score', 0):.4f}"},
                            {"Signal": "Window variance", "Value": f"{r.get('window_variance', 0):.4f}"},
                            {"Signal": "Hot-span length", "Value": str(r.get('window_hot_span', 0))},
                            {"Signal": "N windows", "Value": str(r.get('window_n_windows', 0))},
                            {"Signal": "Mixed signal", "Value": str(r.get('window_mixed_signal', False))},
                            {"Signal": "FW trajectory CV", "Value": f"{r.get('window_fw_trajectory_cv', 0):.4f}"},
                            {"Signal": "Comp trajectory mean", "Value": f"{r.get('window_comp_trajectory_mean', 0):.4f}"},
                            {"Signal": "Comp trajectory CV", "Value": f"{r.get('window_comp_trajectory_cv', 0):.4f}"},
                            {"Signal": "Changepoint", "Value": str(r.get('window_changepoint') or 'none')},
                        ]
                        st.dataframe(pd.DataFrame(win_data), use_container_width=True, hide_index=True)

                        # ── Self-Similarity (NSSI) ─────────────────────
                        st.markdown("**🔁 Self-Similarity (NSSI)**")
                        nssi_data = [
                            {"Signal": "NSSI score", "Value": f"{r.get('self_similarity_nssi_score', 0):.4f}"},
                            {"Signal": "NSSI signals", "Value": str(r.get('self_similarity_nssi_signals', 0))},
                            {"Signal": "Determination", "Value": str(r.get('self_similarity_determination') or 'n/a')},
                            {"Signal": "Confidence", "Value": f"{r.get('self_similarity_confidence', 0):.3f}"},
                            {"Signal": "Formulaic density", "Value": f"{r.get('self_similarity_formulaic_density', 0):.4f}"},
                            {"Signal": "Power-adj density", "Value": f"{r.get('self_similarity_power_adj_density', 0):.4f}"},
                            {"Signal": "Demonstrative density", "Value": f"{r.get('self_similarity_demonstrative_density', 0):.4f}"},
                            {"Signal": "Transition density", "Value": f"{r.get('self_similarity_transition_density', 0):.4f}"},
                            {"Signal": "Scare-quote density", "Value": f"{r.get('self_similarity_scare_quote_density', 0):.4f}"},
                            {"Signal": "Em-dash density", "Value": f"{r.get('self_similarity_emdash_density', 0):.4f}"},
                            {"Signal": "This/the start rate", "Value": f"{r.get('self_similarity_this_the_start_rate', 0):.4f}"},
                            {"Signal": "Section depth", "Value": str(r.get('self_similarity_section_depth', 0))},
                            {"Signal": "Sent length CV", "Value": f"{r.get('self_similarity_sent_length_cv', 0):.4f}"},
                            {"Signal": "Comp ratio", "Value": f"{r.get('self_similarity_comp_ratio', 0):.4f}"},
                            {"Signal": "Hapax ratio", "Value": f"{r.get('self_similarity_hapax_ratio', 0):.4f}"},
                            {"Signal": "Hapax count", "Value": str(r.get('self_similarity_hapax_count', 0))},
                            {"Signal": "Unique words", "Value": str(r.get('self_similarity_unique_words', 0))},
                            {"Signal": "Shuffled comp ratio", "Value": f"{r.get('self_similarity_shuffled_comp_ratio', 0):.4f}"},
                            {"Signal": "Structural compression delta", "Value": f"{r.get('self_similarity_structural_compression_delta', 0):.4f}"},
                        ]
                        st.dataframe(pd.DataFrame(nssi_data), use_container_width=True, hide_index=True)

                        # ── Continuation / DNA-GPT ─────────────────────
                        st.markdown("**🧬 Continuation (DNA-GPT)**")
                        cont_data = [
                            {"Signal": "B-score", "Value": f"{r.get('continuation_bscore', 0):.4f}"},
                            {"Signal": "B-score max", "Value": f"{r.get('continuation_bscore_max', 0):.4f}"},
                            {"Signal": "Determination", "Value": str(r.get('continuation_determination') or 'n/a')},
                            {"Signal": "Confidence", "Value": f"{r.get('continuation_confidence', 0):.3f}"},
                            {"Signal": "N samples", "Value": str(r.get('continuation_n_samples', 0))},
                            {"Signal": "Mode", "Value": str(r.get('continuation_mode') or 'n/a')},
                            {"Signal": "NCD", "Value": f"{r.get('continuation_ncd', 0):.4f}"},
                            {"Signal": "Internal overlap", "Value": f"{r.get('continuation_internal_overlap', 0):.4f}"},
                            {"Signal": "Cond surprisal", "Value": f"{r.get('continuation_cond_surprisal', 0):.4f}"},
                            {"Signal": "Repeat-4 rate", "Value": f"{r.get('continuation_repeat4', 0):.4f}"},
                            {"Signal": "TTR (continuation)", "Value": f"{r.get('continuation_ttr', 0):.4f}"},
                            {"Signal": "Composite", "Value": f"{r.get('continuation_composite', 0):.4f}"},
                            {"Signal": "Composite variance", "Value": f"{r.get('continuation_composite_variance', 0):.4f}"},
                            {"Signal": "Composite stability", "Value": f"{r.get('continuation_composite_stability', 0):.4f}"},
                            {"Signal": "Improvement rate", "Value": f"{r.get('continuation_improvement_rate', 0):.4f}"},
                            {"Signal": "NCD matrix mean", "Value": f"{r.get('continuation_ncd_matrix_mean', 0):.4f}"},
                            {"Signal": "NCD matrix variance", "Value": f"{r.get('continuation_ncd_matrix_variance', 0):.4f}"},
                            {"Signal": "NCD matrix min", "Value": f"{r.get('continuation_ncd_matrix_min', 0):.4f}"},
                        ]
                        st.dataframe(pd.DataFrame(cont_data), use_container_width=True, hide_index=True)

                        # ── Perplexity ─────────────────────────────────
                        st.markdown("**📊 Perplexity**")
                        ppl_data = [
                            {"Signal": "Perplexity value", "Value": f"{r.get('perplexity_value', 0):.3f}"},
                            {"Signal": "Determination", "Value": str(r.get('perplexity_determination') or 'n/a')},
                            {"Signal": "Confidence", "Value": f"{r.get('perplexity_confidence', 0):.3f}"},
                            {"Signal": "Surprisal variance", "Value": f"{r.get('surprisal_variance', 0):.4f}"},
                            {"Signal": "Surprisal var (first half)", "Value": f"{r.get('surprisal_first_half_var', 0):.4f}"},
                            {"Signal": "Surprisal var (second half)", "Value": f"{r.get('surprisal_second_half_var', 0):.4f}"},
                            {"Signal": "Volatility decay ratio", "Value": f"{r.get('volatility_decay_ratio', 1):.4f}"},
                            {"Signal": "Binoculars score", "Value": f"{r.get('binoculars_score', 0):.4f}"},
                            {"Signal": "Binoculars determination", "Value": str(r.get('binoculars_determination') or 'n/a')},
                            {"Signal": "Comp ratio", "Value": f"{r.get('perplexity_comp_ratio', 0):.4f}"},
                            {"Signal": "zlib-normalised PPL", "Value": f"{r.get('perplexity_zlib_normalized_ppl', 0):.4f}"},
                            {"Signal": "Comp/PPL ratio", "Value": f"{r.get('perplexity_comp_ppl_ratio', 0):.4f}"},
                        ]
                        st.dataframe(pd.DataFrame(ppl_data), use_container_width=True, hide_index=True)

                        # ── Surprisal Trajectory ───────────────────────
                        st.markdown("**📈 Surprisal Trajectory**")
                        traj_data = [
                            {"Signal": "Trajectory CV", "Value": f"{r.get('surprisal_trajectory_cv', 0):.4f}"},
                            {"Signal": "Var-of-var", "Value": f"{r.get('surprisal_var_of_var', 0):.4f}"},
                            {"Signal": "Stationarity", "Value": f"{r.get('surprisal_stationarity', 0):.4f}"},
                        ]
                        st.dataframe(pd.DataFrame(traj_data), use_container_width=True, hide_index=True)

                        # ── TOCSIN ─────────────────────────────────────
                        st.markdown("**🔔 TOCSIN (Token Cohesiveness)**")
                        toc_data = [
                            {"Signal": "Cohesiveness", "Value": f"{r.get('tocsin_cohesiveness', 0):.4f}"},
                            {"Signal": "Cohesiveness std", "Value": f"{r.get('tocsin_cohesiveness_std', 0):.4f}"},
                            {"Signal": "Determination", "Value": str(r.get('tocsin_determination') or 'n/a')},
                            {"Signal": "Confidence", "Value": f"{r.get('tocsin_confidence', 0):.3f}"},
                        ]
                        st.dataframe(pd.DataFrame(toc_data), use_container_width=True, hide_index=True)

                        # ── Semantic Resonance ─────────────────────────
                        st.markdown("**🌐 Semantic Resonance**")
                        sem_data = [
                            {"Signal": "AI score", "Value": f"{r.get('semantic_resonance_ai_score', 0):.4f}"},
                            {"Signal": "Human score", "Value": f"{r.get('semantic_resonance_human_score', 0):.4f}"},
                            {"Signal": "AI centroid mean", "Value": f"{r.get('semantic_resonance_ai_mean', 0):.4f}"},
                            {"Signal": "Human centroid mean", "Value": f"{r.get('semantic_resonance_human_mean', 0):.4f}"},
                            {"Signal": "Delta (AI − human)", "Value": f"{r.get('semantic_resonance_delta', 0):.4f}"},
                            {"Signal": "Determination", "Value": str(r.get('semantic_resonance_determination') or 'n/a')},
                            {"Signal": "Confidence", "Value": f"{r.get('semantic_resonance_confidence', 0):.3f}"},
                        ]
                        st.dataframe(pd.DataFrame(sem_data), use_container_width=True, hide_index=True)

                # Detection spans
                spans = r.get("detection_spans", [])
                if spans:
                    span_sources = Counter(
                        s.get("source", "?") for s in spans
                    )
                    st.caption(
                        f"Detection spans: {len(spans)} "
                        f"({', '.join(f'{src}={ct}' for src, ct in span_sources.items())})"
                    )

                # Attack types
                attacks = r.get("norm_attack_types", [])
                if attacks:
                    st.warning(
                        f"Attacks neutralized: {', '.join(attacks)}"
                    )

                # Shadow model
                shadow = r.get("shadow_disagreement")
                if shadow:
                    st.error(
                        f"Shadow: {shadow.get('interpretation', 'disagrees')} — "
                        f"Rule={shadow.get('rule_determination', '?')}, "
                        f"Model={shadow.get('shadow_ai_prob', 0):.1%} AI"
                    )

        # Export actions
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            # CSV export
            flat = []
            for r in results:
                row = {
                    k: v for k, v in r.items() if k != "preamble_details"
                }
                row["preamble_details"] = str(
                    r.get("preamble_details", [])
                )
                flat.append(row)
            csv_data = pd.DataFrame(flat).to_csv(index=False)
            st.download_button(
                "\U0001f4be Download CSV",
                csv_data,
                "results.csv",
                "text/csv",
            )

        with c2:
            # HTML report
            try:
                text_map = st.session_state.get("text_map", {})
                if len(results) == 1:
                    # Single text analysis — generate report for any determination
                    from llm_detector.html_report import generate_html_report
                    r = results[0]
                    # Text is stored under "_single" for single-text analysis;
                    # fall back to task_id key for any other single-result case.
                    text = text_map.get("_single") or text_map.get(
                        r.get("task_id", ""), ""
                    )
                    html_content = generate_html_report(text, r)
                    st.download_button(
                        "\U0001f4c4 Download HTML Report",
                        html_content,
                        "report.html",
                        "text/html",
                    )
                else:
                    # Batch analysis — report for flagged results only
                    flagged = [
                        r
                        for r in results
                        if r["determination"] in ("RED", "AMBER", "MIXED")
                    ]
                    if flagged:
                        from llm_detector.html_report import (
                            generate_batch_html_report,
                        )
                        html_content = generate_batch_html_report(
                            flagged, text_map
                        )
                        st.download_button(
                            "\U0001f4c4 Download HTML Report",
                            html_content,
                            "report.html",
                            "text/html",
                        )
                    else:
                        st.caption("No flagged results for HTML report")
            except Exception:
                st.caption("HTML report generation unavailable")


# ── Page: Configuration ──────────────────────────────────────────────────────

def _page_configuration():
    st.markdown("### \u2699\ufe0f Configuration")
    st.caption("API keys, similarity settings, and output options")

    # Continuation Analysis
    with st.expander("\U0001f9ec Continuation Analysis (DNA-GPT)", expanded=True):
        st.caption(
            "Optional: provide an API key to enable DNA-GPT continuation analysis. "
            "Leave blank to skip API continuation and run faster."
        )
        c1, c2 = st.columns(2)
        with c1:
            provider = st.selectbox(
                "Provider",
                ["anthropic", "openai"],
                key="dna_provider",
            )
        with c2:
            api_key = st.text_input(
                "API Key", type="password", key="api_key",
                placeholder="sk-... or leave blank to skip API continuation",
            )
        # Show whether a key is configured (without revealing it)
        if st.session_state.get("api_key", "").strip():
            st.success("\u2705 API key set — DNA-GPT continuation analysis enabled")
        else:
            st.info("\u2139\ufe0f No API key — API continuation will be skipped (faster, less accurate)")

        c1, c2, c3 = st.columns(3)
        with c1:
            dna_model = st.text_input(
                "Model (optional)", key="dna_model",
                placeholder="Leave blank for default",
            )
        with c2:
            dna_samples = st.number_input(
                "Samples",
                min_value=1,
                max_value=10,
                value=3,
                key="dna_samples",
            )
        with c3:
            batch_api = st.checkbox(
                "Batch API (50% cheaper)", key="batch_api"
            )

    # Similarity
    with st.expander("\U0001f50d Similarity Analysis", expanded=True):
        st.caption(
            "Detect near-duplicate submissions within a batch. "
            "Lower threshold = more aggressive matching."
        )
        c1, c2 = st.columns(2)
        with c1:
            no_similarity = st.checkbox(
                "Disable similarity", key="no_similarity"
            )
        with c2:
            sim_threshold = st.number_input(
                "Threshold",
                min_value=0.0,
                max_value=1.0,
                value=_DEFAULT_SIMILARITY_THRESHOLD,
                step=0.05,
                key="sim_threshold",
            )

        sim_store = st.text_input(
            "Sim store (JSONL path) — optional, for cross-batch memory",
            key="sim_store",
            placeholder="/path/to/sim_store.jsonl",
        )
        instructions = st.text_input(
            "Instructions file path — optional, filters out boilerplate text",
            key="instructions_file",
            placeholder="/path/to/instructions.txt",
        )

    # Output Options
    with st.expander("\U0001f4e4 Output Options", expanded=True):
        output_dir = st.text_input(
            "\U0001f4c2 Batch output folder",
            key="output_dir",
            placeholder="/path/to/output/folder",
            help=(
                "When set, a timestamped subfolder (run_YYYYMMDD_HHMMSS) is "
                "created here after each batch analysis and all outputs are "
                "saved automatically: results CSV, HTML report for flagged "
                "submissions, and a memory store. Individual download buttons "
                "are still available for manual export."
            ),
        )
        c1, c2 = st.columns(2)
        with c1:
            cost = st.number_input(
                "Cost per prompt ($) — used for financial impact estimate",
                min_value=0.0,
                value=400.0,
                step=50.0,
                key="cost_per_prompt",
            )
        with c2:
            collect_path = st.text_input(
                "Collect baselines to JSONL",
                key="collect_path",
                placeholder="/path/to/baselines.jsonl",
            )

    st.success("\u2705 Configuration is saved in the session automatically.")


# ── Page: Memory & Learning ──────────────────────────────────────────────────

def _page_memory():
    st.markdown("### \U0001f9e0 Memory & Learning")
    st.caption("BEET memory store, ground truth, and learning tools")

    # Quick-start callout when no memory store is loaded
    if st.session_state.get("memory_store") is None:
        st.info(
            "\U0001f4a1 **Getting started**: Click **Use Default (.beet)** below to "
            "automatically create and load a memory store in the current working "
            "directory. You can also specify a custom path and click **Load Memory Store**."
        )

    # Memory Store
    with st.expander("\U0001f4be BEET Memory Store", expanded=True):
        st.caption(
            "The memory store persists analysis history, attempter profiles, "
            "and learned models across sessions."
        )

        c_input, c_default = st.columns([3, 1])
        with c_input:
            mem_dir = st.text_input(
                "Store directory",
                placeholder="/path/to/.beet",
                key="memory_dir",
            )
        with c_default:
            st.markdown("&nbsp;", unsafe_allow_html=True)  # vertical spacer
            use_default = st.button("Use Default (.beet)", use_container_width=True)

        if use_default:
            try:
                from llm_detector.memory import MemoryStore
                st.session_state["memory_store"] = MemoryStore(".beet")
                st.success("Memory store ready: .beet")
                _rerun()
            except Exception as e:
                st.error(str(e))

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Load Memory Store"):
                _dir = mem_dir.strip()
                if not _dir:
                    st.warning("Enter a directory path or click **Use Default (.beet)**.")
                else:
                    try:
                        from llm_detector.memory import MemoryStore
                        st.session_state["memory_store"] = MemoryStore(_dir)
                        st.success(f"Loaded: {_dir}")
                        _rerun()
                    except Exception as e:
                        st.error(str(e))

        with c2:
            if st.button("Print Summary"):
                mem = st.session_state.get("memory_store")
                if mem:
                    buf = io.StringIO()
                    old = sys.stdout
                    sys.stdout = buf
                    try:
                        mem.print_summary()
                    finally:
                        sys.stdout = old
                    st.code(buf.getvalue())
                else:
                    st.warning("Load a memory store first.")

        # Show metadata if memory store is loaded
        mem = st.session_state.get("memory_store")
        if mem is not None:
            cfg = getattr(mem, "_config", {})
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("Submissions", cfg.get("total_submissions", 0))
            with mc2:
                st.metric("Batches", cfg.get("total_batches", 0))
            with mc3:
                st.metric("Attempters", cfg.get("total_attempters", 0))
            with mc4:
                st.metric("Confirmed", cfg.get("total_confirmed", 0))
            st.caption(f"Store path: `{mem.store_dir}`  |  Created: {cfg.get('created', 'unknown')[:10]}")

    # Ground Truth Confirmation
    with st.expander("\u2705 Record Ground Truth Confirmation", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            confirm_task = st.text_input("Task ID", key="confirm_task")
        with c2:
            confirm_label = st.selectbox(
                "Label", ["ai", "human", "unsure"], key="confirm_label"
            )
        with c3:
            confirm_reviewer = st.text_input(
                "Reviewer", key="confirm_reviewer"
            )

        if st.button("Confirm"):
            mem = st.session_state.get("memory_store")
            if not mem:
                st.warning("Load a memory store first.")
            elif not confirm_task or not confirm_reviewer:
                st.warning("Task ID and Reviewer are required.")
            else:
                mem.record_confirmation(
                    confirm_task, confirm_label, verified_by=confirm_reviewer
                )
                st.success(
                    f"Confirmed: {confirm_task} = {confirm_label} "
                    f"by {confirm_reviewer}"
                )

    # Quick-confirm from recent results
    with st.expander("\U0001f50d Quick Confirm — Recent Scanned Samples", expanded=False):
        results = st.session_state.get("results", [])
        text_map = st.session_state.get("text_map", {})
        if not results:
            st.info("Run an analysis first to see recent samples here.")
        else:
            qc_reviewer = st.text_input("Reviewer name", key="qc_reviewer")
            for idx, r in enumerate(results):
                tid = r.get("task_id", f"#{idx+1}")
                det = r.get("determination", "?")
                emoji = _DET_EMOJI.get(det, "")
                preview = (text_map.get(tid, "") or "")[:120].replace("\n", " ")
                safe_tid = _html.escape(str(tid))
                safe_det = _html.escape(str(det))
                safe_preview = _html.escape(preview)
                ellipsis = "…" if len(text_map.get(tid, "") or "") > 120 else ""
                st.markdown(
                    f"**{emoji} [{safe_det}] {safe_tid}**  \n"
                    f"<small style='color:#6b7280'>{safe_preview}{ellipsis}</small>",
                    unsafe_allow_html=True,
                )
                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    if st.button("\U0001f9d1 Human", key=f"qc_human_{idx}"):
                        mem = st.session_state.get("memory_store")
                        if not mem:
                            st.warning("Load a memory store first.")
                        elif not qc_reviewer.strip():
                            st.warning("Enter a reviewer name above.")
                        else:
                            mem.record_confirmation(tid, "human", verified_by=qc_reviewer.strip())
                            st.success(f"Confirmed: {tid} = human")
                with bc2:
                    if st.button("\U0001f916 AI", key=f"qc_ai_{idx}"):
                        mem = st.session_state.get("memory_store")
                        if not mem:
                            st.warning("Load a memory store first.")
                        elif not qc_reviewer.strip():
                            st.warning("Enter a reviewer name above.")
                        else:
                            mem.record_confirmation(tid, "ai", verified_by=qc_reviewer.strip())
                            st.success(f"Confirmed: {tid} = ai")
                with bc3:
                    if st.button("? Unsure", key=f"qc_unsure_{idx}"):
                        mem = st.session_state.get("memory_store")
                        if not mem:
                            st.warning("Load a memory store first.")
                        elif not qc_reviewer.strip():
                            st.warning("Enter a reviewer name above.")
                        else:
                            mem.record_confirmation(tid, "unsure", verified_by=qc_reviewer.strip())
                            st.success(f"Confirmed: {tid} = unsure")
                st.divider()

    # Attempter History
    with st.expander("\U0001f464 Attempter History", expanded=False):
        attempter_name = st.text_input("Attempter name", key="hist_attempter")
        if st.button("Show History"):
            mem = st.session_state.get("memory_store")
            if not mem:
                st.warning("Load a memory store first.")
            elif attempter_name.strip():
                buf = io.StringIO()
                old = sys.stdout
                sys.stdout = buf
                try:
                    mem.print_attempter_history(attempter_name.strip())
                finally:
                    sys.stdout = old
                st.code(buf.getvalue())

    # Learning Tools
    with st.expander("\U0001f4da Learning Tools", expanded=False):
        corpus_path = st.text_input(
            "Labeled corpus (JSONL)",
            placeholder="/path/to/corpus.jsonl",
            key="corpus_path",
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            rebuild_shadow_btn = st.button("Rebuild Shadow")
        with c2:
            rebuild_centroids_btn = st.button("Rebuild Centroids")
        with c3:
            discover_lexicon_btn = st.button("Discover Lexicon")
        with c4:
            rebuild_all_btn = st.button("Rebuild All")

        mem = st.session_state.get("memory_store")
        if rebuild_shadow_btn:
            if not mem:
                st.warning("Load a memory store first.")
            else:
                with st.spinner("Rebuilding shadow model..."):
                    pkg = mem.rebuild_shadow_model()
                if pkg:
                    st.success(f"Shadow model rebuilt: AUC={pkg['cv_auc']:.3f}")
                else:
                    st.error("Insufficient labeled data")

        if rebuild_centroids_btn:
            if not mem:
                st.warning("Load a memory store first.")
            elif not corpus_path.strip():
                st.warning("Set a labeled corpus path.")
            else:
                with st.spinner("Rebuilding centroids..."):
                    result = mem.rebuild_semantic_centroids(corpus_path.strip())
                if result:
                    st.success(
                        f"Centroids rebuilt: separation={result['separation']:.4f}"
                    )
                else:
                    st.error("Insufficient labeled text")

        if discover_lexicon_btn:
            if not mem:
                st.warning("Load a memory store first.")
            elif not corpus_path.strip():
                st.warning("Set a labeled corpus path.")
            else:
                with st.spinner("Discovering lexicon..."):
                    candidates = mem.discover_lexicon_candidates(
                        corpus_path.strip()
                    )
                n_new = sum(
                    1
                    for c in candidates
                    if not c.get("already_in_fingerprints")
                    and not c.get("already_in_packs")
                )
                st.success(
                    f"Lexicon discovery: {len(candidates)} candidates "
                    f"({n_new} new)"
                )

        if rebuild_all_btn:
            if not mem:
                st.warning("Load a memory store first.")
            else:
                with st.spinner("Rebuilding all artifacts..."):
                    msgs = []
                    cal = mem.rebuild_calibration()
                    if cal:
                        st.session_state["cal_table"] = cal
                        msgs.append(
                            f"Calibration: {cal['n_calibration']} samples"
                        )
                    else:
                        msgs.append("Calibration: insufficient data")

                    shadow = mem.rebuild_shadow_model()
                    if shadow:
                        msgs.append(
                            f"Shadow model: AUC={shadow['cv_auc']:.3f}"
                        )
                    else:
                        msgs.append("Shadow model: insufficient data")

                    if corpus_path.strip():
                        centroids = mem.rebuild_semantic_centroids(
                            corpus_path.strip()
                        )
                        if centroids:
                            msgs.append(
                                f"Centroids: separation="
                                f"{centroids['separation']:.4f}"
                            )
                        cands = mem.discover_lexicon_candidates(
                            corpus_path.strip()
                        )
                        n_new = sum(
                            1
                            for c in cands
                            if not c.get("already_in_fingerprints")
                            and not c.get("already_in_packs")
                        )
                        msgs.append(
                            f"Lexicon: {len(cands)} candidates ({n_new} new)"
                        )
                for m in msgs:
                    st.info(m)

    # Interactive Labeling
    with st.expander("\U0001f3f7\ufe0f Interactive Labeling", expanded=False):
        st.caption(
            "Review pipeline results one by one and assign ground-truth labels "
            "(AI / Human / Unsure). Labels are written to a JSONL file for "
            "calibration and optionally stored in the memory store. "
            "Run an analysis on the **Analysis** page first."
        )

        results = st.session_state.get("results", [])
        if not results:
            st.info("No results available. Run an analysis on the **Analysis** page first.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                lbl_reviewer = st.text_input(
                    "Reviewer name/ID", key="lbl_reviewer"
                )
                lbl_output = st.text_input(
                    "Output JSONL path",
                    key="lbl_output",
                    placeholder="beet_labels_YYYYMM_HHMM.jsonl",
                )
                lbl_max = st.number_input(
                    "Max labels (0 = all)", min_value=0, value=0, key="lbl_max"
                )
            with c2:
                lbl_skip_green = st.checkbox("Skip GREEN determinations", key="lbl_skip_green")
                lbl_skip_red = st.checkbox("Skip RED determinations", key="lbl_skip_red")

            if st.button("▶ Start / Resume Labeling Session", type="primary"):
                if not lbl_reviewer.strip():
                    st.warning("Enter a reviewer name.")
                else:
                    from llm_detector.cli import _sort_for_labeling
                    from datetime import datetime
                    sorted_res = _sort_for_labeling(results)
                    if lbl_skip_green:
                        sorted_res = [r for r in sorted_res if r["determination"] != "GREEN"]
                    if lbl_skip_red:
                        sorted_res = [r for r in sorted_res if r["determination"] != "RED"]
                    if lbl_max > 0:
                        sorted_res = sorted_res[:lbl_max]
                    out_path = lbl_output.strip() or (
                        f"beet_labels_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
                    )
                    st.session_state["lbl_queue"] = sorted_res
                    st.session_state["lbl_idx"] = 0
                    st.session_state["lbl_out_path"] = out_path
                    st.session_state["lbl_reviewer_name"] = lbl_reviewer.strip()
                    st.session_state["lbl_stats"] = {
                        "labeled_ai": 0, "labeled_human": 0,
                        "labeled_unsure": 0, "skipped": 0,
                    }
                    _rerun()

            # Active labeling session
            queue = st.session_state.get("lbl_queue", [])
            idx = st.session_state.get("lbl_idx", 0)
            if queue and idx < len(queue):
                r = queue[idx]
                stats = st.session_state.get("lbl_stats", {})
                text_map = st.session_state.get("text_map", {})
                session_reviewer = st.session_state.get("lbl_reviewer_name", "")
                out_path = st.session_state.get("lbl_out_path", "labels.jsonl")

                icons = {
                    "RED": "\U0001f534", "AMBER": "\U0001f7e0",
                    "YELLOW": "\U0001f7e1", "GREEN": "\U0001f7e2",
                    "MIXED": "\U0001f535", "REVIEW": "\u26aa",
                }
                icon = icons.get(r.get("determination", ""), "?")
                st.markdown(f"**{icon} [{r.get('determination','?')}]** — "
                            f"conf={r.get('confidence', 0):.2f} | "
                            f"words={r.get('word_count', 0)} | "
                            f"mode={r.get('mode','?')}")
                st.markdown(f"**Task:** {r.get('task_id','?')}  &nbsp; "
                            f"**Attempter:** {r.get('attempter','(unknown)')}  &nbsp; "
                            f"**Occupation:** {r.get('occupation','(unknown)')}")
                st.caption(f"Reason: {r.get('reason','')[:200]}")

                tid = r.get("task_id", "")
                if tid in text_map:
                    with st.expander("Text preview"):
                        st.text(text_map[tid][:500])

                st.progress((idx) / len(queue),
                            text=f"{idx + 1}/{len(queue)} — "
                                 f"{stats.get('labeled_ai',0)} AI / "
                                 f"{stats.get('labeled_human',0)} human / "
                                 f"{stats.get('labeled_unsure',0)} unsure / "
                                 f"{stats.get('skipped',0)} skipped")

                lbl_notes = st.text_input("Notes (optional)", key=f"lbl_notes_{idx}")

                bc1, bc2, bc3, bc4, bc5 = st.columns(5)

                def _record_and_advance(ground_truth):
                    import json
                    from datetime import datetime
                    wc = r.get("word_count", 0)
                    record = {
                        "task_id": r.get("task_id", ""),
                        "attempter": r.get("attempter", ""),
                        "occupation": r.get("occupation", ""),
                        "ground_truth": ground_truth,
                        "pipeline_determination": r.get("determination", ""),
                        "pipeline_confidence": r.get("confidence", 0),
                        "reviewer": session_reviewer,
                        "notes": st.session_state.get(f"lbl_notes_{idx}", ""),
                        "timestamp": datetime.now().isoformat(),
                        "pipeline_version": "v0.66",
                        "confidence": r.get("confidence", 0),
                        "word_count": wc,
                        "domain": r.get("domain", ""),
                        "mode": r.get("mode", ""),
                        "length_bin": (
                            "short" if wc < 100 else
                            "medium" if wc < 300 else
                            "long" if wc < 800 else
                            "very_long"
                        ),
                    }
                    with open(out_path, "a") as fh:
                        fh.write(json.dumps(record) + "\n")
                    mem = st.session_state.get("memory_store")
                    if mem and ground_truth in ("ai", "human"):
                        mem.record_confirmation(
                            r.get("task_id", ""), ground_truth,
                            verified_by=session_reviewer,
                            notes=record["notes"],
                        )
                    if ground_truth == "ai":
                        st.session_state["lbl_stats"]["labeled_ai"] += 1
                    elif ground_truth == "human":
                        st.session_state["lbl_stats"]["labeled_human"] += 1
                    else:
                        st.session_state["lbl_stats"]["labeled_unsure"] += 1
                    st.session_state["lbl_idx"] += 1
                    _rerun()

                with bc1:
                    if st.button("\U0001f916 AI", use_container_width=True, key=f"lbl_ai_{idx}"):
                        _record_and_advance("ai")
                with bc2:
                    if st.button("\U0001f9d1 Human", use_container_width=True, key=f"lbl_human_{idx}"):
                        _record_and_advance("human")
                with bc3:
                    if st.button("? Unsure", use_container_width=True, key=f"lbl_unsure_{idx}"):
                        _record_and_advance("unsure")
                with bc4:
                    if st.button("Skip", use_container_width=True, key=f"lbl_skip_{idx}"):
                        st.session_state["lbl_stats"]["skipped"] += 1
                        st.session_state["lbl_idx"] += 1
                        _rerun()
                with bc5:
                    if st.button("Quit", use_container_width=True, key=f"lbl_quit_{idx}"):
                        st.session_state["lbl_queue"] = []
                        _rerun()

            elif queue and idx >= len(queue):
                stats = st.session_state.get("lbl_stats", {})
                out_path = st.session_state.get("lbl_out_path", "labels.jsonl")
                st.success(
                    f"\u2705 Session complete — "
                    f"{stats.get('labeled_ai', 0)} AI, "
                    f"{stats.get('labeled_human', 0)} human, "
                    f"{stats.get('labeled_unsure', 0)} unsure, "
                    f"{stats.get('skipped', 0)} skipped. "
                    f"Saved to `{out_path}`."
                )
                st.session_state["lbl_queue"] = []


# ── Page: Calibration ────────────────────────────────────────────────────────

def _page_calibration():
    st.markdown("### \u2696\ufe0f Calibration & Baselines")
    st.caption("Conformal calibration and baseline analysis")

    # Calibration
    with st.expander("\U0001f4cf Conformal Calibration", expanded=True):
        cal_path = st.text_input(
            "Calibration table (JSON)",
            placeholder="/path/to/calibration.json",
            key="cal_table_path",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Load Calibration"):
                if cal_path.strip() and os.path.exists(cal_path.strip()):
                    from llm_detector.calibration import load_calibration

                    cal = load_calibration(cal_path.strip())
                    st.session_state["cal_table"] = cal
                    st.success(
                        f"Loaded: {cal['n_calibration']} records, "
                        f"{len(cal.get('strata', {}))} strata"
                    )
                else:
                    st.warning("Select a valid JSON file.")

        with c2:
            if st.button("Rebuild from Memory"):
                mem = st.session_state.get("memory_store")
                if not mem:
                    st.warning("Load a memory store first.")
                else:
                    cal = mem.rebuild_calibration()
                    if cal:
                        st.session_state["cal_table"] = cal
                        st.success(
                            f"Rebuilt: {cal['n_calibration']} samples"
                        )
                    else:
                        st.error("Insufficient data")

        build_from = st.text_input(
            "Build from JSONL",
            placeholder="/path/to/baselines.jsonl",
            key="cal_build_jsonl",
        )
        if st.button("Build & Save Calibration"):
            if not build_from.strip() or not os.path.exists(
                build_from.strip()
            ):
                st.warning("Select a valid baselines JSONL file.")
            else:
                from llm_detector.calibration import (
                    calibrate_from_baselines,
                    save_calibration,
                )

                cal = calibrate_from_baselines(build_from.strip())
                if cal is None:
                    st.error(
                        "Need >= 20 labeled human samples for calibration"
                    )
                else:
                    out_path = build_from.strip().replace(
                        ".jsonl", "_calibration.json"
                    )
                    save_calibration(cal, out_path)
                    st.session_state["cal_table"] = cal
                    st.success(
                        f"Built: {cal['n_calibration']} records, "
                        f"{len(cal.get('strata', {}))} strata "
                        f"→ {out_path}"
                    )

    # Baseline Analysis
    with st.expander("\U0001f4ca Baseline Analysis", expanded=True):
        bl_jsonl = st.text_input(
            "Baselines JSONL",
            placeholder="/path/to/baselines.jsonl",
            key="bl_jsonl_path",
        )
        bl_csv = st.text_input(
            "Output CSV (optional)",
            key="bl_csv_path",
        )
        if st.button("Analyze Baselines"):
            if not bl_jsonl.strip() or not os.path.exists(bl_jsonl.strip()):
                st.warning("Select a valid baselines JSONL file.")
            else:
                buf = io.StringIO()
                old = sys.stdout
                sys.stdout = buf
                try:
                    from llm_detector.baselines import analyze_baselines

                    analyze_baselines(
                        bl_jsonl.strip(),
                        output_csv=bl_csv.strip() or None,
                    )
                finally:
                    sys.stdout = old
                st.code(buf.getvalue())

    # Calibration Diagnostics Report
    with st.expander("\U0001f50e Calibration Diagnostics Report", expanded=False):
        st.caption(
            "Generate a diagnostics report from a labeled JSONL file "
            "(records with ground_truth='ai' or 'human'). "
            "Shows confusion matrix, reliability diagram, TPR at fixed FPR, "
            "and per-stratum calibration."
        )
        cr_jsonl = st.text_input(
            "Labeled JSONL",
            placeholder="/path/to/labeled.jsonl",
            key="cal_report_jsonl",
        )
        cr_csv = st.text_input(
            "Export labeled data to CSV (optional)",
            key="cal_report_csv",
            placeholder="/path/to/output.csv",
        )
        if st.button("Generate Calibration Report"):
            if not cr_jsonl.strip() or not os.path.exists(cr_jsonl.strip()):
                st.warning("Select a valid labeled JSONL file.")
            else:
                buf = io.StringIO()
                old = sys.stdout
                sys.stdout = buf
                try:
                    from llm_detector.cli import calibration_report
                    calibration_report(
                        cr_jsonl.strip(),
                        cal_table=st.session_state.get("cal_table"),
                        output_csv=cr_csv.strip() or None,
                    )
                finally:
                    sys.stdout = old
                st.code(buf.getvalue())


# ── Page: Reports ────────────────────────────────────────────────────────────

def _page_reports():
    st.markdown("### \U0001f4ca Reports")
    st.caption("Batch summaries, attempter profiles, and financial impact")

    results = st.session_state.get("results", [])
    if not results:
        st.info(
            "No results available. Run a batch analysis on the "
            "**Analysis** page first."
        )
        return

    # Determination Distribution
    with st.expander("\U0001f4ca Determination Distribution", expanded=True):
        counts = Counter(r["determination"] for r in results)
        det_order = ["RED", "AMBER", "MIXED", "YELLOW", "REVIEW", "GREEN"]
        chart_data = pd.DataFrame(
            {
                "Determination": det_order,
                "Count": [counts.get(d, 0) for d in det_order],
            }
        )
        chart_data = chart_data[chart_data["Count"] > 0]
        st.bar_chart(chart_data.set_index("Determination"))

        # Percentage breakdown
        total = len(results)
        for det in det_order:
            ct = counts.get(det, 0)
            if ct > 0:
                pct = ct / total * 100
                emoji = _DET_EMOJI.get(det, "")
                st.markdown(
                    f"{emoji} **{det}**: {ct} ({pct:.1f}%)"
                )

    # Attempter Profiles
    if len(results) >= 5:
        with st.expander("\U0001f464 Attempter Risk Profiles", expanded=True):
            try:
                from llm_detector.reporting import profile_attempters

                profiles = profile_attempters(results)
                if profiles:
                    df = pd.DataFrame(profiles[:20])
                    display_cols = [
                        c
                        for c in [
                            "attempter",
                            "n_submissions",
                            "flag_rate",
                            "mean_confidence",
                        ]
                        if c in df.columns
                    ]
                    st.dataframe(
                        df[display_cols],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No attempter profiles available.")
            except Exception:
                st.info("Attempter profiling unavailable.")

    # Channel Patterns
    flagged = [
        r
        for r in results
        if r["determination"] in ("RED", "AMBER", "MIXED")
    ]
    if flagged:
        with st.expander(
            "\U0001f50d Channel Patterns (flagged)", expanded=True
        ):
            channel_counts = Counter()
            for r in flagged:
                cd = r.get("channel_details", {}).get("channels", {})
                for ch_name, info in cd.items():
                    if info.get("severity") not in ("GREEN", None):
                        channel_counts[ch_name] += 1
            if channel_counts:
                df = pd.DataFrame(
                    channel_counts.most_common(),
                    columns=["Channel", "Flags"],
                )
                st.bar_chart(df.set_index("Channel"))
            else:
                st.info("No channel patterns detected.")

    # Financial Impact
    if len(results) >= 10:
        with st.expander(
            "\U0001f4b0 Financial Impact Estimate", expanded=True
        ):
            try:
                from llm_detector.reporting import financial_impact

                cost = st.session_state.get("cost_per_prompt", 400.0)
                impact = financial_impact(results, cost_per_prompt=cost)

                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric(
                        "Total Submissions",
                        impact["total_submissions"],
                    )
                with mc2:
                    st.metric(
                        "Flag Rate", f"{impact['flag_rate']:.1%}"
                    )
                with mc3:
                    st.metric(
                        "Waste Estimate",
                        f"${impact['waste_estimate']:,.0f}",
                    )
                with mc4:
                    st.metric(
                        "Projected Annual",
                        f"${impact.get('projected_annual_waste', 0):,.0f}",
                    )
            except Exception:
                st.info("Financial impact calculation unavailable.")

    # Export
    st.markdown("---")
    if st.button("Export Baselines"):
        try:
            from llm_detector.baselines import collect_baselines

            buf = io.BytesIO()
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".jsonl"
            ) as tmp:
                collect_baselines(results, tmp.name)
                with open(tmp.name, "r") as f:
                    data = f.read()
                os.unlink(tmp.name)
            st.download_button(
                "\U0001f4be Download Baselines JSONL",
                data,
                "baselines.jsonl",
                "application/jsonl",
            )
        except Exception as e:
            st.error(str(e))


# ── Page: Quick Reference ────────────────────────────────────────────────────

_QUICK_REFERENCE_TEXT = """\
CHANNELS (fusion combines these into a final determination)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• **prompt_structure** — Rule-based structural analysis: preamble detection, \
fingerprint matching, prompt-signature scoring (CFD, MFSR, framing), \
instruction density (IDI), and voice dissonance (VSD).
• **stylometry** — Statistical stylometric features: function-word ratio, \
sentence-length dispersion, TTR, avg word length, short-word ratio, masked \
topical tokens.
• **continuation** — DNA-GPT continuation analysis: generates LLM \
continuations and measures B-score, NCD, overlap, conditional surprisal, \
repeat-4 rate, TTR.  Requires API key (Anthropic / OpenAI) or local model.
• **windowing** — Sliding-window analysis: max/mean window score, variance, \
hot-span detection, FW trajectory CV, comp trajectory, changepoint detection.

INDIVIDUAL ANALYZERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
| Analyzer | Key Signals |
|---|---|
| **Preamble** | Detects LLM boilerplate openings. Score, severity, hit count. |
| **Fingerprint** | Pattern-matches known LLM output fingerprints. |
| **Prompt Signature** | CFD, MFSR, distinct frames, framing completeness, conditional density, meta-design, contractions, must-rate, numbered criteria. |
| **IDI** | Instruction Density Index: imperatives, conditionals, binary specs, missing references, flag count. |
| **VSD** | Voice Dissonance Score: voice × spec, casual markers, misspellings, hedges, CamelCase columns, calculations. |
| **SSI / NSSI** | Self-Similarity (Normalised): formulaic density, transition density, scare-quote/em-dash density, compression ratio, hapax ratio. |
| **DNA-GPT** | Continuation B-score, NCD, internal overlap, conditional surprisal, repeat-4, TTR, composite stability. |
| **Perplexity** | Mean PPL, burstiness, surprisal variance, volatility decay, Binoculars score, zlib-normalised PPL. |
| **TOCSIN** | Token Cohesiveness: cohesiveness score and std. |
| **Semantic Resonance** | AI/human centroid similarity: AI score, human score, delta. |
| **Semantic Flow** | Cross-sentence semantic coherence (cosine similarity). |
| **Lexicon Packs** | Domain-specific lexicon: constraint, exec-spec, schema scores. |

POST-PROCESSING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• **Similarity** — Within-batch Jaccard similarity for near-duplicate detection.
• **Cross-batch** — Persistent similarity store (JSONL) across sessions.
• **Shadow Model** — ML classifier trained on confirmed labels; detects disagreements.
• **Calibration** — Conformal prediction tables for well-calibrated probabilities.
• **Normalization** — Unicode obfuscation detection: invisible chars, homoglyphs.
• **Language Gate** — Non-English/non-Latin language support level check.
"""


def _page_quick_reference():
    st.markdown("### \U0001f4d6 Quick Reference")
    st.caption("Summary of every analysis, channel, and signal in the detection pipeline")
    st.markdown(_QUICK_REFERENCE_TEXT)


# ── Page: Precheck ───────────────────────────────────────────────────────────

def _check_dependencies_st():
    """Return list of (status, name, category, notes) for dependency checks."""
    import importlib.util as iu
    checks = []

    def _probe(mod, display, cat, required=True, note_ok="", note_miss=""):
        try:
            ok = iu.find_spec(mod) is not None
        except (ModuleNotFoundError, ValueError):
            ok = False
        if ok:
            checks.append(("\u2705", display, cat, note_ok or "Available"))
        elif required:
            checks.append(("\u274c", display, cat, note_miss or "Missing — required"))
        else:
            checks.append(("\u2757", display, cat, note_miss or "Missing — optional"))

    _probe("pandas", "pandas", "Core", True, "DataFrame processing")
    _probe("openpyxl", "openpyxl", "Core", True, "Excel I/O")
    _probe("spacy", "spacy", "NLP", False, note_miss="Optional — regex fallback used")
    _probe("ftfy", "ftfy", "NLP", False, note_miss="Optional — text normalisation")
    _probe("sentence_transformers", "sentence-transformers", "NLP", False,
           note_miss="Optional — semantic resonance disabled")
    _probe("sklearn", "scikit-learn", "NLP", False,
           note_miss="Optional — ML fusion disabled")
    _probe("transformers", "transformers (HuggingFace)", "Perplexity", False,
           note_miss="Optional — perplexity analyser disabled")
    _probe("torch", "PyTorch", "Perplexity", False,
           note_miss="Optional — perplexity analyser disabled")
    _probe("anthropic", "anthropic SDK", "API", False,
           note_miss="Optional — Anthropic continuation disabled")
    _probe("openai", "openai SDK", "API", False,
           note_miss="Optional — OpenAI continuation disabled")
    _probe("pypdf", "pypdf", "PDF", False, note_miss="Optional — PDF ingestion disabled")
    _probe("streamlit", "streamlit", "Web", False,
           note_miss="Optional — auto-install available")
    _probe("tkinter", "tkinter", "GUI", False,
           note_miss="Optional — desktop GUI unavailable")
    _probe("PyInstaller", "PyInstaller", "Build", False,
           note_miss="Optional — executable building only")
    return checks


_PIP_INSTALL_MAP_ST = {
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


def _page_precheck():
    st.markdown("### \u2705 Precheck")
    st.caption(
        "Required and optional Python modules, models, and external programs. "
        "\u2705 = available   \u2757 = missing (optional)   \u274c = missing (breaks analysis)"
    )
    rows = _check_dependencies_st()
    df = pd.DataFrame(rows, columns=["Status", "Component", "Category", "Notes"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Summary counts
    ok_count = sum(1 for r in rows if r[0] == "\u2705")
    warn_count = sum(1 for r in rows if r[0] == "\u2757")
    err_count = sum(1 for r in rows if r[0] == "\u274c")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Available", ok_count)
    with c2:
        st.metric("Optional Missing", warn_count)
    with c3:
        st.metric("Required Missing", err_count)

    # Install missing optional dependencies
    missing = [r[1] for r in rows if r[0] == "\u2757" and r[1] in _PIP_INSTALL_MAP_ST]
    if missing:
        st.markdown("---")
        st.markdown("**Install missing optional dependencies**")
        selected = st.multiselect("Select dependencies to install:", missing, default=[])
        if st.button("Install Selected", disabled=not selected):
            packages = sorted({_PIP_INSTALL_MAP_ST[name] for name in selected})
            with st.spinner(f"Installing {', '.join(packages)}\u2026"):
                try:
                    subprocess.check_call(
                        [sys.executable, '-m', 'pip', 'install'] + packages,
                        stdout=subprocess.DEVNULL,
                    )
                    st.success("Installation complete. Refresh the page to update the precheck.")
                    st.rerun()
                except (subprocess.CalledProcessError, FileNotFoundError):
                    st.error(
                        "Installation failed. Try manually:\n\n"
                        f"```\npip install {' '.join(packages)}\n```"
                    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Streamlit dashboard entry point."""
    _configure_page()
    _init_state()
    page = _render_sidebar()

    if page.endswith("Analysis"):
        _page_analysis()
    elif page.endswith("Configuration"):
        _page_configuration()
    elif page.endswith("Memory & Learning"):
        _page_memory()
    elif page.endswith("Calibration"):
        _page_calibration()
    elif page.endswith("Reports"):
        _page_reports()
    elif page.endswith("Quick Reference"):
        _page_quick_reference()
    elif page.endswith("Precheck"):
        _page_precheck()


if __name__ == "__main__":
    main()
