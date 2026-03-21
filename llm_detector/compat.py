"""
Feature detection and optional dependency management.

Centralizes all try/except ImportError blocks so other modules can check
availability flags without repeating import logic.

Models are lazily loaded on first use via getter functions to avoid
slow imports when only checking flags or running --help.
"""

import os
import re
import logging

logger = logging.getLogger(__name__)

def _get_hf_token():
    """Return Hugging Face token from environment (HF_TOKEN → HUGGINGFACEHUB_API_TOKEN)."""
    return os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')

# ── tkinter ──────────────────────────────────────────────────────────────────
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    # Verify that a display is available (fails in headless CI environments)
    try:
        _tk_test = tk.Tk()
        _tk_test.withdraw()
        _tk_test.destroy()
        del _tk_test
        HAS_TK = True
    except tk.TclError:
        HAS_TK = False
except ImportError:
    HAS_TK = False

# ── spaCy: lightweight sentencizer ──────────────────────────────────────────
try:
    import spacy  # noqa: F401
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    logger.info("spacy not installed. Sentence segmentation will use regex fallback.")

_nlp = None

def get_nlp():
    """Return spaCy sentencizer, initializing on first call."""
    global _nlp
    if _nlp is None and HAS_SPACY:
        from spacy.lang.en import English
        _nlp = English()
        _nlp.add_pipe("sentencizer")
    return _nlp

# ── ftfy: robust text encoding repair ───────────────────────────────────────
try:
    import ftfy  # noqa: F401
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False

# ── sentence-transformers: semantic vector analysis ─────────────────────────
try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
    from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401
    import numpy  # noqa: F401
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False
except Exception as e:
    HAS_SEMANTIC = False
    logger.info("sentence-transformers setup failed (%s). Semantic layer disabled.", e)

_EMBEDDER = None
_AI_CENTROIDS = None
_HUMAN_CENTROIDS = None
_EXTRA_CENTROID_PATHS = []


def register_centroid_path(directory):
    """Register an additional directory to search for centroid files.

    Call this before the first analysis so that get_semantic_models()
    picks up centroids from a custom --memory store directory.
    Resets cached centroids so the new path takes effect.
    """
    global _AI_CENTROIDS, _HUMAN_CENTROIDS
    centroid_file = os.path.join(str(directory), 'centroids', 'centroids_latest.npz')
    if centroid_file not in _EXTRA_CENTROID_PATHS:
        _EXTRA_CENTROID_PATHS.insert(0, centroid_file)
        # Reset cached centroids so they reload from the new path
        _AI_CENTROIDS = None
        _HUMAN_CENTROIDS = None


def get_semantic_models():
    """Return (embedder, ai_centroids, human_centroids), loading on first call.

    Checks for data-derived centroids in .beet/centroids/centroids_latest.npz
    before falling back to hardcoded archetypes.
    """
    global _EMBEDDER, _AI_CENTROIDS, _HUMAN_CENTROIDS
    if _EMBEDDER is None and HAS_SEMANTIC:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        _EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2', token=_get_hf_token())

        centroid_paths = [
            '.beet/centroids/centroids_latest.npz',
            os.path.expanduser('~/.beet/centroids/centroids_latest.npz'),
        ]

        loaded = False
        for cpath in centroid_paths:
            if os.path.exists(cpath):
                try:
                    data = np.load(cpath)
                    if 'ai_multi' in data and data['ai_multi'].shape[0] > 1:
                        _AI_CENTROIDS = data['ai_multi']
                        _HUMAN_CENTROIDS = data['human_multi']
                    else:
                        _AI_CENTROIDS = data['ai_centroid']
                        _HUMAN_CENTROIDS = data['human_centroid']
                    loaded = True
                    break
                except Exception:
                    continue

        if not loaded:
            _AI_ARCHETYPES = [
                "As an AI language model, I cannot provide personal opinions.",
                "Here is a comprehensive breakdown of the key factors to consider.",
                "To address this challenge, we must consider multiple perspectives.",
                "This thorough analysis demonstrates the critical importance of the topic.",
                "Furthermore, it is essential to note that this approach ensures alignment.",
                "In conclusion, by leveraging these strategies we can achieve optimal results.",
            ]
            _HUMAN_ARCHETYPES = [
                "honestly idk maybe try restarting it lol",
                "so I went ahead and just hacked together a quick script",
                "tbh the whole thing is kinda janky but it works",
                "yeah no that's totally wrong, here's what actually happened",
                "I messed around with it for a bit and got something working",
            ]
            _AI_CENTROIDS = _EMBEDDER.encode(_AI_ARCHETYPES)
            _HUMAN_CENTROIDS = _EMBEDDER.encode(_HUMAN_ARCHETYPES)
    return _EMBEDDER, _AI_CENTROIDS, _HUMAN_CENTROIDS

# ── transformers: local perplexity scoring ──────────────────────────────────
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    import torch  # noqa: F401
    HAS_PERPLEXITY = True
except ImportError:
    HAS_PERPLEXITY = False
except Exception as e:
    HAS_PERPLEXITY = False
    logger.info("transformers/torch setup failed (%s). Perplexity scoring disabled.", e)

# Default and available perplexity models (model_id -> short label)
PPL_MODELS = {
    'Qwen/Qwen2.5-0.5B': 'Qwen2.5-0.5B',
    'HuggingFaceTB/SmolLM2-360M': 'SmolLM2-360M',
    'HuggingFaceTB/SmolLM2-135M': 'SmolLM2-135M',
    'distilgpt2': 'DistilGPT-2 (legacy)',
    'gpt2': 'GPT-2',
}
PPL_DEFAULT_MODEL = 'Qwen/Qwen2.5-0.5B'

_PPL_MODEL = None
_PPL_TOKENIZER = None
_PPL_MODEL_ID = None

def get_perplexity_model(model_id=None):
    """Return (model, tokenizer), loading on first call.

    Args:
        model_id: HuggingFace model identifier. Defaults to PPL_DEFAULT_MODEL.
                  If a different model_id is passed than what's currently loaded,
                  the model is reloaded.
    """
    global _PPL_MODEL, _PPL_TOKENIZER, _PPL_MODEL_ID
    if model_id is None:
        model_id = PPL_DEFAULT_MODEL
    if (_PPL_MODEL is None or _PPL_MODEL_ID != model_id) and HAS_PERPLEXITY:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        logger.info("Loading perplexity model: %s", model_id)
        _PPL_MODEL_ID = model_id
        hf_token = _get_hf_token()
        _PPL_MODEL = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, token=hf_token,
            torch_dtype=torch.float32,
        )
        _PPL_TOKENIZER = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, token=hf_token
        )
        _PPL_MODEL.eval()
    return _PPL_MODEL, _PPL_TOKENIZER

# ── Binoculars: contrastive LM scoring (observer model) ─────────────────────
HAS_BINOCULARS = HAS_PERPLEXITY  # Same deps, just needs second model

_BINO_MODEL = None
_BINO_TOKENIZER = None

def get_binoculars_model():
    """Return (observer_model, observer_tokenizer) for contrastive scoring."""
    global _BINO_MODEL, _BINO_TOKENIZER
    if _BINO_MODEL is None and HAS_BINOCULARS:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        # Use distilgpt2 as observer — small and different enough for contrastive signal
        hf_token = _get_hf_token()
        _BINO_MODEL = AutoModelForCausalLM.from_pretrained(
            'distilgpt2', token=hf_token, torch_dtype=torch.float32,
        )
        _BINO_TOKENIZER = AutoTokenizer.from_pretrained('distilgpt2', token=hf_token)
        _BINO_MODEL.eval()
    return _BINO_MODEL, _BINO_TOKENIZER

# ── pypdf: PDF text extraction ──────────────────────────────────────────────
try:
    from pypdf import PdfReader  # noqa: F401
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
