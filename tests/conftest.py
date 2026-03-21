"""Shared fixtures and sample texts for pipeline tests."""

import pytest

AI_TEXT = (
    "This comprehensive analysis provides a thorough examination of the "
    "key factors that contribute to the overall effectiveness of the proposed "
    "framework. Furthermore, it is essential to note that the implementation "
    "of these strategies ensures alignment with best practices and industry "
    "standards. To address this challenge, we must consider multiple perspectives "
    "and leverage data-driven insights to achieve optimal outcomes. Additionally, "
    "this approach demonstrates the critical importance of systematic evaluation "
    "and evidence-based decision making in the modern landscape."
)

HUMAN_TEXT = (
    "so yeah I just kinda threw together a quick script to parse the logs "
    "and honestly it's pretty janky but it works lol. the main thing was "
    "getting the regex right for the timestamps because some of them had "
    "weird formats and I kept hitting edge cases. anyway I pushed it to the "
    "repo if you wanna take a look, but fair warning it's not exactly "
    "production ready haha. oh and I forgot to mention, there's a bug where "
    "it chokes on empty lines but I'll fix that tomorrow probably."
)

CLINICAL_TEXT = (
    "The patient presented to the emergency department with acute chest pain "
    "radiating to the left arm. Vital signs were stable with blood pressure "
    "of 130/85 mmHg and heart rate of 92 beats per minute. An electrocardiogram "
    "was performed which showed ST-segment elevation in leads V1 through V4. "
    "The patient was immediately started on aspirin and heparin therapy."
)


# ── pytest fixtures for edge-case testing ────────────────────────────────────

@pytest.fixture
def short_text():
    """Text under 50 words for short-text path testing."""
    return "Short text. Brief. Minimal content here for testing purposes only."


@pytest.fixture
def non_english_text():
    """Non-English text for language gate / fairness testing."""
    return (
        "这是一个中文测试文本，用于测试语言检测功能和公平性限制逻辑。"
        "本测试涵盖多个常见的中文句子，确保系统能够正确识别非英语内容。"
    ) * 5


@pytest.fixture
def obfuscated_text():
    """Text with zero-width characters simulating obfuscation attacks."""
    return (
        "This text cont\u200bains zero-w\u200bidth chars and inv\u200bisible "
        "Unicode separ\u200bators inserted to evade detection systems. "
        "The qu\u200bick brown fox jumps over the lazy dog."
    )


@pytest.fixture
def mixed_human_ai_text():
    """Text with distinct human and AI sections for windowed testing."""
    return (
        "Hey, so I was thinking about this thing... you know how sometimes you "
        "just need to get stuff done quickly? Yeah, totally. Anyway, I tried it "
        "and it worked, kind of. Not perfect but good enough for now I guess. "
        "The implementation of advanced algorithmic methodologies necessitates "
        "a comprehensive evaluation framework that systematically addresses "
        "the multifaceted challenges inherent in contemporary computational "
        "paradigms. Furthermore, the holistic approach ensures alignment with "
        "industry best practices and evidence-based decision-making strategies."
    )


@pytest.fixture
def ai_task_prompt():
    """Typical AI-generated task prompt with structural signals."""
    return (
        "Please analyze the following dataset and provide a comprehensive "
        "assessment. You must include: (1) statistical summary, (2) trend "
        "analysis, and (3) actionable recommendations. Ensure all findings "
        "are presented with supporting evidence. The analysis should follow "
        "structured methodology and adhere to best practices. Additionally, "
        "you must validate your results against established benchmarks."
    )


@pytest.fixture
def human_task_prompt():
    """Typical human-written task prompt without AI structural signals."""
    return (
        "hey can you look at this data? I'm trying to figure out what's "
        "going on with the numbers - they seem off but I'm not sure why. "
        "maybe just pull out the main trends? don't need anything fancy, "
        "just enough to understand what's happening. thanks"
    )
