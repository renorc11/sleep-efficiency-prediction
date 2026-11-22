"""Shared checks for text produced by the local BioMistral model.

These checks never trust the LLM. Each returns a list of human-readable
problems; an empty list means the check passed. They are deliberately
conservative pattern checks, not a guarantee of medical safety, and are
combined with deterministic fallbacks by the calling services.
"""

from __future__ import annotations

import re
from typing import Iterable

# Knowledge-base identifiers: sources (SRC-001) and chunks (SLEEP-003,
# CAFF-001, ...). Anything shaped like these must have been retrieved.
SOURCE_ID_PATTERN = re.compile(r"\bSRC-\d{3}\b")
CHUNK_ID_PATTERN = re.compile(
    r"\b(?:SLEEP|CAFF|ALC|MEAL|ACT|STRESS|SCREEN|NAP|REG|HISTORY|ENV|DIS|TECH|HYGIENE)-\d{3}\b"
)

PLACEHOLDER_PATTERN = re.compile(r"\[[A-Z_ ]{3,}\]|\[[^\]]*(?:name|date|email|user|website)[^\]]*\]", re.IGNORECASE)
LETTER_PATTERN = re.compile(
    r"^\s*(?:subject\s*:|dear\b|to whom it may concern|best regards|sincerely|yours (?:truly|sincerely))",
    re.IGNORECASE | re.MULTILINE,
)

# Medication / supplement advice is outside the assistant's scope.
MEDICATION_PATTERN = re.compile(
    r"\b(?:melatonin|zolpidem|ambien|eszopiclone|zaleplon|benzodiazepines?|temazepam|"
    r"lorazepam|diazepam|trazodone|doxepin|suvorexant|diphenhydramine|doxylamine|"
    r"antihistamines?|sleeping pills?|sleep(?:ing)? aids?|sedatives?|hypnotics?|"
    r"valerian|magnesium supplements?|cbd)\b",
    re.IGNORECASE,
)
DOSAGE_PATTERN = re.compile(r"\b(?:take|taking|dose of|dosage)\b[^.]{0,40}\b\d+\s*(?:mg|mcg|milligrams?)\b", re.IGNORECASE)

# Statements that assign a diagnosis to the user.
DIAGNOSIS_PATTERN = re.compile(
    r"\byou (?:have|suffer from|are suffering from|likely have|probably have|have been diagnosed with)\s+"
    r"(?:an?\s+)?(?:insomnia|sleep apnea|sleep apnoea|narcolepsy|restless legs|"
    r"circadian rhythm (?:sleep )?disorder|(?:a )?sleep disorder|depression|anxiety disorder)\b",
    re.IGNORECASE,
)

# Inactivity is a behavioural proxy and never proof of sleep.
INACTIVITY_PROOF_PATTERN = re.compile(
    r"\b(?:inactivity|screen inactivity|phone inactivity|no (?:screen|phone) activity)\b[^.]{0,60}"
    r"\b(?:proves?|confirms?|shows? that you (?:slept|were asleep)|means you (?:slept|were asleep))\b",
    re.IGNORECASE,
)

# Causal certainty drawn from feature attributions.
ATTRIBUTION_CAUSAL_PATTERN = re.compile(
    r"\b(?:will|guaranteed to|certainly|definitely)\s+(?:increase|improve|raise|boost)\b[^.]{0,60}"
    r"\b(?:sleep efficiency|your score|by \d)",
    re.IGNORECASE,
)

UNSAFE_EXTREME_PATTERN = re.compile(
    r"\b(?:stop eating|fast for|skip (?:all )?meals|"
    r"(?:no sleep|stay awake|go without sleep) for \d+\s*(?:hours?|hrs?|days?)|"
    r"\d{2,}\s*(?:hours?|hrs?) of (?:exercise|training))\b",
    re.IGNORECASE,
)


LEADING_META_PATTERN = re.compile(
    r"^\s*(?:(?:the )?assistant (?:should|would|will) (?:respond|reply|answer)(?: with)?\s*:|answer\s*:)\s*",
    re.IGNORECASE,
)


def strip_meta_prefix(text: str) -> str:
    """Remove leaked prompt scaffolding such as 'Answer:' (cosmetic only)."""
    cleaned = LEADING_META_PATTERN.sub("", text, count=1).strip()
    if cleaned != text.strip() and len(cleaned) > 1 and cleaned[0] == cleaned[-1] == '"':
        cleaned = cleaned[1:-1].strip()
    return cleaned or text.strip()


def find_unknown_identifiers(text: str, allowed_ids: Iterable[str]) -> list[str]:
    """Return knowledge identifiers mentioned in text that were not retrieved."""
    allowed = set(allowed_ids)
    mentioned = set(SOURCE_ID_PATTERN.findall(text)) | set(CHUNK_ID_PATTERN.findall(text))
    return sorted(mentioned - allowed)


def check_format(text: str) -> list[str]:
    problems = []
    if not isinstance(text, str) or not text.strip():
        return ["Empty output."]
    if PLACEHOLDER_PATTERN.search(text):
        problems.append("Unresolved placeholder found.")
    if LETTER_PATTERN.search(text):
        problems.append("Letter or email format found.")
    return problems


def check_safety(text: str) -> list[str]:
    problems = []
    checks = (
        (MEDICATION_PATTERN, "Medication or supplement content found."),
        (DOSAGE_PATTERN, "Dosage instruction found."),
        (DIAGNOSIS_PATTERN, "Diagnostic statement about the user found."),
        (INACTIVITY_PROOF_PATTERN, "Claim that inactivity proves sleep found."),
        (ATTRIBUTION_CAUSAL_PATTERN, "Guaranteed-improvement claim found."),
        (UNSAFE_EXTREME_PATTERN, "Extreme lifestyle advice found."),
    )
    for pattern, message in checks:
        if pattern.search(text):
            problems.append(message)
    return problems


def validate_free_text(text: str, allowed_ids: Iterable[str]) -> list[str]:
    """Format, citation, and safety problems for free-text LLM output."""
    problems = check_format(text)
    if problems == ["Empty output."]:
        return problems
    unknown = find_unknown_identifiers(text, allowed_ids)
    if unknown:
        problems.append(f"Citation not present in retrieved evidence: {', '.join(unknown)}.")
    problems.extend(check_safety(text))
    return problems
