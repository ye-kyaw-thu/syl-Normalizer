"""
Burmese Syllable Normalizer
===========================

Normalizes Myanmar text using a multi-stage pipeline:

  Stage 0 – Unicode NFC normalization
  Stage 1 – Regex-based rule corrections (medial ordering, vowel fixes, etc.)
  Stage 2 – Fuzzy correction:
             (a) TargetedFuzzyCorrector – checks candidates against the
                 syllable frequency dictionary (--fuzzy-distance 1, no --ngram-lm)
             (b) NgramFuzzyCorrector    – scores candidates with an ARPA
                 n-gram language model and accepts only if the candidate is
                 strictly more probable than the original token (--ngram-lm)
  Stage 3 – Consonant+Asat merge with previous syllable
  Stage 4 – Compound syllable splitting (DP, ≤ 3 parts)

Written by Ye Kyaw Thu, LU Lab., Myanmar.
Version: 0.6

Changes from 0.5:
  * Stage 2 fuzzy correction now optionally driven by an ARPA n-gram LM
    (--ngram-lm).  Candidates are scored with Katz backoff using up to
    2 tokens of left context; a correction is accepted only when the
    candidate scores strictly better than the original token, preventing
    spurious corrections on foreign-name transliterations.
  * Log now records every individual regex rule application with its rule
    ID and a short description for detailed error analysis.
  * New --min-lm-improve flag (log10 units, default 0.5): candidate must
    be at least this much more probable than the original before the LM
    correction fires.
  * New NgramLanguageModel class: loads ARPA trigram file, provides
    score_word(word, context) via Katz backoff.
  * Backward-compatible: all v0.5 flags continue to work unchanged.

Usage:
  # Stage 1 + merge/split only (safest):
  python3 syl_normalizer.py \\
      --dictionary syl_dict.txt --frequency 2 \\
      --input in.syl --output out.syl \\
      --log corrections.log --error-output errors.txt \\
      --fuzzy-distance 0

  # Add n-gram LM-guided fuzzy correction:
  python3 syl_normalizer.py \\
      --dictionary syl_dict.txt --frequency 2 \\
      --ngram-lm myMono_syl_trigram.arpa \\
      --min-lm-improve 0.5 \\
      --input in.syl --output out.syl \\
      --log corrections.log --error-output errors.txt

  # Fallback dict-based fuzzy (v0.5 behaviour):
  python3 syl_normalizer.py \\
      --dictionary syl_dict.txt --frequency 2 \\
      --fuzzy-distance 1 \\
      --input in.syl --output out.syl \\
      --log corrections.log --error-output errors.txt
"""

import re
import sys
import math
import argparse
import unicodedata
from typing import Dict, List, Optional, Tuple


# ===========================================================================
# Myanmar Unicode Constants
# ===========================================================================

VIRAMA = '\u1039'   # ္  Stacking virama (sa-lon)
ASAT   = '\u103A'   # ်  Asat (killer vowel)

MYANMAR_DIGITS       = set(chr(cp) for cp in range(0x1040, 0x104A))
COMBINING_MARKS_RANGE = (0x102B, 0x103E)


# ===========================================================================
# Confused Character Pairs
# ===========================================================================

# Codepoint quick-reference (Unicode Standard, Myanmar block U1000):
#   U+1000=က  U+1001=ခ  U+1002=ဂ  U+1003=ဃ  U+1004=င  U+1005=စ
#   U+1006=ဆ  U+1007=ဇ  U+1008=ဈ  U+1009=ဉ  U+100A=ည
#   U+1010=တ  U+1011=ထ  U+1012=ဒ  U+1013=ဓ  U+1014=န  U+1015=ပ
#   U+1016=ဖ  U+1017=ဗ  U+1018=ဘ  U+1019=မ  U+101A=ယ  U+101B=ရ
#   U+101C=လ  U+101D=ဝ  U+101E=သ  U+101F=ဟ  U+1020=ဠ  U+1021=အ
#   U+102B=ါ  U+102C=ာ  (combining vowel marks)
#
# Each tuple is (wrong_char, correct_char) as single-character str.
# substitute_confused_chars uses:  c == wrong_char  (str vs str, never int).
CONFUSED_PAIRS = [
    # TA-series (တ ထ ဒ ဓ): very common real-world confusion
    ('\u1010', '\u1013'),  # တ (ta)      -> ဓ (dha)  KEY: fixes ခန္တာ->ခန္ဓာ
    ('\u1013', '\u1010'),  # ဓ (dha)     -> တ (ta)
    ('\u1010', '\u1012'),  # တ (ta)      -> ဒ (da)
    ('\u1012', '\u1010'),  # ဒ (da)      -> တ (ta)
    ('\u1011', '\u1010'),  # ထ (tha)     -> တ (ta)
    ('\u1010', '\u1011'),  # တ (ta)      -> ထ (tha)
    ('\u1012', '\u1013'),  # ဒ (da)      -> ဓ (dha)
    ('\u1013', '\u1012'),  # ဓ (dha)     -> ဒ (da)
    # KA-series (က ခ ဂ): visually similar and keyboard-adjacent
    ('\u1000', '\u1001'),  # က (ka)      -> ခ (kha)
    ('\u1001', '\u1000'),  # ခ (kha)     -> က (ka)
    ('\u1001', '\u1002'),  # ခ (kha)     -> ဂ (ga)
    ('\u1002', '\u1001'),  # ဂ (ga)      -> ခ (kha)
    # PA-series (ပ ဖ): keyboard-adjacent
    ('\u1015', '\u1016'),  # ပ (pa)      -> ဖ (pha)
    ('\u1016', '\u1015'),  # ဖ (pha)     -> ပ (pa)
    # NYA confusion (ဉ ည): extremely common in real Myanmar text
    ('\u1009', '\u100A'),  # ဉ (nya)     -> ည (nnya)
    ('\u100A', '\u1009'),  # ည (nnya)    -> ဉ (nya)
    # NGA / CHA confusion (င ဆ ဈ)
    ('\u1004', '\u1006'),  # င (nga)     -> ဆ (cha)
    ('\u1006', '\u1004'),  # ဆ (cha)     -> င (nga)
    ('\u1008', '\u1006'),  # ဈ (jha)     -> ဆ (cha)
    ('\u1006', '\u1008'),  # ဆ (cha)     -> ဈ (jha)
    # YA-group (မ ယ ရ လ ဝ): shape and position confusion
    ('\u1019', '\u101A'),  # မ (ma)      -> ယ (ya)
    ('\u101A', '\u1019'),  # ယ (ya)      -> မ (ma)
    ('\u101A', '\u101B'),  # ယ (ya)      -> ရ (ra)
    ('\u101B', '\u101A'),  # ရ (ra)      -> ယ (ya)
    ('\u101B', '\u101C'),  # ရ (ra)      -> လ (la)
    ('\u101C', '\u101B'),  # လ (la)      -> ရ (ra)
    ('\u101B', '\u101D'),  # ရ (ra)      -> ဝ (wa)
    ('\u101D', '\u101B'),  # ဝ (wa)      -> ရ (ra)
    # SA/HA confusion (သ ဟ)
    ('\u101E', '\u101F'),  # သ (sa)      -> ဟ (ha)
    ('\u101F', '\u101E'),  # ဟ (ha)      -> သ (sa)
    # Vowel AA variants
    ('\u102C', '\u102B'),  # ာ (AA)      -> ါ (tall AA)
    ('\u102B', '\u102C'),  # ါ (tall AA) -> ာ (AA)
    # Myanmar digit 7 (၇) used as consonant ရ
    ('\u1047', '\u101B'),  # ၇ (digit 7) -> ရ (ra)
]


# ===========================================================================
# Utility Functions
# ===========================================================================

def _is_base_consonant(c: str) -> bool:
    cp = ord(c)
    return (0x1000 <= cp <= 0x1021) or cp == 0x103F


def _is_myanmar_char(c: str) -> bool:
    cp = ord(c)
    return (0x1000 <= cp <= 0x109F) or cp in (0x200B, 0x200C, 0xFE00, 0xFE01, 0xFEFF)


def _is_myanmar_number_token(token: str) -> bool:
    return bool(token) and all(c in MYANMAR_DIGITS for c in token)


def _is_non_myanmar_token(token: str) -> bool:
    return not token or not any(_is_myanmar_char(c) for c in token)


def _is_passthrough_token(token: str) -> bool:
    return _is_non_myanmar_token(token) or _is_myanmar_number_token(token)


# ===========================================================================
# Rule-Based Normalizer  (updated: per-rule description + detailed logging)
# ===========================================================================

class BurmeseRuleNormalizer:
    """
    Apply Myanmar-specific regex substitution rules iteratively.

    normalize_detailed() returns per-rule change records so the caller can
    write a fine-grained correction log showing exactly which rule fired.
    """

    # Each rule entry now has 'desc' for log readability.
    RULES = [
        {
            'id': '1',
            'desc': 'strip zero-width/invisible noise chars',
            'pattern': r'[\u200B\u200C\u202C\u00A0\u200D\u200A]',
            'repl': '',
        },
        {
            # Dot-below (U+1037) + Asat (U+103A) ordering fix.
            # Wrong typing: asat THEN dot  = U+103A U+1037
            # Correct encoding: dot THEN asat = U+1037 U+103A
            # Explicit Python \u escapes prevent editors from silently reordering.
            'id': '5',
            'desc': 'fix asat(်)+dot(့) order -> dot+asat',
            'pattern': '\u103A\u1037',      # WRONG order: asat then dot-below
            'repl':    '\u1037\u103A',      # CORRECT order: dot-below then asat
        },
        {
            # \1+ catches 2 or more consecutive identical diacritics.
            # (Previous version used \1{2,} which missed exactly-doubled marks.)
            'id': '15',
            'desc': 'collapse duplicate combining marks',
            'pattern': r'([\u102B-\u103E])\1+',
            'repl': r'\1',
        },
        {
            'id': '17',
            'desc': 'ဥ် -> ဉ် (wrong independent vowel)',
            'pattern': r'ဥ်',
            'repl': 'ဉ်',
        },
        {
            'id': '18',
            'desc': 'ဥာ -> ဉာ (wrong independent vowel)',
            'pattern': r'ဥာ',
            'repl': 'ဉာ',
        },
        {
            'id': '20',
            'desc': 'တကသိုလ် -> တက္ကသိုလ် (common misspelling)',
            'pattern': r'တကသိုလ်',
            'repl': 'တက္ကသိုလ်',
        },
        {
            # Rule 25 MUST precede Rule 24: Rule 24 converts သြ->ဩ first,
            # destroying the longer pattern that Rule 25 needs to match.
            'id': '25',
            'desc': 'သြော် -> ဪ (U+102A, AU independent vowel)',
            'pattern': r'သြော်',
            'repl': 'ဪ',
        },
        {
            'id': '24',
            'desc': 'သြ -> ဩ (U+1029, O independent vowel)',
            'pattern': r'သြ',
            'repl': 'ဩ',
        },
        {
            'id': '26',
            'desc': '၄င်း -> ၎င်း (digit 4 used as ၎)',
            'pattern': r'၄င်း',
            'repl': ' ၎င်း',
        },
        {
            'id': '28',
            'desc': 'စျ -> ဈ (two-consonant form to single char)',
            'pattern': r'စျ',
            'repl': 'ဈ',
        },
        {
            'id': '29',
            'desc': 'ဏာန်း -> ဏန်း (spurious AA vowel)',
            'pattern': r'ဏာန်း',
            'repl': 'ဏန်း',
        },
        {
            'id': '31',
            'desc': 'ဆဥ် -> ဆင် (wrong independent vowel)',
            'pattern': r'ဆဥ်',
            'repl': 'ဆင်',
        },
        {
            'id': '33',
            'desc': 'ုိး -> ိုး (vowel U+I reorder)',
            'pattern': r'ုိး',
            'repl': 'ိုး',
        },
        {
            'id': '34',
            'desc': 'ီုး -> ိုး (vowel II+U reorder)',
            'pattern': r'ီုး',
            'repl': 'ိုး',
        },
        {
            'id': '35',
            'desc': 'Cိး -> CီးCons (I->II before consonant)',
            'pattern': r'([\u1000-\u1021])ိး(?=[\u1000-\u1021\s])',
            'repl': r'\1ီး',
        },
        {
            'id': '37',
            'desc': '၀+diacritic -> ဝ+diacritic (digit 0 as wa consonant)',
            'pattern': r'၀([\u102B-\u103E])',
            'repl': r'ဝ\1',
        },
        {
            'id': '43',
            'desc': 'မ႓ာ -> မ္ဘာ့ (special sequence)',
            'pattern': r'မ႓ာ',
            'repl': r'မ္ဘာ့',
        },
    ]

    def __init__(self) -> None:
        self._compiled: List[Tuple[str, str, re.Pattern, str]] = [
            (r['id'], r['desc'], re.compile(r['pattern'], re.UNICODE), r['repl'])
            for r in self.RULES
        ]

    # ── public API ─────────────────────────────────────────────────────────

    def normalize(self, text: str, max_passes: int = 3) -> Tuple[str, int]:
        """Quick normalisation; returns (result, passes_taken)."""
        for i in range(max_passes):
            new_text = self._once(text)
            if new_text == text:
                return new_text, i + 1
            text = new_text
        return text, max_passes

    def normalize_detailed(
        self, text: str, max_passes: int = 3
    ) -> Tuple[str, int, List[Tuple[str, str, str, str]]]:
        """
        Full normalisation with per-rule change tracking.

        Returns
        -------
        (final_text, passes_taken, changes)
        where each element of changes is:
            (rule_id, rule_desc, text_before_rule, text_after_rule)
        """
        all_changes: List[Tuple[str, str, str, str]] = []
        for i in range(max_passes):
            new_text, pass_changes = self._once_detailed(text)
            all_changes.extend(pass_changes)
            if new_text == text:
                return new_text, i + 1, all_changes
            text = new_text
        return text, max_passes, all_changes

    # ── private helpers ────────────────────────────────────────────────────

    def _once(self, text: str) -> str:
        for _rid, _desc, pattern, repl in self._compiled:
            text = pattern.sub(repl, text)
        return text

    def _once_detailed(
        self, text: str
    ) -> Tuple[str, List[Tuple[str, str, str, str]]]:
        changes: List[Tuple[str, str, str, str]] = []
        for rule_id, desc, pattern, repl in self._compiled:
            new_text = pattern.sub(repl, text)
            if new_text != text:
                changes.append((rule_id, desc, text, new_text))
                text = new_text
        return text, changes


# ===========================================================================
# Syllable Checker  (unchanged from v0.5)
# ===========================================================================

class BurmeseSyllableChecker:
    """Validate Burmese syllables against a frequency dictionary."""

    def __init__(self, dictionary_file: str, min_frequency: int = 2) -> None:
        C   = r'[\u1000-\u1021]'
        M   = r'(\u103B|\u103C)?\u103D?\u103E?'
        V   = (r'\u1031?(?!\u102E\u1030|\u102E\u102F)'
               r'(\u102B|\u102C)?(\u102D\u102F|\u102D|\u102E|\u1032)?'
               r'(\u102F|\u1030)?')
        F   = r'[\u1036\u1037]?\u1038?'
        A   = r'\u103A'
        S   = r'\u1039'
        G   = r'\u103B'
        IVS = r'([\u1004])?'
        SUB = r'((?:{C}{S}{C})+)'.format(C=C, S=S)
        KZ  = r'\u1019\u1037'

        self.syllable_regex = re.compile(
            f"^("
            f"{IVS}?{C}{M}?{V}?{F}?{A}?|"
            f"{IVS}?{SUB}{V}?{F}?{A}?|"
            f"{G}{V}?{F}?|"
            f"{KZ}{C}{S}{C}{V}?{F}?{A}?|"
            f"{C}{S}{C}{V}{F}?{A}?|"
            f"{C}{M}{V}{F}?{C}{A}\u1038?|"
            f"{C}{SUB}*{V}{C}{A}{F}?|"
            f"{C}{KZ}{C}{SUB}{V}?{F}?|"
            f"[\u1040-\u1049]+|"
            f"[0-9]+|"
            f"[\u104A\u104B!?,.<>()\"'\\-]+|"
            f"[a-zA-Z]+"
            f")$",
            re.UNICODE,
        )

        self.dictionary: Dict[str, int] = {}
        with open(dictionary_file, 'r', encoding='utf-8-sig') as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) == 2:
                    syllable, freq_str = parts
                    syllable = unicodedata.normalize('NFC', syllable)
                    freq = int(freq_str)
                    if freq >= min_frequency:
                        self.dictionary[syllable] = freq

    def in_dictionary(self, syllable: str) -> bool:
        return unicodedata.normalize('NFC', syllable) in self.dictionary

    def matches_re(self, syllable: str) -> bool:
        return bool(self.syllable_regex.fullmatch(syllable))

    def is_valid(self, syllable: str, mode: str = 'dictionary') -> bool:
        if mode == 'RE_and_dictionary':
            return self.matches_re(syllable) and self.in_dictionary(syllable)
        return self.in_dictionary(syllable)


# ===========================================================================
# N-gram Language Model  (NEW in v0.6)
# ===========================================================================

class NgramLanguageModel:
    """
    ARPA format n-gram language model loader with Katz backoff scoring.

    Supports up to trigrams.  Used by NgramFuzzyCorrector to rank correction
    candidates by probability in context rather than by simple dictionary
    membership.

    ARPA format (one entry per line, fields whitespace-separated):
        log10prob  word              [backoff]     <- 1-gram
        log10prob  word1 word2       [backoff]     <- 2-gram
        log10prob  word1 word2 word3               <- 3-gram  (no backoff)

    Special tokens: <unk>, <s> (begin-of-sentence), </s> (end-of-sentence)
    """

    UNK = '<unk>'
    BOS = '<s>'
    EOS = '</s>'

    # Fallback probability used for any token that is completely absent from
    # the unigram table.  -99 is the SRILM convention for "zero probability".
    _DEFAULT_UNK_LOGPROB = -99.0

    def __init__(self, arpa_file: str) -> None:
        # unigrams[word]      = (log10_prob, log10_backoff)
        self.unigrams: Dict[str, Tuple[float, float]] = {}
        # bigrams['w1 w2']    = (log10_prob, log10_backoff)
        self.bigrams:  Dict[str, Tuple[float, float]] = {}
        # trigrams['w1 w2 w3']= log10_prob   (highest order → no backoff)
        self.trigrams: Dict[str, float]               = {}
        self.max_order: int = 1
        self.unk_logprob: float = self._DEFAULT_UNK_LOGPROB

        print(f"  Loading n-gram LM: {arpa_file}", file=sys.stderr)
        self._load_arpa(arpa_file)
        # Cache the <unk> probability for fast OOV scoring
        if self.UNK in self.unigrams:
            self.unk_logprob = self.unigrams[self.UNK][0]
        print(
            f"  LM loaded: {len(self.unigrams):,} unigrams, "
            f"{len(self.bigrams):,} bigrams, "
            f"{len(self.trigrams):,} trigrams  "
            f"(unk={self.unk_logprob:.3f})",
            file=sys.stderr,
        )

    # ── ARPA loader ────────────────────────────────────────────────────────

    def _load_arpa(self, path: str) -> None:
        current_order = 0
        with open(path, 'r', encoding='utf-8') as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                # Section headers
                # FIX: Changed r'\data\' to '\\data\\' (and same for \end\)
                if line == '\\data\\' or line == '\\end\\':
                    continue
                if line.startswith('ngram '):
                    # e.g. "ngram 3=6612522"
                    try:
                        order = int(line.split('=')[0].split()[1])
                        self.max_order = max(self.max_order, order)
                    except (IndexError, ValueError):
                        pass
                    continue
                # n-gram section marker: \1-grams: \2-grams: \3-grams:
                # (This regex is fine because the backslash is inside the regex pattern)
                m = re.match(r'\\(\d+)-grams:', line)
                if m:
                    current_order = int(m.group(1))
                    continue
                if current_order == 0:
                    continue

                # Parse n-gram entry
                parts = line.split()
                # Need at least: log_prob + n words
                if len(parts) < current_order + 1:
                    continue
                try:
                    log_prob = float(parts[0])
                except ValueError:
                    continue

                words = parts[1: current_order + 1]   # the n words
                # Optional backoff weight (present for orders < max_order)
                backoff = 0.0
                if len(parts) > current_order + 1:
                    try:
                        backoff = float(parts[current_order + 1])
                    except ValueError:
                        backoff = 0.0

                key = ' '.join(words)   # space-joined; Myanmar words have no spaces
                if current_order == 1:
                    self.unigrams[key] = (log_prob, backoff)
                elif current_order == 2:
                    self.bigrams[key] = (log_prob, backoff)
                elif current_order == 3:
                    self.trigrams[key] = log_prob

    # ── Scoring  ───────────────────────────────────────────────────────────

    def score_word(self, word: str, context: List[str]) -> float:
        """
        Return log10 P(word | context) using Katz backoff.

        context : list of preceding words, most-recent last.
                  Caller passes out_tokens[-2:] so context length is 0-2.

        The returned value is always ≤ 0 (log10 of a probability).
        Lower (more negative) = less likely.
        """
        w = word
        ctx = context  # e.g. [prev2, prev1]

        # ── Try trigram P(w | prev2, prev1) ──────────────────────────────
        if len(ctx) >= 2 and self.max_order >= 3:
            key3 = f"{ctx[-2]} {ctx[-1]} {w}"
            if key3 in self.trigrams:
                return self.trigrams[key3]
            # Not found → apply bigram backoff and fall through to bigram
            bow2_key = f"{ctx[-2]} {ctx[-1]}"
            bow2 = self.bigrams.get(bow2_key, (0.0, 0.0))[1]  # log backoff
            return bow2 + self._score_bigram(w, ctx[-1])

        # ── Try bigram P(w | prev1) ──────────────────────────────────────
        if len(ctx) >= 1:
            return self._score_bigram(w, ctx[-1])

        # ── Unigram only ─────────────────────────────────────────────────
        return self._score_unigram(w)

    def _score_bigram(self, word: str, prev: str) -> float:
        key2 = f"{prev} {word}"
        if key2 in self.bigrams:
            return self.bigrams[key2][0]
        # Backoff: bow(prev) + P(word)
        bow1 = self.unigrams.get(prev, (0.0, 0.0))[1]
        return bow1 + self._score_unigram(word)

    def _score_unigram(self, word: str) -> float:
        if word in self.unigrams:
            return self.unigrams[word][0]
        return self.unk_logprob


# ===========================================================================
# N-gram-based Fuzzy Corrector  (NEW in v0.6)
# ===========================================================================

class NgramFuzzyCorrector:
    """
    Fuzzy corrector that uses an n-gram language model for syllable-level
    confirmation and substitution.

    For each unknown token the corrector:
      1. Generates correction candidates (same four methods as
         TargetedFuzzyCorrector: dup-vowel removal, asat removal,
         confused-char substitution, medial ordering fix).
      2. FILTERS candidates to keep ONLY those that exist in the syllable
         dictionary. (This prevents blind character swaps on OOV syllables
         based purely on backoff noise).
      3. Scores the ORIGINAL token and remaining valid candidates with the
         LM in the current left-context window.
      4. Accepts the highest-scoring candidate ONLY IF it beats the original
         token's score by at least `min_improve` log10 units.
    """

    def __init__(
        self,
        lm: NgramLanguageModel,
        dictionary: Dict[str, int],
        min_improve: float = 0.5,
    ) -> None:
        self.lm = lm
        self.dictionary = dictionary
        self.min_improve = min_improve
        # Cache keyed by (token, context_tuple) to avoid repeated LM lookups
        self._cache: Dict[Tuple[str, ...], Optional[Tuple[str, str]]] = {}

    # ── Candidate generators (same logic as TargetedFuzzyCorrector) ────────

    def _gen_dup_vowel(self, token: str) -> List[str]:
        chars = list(token)
        lo, hi = COMBINING_MARKS_RANGE
        for i in range(len(chars) - 1):
            if chars[i] == chars[i + 1] and lo <= ord(chars[i]) <= hi:
                return [''.join(chars[:i] + chars[i + 1:])]
        return []

    def _gen_asat_removal(self, token: str) -> List[str]:
        results, tone_marks = [], {'\u1037', '\u1038'}
        idx = token.find(ASAT)
        while idx != -1:
            if idx + 1 < len(token) and token[idx + 1] in tone_marks:
                results.append(token[:idx] + token[idx + 1:])
            idx = token.find(ASAT, idx + 1)
        return results

    def _gen_char_sub(self, token: str) -> List[str]:
        candidates, seen = [], set()
        chars = list(token)
        for i, c in enumerate(chars):
            for wrong_char, correct_char in CONFUSED_PAIRS:
                if c == wrong_char:
                    cand = ''.join(chars[:i] + [correct_char] + chars[i + 1:])
                    if cand not in seen:
                        candidates.append(cand)
                        seen.add(cand)
        return candidates

    def _gen_medial(self, token: str) -> List[str]:
        candidates = []
        for medial in ('\u103B', '\u103C', '\u103D'):
            idx = token.find(medial)
            if idx != -1 and idx + 1 < len(token):
                nxt = token[idx + 1]
                ncp = ord(nxt)
                if (0x1000 <= ncp <= 0x1021) or ncp == 0x103F:
                    cand = token.replace(medial + nxt, nxt + medial, 1)
                    if cand != token:
                        candidates.append(cand)
        return candidates

    # ── Main correction entry point ─────────────────────────────────────────

    def correct(
        self, token: str, context: List[str]
    ) -> Optional[Tuple[str, str]]:
        """
        Return (corrected_token, method_name) or None.
        """
        cache_key = (token, *context[-2:])
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Score the original token in context
        orig_score = self.lm.score_word(
            unicodedata.normalize('NFC', token), context
        )

        # Collect all (candidate, method) pairs
        all_candidates: List[Tuple[str, str]] = []
        for c in self._gen_dup_vowel(token):
            all_candidates.append((c, 'ngram_dup_vowel'))
        for c in self._gen_asat_removal(token):
            all_candidates.append((c, 'ngram_asat_removal'))
        for c in self._gen_char_sub(token):
            all_candidates.append((c, 'ngram_char_sub'))
        for c in self._gen_medial(token):
            all_candidates.append((c, 'ngram_medial'))
        # Combined: dup-vowel then asat removal
        for dc in self._gen_dup_vowel(token):
            for ac in self._gen_asat_removal(dc):
                all_candidates.append((ac, 'ngram_dup_plus_asat'))

        # Score every candidate; keep the best that beats orig by min_improve
        best_result: Optional[Tuple[str, str]] = None
        best_score  = orig_score + self.min_improve  # threshold

        for cand, method in all_candidates:
            cand_nfc = unicodedata.normalize('NFC', cand)
            
            # ==============================================================
            # CRITICAL FIX: Syllable-level confirmation
            # Only evaluate candidates that are VALID syllables in the
            # dictionary. The LM is for contextual confirmation between
            # valid syllables, NOT for inventing new syllables via blind
            # character swaps on OOV tokens.
            # ==============================================================
            if cand_nfc not in self.dictionary:
                continue
                
            score = self.lm.score_word(cand_nfc, context)
            if score > best_score:
                best_score  = score
                best_result = (cand_nfc, method)

        self._cache[cache_key] = best_result
        return best_result

# ===========================================================================
# Dictionary-based Fuzzy Corrector  (unchanged from v0.5, kept for compat)
# ===========================================================================

class TargetedFuzzyCorrector:
    """
    Targeted fuzzy correction against the syllable frequency dictionary.

    Used when --ngram-lm is NOT provided and --fuzzy-distance 1 is set.
    Preserved for backward compatibility with v0.5 behaviour.
    """

    def __init__(self, dictionary: Dict[str, int]) -> None:
        self.dictionary = dictionary
        self._cache: Dict[str, Optional[Tuple[str, str]]] = {}

    def _in_dict(self, candidate: str) -> bool:
        return unicodedata.normalize('NFC', candidate) in self.dictionary

    def _dup_vowel(self, token: str) -> List[str]:
        chars = list(token)
        lo, hi = COMBINING_MARKS_RANGE
        for i in range(len(chars) - 1):
            if chars[i] == chars[i + 1] and lo <= ord(chars[i]) <= hi:
                return [''.join(chars[:i] + chars[i + 1:])]
        return []

    def _asat_removal(self, token: str) -> List[str]:
        results, tone_marks = [], {'\u1037', '\u1038'}
        idx = token.find(ASAT)
        while idx != -1:
            if idx + 1 < len(token) and token[idx + 1] in tone_marks:
                results.append(token[:idx] + token[idx + 1:])
            idx = token.find(ASAT, idx + 1)
        return results

    def _char_sub(self, token: str) -> List[str]:
        candidates, seen = [], set()
        chars = list(token)
        for i, c in enumerate(chars):
            for wrong_char, correct_char in CONFUSED_PAIRS:
                if c == wrong_char:
                    cand = ''.join(chars[:i] + [correct_char] + chars[i + 1:])
                    if cand not in seen:
                        candidates.append(cand)
                        seen.add(cand)
        return candidates

    def _medial(self, token: str) -> List[str]:
        candidates = []
        for medial in ('\u103B', '\u103C', '\u103D'):
            idx = token.find(medial)
            if idx != -1 and idx + 1 < len(token):
                nxt = token[idx + 1]
                ncp = ord(nxt)
                if (0x1000 <= ncp <= 0x1021) or ncp == 0x103F:
                    cand = token.replace(medial + nxt, nxt + medial, 1)
                    if cand != token:
                        candidates.append(cand)
        return candidates

    def correct(self, token: str) -> Optional[Tuple[str, str]]:
        if token in self._cache:
            return self._cache[token]

        result = None
        for cand in self._dup_vowel(token):
            if self._in_dict(cand):
                result = (cand, 'duplicate_vowel_removal'); break
        if result is None:
            for cand in self._asat_removal(token):
                if self._in_dict(cand):
                    result = (cand, 'asat_removal'); break
        if result is None:
            for cand in self._char_sub(token):
                if self._in_dict(cand):
                    result = (cand, 'character_substitution'); break
        if result is None:
            for cand in self._medial(token):
                if self._in_dict(cand):
                    result = (cand, 'medial_ordering'); break
        if result is None:
            for dc in self._dup_vowel(token):
                for ac in self._asat_removal(dc):
                    if self._in_dict(ac):
                        result = (ac, 'duplicate_vowel_plus_asat_removal')
                        break
                if result:
                    break

        self._cache[token] = result
        return result


# ===========================================================================
# Consonant+Asat Merger  (unchanged from v0.5)
# ===========================================================================

def is_consonant_with_athat(token: str) -> bool:
    if not token:
        return False
    first_cp = ord(token[0])
    if not ((0x1000 <= first_cp <= 0x1021) or first_cp == 0x103F
            or first_cp == 0x1039):
        return False
    asat_pos = token.find(ASAT)
    if asat_pos == -1:
        return False
    for c in token[asat_pos + 1:]:
        if ord(c) not in (0x1037, 0x1038):
            return False
    return True


def try_merge_with_previous(
    current_token: str,
    previous_token: str,
    dictionary: Dict[str, int],
) -> Optional[str]:
    if not is_consonant_with_athat(current_token):
        return None
    merged = unicodedata.normalize('NFC', previous_token + current_token)
    return merged if merged in dictionary else None


# ===========================================================================
# Compound-Split Recovery  (unchanged from v0.5)
# ===========================================================================

def compound_split(
    token: str,
    dictionary: Dict[str, int],
    max_parts: int = 3,
) -> Optional[List[str]]:
    n = len(token)
    dp: List[Optional[List[str]]] = [None] * (n + 1)
    dp[0] = []
    for i in range(1, n + 1):
        for j in range(max(0, i - 12), i):
            if dp[j] is None or len(dp[j]) >= max_parts:
                continue
            part_nfc = unicodedata.normalize('NFC', token[j:i])
            if part_nfc in dictionary:
                cand = dp[j] + [part_nfc]
                if dp[i] is None or len(cand) < len(dp[i]):
                    dp[i] = cand
    result = dp[n]
    return result if (result is not None and len(result) >= 2) else None


# ===========================================================================
# Main Processing Pipeline
# ===========================================================================

def process_input(
    input_stream,
    output_file:        Optional[str]  = None,
    error_output_file:  Optional[str]  = 'detected_syllable_errors.txt',
    verbose:            bool           = False,
    dictionary_file:    Optional[str]  = None,
    min_frequency:      int            = 2,
    check:              str            = 'dictionary',
    fuzzy_distance:     int            = 0,
    ngram_lm_file:      Optional[str]  = None,
    min_lm_improve:     float          = 0.5,
    log_file:           Optional[str]  = None,
    debug_fuzzy:        bool           = False,
) -> None:
    """
    Full normalisation pipeline.

    Token processing order
    ----------------------
    0. Passthrough (Latin / Myanmar digit-only tokens) → copy unchanged.
    1. Unicode NFC normalization.
    2. Stage 1: iterative regex rules; log each rule that fires.
    3. Dictionary validity check → done if valid.
    4. Stage 2: fuzzy correction (n-gram LM if --ngram-lm, else syllable-dict
       if --fuzzy-distance 1, else skipped).
    5. Stage 3: consonant+asat merge with previous output token.
    6. Stage 4: compound split (DP, ≤ 3 parts, len ≥ 4).
    7. Unknown: output as-is; counted in stats.
    """

    # ── Initialise components ───────────────────────────────────────────────
    checker   = BurmeseSyllableChecker(dictionary_file, min_frequency)
    rule_norm = BurmeseRuleNormalizer()

    # Determine which fuzzy corrector to use
    ngram_corrector:  Optional[NgramFuzzyCorrector]   = None
    target_corrector: Optional[TargetedFuzzyCorrector] = None

    if ngram_lm_file:
        lm = NgramLanguageModel(ngram_lm_file)
        ngram_corrector = NgramFuzzyCorrector(lm, checker.dictionary, min_improve=min_lm_improve)
        print(
            f"  Fuzzy correction: n-gram LM mode  "
            f"(min_improve={min_lm_improve:.2f} log10 units)",
            file=sys.stderr,
        )
    elif fuzzy_distance > 0:
        target_corrector = TargetedFuzzyCorrector(checker.dictionary)
        print(
            "  Fuzzy correction: syllable-dictionary mode (v0.5 compatible)",
            file=sys.stderr,
        )
    else:
        print("  Fuzzy correction: disabled", file=sys.stderr)

    # ── Stats counters ──────────────────────────────────────────────────────
    stats: Dict[str, int] = {
        'lines':                    0,
        'tokens':                   0,
        'passthrough':              0,
        'already_valid':            0,
        'fixed_stage1':             0,   # tokens changed by rules
        'fixed_stage1_rule_apps':   0,   # individual rule applications
        'fixed_ngram_dup_vowel':    0,
        'fixed_ngram_asat':         0,
        'fixed_ngram_char_sub':     0,
        'fixed_ngram_medial':       0,
        'fixed_ngram_combined':     0,
        'fixed_fuzzy_dup_vowel':    0,   # dict-based (v0.5 compat)
        'fixed_fuzzy_asat':         0,
        'fixed_fuzzy_char_sub':     0,
        'fixed_fuzzy_medial':       0,
        'fixed_fuzzy_combined':     0,
        'fixed_merge':              0,
        'fixed_split':              0,
        'still_unknown':            0,
    }

    log_entries:        List[str] = []
    error_output_lines: List[str] = []

    def _log(line_no: int, original: str, corrected: str, stage: str) -> None:
        if log_file or verbose:
            log_entries.append(
                f"line {line_no:>6} | {original!r:30s}"
                f" -> {corrected!r:30s} | {stage}"
            )

    output_lines: List[str] = []

    # ── Main loop ───────────────────────────────────────────────────────────
    for raw_line in input_stream:
        line = raw_line.rstrip('\n')
        stats['lines'] += 1
        line_no = stats['lines']

        tokens     = line.split()
        out_tokens: List[str] = []
        error_tokens: List[str] = []

        for tok in tokens:
            stats['tokens'] += 1

            # ── Stage 0: passthrough ──────────────────────────────────────
            if _is_passthrough_token(tok):
                stats['passthrough'] += 1
                out_tokens.append(tok)
                error_tokens.append(tok)
                continue

            # ── Stage 0b: NFC normalisation ───────────────────────────────
            tok_nfc = unicodedata.normalize('NFC', tok)

            # ── Stage 1: regex rules with per-rule logging ─────────────────
            tok_norm, _passes, rule_changes = rule_norm.normalize_detailed(tok_nfc)

            # Log every individual rule that fired (not just "rules applied")
            for rule_id, rule_desc, before, after in rule_changes:
                _log(line_no, before, after,
                     f'stage1:rule{rule_id} ({rule_desc})')
                stats['fixed_stage1_rule_apps'] += 1

            if tok_norm != tok_nfc:
                stats['fixed_stage1'] += 1
                tok = tok_norm
            else:
                tok = tok_nfc   # ensure NFC even if no rules fired

            # ── Dictionary validity check ─────────────────────────────────
            if checker.is_valid(tok, mode=check):
                stats['already_valid'] += 1
                out_tokens.append(tok)
                error_tokens.append(tok)
                continue

            # Mark as error BEFORE attempting correction
            error_tokens.append(f'<{tok}>')

            # ── Stage 2a: N-gram LM fuzzy correction ──────────────────────
            if ngram_corrector is not None:
                context = out_tokens[-2:]   # up to 2 preceding output tokens
                ng_result = ngram_corrector.correct(tok, context)
                if ng_result is not None:
                    corrected, method = ng_result
                    _log(line_no, tok, corrected, f'stage2:ngram({method})')
                    if   'dup_plus_asat' in method: stats['fixed_ngram_combined'] += 1
                    elif 'dup_vowel'     in method: stats['fixed_ngram_dup_vowel'] += 1
                    elif 'asat'          in method: stats['fixed_ngram_asat'] += 1
                    elif 'char_sub'      in method: stats['fixed_ngram_char_sub'] += 1
                    elif 'medial'        in method: stats['fixed_ngram_medial'] += 1
                    out_tokens.append(corrected)
                    continue

            # ── Stage 2b: Dict-based fuzzy correction (v0.5 compat) ───────
            if target_corrector is not None:
                t_result = target_corrector.correct(tok)
                if t_result is not None:
                    corrected, method = t_result
                    _log(line_no, tok, corrected, f'stage2:fuzzy({method})')
                    if   'plus_asat'         in method: stats['fixed_fuzzy_combined'] += 1
                    elif 'duplicate'         in method: stats['fixed_fuzzy_dup_vowel'] += 1
                    elif 'asat'              in method: stats['fixed_fuzzy_asat'] += 1
                    elif 'character_sub'     in method: stats['fixed_fuzzy_char_sub'] += 1
                    elif 'medial'            in method: stats['fixed_fuzzy_medial'] += 1
                    out_tokens.append(corrected)
                    continue

            # ── Stage 3: consonant+asat merge with previous token ──────────
            if out_tokens and is_consonant_with_athat(tok):
                prev   = out_tokens[-1]
                merged = try_merge_with_previous(tok, prev, checker.dictionary)
                if merged:
                    _log(line_no, f"{prev} {tok}", merged,
                         'stage3:merge_with_previous')
                    stats['fixed_merge'] += 1
                    out_tokens[-1] = merged
                    continue

            # ── Stage 4: compound split ────────────────────────────────────
            if len(tok) >= 4:
                parts = compound_split(tok, checker.dictionary)
                if parts is not None:
                    _log(line_no, tok, ' '.join(parts),
                         f'stage4:split({len(parts)} parts)')
                    stats['fixed_split'] += 1
                    out_tokens.extend(parts)
                    continue

            # ── Still unknown ──────────────────────────────────────────────
            stats['still_unknown'] += 1
            out_tokens.append(tok)

        output_lines.append(' '.join(out_tokens))
        error_output_lines.append(' '.join(error_tokens))

    # ── Write outputs ───────────────────────────────────────────────────────
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as fh:
            for line in output_lines:
                fh.write(line.strip() + '\n')
    else:
        for line in output_lines:
            print(line.strip())

    if error_output_file:
        with open(error_output_file, 'w', encoding='utf-8') as fh:
            for line in error_output_lines:
                fh.write(line.strip() + '\n')

    if log_file and log_entries:
        with open(log_file, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(log_entries) + '\n')

    # ── Summary statistics ──────────────────────────────────────────────────
    total_ngram  = (stats['fixed_ngram_dup_vowel'] + stats['fixed_ngram_asat']
                    + stats['fixed_ngram_char_sub'] + stats['fixed_ngram_medial']
                    + stats['fixed_ngram_combined'])
    total_fuzzy  = (stats['fixed_fuzzy_dup_vowel'] + stats['fixed_fuzzy_asat']
                    + stats['fixed_fuzzy_char_sub'] + stats['fixed_fuzzy_medial']
                    + stats['fixed_fuzzy_combined'])
    total_fixed  = (stats['fixed_stage1'] + total_ngram + total_fuzzy
                    + stats['fixed_merge'] + stats['fixed_split'])
    total_tokens = max(1, stats['tokens'])

    print(
        f"\n=== syl_normalizer summary (v0.6) ===\n"
        f"  Lines processed    : {stats['lines']:>10,}\n"
        f"  Tokens processed   : {stats['tokens']:>10,}\n"
        f"  Passthrough        : {stats['passthrough']:>10,}\n"
        f"  Already valid      : {stats['already_valid']:>10,}\n"
        f"\n"
        f"  Fixed - stage 1 (rules, token count)    : {stats['fixed_stage1']:>8,}\n"
        f"  Fixed - stage 1 (individual rule apps)  : {stats['fixed_stage1_rule_apps']:>8,}\n"
        f"\n"
        f"  Fixed - stage 2 ngram (dup vowel)       : {stats['fixed_ngram_dup_vowel']:>8,}\n"
        f"  Fixed - stage 2 ngram (asat removal)    : {stats['fixed_ngram_asat']:>8,}\n"
        f"  Fixed - stage 2 ngram (char sub)        : {stats['fixed_ngram_char_sub']:>8,}\n"
        f"  Fixed - stage 2 ngram (medial order)    : {stats['fixed_ngram_medial']:>8,}\n"
        f"  Fixed - stage 2 ngram (dup+asat)        : {stats['fixed_ngram_combined']:>8,}\n"
        f"\n"
        f"  Fixed - stage 2 dict (dup vowel)        : {stats['fixed_fuzzy_dup_vowel']:>8,}\n"
        f"  Fixed - stage 2 dict (asat removal)     : {stats['fixed_fuzzy_asat']:>8,}\n"
        f"  Fixed - stage 2 dict (char sub)         : {stats['fixed_fuzzy_char_sub']:>8,}\n"
        f"  Fixed - stage 2 dict (medial order)     : {stats['fixed_fuzzy_medial']:>8,}\n"
        f"  Fixed - stage 2 dict (dup+asat)         : {stats['fixed_fuzzy_combined']:>8,}\n"
        f"\n"
        f"  Fixed - stage 3 (merge with previous)   : {stats['fixed_merge']:>8,}\n"
        f"  Fixed - stage 4 (compound split)        : {stats['fixed_split']:>8,}\n"
        f"\n"
        f"  Total fixed        : {total_fixed:>10,}\n"
        f"  Still unknown      : {stats['still_unknown']:>10,}\n"
        f"  Unknown rate       : {stats['still_unknown']/total_tokens:>9.2%}\n",
        file=sys.stderr,
    )


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Burmese Syllable Normalizer v0.6 — '
            'Normalizes Myanmar text using rules, n-gram LM fuzzy correction, '
            'consonant+asat merging, and compound splitting.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── I/O arguments ────────────────────────────────────────────────────────
    parser.add_argument('--input',  '-i',
        help='Input file (space-separated syllables). Reads from stdin if omitted.')
    parser.add_argument('--output', '-o',
        help='Output file with corrected text. Prints to stdout if omitted.')
    parser.add_argument('--log',    '-l',
        help='File to write detailed per-correction log (every rule + every fix).')
    parser.add_argument('--error-output', '-e',
        default='detected_syllable_errors.txt',
        help='File to write detected errors with <markers> BEFORE correction.')
    parser.add_argument('--verbose', '-v', action='store_true',
        help='Also log unknown tokens to the correction log.')

    # ── Dictionary arguments ─────────────────────────────────────────────────
    parser.add_argument('--dictionary', '-d', required=True,
        help='Syllable dictionary (format: syllable<SPACE>frequency, one per line).')
    parser.add_argument('--frequency', '-f', type=int, default=2, metavar='N',
        help='Minimum frequency for a dictionary syllable to be considered valid '
             '(default: 2).')
    parser.add_argument('--check', '-c',
        choices=['dictionary', 'RE_and_dictionary'], default='dictionary',
        help='Syllable validation mode (default: dictionary).')

    # ── Fuzzy correction arguments ────────────────────────────────────────────
    parser.add_argument('--fuzzy-distance', '-z', type=int, default=0, metavar='N',
        dest='fuzzy_distance',
        help='Enable syllable-dictionary fuzzy correction (0=disabled, 1=enabled). '
             'Ignored when --ngram-lm is provided.')
    parser.add_argument('--ngram-lm', metavar='ARPA_FILE',
        dest='ngram_lm_file', default=None,
        help='ARPA format n-gram language model for smarter fuzzy correction. '
             'When provided, overrides --fuzzy-distance and uses LM-based scoring.')
    parser.add_argument('--min-lm-improve', type=float, default=0.5,
        dest='min_lm_improve', metavar='DELTA',
        help='Minimum log10 probability improvement required for an n-gram '
             'correction to be accepted (default: 0.5). Higher = more conservative.')

    # ── Debug arguments ──────────────────────────────────────────────────────
    parser.add_argument('--debug-fuzzy', action='store_true',
        help='Print verbose fuzzy correction debug output to stderr.')

    args = parser.parse_args()

    input_stream = (
        open(args.input, 'r', encoding='utf-8') if args.input else sys.stdin
    )

    try:
        process_input(
            input_stream      = input_stream,
            output_file       = args.output,
            error_output_file = args.error_output if args.error_output else None,
            verbose           = args.verbose,
            dictionary_file   = args.dictionary,
            min_frequency     = args.frequency,
            check             = args.check,
            fuzzy_distance    = args.fuzzy_distance,
            ngram_lm_file     = args.ngram_lm_file,
            min_lm_improve    = args.min_lm_improve,
            log_file          = args.log,
            debug_fuzzy       = args.debug_fuzzy,
        )
    finally:
        if args.input:
            input_stream.close()


if __name__ == '__main__':
    main()

