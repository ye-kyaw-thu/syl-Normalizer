# Myanmar Syllable Normalizer (v0.6)

A multi-stage pipeline tool for normalizing Myanmar text using rule-based corrections, n-gram language model-guided fuzzy correction, and compound syllable recovery. This tool is designed to improve the quality of raw Myanmar text for downstream NLP tasks such as POS tagging, NER, and Machine Translation.

**Author:** Ye Kyaw Thu  
**Affiliation:** Language Understanding Lab (LU Lab), Myanmar  
**Status:** Version 0.6 (Active Development)  

---

## 🛠 Syllable Normalization Pipeline

The normalizer operates through a 5-stage process to ensure linguistic accuracy and encoding consistency.

1. **Stage 0: Unicode Normalization** – Converts text to Unicode NFC (Normalization Form C).
2. **Stage 1: Regex-based Rules** – Iterative corrections for medial ordering, vowel fixes, and common keyboarding errors.
3. **Stage 2: Fuzzy Correction** – Corrects syllables by checking against a frequency dictionary or scoring candidates using an ARPA n-gram language model.
4. **Stage 3: Merge** – Recombines dangling Consonant+Asat sequences with their preceding syllables.
5. **Stage 4: Compound Splitting** – Uses Dynamic Programming to split compound syllables into valid sub-parts (up to 3 parts).

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ye-kyaw-thu/syl-Normalizer
cd syl-Normalizer/ver_0.6

```

### 2. Rebuild the n-gram Language Model

Due to GitHub file size limits, the 301MB ARPA language model is stored in chunks within the `lm_chunks/` directory. **You must rebuild the file before using the n-gram fuzzy correction feature.**

```bash
chmod +x rebuild_lm.sh
./rebuild_lm.sh

```

This will merge the 13 parts and verify the integrity using a SHA256 checksum to produce `myMono_syl_trigram.arpa`.

---

## 📖 Usage

Navigate to the `ver_0.6` directory to run the following commands.

### Option A: Standard Normalization (Recommended/Safest)

This mode uses Stage 1 rules, merging, and splitting. It is the most conservative and reliable for general text.

```bash
python3 syl_normalizer.py \
    --dictionary final_syl_dictionary_13Feb2024.sorted.txt \
    --frequency 2 \
    --input test.my \
    --output out.syl \
    --log corrections.log \
    --error-output errors.txt \
    --fuzzy-distance 0

```

### Option B: n-gram LM-Guided Normalization

Uses the Katz backoff trigram model to provide context-aware corrections. Use this if your text contains many visual typos that require probabilistic confirmation.

```bash
python3 syl_normalizer.py \
    --dictionary final_syl_dictionary_13Feb2024.sorted.txt \
    --frequency 2 \
    --ngram-lm myMono_syl_trigram.arpa \
    --min-lm-improve 0.5 \
    --input test.my \
    --output out.syl \
    --log corrections.log \
    --error-output errors.txt

```

---

## 📂 Repository Structure

* `syl_normalizer.py`: The main Python implementation.
* `final_syl_dictionary_13Feb2024.sorted.txt`: The reference syllable frequency dictionary.
* `lm_chunks/`: Segmented ARPA language model files.
* `notebook/`: Contains `Syllable_Normalizer.ipynb` and a PDF export for a detailed walkthrough and demonstration.
* `overview/`: Pipeline diagrams and documentation assets.
* `test.my`: Sample input data for testing.

---

## 📑 Citation

If you use this tool or the provided datasets in your research, please cite it as follows:

```bibtex
@misc{syl_normalizer,
  author       = {Ye Kyaw Thu},
  title        = {{Syllable Normalization Tool for Myanmar Language}},
  version      = {0.6},
  month        = {May},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/ye-kyaw-thu/syl-Normalizer/tree/main/ver_0.6},
  note         = {Accessed: YYYY-MM-DD},
  institution  = {Language Understanding Lab (LU Lab), Myanmar}
}
```

