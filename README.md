# Thesis: PII Detection and Anonymisation

Synthetic document generation, defender (detection + redaction), attacker (re-identification), and evaluation for RQ1–RQ4.

## Setup

```bash
pip install -r requirements.txt
```

Optional: install LibreOffice for DOCX→PDF conversion.

## Generate a synthetic dataset

```bash
python -m document_generator.generate --config path/to/config.json
```

## Run the defender on documents

```bash
python -m thesis_project.defender path/to/file_or_folder --output-root out/
```

Useful flags: `--redact-quasi`, `--no-pdf`, `--ner-model NAME`, `--pdf-ocr-all-pages`.

### Output guarantees

The defender writes a brand new document to `<output-root>/<stem>_anonymized.<ext>` the source file is never modified.
Redacted content is actually removed, not just visually covered:
- **DOCX text**: paragraph runs are rewritten in-place via `replace_paragraph_text`, so the masked tokens replace the original text in the underlying XML.
- **DOCX images**: embedded media parts are repacked with the redacted regions drawn as opaque rectangles before the ZIP is rewritten ([redact.py:99-143](thesis_project/defender/redact.py#L99-L143)).
- **PDF text and images**: redactions go through PyMuPDF's `add_redact_annot` + `apply_redactions()` ([redact.py:316-327](thesis_project/defender/redact.py#L316-L327)), which strips the underlying content stream — copy/paste, text search, and PDF inspection tools cannot recover the original.

Each run also writes an audit trail (`audit_log.jsonl`) and a redaction mapping JSON so you can verify what was removed.

### Privacy of the NER backend

All detection runs **locally on your machine**. GLiNER, the candidate name model, EasyOCR, and any transformer NER are loaded into the local Python process  no document text, no detections, and no PII ever leave the machine. Pass `--ner-local-files-only` to additionally block GLiNER from contacting Hugging Face for model downloads (use a pre-downloaded local model path).

## Run the experiments (RQ1–RQ4)

```python
from thesis_project import Experiments, Config

experiments = Experiments(dataset_root="notebooks/workspace/dataset", config=Config())
results = experiments.run_all()        # all four research questions
# or run one at a time: experiments.rq1(), .rq2(), .rq3(), .rq4()
```
or go in the notebooks

## Run the tests

```bash
pytest test_pipeline_integration.py -v
```

## UIs

Streamlit review app:

```bash
streamlit run UI/ui.py
```

Flask labeling app (RQ3 annotation review):

```bash
python UI/label.py
# open http://127.0.0.1:7998
```

## AI tool usage

The codebase was designed and written by the author. GitHub Copilot provided in-editor autocomplete during development, and ChatGPT was
used to draft isolated presentation code (plotting helpers in the
notebooks and the two UIs oen for labeling and one for general use case) and to draft this README (: The pipeline was otherwise written by the author, except for the regex patterns
and the UI front-ends. AI tools were also used for occasional debugging, chart styling decisions, and cross-checking numerical claims against the code.