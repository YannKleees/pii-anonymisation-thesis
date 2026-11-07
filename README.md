# PII detection and anonymisation of student project documents

Bachelor's thesis code (BSc Data Science and AI, Maastricht University, 2026). It generates realistic synthetic student project documents (agendas, minutes, project plans as DOCX and PDF), detects and removes personal data in them with a layout-aware pipeline (patterns, GLiNER, OCR, cross-region linking), and then attacks the redacted documents to measure how much re-identification risk is left.

## Results

On 500 synthetic documents with 8,768 annotated identifier mentions:

- **RQ1, detection:** the full pipeline finds 8,750 of 8,768 mentions (recall 0.998; 18 missed, 13 of them on cover pages). Precision is 0.70 because it over-redacts, mostly dates (2,603 of 3,768 extra redactions). For direct identifiers alone precision is 0.90.
- **RQ2, components:** a text-only baseline finds 44% of mentions. Layout-aware zoning (cover pages, headers, footers, tables) raises this to 98%, OCR and cross-region linking to 99.8%. GLiNER removes the most leakage of all components.
- **RQ4, re-identification:** after redaction, an attacker has 0.16 usable signals per document instead of 17.3 in the original and can no longer narrow down who wrote it (median 3,500 candidates, about the whole synthetic population of 3,533). Removing only direct identifiers still leaves 4 signals and narrows the field to 76 candidates.
- **RQ3** repeats the evaluation on real, manually reviewed student documents. Those documents and their knowledge base contain personal data and are not part of this repository, so RQ3 and the real-document part of RQ4 cannot be re-run from it; their aggregated results are kept in the notebook outputs.

## Stack

Python · GLiNER, transformers, PyTorch · EasyOCR, OpenCV · PyMuPDF, python-docx, ReportLab · scikit-learn · Faker · pandas · Streamlit and Flask (review UIs)

## Run it

```
pip install -r requirements.txt
```

LibreOffice is optional (DOCX to PDF export). GLiNER and EasyOCR download their models on first use.

Generate a synthetic dataset (the config used for the thesis is included):

```
python -m document_generator.generate --config notebooks/workspace/generation-config.json
```

Redact documents:

```
python -m thesis_project.defender path/to/file_or_folder --output-root out/
```

By default only direct identifiers (names, student IDs, emails) are removed; `--redact-quasi` also removes quasi-identifiers such as course codes, dates and group labels. Other flags: `--no-pdf`, `--ner-model NAME`, `--pdf-ocr-all-pages`, `--ner-local-files-only`.

The defender writes a new file (`<stem>_anonymized.<ext>`) and never changes the source. Redacted content is removed, not covered: DOCX text runs are rewritten, embedded images are repacked with the regions blacked out, and PDFs go through PyMuPDF's `apply_redactions()`, so copy and paste or text search cannot recover the original. Each run also writes an audit log and a redaction map. Everything runs locally; no document text leaves the machine.

Experiments: the notebooks `notebooks/RQ1.ipynb` to `RQ4.ipynb`, or

```python
from thesis_project import Experiments, Config
Experiments(dataset_root="notebooks/workspace/dataset", config=Config()).rq1()   # also rq2(), rq4()
```

Review UIs (need `pip install streamlit flask`): `streamlit run UI/ui.py` and `python UI/label.py` (http://127.0.0.1:7998).

## Data

All documents in `notebooks/workspace/dataset` are synthetic: people come from Faker, and any match with a real person is a coincidence. The real documents used for RQ3 are not included.

## AI tool usage

The codebase was designed and written by me. GitHub Copilot provided in-editor autocomplete, and ChatGPT was used to draft isolated presentation code (plotting helpers in the notebooks and the two UIs, one for labeling and one for general use) and a first draft of this README. The pipeline was otherwise written by me, except for the regex patterns and the UI front-ends. AI tools were also used for occasional debugging, chart styling decisions and cross-checking numerical claims against the code.
