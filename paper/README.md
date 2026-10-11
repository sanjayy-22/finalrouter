# Paper: Adaptive LLM Model Routing Using Contextual Bandits with Feedback

IEEE-format manuscript describing the LinUCB-based routing framework.

| File | Description |
|---|---|
| `Adaptive_LLM_Model_Routing_IEEE_Paper.docx` | Manuscript (Word) |
| `Adaptive_LLM_Model_Routing_IEEE_Paper_preview.pdf` | PDF preview rendered with LibreOffice |
| `figures/` | Fig. 1 routing loop, Fig. 2 evaluation design, Fig. 3 implementation view |
| `build/content.py` | Paper text, tables, equations and references (citations as `{key}`, numbered automatically) |
| `build/figs.py` | Generates the three figures (matplotlib) |
| `build/build.py` | Rewrites `word/document.xml` of an unpacked .docx from `content.py` |

## Status

Experiments have not been run yet. Section VI defines the reporting format only:
cells marked **[TBD]** (Tables VII–VIII) and **[To be recorded]** (Table I) are
highlighted placeholders to be replaced with measured values. The author block
still needs to be confirmed.

## Rebuilding after editing `content.py`

```bash
mkdir -p /tmp/paper && cd /tmp/paper
unzip -q /path/to/paper/Adaptive_LLM_Model_Routing_IEEE_Paper.docx -d unpacked
python3 /path/to/paper/build/figs.py unpacked/word/media      # optional: regenerate figures (RGBA PNG)
python3 /path/to/paper/build/build.py unpacked
(cd unpacked && zip -qXr ../paper.docx .)
```
