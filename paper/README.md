# Paper: Adaptive LLM Model Routing Using Contextual Bandits with Feedback

IEEE-format manuscript describing the LinUCB-based routing framework.

| File | Description |
|---|---|
| `Adaptive_LLM_Model_Routing_IEEE_Paper.docx` | Manuscript (Word) |
| `Adaptive_LLM_Model_Routing_IEEE_Paper_preview.pdf` | PDF preview rendered with LibreOffice |
| `figures/` | Fig. 1 routing loop, Fig. 2 evaluation design (Fig. 3, implementation view, is no longer used) |
| `build/content.py` | Paper text, tables, equations and references (citations as `{key}`, numbered automatically) |
| `build/figs.py` | Generates the three figures (matplotlib) |
| `build/build.py` | Rewrites `word/document.xml` of an unpacked .docx from `content.py` |

## Status

Six pages. Section VI reports the GSM8K pilot (300 problems, 90 dev / 210 test; Qwen3-4B/8B/14B FP8 on one
RTX 4090; 5 seeds). Numbers come from `results/gsm8k_pilot_models_4090/` (`tables.md`, `results.json`,
`sweeps/sweeps.md`) and per-model statistics from `data/outcomes_gsm8k_models_4090.jsonl`. Only the reasoning
category has been run, so per-category (E3) and workload-shift (E5) results are not reported yet.
The author block still needs to be confirmed (`[Guide Email ID]`).

## Rebuilding after editing `content.py`

```bash
mkdir -p /tmp/paper && cd /tmp/paper
unzip -q /path/to/paper/Adaptive_LLM_Model_Routing_IEEE_Paper.docx -d unpacked
python3 /path/to/paper/build/figs.py unpacked/word/media      # optional: regenerate figures (RGBA PNG)
python3 /path/to/paper/build/build.py unpacked
(cd unpacked && zip -qXr ../paper.docx .)
```
