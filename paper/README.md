# Paper: Adaptive LLM Model Routing Using Contextual Bandits with Feedback

IEEE-format manuscript describing the LinUCB-based routing framework.

| File | Description |
|---|---|
| `Adaptive_LLM_Model_Routing_IEEE_Paper.docx` | Manuscript (Word) |
| `Adaptive_LLM_Model_Routing_IEEE_Paper_preview.pdf` | PDF preview rendered with LibreOffice |
| `figures/` | Figure images. The paper uses Fig. 1 (routing loop, `image1.png`) and Fig. 2 (pilot results, `image3.png`, drawn from `results/gsm8k_pilot_models_4090/results.json`); `image2.png` (evaluation design) is generated but not placed |
| `build/content.py` | Paper text, tables, equations and references (citations as `{key}`, numbered automatically) |
| `build/figs.py` | Generates the figures (matplotlib); the results plot reads `results/gsm8k_pilot_models_4090/results.json` |
| `build/build.py` | Rewrites `word/document.xml` of an unpacked .docx from `content.py` |

## Status

Six pages: five pages of text and a sixth page that holds only the references. Figures are borderless floating tables pinned to the top of a page (out of the text flow) and the three results tables fit in one column, so no blank bands appear when content moves between pages. Section VI reports the GSM8K pilot (300 problems, 90 dev / 210 test; Qwen3-4B/8B/14B FP8 on one
RTX 4090; 5 seeds). Numbers come from `results/gsm8k_pilot_models_4090/` (`tables.md`, `results.json`,
`sweeps/sweeps.md`) and per-model statistics from `data/outcomes_gsm8k_models_4090.jsonl`. Only the reasoning
category has been run, so per-category (E3) and workload-shift (E5) results are not reported yet.
The guide's email line was a placeholder and has been removed from the author block; add it back in Word if needed.

## Rebuilding after editing `content.py`

```bash
mkdir -p /tmp/paper && cd /tmp/paper
unzip -q /path/to/paper/Adaptive_LLM_Model_Routing_IEEE_Paper.docx -d unpacked
python3 /path/to/paper/build/figs.py unpacked/word/media      # optional: regenerate figures (RGBA PNG)
python3 /path/to/paper/build/build.py unpacked
(cd unpacked && zip -qXr ../paper.docx .)
```
