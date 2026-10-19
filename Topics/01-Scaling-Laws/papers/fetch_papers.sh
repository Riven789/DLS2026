#!/bin/bash
# Downloads the reading list for 'Scaling Laws, Visually' from arXiv and zips it.
# Usage:  bash fetch_papers.sh          (needs curl and zip; ~43 PDFs; takes a few minutes)
# arXiv asks for no more than one request every few seconds from scripts; the sleep below respects that.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)/scaling_laws_papers"
mkdir -p "$ROOT"
fail=0; ok=0
get() {  # get <dir> <filename> <url>
  mkdir -p "$ROOT/$1"
  if [ -s "$ROOT/$1/$2" ]; then ok=$((ok+1)); return; fi
  if curl -sS -L --retry 3 --retry-delay 5 -A "fetch_papers.sh (reading list downloader)" -o "$ROOT/$1/$2" "$3" && [ -s "$ROOT/$1/$2" ] && head -c 4 "$ROOT/$1/$2" | grep -q "%PDF"; then
    ok=$((ok+1)); echo "ok   $1/$2"
  else
    rm -f "$ROOT/$1/$2"; fail=$((fail+1)); echo "FAIL $1/$2  ($3)"
  fi
  sleep 3
}

get "1_before_language_models" "cortes1993learning.pdf" "https://proceedings.neurips.cc/paper_files/paper/1993/file/1aa48fc4880bb0c9b8a3bf979d3b917e-Paper.pdf"
get "1_before_language_models" "banko2001scaling.pdf" "https://aclanthology.org/P01-1005.pdf"
echo "manual: 1_before_language_models/halevy2009unreasonable  ->  https://doi.org/10.1109/MIS.2009.36"   # not a direct PDF; open in a browser
get "1_before_language_models" "sun2017revisiting.pdf" "https://arxiv.org/pdf/1707.02968"
get "1_before_language_models" "hestness2017predictable.pdf" "https://arxiv.org/pdf/1712.00409"
echo "manual: 1_before_language_models/amodei2018compute  ->  https://openai.com/index/ai-and-compute/"   # not a direct PDF; open in a browser
get "1_before_language_models" "rosenfeld2019constructive.pdf" "https://arxiv.org/pdf/1909.12673"
get "1_before_language_models" "sevilla2022compute.pdf" "https://arxiv.org/pdf/2202.05924"
get "2_core_results_kaplan_and_chinchilla" "kaplan2020scaling.pdf" "https://arxiv.org/pdf/2001.08361"
get "2_core_results_kaplan_and_chinchilla" "brown2020gpt3.pdf" "https://arxiv.org/pdf/2005.14165"
get "2_core_results_kaplan_and_chinchilla" "henighan2020autoregressive.pdf" "https://arxiv.org/pdf/2010.14701"
get "2_core_results_kaplan_and_chinchilla" "hoffmann2022chinchilla.pdf" "https://arxiv.org/pdf/2203.15556"
get "2_core_results_kaplan_and_chinchilla" "besiroglu2024replication.pdf" "https://arxiv.org/pdf/2404.10102"
get "2_core_results_kaplan_and_chinchilla" "porian2024discrepancies.pdf" "https://arxiv.org/pdf/2406.19146"
get "2_core_results_kaplan_and_chinchilla" "mccandlish2018batch.pdf" "https://arxiv.org/pdf/1812.06162"
get "3_refinements" "sardana2023inference.pdf" "https://arxiv.org/pdf/2401.00448"
get "3_refinements" "gadre2024overtraining.pdf" "https://arxiv.org/pdf/2403.08540"
get "3_refinements" "muennighoff2023dataconstrained.pdf" "https://arxiv.org/pdf/2305.16264"
get "3_refinements" "villalobos2022data.pdf" "https://arxiv.org/pdf/2211.04325"
get "3_refinements" "deepseek2024llm.pdf" "https://arxiv.org/pdf/2401.02954"
get "3_refinements" "yang2022mutransfer.pdf" "https://arxiv.org/pdf/2203.03466"
get "3_refinements" "hu2024minicpm.pdf" "https://arxiv.org/pdf/2404.06395"
get "3_refinements" "hagele2024beyond.pdf" "https://arxiv.org/pdf/2405.18392"
get "3_refinements" "dey2023cerebras.pdf" "https://arxiv.org/pdf/2304.03208"
get "3_refinements" "wortsman2023instabilities.pdf" "https://arxiv.org/pdf/2309.14322"
get "4_architecture_knobs" "clark2022unified.pdf" "https://arxiv.org/pdf/2202.01169"
get "4_architecture_knobs" "krajewski2024finegrained.pdf" "https://arxiv.org/pdf/2402.07871"
get "4_architecture_knobs" "tao2024vocabulary.pdf" "https://arxiv.org/pdf/2407.13623"
get "4_architecture_knobs" "kumar2024precision.pdf" "https://arxiv.org/pdf/2411.04330"
get "4_architecture_knobs" "ye2024datamixing.pdf" "https://arxiv.org/pdf/2403.16952"
get "5_loss_to_capability" "wei2022emergent.pdf" "https://arxiv.org/pdf/2206.07682"
get "5_loss_to_capability" "schaeffer2023mirage.pdf" "https://arxiv.org/pdf/2304.15004"
get "5_loss_to_capability" "openai2023gpt4.pdf" "https://arxiv.org/pdf/2303.08774"
get "5_loss_to_capability" "grattafiori2024llama3.pdf" "https://arxiv.org/pdf/2407.21783"
get "5_loss_to_capability" "bhagia2024ladders.pdf" "https://arxiv.org/pdf/2412.04403"
get "5_loss_to_capability" "isik2024downstream.pdf" "https://arxiv.org/pdf/2402.04177"
get "6_why_power_laws" "bahri2021explaining.pdf" "https://arxiv.org/pdf/2102.06701"
get "6_why_power_laws" "hutter2021learningcurve.pdf" "https://arxiv.org/pdf/2102.04074"
get "6_why_power_laws" "michaud2023quantization.pdf" "https://arxiv.org/pdf/2303.13506"
get "6_why_power_laws" "sharma2020neural.pdf" "https://arxiv.org/pdf/2004.10802"
get "7_ladders_and_practice" "choshen2024hitchhiker.pdf" "https://arxiv.org/pdf/2410.11840"
get "7_ladders_and_practice" "olmo2024olmo2.pdf" "https://arxiv.org/pdf/2501.00656"
get "7_ladders_and_practice" "wen2024cool.pdf" "https://arxiv.org/pdf/2410.05192"
get "7_ladders_and_practice" "hernandez2021transfer.pdf" "https://arxiv.org/pdf/2102.01293"
echo "manual: 8_frontier_case_studies/microsoft2026frontier  ->  https://microsoft.ai/news/"   # not a direct PDF; open in a browser
get "8_frontier_case_studies" "deepseek2024v3.pdf" "https://arxiv.org/pdf/2412.19437"

cp "$(dirname "$0")/reading_list.csv" "$(dirname "$0")/reading_list.bib" "$(dirname "$0")/README.md" "$ROOT/" 2>/dev/null
echo "downloaded: $ok   failed: $fail"
( cd "$(dirname "$ROOT")" && rm -f scaling_laws_papers.zip && zip -qr scaling_laws_papers.zip "$(basename "$ROOT")" ) && echo "wrote $(dirname "$ROOT")/scaling_laws_papers.zip"
