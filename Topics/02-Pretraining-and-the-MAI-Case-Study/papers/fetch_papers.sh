#!/bin/bash
# Downloads the reading list for 'Frontier Pre-training, Visually' from arXiv and zips it.
# Usage:  bash fetch_papers.sh          (needs curl and zip; ~51 PDFs; takes a few minutes)
# arXiv asks for no more than one request every few seconds from scripts; the sleep below respects that.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)/pretraining_papers"
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

echo "manual: 1_overview_and_results/microsoft2026frontier  ->  https://microsoft.ai/news/"   # not a direct PDF; open in a browser
get "1_overview_and_results" "chowdhery2022palm.pdf" "https://arxiv.org/pdf/2204.02311"
get "1_overview_and_results" "deletang2023compression.pdf" "https://arxiv.org/pdf/2309.10668"
get "2_the_model" "shazeer2017moe.pdf" "https://arxiv.org/pdf/1701.06538"
get "2_the_model" "lepikhin2020gshard.pdf" "https://arxiv.org/pdf/2006.16668"
get "2_the_model" "fedus2021switch.pdf" "https://arxiv.org/pdf/2101.03961"
get "2_the_model" "dai2024deepseekmoe.pdf" "https://arxiv.org/pdf/2401.06066"
get "2_the_model" "ainslie2023gqa.pdf" "https://arxiv.org/pdf/2305.13245"
get "2_the_model" "press2017tying.pdf" "https://arxiv.org/pdf/1608.05859"
get "3_scaling_ladders" "kaplan2020scaling.pdf" "https://arxiv.org/pdf/2001.08361"
get "3_scaling_ladders" "hoffmann2022chinchilla.pdf" "https://arxiv.org/pdf/2203.15556"
get "3_scaling_ladders" "porian2024discrepancies.pdf" "https://arxiv.org/pdf/2406.19146"
get "3_scaling_ladders" "muennighoff2023dataconstrained.pdf" "https://arxiv.org/pdf/2305.16264"
get "3_scaling_ladders" "bhagia2024ladders.pdf" "https://arxiv.org/pdf/2412.04403"
get "3_scaling_ladders" "choshen2024hitchhiker.pdf" "https://arxiv.org/pdf/2410.11840"
get "4_hyperparameter_laws" "loshchilov2019adamw.pdf" "https://arxiv.org/pdf/1711.05101"
get "4_hyperparameter_laws" "wang2024weightdecay.pdf" "https://arxiv.org/pdf/2405.13698"
get "4_hyperparameter_laws" "malladi2022sdes.pdf" "https://arxiv.org/pdf/2205.10287"
get "4_hyperparameter_laws" "yang2022mutransfer.pdf" "https://arxiv.org/pdf/2203.03466"
get "4_hyperparameter_laws" "deepseek2024llm.pdf" "https://arxiv.org/pdf/2401.02954"
get "4_hyperparameter_laws" "hu2024minicpm.pdf" "https://arxiv.org/pdf/2404.06395"
echo "manual: 4_hyperparameter_laws/srivastava2014dropout  ->  https://jmlr.org/papers/v15/srivastava14a.html"   # not a direct PDF; open in a browser
get "5_moe_systems" "gale2022megablocks.pdf" "https://arxiv.org/pdf/2211.15841"
get "5_moe_systems" "rajbhandari2022deepspeedmoe.pdf" "https://arxiv.org/pdf/2201.05596"
get "5_moe_systems" "hwang2022tutel.pdf" "https://arxiv.org/pdf/2206.03382"
get "5_moe_systems" "zoph2022stmoe.pdf" "https://arxiv.org/pdf/2202.08906"
get "5_moe_systems" "wang2024auxlossfree.pdf" "https://arxiv.org/pdf/2408.15664"
get "5_moe_systems" "korthikanti2022recomputation.pdf" "https://arxiv.org/pdf/2205.05198"
echo "manual: 6_muon/jordan2024muon  ->  https://kellerjordan.github.io/posts/muon/"   # not a direct PDF; open in a browser
get "6_muon" "bernstein2024anthology.pdf" "https://arxiv.org/pdf/2409.20325"
get "6_muon" "liu2025muonscalable.pdf" "https://arxiv.org/pdf/2502.16982"
get "6_muon" "kimi2025k2.pdf" "https://arxiv.org/pdf/2507.20534"
get "6_muon" "rajbhandari2020zero.pdf" "https://arxiv.org/pdf/1910.02054"
get "6_muon" "wen2025fantastic.pdf" "https://arxiv.org/pdf/2509.02046"
get "7_numerics_and_stability" "micikevicius2022fp8.pdf" "https://arxiv.org/pdf/2209.05433"
get "7_numerics_and_stability" "rouhani2023mx.pdf" "https://arxiv.org/pdf/2310.10537"
get "7_numerics_and_stability" "nvidia2025mxfp8.pdf" "https://arxiv.org/pdf/2506.08027"
get "7_numerics_and_stability" "fishman2024fp8.pdf" "https://arxiv.org/pdf/2409.12517"
get "7_numerics_and_stability" "pascanu2013clipping.pdf" "https://arxiv.org/pdf/1211.5063"
get "7_numerics_and_stability" "dehghani2023vit22b.pdf" "https://arxiv.org/pdf/2302.05442"
get "7_numerics_and_stability" "wortsman2023instabilities.pdf" "https://arxiv.org/pdf/2309.14322"
get "7_numerics_and_stability" "olmo2024olmo2.pdf" "https://arxiv.org/pdf/2501.00656"
get "8_data_and_vision" "penedo2024fineweb.pdf" "https://arxiv.org/pdf/2406.17557"
get "8_data_and_vision" "li2024dclm.pdf" "https://arxiv.org/pdf/2406.11794"
get "8_data_and_vision" "abbas2023semdedup.pdf" "https://arxiv.org/pdf/2303.09540"
get "8_data_and_vision" "tirumala2023d4.pdf" "https://arxiv.org/pdf/2308.12284"
get "8_data_and_vision" "ye2024mixinglaws.pdf" "https://arxiv.org/pdf/2403.16952"
get "8_data_and_vision" "dosovitskiy2020vit.pdf" "https://arxiv.org/pdf/2010.11929"
echo "manual: 8_data_and_vision/bavishi2023fuyu  ->  https://www.adept.ai/blog/fuyu-8b"   # not a direct PDF; open in a browser
get "8_data_and_vision" "diao2024eve.pdf" "https://arxiv.org/pdf/2406.11832"
get "8_data_and_vision" "beyer2024paligemma.pdf" "https://arxiv.org/pdf/2407.07726"
get "9_comparators_and_frontier_reports" "deepseek2024v3.pdf" "https://arxiv.org/pdf/2412.19437"
get "9_comparators_and_frontier_reports" "qwen2025qwen3.pdf" "https://arxiv.org/pdf/2505.09388"
get "9_comparators_and_frontier_reports" "yang2024gateddeltanet.pdf" "https://arxiv.org/pdf/2412.06464"
get "9_comparators_and_frontier_reports" "kimi2025linear.pdf" "https://arxiv.org/pdf/2510.26692"
echo "manual: 9_comparators_and_frontier_reports/meta2025llama4  ->  https://ai.meta.com/blog/llama-4-multimodal-intelligence/"   # not a direct PDF; open in a browser
echo "manual: 9_comparators_and_frontier_reports/deepseek2026v4card  ->  https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base"   # not a direct PDF; open in a browser
echo "manual: 9_comparators_and_frontier_reports/moonshot2026k3card  ->  https://huggingface.co/moonshotai/Kimi-K3"   # not a direct PDF; open in a browser
echo "manual: 9_comparators_and_frontier_reports/qwen2026qwen38card  ->  https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B"   # not a direct PDF; open in a browser
echo "manual: 9_comparators_and_frontier_reports/xiaomi2026mimocard  ->  https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL"   # not a direct PDF; open in a browser

cp "$(dirname "$0")/reading_list.csv" "$(dirname "$0")/reading_list.bib" "$(dirname "$0")/README.md" "$ROOT/" 2>/dev/null
echo "downloaded: $ok   failed: $fail"
( cd "$(dirname "$ROOT")" && rm -f pretraining_papers.zip && zip -qr pretraining_papers.zip "$(basename "$ROOT")" ) && echo "wrote $(dirname "$ROOT")/pretraining_papers.zip"
