#!/bin/bash
# Downloads the reading list for 'Post-Training, Visually' from arXiv and zips it.
# Usage:  bash fetch_papers.sh          (needs curl and zip; ~78 PDFs; takes a few minutes)
# arXiv asks for no more than one request every few seconds from scripts; the sleep below respects that.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)/posttraining_papers"
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

get "1_overview_sft_and_instruction_data" "wei2021flan.pdf" "https://arxiv.org/pdf/2109.01652"
get "1_overview_sft_and_instruction_data" "sanh2021t0.pdf" "https://arxiv.org/pdf/2110.08207"
get "1_overview_sft_and_instruction_data" "ouyang2022instructgpt.pdf" "https://arxiv.org/pdf/2203.02155"
get "1_overview_sft_and_instruction_data" "wang2022selfinstruct.pdf" "https://arxiv.org/pdf/2212.10560"
echo "manual: 1_overview_sft_and_instruction_data/taori2023alpaca  ->  https://github.com/tatsu-lab/stanford_alpaca"   # not a direct PDF; open in a browser
get "1_overview_sft_and_instruction_data" "zhou2023lima.pdf" "https://arxiv.org/pdf/2305.11206"
get "1_overview_sft_and_instruction_data" "lambert2024tulu3.pdf" "https://arxiv.org/pdf/2411.15124"
get "1_overview_sft_and_instruction_data" "hu2021lora.pdf" "https://arxiv.org/pdf/2106.09685"
get "1_overview_sft_and_instruction_data" "dettmers2023qlora.pdf" "https://arxiv.org/pdf/2305.14314"
get "1_overview_sft_and_instruction_data" "zelikman2022star.pdf" "https://arxiv.org/pdf/2203.14465"
get "1_overview_sft_and_instruction_data" "bai2022hh.pdf" "https://arxiv.org/pdf/2204.05862"
get "2_preference_optimization_rlhf_reward_models_ppo_dpo" "christiano2017preferences.pdf" "https://arxiv.org/pdf/1706.03741"
get "2_preference_optimization_rlhf_reward_models_ppo_dpo" "ziegler2019finetuning.pdf" "https://arxiv.org/pdf/1909.08593"
get "2_preference_optimization_rlhf_reward_models_ppo_dpo" "stiennon2020summarize.pdf" "https://arxiv.org/pdf/2009.01325"
echo "manual: 2_preference_optimization_rlhf_reward_models_ppo_dpo/bradley1952rank  ->  https://doi.org/10.2307/2334029"   # not a direct PDF; open in a browser
get "2_preference_optimization_rlhf_reward_models_ppo_dpo" "gao2022overoptimization.pdf" "https://arxiv.org/pdf/2210.10760"
get "2_preference_optimization_rlhf_reward_models_ppo_dpo" "schulman2017ppo.pdf" "https://arxiv.org/pdf/1707.06347"
get "2_preference_optimization_rlhf_reward_models_ppo_dpo" "rafailov2023dpo.pdf" "https://arxiv.org/pdf/2305.18290"
get "2_preference_optimization_rlhf_reward_models_ppo_dpo" "azar2023ipo.pdf" "https://arxiv.org/pdf/2310.12036"
get "2_preference_optimization_rlhf_reward_models_ppo_dpo" "ethayarajh2024kto.pdf" "https://arxiv.org/pdf/2402.01306"
get "2_preference_optimization_rlhf_reward_models_ppo_dpo" "hong2024orpo.pdf" "https://arxiv.org/pdf/2403.07691"
get "2_preference_optimization_rlhf_reward_models_ppo_dpo" "meng2024simpo.pdf" "https://arxiv.org/pdf/2405.14734"
get "2_preference_optimization_rlhf_reward_models_ppo_dpo" "ahmadian2024rloo.pdf" "https://arxiv.org/pdf/2402.14740"
get "2_preference_optimization_rlhf_reward_models_ppo_dpo" "cui2023ultrafeedback.pdf" "https://arxiv.org/pdf/2310.01377"
get "3_rl_with_verifiable_rewards_and_rl_systems" "shao2024deepseekmath.pdf" "https://arxiv.org/pdf/2402.03300"
get "3_rl_with_verifiable_rewards_and_rl_systems" "yu2025dapo.pdf" "https://arxiv.org/pdf/2503.14476"
get "3_rl_with_verifiable_rewards_and_rl_systems" "liu2025drgrpo.pdf" "https://arxiv.org/pdf/2503.20783"
get "3_rl_with_verifiable_rewards_and_rl_systems" "sheng2024hybridflow.pdf" "https://arxiv.org/pdf/2409.19256"
get "3_rl_with_verifiable_rewards_and_rl_systems" "hu2025openreasonerzero.pdf" "https://arxiv.org/pdf/2503.24290"
echo "manual: 3_rl_with_verifiable_rewards_and_rl_systems/ring2025icepop  ->  https://huggingface.co/inclusionAI/Ring-1T"   # not a direct PDF; open in a browser
get "4_reasoning_test-time_compute_r1_distillation_budgets" "wei2022cot.pdf" "https://arxiv.org/pdf/2201.11903"
get "4_reasoning_test-time_compute_r1_distillation_budgets" "lightman2023prm.pdf" "https://arxiv.org/pdf/2305.20050"
get "4_reasoning_test-time_compute_r1_distillation_budgets" "snell2024testtime.pdf" "https://arxiv.org/pdf/2408.03314"
get "4_reasoning_test-time_compute_r1_distillation_budgets" "deepseek2025r1.pdf" "https://arxiv.org/pdf/2501.12948"
get "4_reasoning_test-time_compute_r1_distillation_budgets" "kimi2025k15.pdf" "https://arxiv.org/pdf/2501.12599"
get "4_reasoning_test-time_compute_r1_distillation_budgets" "muennighoff2025s1.pdf" "https://arxiv.org/pdf/2501.19393"
get "4_reasoning_test-time_compute_r1_distillation_budgets" "qwen2025qwen3.pdf" "https://arxiv.org/pdf/2505.09388"
get "4_reasoning_test-time_compute_r1_distillation_budgets" "yue2025passk.pdf" "https://arxiv.org/pdf/2504.13837"
get "4_reasoning_test-time_compute_r1_distillation_budgets" "guha2025openthoughts.pdf" "https://arxiv.org/pdf/2506.04178"
echo "manual: 4_reasoning_test-time_compute_r1_distillation_budgets/openai2024o1  ->  https://openai.com/index/learning-to-reason-with-llms/"   # not a direct PDF; open in a browser
get "5_long_context" "su2021rope.pdf" "https://arxiv.org/pdf/2104.09864"
get "5_long_context" "chen2023pi.pdf" "https://arxiv.org/pdf/2306.15595"
echo "manual: 5_long_context/bloc97_2023_ntk  ->  https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/"   # not a direct PDF; open in a browser
get "5_long_context" "peng2023yarn.pdf" "https://arxiv.org/pdf/2309.00071"
get "5_long_context" "xiong2023abf.pdf" "https://arxiv.org/pdf/2309.16039"
get "5_long_context" "ding2024longrope.pdf" "https://arxiv.org/pdf/2402.13753"
get "5_long_context" "grattafiori2024llama3.pdf" "https://arxiv.org/pdf/2407.21783"
get "5_long_context" "liu2023lostmiddle.pdf" "https://arxiv.org/pdf/2307.03172"
get "5_long_context" "hsieh2024ruler.pdf" "https://arxiv.org/pdf/2404.06654"
echo "manual: 5_long_context/kamradt2023niah  ->  https://github.com/gkamradt/LLMTest_NeedleInAHaystack"   # not a direct PDF; open in a browser
get "5_long_context" "deepseek2024v3.pdf" "https://arxiv.org/pdf/2412.19437"
get "6_safety_alignment" "askell2021hhh.pdf" "https://arxiv.org/pdf/2112.00861"
get "6_safety_alignment" "bai2022cai.pdf" "https://arxiv.org/pdf/2212.08073"
get "6_safety_alignment" "lee2023rlaif.pdf" "https://arxiv.org/pdf/2309.00267"
get "6_safety_alignment" "touvron2023llama2.pdf" "https://arxiv.org/pdf/2307.09288"
get "6_safety_alignment" "rottger2023xstest.pdf" "https://arxiv.org/pdf/2308.01263"
get "6_safety_alignment" "zou2023gcg.pdf" "https://arxiv.org/pdf/2307.15043"
echo "manual: 6_safety_alignment/anil2024manyshot  ->  https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea456e232efb72d261715e33ce25f208-Abstract.html"   # not a direct PDF; open in a browser
get "6_safety_alignment" "mazeika2024harmbench.pdf" "https://arxiv.org/pdf/2402.04249"
get "6_safety_alignment" "inan2023llamaguard.pdf" "https://arxiv.org/pdf/2312.06674"
echo "manual: 6_safety_alignment/meta2024llamaguard3  ->  https://huggingface.co/meta-llama/Llama-Guard-3-8B"   # not a direct PDF; open in a browser
get "6_safety_alignment" "guan2024deliberative.pdf" "https://arxiv.org/pdf/2412.16339"
get "6_safety_alignment" "qi2023finetuning.pdf" "https://arxiv.org/pdf/2310.03693"
get "6_safety_alignment" "qi2024shallow.pdf" "https://arxiv.org/pdf/2406.05946"
get "6_safety_alignment" "li2024wmdp.pdf" "https://arxiv.org/pdf/2403.03218"
get "6_safety_alignment" "tamirisa2024tar.pdf" "https://arxiv.org/pdf/2408.00761"
get "7_tools_and_agents" "schick2023toolformer.pdf" "https://arxiv.org/pdf/2302.04761"
get "7_tools_and_agents" "yao2022react.pdf" "https://arxiv.org/pdf/2210.03629"
get "7_tools_and_agents" "qin2023toolllm.pdf" "https://arxiv.org/pdf/2307.16789"
get "7_tools_and_agents" "liu2024apigen.pdf" "https://arxiv.org/pdf/2406.18518"
get "7_tools_and_agents" "patil2025bfcl.pdf" "https://proceedings.mlr.press/v267/patil25a/patil25a.pdf"
get "7_tools_and_agents" "yao2024taubench.pdf" "https://arxiv.org/pdf/2406.12045"
get "7_tools_and_agents" "jimenez2023swebench.pdf" "https://arxiv.org/pdf/2310.06770"
echo "manual: 7_tools_and_agents/openai2024swebenchverified  ->  https://openai.com/index/introducing-swe-bench-verified/"   # not a direct PDF; open in a browser
get "7_tools_and_agents" "yang2024sweagent.pdf" "https://arxiv.org/pdf/2405.15793"
get "7_tools_and_agents" "kimi2025k2.pdf" "https://arxiv.org/pdf/2507.20534"
echo "manual: 7_tools_and_agents/nvidia2025nemogym  ->  https://github.com/NVIDIA-NeMo/Gym"   # not a direct PDF; open in a browser
get "8_open_recipes_and_on-policy_distillation" "agarwal2023gkd.pdf" "https://arxiv.org/pdf/2306.13649"
echo "manual: 8_open_recipes_and_on-policy_distillation/thinkingmachines2025opd  ->  https://thinkingmachines.ai/blog/on-policy-distillation/"   # not a direct PDF; open in a browser
get "8_open_recipes_and_on-policy_distillation" "ma2026mopd.pdf" "https://arxiv.org/pdf/2606.30406"
echo "manual: 8_open_recipes_and_on-policy_distillation/ai2_2025_olmo3  ->  https://allenai.org/blog/olmo3"   # not a direct PDF; open in a browser
get "8_open_recipes_and_on-policy_distillation" "tunstall2023zephyr.pdf" "https://arxiv.org/pdf/2310.16944"
get "8_open_recipes_and_on-policy_distillation" "ivison2023tulu2.pdf" "https://arxiv.org/pdf/2311.10702"
get "9_evaluation" "zhou2023ifeval.pdf" "https://arxiv.org/pdf/2311.07911"
get "9_evaluation" "zheng2023mtbench.pdf" "https://arxiv.org/pdf/2306.05685"
get "9_evaluation" "dubois2024alpacaeval.pdf" "https://arxiv.org/pdf/2404.04475"
get "9_evaluation" "li2024arenahard.pdf" "https://arxiv.org/pdf/2406.11939"
get "9_evaluation" "jain2024livecodebench.pdf" "https://arxiv.org/pdf/2403.07974"
get "9_evaluation" "rein2023gpqa.pdf" "https://arxiv.org/pdf/2311.12022"
get "9_evaluation" "hendrycks2020mmlu.pdf" "https://arxiv.org/pdf/2009.03300"

cp "$(dirname "$0")/reading_list.csv" "$(dirname "$0")/reading_list.bib" "$(dirname "$0")/README.md" "$ROOT/" 2>/dev/null
echo "downloaded: $ok   failed: $fail"
( cd "$(dirname "$ROOT")" && rm -f posttraining_papers.zip && zip -qr posttraining_papers.zip "$(basename "$ROOT")" ) && echo "wrote $(dirname "$ROOT")/posttraining_papers.zip"
