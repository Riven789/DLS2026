#!/bin/bash
# Downloads the reading list for 'Reasoning Models' from arXiv and zips it.
# Usage:  bash fetch_papers.sh          (needs curl and zip; ~118 PDFs; takes a few minutes)
# arXiv asks for no more than one request every few seconds from scripts; the sleep below respects that.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)/reasoning_models_papers"
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

echo "manual: 1_overview_and_lineage/openai2024o1  ->  https://openai.com/index/learning-to-reason-with-llms/"   # not a direct PDF; open in a browser
get "1_overview_and_lineage" "nye2021scratchpad.pdf" "https://arxiv.org/pdf/2112.00114"
get "1_overview_and_lineage" "zelikman2022star.pdf" "https://arxiv.org/pdf/2203.14465"
get "1_overview_and_lineage" "lambert2024tulu3.pdf" "https://arxiv.org/pdf/2411.15124"
get "1_overview_and_lineage" "deepseek2025r1.pdf" "https://arxiv.org/pdf/2501.12948"
get "1_overview_and_lineage" "xu2025lrmsurvey.pdf" "https://arxiv.org/pdf/2501.09686"
get "1_overview_and_lineage" "kumar2025posttraining.pdf" "https://arxiv.org/pdf/2502.21321"
echo "manual: 1_overview_and_lineage/weng2025think  ->  https://lilianweng.github.io/posts/2025-05-01-thinking/"   # not a direct PDF; open in a browser
echo "manual: 1_overview_and_lineage/raschka2025understanding  ->  https://magazine.sebastianraschka.com/p/understanding-reasoning-llms"   # not a direct PDF; open in a browser
get "1_overview_and_lineage" "lambert2025rlhfbook.pdf" "https://arxiv.org/pdf/2504.12501"
get "2_chains_of_thought_the_effect_the_theory_faithfulness" "wei2022cot.pdf" "https://arxiv.org/pdf/2201.11903"
get "2_chains_of_thought_the_effect_the_theory_faithfulness" "kojima2022zeroshot.pdf" "https://arxiv.org/pdf/2205.11916"
get "2_chains_of_thought_the_effect_the_theory_faithfulness" "wang2022selfconsistency.pdf" "https://arxiv.org/pdf/2203.11171"
get "2_chains_of_thought_the_effect_the_theory_faithfulness" "zhou2022leasttomost.pdf" "https://arxiv.org/pdf/2205.10625"
get "2_chains_of_thought_the_effect_the_theory_faithfulness" "madaan2022textpatterns.pdf" "https://arxiv.org/pdf/2209.07686"
get "2_chains_of_thought_the_effect_the_theory_faithfulness" "prystawski2023locality.pdf" "https://arxiv.org/pdf/2304.03843"
get "2_chains_of_thought_the_effect_the_theory_faithfulness" "feng2023mystery.pdf" "https://arxiv.org/pdf/2305.15408"
get "2_chains_of_thought_the_effect_the_theory_faithfulness" "merrill2023expressive.pdf" "https://arxiv.org/pdf/2310.07923"
get "2_chains_of_thought_the_effect_the_theory_faithfulness" "li2024serial.pdf" "https://arxiv.org/pdf/2402.12875"
get "2_chains_of_thought_the_effect_the_theory_faithfulness" "hao2024coconut.pdf" "https://arxiv.org/pdf/2412.06769"
get "2_chains_of_thought_the_effect_the_theory_faithfulness" "geiping2025recurrent.pdf" "https://arxiv.org/pdf/2502.05171"
get "2_chains_of_thought_the_effect_the_theory_faithfulness" "zelikman2024quietstar.pdf" "https://arxiv.org/pdf/2403.09629"
get "2_chains_of_thought_the_effect_the_theory_faithfulness" "turpin2023unfaithful.pdf" "https://arxiv.org/pdf/2305.04388"
get "2_chains_of_thought_the_effect_the_theory_faithfulness" "lanham2023faithfulness.pdf" "https://arxiv.org/pdf/2307.13702"
get "2_chains_of_thought_the_effect_the_theory_faithfulness" "arcuschin2025wild.pdf" "https://arxiv.org/pdf/2503.08679"
get "2_chains_of_thought_the_effect_the_theory_faithfulness" "chen2025dontsay.pdf" "https://arxiv.org/pdf/2505.05410"
get "3_verifiers_search_and_test-time_compute" "cobbe2021verifiers.pdf" "https://arxiv.org/pdf/2110.14168"
get "3_verifiers_search_and_test-time_compute" "uesato2022process.pdf" "https://arxiv.org/pdf/2211.14275"
get "3_verifiers_search_and_test-time_compute" "lightman2023verify.pdf" "https://arxiv.org/pdf/2305.20050"
get "3_verifiers_search_and_test-time_compute" "wang2023mathshepherd.pdf" "https://arxiv.org/pdf/2312.08935"
get "3_verifiers_search_and_test-time_compute" "zhang2025prmlessons.pdf" "https://arxiv.org/pdf/2501.07301"
get "3_verifiers_search_and_test-time_compute" "zhang2024genrm.pdf" "https://arxiv.org/pdf/2408.15240"
get "3_verifiers_search_and_test-time_compute" "yao2023tot.pdf" "https://arxiv.org/pdf/2305.10601"
get "3_verifiers_search_and_test-time_compute" "besta2023got.pdf" "https://arxiv.org/pdf/2308.09687"
get "3_verifiers_search_and_test-time_compute" "snell2024testtime.pdf" "https://arxiv.org/pdf/2408.03314"
get "3_verifiers_search_and_test-time_compute" "brown2024monkeys.pdf" "https://arxiv.org/pdf/2407.21787"
get "3_verifiers_search_and_test-time_compute" "guan2025rstar.pdf" "https://arxiv.org/pdf/2501.04519"
get "3_verifiers_search_and_test-time_compute" "openai2025competitive.pdf" "https://arxiv.org/pdf/2502.06807"
echo "manual: 3_verifiers_search_and_test-time_compute/hf2024testtime  ->  https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute"   # not a direct PDF; open in a browser
get "3_verifiers_search_and_test-time_compute" "liu2025computeoptimal.pdf" "https://arxiv.org/pdf/2502.06703"
get "3_verifiers_search_and_test-time_compute" "fu2025deepconf.pdf" "https://arxiv.org/pdf/2508.15260"
get "3_verifiers_search_and_test-time_compute" "wen2025parathinker.pdf" "https://arxiv.org/pdf/2509.04475"
get "3_verifiers_search_and_test-time_compute" "ghosal2025mirage.pdf" "https://arxiv.org/pdf/2506.04210"
get "3_verifiers_search_and_test-time_compute" "zhang2026reasoningtokens.pdf" "https://arxiv.org/pdf/2604.01302"
get "4_reinforcement_learning_on_verifiable_rewards_rlvr_r1_grpo_reward_design" "shao2024deepseekmath.pdf" "https://arxiv.org/pdf/2402.03300"
get "4_reinforcement_learning_on_verifiable_rewards_rlvr_r1_grpo_reward_design" "singh2023restem.pdf" "https://arxiv.org/pdf/2312.06585"
get "4_reinforcement_learning_on_verifiable_rewards_rlvr_r1_grpo_reward_design" "ahmadian2024rloo.pdf" "https://arxiv.org/pdf/2402.14740"
get "4_reinforcement_learning_on_verifiable_rewards_rlvr_r1_grpo_reward_design" "hu2025reinforcepp.pdf" "https://arxiv.org/pdf/2501.03262"
get "4_reinforcement_learning_on_verifiable_rewards_rlvr_r1_grpo_reward_design" "kimi2025k15.pdf" "https://arxiv.org/pdf/2501.12599"
get "4_reinforcement_learning_on_verifiable_rewards_rlvr_r1_grpo_reward_design" "yu2025dapo.pdf" "https://arxiv.org/pdf/2503.14476"
get "4_reinforcement_learning_on_verifiable_rewards_rlvr_r1_grpo_reward_design" "liu2025drgrpo.pdf" "https://arxiv.org/pdf/2503.20783"
get "4_reinforcement_learning_on_verifiable_rewards_rlvr_r1_grpo_reward_design" "yue2025vapo.pdf" "https://arxiv.org/pdf/2504.05118"
get "4_reinforcement_learning_on_verifiable_rewards_rlvr_r1_grpo_reward_design" "zheng2025gspo.pdf" "https://arxiv.org/pdf/2507.18071"
get "4_reinforcement_learning_on_verifiable_rewards_rlvr_r1_grpo_reward_design" "minimax2025m1.pdf" "https://arxiv.org/pdf/2506.13585"
get "4_reinforcement_learning_on_verifiable_rewards_rlvr_r1_grpo_reward_design" "magistral2025.pdf" "https://arxiv.org/pdf/2506.10910"
get "4_reinforcement_learning_on_verifiable_rewards_rlvr_r1_grpo_reward_design" "skalse2022rewardhacking.pdf" "https://arxiv.org/pdf/2209.13085"
get "4_reinforcement_learning_on_verifiable_rewards_rlvr_r1_grpo_reward_design" "huang2025rubrics.pdf" "https://arxiv.org/pdf/2508.12790"
get "4_reinforcement_learning_on_verifiable_rewards_rlvr_r1_grpo_reward_design" "zhao2025intuitor.pdf" "https://arxiv.org/pdf/2505.19590"
get "4_reinforcement_learning_on_verifiable_rewards_rlvr_r1_grpo_reward_design" "zhao2025absolutezero.pdf" "https://arxiv.org/pdf/2505.03335"
get "4_reinforcement_learning_on_verifiable_rewards_rlvr_r1_grpo_reward_design" "wei2025swerl.pdf" "https://arxiv.org/pdf/2502.18449"
get "4_reinforcement_learning_on_verifiable_rewards_rlvr_r1_grpo_reward_design" "feng2025retool.pdf" "https://arxiv.org/pdf/2504.11536"
get "4_reinforcement_learning_on_verifiable_rewards_rlvr_r1_grpo_reward_design" "gou2023tora.pdf" "https://arxiv.org/pdf/2309.17452"
get "4_reinforcement_learning_on_verifiable_rewards_rlvr_r1_grpo_reward_design" "helff2026gaming.pdf" "https://arxiv.org/pdf/2604.15149"
get "5_training_dynamics_what_rl_changes_and_what_was_already_there" "yue2025passk.pdf" "https://arxiv.org/pdf/2504.13837"
get "5_training_dynamics_what_rl_changes_and_what_was_already_there" "shao2025spurious.pdf" "https://arxiv.org/pdf/2506.10947"
get "5_training_dynamics_what_rl_changes_and_what_was_already_there" "wang2025oneexample.pdf" "https://arxiv.org/pdf/2504.20571"
get "5_training_dynamics_what_rl_changes_and_what_was_already_there" "liu2025prorl.pdf" "https://arxiv.org/pdf/2505.24864"
get "5_training_dynamics_what_rl_changes_and_what_was_already_there" "hu2025brorl.pdf" "https://arxiv.org/pdf/2510.01180"
get "5_training_dynamics_what_rl_changes_and_what_was_already_there" "cui2025entropy.pdf" "https://arxiv.org/pdf/2505.22617"
get "5_training_dynamics_what_rl_changes_and_what_was_already_there" "wang2025forktokens.pdf" "https://arxiv.org/pdf/2506.01939"
get "5_training_dynamics_what_rl_changes_and_what_was_already_there" "khatri2025scalerl.pdf" "https://arxiv.org/pdf/2510.13786"
get "5_training_dynamics_what_rl_changes_and_what_was_already_there" "tan2025rlscaling.pdf" "https://arxiv.org/pdf/2509.25300"
get "5_training_dynamics_what_rl_changes_and_what_was_already_there" "chen2025passktraining.pdf" "https://arxiv.org/pdf/2508.10751"
get "5_training_dynamics_what_rl_changes_and_what_was_already_there" "cheng2026isocompute.pdf" "https://arxiv.org/pdf/2603.12151"
get "5_training_dynamics_what_rl_changes_and_what_was_already_there" "zhai2026real.pdf" "https://arxiv.org/pdf/2602.05630"
get "6_distillation_and_data" "hsieh2023distilling.pdf" "https://arxiv.org/pdf/2305.02301"
get "6_distillation_and_data" "muennighoff2025s1.pdf" "https://arxiv.org/pdf/2501.19393"
get "6_distillation_and_data" "ye2025limo.pdf" "https://arxiv.org/pdf/2502.03387"
get "6_distillation_and_data" "guha2025openthoughts.pdf" "https://arxiv.org/pdf/2506.04178"
get "6_distillation_and_data" "moshkov2025openmath.pdf" "https://arxiv.org/pdf/2504.16891"
get "6_distillation_and_data" "wen2025lightr1.pdf" "https://arxiv.org/pdf/2503.10460"
get "6_distillation_and_data" "chen2025acereason.pdf" "https://arxiv.org/pdf/2505.16400"
echo "manual: 6_distillation_and_data/hf2025openr1  ->  https://huggingface.co/blog/open-r1"   # not a direct PDF; open in a browser
echo "manual: 6_distillation_and_data/numina2024  ->  https://huggingface.co/collections/AI-MO/numinamath"   # not a direct PDF; open in a browser
echo "manual: 6_distillation_and_data/novasky2025skyt1  ->  https://novasky-ai.github.io/posts/sky-t1/"   # not a direct PDF; open in a browser
echo "manual: 6_distillation_and_data/bespoke2025stratos  ->  https://www.bespokelabs.ai/blog/bespoke-stratos-the-unreasonable-effectiveness-of-reasoning-distillation"   # not a direct PDF; open in a browser
echo "manual: 6_distillation_and_data/agentica2025deepscaler  ->  https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview"   # not a direct PDF; open in a browser
get "6_distillation_and_data" "qwen2025qwen3.pdf" "https://arxiv.org/pdf/2505.09388"
get "6_distillation_and_data" "openai2025gptoss.pdf" "https://arxiv.org/pdf/2508.10925"
get "6_distillation_and_data" "abdin2025phi4r.pdf" "https://arxiv.org/pdf/2504.21318"
get "6_distillation_and_data" "bercovich2025nemotron.pdf" "https://arxiv.org/pdf/2505.00949"
get "6_distillation_and_data" "zeng2025glm45.pdf" "https://arxiv.org/pdf/2508.06471"
get "6_distillation_and_data" "kimi2025k2.pdf" "https://arxiv.org/pdf/2507.20534"
get "7_inference_budgets_length_control_parallel_compute_tools_serving" "chen2024overthinking.pdf" "https://arxiv.org/pdf/2412.21187"
get "7_inference_budgets_length_control_parallel_compute_tools_serving" "wang2025underthinking.pdf" "https://arxiv.org/pdf/2501.18585"
get "7_inference_budgets_length_control_parallel_compute_tools_serving" "aggarwal2025l1.pdf" "https://arxiv.org/pdf/2503.04697"
get "7_inference_budgets_length_control_parallel_compute_tools_serving" "hou2025thinkprune.pdf" "https://arxiv.org/pdf/2504.01296"
get "7_inference_budgets_length_control_parallel_compute_tools_serving" "ma2025nothinking.pdf" "https://arxiv.org/pdf/2504.09858"
get "7_inference_budgets_length_control_parallel_compute_tools_serving" "zhang2025adaptthink.pdf" "https://arxiv.org/pdf/2505.13417"
get "7_inference_budgets_length_control_parallel_compute_tools_serving" "fang2025thinkless.pdf" "https://arxiv.org/pdf/2505.13379"
get "7_inference_budgets_length_control_parallel_compute_tools_serving" "cuadron2025danger.pdf" "https://arxiv.org/pdf/2502.08235"
get "7_inference_budgets_length_control_parallel_compute_tools_serving" "aghajohari2025markovian.pdf" "https://arxiv.org/pdf/2510.06557"
get "7_inference_budgets_length_control_parallel_compute_tools_serving" "sui2025stopoverthinking.pdf" "https://arxiv.org/pdf/2503.16419"
get "7_inference_budgets_length_control_parallel_compute_tools_serving" "shen2025thinkingdoing.pdf" "https://arxiv.org/pdf/2506.07976"
get "7_inference_budgets_length_control_parallel_compute_tools_serving" "zheng2025parallelr1.pdf" "https://arxiv.org/pdf/2509.07980"
get "7_inference_budgets_length_control_parallel_compute_tools_serving" "wang2025scalingoverscaling.pdf" "https://arxiv.org/pdf/2505.20522"
get "7_inference_budgets_length_control_parallel_compute_tools_serving" "gou2023critic.pdf" "https://arxiv.org/pdf/2305.11738"
echo "manual: 8_the_landscape_in_october_2026/openai2026docs  ->  https://platform.openai.com/docs/guides/reasoning"   # not a direct PDF; open in a browser
echo "manual: 8_the_landscape_in_october_2026/anthropic2026thinking  ->  https://platform.claude.com/docs/en/build-with-claude/thinking"   # not a direct PDF; open in a browser
echo "manual: 8_the_landscape_in_october_2026/google2026thinking  ->  https://ai.google.dev/gemini-api/docs/thinking"   # not a direct PDF; open in a browser
echo "manual: 8_the_landscape_in_october_2026/deepseek2026v4  ->  https://api-docs.deepseek.com/"   # not a direct PDF; open in a browser
echo "manual: 8_the_landscape_in_october_2026/qwen2026qwen38  ->  https://huggingface.co/Qwen"   # not a direct PDF; open in a browser
echo "manual: 8_the_landscape_in_october_2026/moonshot2026k3  ->  https://huggingface.co/moonshotai"   # not a direct PDF; open in a browser
echo "manual: 8_the_landscape_in_october_2026/openai2025ioi  ->  https://x.com/mostafarohani/status/1968361152741826849"   # not a direct PDF; open in a browser
echo "manual: 8_the_landscape_in_october_2026/gdm2025imo  ->  https://deepmind.google/discover/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/"   # not a direct PDF; open in a browser
get "8_the_landscape_in_october_2026" "balunovic2025matharena.pdf" "https://arxiv.org/pdf/2505.23281"
get "8_the_landscape_in_october_2026" "chollet2025arcagi2.pdf" "https://arxiv.org/pdf/2505.11831"
get "8_the_landscape_in_october_2026" "phan2025hle.pdf" "https://arxiv.org/pdf/2501.14249"
echo "manual: 8_the_landscape_in_october_2026/tinyzero2025  ->  https://github.com/Jiayi-Pan/TinyZero"   # not a direct PDF; open in a browser
echo "manual: 8_the_landscape_in_october_2026/verl2025  ->  https://github.com/volcengine/verl"   # not a direct PDF; open in a browser
get "9_limits_evaluation_and_safety" "mirzadeh2024gsmsymbolic.pdf" "https://arxiv.org/pdf/2410.05229"
get "9_limits_evaluation_and_safety" "zhang2024gsm1k.pdf" "https://arxiv.org/pdf/2405.00332"
get "9_limits_evaluation_and_safety" "shojaee2025illusion.pdf" "https://arxiv.org/pdf/2506.06941"
get "9_limits_evaluation_and_safety" "lawsen2025comment.pdf" "https://arxiv.org/pdf/2506.09250"
get "9_limits_evaluation_and_safety" "hendrycks2021math.pdf" "https://arxiv.org/pdf/2103.03874"
get "9_limits_evaluation_and_safety" "rein2023gpqa.pdf" "https://arxiv.org/pdf/2311.12022"
get "9_limits_evaluation_and_safety" "glazer2024frontiermath.pdf" "https://arxiv.org/pdf/2411.04872"
get "9_limits_evaluation_and_safety" "jain2024livecodebench.pdf" "https://arxiv.org/pdf/2403.07974"
echo "manual: 9_limits_evaluation_and_safety/openai2024swebenchverified  ->  https://openai.com/index/introducing-swe-bench-verified/"   # not a direct PDF; open in a browser
get "9_limits_evaluation_and_safety" "yoon2025confidence.pdf" "https://arxiv.org/pdf/2505.14489"
get "9_limits_evaluation_and_safety" "mei2025uncertainty.pdf" "https://arxiv.org/pdf/2506.18183"
get "9_limits_evaluation_and_safety" "zeng2025thinkingoutloud.pdf" "https://arxiv.org/pdf/2504.06564"
get "9_limits_evaluation_and_safety" "baker2025monitoring.pdf" "https://arxiv.org/pdf/2503.11926"
get "9_limits_evaluation_and_safety" "korbak2025monitorability.pdf" "https://arxiv.org/pdf/2507.11473"
get "9_limits_evaluation_and_safety" "guan2025monitoringmonitorability.pdf" "https://arxiv.org/pdf/2512.18311"
echo "manual: 9_limits_evaluation_and_safety/openai2025o3card  ->  https://openai.com/index/o3-o4-mini-system-card/"   # not a direct PDF; open in a browser
echo "manual: 9_limits_evaluation_and_safety/openai2026cotcontrol  ->  https://openai.com/index/reasoning-models-chain-of-thought-controllability/"   # not a direct PDF; open in a browser
echo "manual: 9_limits_evaluation_and_safety/openai2026astracard  ->  https://deploymentsafety.openai.com/gpt-6-astra"   # not a direct PDF; open in a browser
get "10_practice_summary_and_surveys" "sun2023foundationsurvey.pdf" "https://arxiv.org/pdf/2312.11562"
get "10_practice_summary_and_surveys" "zhu2025concisesurvey.pdf" "https://arxiv.org/pdf/2507.09662"
echo "manual: 10_practice_summary_and_surveys/raschka2025state  ->  https://magazine.sebastianraschka.com/p/state-of-llm-reasoning-and-inference-scaling"   # not a direct PDF; open in a browser
get "10_practice_summary_and_surveys" "hariri2026ttsregimes.pdf" "https://arxiv.org/pdf/2608.04001"
echo "manual: 10_practice_summary_and_surveys/openai2025gpt5  ->  https://openai.com/index/introducing-gpt-5/"   # not a direct PDF; open in a browser
echo "manual: 10_practice_summary_and_surveys/scientificamerican2025r1cost  ->  https://www.scientificamerican.com/article/secrets-of-chinese-ai-model-deepseek-revealed-in-landmark-paper/"   # not a direct PDF; open in a browser

cp "$(dirname "$0")/reading_list.csv" "$(dirname "$0")/reading_list.bib" "$(dirname "$0")/README.md" "$ROOT/" 2>/dev/null
echo "downloaded: $ok   failed: $fail"
( cd "$(dirname "$ROOT")" && rm -f reasoning_models_papers.zip && zip -qr reasoning_models_papers.zip "$(basename "$ROOT")" ) && echo "wrote $(dirname "$ROOT")/reasoning_models_papers.zip"
