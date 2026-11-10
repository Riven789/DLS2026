#!/bin/bash
# Downloads the reading list for 'Decision Models' from arXiv and zips it.
# Usage:  bash fetch_papers.sh          (needs curl and zip; ~79 PDFs; takes a few minutes)
# arXiv asks for no more than one request every few seconds from scripts; the sleep below respects that.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)/decision_models_papers"
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

echo "manual: 1_overview_and_lineage/typesafe_systemone  ->  https://docs.typesafe.ai/concepts/system-one"   # not a direct PDF; open in a browser
echo "manual: 1_overview_and_lineage/typesafe_primer  ->  https://docs.typesafe.ai/introduction/machine-learning-primer"   # not a direct PDF; open in a browser
echo "manual: 1_overview_and_lineage/kahneman2011  ->  https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow"   # not a direct PDF; open in a browser
get "1_overview_and_lineage" "booch2020thinking.pdf" "https://arxiv.org/pdf/2010.06002"
get "1_overview_and_lineage" "yu2024distilling.pdf" "https://arxiv.org/pdf/2407.06023"
get "1_overview_and_lineage" "niculescu2005predicting.pdf" "https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf"
get "1_overview_and_lineage" "hendrycks2020mmlu.pdf" "https://arxiv.org/pdf/2009.03300"
get "1_overview_and_lineage" "ouyang2022instructgpt.pdf" "https://arxiv.org/pdf/2203.02155"
get "1_overview_and_lineage" "zheng2023judge.pdf" "https://arxiv.org/pdf/2306.05685"
get "1_overview_and_lineage" "kalai2025hallucinate.pdf" "https://arxiv.org/pdf/2509.04664"
echo "manual: 2_primitives_state_and_the_api/typesafe_primitives  ->  https://docs.typesafe.ai/primitives"   # not a direct PDF; open in a browser
echo "manual: 2_primitives_state_and_the_api/typesafe_confidence  ->  https://docs.typesafe.ai/confidence"   # not a direct PDF; open in a browser
echo "manual: 2_primitives_state_and_the_api/typesafe_state  ->  https://docs.typesafe.ai/concepts/state"   # not a direct PDF; open in a browser
echo "manual: 2_primitives_state_and_the_api/typesafe_howto  ->  https://docs.typesafe.ai/concepts/how-to-build-with-system-one"   # not a direct PDF; open in a browser
echo "manual: 2_primitives_state_and_the_api/typesafe_api  ->  https://docs.typesafe.ai/api/system-one"   # not a direct PDF; open in a browser
echo "manual: 2_primitives_state_and_the_api/typesafe_parallel  ->  https://docs.typesafe.ai/cookbooks/parallel_questions"   # not a direct PDF; open in a browser
echo "manual: 2_primitives_state_and_the_api/openrouter_decisions  ->  https://openrouter.ai/docs/guides/community/jev"   # not a direct PDF; open in a browser
echo "manual: 2_primitives_state_and_the_api/openai_decisions  ->  https://developers.openai.com/api/docs/guides/decisions"   # not a direct PDF; open in a browser
get "2_primitives_state_and_the_api" "cheng2023batch.pdf" "https://arxiv.org/pdf/2301.08721"
get "2_primitives_state_and_the_api" "son2024multitask.pdf" "https://arxiv.org/pdf/2402.11597"
get "2_primitives_state_and_the_api" "liu2023geval.pdf" "https://arxiv.org/pdf/2303.16634"
get "2_primitives_state_and_the_api" "robinson2022mcqa.pdf" "https://arxiv.org/pdf/2210.12353"
get "3_calibration_and_proper_scoring_rules" "guo2017calibration.pdf" "https://arxiv.org/pdf/1706.04599"
echo "manual: 3_calibration_and_proper_scoring_rules/gneiting2007proper  ->  https://doi.org/10.1198/016214506000001437"   # not a direct PDF; open in a browser
echo "manual: 3_calibration_and_proper_scoring_rules/degroot1983forecasters  ->  https://doi.org/10.2307/2987588"   # not a direct PDF; open in a browser
echo "manual: 3_calibration_and_proper_scoring_rules/platt1999probabilistic  ->  https://www.researchgate.net/publication/2594015"   # not a direct PDF; open in a browser
get "3_calibration_and_proper_scoring_rules" "nixon2019measuring.pdf" "https://arxiv.org/pdf/1904.01685"
get "3_calibration_and_proper_scoring_rules" "kumar2019verified.pdf" "https://arxiv.org/pdf/1909.10155"
get "3_calibration_and_proper_scoring_rules" "kull2019dirichlet.pdf" "https://arxiv.org/pdf/1910.12656"
get "3_calibration_and_proper_scoring_rules" "minderer2021revisiting.pdf" "https://arxiv.org/pdf/2106.07998"
get "3_calibration_and_proper_scoring_rules" "blasiok2022unifying.pdf" "https://arxiv.org/pdf/2211.16886"
get "3_calibration_and_proper_scoring_rules" "ovadia2019trust.pdf" "https://arxiv.org/pdf/1906.02530"
get "3_calibration_and_proper_scoring_rules" "zhao2021calibrate.pdf" "https://arxiv.org/pdf/2102.09690"
get "3_calibration_and_proper_scoring_rules" "kadavath2022know.pdf" "https://arxiv.org/pdf/2207.05221"
get "3_calibration_and_proper_scoring_rules" "openai2023gpt4.pdf" "https://arxiv.org/pdf/2303.08774"
get "3_calibration_and_proper_scoring_rules" "lin2022teaching.pdf" "https://arxiv.org/pdf/2205.14334"
get "3_calibration_and_proper_scoring_rules" "tian2023justask.pdf" "https://arxiv.org/pdf/2305.14975"
get "3_calibration_and_proper_scoring_rules" "xiong2023express.pdf" "https://arxiv.org/pdf/2306.13063"
get "3_calibration_and_proper_scoring_rules" "zhu2023alignment.pdf" "https://arxiv.org/pdf/2311.13240"
get "3_calibration_and_proper_scoring_rules" "kapoor2024taught.pdf" "https://arxiv.org/pdf/2406.08391"
get "3_calibration_and_proper_scoring_rules" "xiao2025restoring.pdf" "https://arxiv.org/pdf/2505.01997"
get "3_calibration_and_proper_scoring_rules" "sanz2026calibration.pdf" "https://arxiv.org/pdf/2609.26489"
get "3_calibration_and_proper_scoring_rules" "wang2024myanswer.pdf" "https://arxiv.org/pdf/2402.14499"
get "4_training_for_calibration" "yang2023rlcd.pdf" "https://arxiv.org/pdf/2307.12950"
get "4_training_for_calibration" "mielke2022linguistic.pdf" "https://arxiv.org/pdf/2012.14983"
get "4_training_for_calibration" "band2024linguistic.pdf" "https://arxiv.org/pdf/2404.00474"
get "4_training_for_calibration" "stengel2024lacie.pdf" "https://arxiv.org/pdf/2405.21028"
get "4_training_for_calibration" "xu2024sayself.pdf" "https://arxiv.org/pdf/2405.20974"
get "4_training_for_calibration" "leng2024taming.pdf" "https://arxiv.org/pdf/2410.09724"
get "4_training_for_calibration" "baniharouni2025doubt.pdf" "https://arxiv.org/pdf/2503.02623"
get "4_training_for_calibration" "damani2025rlcr.pdf" "https://arxiv.org/pdf/2507.16806"
get "4_training_for_calibration" "yoon2025reasoning.pdf" "https://arxiv.org/pdf/2505.14489"
get "4_training_for_calibration" "mei2025uncertainty.pdf" "https://arxiv.org/pdf/2506.18183"
get "4_training_for_calibration" "zeng2025thinking.pdf" "https://arxiv.org/pdf/2504.06564"
get "4_training_for_calibration" "kirichenko2025abstention.pdf" "https://arxiv.org/pdf/2506.09038"
get "4_training_for_calibration" "hinton2015distilling.pdf" "https://arxiv.org/pdf/1503.02531"
get "4_training_for_calibration" "hsieh2023distilling.pdf" "https://arxiv.org/pdf/2305.02301"
get "4_training_for_calibration" "pangakis2024distillation.pdf" "https://arxiv.org/pdf/2406.17633"
echo "manual: 5_selective_prediction_thresholds_conformal_sets_cascades/elyaniv2010selective  ->  https://jmlr.org/papers/v11/el-yaniv10a.html"   # not a direct PDF; open in a browser
get "5_selective_prediction_thresholds_conformal_sets_cascades" "geifman2017selective.pdf" "https://arxiv.org/pdf/1705.08500"
get "5_selective_prediction_thresholds_conformal_sets_cascades" "geifman2019bias.pdf" "https://arxiv.org/pdf/1805.08206"
get "5_selective_prediction_thresholds_conformal_sets_cascades" "hendrycks2016baseline.pdf" "https://arxiv.org/pdf/1610.02136"
get "5_selective_prediction_thresholds_conformal_sets_cascades" "madras2018defer.pdf" "https://arxiv.org/pdf/1711.06664"
get "5_selective_prediction_thresholds_conformal_sets_cascades" "mozannar2020defer.pdf" "https://arxiv.org/pdf/2006.01862"
get "5_selective_prediction_thresholds_conformal_sets_cascades" "kamath2020selective.pdf" "https://arxiv.org/pdf/2006.09462"
get "5_selective_prediction_thresholds_conformal_sets_cascades" "elkan2001cost.pdf" "https://cseweb.ucsd.edu/~elkan/rescale.pdf"
echo "manual: 5_selective_prediction_thresholds_conformal_sets_cascades/vickers2006decision  ->  https://doi.org/10.1177/0272989X06295361"   # not a direct PDF; open in a browser
get "5_selective_prediction_thresholds_conformal_sets_cascades" "angelopoulos2021gentle.pdf" "https://arxiv.org/pdf/2107.07511"
get "5_selective_prediction_thresholds_conformal_sets_cascades" "kumar2023conformal.pdf" "https://arxiv.org/pdf/2305.18404"
get "5_selective_prediction_thresholds_conformal_sets_cascades" "quach2023conformal.pdf" "https://arxiv.org/pdf/2306.10193"
get "5_selective_prediction_thresholds_conformal_sets_cascades" "ren2023knowno.pdf" "https://arxiv.org/pdf/2307.01928"
get "5_selective_prediction_thresholds_conformal_sets_cascades" "mohri2024factuality.pdf" "https://arxiv.org/pdf/2402.10978"
get "5_selective_prediction_thresholds_conformal_sets_cascades" "chen2023frugalgpt.pdf" "https://arxiv.org/pdf/2305.05176"
get "5_selective_prediction_thresholds_conformal_sets_cascades" "jitkrittum2023deferral.pdf" "https://arxiv.org/pdf/2307.02764"
get "5_selective_prediction_thresholds_conformal_sets_cascades" "yue2023cascades.pdf" "https://arxiv.org/pdf/2310.03094"
get "5_selective_prediction_thresholds_conformal_sets_cascades" "ong2024routellm.pdf" "https://arxiv.org/pdf/2406.18665"
get "5_selective_prediction_thresholds_conformal_sets_cascades" "gupta2024cascades.pdf" "https://arxiv.org/pdf/2404.10136"
get "5_selective_prediction_thresholds_conformal_sets_cascades" "chuang2024selfref.pdf" "https://arxiv.org/pdf/2410.13284"
echo "manual: 5_selective_prediction_thresholds_conformal_sets_cascades/typesafe_cascade  ->  https://docs.typesafe.ai/cookbooks/sde_cascade"   # not a direct PDF; open in a browser
echo "manual: 5_selective_prediction_thresholds_conformal_sets_cascades/typesafe_routing  ->  https://docs.typesafe.ai/patterns/confidence-routing"   # not a direct PDF; open in a browser
echo "manual: 6_the_landscape_and_open-weight_decision_models/typesafe_models  ->  https://docs.typesafe.ai/models"   # not a direct PDF; open in a browser
echo "manual: 6_the_landscape_and_open-weight_decision_models/cloudflare_clef  ->  https://blog.cloudflare.com/clef-decision-models/"   # not a direct PDF; open in a browser
echo "manual: 6_the_landscape_and_open-weight_decision_models/perplexity_decider  ->  https://huggingface.co/perplexity-ai/pplx-decider-v1.2"   # not a direct PDF; open in a browser
echo "manual: 6_the_landscape_and_open-weight_decision_models/palmer_kev  ->  https://github.com/jaredpalmer/kev/blob/main/README.md"   # not a direct PDF; open in a browser
echo "manual: 6_the_landscape_and_open-weight_decision_models/laya_card  ->  https://huggingface.co/convaiinnovations/laya"   # not a direct PDF; open in a browser
echo "manual: 6_the_landscape_and_open-weight_decision_models/ggml_decisions  ->  https://huggingface.co/blog/ggml-org/decision-models-in-llamacpp"   # not a direct PDF; open in a browser
echo "manual: 6_the_landscape_and_open-weight_decision_models/microsoft_decision1  ->  https://commandline.microsoft.com/microsoft-decision-1-model-foundry/"   # not a direct PDF; open in a browser
echo "manual: 6_the_landscape_and_open-weight_decision_models/liquid_d1  ->  https://www.liquid.ai/blog/d1-ultra"   # not a direct PDF; open in a browser
get "6_the_landscape_and_open-weight_decision_models" "li2026llmasjev.pdf" "https://arxiv.org/pdf/2610.02076"
get "6_the_landscape_and_open-weight_decision_models" "warner2024modernbert.pdf" "https://arxiv.org/pdf/2412.13663"
get "6_the_landscape_and_open-weight_decision_models" "tunstall2022setfit.pdf" "https://arxiv.org/pdf/2209.11055"
echo "manual: 6_the_landscape_and_open-weight_decision_models/raschka2026classifier  ->  https://magazine.sebastianraschka.com/p/classifier-history-and-jev"   # not a direct PDF; open in a browser
echo "manual: 7_limits_jaggedness_order_injection_distribution/typesafe_jagged  ->  https://docs.typesafe.ai/model-jaggedness/jev-1.13"   # not a direct PDF; open in a browser
get "7_limits_jaggedness_order_injection_distribution" "zheng2023selectors.pdf" "https://arxiv.org/pdf/2309.03882"
get "7_limits_jaggedness_order_injection_distribution" "pezeshkpour2023order.pdf" "https://arxiv.org/pdf/2308.11483"
get "7_limits_jaggedness_order_injection_distribution" "wang2023fair.pdf" "https://arxiv.org/pdf/2305.17926"
get "7_limits_jaggedness_order_injection_distribution" "xu2026jevout.pdf" "https://arxiv.org/pdf/2609.30243"
get "7_limits_jaggedness_order_injection_distribution" "hu2026jevadvbench.pdf" "https://arxiv.org/pdf/2609.31142"
echo "manual: 7_limits_jaggedness_order_injection_distribution/anaconda2026decision  ->  https://www.anaconda.com/blog/decision-time"   # not a direct PDF; open in a browser
get "7_limits_jaggedness_order_injection_distribution" "typedbench2026.pdf" "https://arxiv.org/pdf/2610.11392"
echo "manual: 7_limits_jaggedness_order_injection_distribution/molas2026calibrated  ->  https://alexmolas.com/2026/09/23/jev-cant-be-calibrated.html"   # not a direct PDF; open in a browser
get "8_evaluation_and_the_independent_evidence" "rafe2026crash.pdf" "https://arxiv.org/pdf/2609.24052"
get "8_evaluation_and_the_independent_evidence" "guo2026justask.pdf" "https://arxiv.org/pdf/2609.29429"
get "8_evaluation_and_the_independent_evidence" "wu2026reflex.pdf" "https://arxiv.org/pdf/2609.26532"
echo "manual: 8_evaluation_and_the_independent_evidence/openrouter_banking77  ->  https://openrouter.ai/blog/insights/jev-vs-claude-opus-5-classification/"   # not a direct PDF; open in a browser
echo "manual: 8_evaluation_and_the_independent_evidence/openrouter_judge  ->  https://openrouter.ai/blog/tutorials/jev-vs-llm-as-a-judge/"   # not a direct PDF; open in a browser
echo "manual: 8_evaluation_and_the_independent_evidence/archestra2026agent  ->  https://archestra.ai/blog/we-tested-jev-on-100-real-agent-calls"   # not a direct PDF; open in a browser
echo "manual: 8_evaluation_and_the_independent_evidence/dibia2026howjev  ->  https://newsletter.victordibia.com/p/how-jev-works-calibrated-decision"   # not a direct PDF; open in a browser
get "8_evaluation_and_the_independent_evidence" "lambert2024rewardbench.pdf" "https://arxiv.org/pdf/2403.13787"
get "8_evaluation_and_the_independent_evidence" "galil2023imagenet.pdf" "https://arxiv.org/pdf/2302.11874"
get "8_evaluation_and_the_independent_evidence" "gilardi2023chatgpt.pdf" "https://arxiv.org/pdf/2303.15056"
echo "manual: 9_practice_agents_questions_building_your_own/openrouter_autoapprove  ->  https://openrouter.ai/docs/cookbook/coding-agents/auto-approve-permission-prompts-with-jev"   # not a direct PDF; open in a browser
echo "manual: 9_practice_agents_questions_building_your_own/typesafe_coding  ->  https://docs.typesafe.ai/introduction/coding-agents"   # not a direct PDF; open in a browser
echo "manual: 9_practice_agents_questions_building_your_own/langchain_harness  ->  https://www.langchain.com/blog/building-a-harness-with-jev"   # not a direct PDF; open in a browser
echo "manual: 9_practice_agents_questions_building_your_own/typesafe_guardrails  ->  https://docs.typesafe.ai/cookbooks/llm_guardrails"   # not a direct PDF; open in a browser
echo "manual: 9_practice_agents_questions_building_your_own/typesafe_hier  ->  https://docs.typesafe.ai/cookbooks/hierarchical_classification"   # not a direct PDF; open in a browser
get "9_practice_agents_questions_building_your_own" "inan2023llamaguard.pdf" "https://arxiv.org/pdf/2312.06674"
get "9_practice_agents_questions_building_your_own" "bucher2024finetuned.pdf" "https://arxiv.org/pdf/2406.08660"

cp "$(dirname "$0")/reading_list.csv" "$(dirname "$0")/reading_list.bib" "$(dirname "$0")/README.md" "$ROOT/" 2>/dev/null
echo "downloaded: $ok   failed: $fail"
( cd "$(dirname "$ROOT")" && rm -f decision_models_papers.zip && zip -qr decision_models_papers.zip "$(basename "$ROOT")" ) && echo "wrote $(dirname "$ROOT")/decision_models_papers.zip"
