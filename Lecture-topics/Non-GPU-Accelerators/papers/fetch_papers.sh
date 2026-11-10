#!/bin/bash
# Downloads the reading list for 'Non-GPU AI Accelerators' from arXiv and zips it.
# Usage:  bash fetch_papers.sh          (needs curl and zip; ~38 PDFs; takes a few minutes)
# arXiv asks for no more than one request every few seconds from scripts; the sleep below respects that.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)/non_gpu_accelerators_papers"
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

get "1_overview_a_decade_of_alternatives" "reuther2022survey.pdf" "https://arxiv.org/pdf/2210.04055"
get "1_overview_a_decade_of_alternatives" "reuther2021survey.pdf" "https://arxiv.org/pdf/2109.08957"
echo "manual: 1_overview_a_decade_of_alternatives/hennessy2019golden  ->  https://cacm.acm.org/?p=7368"   # not a direct PDF; open in a browser
get "1_overview_a_decade_of_alternatives" "sastry2026atpesc.pdf" "https://extremecomputingtraining.anl.gov/sites/atpesc/files/2026-08/AI%20Testbeds.pdf"
echo "manual: 1_overview_a_decade_of_alternatives/alcf2026testbed  ->  https://ai.alcf.anl.gov"   # not a direct PDF; open in a browser
echo "manual: 1_overview_a_decade_of_alternatives/hotchips2026program  ->  https://hc2026.hotchips.org/program/conference/"   # not a direct PDF; open in a browser
echo "manual: 1_overview_a_decade_of_alternatives/sth2025hotchips  ->  https://www.servethehome.com/hot-chips-2025-preliminary-schedule-released/"   # not a direct PDF; open in a browser
echo "manual: 1_overview_a_decade_of_alternatives/register2026vc  ->  https://www.theregister.com/2026/02/25/ai_chips_vc_funding_1point1billion/"   # not a direct PDF; open in a browser
get "2_principles_the_memory_wall_dataflow_and_sram" "kung1982systolic.pdf" "https://www.cs.virginia.edu/~smk9u/CS6501F19/kung_-_1982_-_why_systolic_architectures.pdf"
get "2_principles_the_memory_wall_dataflow_and_sram" "horowitz2014energy.pdf" "https://www.Gwern.net/doc/cs/hardware/2014-horowitz-2.pdf"
get "2_principles_the_memory_wall_dataflow_and_sram" "dally2016berkeley.pdf" "https://berkeley-deep-learning.github.io/cs294-dl-f16/slides/DL_HW_Berkeley_0916.pdf"
get "2_principles_the_memory_wall_dataflow_and_sram" "dally2020dsa.pdf" "https://www.doc.ic.ac.uk/~wl/teachlocal/arch/papers/cacm20dsa.pdf"
get "2_principles_the_memory_wall_dataflow_and_sram" "gholami2024memwall.pdf" "https://arxiv.org/pdf/2403.14123"
echo "manual: 2_principles_the_memory_wall_dataflow_and_sram/valiant1990bsp  ->  https://www.osti.gov/biblio/6502724"   # not a direct PDF; open in a browser
get "2_principles_the_memory_wall_dataflow_and_sram" "chen2016eyeriss.pdf" "https://sands.kaust.edu.sa/classes/CS290E/F19/papers/eyeriss.pdf"
get "2_principles_the_memory_wall_dataflow_and_sram" "sze2017survey.pdf" "https://arxiv.org/pdf/1703.09039"
get "2_principles_the_memory_wall_dataflow_and_sram" "prabhakar2017plasticine.pdf" "https://sands.kaust.edu.sa/classes/CS290E/F19/papers/plasticine.pdf"
echo "manual: 2_principles_the_memory_wall_dataflow_and_sram/fowers2018brainwave  ->  https://www.microsoft.com/en-us/research/publication/a-configurable-cloud-scale-dnn-processor-for-real-time-ai/"   # not a direct PDF; open in a browser
echo "manual: 2_principles_the_memory_wall_dataflow_and_sram/shilov2022sram  ->  https://tomshardware.com/news/no-sram-scaling-implies-on-more-expensive-cpus-and-gpus"   # not a direct PDF; open in a browser
echo "manual: 2_principles_the_memory_wall_dataflow_and_sram/epoch2026hbmcost  ->  https://epoch.ai/data-insights/ai-chip-component-cost-shares"   # not a direct PDF; open in a browser
get "3_the_google_tpu" "jouppi2017tpu.pdf" "https://arxiv.org/pdf/1704.04760"
get "3_the_google_tpu" "norrie2020hc32.pdf" "https://www.hc32.hotchips.org/assets/program/conference/day2/HotChips2020_ML_Training_Google_Norrie_Patil.v01.pdf"
get "3_the_google_tpu" "norrie2021micro.pdf" "https://www.Gwern.net/doc/ai/scaling/hardware/2021-norrie.pdf"
get "3_the_google_tpu" "jouppi2021tpuv4i.pdf" "https://www.cs.cmu.edu/~18742/papers/Jouppi2021.pdf"
get "3_the_google_tpu" "jouppi2023tpuv4.pdf" "https://arxiv.org/pdf/2304.01433"
get "3_the_google_tpu" "jouppi2026fivegen.pdf" "https://arxiv.org/pdf/2606.15870"
get "3_the_google_tpu" "jouppi2025ironwoodhc.pdf" "https://hc2025.hotchips.org/assets/program/conference/day2/61_Google_Ironwood-Final.pdf"
echo "manual: 3_the_google_tpu/googletpu7x  ->  https://docs.cloud.google.com/tpu/docs/tpu7x"   # not a direct PDF; open in a browser
echo "manual: 3_the_google_tpu/googletpuv6e  ->  https://docs.cloud.google.com/tpu/docs/v6e"   # not a direct PDF; open in a browser
echo "manual: 3_the_google_tpu/googletpuv5p  ->  https://docs.cloud.google.com/tpu/docs/v5p"   # not a direct PDF; open in a browser
echo "manual: 3_the_google_tpu/googletpuv5e  ->  https://docs.cloud.google.com/tpu/docs/v5e"   # not a direct PDF; open in a browser
echo "manual: 3_the_google_tpu/google2024trilliumga  ->  https://cloud.google.com/blog/products/compute/trillium-tpu-is-ga"   # not a direct PDF; open in a browser
echo "manual: 3_the_google_tpu/google2024trilliummlperf  ->  https://cloud.google.com/blog/products/compute/trillium-mlperf-41-training-benchmarks"   # not a direct PDF; open in a browser
echo "manual: 3_the_google_tpu/google2025ironwoodga  ->  https://cloud.google.com/blog/products/compute/ironwood-tpus-and-new-axion-based-vms-for-your-ai-workloads/"   # not a direct PDF; open in a browser
echo "manual: 3_the_google_tpu/storagereview2026tpu8  ->  https://www.storagereview.com/news/google-announces-tpu-8t-sunfish-and-tpu-8i-zebrafish"   # not a direct PDF; open in a browser
echo "manual: 4_cerebras_and_the_wafer/lie2023micro  ->  https://www.cerebras.ai/blog/cerebras-architecture-deep-dive-first-look-inside-the-hw-sw-co-design-for-deep-learning"   # not a direct PDF; open in a browser
get "4_cerebras_and_the_wafer" "rocki2020stencil.pdf" "https://arxiv.org/pdf/2010.03660"
echo "manual: 4_cerebras_and_the_wafer/cerebras2024wse3  ->  https://www.cerebras.ai/press-release/cerebras-announces-third-generation-wafer-scale-engine"   # not a direct PDF; open in a browser
echo "manual: 4_cerebras_and_the_wafer/morgan2024cs3  ->  https://www.nextplatform.com/ai/2024/03/14/cerebras-goes-hyperscale-with-third-gen-waferscale-supercomputers/1642584"   # not a direct PDF; open in a browser
echo "manual: 4_cerebras_and_the_wafer/cerebras2024cg3  ->  https://cerebras.ai/press-release/cerebras-g42-announce-condor-galaxy-3"   # not a direct PDF; open in a browser
echo "manual: 4_cerebras_and_the_wafer/cerebras2024s1  ->  https://www.sec.gov/Archives/edgar/data/2021728/000162828024041596/cerebras-sx1.htm"   # not a direct PDF; open in a browser
echo "manual: 4_cerebras_and_the_wafer/cerebras2024inference  ->  https://cerebras.ai/blog/introducing-cerebras-inference-ai-at-instant-speed"   # not a direct PDF; open in a browser
echo "manual: 4_cerebras_and_the_wafer/cerebras2024triples  ->  https://www.cerebras.ai/press-release/cerebras-triples-its-industry-leading-inference-performance-setting-new-all-time-record"   # not a direct PDF; open in a browser
echo "manual: 4_cerebras_and_the_wafer/futurum2026cerebrass1  ->  https://futurumgroup.com/insights/cerebras-s-1-teardown-is-the-23b-wafer-scale-ipo-the-end-of-gpu-homogeneity/"   # not a direct PDF; open in a browser
echo "manual: 4_cerebras_and_the_wafer/techcrunch2026cerebrasipo  ->  https://techcrunch.com/2026/05/14/cerebras-raises-5-5b-kicking-off-2026s-ipo-season-with-a-bang"   # not a direct PDF; open in a browser
echo "manual: 4_cerebras_and_the_wafer/cerebras2026q2  ->  https://www.sec.gov/Archives/edgar/data/0002021728/000162828026056186/cbrsannouncesfinancialresu.htm"   # not a direct PDF; open in a browser
echo "manual: 4_cerebras_and_the_wafer/cerebras2026cs4  ->  https://investors.cerebras.ai/node/7401/pdf"   # not a direct PDF; open in a browser
echo "manual: 4_cerebras_and_the_wafer/morgan2026cs4  ->  https://www.nextplatform.com/compute/2026/08/19/cerebras-overclocks-wse-3-waferscale-engine-to-boost-inference-oomph-in-nexus-cs-4/5289400"   # not a direct PDF; open in a browser
get "5_dataflow_and_streaming_machines" "abts2020tsp.pdf" "https://groq.com/wp-content/uploads/2024/02/2020-Isca.pdf"
get "5_dataflow_and_streaming_machines" "abts2022tsm.pdf" "https://groq.com/wp-content/uploads/2024/02/GroqISCAPaper2022_ASoftwareDefinedTensorStreamingMultiprocessorForLargeScaleMachineLearning.pdf"
echo "manual: 5_dataflow_and_streaming_machines/groq2024aa  ->  https://groq.com/newsroom/groq-lpu-inference-engine-leads-in-first-independent-llm-benchmark"   # not a direct PDF; open in a browser
echo "manual: 5_dataflow_and_streaming_machines/dcd2025nvidiagroq  ->  https://www.datacenterdynamics.com/en/news/nvidia-to-license-tech-from-ai-inference-chip-company-groq-hire-its-leadership/"   # not a direct PDF; open in a browser
echo "manual: 5_dataflow_and_streaming_machines/register2026lpx  ->  https://www.theregister.com/2026/03/16/nvidia_lpx_groq_3/"   # not a direct PDF; open in a browser
echo "manual: 5_dataflow_and_streaming_machines/tomshw2026lpx  ->  https://tomshardware.com/tech-industry/semiconductors/nvidia-presents-groq-3-lpx-architecture-and-unveils-its-first-third-party-inference-benchmark"   # not a direct PDF; open in a browser
get "5_dataflow_and_streaming_machines" "prabhakar2024sn40l.pdf" "https://arxiv.org/pdf/2405.07518"
get "5_dataflow_and_streaming_machines" "prabhakar2024hotchips.pdf" "https://www.hc2024.hotchips.org/assets/program/conference/day1/48_HC2024.Sambanova.Prabhakar.final-withoutvideo.pdf"
echo "manual: 5_dataflow_and_streaming_machines/sambanova2026sn50  ->  https://www.businesswire.com/news/home/20260224971025/en/"   # not a direct PDF; open in a browser
get "5_dataflow_and_streaming_machines" "jia2019ipu.pdf" "https://arxiv.org/pdf/1912.03413"
echo "manual: 5_dataflow_and_streaming_machines/wardfoxton2022bow  ->  https://www.eetasia.com/?p=36263"   # not a direct PDF; open in a browser
echo "manual: 5_dataflow_and_streaming_machines/register2023graphcore  ->  https://www.theregister.com/2023/10/06/brit_ai_chip_outfit_graphcore/"   # not a direct PDF; open in a browser
echo "manual: 5_dataflow_and_streaming_machines/register2024blackhole  ->  https://www.theregister.com/2024/08/27/tenstorrent_ai_blackhole/"   # not a direct PDF; open in a browser
echo "manual: 5_dataflow_and_streaming_machines/sth2025rebelquad  ->  https://www.servethehome.com/rebellions-rebel-quad-ucie-and-144gb-hbm3e-accelerator-at-hot-chips-2025/"   # not a direct PDF; open in a browser
echo "manual: 5_dataflow_and_streaming_machines/lam2024rngd  ->  https://chipsandcheese.com/p/furiosaais-rngd-at-hot-chips-2024-accelerating-ai-with-a-more-flexible-primitive"   # not a direct PDF; open in a browser
echo "manual: 6_the_hyperscalers'_chips_and_the_established_vendors/aws2023inferentia2  ->  https://aws.amazon.com/blogs/machine-learning/aws-inferentia2-builds-on-aws-inferentia1-by-delivering-4x-higher-throughput-and-10x-lower-latency/"   # not a direct PDF; open in a browser
echo "manual: 6_the_hyperscalers'_chips_and_the_established_vendors/barr2024trn2  ->  https://aws.amazon.com/blogs/aws/amazon-ec2-trn2-instances-and-trn2-ultraservers-for-aiml-training-and-inference-is-now-available"   # not a direct PDF; open in a browser
echo "manual: 6_the_hyperscalers'_chips_and_the_established_vendors/aws2025trn3  ->  https://aws.amazon.com/ec2/instance-types/trn3/"   # not a direct PDF; open in a browser
echo "manual: 6_the_hyperscalers'_chips_and_the_established_vendors/techcrunch2025trn3  ->  https://techcrunch.com/2025/12/02/amazon-releases-an-impressive-new-ai-chip-and-teases-a-nvidia-friendly-roadmap"   # not a direct PDF; open in a browser
echo "manual: 6_the_hyperscalers'_chips_and_the_established_vendors/sth2024maia100  ->  https://servethehome.com/microsoft-maia-100-ai-accelerator-for-azure"   # not a direct PDF; open in a browser
echo "manual: 6_the_hyperscalers'_chips_and_the_established_vendors/microsoft2026maia200  ->  https://techcommunity.microsoft.com/blog/azureinfrastructureblog/deep-dive-into-the-maia-200-architecture/4489312"   # not a direct PDF; open in a browser
echo "manual: 6_the_hyperscalers'_chips_and_the_established_vendors/meta2024mtiav2  ->  https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/"   # not a direct PDF; open in a browser
echo "manual: 6_the_hyperscalers'_chips_and_the_established_vendors/hpcwire2026mtia  ->  https://www.hpcwire.com/2026/03/11/meta-packs-gobs-of-hbm-into-homegrown-ai-accelerators/"   # not a direct PDF; open in a browser
echo "manual: 6_the_hyperscalers'_chips_and_the_established_vendors/tomshw2025openaibroadcom  ->  https://www.tomshardware.com/openai-broadcom-to-co-develop-10gw-of-custom-ai-chips"   # not a direct PDF; open in a browser
get "6_the_hyperscalers'_chips_and_the_established_vendors" "kaplan2024gaudi3.pdf" "https://www.hc2024.hotchips.org/assets/program/conference/day1/60_HC2024.Intel.RomanKaplan.Gaudi3-0826.pdf"
get "6_the_hyperscalers'_chips_and_the_established_vendors" "qualcomm2023ultra.pdf" "https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Prod-Brief-QCOM-Cloud-AI-100-Ultra.pdf"
echo "manual: 6_the_hyperscalers'_chips_and_the_established_vendors/sth2025qualcommai200  ->  https://www.servethehome.com/qualcomm-announces-new-integrated-ai-racks-with-768gb-cards-and-a-200mw-ai-deal/"   # not a direct PDF; open in a browser
echo "manual: 6_the_hyperscalers'_chips_and_the_established_vendors/sth2024spyre  ->  https://servethehome.com/ibm-telum-ii-processor-and-spyre-ai-updates-at-hot-chips-2024"   # not a direct PDF; open in a browser
get "6_the_hyperscalers'_chips_and_the_established_vendors" "talpes2022dojo.pdf" "https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips_tesla_dojo_uarch.pdf"
echo "manual: 6_the_hyperscalers'_chips_and_the_established_vendors/techcrunch2025dojo  ->  https://techcrunch.com/2025/09/02/tesla-dojo-the-rise-and-fall-of-elon-musks-ai-supercomputer/"   # not a direct PDF; open in a browser
echo "manual: 6_the_hyperscalers'_chips_and_the_established_vendors/techcrunch2026dojo3  ->  https://techcrunch.com/2026/01/20/elon-musk-says-teslas-restarted-dojo3-will-be-for-space-based-ai-compute"   # not a direct PDF; open in a browser
get "7_china_ascend_cloudmatrix_and_the_policy_backdrop" "liao2019davinci.pdf" "https://www.old.hotchips.org/hc31/HC31_1.11_Huawei.Davinci.HengLiao_v4.0.pdf"
echo "manual: 7_china_ascend_cloudmatrix_and_the_policy_backdrop/feldgoise2024cset  ->  https://cset.georgetown.edu/publication/pushing-the-limits-huaweis-ai-chip-tests-u-s-export-controls/"   # not a direct PDF; open in a browser
echo "manual: 7_china_ascend_cloudmatrix_and_the_policy_backdrop/patel2025cm384  ->  https://newsletter.semianalysis.com/p/huawei-ai-cloudmatrix-384-chinas-answer-to-nvidia-gb200-nvl72"   # not a direct PDF; open in a browser
get "7_china_ascend_cloudmatrix_and_the_policy_backdrop" "zuo2025cm384.pdf" "https://arxiv.org/pdf/2506.12708"
echo "manual: 7_china_ascend_cloudmatrix_and_the_policy_backdrop/huawei2025xukeynote  ->  https://www.huawei.com/en/news/2025/9/hc-xu-keynote-speech"   # not a direct PDF; open in a browser
echo "manual: 7_china_ascend_cloudmatrix_and_the_policy_backdrop/tomshw2026ascendroadmap  ->  https://www.tomshardware.com/tech-industry/artificial-intelligence/huawei-details-ai-accelerator-roadmap-pulls-in-next-generation-ascend-npus-by-quarters-fp4-performance-of-the-ascend-960pr-doubles-expectations"   # not a direct PDF; open in a browser
echo "manual: 7_china_ascend_cloudmatrix_and_the_policy_backdrop/register2025deepseekascend  ->  https://www.theregister.com/2025/08/14/dodgy_huawei_deepseek/"   # not a direct PDF; open in a browser
echo "manual: 7_china_ascend_cloudmatrix_and_the_policy_backdrop/recode2025chinachips  ->  https://recodechinaai.substack.com/p/who-will-fill-nvidias-ai-chip-void"   # not a direct PDF; open in a browser
echo "manual: 7_china_ascend_cloudmatrix_and_the_policy_backdrop/caixin2025kunlun  ->  https://www.caixinglobal.com/2025-11-14/baidu-unveils-ambitious-ai-chip-roadmap-targeting-1-million-card-cluster-by-2030-102382580.html"   # not a direct PDF; open in a browser
echo "manual: 7_china_ascend_cloudmatrix_and_the_policy_backdrop/trendforce2025thead  ->  https://www.trendforce.com/news/2025/09/17/news-alibaba-t-head-ppu-lands-china-unicom-deal-reportedly-matches-nvidia-h20-performance/"   # not a direct PDF; open in a browser
echo "manual: 7_china_ascend_cloudmatrix_and_the_policy_backdrop/trendforce2026cambricon  ->  https://www.trendforce.com/news/2026/03/09/news-china-gpu-race-intensifies-cambricon-turns-profitable-in-2025-moore-threads-metax-narrow-losses/"   # not a direct PDF; open in a browser
echo "manual: 7_china_ascend_cloudmatrix_and_the_policy_backdrop/nvidia2025h20  ->  https://www.sec.gov/Archives/edgar/data/1045810/000104581025000082/nvda-20250409.htm"   # not a direct PDF; open in a browser
echo "manual: 7_china_ascend_cloudmatrix_and_the_policy_backdrop/mofo2025diffusion  ->  https://www.mofo.com/resources/insights/250204-bis-issues-interim-final-rule"   # not a direct PDF; open in a browser
echo "manual: 7_china_ascend_cloudmatrix_and_the_policy_backdrop/kirkland2025rescission  ->  https://www.kirkland.com/publications/kirkland-alert/2025/05/bis-rescission-of-the-biden-administration"   # not a direct PDF; open in a browser
echo "manual: 7_china_ascend_cloudmatrix_and_the_policy_backdrop/ajc2025fifteenpercent  ->  https://www.ajc.com/news/2025/08/nvidia-amd-to-pay-15-of-china-chip-sale-revenue-to-us-government-2"   # not a direct PDF; open in a browser
echo "manual: 7_china_ascend_cloudmatrix_and_the_policy_backdrop/bloomberg2025rtxpro  ->  https://www.bloomberg.com/news/articles/2025-09-17/china-agency-orders-firms-to-stop-buying-nvidia-ai-chip-ft-says"   # not a direct PDF; open in a browser
echo "manual: 8_emerging_designs_in-memory_hardwired_photonic_neuromorphic/siliconangle2025dmatrix  ->  https://siliconangle.com/2025/11/12/chip-startup-d-matrix-raises-275m-speed-inference-memory-compute/"   # not a direct PDF; open in a browser
echo "manual: 8_emerging_designs_in-memory_hardwired_photonic_neuromorphic/sth2025dmatrixhc  ->  https://www.servethehome.com/d-matrix-presents-corsair-an-in-memory-computing-architecture-for-inference-at-hot-chips-2025/"   # not a direct PDF; open in a browser
get "8_emerging_designs_in-memory_hardwired_photonic_neuromorphic" "gomezluna2021upmem.pdf" "https://arxiv.org/pdf/2105.03814"
echo "manual: 8_emerging_designs_in-memory_hardwired_photonic_neuromorphic/techcrunch2023mythic  ->  https://techcrunch.com/2023/03/09/ai-chip-startup-mythic-rises-from-the-ashes-with-13m-new-ceo"   # not a direct PDF; open in a browser
echo "manual: 8_emerging_designs_in-memory_hardwired_photonic_neuromorphic/businesswire2025mythic  ->  https://www.businesswire.com/news/home/20251217546249/en/"   # not a direct PDF; open in a browser
echo "manual: 8_emerging_designs_in-memory_hardwired_photonic_neuromorphic/hpcwire2025untether  ->  https://www.hpcwire.com/off-the-wire/untether-ai-team-acquired-by-amd-product-support-discontinued/"   # not a direct PDF; open in a browser
echo "manual: 8_emerging_designs_in-memory_hardwired_photonic_neuromorphic/cnx2026taalas  ->  https://www.cnx-software.com/2026/02/22/taalas-hc1-hardwired-llama-3-1-8b-ai-accelerator-delivers-up-to-17000-tokens-s/"   # not a direct PDF; open in a browser
echo "manual: 8_emerging_designs_in-memory_hardwired_photonic_neuromorphic/nyu2026amdtaalas  ->  https://rits.shanghai.nyu.edu/ai/amd-acquires-taalas-the-startup-etching-llms-into-silicon/"   # not a direct PDF; open in a browser
echo "manual: 8_emerging_designs_in-memory_hardwired_photonic_neuromorphic/gadgetscout2026etched  ->  https://www.gadgetscout.co.uk/articles/etched-reaches-10-3bn-valuation-with-sk-hynix-backing.html"   # not a direct PDF; open in a browser
echo "manual: 8_emerging_designs_in-memory_hardwired_photonic_neuromorphic/aiweekly2026etched  ->  https://aiweekly.co/alerts/etched-doubles-to-21b-in-700m-round-led-by-jane-street"   # not a direct PDF; open in a browser
echo "manual: 8_emerging_designs_in-memory_hardwired_photonic_neuromorphic/implicator2026positron  ->  https://www.implicator.ai/positron-raises-875-million-at-5-billion-before-its-asimov-chip-tapes-out.md"   # not a direct PDF; open in a browser
echo "manual: 8_emerging_designs_in-memory_hardwired_photonic_neuromorphic/marvell2025celestial  ->  https://www.sec.gov/Archives/edgar/data/1835632/000119312525305289/d34367dex991.htm"   # not a direct PDF; open in a browser
echo "manual: 8_emerging_designs_in-memory_hardwired_photonic_neuromorphic/lightmatter2026llms  ->  https://lightmatter.co/llms.txt"   # not a direct PDF; open in a browser
echo "manual: 8_emerging_designs_in-memory_hardwired_photonic_neuromorphic/dcd2025lightmatter  ->  https://datacenterdynamics.com/en/news/lightmatter-unveils-m1000-and-l200-passage-photonic-interconnects"   # not a direct PDF; open in a browser
echo "manual: 8_emerging_designs_in-memory_hardwired_photonic_neuromorphic/techpowerup2024halapoint  ->  https://Techpowerup.com/321645/intel-builds-worlds-largest-neuromorphic-system-to-enable-more-sustainable-ai"   # not a direct PDF; open in a browser
get "8_emerging_designs_in-memory_hardwired_photonic_neuromorphic" "gonzalez2024spinnaker2.pdf" "https://arxiv.org/pdf/2401.04491"
echo "manual: 8_emerging_designs_in-memory_hardwired_photonic_neuromorphic/modha2023northpole  ->  https://modha.org/2023/10/northpole-neural-inference-at-the-frontier-of-energy-space-and-time"   # not a direct PDF; open in a browser
echo "manual: 8_emerging_designs_in-memory_hardwired_photonic_neuromorphic/modha2024northpolellm  ->  https://modha.org/2024/09/breakthrough-low-latency-high-energy-efficiency-llm-inference-performance-using-northpole"   # not a direct PDF; open in a browser
get "8_emerging_designs_in-memory_hardwired_photonic_neuromorphic" "debole2025northpole.pdf" "https://arxiv.org/pdf/2511.15950"
get "9_measured_specifications_mlperf_speed_and_deals" "peng2023ipurdu.pdf" "https://arxiv.org/pdf/2311.04417"
get "9_measured_specifications_mlperf_speed_and_deals" "emani2023llm.pdf" "https://arxiv.org/pdf/2310.04607"
get "9_measured_specifications_mlperf_speed_and_deals" "mattson2020mlperf.pdf" "https://arxiv.org/pdf/1910.01500"
echo "manual: 9_measured_specifications_mlperf_speed_and_deals/mlcommons_training_results  ->  https://github.com/mlcommons/training_results_v2.1"   # not a direct PDF; open in a browser
echo "manual: 9_measured_specifications_mlperf_speed_and_deals/mlcommons_inference_results  ->  https://github.com/mlcommons/inference_results_v4.0"   # not a direct PDF; open in a browser
echo "manual: 9_measured_specifications_mlperf_speed_and_deals/mlcommons2025training50  ->  https://mlcommons.org/2025/06/mlperf-training-v5-0-results"   # not a direct PDF; open in a browser
get "9_measured_specifications_mlperf_speed_and_deals" "mlcommons2026training60.pdf" "https://mlcommons.org/wp-content/uploads/2026/06/MLPerf-Training-v6.0-Supplemental-Discussion-UNDER-EMBARGO-UNTIL-6_16_26-8_00-AM-PT-1.pdf"
echo "manual: 9_measured_specifications_mlperf_speed_and_deals/aa2026gptoss  ->  https://artificialanalysis.ai/models/gpt-oss-120b/providers"   # not a direct PDF; open in a browser
echo "manual: 9_measured_specifications_mlperf_speed_and_deals/patel2024groq  ->  https://newsletter.semianalysis.com/p/groq-inference-tokenomics-speed-but"   # not a direct PDF; open in a browser
echo "manual: 9_measured_specifications_mlperf_speed_and_deals/scmp2025anthropictpu  ->  https://www.scmp.com/tech/big-tech/article/3330143/google-anthropic-unveil-cloud-deal-worth-tens-billions-dollars"   # not a direct PDF; open in a browser
echo "manual: 9_measured_specifications_mlperf_speed_and_deals/aim2025broadcomanthropic  ->  https://analyticsindiamag.com/ai-news-updates/broadcom-reveals-21-billion-google-tpus-order-from-anthropic/"   # not a direct PDF; open in a browser
echo "manual: 9_measured_specifications_mlperf_speed_and_deals/dataconomy2026metatpu  ->  https://dataconomy.com/2026/02/27/meta-signs-multibillion-dollar-deal-to-rent-google-tpus-for-ai-training/"   # not a direct PDF; open in a browser
echo "manual: 9_measured_specifications_mlperf_speed_and_deals/siliconangle2025rainier  ->  https://siliconangle.com/2025/10/29/aws-opens-11b-project-rainier-data-center-campus-built-anthropic/"   # not a direct PDF; open in a browser
echo "manual: 9_measured_specifications_mlperf_speed_and_deals/broadcom2025q4  ->  https://www.sec.gov/Archives/edgar/data/1730168/000173016825000116/avgo-11022025x8kxex99.htm"   # not a direct PDF; open in a browser
echo "manual: 9_measured_specifications_mlperf_speed_and_deals/trendforce2025share  ->  https://www.trendforce.com/presscenter/news/20251030-12762.html"   # not a direct PDF; open in a browser
echo "manual: 9_measured_specifications_mlperf_speed_and_deals/crunchbase2026semis  ->  https://news.crunchbase.com/venture/record-high-us-semiconductor-startup-funding-eoy-2025"   # not a direct PDF; open in a browser
echo "manual: 10_wrap-up/emani2022pmbs  ->  https://sc22.supercomputing.org/proceedings/workshops/workshop_pages/ws_pmbsf120.html"   # not a direct PDF; open in a browser
get "10_wrap-up" "lattner2021mlir.pdf" "https://arxiv.org/pdf/2002.11054"
get "10_wrap-up" "chen2018tvm.pdf" "https://arxiv.org/pdf/1802.04799"
echo "manual: 10_wrap-up/google2023openxla  ->  https://opensource.googleblog.com/2023/03/openxla-is-ready-to-accelerate-and-simplify-ml-development.html"   # not a direct PDF; open in a browser
echo "manual: 10_wrap-up/vllm2025plugin  ->  https://vllm.ai/blog/2025-05-12-hardware-plugin"   # not a direct PDF; open in a browser
echo "manual: 10_wrap-up/register2020springcrest  ->  https://www.theregister.com/2020/01/31/intel_kills_spring_crest/"   # not a direct PDF; open in a browser
echo "manual: 10_wrap-up/mips2021wave  ->  https://prnewswire.com/news-releases/wave-computing-and-mips-emerge-from-chapter-11-bankruptcy-301237051.html"   # not a direct PDF; open in a browser
echo "manual: 10_wrap-up/corsix2025esperanto  ->  https://www.corsix.org/content/esperanto-lives-on"   # not a direct PDF; open in a browser

cp "$(dirname "$0")/reading_list.csv" "$(dirname "$0")/reading_list.bib" "$(dirname "$0")/README.md" "$ROOT/" 2>/dev/null
echo "downloaded: $ok   failed: $fail"
( cd "$(dirname "$ROOT")" && rm -f non_gpu_accelerators_papers.zip && zip -qr non_gpu_accelerators_papers.zip "$(basename "$ROOT")" ) && echo "wrote $(dirname "$ROOT")/non_gpu_accelerators_papers.zip"
