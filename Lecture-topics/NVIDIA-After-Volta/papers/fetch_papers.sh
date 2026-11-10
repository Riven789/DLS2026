#!/bin/bash
# Downloads the reading list for 'NVIDIA GPU Architectures after Volta' from arXiv and zips it.
# Usage:  bash fetch_papers.sh          (needs curl and zip; ~27 PDFs; takes a few minutes)
# arXiv asks for no more than one request every few seconds from scripts; the sleep below respects that.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)/nvidia_after_volta_papers"
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

echo "manual: 1_overview_nine_years_after_volta/dally2021evolution  ->  https://doi.org/10.1109/MM.2021.3113475"   # not a direct PDF; open in a browser
echo "manual: 1_overview_nine_years_after_volta/computex2024roadmap  ->  https://www.theregister.com/2024/06/05/chipmakers_computex_roadmaps"   # not a direct PDF; open in a browser
echo "manual: 1_overview_nine_years_after_volta/lockwood2026gtc  ->  https://blog.glennklockwood.com/2026/03/gtc-2026-recap.html"   # not a direct PDF; open in a browser
echo "manual: 1_overview_nine_years_after_volta/lockwood_b200  ->  https://glennklockwood.com/garden/processors/B200"   # not a direct PDF; open in a browser
echo "manual: 1_overview_nine_years_after_volta/lockwood_b300  ->  https://glennklockwood.com/garden/processors/B300"   # not a direct PDF; open in a browser
echo "manual: 1_overview_nine_years_after_volta/lockwood_r200  ->  https://glennklockwood.com/garden/processors/R200"   # not a direct PDF; open in a browser
get "1_overview_nine_years_after_volta" "micikevicius2017mixed.pdf" "https://arxiv.org/pdf/1710.03740"
get "2_turing" "nvidia2018turingwp.pdf" "https://images.nvidia.com/aem-dam/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf"
get "2_turing" "nvidia2018t4ds.pdf" "https://nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf"
get "2_turing" "jia2019turing.pdf" "https://arxiv.org/pdf/1903.07486"
get "2_turing" "jia2018volta.pdf" "https://arxiv.org/pdf/1804.06826"
echo "manual: 2_turing/nvidia2019dlss  ->  https://developer.nvidia.com/blog/nvidia-dlss-2-0-a-big-leap-in-ai-rendering/"   # not a direct PDF; open in a browser
get "3_ampere" "nvidia2020amperewp.pdf" "https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf"
get "3_ampere" "choquette2020hc32.pdf" "https://hc32.hotchips.org/assets/program/conference/day1/HotChips2020_GPU_NVIDIA_Choquette_v01.pdf"
echo "manual: 3_ampere/choquette2021micro  ->  https://doi.org/10.1109/MM.2021.3061394"   # not a direct PDF; open in a browser
get "3_ampere" "abdelkhalik2022ampere.pdf" "https://arxiv.org/pdf/2208.11174"
echo "manual: 3_ampere/tomshw2020ga102  ->  https://www.tomshardware.com/features/nvidia-ampere-architecture-deep-dive"   # not a direct PDF; open in a browser
get "3_ampere" "mishra2021sparsity.pdf" "https://arxiv.org/pdf/2104.08378"
get "3_ampere" "kalamkar2019bf16.pdf" "https://arxiv.org/pdf/1905.12322"
echo "manual: 3_ampere/nvidia2020mig  ->  https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html"   # not a direct PDF; open in a browser
get "3_ampere" "nvidia2020dgxa100.pdf" "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/HGX/pdf/nvidia-hgx-a100-datasheet.pdf"
get "4_hopper" "nvidia2022hopperwp.pdf" "https://www.ece.lsu.edu/koppel/gp/refs/gtc22-whitepaper-hopper.pdf"
get "4_hopper" "nvidia2023h100ds.pdf" "https://openzeka.com/wp-content/uploads/2023/12/h100-datasheet-1.pdf"
echo "manual: 4_hopper/choquette2023micro  ->  https://ieeexplore.ieee.org/document/10070122"   # not a direct PDF; open in a browser
echo "manual: 4_hopper/andersch2022hopperblog  ->  https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/"   # not a direct PDF; open in a browser
get "4_hopper" "micikevicius2022fp8.pdf" "https://arxiv.org/pdf/2209.05433"
get "4_hopper" "luo2024hopper.pdf" "https://arxiv.org/pdf/2402.13499"
echo "manual: 4_hopper/nvidia2023h200pr  ->  https://nvidianews.nvidia.com/news/nvidia-supercharges-hopper-the-worlds-leading-ai-computing-platform"   # not a direct PDF; open in a browser
get "4_hopper" "nvidia2023gh200ds.pdf" "https://www.amax.com/content/files/2023/12/NVIDIA_GH200_Grace_Hopper_Superchip_Datasheet.pdf"
get "4_hopper" "nvidia2024gracetuning.pdf" "https://docs.nvidia.com/grace-performance-tuning-guide.pdf"
echo "manual: 4_hopper/nvidia2023awsgh200  ->  https://nvidianews.nvidia.com/news/aws-nvidia-strategic-collaboration-for-generative-ai"   # not a direct PDF; open in a browser
get "4_hopper" "shah2024fa3.pdf" "https://arxiv.org/pdf/2407.08691"
echo "manual: 4_hopper/nvidia2023nvswitch3  ->  https://www.nvidia.com/en-us/data-center/nvlink/"   # not a direct PDF; open in a browser
get "5_ada_lovelace" "nvidia2022adawp.pdf" "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf"
get "5_ada_lovelace" "supermicro2023l40s.pdf" "https://www.supermicro.com/datasheet/datasheet_NVIDIA_L40S_Systems.pdf"
echo "manual: 6_blackwell_and_blackwell_ultra/nvidia2024blackwellpr  ->  https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing"   # not a direct PDF; open in a browser
get "6_blackwell_and_blackwell_ultra" "nvidia2024blackwelldeck.pdf" "https://mcomputers.cz/wp-content/uploads/2024/04/NVIDIA-Blackwell-Praha-PDF.pdf"
echo "manual: 6_blackwell_and_blackwell_ultra/nvidia2024blackwellbrief  ->  https://resources.nvidia.com/en-us-blackwell-architecture"   # not a direct PDF; open in a browser
echo "manual: 6_blackwell_and_blackwell_ultra/tirumala2024hc36  ->  https://doi.org/10.1109/HCS61935.2024.10665247"   # not a direct PDF; open in a browser
echo "manual: 6_blackwell_and_blackwell_ultra/nvidia2025gb200page  ->  https://www.nvidia.com/en-us/data-center/gb200-nvl72/"   # not a direct PDF; open in a browser
echo "manual: 6_blackwell_and_blackwell_ultra/lenovo2026b200  ->  https://lenovopress.lenovo.com/lp2226-thinksystem-nvidia-b200-180gb-1000w-gpu"   # not a direct PDF; open in a browser
echo "manual: 6_blackwell_and_blackwell_ultra/heise2024nvlinkswitch  ->  https://heise.de/en/news/Blackwell-Nvidia-unveils-its-next-generation-of-AI-accelerators-9782399.html?seite=2"   # not a direct PDF; open in a browser
echo "manual: 6_blackwell_and_blackwell_ultra/register2024fp64  ->  https://www.theregister.com/2024/05/22/nvidias_future_in_scientific_computing/"   # not a direct PDF; open in a browser
echo "manual: 6_blackwell_and_blackwell_ultra/register2024mask  ->  https://www.theregister.com/2024/08/29/nvidia_blackwell_manufacturing/"   # not a direct PDF; open in a browser
echo "manual: 6_blackwell_and_blackwell_ultra/neowin2024azure  ->  https://www.neowin.net/amp/microsoft-azure-is-the-first-hyperscaler-to-have-nvidias-blackwell-system-with-gb200/"   # not a direct PDF; open in a browser
get "6_blackwell_and_blackwell_ultra" "jarmusch2025blackwell1.pdf" "https://arxiv.org/pdf/2507.10789"
get "6_blackwell_and_blackwell_ultra" "jarmusch2025blackwell2.pdf" "https://arxiv.org/pdf/2512.02189"
echo "manual: 6_blackwell_and_blackwell_ultra/nvidia2025blackwellultrablog  ->  https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/"   # not a direct PDF; open in a browser
echo "manual: 6_blackwell_and_blackwell_ultra/tomshw2025blackwellultra  ->  https://www.tomshardware.com/pc-components/gpus/nvidia-shares-blackwell-ultras-secrets-nvfp4-boost-detailed-and-pcie-6-0-support"   # not a direct PDF; open in a browser
echo "manual: 6_blackwell_and_blackwell_ultra/nvidia2025gb300page  ->  https://www.nvidia.com/en-au/data-center/gb300-nvl72"   # not a direct PDF; open in a browser
echo "manual: 6_blackwell_and_blackwell_ultra/nvidia2025blackwellultrads  ->  https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-ultra-datasheet"   # not a direct PDF; open in a browser
echo "manual: 6_blackwell_and_blackwell_ultra/tomshw2025gb300coreweave  ->  https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidias-newest-top-tier-ai-supercomputers-deployed-for-the-first-time-grace-blackwell-ultra-superchip-systems-deployed-at-coreweave"   # not a direct PDF; open in a browser
echo "manual: 6_blackwell_and_blackwell_ultra/nvidia2025nvfp4blog  ->  https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/"   # not a direct PDF; open in a browser
get "6_blackwell_and_blackwell_ultra" "abecassis2025nvfp4.pdf" "https://arxiv.org/pdf/2509.25149"
get "6_blackwell_and_blackwell_ultra" "rouhani2023mx.pdf" "https://arxiv.org/pdf/2310.10537"
get "6_blackwell_and_blackwell_ultra" "ocp2023mxspec.pdf" "https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf"
echo "manual: 6_blackwell_and_blackwell_ultra/ozaki2012scheme  ->  https://doi.org/10.1007/s11075-011-9478-1"   # not a direct PDF; open in a browser
echo "manual: 6_blackwell_and_blackwell_ultra/ootomo2024dgemm  ->  https://doi.org/10.1177/10943420241239588"   # not a direct PDF; open in a browser
echo "manual: 6_blackwell_and_blackwell_ultra/nvidia2025fp64emulation  ->  https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/"   # not a direct PDF; open in a browser
echo "manual: 6_blackwell_and_blackwell_ultra/nvidia2025cublas129  ->  https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9/"   # not a direct PDF; open in a browser
echo "manual: 6_blackwell_and_blackwell_ultra/register2024nvl72power  ->  https://www.theregister.com/2024/03/21/nvidia_dgx_gb200_nvk72"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/tomshw2025rubinroadmap  ->  https://www.tomshardware.com/pc-components/gpus/nvidia-announces-rubin-gpus-in-2026-rubin-ultra-in-2027-feynam-after"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/nvidia2025cpxpr  ->  https://nvidianews.nvidia.com/news/nvidia-unveils-rubin-cpx-a-new-class-of-gpu-designed-for-massive-context-inference"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/rcrtech2026cpx  ->  https://rcrtech.com/semiconductor-news/nvidia-reportedly-revives-rubin-cpx/"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/nvidia2026cespr  ->  https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-Kicks-Off-the-Next-Generation-of-AI-With-Rubin--Six-New-Chips-One-Incredible-AI-Supercomputer/default.aspx"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/sth2026ces  ->  https://www.servethehome.com/nvidia-launches-next-generation-rubin-ai-compute-platform-at-ces-2026/"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/tomshw2026sevenchips  ->  https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidias-seven-chip-vera-rubin-platforms-turns-the-data-center-into-an-ai-factory"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/storagereview2026gtc  ->  https://www.storagereview.com/news/nvidia-gtc-2026-rubin-gpus-groq-lpus-vera-cpus-and-what-nvidia-is-building-for-trillion-parameter-inference"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/nvidia2026vrpage  ->  https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/hwbusters2026speccut  ->  https://hwbusters.com/news/nvidia-quietly-lowers-vera-rubin-hbm4-bandwidth-and-the-hgx-nvl8-loses-a-quarter-of-it/"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/wccftech2026statement  ->  https://wccftech.com/nvidia-official-statement-on-vera-rubin-spec-change/"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/techpowerup2026speccut  ->  https://www.techpowerup.com/354133/nvidia-cuts-hbm4-speed-and-nvlink-bandwidth-targets-for-vera-rubin"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/hotchips2026program  ->  https://hc2026.hotchips.org/program/"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/sth2026vrhotchips  ->  https://www.servethehome.com/nvidia-vera-rubin-nvl72-rack-at-hot-chips-2026/"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/sth2026vera  ->  https://www.servethehome.com/nvidia-vera-cpu-at-hot-chips-2026/"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/nvidia2026fullproduction  ->  https://nvidianews.nvidia.com/news/vera-rubin-full-production-agentic-ai-factory"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/nvidia2026q2fy27  ->  https://www.sec.gov/Archives/edgar/data/0001045810/000104581026000073/q2fy27pr.htm"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/tomshw2026vr20b  ->  https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-expects-to-sell-usd20-billion-worth-of-vera-rubin-hardware-this-quarter-would-account-for-20-percent-of-data-center-revenue-its-fastest-ramp-in-company-history"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/coreweave2026vr  ->  https://investors.coreweave.com/news/news-details/2026/CoreWeave-Delivers-NVIDIA-Vera-Rubin-NVL72-Performance-at-Production-Scale-Starting-With-Cognition/default.aspx"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/nvidia2025nvlinkfusion  ->  https://investor.nvidia.com/news/press-release-details/2025/NVIDIA-Unveils-NVLink-Fusion-for-Industry-to-Build-Semi-Custom-AI-Infrastructure-With-NVIDIA-Partner-Ecosystem/default.aspx"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/constellation2025groq  ->  https://www.constellationr.com/insights/news/nvidias-groq-deal-acquisition-acquihire-or-creative-licensing-deal"   # not a direct PDF; open in a browser
echo "manual: 7_rubin_and_vera/nvidia2025dynamo  ->  https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_scale/top500_2026jun  ->  https://www.top500.org/news/lineshine-debuts-no-1-top500-enters-new-global-exascale-era/"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_scale/top500_2025nov  ->  https://top500.org/news/el-capitan-retains-1-as-jupiter-becomes-europes-first-exascale-system-in-the-66th-top500-list/"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_scale/fzj2025jupiter  ->  https://www.fz-juelich.de/en/news/archive/press-release/2025/europes-first-supercomputer-reaches-1-exaflop-s"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_scale/lbl2026cech  ->  https://newscenter.lbl.gov/2026/03/17/berkeley-lab-takes-major-step-toward-doudna-with-delivery-of-early-access-system-cech/"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_scale/nersc_doudna  ->  https://www.nersc.gov/what-we-do/resources/doudna-system"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_scale/nvidia2025bluelion  ->  https://blogs.nvidia.com/blog/blue-lion-vera-rubin/"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_scale/nvidia2025solstice  ->  https://nvidianews.nvidia.com/news/nvidia-oracle-us-department-of-energy-ai-supercomputer-scientific-discovery"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_scale/doe2025solstice  ->  https://www.energy.gov/articles/energy-department-announces-new-partnership-nvidia-and-oracle-build-largest-doe-ai"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_scale/dcd2025stargate  ->  https://www.datacenterdynamics.com/en/news/openai-and-oracle-to-deploy-450000-gb200-gpus-at-stargate-abilene-data-center/"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_scale/nvidia2026superpodrubin  ->  https://blogs.nvidia.com/blog/dgx-superpod-rubin/"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_scale/nvidia2025mlperftraining51  ->  https://blogs.nvidia.com/blog/mlperf-training-benchmark-blackwell-ultra/"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_scale/nvidia2026mlperftraining60  ->  https://developer.nvidia.com/blog/nvidia-blackwell-tops-mlperf-training-6-0-with-industry-leading-scale-and-performance/"   # not a direct PDF; open in a browser
get "8_systems_and_scale" "mlcommons2026training60.pdf" "https://mlcommons.org/wp-content/uploads/2026/06/Final-MLPerf-Training-v6.0-Supplemental-Discussion-UNDER-EMBARGO-UNTIL-6_16_26-8_00-AM-PT.pdf"
echo "manual: 8_systems_and_scale/mlcommons2026inference60  ->  https://www.streetinsider.com/Press+Releases/MLCommons+Releases+New+MLPerf+Inference+v6.0+Benchmark+Results/26256741.html"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_scale/trendforce2026rubin  ->  https://www.trendforce.com/presscenter/news/20260408-13003.html"   # not a direct PDF; open in a browser
echo "manual: 9_software/nvidia2025cuda13notes  ->  https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html"   # not a direct PDF; open in a browser
echo "manual: 9_software/nvidia2022cuda12notes  ->  https://docs.nvidia.com/cuda/archive/12.0.0/cuda-toolkit-release-notes/index.html"   # not a direct PDF; open in a browser
echo "manual: 9_software/nvidia_transformerengine  ->  https://github.com/NVIDIA/TransformerEngine"   # not a direct PDF; open in a browser
echo "manual: 9_software/thakkar2023cutlass  ->  https://github.com/NVIDIA/cutlass"   # not a direct PDF; open in a browser
echo "manual: 9_software/semianalysis2024cudamoat  ->  https://newsletter.semianalysis.com/p/mi300x-vs-h100-vs-h200-benchmark-part-1-training"   # not a direct PDF; open in a browser
echo "manual: 9_software/nvidia2025cutile  ->  https://developer.nvidia.com/blog/cuda-python-and-cutile/"   # not a direct PDF; open in a browser
echo "manual: 10_wrap-up_retrospectives_and_the_spec-sheet_record/nvidia_hopperwp_official  ->  https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper"   # not a direct PDF; open in a browser
echo "manual: 10_wrap-up_retrospectives_and_the_spec-sheet_record/nvidia_h100ds_official  ->  https://resources.nvidia.com/en-us-grace-cpu/h100-datasheet-24306-1"   # not a direct PDF; open in a browser
echo "manual: 10_wrap-up_retrospectives_and_the_spec-sheet_record/wikipedia_rubin  ->  https://en.wikipedia.org/wiki/Rubin_(microarchitecture)"   # not a direct PDF; open in a browser
echo "manual: 10_wrap-up_retrospectives_and_the_spec-sheet_record/nvidia2026ceskeynote  ->  https://www.youtube.com/live/0NBILspM4c4"   # not a direct PDF; open in a browser
echo "manual: 10_wrap-up_retrospectives_and_the_spec-sheet_record/hpcwire2026roadmap  ->  https://www.hpcwire.com/aiwire/2026/03/18/huang-shares-nvidia-roadmap-showing-more-chips-nvl1152-scale-up-cpo/"   # not a direct PDF; open in a browser
echo "manual: 10_wrap-up_retrospectives_and_the_spec-sheet_record/semihub2026hotchips  ->  https://semihub.io/en/blog/hot-chips-2026-recap.html"   # not a direct PDF; open in a browser

cp "$(dirname "$0")/reading_list.csv" "$(dirname "$0")/reading_list.bib" "$(dirname "$0")/README.md" "$ROOT/" 2>/dev/null
echo "downloaded: $ok   failed: $fail"
( cd "$(dirname "$ROOT")" && rm -f nvidia_after_volta_papers.zip && zip -qr nvidia_after_volta_papers.zip "$(basename "$ROOT")" ) && echo "wrote $(dirname "$ROOT")/nvidia_after_volta_papers.zip"
