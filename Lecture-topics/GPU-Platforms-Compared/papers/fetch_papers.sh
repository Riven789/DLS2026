#!/bin/bash
# Downloads the reading list for 'Data-Center GPU Platforms Compared' from arXiv and zips it.
# Usage:  bash fetch_papers.sh          (needs curl and zip; ~24 PDFs; takes a few minutes)
# arXiv asks for no more than one request every few seconds from scripts; the sleep below respects that.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)/gpu_platforms_compared_papers"
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

echo "manual: 1_overview_three_vendors_six_levels/inferencex2026specs  ->  https://inferencex.semianalysis.com/gpu-specs"   # not a direct PDF; open in a browser
echo "manual: 1_overview_three_vendors_six_levels/siliconanalysts2026share  ->  https://siliconanalysts.com/analysis/amd-vs-nvidia-ai-gpu-market-share-2026"   # not a direct PDF; open in a browser
echo "manual: 1_overview_three_vendors_six_levels/trendforce2026rubin  ->  https://www.trendforce.com/presscenter/news/20260408-13003.html"   # not a direct PDF; open in a browser
echo "manual: 1_overview_three_vendors_six_levels/lockwood2026r200  ->  https://glennklockwood.com/garden/processors/R200"   # not a direct PDF; open in a browser
echo "manual: 1_overview_three_vendors_six_levels/ocp2019oam  ->  https://www.opencompute.org/wiki/Server/OAM"   # not a direct PDF; open in a browser
get "2_the_nvidia_line_as_platforms" "nvidia2020hgxa100.pdf" "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/HGX/pdf/nvidia-hgx-a100-datasheet.pdf"
get "2_the_nvidia_line_as_platforms" "nvidia2023h100ds.pdf" "https://openzeka.com/wp-content/uploads/2023/12/h100-datasheet-1.pdf"
echo "manual: 2_the_nvidia_line_as_platforms/coreweave2024hgxh100  ->  https://docs.coreweave.com/coreweave-machine-learning-and-ai/nvidia-hgx-h100"   # not a direct PDF; open in a browser
echo "manual: 2_the_nvidia_line_as_platforms/nvidia2023h200pr  ->  https://nvidianews.nvidia.com/news/nvidia-supercharges-hopper-the-worlds-leading-ai-computing-platform"   # not a direct PDF; open in a browser
get "2_the_nvidia_line_as_platforms" "lenovo2025h200.pdf" "https://lenovopress.lenovo.com/lp1944.pdf"
echo "manual: 2_the_nvidia_line_as_platforms/nvidia2025gb200page  ->  https://www.nvidia.com/en-us/data-center/gb200-nvl72/"   # not a direct PDF; open in a browser
echo "manual: 2_the_nvidia_line_as_platforms/theregister2024gb200  ->  https://www.theregister.com/2024/03/21/nvidia_dgx_gb200_nvk72"   # not a direct PDF; open in a browser
echo "manual: 2_the_nvidia_line_as_platforms/nvidia2025gb300page  ->  https://www.nvidia.com/en-gb/data-center/gb300-nvl72"   # not a direct PDF; open in a browser
echo "manual: 2_the_nvidia_line_as_platforms/tomshw2025gb300coreweave  ->  https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidias-newest-top-tier-ai-supercomputers-deployed-for-the-first-time-grace-blackwell-ultra-superchip-systems-deployed-at-coreweave"   # not a direct PDF; open in a browser
echo "manual: 2_the_nvidia_line_as_platforms/tomshw2024dellgb200  ->  https://www.tomshardware.com/tech-industry/artificial-intelligence/dell-reaches-milestone-with-industrys-first-enterprise-ready-nvidia-blackwell-poweredge-xe9712-server-racks"   # not a direct PDF; open in a browser
echo "manual: 2_the_nvidia_line_as_platforms/tomshw2026vrces  ->  https://www.tomshardware.com/pc-components/gpus/nvidia-launches-vera-rubin-nvl72-ai-supercomputer-at-ces-promises-up-to-5x-greater-inference-performance-and-10x-lower-cost-per-token-than-blackwell-coming-2h-2026"   # not a direct PDF; open in a browser
echo "manual: 2_the_nvidia_line_as_platforms/nvidia2026vrpage  ->  https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72"   # not a direct PDF; open in a browser
echo "manual: 2_the_nvidia_line_as_platforms/hwbusters2026speccut  ->  https://hwbusters.com/news/nvidia-quietly-lowers-vera-rubin-hbm4-bandwidth-and-the-hgx-nvl8-loses-a-quarter-of-it/"   # not a direct PDF; open in a browser
echo "manual: 2_the_nvidia_line_as_platforms/wccftech2026statement  ->  https://wccftech.com/nvidia-official-statement-on-vera-rubin-spec-change/"   # not a direct PDF; open in a browser
get "3_amd_instinct_and_cdna" "amd2020cdna1wp.pdf" "https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna-white-paper.pdf"
get "3_amd_instinct_and_cdna" "amd2021cdna2wp.pdf" "https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf"
get "3_amd_instinct_and_cdna" "amd2023cdna3wp.pdf" "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf"
get "3_amd_instinct_and_cdna" "amd2025cdna4wp.pdf" "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf"
get "3_amd_instinct_and_cdna" "amd2021mi200ds.pdf" "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instinct-mi200-datasheet.pdf"
get "3_amd_instinct_and_cdna" "amd2022mi210br.pdf" "https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/product-briefs/instinct-mi210-brochure.pdf"
get "3_amd_instinct_and_cdna" "amd2023mi300ads.pdf" "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300a-data-sheet.pdf"
get "3_amd_instinct_and_cdna" "amd2023mi300xpb.pdf" "https://www.amd.com/content/dam/amd/en/documents/partner-hub/instinct/instinct-mi300x-accelerator-product-briefs.pdf"
get "3_amd_instinct_and_cdna" "amd2024mi325xds.pdf" "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-datasheet.pdf"
get "3_amd_instinct_and_cdna" "amd2025mi355xbr.pdf" "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf"
echo "manual: 3_amd_instinct_and_cdna/amd2024mi300arch  ->  https://instinct.docs.amd.com/latest/_sources/gpu-arch/mi300.md"   # not a direct PDF; open in a browser
echo "manual: 3_amd_instinct_and_cdna/amd2026mi250arch  ->  https://rocm.docs.amd.com/en/docs-7.14.1/_sources/reference/gpu-arch/mi250.md"   # not a direct PDF; open in a browser
echo "manual: 3_amd_instinct_and_cdna/amd2026mi100arch  ->  https://rocm.docs.amd.com/en/docs-7.14.1/reference/gpu-arch/mi100.html"   # not a direct PDF; open in a browser
get "3_amd_instinct_and_cdna" "smith2024mi300xhc36.pdf" "https://www.hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith(MI300X).v1.Final.20240817.pdf"
echo "manual: 3_amd_instinct_and_cdna/sth2025cdna4hotchips  ->  https://www.servethehome.com/amd-dives-deep-on-cdna-4-architecture-and-mi350-accelerator-at-hot-chips-2025/"   # not a direct PDF; open in a browser
echo "manual: 3_amd_instinct_and_cdna/amd2024isscc_mi300  ->  https://www.aihardware.mit.edu/amd-instincttm-mi300-series-modular-chiplet-package-hpc-and-ai-accelerator-for-exa-class-systems/"   # not a direct PDF; open in a browser
echo "manual: 3_amd_instinct_and_cdna/chipsandcheese2023cdna3  ->  https://chipsandcheese.com/p/amds-cdna-3-compute-architecture"   # not a direct PDF; open in a browser
echo "manual: 3_amd_instinct_and_cdna/chipsandcheese2025mi430x  ->  https://old.chipsandcheese.com/?p=37639"   # not a direct PDF; open in a browser
echo "manual: 3_amd_instinct_and_cdna/phoronix2025mi400  ->  https://www.phoronix.com/news/AMD-Instinct-MI400-Preview"   # not a direct PDF; open in a browser
echo "manual: 3_amd_instinct_and_cdna/tomshw2025helios_ocp  ->  https://www.tomshardware.com/tech-industry/amd-debuts-helios-rack-scale-ai-hardware-platform-at-ocp-global-summit-2025-promises-easier-serviceability-and-50-percent-more-memory-than-nvidias-vera-rubin"   # not a direct PDF; open in a browser
echo "manual: 3_amd_instinct_and_cdna/tomshw2026mi455x  ->  https://www.tomshardware.com/pc-components/gpus/amd-takes-the-wraps-off-its-instinct-mi455x-ai-accelerator-cdna-5-and-helios-rack-scale-architecture-combine-to-take-the-fight-to-nvidia-in-the-data-center"   # not a direct PDF; open in a browser
echo "manual: 3_amd_instinct_and_cdna/storagereview2026mi455x  ->  https://www.storagereview.com/news/amd-mi455x-and-helios-432gb-hbm4-72-gpu-racks-and-a-real-answer-to-vera-rubin"   # not a direct PDF; open in a browser
get "3_amd_instinct_and_cdna" "supermicro2026helios.pdf" "https://www.supermicro.com/datasheet/datasheet_AMD_Helios.pdf"
echo "manual: 3_amd_instinct_and_cdna/sth2026advancingai  ->  https://www.servethehome.com/?p=102176"   # not a direct PDF; open in a browser
echo "manual: 3_amd_instinct_and_cdna/techwireasia2026advancingai  ->  https://techwireasia.com/2026/07/amd-advancing-ai-2026-helios-openai-meta-anthropic/"   # not a direct PDF; open in a browser
echo "manual: 3_amd_instinct_and_cdna/amd2025mi430xblog  ->  https://www.amd.com/en/blogs/2025/amd-instinct-mi430x-powering-the-next-wave-of-ai.html"   # not a direct PDF; open in a browser
echo "manual: 3_amd_instinct_and_cdna/amd2026q3results  ->  https://www.sec.gov/Archives/edgar/data/0000002488/000000248826000188/q32026991.htm"   # not a direct PDF; open in a browser
echo "manual: 3_amd_instinct_and_cdna/amd2026q3call  ->  https://singjupost.com/amd-advanced-micro-devices-q3-2026-earnings-call-transcript/"   # not a direct PDF; open in a browser
echo "manual: 3_amd_instinct_and_cdna/storagenewsletter2026amdfy25  ->  https://www.storagenewsletter.com/2026/02/18/amd-fiscal-4q25-and-fy25-financial-results/"   # not a direct PDF; open in a browser
echo "manual: 3_amd_instinct_and_cdna/amd2025openai  ->  https://ir.amd.com/news-events/press-releases/detail/1260/amd-and-openai-announce-strategic-partnership-to-deploy-6-gigawatts-of-amd-gpus"   # not a direct PDF; open in a browser
get "4_intel_the_gen_families_xe_and_gaudi" "junkins2015gen9.pdf" "https://www.intel.com/content/dam/develop/external/us/en/documents/the-compute-architecture-of-intel-processor-graphics-gen9-v1d0-843410.pdf"
get "4_intel_the_gen_families_xe_and_gaudi" "intel2019gen11.pdf" "https://www.intel.com/content/dam/develop/external/us/en/documents/the-architecture-of-intel-processor-graphics-gen11-r1new.pdf"
echo "manual: 4_intel_the_gen_families_xe_and_gaudi/tomshw2020xelp  ->  https://www.tomshardware.com/news/intel-xe-lp-graphics-specs"   # not a direct PDF; open in a browser
echo "manual: 4_intel_the_gen_families_xe_and_gaudi/intel2022xehpg  ->  https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-to-the-xe-hpg-architecture.html"   # not a direct PDF; open in a browser
echo "manual: 4_intel_the_gen_families_xe_and_gaudi/intel2024xearch  ->  https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/xe-arch.html"   # not a direct PDF; open in a browser
echo "manual: 4_intel_the_gen_families_xe_and_gaudi/tomshw2022arca770  ->  https://www.tomshardware.com/news/intel-reveals-specifications-for-arc-alchemist-desktop-gpus"   # not a direct PDF; open in a browser
echo "manual: 4_intel_the_gen_families_xe_and_gaudi/tomshw2024b580  ->  https://www.tomshardware.com/pc-components/gpus/intel-arc-b580-review-the-new-usd249-gpu-champion-has-arrived"   # not a direct PDF; open in a browser
echo "manual: 4_intel_the_gen_families_xe_and_gaudi/tomshw2025arcprob60  ->  https://www.tomshardware.com/pc-components/gpus/intel-launches-usd299-arc-pro-b50-with-16gb-of-memory-project-battlematrix-workstations-with-24gb-arc-pro-b60-gpus"   # not a direct PDF; open in a browser
echo "manual: 4_intel_the_gen_families_xe_and_gaudi/hothardware2026pantherlake  ->  https://hothardware.com/news/intel-ces-2026-panther-lake-is-a-go"   # not a direct PDF; open in a browser
get "4_intel_the_gen_families_xe_and_gaudi" "kaplan2024gaudi3hc.pdf" "https://www.hc2024.hotchips.org/assets/program/conference/day1/60_HC2024.Intel.RomanKaplan.Gaudi3-0826.pdf"
echo "manual: 4_intel_the_gen_families_xe_and_gaudi/theregister2022gaudi2  ->  https://www.theregister.com/2022/05/10/intels_habana_unit_reveals_new/"   # not a direct PDF; open in a browser
echo "manual: 4_intel_the_gen_families_xe_and_gaudi/tomshw2024gaudimiss  ->  https://www.tomshardware.com/tech-industry/artificial-intelligence/intel-says-it-will-miss-its-ai-goals-with-gaudi-3-unbaked-software-leaves-intels-usd500-million-ai-goal-unachievable-as-competitors-rake-in-billions"   # not a direct PDF; open in a browser
echo "manual: 5_xe-hpc_ponte_vecchio_aurora_and_after/intel2023ark1550  ->  https://www.intel.com/content/www/us/en/products/sku/232873/intel-data-center-gpu-max-1550/specifications.html"   # not a direct PDF; open in a browser
echo "manual: 5_xe-hpc_ponte_vecchio_aurora_and_after/intel2022isscc  ->  https://intel.com/content/www/us/en/research/blogs/isscc-2022.html"   # not a direct PDF; open in a browser
echo "manual: 5_xe-hpc_ponte_vecchio_aurora_and_after/techpowerup2022pvcisscc  ->  https://techpowerup.com/292250/intel-details-ponte-vecchio-accelerator-63-tiles-600-watt-tdp-and-lots-of-bandwidth"   # not a direct PDF; open in a browser
echo "manual: 5_xe-hpc_ponte_vecchio_aurora_and_after/theregister2022pvchc34  ->  https://www.theregister.com/2022/08/23/intel_ponte_vecchio/"   # not a direct PDF; open in a browser
echo "manual: 5_xe-hpc_ponte_vecchio_aurora_and_after/tomshw2021pvcarchday  ->  https://www.tomshardware.com/features/intel-ponte-vecchio-and-xe-hpc-architecture-built-for-big-data"   # not a direct PDF; open in a browser
echo "manual: 5_xe-hpc_ponte_vecchio_aurora_and_after/techpowerup2023max1550  ->  https://www.techpowerup.com/gpu-specs/data-center-gpu-max-1550.c4068"   # not a direct PDF; open in a browser
echo "manual: 5_xe-hpc_ponte_vecchio_aurora_and_after/lockwood2025pvc  ->  https://glennklockwood.com/garden/processors/PVC"   # not a direct PDF; open in a browser
echo "manual: 5_xe-hpc_ponte_vecchio_aurora_and_after/phoronix2022maxlaunch  ->  https://www.phoronix.com/review/intel-xeon-max-dc/3"   # not a direct PDF; open in a browser
echo "manual: 5_xe-hpc_ponte_vecchio_aurora_and_after/alcf2026aurora  ->  https://docs.alcf.anl.gov/aurora/hardware-overview/machine-overview"   # not a direct PDF; open in a browser
get "5_xe-hpc_ponte_vecchio_aurora_and_after" "goto2026aurorahpl.pdf" "https://arxiv.org/pdf/2604.09517"
echo "manual: 5_xe-hpc_ponte_vecchio_aurora_and_after/hpcwire2023aurorasurvivor  ->  https://www.hpcwire.com/2023/11/13/aurora-the-survivor-exascale-supercomputer-arrives-after-eight-years-of-doom/"   # not a direct PDF; open in a browser
echo "manual: 5_xe-hpc_ponte_vecchio_aurora_and_after/intel2023top500nov  ->  https://www.intel.com/content/www/us/en/newsroom/news/top500-supercomputers-nov-2023.html"   # not a direct PDF; open in a browser
echo "manual: 5_xe-hpc_ponte_vecchio_aurora_and_after/alcf2025gordonbell  ->  https://alcf.anl.gov/news/argonne-researchers-and-supercomputers-contribute-four-projects-named-gordon-bell-prize"   # not a direct PDF; open in a browser
get "5_xe-hpc_ponte_vecchio_aurora_and_after" "hatanpaa2025aeris.pdf" "https://arxiv.org/pdf/2509.13523"
echo "manual: 5_xe-hpc_ponte_vecchio_aurora_and_after/insidehpc2023rialto  ->  https://insidehpc.com/2023/03/intel-alters-hpc-ai-roadmap-rialto-bridge-gpu-discontinued/"   # not a direct PDF; open in a browser
echo "manual: 5_xe-hpc_ponte_vecchio_aurora_and_after/tomshw2025falconshores  ->  https://www.tomshardware.com/tech-industry/artificial-intelligence/intel-cancels-falcon-shores-gpu-for-ai-workloads-jaguar-shores-to-be-successor"   # not a direct PDF; open in a browser
echo "manual: 5_xe-hpc_ponte_vecchio_aurora_and_after/phoronix2025crescentisland  ->  https://www.phoronix.com/review/intel-crescent-island"   # not a direct PDF; open in a browser
echo "manual: 5_xe-hpc_ponte_vecchio_aurora_and_after/storagereview2026dseries  ->  https://www.storagereview.com/news/intel-gpu-d-series-480gb-lpddr5x-1-3-pflops-fp8-350w-pcie-card"   # not a direct PDF; open in a browser
echo "manual: 5_xe-hpc_ponte_vecchio_aurora_and_after/intel2026q3results  ->  https://www.intc.com/news-events/press-releases/detail/1786/intel-reports-third-quarter-2026-financial-results"   # not a direct PDF; open in a browser
echo "manual: 5_xe-hpc_ponte_vecchio_aurora_and_after/intel2025nvidia  ->  https://www.intc.com/news-events/press-releases/detail/1750/"   # not a direct PDF; open in a browser
get "6_fabrics_racks_and_standards" "ualink2025whitepaper.pdf" "https://ualinkconsortium.org/wp-content/uploads/2025/04/UALink-1.0-White_Paper_FINAL.pdf"
echo "manual: 6_fabrics_racks_and_standards/uec2025spec10  ->  https://www.linuxfoundation.org/press/uec-launches-spec-1.0"   # not a direct PDF; open in a browser
echo "manual: 6_fabrics_racks_and_standards/nvidia2025nvlinkfusion  ->  https://investor.nvidia.com/news/press-release-details/2025/NVIDIA-Unveils-NVLink-Fusion-for-Industry-to-Build-Semi-Custom-AI-Infrastructure-With-NVIDIA-Partner-Ecosystem/default.aspx"   # not a direct PDF; open in a browser
get "6_fabrics_racks_and_standards" "olcf2023frontierarch.pdf" "https://www.olcf.ornl.gov/wp-content/uploads/Frontier-Architecture-Overview_Abraham.pdf"
echo "manual: 6_fabrics_racks_and_standards/llnl2026elcapitan  ->  https://hpc.llnl.gov/node/1269"   # not a direct PDF; open in a browser
echo "manual: 6_fabrics_racks_and_standards/morgan2024elcapitan  ->  https://www.nextplatform.com/2024/11/18/el-capitan-supercomputer-blazes-the-trail-for-converged-cpu-gpu-compute/"   # not a direct PDF; open in a browser
echo "manual: 7_software_stacks_and_portability/amd2026rocmversions  ->  https://rocm.docs.amd.com/en/latest/release/versions.html"   # not a direct PDF; open in a browser
echo "manual: 7_software_stacks_and_portability/amd2026hipify  ->  https://rocm.docs.amd.com/projects/HIPIFY/en/latest"   # not a direct PDF; open in a browser
echo "manual: 7_software_stacks_and_portability/khronos2021sycl2020  ->  https://www.khronos.org/news/press/khronos-releases-sycl-2020-final-specification"   # not a direct PDF; open in a browser
echo "manual: 7_software_stacks_and_portability/phoronix2020oneapi10  ->  https://www.phoronix.com/review/intel-oneapi-10"   # not a direct PDF; open in a browser
echo "manual: 7_software_stacks_and_portability/techcrunch2023uxl  ->  https://techcrunch.com/2023/09/19/the-unified-acceleration-foundation-wants-to-create-an-open-standard-for-accelerator-programming/"   # not a direct PDF; open in a browser
echo "manual: 7_software_stacks_and_portability/codeplay2022plugins  ->  https://codeplay.com/portal/press-releases/2022/12/16/codeplay-announces-oneapi-for-nvidia-and-amd-gpu-hardware"   # not a direct PDF; open in a browser
echo "manual: 7_software_stacks_and_portability/intel2026syclomatic  ->  https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/migrate-from-cuda-to-cpp-with-sycl.html"   # not a direct PDF; open in a browser
get "7_software_stacks_and_portability" "reguly2023sycl.pdf" "https://arxiv.org/pdf/2309.10075"
get "7_software_stacks_and_portability" "davis2025portability.pdf" "https://arxiv.org/pdf/2402.08950"
echo "manual: 7_software_stacks_and_portability/ornl2025kokkossycl  ->  https://www.osti.gov/biblio/3016989"   # not a direct PDF; open in a browser
echo "manual: 7_software_stacks_and_portability/semianalysis2024cudamoat  ->  https://newsletter.semianalysis.com/p/mi300x-vs-h100-vs-h200-benchmark-part-1-training"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_procurements/top500_2026jun  ->  https://top500.org/lists/2026/06"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_procurements/hpcg_2026jun  ->  https://top500.org/lists/hpcg/2026/06/"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_procurements/green500_2026jun  ->  https://top500.org/lists/green500/green500-june-2026/"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_procurements/hpe2026top500  ->  https://www.hpe.com/us/en/newsroom/blog-post/2026/06/hpe-delivers-six-out-of-ten-of-worlds-most-powerful-supercomputers.html"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_procurements/csc2021lumiarch  ->  https://www.lumi-supercomputer.eu/lumis-full-system-architecture-revealed/"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_procurements/cines2024adastra  ->  https://dci.dci-gitlab.cines.fr/webextranet/architecture/index.html"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_procurements/hlrs2025hunter  ->  https://www.hlrs.de/solutions/systems/hunter"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_procurements/ornl2025discoverylux  ->  https://www.ornl.gov/news/ornl-amd-and-hpe-deliver-does-newest-ai-supercomputers-discovery-and-lux"   # not a direct PDF; open in a browser
echo "manual: 8_systems_and_procurements/amd2025eviden_recoque  ->  https://newsroom.amd.com/news/amd-and-eviden-to-power-europe-s-new-exascale-supe/"   # not a direct PDF; open in a browser
echo "manual: 9_benchmarks/amd2024mlperfinf41  ->  https://www.amd.com/en/blogs/2024/engineering-insights-unveiling-mlperf-results-on.html"   # not a direct PDF; open in a browser
echo "manual: 9_benchmarks/amd2025mlperfinf50  ->  https://rocm.blogs.amd.com/artificial-intelligence/mi325x-accelerates-mlperf-inference/README.html"   # not a direct PDF; open in a browser
echo "manual: 9_benchmarks/amd2025mlperfinf51  ->  https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v5.1/README.html"   # not a direct PDF; open in a browser
echo "manual: 9_benchmarks/amd2026mlperfinf60  ->  https://amd.com/en/blogs/2026/amd-delivers-breakthrough-mlperf-inference-6-0-results.html"   # not a direct PDF; open in a browser
echo "manual: 9_benchmarks/mlcommons2025training50  ->  https://mlcommons.org/2025/06/mlperf-training-v5-0-results"   # not a direct PDF; open in a browser
echo "manual: 9_benchmarks/amd2025mlperftrain51  ->  https://www.amd.com/en/blogs/2025/accelerating-ai-training.html"   # not a direct PDF; open in a browser
echo "manual: 9_benchmarks/nvidia2026mlperftraining60  ->  https://developer.nvidia.com/blog/nvidia-blackwell-tops-mlperf-training-6-0-with-industry-leading-scale-and-performance/"   # not a direct PDF; open in a browser
echo "manual: 10_wrap-up_and_the_record/lockwood2026gtc  ->  https://blog.glennklockwood.com/2026/03/gtc-2026-recap.html"   # not a direct PDF; open in a browser
echo "manual: 10_wrap-up_and_the_record/sth2026vrhotchips  ->  https://www.servethehome.com/nvidia-vera-rubin-nvl72-rack-at-hot-chips-2026/"   # not a direct PDF; open in a browser
echo "manual: 10_wrap-up_and_the_record/wikipedia2026instinct  ->  https://en.wikipedia.org/wiki/AMD_Instinct"   # not a direct PDF; open in a browser

cp "$(dirname "$0")/reading_list.csv" "$(dirname "$0")/reading_list.bib" "$(dirname "$0")/README.md" "$ROOT/" 2>/dev/null
echo "downloaded: $ok   failed: $fail"
( cd "$(dirname "$ROOT")" && rm -f gpu_platforms_compared_papers.zip && zip -qr gpu_platforms_compared_papers.zip "$(basename "$ROOT")" ) && echo "wrote $(dirname "$ROOT")/gpu_platforms_compared_papers.zip"
