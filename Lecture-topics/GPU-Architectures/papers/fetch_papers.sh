#!/bin/bash
# Downloads the reading list for 'Introduction to GPU Architectures' from arXiv and zips it.
# Usage:  bash fetch_papers.sh          (needs curl and zip; ~36 PDFs; takes a few minutes)
# arXiv asks for no more than one request every few seconds from scripts; the sleep below respects that.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)/gpu_architectures_papers"
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

echo "manual: 1_overview_and_the_long_view/dally2021evolution  ->  https://doi.org/10.1109/MM.2021.3113475"   # not a direct PDF; open in a browser
echo "manual: 1_overview_and_the_long_view/nickolls2010era  ->  https://doi.org/10.1109/MM.2010.41"   # not a direct PDF; open in a browser
echo "manual: 1_overview_and_the_long_view/garland2010throughput  ->  https://doi.org/10.1145/1839676.1839694"   # not a direct PDF; open in a browser
echo "manual: 1_overview_and_the_long_view/fatahalian2008closer  ->  https://doi.org/10.1145/1400181.1400197"   # not a direct PDF; open in a browser
echo "manual: 1_overview_and_the_long_view/owens2008gpucomputing  ->  https://doi.org/10.1109/JPROC.2008.917757"   # not a direct PDF; open in a browser
echo "manual: 1_overview_and_the_long_view/luebke2007howgpuswork  ->  https://doi.org/10.1109/MC.2007.59"   # not a direct PDF; open in a browser
echo "manual: 1_overview_and_the_long_view/peddie2022v1  ->  https://doi.org/10.1007/978-3-031-10968-3"   # not a direct PDF; open in a browser
echo "manual: 1_overview_and_the_long_view/peddie2022v2  ->  https://doi.org/10.1007/978-3-031-13581-1"   # not a direct PDF; open in a browser
echo "manual: 1_overview_and_the_long_view/peddie2023v3  ->  https://doi.org/10.1007/978-3-031-14047-1"   # not a direct PDF; open in a browser
echo "manual: 1_overview_and_the_long_view/hennessy2017caqa  ->  https://www.elsevier.com/books/computer-architecture/hennessy/978-0-12-811905-1"   # not a direct PDF; open in a browser
echo "manual: 2_the_graphics_pipeline_fixed_and_programmable/clark1982geometry  ->  https://doi.org/10.1145/800064.801272"   # not a direct PDF; open in a browser
echo "manual: 2_the_graphics_pipeline_fixed_and_programmable/akeley1993realityengine  ->  https://doi.org/10.1145/166117.166131"   # not a direct PDF; open in a browser
echo "manual: 2_the_graphics_pipeline_fixed_and_programmable/register1999geforce256  ->  https://www.theregister.com/1999/08/31/nvidia_unveils_256cpu_cray_geforce/"   # not a direct PDF; open in a browser
echo "manual: 2_the_graphics_pipeline_fixed_and_programmable/lindholm2001vertex  ->  https://doi.org/10.1145/383259.383274"   # not a direct PDF; open in a browser
echo "manual: 2_the_graphics_pipeline_fixed_and_programmable/mark2003cg  ->  https://doi.org/10.1145/882262.882362"   # not a direct PDF; open in a browser
get "2_the_graphics_pipeline_fixed_and_programmable" "kilgard2003cgtwopages.pdf" "https://arxiv.org/pdf/cs/0302013"
echo "manual: 2_the_graphics_pipeline_fixed_and_programmable/montrym2005geforce6800  ->  https://doi.org/10.1109/MM.2005.37"   # not a direct PDF; open in a browser
echo "manual: 2_the_graphics_pipeline_fixed_and_programmable/andrews2006xbox360  ->  https://doi.org/10.1109/MM.2006.45"   # not a direct PDF; open in a browser
echo "manual: 2_the_graphics_pipeline_fixed_and_programmable/blythe2006direct3d10  ->  https://doi.org/10.1145/1141911.1141947"   # not a direct PDF; open in a browser
echo "manual: 2_the_graphics_pipeline_fixed_and_programmable/peddie2021unifiedshader  ->  https://www.computer.org/publications/tech-news/chasing-pixels/amd-unified-shader-gpu-history"   # not a direct PDF; open in a browser
echo "manual: 3_gpgpu_and_the_unified_g80_with_cuda/buck2004brook  ->  https://doi.org/10.1145/1015706.1015800"   # not a direct PDF; open in a browser
echo "manual: 3_gpgpu_and_the_unified_g80_with_cuda/owens2007survey  ->  https://doi.org/10.1111/j.1467-8659.2007.01012.x"   # not a direct PDF; open in a browser
echo "manual: 3_gpgpu_and_the_unified_g80_with_cuda/steinkraus2005gpuml  ->  https://doi.org/10.1109/ICDAR.2005.251"   # not a direct PDF; open in a browser
echo "manual: 3_gpgpu_and_the_unified_g80_with_cuda/chellapilla2006cnn  ->  https://inria.hal.science/inria-00112631"   # not a direct PDF; open in a browser
echo "manual: 3_gpgpu_and_the_unified_g80_with_cuda/lindholm2008tesla  ->  https://doi.org/10.1109/MM.2008.31"   # not a direct PDF; open in a browser
echo "manual: 3_gpgpu_and_the_unified_g80_with_cuda/nickolls2008cuda  ->  https://doi.org/10.1145/1365490.1365500"   # not a direct PDF; open in a browser
echo "manual: 3_gpgpu_and_the_unified_g80_with_cuda/kirk2007ismm  ->  https://doi.org/10.1145/1296907.1296909"   # not a direct PDF; open in a browser
echo "manual: 3_gpgpu_and_the_unified_g80_with_cuda/eweek2006g80cuda  ->  https://www.eweek.com/networking/nvidia-launches-new-architecture-for-its-graphics-chips/"   # not a direct PDF; open in a browser
get "3_gpgpu_and_the_unified_g80_with_cuda" "nvidia2007teslaoverview.pdf" "https://www.nvidia.com/docs/io/43395/tesla_product_overview.pdf"
echo "manual: 3_gpgpu_and_the_unified_g80_with_cuda/khronos2008opencl  ->  https://www.khronos.org/news/press/the_khronos_group_releases_opencl_1.0_specification"   # not a direct PDF; open in a browser
echo "manual: 3_gpgpu_and_the_unified_g80_with_cuda/techreport2006g80  ->  https://www.techreport.com/reviews/2006q4/geforce-8800/index.x?pg=1"   # not a direct PDF; open in a browser
echo "manual: 4_the_cuda_model_and_its_mapping_to_the_chip/kirk2016pmpp  ->  https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-12-811986-0"   # not a direct PDF; open in a browser
get "4_the_cuda_model_and_its_mapping_to_the_chip" "harris2007reduction.pdf" "https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf"
get "4_the_cuda_model_and_its_mapping_to_the_chip" "volkov2008benchmarking.pdf" "https://bebop.cs.berkeley.edu/pubs/volkov2008-benchmarking.pdf"
echo "manual: 4_the_cuda_model_and_its_mapping_to_the_chip/wittenbrink2011fermi  ->  https://doi.org/10.1109/MM.2011.24"   # not a direct PDF; open in a browser
get "4_the_cuda_model_and_its_mapping_to_the_chip" "nvidia2009fermiwp.pdf" "https://www.ece.lsu.edu/koppel/gp/refs/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf"
get "4_the_cuda_model_and_its_mapping_to_the_chip" "glaskowsky2009fermi.pdf" "https://www.nvidia.com/content/PDF/fermi_white_papers/P.Glaskowsky_NVIDIAFermi-TheFirstCompleteGPUComputingArchitecture.pdf"
get "5_simt_latency_hiding_and_the_energy_argument" "fung2007dwf.pdf" "https://people.ece.ubc.ca/aamodt/papers/wwlfung.micro2007.pdf"
echo "manual: 5_simt_latency_hiding_and_the_energy_argument/hong2009analytical  ->  https://doi.org/10.1145/1555754.1555775"   # not a direct PDF; open in a browser
get "5_simt_latency_hiding_and_the_energy_argument" "bakhoda2009gpgpusim.pdf" "https://people.ece.ubc.ca/aamodt/papers/gpgpusim.ispass09.pdf"
get "5_simt_latency_hiding_and_the_energy_argument" "rogers2012ccws.pdf" "https://engineering.purdue.edu/tgrogers/publication/rogers-micro-2012/rogers-micro-2012.pdf"
get "5_simt_latency_hiding_and_the_energy_argument" "volkov2010occupancy.pdf" "https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf"
get "5_simt_latency_hiding_and_the_energy_argument" "volkov2016thesis.pdf" "https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.pdf"
get "5_simt_latency_hiding_and_the_energy_argument" "keckler2011future.pdf" "https://www.cs.toronto.edu/~pekhimenko/courses/csc2231-f17/Papers/GPUs.pdf"
echo "manual: 5_simt_latency_hiding_and_the_energy_argument/esmaeilzadeh2011darksilicon  ->  https://doi.org/10.1145/2000064.2000108"   # not a direct PDF; open in a browser
echo "manual: 5_simt_latency_hiding_and_the_energy_argument/dennard1974scaling  ->  https://doi.org/10.1109/JSSC.1974.1050511"   # not a direct PDF; open in a browser
echo "manual: 5_simt_latency_hiding_and_the_energy_argument/bohr2007dennard  ->  https://doi.org/10.1109/N-SSC.2007.4785534"   # not a direct PDF; open in a browser
get "5_simt_latency_hiding_and_the_energy_argument" "lee2010debunking.pdf" "https://www.cs.umd.edu/class/fall2019/cmsc714/readings/Lee-GPU.pdf"
echo "manual: 6_the_memory_system_and_the_roofline/williams2009roofline  ->  https://doi.org/10.1145/1498765.1498785"   # not a direct PDF; open in a browser
get "6_the_memory_system_and_the_roofline" "wong2010demystifying.pdf" "https://www.eecg.utoronto.ca/~moshovos/CUDA08/arx/microbenchmark_report.pdf"
get "6_the_memory_system_and_the_roofline" "mei2017dissecting.pdf" "https://arxiv.org/pdf/1509.02308"
echo "manual: 6_the_memory_system_and_the_roofline/jedec2016hbm2  ->  https://www.jedec.org/node/5916"   # not a direct PDF; open in a browser
echo "manual: 6_the_memory_system_and_the_roofline/jedec2016gddr5x  ->  https://www.jedec.org/node/5917"   # not a direct PDF; open in a browser
get "6_the_memory_system_and_the_roofline" "nvidia2016gtx1080wp.pdf" "https://kolegite.com/EE_library/datasheets_and_manuals/GPU/GeForce_GTX_1080_Whitepaper_FINAL.pdf"
get "7_the_generations_tesla_fermi_kepler_maxwell_pascal" "nvidia2010c1060spec.pdf" "https://www.nvidia.com/docs/io/43395/bd-04111-001_v06.pdf"
get "7_the_generations_tesla_fermi_kepler_maxwell_pascal" "nvidia2010c2050ds.pdf" "https://www.nvidia.com/docs/io/43395/nv_ds_tesla_c2050_c2070_jul10_lores.pdf"
get "7_the_generations_tesla_fermi_kepler_maxwell_pascal" "nvidia2012gk110wp.pdf" "https://www.nvidia.co.jp/content/apac/pdf/tesla/nvidia-kepler-gk110-architecture-whitepaper-jp.pdf"
get "7_the_generations_tesla_fermi_kepler_maxwell_pascal" "nvidia2014k80k40ds.pdf" "https://hpc-test.llnl.gov/sites/default/files/Tesla-K80K40-datasheet.pdf"
get "7_the_generations_tesla_fermi_kepler_maxwell_pascal" "nvidia2016m40ds.pdf" "https://images.nvidia.com/content/tesla/pdf/nvidia-teslam40-datasheet.pdf"
get "7_the_generations_tesla_fermi_kepler_maxwell_pascal" "nvidia2016p100wp.pdf" "https://techpowerup.com/gpu-specs/docs/nvidia-gp100-architecture.pdf"
echo "manual: 7_the_generations_tesla_fermi_kepler_maxwell_pascal/nvidia2016p100pr  ->  https://www.globenewswire.com/news-release/2016/04/05/1149699/0/en/NVIDIA-Delivers-Massive-Performance-Leap-for-Deep-Learning-HPC-Applications-With-NVIDIA-Tesla-P100-Accelerators.html"   # not a direct PDF; open in a browser
echo "manual: 7_the_generations_tesla_fermi_kepler_maxwell_pascal/foley2017pascal  ->  https://doi.org/10.1109/MM.2017.37"   # not a direct PDF; open in a browser
echo "manual: 7_the_generations_tesla_fermi_kepler_maxwell_pascal/nvidia2015m40pr  ->  https://www.globenewswire.com/news-release/2015/11/10/1149678/0/en/New-NVIDIA-Hyperscale-Accelerators-Boost-Machine-Learning-Throughput-for-Web-Data-Centers.html"   # not a direct PDF; open in a browser
get "7_the_generations_tesla_fermi_kepler_maxwell_pascal" "krizhevsky2012alexnet.pdf" "https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf"
get "7_the_generations_tesla_fermi_kepler_maxwell_pascal" "chetlur2014cudnn.pdf" "https://arxiv.org/pdf/1410.0759"
get "7_the_generations_tesla_fermi_kepler_maxwell_pascal" "raina2009gpudl.pdf" "https://icml.cc/imls/conferences/2009/papers/218.pdf"
get "7_the_generations_tesla_fermi_kepler_maxwell_pascal" "ciresan2010deepbig.pdf" "https://arxiv.org/pdf/1003.0358"
echo "manual: 7_the_generations_tesla_fermi_kepler_maxwell_pascal/nvidia2015m40alexnet  ->  https://www.tomshardware.com/reviews/nvidia-geforce-gtx-titan-x-gm200-maxwell,4091.html"   # not a direct PDF; open in a browser
get "8_volta_and_the_tensor_core" "nvidia2017voltawp.pdf" "https://www.ece.lsu.edu/gp/refs/volta-architecture-whitepaper.pdf"
echo "manual: 8_volta_and_the_tensor_core/choquette2018volta  ->  https://doi.org/10.1109/MM.2018.022071134"   # not a direct PDF; open in a browser
get "8_volta_and_the_tensor_core" "nvidia2018v100ds.pdf" "https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf"
get "8_volta_and_the_tensor_core" "jia2018volta.pdf" "https://arxiv.org/pdf/1804.06826"
get "8_volta_and_the_tensor_core" "markidis2018tensorcore.pdf" "https://arxiv.org/pdf/1803.04014"
get "8_volta_and_the_tensor_core" "micikevicius2017mixed.pdf" "https://arxiv.org/pdf/1710.03740"
echo "manual: 8_volta_and_the_tensor_core/nvidia2017voltapr  ->  https://www.edge-ai-vision.com/2017/05/nvidia-launches-revolutionary-volta-gpu-platform-fueling-next-era-of-ai-and-high-performance-computing/"   # not a direct PDF; open in a browser
get "9_systems_interconnect_supercomputers_the_alternatives" "hotchips2016pascal.pdf" "https://old.hotchips.org/wp-content/uploads/hc_archives/hc28/HC28.22-Monday-Epub/HC28.22.10-GPU-HPC-Epub/HC28.22.121-Pascal-GPU-DanskinFoley-NVIDIA-v06-6_7.pdf"
get "9_systems_interconnect_supercomputers_the_alternatives" "hotchips2018nvswitch.pdf" "https://www.old.hotchips.org/hc30/2conf/2.01_Nvidia_NVswitch_HotChips2018_DGX2NVS_Final.pdf"
echo "manual: 9_systems_interconnect_supercomputers_the_alternatives/hpcwire2018dgx2  ->  https://www.hpcwire.com/2018/03/27/nvidia-ups-hardware-game-with-16-gpu-dgx-2-server-and-nvswitch/"   # not a direct PDF; open in a browser
get "9_systems_interconnect_supercomputers_the_alternatives" "olcf2018summit.pdf" "https://www.olcf.ornl.gov/wp-content/uploads/2018/06/Summit_bythenumbers_FIN-1.pdf"
echo "manual: 9_systems_interconnect_supercomputers_the_alternatives/llnl2018sierra  ->  https://hpc.llnl.gov/hardware/platforms/sierra"   # not a direct PDF; open in a browser
echo "manual: 9_systems_interconnect_supercomputers_the_alternatives/top500_2010_11  ->  https://top500.org/lists/top500/2010/11/"   # not a direct PDF; open in a browser
echo "manual: 9_systems_interconnect_supercomputers_the_alternatives/nvidia2010top5pr  ->  https://www.globenewswire.com/news-release/2010/11/15/1149888/0/en/Three-of-the-World-s-Top-Five-Supercomputers-Powered-by-NVIDIA-Tesla-GPUs.html"   # not a direct PDF; open in a browser
echo "manual: 9_systems_interconnect_supercomputers_the_alternatives/top500_2012_11  ->  https://top500.org/lists/top500/2012/11/"   # not a direct PDF; open in a browser
echo "manual: 9_systems_interconnect_supercomputers_the_alternatives/top500_2017_11  ->  https://top500.org/lists/top500/list/2017/11"   # not a direct PDF; open in a browser
echo "manual: 9_systems_interconnect_supercomputers_the_alternatives/top500_2018_06  ->  https://www.top500.org/lists/2018/06/"   # not a direct PDF; open in a browser
echo "manual: 9_systems_interconnect_supercomputers_the_alternatives/top500_2018_11  ->  https://top500.org/lists/2018/11/"   # not a direct PDF; open in a browser
echo "manual: 9_systems_interconnect_supercomputers_the_alternatives/top500_2017_pizdaint  ->  https://www.top500.org/news/swiss-deploy-worlds-fastest-gpu-powered-supercomputer"   # not a direct PDF; open in a browser
echo "manual: 9_systems_interconnect_supercomputers_the_alternatives/seiler2008larrabee  ->  https://doi.org/10.1145/1360612.1360617"   # not a direct PDF; open in a browser
echo "manual: 9_systems_interconnect_supercomputers_the_alternatives/hpcwire2009larrabee  ->  https://www.hpcwire.com/2009/12/07/intel_cancels_2010_larrabee_debut/"   # not a direct PDF; open in a browser
echo "manual: 9_systems_interconnect_supercomputers_the_alternatives/hpcwire2012knc  ->  https://www.hpcwire.com/2012/11/12/intel_brings_manycore_x86_to_market_with_knights_corner/"   # not a direct PDF; open in a browser
echo "manual: 9_systems_interconnect_supercomputers_the_alternatives/top500news2016knl  ->  https://www.top500.org/news/intel-takes-on-nvidia-with-knights-landing-launch/"   # not a direct PDF; open in a browser
echo "manual: 9_systems_interconnect_supercomputers_the_alternatives/hpcwire2018knlend  ->  https://www.hpcwire.com/2018/07/25/end-of-the-road-for-knights-landing-phi"   # not a direct PDF; open in a browser
echo "manual: 9_systems_interconnect_supercomputers_the_alternatives/amd2014s9150  ->  https://www.globenewswire.com/news-release/2014/08/06/1011324/0/en/AMD-Unleashes-World-s-Most-Powerful-Server-GPU-for-HPC.html"   # not a direct PDF; open in a browser
echo "manual: 9_systems_interconnect_supercomputers_the_alternatives/amd2015boltzmann  ->  https://ir.amd.com/news-events/press-releases/detail/652/amd-launches-boltzmann-initiative-to-dramatically-reduce-barriers-to-gpu-computing-on-amd-fireprotm-graphics"   # not a direct PDF; open in a browser
echo "manual: 9_systems_interconnect_supercomputers_the_alternatives/amd2016rocm  ->  https://ir.amd.com/news-events/press-releases/detail/732/amd-releases-new-version-of-rocm-the-most-versatile-open-source-platform-for-gpu-computing"   # not a direct PDF; open in a browser
get "10_practice_textbooks_and_retrospectives" "halfhill2009fermi.pdf" "https://www.halfhill.com/mpr/2009/234001.pdf"
get "10_practice_textbooks_and_retrospectives" "nvidia2014gk110210wp.pdf" "https://techpowerup.com/gpu-specs/docs/nvidia-gk110-210-compute-architecture.pdf"
echo "manual: 10_practice_textbooks_and_retrospectives/nvidia_cuda_guide  ->  https://docs.nvidia.com/cuda/archive/9.0/cuda-c-programming-guide/index.html"   # not a direct PDF; open in a browser
echo "manual: 10_practice_textbooks_and_retrospectives/hwu2022pmpp4  ->  https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0"   # not a direct PDF; open in a browser
echo "manual: 10_practice_textbooks_and_retrospectives/patterson2011caqa5  ->  https://www.elsevier.com/books/computer-architecture/hennessy/978-0-12-383872-8"   # not a direct PDF; open in a browser
echo "manual: 10_practice_textbooks_and_retrospectives/peddie2020geometryengine  ->  https://www.computer.org/publications/tech-news/chasing-pixels/geometry-engine"   # not a direct PDF; open in a browser

cp "$(dirname "$0")/reading_list.csv" "$(dirname "$0")/reading_list.bib" "$(dirname "$0")/README.md" "$ROOT/" 2>/dev/null
echo "downloaded: $ok   failed: $fail"
( cd "$(dirname "$ROOT")" && rm -f gpu_architectures_papers.zip && zip -qr gpu_architectures_papers.zip "$(basename "$ROOT")" ) && echo "wrote $(dirname "$ROOT")/gpu_architectures_papers.zip"
