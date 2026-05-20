#!/bin/bash
# Fetch published-baseline repos as snapshots into methods/<name>/.
# Re-runnable (skips clone if dir already exists).
#
# Vendored shas (recorded 2026-05-20):
#   Fast-dLLM:      huggingface/Fast-dLLM @ <main, latest>
#   Dynamic-DLLM:   <ICLR 2026 repo URL — fill after public release>
#   Elastic-Cache:  YuyangSunshine/Elastic-Cache @ <main, 2025-10-21 sha>
#
# Usage: ./scripts/clone_baselines.sh

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p methods

if [ ! -d methods/fast_dllm ]; then
    echo "[clone] Fast-dLLM"
    git clone --depth 1 https://github.com/NVlabs/Fast-dLLM.git methods/fast_dllm
fi

if [ ! -d methods/elastic_cache ]; then
    echo "[clone] Elastic-Cache"
    git clone --depth 1 https://github.com/YuyangSunshine/Elastic-Cache.git methods/elastic_cache
fi

if [ ! -d methods/dynamic_dllm ]; then
    echo "[clone] Dynamic-DLLM  (skipping — public URL TBD; see paper Appendix)"
    # git clone --depth 1 <Dynamic-DLLM URL> methods/dynamic_dllm
fi

cat <<EOF

Vendored baselines installed under methods/{fast_dllm, elastic_cache, dynamic_dllm}.
Pin shas with:
  for d in methods/fast_dllm methods/elastic_cache methods/dynamic_dllm; do
    [ -d "\$d/.git" ] && (cd "\$d" && echo "\$d @ \$(git rev-parse HEAD)")
  done
EOF
