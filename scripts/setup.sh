#!/usr/bin/env bash
# Clone SocioVerse2 at the pinned commit next to this repo's files, link the adacall studies into
# it, and install it (editable) into the active Python environment.
set -euo pipefail

SV2_COMMIT="${SV2_COMMIT:-9c3ac77}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -d SocioVerse2/.git ]; then
  git clone --quiet https://github.com/sii-research/SocioVerse2 SocioVerse2
fi
if ! git -C SocioVerse2 cat-file -e "${SV2_COMMIT}^{commit}" 2>/dev/null; then
  git -C SocioVerse2 fetch --quiet --unshallow 2>/dev/null || git -C SocioVerse2 fetch --quiet
fi
git -C SocioVerse2 -c advice.detachedHead=false checkout --quiet "$SV2_COMMIT"

for s in adacall_od adacall_germany adacall_campus; do
  ln -sfn "../../studies/$s" "SocioVerse2/studies/$s"
done
python -m pip install --quiet -e "SocioVerse2[dev,llm,abm,viz]" scipy openpyxl
echo "SocioVerse2 @ $(git -C SocioVerse2 rev-parse --short HEAD), adacall studies linked"
