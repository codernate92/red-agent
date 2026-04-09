#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
STAGE_DIR="${BUILD_DIR}/arxiv_source"
DIST_DIR="${ROOT_DIR}/dist"
BUNDLE_PATH="${DIST_DIR}/red-agent-arxiv-source.tar.gz"

mkdir -p "${BUILD_DIR}" "${DIST_DIR}"
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"

# Compile once to ensure a fresh manuscript PDF exists.
tectonic --outdir "${ROOT_DIR}" "${ROOT_DIR}/main.tex"

cp "${ROOT_DIR}/main.tex" "${STAGE_DIR}/"
cp "${ROOT_DIR}/abstract.tex" "${STAGE_DIR}/"
cp "${ROOT_DIR}/references.bib" "${STAGE_DIR}/"
cp -R "${ROOT_DIR}/sections" "${STAGE_DIR}/"
cp -R "${ROOT_DIR}/tables" "${STAGE_DIR}/"
cp -R "${ROOT_DIR}/figures" "${STAGE_DIR}/"

# Build with intermediates and include main.bbl when available.
tectonic --keep-intermediates --outdir "${BUILD_DIR}" "${ROOT_DIR}/main.tex"
if [[ -f "${BUILD_DIR}/main.bbl" ]]; then
  cp "${BUILD_DIR}/main.bbl" "${STAGE_DIR}/"
fi

tar -czf "${BUNDLE_PATH}" -C "${STAGE_DIR}" .
echo "Created ${BUNDLE_PATH}"
