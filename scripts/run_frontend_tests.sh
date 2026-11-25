#!/usr/bin/env bash
# Run the Phase 0 frontend baseline suite WITHOUT modifying frontend/package.json or
# frontend/package-lock.json (the lockfile is out of sync at the baseline -- D-12).
# Dependencies are installed into a scratch copy of the manifests and exposed to
# frontend/ via a node_modules symlink (node_modules/ is gitignored).
#   usage: scripts/run_frontend_tests.sh [--coverage]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${DOCINTEL_FE_DEPS:-${TMPDIR:-/tmp}/docintel-frontend-deps}"
TEST_DEPS=(vitest@4.1.11 @vitest/coverage-v8@4.1.11 jsdom@26.1.0
           @testing-library/react@16.3.0 @testing-library/dom@10.4.0 @testing-library/user-event@14.6.1)

if [ ! -x "$WORK/node_modules/.bin/vitest" ]; then
  mkdir -p "$WORK"
  cp "$ROOT/frontend/package.json" "$ROOT/frontend/package-lock.json" "$WORK/"
  # 1) app deps exactly as resolved from the committed lockfile; 2) test-only tools on top.
  #    --legacy-peer-deps works around an npm 10 arborist crash on vitest 4's optional peers.
  (cd "$WORK" && npm install --no-audit --no-fund \
     && rm -f package-lock.json node_modules/.package-lock.json \
     && npm install --no-audit --no-fund --legacy-peer-deps "${TEST_DEPS[@]}")
fi
if [ -e "$ROOT/frontend/node_modules" ] && [ ! -L "$ROOT/frontend/node_modules" ]; then
  echo "frontend/node_modules exists and is not a symlink; remove it or set DOCINTEL_FE_DEPS" >&2; exit 1
fi
ln -sfn "$WORK/node_modules" "$ROOT/frontend/node_modules"
cd "$ROOT/frontend"
if [ "${1:-}" = "--coverage" ]; then
  "$WORK/node_modules/.bin/vitest" run --coverage --coverage.include='src/**' --coverage.reporter=text
else
  "$WORK/node_modules/.bin/vitest" run
fi
git -C "$ROOT" diff --quiet -- frontend/package.json frontend/package-lock.json \
  || { echo "ERROR: frontend manifests were modified" >&2; exit 1; }
