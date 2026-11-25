// Test-only config for the Phase 0 baseline frontend suite. Does not affect `vite`/`vite build`.
// Tooling is installed with `npm install --no-save` (see scripts/run_frontend_tests.sh) because
// package-lock.json is out of sync with package.json at the baseline checkpoint (D-12).
import { defineConfig } from "vitest/config"

export default defineConfig({
  test: {
    environment: "jsdom",
    include: ["tests/**/*.test.jsx"],
    restoreMocks: true,
  },
})
