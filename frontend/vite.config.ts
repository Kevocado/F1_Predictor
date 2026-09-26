/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5174,
    // Bind all interfaces, not just localhost — needed to reach this from
    // another device (phone, or another machine on the LAN/Tailscale).
    host: true,
    // Same-origin /api, as in the public Docker build: the dev server
    // forwards it to the local backend.
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/test/setup.ts',
    // Session times render in the viewer's zone; pin one so tests are stable.
    env: { TZ: 'America/Chicago' },
    // These are real DOM interaction tests in jsdom, and 5s is vitest's
    // default, not a budget derived from anything. Under load (four
    // concurrent vitest processes on ten cores) whole files blew past it and
    // failed with "Test timed out in 5000ms" while asserting nothing wrong:
    // reproduced 0-for-12 before the determinism fixes in
    // src/test/no-real-time.test.ts, and the residue after them was still
    // wall-clock, not logic. 20s costs an idle run nothing — a passing test
    // exits when it is done, not when the timer expires — and a genuinely
    // hung test still fails in a time a human will wait for.
    testTimeout: 20_000,
  },
})
