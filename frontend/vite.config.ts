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
  },
})
