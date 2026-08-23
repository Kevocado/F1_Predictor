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
  },
})
