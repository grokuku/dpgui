import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiUrl = env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

  console.log(`[Vite Proxy] Configured to forward /api requests to: ${apiUrl}`);

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api': {
          target: apiUrl,
          changeOrigin: true,
          ws: true, // <--- AJOUT CRUCIAL : Active le proxy pour les WebSockets
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      },
    },
  }
})