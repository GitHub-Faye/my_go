import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// 本地开发代理：/api → 后端 my_go (uvicorn :8000)
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});
