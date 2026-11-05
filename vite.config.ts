import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(async ({ mode }) => ({
  plugins: [
    react(),
    ...(mode === 'cloudflare'
      ? [
          (await import('@cloudflare/vite-plugin')).cloudflare({
            assetsOnly: true,
            remoteBindings: false,
            types: { generate: false },
          }),
        ]
      : []),
  ],
  server: { port: 5173, strictPort: true },
  preview: { port: 4173, strictPort: true },
}));
