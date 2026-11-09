import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';

// Cloudflare Web Analytics for transformer-animation.cjyhobbyprojects.workers.dev.
// The token is public by design (it ships in the page). Only `--mode deploy` builds include it,
// so dev, preview and e2e runs never report visits.
const WEB_ANALYTICS_SNIPPET =
  `<script type='module' src='https://static.cloudflareinsights.com/beacon.min.js' ` +
  `data-cf-beacon='{"token": "bfd1f7ed8b734b309e39dcf41074419c"}'></script>`;

function cloudflareWebAnalytics(): Plugin {
  return {
    name: 'cloudflare-web-analytics',
    transformIndexHtml(html) {
      if (!html.includes('</body>'))
        throw new Error('index.html has no </body> for the analytics snippet');
      return html.replace('</body>', `  ${WEB_ANALYTICS_SNIPPET}\n  </body>`);
    },
  };
}

export default defineConfig(async ({ mode }) => ({
  plugins: [
    react(),
    ...(mode === 'deploy' ? [cloudflareWebAnalytics()] : []),
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
