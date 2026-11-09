import { defineConfig } from 'cf/config';

// Local `cf build` only. Production deploys use wrangler.jsonc via Workers Builds;
// keep the name and URL settings here in sync with that file.
export default defineConfig({
  worker: {
    name: 'transformer-animation',
    compatibilityDate: '2026-11-05',
    assets: { notFoundHandling: 'single-page-application' },
    workersDev: true,
    previewUrls: false,
  },
});
