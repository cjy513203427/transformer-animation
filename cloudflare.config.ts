import { defineConfig } from 'cf/config';

// Local preparation only. Choose identity/access policy before enabling an entry point.
export default defineConfig({
  worker: {
    name: 'llm-mechanism-lab',
    compatibilityDate: '2026-11-05',
    assets: { notFoundHandling: 'single-page-application' },
    workersDev: false,
    previewUrls: false,
  },
});
