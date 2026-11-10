import { defineConfig } from 'cf/config';

// Public workers.dev deployment authorized by the owner; version previews stay disabled.
export default defineConfig({
  accountId: '022d0703f819b09d7c97afd1fdcd191d',
  worker: {
    name: 'llm-mechanism-lab',
    compatibilityDate: '2026-11-05',
    assets: { notFoundHandling: 'single-page-application' },
    workersDev: true,
    previewUrls: false,
  },
});
