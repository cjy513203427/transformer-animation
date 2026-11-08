import { expect, type Page } from '@playwright/test';

export async function openDirectory(page: Page) {
  const toggle = page.getByRole('button', { name: '学习目录', exact: true });
  await expect(toggle).toBeVisible();
  if ((await toggle.getAttribute('aria-expanded')) !== 'true') await toggle.click();
  const directory = page.getByRole('navigation', { name: '实验目录', exact: true });
  await expect(directory).toBeVisible();
  return directory;
}

export async function goToExperiment(page: Page, title: string) {
  const directory = await openDirectory(page);
  await directory.getByRole('link', { name: title, exact: true }).click();
}
