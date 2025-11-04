import { test, expect } from '@playwright/test';
import { testCredentials, testConversations } from '../fixtures/testData.js';

/**
 * E2E Tests for Export/Import Flow
 * Tests conversation export and import functionality
 */

test.describe('Export/Import Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Set up authentication
    await page.goto('/');
    await page.evaluate((apiKey) => {
      localStorage.setItem('ollama_api_key', apiKey);
      localStorage.setItem('ollama_url', 'http://localhost:11434');
      localStorage.setItem('is_configured', 'true');
    }, testCredentials.validApiKey);

    // Mock APIs
    await page.route('**/api/models', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ models: [] }),
      });
    });

    await page.route('**/api/conversations', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ conversations: testConversations }),
      });
    });
  });

  test('should open export/import modal', async ({ page }) => {
    await page.goto('/chat');

    // Click export/import button
    await page.getByRole('button', { name: /Export|Import/i }).click();

    // Check if modal is visible
    await expect(page.getByText(/Export.*Import/i)).toBeVisible();
  });

  test('should export single conversation', async ({ page }) => {
    await page.goto('/chat');

    // Open export modal
    await page.getByRole('button', { name: /Export/i }).click();

    // Select first conversation
    await page.getByRole('checkbox').first().check();

    // Setup download listener
    const downloadPromise = page.waitForEvent('download');

    // Click export button
    await page.getByRole('button', { name: /Export Selected/i }).click();

    // Wait for download
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/conversation.*\.json/);
  });

  test('should export all conversations', async ({ page }) => {
    await page.goto('/chat');

    // Open export modal
    await page.getByRole('button', { name: /Export/i }).click();

    // Click "Select All" checkbox
    await page.getByLabel(/Select All/i).check();

    // Setup download listener
    const downloadPromise = page.waitForEvent('download');

    // Click export button
    await page.getByRole('button', { name: /Export Selected/i }).click();

    // Wait for download
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/conversations.*\.json/);

    // Verify download content
    const content = await download.path();
    expect(content).toBeTruthy();
  });

  test('should import conversations from file', async ({ page }) => {
    await page.goto('/chat');

    // Open import modal
    await page.getByRole('button', { name: /Import/i }).click();

    // Create test file content
    const fileContent = JSON.stringify({
      conversations: testConversations,
    });

    // Mock file input
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.getByLabel(/Choose file|Import file/i).click();

    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles({
      name: 'test-conversations.json',
      mimeType: 'application/json',
      buffer: Buffer.from(fileContent),
    });

    // Mock import API
    await page.route('**/api/conversations/import', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ imported: testConversations.length }),
      });
    });

    // Click import button
    await page.getByRole('button', { name: /Import/i }).click();

    // Check for success message
    await expect(page.getByText(/imported successfully/i)).toBeVisible({ timeout: 3000 });
  });

  test('should show error for invalid import file', async ({ page }) => {
    await page.goto('/chat');

    // Open import modal
    await page.getByRole('button', { name: /Import/i }).click();

    // Create invalid file content
    const invalidContent = 'not valid json';

    // Mock file input
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.getByLabel(/Choose file|Import file/i).click();

    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles({
      name: 'invalid.json',
      mimeType: 'application/json',
      buffer: Buffer.from(invalidContent),
    });

    // Click import button
    await page.getByRole('button', { name: /Import/i }).click();

    // Check for error message
    await expect(page.getByText(/invalid.*file|error/i)).toBeVisible({ timeout: 3000 });
  });

  test('should validate export before downloading', async ({ page }) => {
    await page.goto('/chat');

    // Open export modal
    await page.getByRole('button', { name: /Export/i }).click();

    // Try to export without selecting any conversation
    const exportButton = page.getByRole('button', { name: /Export Selected/i });

    // Button should be disabled or show error
    const isDisabled = await exportButton.isDisabled();
    if (!isDisabled) {
      await exportButton.click();
      await expect(page.getByText(/select at least one/i)).toBeVisible();
    } else {
      expect(isDisabled).toBeTruthy();
    }
  });

  test('should show loading state during export', async ({ page }) => {
    await page.goto('/chat');

    // Open export modal
    await page.getByRole('button', { name: /Export/i }).click();

    // Select conversation
    await page.getByRole('checkbox').first().check();

    // Mock delayed export
    await page.route('**/api/conversations/export', async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(testConversations[0]),
      });
    });

    // Click export
    await page.getByRole('button', { name: /Export Selected/i }).click();

    // Check for loading indicator
    await expect(page.getByRole('status')).toBeVisible();
  });

  test('should show loading state during import', async ({ page }) => {
    await page.goto('/chat');

    // Open import modal
    await page.getByRole('button', { name: /Import/i }).click();

    // Prepare file
    const fileContent = JSON.stringify({ conversations: testConversations });

    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.getByLabel(/Choose file|Import file/i).click();

    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles({
      name: 'test.json',
      mimeType: 'application/json',
      buffer: Buffer.from(fileContent),
    });

    // Mock delayed import
    await page.route('**/api/conversations/import', async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ imported: testConversations.length }),
      });
    });

    // Click import
    await page.getByRole('button', { name: /Import/i }).click();

    // Check for loading indicator
    await expect(page.getByRole('status')).toBeVisible();
  });

  test('should close modal after successful export', async ({ page }) => {
    await page.goto('/chat');

    // Open export modal
    await page.getByRole('button', { name: /Export/i }).click();

    // Select and export
    await page.getByRole('checkbox').first().check();

    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: /Export Selected/i }).click();
    await downloadPromise;

    // Modal should close
    await expect(page.getByText(/Export.*Import/i)).not.toBeVisible({ timeout: 2000 });
  });

  test('should be accessible with keyboard', async ({ page }) => {
    await page.goto('/chat');

    // Open modal with keyboard
    await page.keyboard.press('Tab');
    // Navigate to export button
    let attempts = 0;
    while (attempts < 20) {
      const focusedText = await page.evaluate(() => document.activeElement?.textContent);
      if (focusedText?.match(/Export|Import/i)) {
        await page.keyboard.press('Enter');
        break;
      }
      await page.keyboard.press('Tab');
      attempts++;
    }

    // Check if modal opened
    await expect(page.getByText(/Export.*Import/i)).toBeVisible({ timeout: 3000 });

    // Close with Escape
    await page.keyboard.press('Escape');
    await expect(page.getByText(/Export.*Import/i)).not.toBeVisible();
  });
});
