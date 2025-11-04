import { test, expect } from '@playwright/test';
import { testCredentials } from '../fixtures/testData.js';

/**
 * E2E Tests for Setup Flow
 * Tests the initial configuration and onboarding process
 */

test.describe('Setup Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Clear localStorage before each test
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
  });

  test('should redirect to setup page when not configured', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveURL('/setup');
  });

  test('should display setup form with required fields', async ({ page }) => {
    await page.goto('/setup');

    // Check for form elements
    await expect(page.getByLabel(/API Key/i)).toBeVisible();
    await expect(page.getByLabel(/Ollama URL/i)).toBeVisible();
    await expect(page.getByRole('button', { name: /Connect/i })).toBeVisible();
  });

  test('should show validation errors for empty fields', async ({ page }) => {
    await page.goto('/setup');

    // Click connect without filling fields
    await page.getByRole('button', { name: /Connect/i }).click();

    // Check for validation messages
    await expect(page.getByText(/API key is required/i)).toBeVisible();
  });

  test('should show error for invalid API key', async ({ page }) => {
    await page.goto('/setup');

    // Fill in invalid credentials
    await page.getByLabel(/API Key/i).fill(testCredentials.invalidApiKey);
    await page.getByLabel(/Ollama URL/i).fill(testCredentials.ollamaUrl);

    // Mock API error response
    await page.route('**/api/auth/validate', (route) => {
      route.fulfill({
        status: 401,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Invalid API key' }),
      });
    });

    // Submit form
    await page.getByRole('button', { name: /Connect/i }).click();

    // Check for error message
    await expect(page.getByText(/Invalid API key/i)).toBeVisible();
  });

  test('should successfully configure and redirect to chat', async ({ page }) => {
    await page.goto('/setup');

    // Fill in valid credentials
    await page.getByLabel(/API Key/i).fill(testCredentials.validApiKey);
    await page.getByLabel(/Ollama URL/i).fill(testCredentials.ollamaUrl);

    // Mock successful API response
    await page.route('**/api/auth/validate', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ valid: true }),
      });
    });

    // Submit form
    await page.getByRole('button', { name: /Connect/i }).click();

    // Should redirect to chat
    await expect(page).toHaveURL('/chat');
  });

  test('should persist configuration in localStorage', async ({ page }) => {
    await page.goto('/setup');

    // Fill in credentials
    await page.getByLabel(/API Key/i).fill(testCredentials.validApiKey);
    await page.getByLabel(/Ollama URL/i).fill(testCredentials.ollamaUrl);

    // Mock successful response
    await page.route('**/api/auth/validate', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ valid: true }),
      });
    });

    await page.getByRole('button', { name: /Connect/i }).click();

    // Check localStorage
    const apiKey = await page.evaluate(() => localStorage.getItem('ollama_api_key'));
    expect(apiKey).toBe(testCredentials.validApiKey);
  });

  test('should be keyboard accessible', async ({ page }) => {
    await page.goto('/setup');

    // Tab through form elements
    await page.keyboard.press('Tab');
    await expect(page.getByLabel(/API Key/i)).toBeFocused();

    await page.keyboard.press('Tab');
    await expect(page.getByLabel(/Ollama URL/i)).toBeFocused();

    await page.keyboard.press('Tab');
    await expect(page.getByRole('button', { name: /Connect/i })).toBeFocused();
  });

  test('should show loading state during validation', async ({ page }) => {
    await page.goto('/setup');

    await page.getByLabel(/API Key/i).fill(testCredentials.validApiKey);
    await page.getByLabel(/Ollama URL/i).fill(testCredentials.ollamaUrl);

    // Mock delayed response
    await page.route('**/api/auth/validate', async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ valid: true }),
      });
    });

    await page.getByRole('button', { name: /Connect/i }).click();

    // Check for loading indicator
    await expect(page.getByRole('status')).toBeVisible();
  });
});
