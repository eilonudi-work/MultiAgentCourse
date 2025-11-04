import { test, expect } from '@playwright/test';
import { testCredentials, testModels, testMessages } from '../fixtures/testData.js';

/**
 * E2E Tests for Chat Flow
 * Tests the main chat interface and messaging functionality
 */

test.describe('Chat Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Set up authentication
    await page.goto('/');
    await page.evaluate((apiKey) => {
      localStorage.setItem('ollama_api_key', apiKey);
      localStorage.setItem('ollama_url', 'http://localhost:11434');
      localStorage.setItem('is_configured', 'true');
    }, testCredentials.validApiKey);

    // Mock models API
    await page.route('**/api/models', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ models: testModels }),
      });
    });

    // Mock conversations API
    await page.route('**/api/conversations', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ conversations: [] }),
      });
    });
  });

  test('should load chat page after authentication', async ({ page }) => {
    await page.goto('/chat');
    await expect(page).toHaveURL('/chat');
    await expect(page.getByText(/Select a model/i)).toBeVisible();
  });

  test('should display model selector', async ({ page }) => {
    await page.goto('/chat');

    // Click model selector
    const modelButton = page.getByRole('button', { name: /Select Model/i });
    await modelButton.click();

    // Check if models are displayed
    for (const model of testModels) {
      await expect(page.getByText(model.name)).toBeVisible();
    }
  });

  test('should create a new conversation', async ({ page }) => {
    await page.goto('/chat');

    // Mock create conversation API
    await page.route('**/api/conversations', (route) => {
      if (route.request().method() === 'POST') {
        route.fulfill({
          status: 201,
          contentType: 'application/json',
          body: JSON.stringify({
            id: 'new-conv-1',
            title: 'New Conversation',
            model: 'llama2',
            created_at: new Date().toISOString(),
          }),
        });
      } else {
        route.continue();
      }
    });

    // Click new conversation button
    await page.getByRole('button', { name: /New Chat/i }).click();

    // Check if new conversation is created
    await expect(page.getByText(/New Conversation/i)).toBeVisible();
  });

  test('should send a message', async ({ page }) => {
    await page.goto('/chat');

    // Select a model first
    await page.route('**/api/chat/stream', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: 'data: {"content":"Hello! How can I help you?"}\n\n',
      });
    });

    // Type and send message
    const input = page.getByPlaceholder(/Type your message/i);
    await input.fill(testMessages.simple);
    await input.press('Enter');

    // Check if message appears
    await expect(page.getByText(testMessages.simple)).toBeVisible();
  });

  test('should display streaming response', async ({ page }) => {
    await page.goto('/chat');

    // Mock streaming response
    await page.route('**/api/chat/stream', (route) => {
      const stream = [
        'data: {"content":"Hello"}\n\n',
        'data: {"content":" there"}\n\n',
        'data: {"content":"!"}\n\n',
      ].join('');

      route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: stream,
      });
    });

    const input = page.getByPlaceholder(/Type your message/i);
    await input.fill('Hi');
    await input.press('Enter');

    // Wait for response
    await expect(page.getByText(/Hello there!/i)).toBeVisible({ timeout: 5000 });
  });

  test('should render markdown in messages', async ({ page }) => {
    await page.goto('/chat');

    // Mock response with markdown
    await page.route('**/api/chat/stream', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: 'data: {"content":"# Heading\\n\\n**Bold** text"}\n\n',
      });
    });

    const input = page.getByPlaceholder(/Type your message/i);
    await input.fill('Test markdown');
    await input.press('Enter');

    // Check for rendered markdown
    await expect(page.locator('h1').filter({ hasText: 'Heading' })).toBeVisible();
    await expect(page.locator('strong').filter({ hasText: 'Bold' })).toBeVisible();
  });

  test('should display code blocks with syntax highlighting', async ({ page }) => {
    await page.goto('/chat');

    // Mock response with code
    await page.route('**/api/chat/stream', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: 'data: {"content":"```javascript\\nconst x = 1;\\n```"}\n\n',
      });
    });

    const input = page.getByPlaceholder(/Type your message/i);
    await input.fill('Show me code');
    await input.press('Enter');

    // Check for code block
    await expect(page.locator('pre code')).toBeVisible();
    await expect(page.locator('pre code')).toContainText('const x = 1');
  });

  test('should handle keyboard shortcuts', async ({ page }) => {
    await page.goto('/chat');

    // Test Shift+? for shortcuts modal
    await page.keyboard.press('Shift+?');
    await expect(page.getByText(/Keyboard Shortcuts/i)).toBeVisible();

    // Close with Escape
    await page.keyboard.press('Escape');
    await expect(page.getByText(/Keyboard Shortcuts/i)).not.toBeVisible();
  });

  test('should show error on failed message send', async ({ page }) => {
    await page.goto('/chat');

    // Mock error response
    await page.route('**/api/chat/stream', (route) => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Server error' }),
      });
    });

    const input = page.getByPlaceholder(/Type your message/i);
    await input.fill('Test error');
    await input.press('Enter');

    // Check for error message (toast or inline)
    await expect(page.getByText(/error/i)).toBeVisible({ timeout: 3000 });
  });

  test('should be accessible with keyboard navigation', async ({ page }) => {
    await page.goto('/chat');

    // Tab through interactive elements
    await page.keyboard.press('Tab');

    // Check focus management
    const focusedElement = await page.evaluate(() => document.activeElement.tagName);
    expect(['BUTTON', 'INPUT', 'TEXTAREA', 'A']).toContain(focusedElement);
  });

  test('should support dark mode toggle', async ({ page }) => {
    await page.goto('/chat');

    // Open settings
    await page.getByRole('button', { name: /Settings/i }).click();

    // Toggle dark mode
    const darkModeToggle = page.getByLabel(/Dark Mode/i);
    await darkModeToggle.click();

    // Check if dark class is applied
    const isDark = await page.evaluate(() => document.documentElement.classList.contains('dark'));
    expect(isDark).toBeTruthy();
  });

  test('should show network status indicator when offline', async ({ page }) => {
    await page.goto('/chat');

    // Simulate offline
    await page.context().setOffline(true);

    // Trigger network-dependent action
    const input = page.getByPlaceholder(/Type your message/i);
    await input.fill('Test offline');
    await input.press('Enter');

    // Check for network status indicator or error
    await expect(page.getByText(/offline|no connection/i)).toBeVisible({ timeout: 3000 });

    // Go back online
    await page.context().setOffline(false);
  });

  test('should show onboarding tour for new users', async ({ page }) => {
    // Clear onboarding flag
    await page.evaluate(() => {
      localStorage.removeItem('onboarding_completed');
    });

    await page.goto('/chat');

    // Check if onboarding is shown
    await expect(page.getByText(/Welcome to Ollama Web GUI/i)).toBeVisible({ timeout: 2000 });
  });

  test('should allow skipping onboarding tour', async ({ page }) => {
    await page.evaluate(() => {
      localStorage.removeItem('onboarding_completed');
    });

    await page.goto('/chat');

    // Wait for onboarding to appear
    await expect(page.getByText(/Welcome/i)).toBeVisible({ timeout: 2000 });

    // Click skip
    await page.getByRole('button', { name: /Skip/i }).click();

    // Onboarding should be gone
    await expect(page.getByText(/Welcome to Ollama Web GUI/i)).not.toBeVisible();
  });
});
