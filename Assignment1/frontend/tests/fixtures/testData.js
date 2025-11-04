/**
 * Test Data Fixtures
 * Provides mock data for E2E tests
 */

export const testCredentials = {
  validApiKey: 'test-api-key-12345',
  invalidApiKey: 'invalid-key',
  ollamaUrl: 'http://localhost:11434',
};

export const testModels = [
  {
    name: 'llama2',
    size: '4.1 GB',
    modified_at: '2024-01-15T10:30:00Z',
  },
  {
    name: 'mistral',
    size: '4.7 GB',
    modified_at: '2024-01-20T14:20:00Z',
  },
  {
    name: 'codellama',
    size: '3.8 GB',
    modified_at: '2024-01-10T09:15:00Z',
  },
];

export const testConversations = [
  {
    id: '1',
    title: 'Test Conversation 1',
    model: 'llama2',
    created_at: '2024-01-15T10:00:00Z',
    updated_at: '2024-01-15T10:30:00Z',
    messages: [
      {
        role: 'user',
        content: 'Hello, can you help me with JavaScript?',
      },
      {
        role: 'assistant',
        content: 'Of course! I\'d be happy to help you with JavaScript. What would you like to know?',
      },
    ],
  },
  {
    id: '2',
    title: 'Test Conversation 2',
    model: 'mistral',
    created_at: '2024-01-16T11:00:00Z',
    updated_at: '2024-01-16T11:45:00Z',
    messages: [
      {
        role: 'user',
        content: 'Explain React hooks',
      },
      {
        role: 'assistant',
        content: 'React Hooks are functions that let you use state and other React features in functional components...',
      },
    ],
  },
];

export const testMessages = {
  simple: 'Hello, world!',
  withCode: 'Here is a code example:\n```javascript\nconst hello = "world";\nconsole.log(hello);\n```',
  withMarkdown: '# Heading\n\n**Bold text** and *italic text*\n\n- List item 1\n- List item 2',
  long: 'This is a very long message that should test the scrolling behavior and text wrapping. '.repeat(20),
};

export const mockApiResponses = {
  modelsSuccess: {
    models: testModels,
  },
  conversationsSuccess: {
    conversations: testConversations,
  },
  chatStreamSuccess: [
    { content: 'Hello' },
    { content: ' there' },
    { content: '!' },
    { content: ' How' },
    { content: ' can' },
    { content: ' I' },
    { content: ' help' },
    { content: ' you?' },
  ],
  errorInvalidKey: {
    detail: 'Invalid API key',
  },
  errorServerError: {
    detail: 'Internal server error',
  },
  errorNetworkError: {
    message: 'Network Error',
  },
};

export default {
  testCredentials,
  testModels,
  testConversations,
  testMessages,
  mockApiResponses,
};
