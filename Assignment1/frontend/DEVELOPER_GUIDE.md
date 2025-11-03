# Ollama Web GUI - Developer Guide

## Quick Start

### Prerequisites
- Node.js 18+
- npm or yarn
- Backend API running on http://localhost:8000
- Ollama service running on http://localhost:11434

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
# Opens on http://localhost:5173
```

### Production Build

```bash
npm run build
npm run preview
```

---

## Project Structure

```
src/
├── components/        # Reusable UI components
├── pages/            # Page-level components
├── services/         # API service layers
├── store/            # Zustand state stores
├── utils/            # Utility functions
├── App.jsx           # Root component with routing
├── main.jsx          # Application entry point
└── index.css         # Global styles + Tailwind
```

---

## State Management Architecture

### Stores Overview

#### 1. authStore.js
**Purpose:** Authentication and API key management

```javascript
import useAuthStore from './store/authStore';

// In component
const { apiKey, isAuthenticated, setApiKey, logout } = useAuthStore();
```

**State:**
- `apiKey` - User's API key
- `isAuthenticated` - Auth status
- `ollamaUrl` - Ollama service URL

**Actions:**
- `setApiKey(key)` - Store API key
- `logout()` - Clear authentication
- `checkAuth()` - Verify auth status

#### 2. configStore.js
**Purpose:** App configuration and preferences

```javascript
import useConfigStore from './store/configStore';

const { theme, toggleTheme, systemPrompt, setSystemPrompt } = useConfigStore();
```

**State:**
- `theme` - 'light' | 'dark'
- `temperature` - Model temperature (0-2)
- `maxTokens` - Max response length
- `systemPrompt` - System prompt text
- `sidebarCollapsed` - Sidebar state

**Actions:**
- `toggleTheme()` - Switch themes
- `initializeTheme()` - Load saved theme
- `setSystemPrompt(prompt)` - Update prompt
- `toggleSidebar()` - Toggle sidebar

#### 3. chatStore.js
**Purpose:** Chat messages and streaming state

```javascript
import useChatStore from './store/chatStore';

const { messages, addMessage, isStreaming, stopGeneration } = useChatStore();
```

**State:**
- `messages` - Array of message objects
- `isStreaming` - Streaming status
- `streamingMessageId` - ID of streaming message
- `currentStreamContent` - Accumulated tokens
- `eventSource` - SSE connection

**Actions:**
- `setMessages(messages)` - Load conversation
- `addMessage(message)` - Add new message
- `startStreaming(messageId)` - Begin stream
- `appendStreamToken(token)` - Add token
- `stopStreaming()` - End stream
- `stopGeneration()` - Cancel stream
- `clearMessages()` - Reset chat

#### 4. conversationStore.js
**Purpose:** Conversation list and selection

```javascript
import useConversationStore from './store/conversationStore';

const { conversations, currentConversation, selectConversation } = useConversationStore();
```

**State:**
- `conversations` - List of all conversations
- `currentConversation` - Active conversation
- `selectedModel` - Current model
- `searchQuery` - Filter query
- `isLoading` - Loading state

**Actions:**
- `loadConversations()` - Fetch list
- `createConversation(data)` - New conversation
- `selectConversation(id)` - Load conversation
- `updateConversation(id, data)` - Update
- `deleteConversation(id)` - Delete
- `exportConversation(id, format)` - Export
- `importConversations(data)` - Import

---

## Service Layer

### API Client Configuration

All services use the centralized `apiClient` from `/src/services/api.js`:

```javascript
import apiClient from './services/api';

// Automatically includes:
// - Base URL: VITE_API_BASE_URL
// - Authorization header with API key
// - Timeout: 30 seconds
// - Global error handling
```

### Available Services

#### conversationsService
```javascript
import conversationsService from './services/conversationsService';

// Create conversation
await conversationsService.create({
  title: 'New Chat',
  model_name: 'llama2',
  system_prompt: 'You are a helpful assistant.'
});

// List conversations
const { conversations } = await conversationsService.list({ skip: 0, limit: 50 });

// Get single conversation
const conversation = await conversationsService.get(conversationId);

// Update conversation
await conversationsService.update(conversationId, { title: 'Updated Title' });

// Delete conversation
await conversationsService.delete(conversationId);

// Export conversation
const data = await conversationsService.export(conversationId, 'json');

// Import conversations
await conversationsService.import(importData);
```

#### chatService
```javascript
import chatService from './services/chatService';

// Stream message
const stream = await chatService.streamMessage(
  {
    conversation_id: 'conv-123',
    message: 'Hello!',
    model_name: 'llama2',
    system_prompt: 'You are helpful.'
  },
  (token) => console.log('Token:', token),
  (data) => console.log('Complete:', data),
  (error) => console.error('Error:', error)
);

// Stop streaming
stream.stop();
```

#### modelsService
```javascript
import modelsService from './services/modelsService';

// List models
const { models } = await modelsService.list();
```

#### promptsService
```javascript
import promptsService from './services/promptsService';

// Get templates
const { templates } = await promptsService.getTemplates();
```

---

## Component Usage

### MessageBubble
```jsx
import MessageBubble from './components/MessageBubble';

<MessageBubble
  message={{
    id: 'msg-123',
    role: 'user', // or 'assistant'
    content: 'Hello!',
    timestamp: new Date().toISOString()
  }}
  isStreaming={false}
/>
```

### ChatMessages
```jsx
import ChatMessages from './components/ChatMessages';

<ChatMessages
  messages={messages}
  isStreaming={isStreaming}
  isLoading={false}
/>
```

### ChatInput
```jsx
import ChatInput from './components/ChatInput';

<ChatInput
  onSend={(message) => handleSendMessage(message)}
  disabled={isStreaming}
  placeholder="Type a message..."
/>
```

### ConversationSidebar
```jsx
import ConversationSidebar from './components/ConversationSidebar';

<ConversationSidebar
  onSelectConversation={(id) => handleSelect(id)}
  onNewChat={() => handleNewChat()}
/>
```

### ModelSelectorModal
```jsx
import ModelSelectorModal from './components/ModelSelectorModal';

<ModelSelectorModal
  isOpen={showModal}
  onClose={() => setShowModal(false)}
  currentModel="llama2"
  onSelectModel={(model) => handleModelChange(model)}
/>
```

### SettingsModal
```jsx
import SettingsModal from './components/SettingsModal';

<SettingsModal
  isOpen={showSettings}
  onClose={() => setShowSettings(false)}
/>
```

### ExportImportModal
```jsx
import ExportImportModal from './components/ExportImportModal';

<ExportImportModal
  isOpen={showExportImport}
  onClose={() => setShowExportImport(false)}
  conversationId={currentConversation?.id}
/>
```

---

## Styling Guide

### Tailwind CSS Classes

#### Buttons
```jsx
// Primary button
<button className="btn-primary">Click Me</button>

// Secondary button
<button className="btn-secondary">Cancel</button>

// Icon button
<button className="btn-icon">
  <svg>...</svg>
</button>
```

#### Inputs
```jsx
// Text input
<input className="input-field" />

// Textarea
<textarea className="textarea-field" />
```

#### Cards
```jsx
// Basic card
<div className="card">Content</div>

// Hoverable card
<div className="card card-hover">Content</div>
```

#### Scrollbars
```jsx
// Custom scrollbar
<div className="custom-scrollbar">
  Scrollable content
</div>
```

### Dark Mode

Dark mode is automatically applied via the `dark` class on `<html>`:

```jsx
// Light mode styles
<div className="bg-white text-gray-900">

// Dark mode styles (auto-applied)
<div className="dark:bg-gray-800 dark:text-gray-100">
```

### Markdown Content
```jsx
<div className="markdown-content prose prose-sm dark:prose-invert">
  {renderedMarkdown}
</div>
```

---

## Real-time Streaming Flow

### Implementation Pattern

```javascript
// 1. Add user message
const userMessage = {
  id: `user-${Date.now()}`,
  role: 'user',
  content: userInput,
  timestamp: new Date().toISOString()
};
addMessage(userMessage);

// 2. Create assistant message placeholder
const assistantMessageId = `assistant-${Date.now()}`;
const assistantMessage = {
  id: assistantMessageId,
  role: 'assistant',
  content: '',
  timestamp: new Date().toISOString()
};
addMessage(assistantMessage);

// 3. Start streaming
startStreaming(assistantMessageId);

// 4. Stream response
const stream = await chatService.streamMessage(
  { conversation_id, message: userInput, model_name, system_prompt },
  (token) => appendStreamToken(token), // Token callback
  (data) => stopStreaming(data),        // Complete callback
  (error) => setError(error)            // Error callback
);

// 5. Store EventSource for cancellation
setEventSource(stream.eventSource);
```

### Stopping Generation

```javascript
const { stopGeneration } = useChatStore();

// In component
<button onClick={stopGeneration}>
  Stop Generation
</button>
```

---

## Environment Variables

Create `.env` file in `/frontend`:

```env
# Backend API URL
VITE_API_BASE_URL=http://localhost:8000

# Ollama Service URL
VITE_OLLAMA_DEFAULT_URL=http://localhost:11434
```

Access in code:
```javascript
const apiUrl = import.meta.env.VITE_API_BASE_URL;
```

---

## Error Handling

### Service Layer
```javascript
try {
  const data = await conversationsService.create(payload);
} catch (error) {
  // error.message contains user-friendly message
  console.error(error.message);
}
```

### Component Level
```javascript
const [error, setError] = useState(null);

try {
  await someAsyncOperation();
} catch (err) {
  setError(err.message);
}

// Display error
{error && (
  <div className="text-red-600 dark:text-red-400">
    {error}
  </div>
)}
```

### Global Error Boundary
All routes are wrapped in `ErrorBoundary`:
```jsx
<ErrorBoundary>
  <YourComponent />
</ErrorBoundary>
```

---

## Performance Optimization

### Zustand Best Practices

```javascript
// ✅ Good: Subscribe to specific fields
const messages = useChatStore((state) => state.messages);
const addMessage = useChatStore((state) => state.addMessage);

// ❌ Avoid: Subscribe to entire store
const store = useChatStore();
```

### Memo Optimization

```javascript
import { memo } from 'react';

const MessageBubble = memo(({ message, isStreaming }) => {
  // Component re-renders only when props change
});
```

### Debouncing

```javascript
// Search input example
const [searchQuery, setSearchQuery] = useState('');

const handleSearch = (value) => {
  setSearchQuery(value);
  // Store automatically filters on next render
};
```

---

## Accessibility

### ARIA Labels
```jsx
<button aria-label="Send message">
  <svg>...</svg>
</button>

<input aria-label="Search conversations" />
```

### Keyboard Navigation
```jsx
<div
  role="button"
  tabIndex={0}
  onKeyDown={(e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      handleClick();
    }
  }}
>
  Interactive Element
</div>
```

### Focus Management
```jsx
// In modals
useEffect(() => {
  if (isOpen) {
    modalRef.current?.focus();
  }
}, [isOpen]);
```

---

## Testing

### Component Testing (Example)

```javascript
import { render, screen } from '@testing-library/react';
import MessageBubble from './MessageBubble';

test('renders user message', () => {
  const message = {
    id: '1',
    role: 'user',
    content: 'Hello',
    timestamp: new Date().toISOString()
  };

  render(<MessageBubble message={message} />);
  expect(screen.getByText('Hello')).toBeInTheDocument();
});
```

### Store Testing

```javascript
import { renderHook, act } from '@testing-library/react';
import useChatStore from './chatStore';

test('adds message', () => {
  const { result } = renderHook(() => useChatStore());

  act(() => {
    result.current.addMessage({ id: '1', content: 'Test' });
  });

  expect(result.current.messages).toHaveLength(1);
});
```

---

## Troubleshooting

### Common Issues

#### 1. CORS Errors
**Problem:** Cannot connect to backend
**Solution:** Ensure backend has CORS enabled for `http://localhost:5173`

#### 2. API Key Not Sent
**Problem:** 401 Unauthorized errors
**Solution:** Check localStorage has `ollama_api_key`:
```javascript
localStorage.getItem('ollama_api_key');
```

#### 3. Streaming Not Working
**Problem:** Messages don't stream
**Solution:**
- Verify SSE endpoint returns proper headers
- Check EventSource is supported in browser
- Inspect Network tab for SSE connection

#### 4. Dark Mode Not Persisting
**Problem:** Theme resets on reload
**Solution:** Call `initializeTheme()` in root component:
```javascript
useEffect(() => {
  useConfigStore.getState().initializeTheme();
}, []);
```

#### 5. Sidebar Not Closing on Mobile
**Problem:** Sidebar stays open after selection
**Solution:** Already handled in `ConversationSidebar.jsx` - check `isMobile` state

---

## Build and Deployment

### Production Build

```bash
npm run build
```

Output: `/dist` folder

### Environment-Specific Builds

```bash
# Development
VITE_API_BASE_URL=http://localhost:8000 npm run build

# Production
VITE_API_BASE_URL=https://api.yourdomain.com npm run build
```

### Serve Production Build

```bash
npm run preview
# or
npx serve dist
```

### Deploy to Static Hosting

The `/dist` folder can be deployed to:
- Vercel
- Netlify
- AWS S3 + CloudFront
- GitHub Pages
- Any static hosting service

---

## Code Style Guide

### Component Structure

```javascript
import { useState, useEffect } from 'react';
import useStore from './store/useStore';
import Component from './Component';

/**
 * Component description
 * @param {Object} props - Component props
 */
const MyComponent = ({ prop1, prop2 }) => {
  // 1. Hooks
  const storeValue = useStore((state) => state.value);
  const [localState, setLocalState] = useState(null);

  // 2. Effects
  useEffect(() => {
    // Effect logic
  }, []);

  // 3. Event handlers
  const handleClick = () => {
    // Handler logic
  };

  // 4. Render helpers
  const renderContent = () => {
    // Render logic
  };

  // 5. Return JSX
  return (
    <div className="component">
      {renderContent()}
    </div>
  );
};

export default MyComponent;
```

### Naming Conventions

- **Components:** PascalCase (`MessageBubble.jsx`)
- **Stores:** camelCase (`chatStore.js`)
- **Services:** camelCase (`conversationsService.js`)
- **Utilities:** camelCase (`formatDate.js`)
- **Constants:** UPPER_SNAKE_CASE
- **CSS Classes:** kebab-case or Tailwind utilities

---

## Resources

### Documentation
- [React 19 Docs](https://react.dev/)
- [Zustand Docs](https://docs.pmnd.rs/zustand/getting-started/introduction)
- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [Vite Docs](https://vitejs.dev/)

### Libraries
- [marked.js](https://marked.js.org/) - Markdown parser
- [highlight.js](https://highlightjs.org/) - Syntax highlighting
- [React Router](https://reactrouter.com/) - Routing

### Tools
- [React DevTools](https://react.dev/learn/react-developer-tools)
- [Zustand DevTools](https://github.com/pmndrs/zustand#redux-devtools)

---

## Support

For issues or questions:
1. Check this guide
2. Review component JSDoc comments
3. Inspect browser console for errors
4. Check Network tab for API issues
5. Refer to PHASE2_IMPLEMENTATION_SUMMARY.md

---

**Last Updated:** 2025-11-04
**Version:** 2.0.0 (Phase 2 Complete)
