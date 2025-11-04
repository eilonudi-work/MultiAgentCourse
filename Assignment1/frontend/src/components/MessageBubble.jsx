import { useState, useEffect, useMemo, memo } from 'react';
import { marked } from 'marked';
import hljs from 'highlight.js';
import 'highlight.js/styles/github-dark.css';

/**
 * Configure marked.js with custom renderer
 */
const configureMarked = () => {
  const renderer = new marked.Renderer();

  // Custom code block renderer with syntax highlighting
  renderer.code = (code, language) => {
    const validLanguage = hljs.getLanguage(language) ? language : 'plaintext';
    const highlighted = hljs.highlight(code, { language: validLanguage }).value;

    return `
      <div class="code-block-wrapper relative group mb-4">
        <div class="flex items-center justify-between bg-gray-800 px-4 py-2 rounded-t-md">
          <span class="text-xs text-gray-400 font-mono">${validLanguage}</span>
          <button
            class="copy-button text-xs text-gray-400 hover:text-gray-200 px-2 py-1 rounded hover:bg-gray-700 transition-colors"
            data-code="${encodeURIComponent(code)}"
          >
            Copy
          </button>
        </div>
        <pre class="!mt-0 !rounded-t-none"><code class="hljs language-${validLanguage}">${highlighted}</code></pre>
      </div>
    `;
  };

  // Custom inline code renderer
  renderer.codespan = (code) => {
    return `<code class="inline-code">${code}</code>`;
  };

  marked.setOptions({
    renderer,
    breaks: true,
    gfm: true,
  });
};

// Initialize marked configuration
configureMarked();

/**
 * MessageBubble component
 * Displays a single message with markdown rendering and syntax highlighting
 * Optimized with React.memo and useMemo for better performance
 */
const MessageBubble = memo(({ message, isStreaming = false }) => {
  const isUser = message.role === 'user';

  // Memoize rendered markdown content
  const renderedContent = useMemo(() => {
    if (!message.content || isUser) return '';
    return marked.parse(message.content);
  }, [message.content, isUser]);

  // Handle copy button clicks
  useEffect(() => {
    const handleCopy = (e) => {
      if (e.target.classList.contains('copy-button')) {
        const code = decodeURIComponent(e.target.getAttribute('data-code'));
        navigator.clipboard.writeText(code).then(() => {
          const originalText = e.target.textContent;
          e.target.textContent = 'Copied!';
          setTimeout(() => {
            e.target.textContent = originalText;
          }, 2000);
        });
      }
    };

    document.addEventListener('click', handleCopy);
    return () => document.removeEventListener('click', handleCopy);
  }, []);

  return (
    <div
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4 px-4 animate-fadeIn`}
      role="article"
      aria-label={`${isUser ? 'User' : 'Assistant'} message`}
    >
      <div
        className={`max-w-[85%] md:max-w-[75%] rounded-lg px-4 py-3 ${
          isUser
            ? 'bg-blue-600 text-white dark:bg-blue-700'
            : 'bg-gray-100 text-gray-900 dark:bg-gray-800 dark:text-gray-100'
        }`}
      >
        {/* Message header with role indicator */}
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs font-semibold opacity-75">
            {isUser ? 'You' : 'Assistant'}
          </span>
          {message.timestamp && (
            <span className="text-xs opacity-50">
              {new Date(message.timestamp).toLocaleTimeString()}
            </span>
          )}
          {isStreaming && (
            <span className="flex items-center gap-1 text-xs opacity-75">
              <span className="inline-block w-1.5 h-1.5 bg-current rounded-full animate-pulse"></span>
              Generating...
            </span>
          )}
        </div>

        {/* Message content with markdown rendering */}
        {isUser ? (
          // User messages - plain text with line breaks
          <div className="whitespace-pre-wrap break-words">
            {message.content}
          </div>
        ) : (
          // Assistant messages - rendered markdown
          <div
            className="markdown-content prose prose-sm max-w-none dark:prose-invert"
            dangerouslySetInnerHTML={{ __html: renderedContent }}
          />
        )}

        {/* Streaming cursor */}
        {isStreaming && !isUser && (
          <span className="inline-block w-2 h-4 bg-current animate-pulse ml-1"></span>
        )}
      </div>
    </div>
  );
});

MessageBubble.displayName = 'MessageBubble';

export default MessageBubble;
