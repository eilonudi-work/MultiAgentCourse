import { useState, useRef, useEffect } from 'react';

/**
 * ChatInput component
 * Multi-line textarea with auto-resize, send button, and keyboard shortcuts
 */
const ChatInput = ({ onSend, disabled = false, placeholder = 'Type a message...' }) => {
  const [message, setMessage] = useState('');
  const [charCount, setCharCount] = useState(0);
  const textareaRef = useRef(null);
  const maxChars = 10000;

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [message]);

  const handleChange = (e) => {
    const value = e.target.value;
    if (value.length <= maxChars) {
      setMessage(value);
      setCharCount(value.length);
    }
  };

  const handleSubmit = () => {
    const trimmed = message.trim();
    if (trimmed && !disabled) {
      onSend(trimmed);
      setMessage('');
      setCharCount(0);
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e) => {
    // Submit on Cmd+Enter or Ctrl+Enter
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Input container */}
        <div className="relative flex items-center gap-2">
          {/* Textarea */}
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={handleChange}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              disabled={disabled}
              className="w-full textarea-field min-h-[44px] max-h-[200px] resize-none pr-16"
              aria-label="Message input"
              rows={1}
            />

            {/* Character counter */}
            <div
              className={`absolute bottom-2 right-2 text-xs ${
                charCount > maxChars * 0.9
                  ? 'text-red-500'
                  : 'text-gray-400 dark:text-gray-500'
              }`}
            >
              {charCount}/{maxChars}
            </div>
          </div>

          {/* Send button */}
          <button
            onClick={handleSubmit}
            disabled={disabled || !message.trim()}
            className="btn-primary px-6 py-3 h-[44px] shrink-0"
            aria-label="Send message"
            title="Send message (Cmd/Ctrl + Enter)"
          >
            <svg
              className="w-5 h-5 rotate-90"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
              />
            </svg>
          </button>
        </div>

        {/* Help text */}
        <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 flex items-center justify-between">
          <span>Press Cmd+Enter to send</span>
          {disabled && (
            <span className="text-amber-600 dark:text-amber-500">
              Waiting for response...
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatInput;
