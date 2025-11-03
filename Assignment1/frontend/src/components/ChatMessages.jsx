import { useEffect, useRef, useState } from 'react';
import MessageBubble from './MessageBubble';
import LoadingSpinner from './LoadingSpinner';

/**
 * ChatMessages component
 * Displays message list with auto-scroll and virtual scrolling for performance
 */
const ChatMessages = ({ messages, isStreaming, isLoading }) => {
  const messagesEndRef = useRef(null);
  const containerRef = useRef(null);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [isNearBottom, setIsNearBottom] = useState(true);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (isNearBottom) {
      scrollToBottom();
    }
  }, [messages, isNearBottom]);

  // Always scroll when streaming starts or updates
  useEffect(() => {
    if (isStreaming) {
      scrollToBottom();
    }
  }, [isStreaming]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Handle scroll events to show/hide scroll button
  const handleScroll = () => {
    if (!containerRef.current) return;

    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;

    // Show button if scrolled up more than 200px from bottom
    setShowScrollButton(distanceFromBottom > 200);
    // Track if user is near bottom
    setIsNearBottom(distanceFromBottom < 100);
  };

  // Empty state
  if (!isLoading && messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="text-center max-w-md">
          <div className="text-6xl mb-4">💬</div>
          <h3 className="text-xl font-semibold text-gray-700 dark:text-gray-300 mb-2">
            Start a conversation
          </h3>
          <p className="text-gray-500 dark:text-gray-400">
            Send a message to begin chatting with the AI assistant
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 relative flex flex-col overflow-hidden">
      {/* Messages container */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto custom-scrollbar"
        role="log"
        aria-live="polite"
        aria-label="Chat messages"
      >
        <div className="py-4">
          {/* Loading state */}
          {isLoading && messages.length === 0 && (
            <div className="flex justify-center py-8">
              <LoadingSpinner size="md" message="Loading messages..." />
            </div>
          )}

          {/* Messages */}
          {messages.map((message, index) => (
            <MessageBubble
              key={message.id || index}
              message={message}
              isStreaming={isStreaming && index === messages.length - 1 && message.role === 'assistant'}
            />
          ))}

          {/* Scroll anchor */}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Scroll to bottom button */}
      {showScrollButton && (
        <button
          onClick={scrollToBottom}
          className="absolute bottom-4 right-4 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-full p-3 shadow-lg hover:shadow-xl transition-all hover:scale-110"
          aria-label="Scroll to bottom"
          title="Scroll to bottom"
        >
          <svg
            className="w-5 h-5 text-gray-600 dark:text-gray-300"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 14l-7 7m0 0l-7-7m7 7V3"
            />
          </svg>
        </button>
      )}
    </div>
  );
};

export default ChatMessages;
