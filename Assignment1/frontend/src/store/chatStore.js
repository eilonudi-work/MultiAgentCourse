import { create } from 'zustand';

/**
 * Chat state store using Zustand
 * Manages messages, streaming state, and chat interactions
 */
const useChatStore = create((set, get) => ({
  // State
  messages: [], // Array of message objects
  isStreaming: false,
  streamingMessageId: null,
  currentStreamContent: '',
  error: null,
  eventSource: null,

  // Actions

  /**
   * Set messages for the current conversation
   * @param {Array} messages - Array of message objects
   */
  setMessages: (messages) => set({ messages }),

  /**
   * Add a new message to the chat
   * @param {Object} message - Message object
   */
  addMessage: (message) => set((state) => ({
    messages: [...state.messages, message],
  })),

  /**
   * Update a specific message
   * @param {string} messageId - Message ID
   * @param {Object} updates - Fields to update
   */
  updateMessage: (messageId, updates) => set((state) => ({
    messages: state.messages.map((msg) =>
      msg.id === messageId ? { ...msg, ...updates } : msg
    ),
  })),

  /**
   * Start streaming a new message
   * @param {string} messageId - ID of the assistant message being streamed
   */
  startStreaming: (messageId) => set({
    isStreaming: true,
    streamingMessageId: messageId,
    currentStreamContent: '',
    error: null,
  }),

  /**
   * Append token to the current streaming message
   * @param {string} token - Token to append
   */
  appendStreamToken: (token) => set((state) => {
    const newContent = state.currentStreamContent + token;

    // Update the message in the messages array
    const updatedMessages = state.messages.map((msg) =>
      msg.id === state.streamingMessageId
        ? { ...msg, content: newContent }
        : msg
    );

    return {
      currentStreamContent: newContent,
      messages: updatedMessages,
    };
  }),

  /**
   * Stop streaming and finalize the message
   * @param {Object} finalData - Final message data from server
   */
  stopStreaming: (finalData = {}) => set((state) => {
    // Update the final message with any additional data
    const updatedMessages = state.streamingMessageId
      ? state.messages.map((msg) =>
          msg.id === state.streamingMessageId
            ? { ...msg, ...finalData, isStreaming: false }
            : msg
        )
      : state.messages;

    return {
      isStreaming: false,
      streamingMessageId: null,
      currentStreamContent: '',
      messages: updatedMessages,
      eventSource: null,
    };
  }),

  /**
   * Set the EventSource instance for stream control
   * @param {Object} eventSource - EventSource instance
   */
  setEventSource: (eventSource) => set({ eventSource }),

  /**
   * Stop the current streaming generation
   */
  stopGeneration: () => {
    const { eventSource } = get();
    if (eventSource) {
      eventSource.close();
      set({
        isStreaming: false,
        streamingMessageId: null,
        currentStreamContent: '',
        eventSource: null,
      });
    }
  },

  /**
   * Set error state
   * @param {Error|string} error - Error object or message
   */
  setError: (error) => set({
    error: typeof error === 'string' ? error : error.message,
    isStreaming: false,
    streamingMessageId: null,
    currentStreamContent: '',
  }),

  /**
   * Clear error state
   */
  clearError: () => set({ error: null }),

  /**
   * Clear all messages (for new conversation or reset)
   */
  clearMessages: () => set({
    messages: [],
    isStreaming: false,
    streamingMessageId: null,
    currentStreamContent: '',
    error: null,
    eventSource: null,
  }),

  /**
   * Delete a specific message
   * @param {string} messageId - Message ID to delete
   */
  deleteMessage: (messageId) => set((state) => ({
    messages: state.messages.filter((msg) => msg.id !== messageId),
  })),
}));

export default useChatStore;
