import { create } from 'zustand';
import conversationsService from '../services/conversationsService';

/**
 * Conversation state store using Zustand
 * Manages conversation list, selection, and CRUD operations
 */
const useConversationStore = create((set, get) => ({
  // State
  conversations: [],
  currentConversation: null,
  selectedModel: null,
  isLoading: false,
  error: null,
  hasMore: true,
  searchQuery: '',

  // Actions

  /**
   * Load conversations from the backend
   * @param {Object} params - Query parameters
   */
  loadConversations: async (params = {}) => {
    set({ isLoading: true, error: null });
    try {
      const data = await conversationsService.list(params);
      set({
        conversations: data.conversations || data,
        isLoading: false,
        hasMore: data.has_more ?? false,
      });
    } catch (error) {
      set({
        error: error.message,
        isLoading: false,
      });
    }
  },

  /**
   * Create a new conversation
   * @param {Object} data - Conversation data
   */
  createConversation: async (data) => {
    set({ isLoading: true, error: null });
    try {
      const conversation = await conversationsService.create(data);
      set((state) => ({
        conversations: [conversation, ...state.conversations],
        currentConversation: conversation,
        isLoading: false,
      }));
      return conversation;
    } catch (error) {
      set({
        error: error.message,
        isLoading: false,
      });
      throw error;
    }
  },

  /**
   * Select and load a conversation with its messages
   * @param {string} id - Conversation ID
   */
  selectConversation: async (id) => {
    set({ isLoading: true, error: null });
    try {
      const conversation = await conversationsService.get(id);
      set({
        currentConversation: conversation,
        isLoading: false,
      });
      return conversation;
    } catch (error) {
      set({
        error: error.message,
        isLoading: false,
      });
      throw error;
    }
  },

  /**
   * Update the current conversation
   * @param {string} id - Conversation ID
   * @param {Object} data - Updated data
   */
  updateConversation: async (id, data) => {
    set({ isLoading: true, error: null });
    try {
      const updated = await conversationsService.update(id, data);
      set((state) => ({
        conversations: state.conversations.map((conv) =>
          conv.id === id ? { ...conv, ...updated } : conv
        ),
        currentConversation: state.currentConversation?.id === id
          ? { ...state.currentConversation, ...updated }
          : state.currentConversation,
        isLoading: false,
      }));
      return updated;
    } catch (error) {
      set({
        error: error.message,
        isLoading: false,
      });
      throw error;
    }
  },

  /**
   * Delete a conversation
   * @param {string} id - Conversation ID
   */
  deleteConversation: async (id) => {
    set({ isLoading: true, error: null });
    try {
      await conversationsService.delete(id);
      set((state) => ({
        conversations: state.conversations.filter((conv) => conv.id !== id),
        currentConversation: state.currentConversation?.id === id
          ? null
          : state.currentConversation,
        isLoading: false,
      }));
    } catch (error) {
      set({
        error: error.message,
        isLoading: false,
      });
      throw error;
    }
  },

  /**
   * Set the selected model
   * @param {string} model - Model name
   */
  setSelectedModel: (model) => set({ selectedModel: model }),

  /**
   * Set current conversation directly (for optimistic updates)
   * @param {Object} conversation - Conversation object
   */
  setCurrentConversation: (conversation) => set({ currentConversation: conversation }),

  /**
   * Clear current conversation
   */
  clearCurrentConversation: () => set({ currentConversation: null }),

  /**
   * Set search query for filtering conversations
   * @param {string} query - Search query
   */
  setSearchQuery: (query) => set({ searchQuery: query }),

  /**
   * Get filtered conversations based on search query
   */
  getFilteredConversations: () => {
    const { conversations, searchQuery } = get();
    if (!searchQuery) return conversations;

    const query = searchQuery.toLowerCase();
    return conversations.filter((conv) =>
      conv.title?.toLowerCase().includes(query) ||
      conv.model_name?.toLowerCase().includes(query)
    );
  },

  /**
   * Clear error state
   */
  clearError: () => set({ error: null }),

  /**
   * Export a conversation
   * @param {string} id - Conversation ID
   * @param {string} format - Export format
   */
  exportConversation: async (id, format = 'json') => {
    try {
      const data = await conversationsService.export(id, format);
      return data;
    } catch (error) {
      set({ error: error.message });
      throw error;
    }
  },

  /**
   * Import conversations
   * @param {Object} data - Import data
   */
  importConversations: async (data) => {
    set({ isLoading: true, error: null });
    try {
      const result = await conversationsService.import(data);
      // Reload conversations after import
      await get().loadConversations();
      return result;
    } catch (error) {
      set({
        error: error.message,
        isLoading: false,
      });
      throw error;
    }
  },
}));

export default useConversationStore;
