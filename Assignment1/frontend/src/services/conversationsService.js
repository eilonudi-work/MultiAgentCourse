import apiClient from './api';

/**
 * Conversations service for managing chat conversations
 */
const conversationsService = {
  /**
   * Create a new conversation
   * @param {Object} data - Conversation data
   * @param {string} data.title - Conversation title
   * @param {string} data.model_name - Model to use
   * @param {string} data.system_prompt - System prompt
   * @returns {Promise} Created conversation
   */
  async create(data) {
    try {
      const response = await apiClient.post('/api/conversations', data);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  /**
   * List all conversations for the current user
   * @param {Object} params - Query parameters
   * @param {number} params.skip - Number of items to skip
   * @param {number} params.limit - Number of items to return
   * @returns {Promise} List of conversations
   */
  async list(params = {}) {
    try {
      const response = await apiClient.get('/api/conversations', { params });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  /**
   * Get a single conversation with all messages
   * @param {string} id - Conversation ID
   * @returns {Promise} Conversation with messages
   */
  async get(id) {
    try {
      const response = await apiClient.get(`/api/conversations/${id}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  /**
   * Update a conversation
   * @param {string} id - Conversation ID
   * @param {Object} data - Updated data
   * @returns {Promise} Updated conversation
   */
  async update(id, data) {
    try {
      const response = await apiClient.put(`/api/conversations/${id}`, data);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  /**
   * Delete a conversation (soft delete)
   * @param {string} id - Conversation ID
   * @returns {Promise} Success message
   */
  async delete(id) {
    try {
      const response = await apiClient.delete(`/api/conversations/${id}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  /**
   * Export a conversation
   * @param {string} id - Conversation ID
   * @param {string} format - Export format ('json' or 'markdown')
   * @returns {Promise} Export data
   */
  async export(id, format = 'json') {
    try {
      const response = await apiClient.get(`/api/conversations/${id}/export`, {
        params: { format },
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  /**
   * Import conversations from file
   * @param {Object} data - Import data
   * @returns {Promise} Import result
   */
  async import(data) {
    try {
      const response = await apiClient.post('/api/conversations/import', data);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  /**
   * Handle API errors
   * @param {Error} error - Error object from axios
   * @returns {Error} Formatted error
   */
  handleError(error) {
    if (error.response) {
      const { status, data } = error.response;

      if (status === 401) {
        return new Error('Unauthorized. Please verify your API key.');
      } else if (status === 404) {
        return new Error('Conversation not found.');
      } else if (status === 400) {
        return new Error(data.detail || 'Invalid request data.');
      } else if (status === 500) {
        return new Error('Server error. Please try again later.');
      } else {
        return new Error(data.detail || data.message || 'Request failed.');
      }
    } else if (error.request) {
      return new Error('Cannot connect to server. Please check if the backend is running.');
    } else {
      return new Error(error.message || 'An unexpected error occurred.');
    }
  },
};

export default conversationsService;
