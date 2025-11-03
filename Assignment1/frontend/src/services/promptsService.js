import apiClient from './api';

/**
 * System prompts service for managing prompt templates
 */
const promptsService = {
  /**
   * Get predefined system prompt templates
   * @returns {Promise} List of prompt templates
   */
  async getTemplates() {
    try {
      const response = await apiClient.get('/api/prompts/templates');
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
        return new Error('Prompts endpoint not found.');
      } else if (status === 500) {
        return new Error('Server error. Please try again later.');
      } else {
        return new Error(data.detail || data.message || 'Failed to fetch templates.');
      }
    } else if (error.request) {
      return new Error('Cannot connect to server. Please check if the backend is running.');
    } else {
      return new Error(error.message || 'An unexpected error occurred.');
    }
  },
};

export default promptsService;
