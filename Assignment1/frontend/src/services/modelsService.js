import apiClient from './api';

/**
 * Models service for fetching available Ollama models
 */
const modelsService = {
  /**
   * List all available Ollama models
   * @returns {Promise} List of models
   */
  async list() {
    try {
      const response = await apiClient.get('/api/models/list');
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
        return new Error('Models endpoint not found.');
      } else if (status === 500) {
        return new Error('Server error. Please try again later.');
      } else if (status === 503) {
        return new Error('Ollama service is not available. Please ensure Ollama is running.');
      } else {
        return new Error(data.message || 'Failed to fetch models.');
      }
    } else if (error.request) {
      return new Error('Cannot connect to server. Please check if the backend is running.');
    } else {
      return new Error(error.message || 'An unexpected error occurred.');
    }
  },
};

export default modelsService;
