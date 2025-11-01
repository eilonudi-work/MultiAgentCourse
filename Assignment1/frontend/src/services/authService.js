import apiClient from './api';

/**
 * Authentication service for API key management
 */
const authService = {
  /**
   * Setup API key and Ollama URL
   * @param {string} apiKey - API key for backend authentication
   * @param {string} ollamaUrl - Ollama server URL
   * @returns {Promise} API response
   */
  async setup(apiKey, ollamaUrl) {
    try {
      const response = await apiClient.post('/api/auth/setup', {
        api_key: apiKey,
        ollama_url: ollamaUrl,
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  /**
   * Verify API key
   * @param {string} apiKey - API key to verify
   * @returns {Promise} Verification result
   */
  async verify(apiKey) {
    try {
      const response = await apiClient.post('/api/auth/verify', {
        api_key: apiKey,
      });
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
        return new Error('Invalid API key. Please check your credentials.');
      } else if (status === 404) {
        return new Error('Endpoint not found. Please check backend server.');
      } else if (status === 500) {
        return new Error('Server error. Please try again later.');
      } else {
        return new Error(data.message || 'An error occurred during authentication.');
      }
    } else if (error.request) {
      return new Error('Cannot connect to server. Please check if the backend is running.');
    } else {
      return new Error(error.message || 'An unexpected error occurred.');
    }
  },
};

export default authService;
