import apiClient from './api';

/**
 * Configuration service for saving and retrieving settings
 */
const configService = {
  /**
   * Save configuration to backend
   * @param {Object} config - Configuration object
   * @returns {Promise} API response
   */
  async save(config) {
    try {
      const response = await apiClient.post('/api/config/save', config);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  /**
   * Get saved configuration from backend
   * @returns {Promise} Configuration object
   */
  async get() {
    try {
      const response = await apiClient.get('/api/config/get');
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
        return new Error('Configuration not found.');
      } else if (status === 500) {
        return new Error('Server error. Please try again later.');
      } else {
        return new Error(data.message || 'Configuration error occurred.');
      }
    } else if (error.request) {
      return new Error('Cannot connect to server. Please check if the backend is running.');
    } else {
      return new Error(error.message || 'An unexpected error occurred.');
    }
  },
};

export default configService;
