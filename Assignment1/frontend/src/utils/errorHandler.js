/**
 * Error handling utilities
 */

/**
 * Format API error for display
 * @param {Error} error - Error object
 * @returns {string} User-friendly error message
 */
export const formatApiError = (error) => {
  if (error.response) {
    const { status, data } = error.response;

    switch (status) {
      case 400:
        return data.message || 'Invalid request. Please check your input.';
      case 401:
        return 'Invalid API key. Please check your credentials.';
      case 403:
        return 'Access forbidden. You do not have permission to perform this action.';
      case 404:
        return 'Resource not found. Please check the endpoint.';
      case 500:
        return 'Server error. Please try again later.';
      case 503:
        return 'Service unavailable. Please ensure Ollama is running.';
      default:
        return data.message || `Request failed with status ${status}`;
    }
  } else if (error.request) {
    return 'Cannot connect to server. Please check if the backend is running on http://localhost:8000';
  } else {
    return error.message || 'An unexpected error occurred.';
  }
};

/**
 * Check if error is a network error
 * @param {Error} error - Error object
 * @returns {boolean}
 */
export const isNetworkError = (error) => {
  return error.message === 'Network Error' ||
         error.code === 'ECONNABORTED' ||
         !error.response;
};

/**
 * Check if error is an authentication error
 * @param {Error} error - Error object
 * @returns {boolean}
 */
export const isAuthError = (error) => {
  return error.response && error.response.status === 401;
};

/**
 * Retry failed request with exponential backoff
 * @param {Function} fn - Function to retry
 * @param {number} maxRetries - Maximum number of retries
 * @param {number} delay - Initial delay in ms
 * @returns {Promise}
 */
export const retryWithBackoff = async (fn, maxRetries = 3, delay = 1000) => {
  let lastError;

  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      // Don't retry on auth errors or client errors (4xx)
      if (isAuthError(error) || (error.response && error.response.status < 500)) {
        throw error;
      }

      // Wait before retrying (exponential backoff)
      if (i < maxRetries - 1) {
        const waitTime = delay * Math.pow(2, i);
        await new Promise(resolve => setTimeout(resolve, waitTime));
      }
    }
  }

  throw lastError;
};

/**
 * Log error to console (in development) or analytics (in production)
 * @param {Error} error - Error object
 * @param {string} context - Error context
 */
export const logError = (error, context = '') => {
  if (import.meta.env.DEV) {
    console.error(`[${context}]`, error);
  } else {
    // In production, you might want to send to analytics/monitoring service
    // e.g., Sentry, LogRocket, etc.
    console.error(`[${context}]`, error.message);
  }
};
