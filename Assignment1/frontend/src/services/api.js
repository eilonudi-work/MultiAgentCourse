import axios from 'axios';
import { retryRequest, isRetryableError } from '../utils/retryHandler';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Create axios instance with base configuration
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor - Add API key to headers
apiClient.interceptors.request.use(
  (config) => {
    const apiKey = localStorage.getItem('ollama_api_key');
    if (apiKey) {
      config.headers.Authorization = `Bearer ${apiKey}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor - Handle errors globally
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Create a structured error object
    const structuredError = {
      message: 'An error occurred',
      type: 'general',
      status: null,
      data: null,
      isRetryable: false,
    };

    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      structuredError.status = status;
      structuredError.data = data;

      switch (status) {
        case 401:
          structuredError.type = 'invalidApiKey';
          structuredError.message = 'Invalid API key. Please check your credentials.';
          break;
        case 404:
          structuredError.type = 'notFound';
          structuredError.message = data?.detail || 'Resource not found';
          break;
        case 500:
        case 502:
        case 503:
        case 504:
          structuredError.type = 'serverError';
          structuredError.message = 'Server error. Please try again later.';
          structuredError.isRetryable = true;
          break;
        case 429:
          structuredError.type = 'rateLimited';
          structuredError.message = 'Too many requests. Please slow down.';
          structuredError.isRetryable = true;
          break;
        default:
          structuredError.message = data?.detail || 'Request failed';
      }
    } else if (error.request) {
      // Request made but no response
      structuredError.type = 'networkError';
      structuredError.message = 'Unable to connect to the server. Please check your connection.';
      structuredError.isRetryable = true;
    } else if (error.code === 'ECONNABORTED') {
      // Request timeout
      structuredError.type = 'timeout';
      structuredError.message = 'Request timed out. Please try again.';
      structuredError.isRetryable = true;
    } else {
      // Something else happened
      structuredError.message = error.message || 'An unexpected error occurred';
    }

    // Check if error is retryable
    structuredError.isRetryable = structuredError.isRetryable || isRetryableError(error);

    // Log error details in development
    if (import.meta.env.DEV) {
      console.error('API Error:', structuredError);
    }

    // Attach structured error to the original error
    error.structuredError = structuredError;

    return Promise.reject(error);
  }
);

/**
 * Make a request with automatic retry logic
 * @param {Function} requestFn - Function that makes the API call
 * @param {Object} options - Retry options
 * @returns {Promise} - API response
 */
export const apiRequest = async (requestFn, options = {}) => {
  const {
    maxAttempts = 3,
    shouldRetry = (error) => error.structuredError?.isRetryable,
    onRetry = null,
    ...retryOptions
  } = options;

  return retryRequest(requestFn, {
    maxAttempts,
    shouldRetry,
    onRetry,
    ...retryOptions,
  });
};

/**
 * Get error type from API error
 * @param {Error} error - The error object
 * @returns {string} - Error type
 */
export const getErrorType = (error) => {
  return error.structuredError?.type || 'general';
};

/**
 * Get user-friendly error message
 * @param {Error} error - The error object
 * @returns {string} - Error message
 */
export const getErrorMessage = (error) => {
  return error.structuredError?.message || error.message || 'An unexpected error occurred';
};

export default apiClient;
