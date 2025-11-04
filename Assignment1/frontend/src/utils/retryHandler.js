/**
 * Retry Handler Utility
 * Implements exponential backoff retry logic
 */

/**
 * Retry a function with exponential backoff
 * @param {Function} fn - The async function to retry
 * @param {Object} options - Retry options
 * @returns {Promise} - Result of the function
 */
export const retryWithBackoff = async (fn, options = {}) => {
  const {
    maxAttempts = 3,
    initialDelay = 1000,
    maxDelay = 10000,
    backoffFactor = 2,
    onRetry = null,
  } = options;

  let attempt = 0;
  let delay = initialDelay;

  while (attempt < maxAttempts) {
    try {
      return await fn();
    } catch (error) {
      attempt++;

      if (attempt >= maxAttempts) {
        throw error;
      }

      // Call onRetry callback if provided
      if (onRetry) {
        onRetry(attempt, delay, error);
      }

      // Wait before retrying
      await new Promise((resolve) => setTimeout(resolve, delay));

      // Increase delay with exponential backoff
      delay = Math.min(delay * backoffFactor, maxDelay);
    }
  }
};

/**
 * Check if an error is retryable
 * @param {Error} error - The error to check
 * @returns {boolean} - Whether the error is retryable
 */
export const isRetryableError = (error) => {
  // Network errors are retryable
  if (!error.response) {
    return true;
  }

  // Server errors (5xx) are retryable
  const status = error.response?.status;
  if (status >= 500 && status < 600) {
    return true;
  }

  // Rate limiting (429) is retryable
  if (status === 429) {
    return true;
  }

  // Request timeout is retryable
  if (error.code === 'ECONNABORTED') {
    return true;
  }

  return false;
};

/**
 * Retry a request with conditional retry based on error type
 * @param {Function} fn - The async function to retry
 * @param {Object} options - Retry options
 * @returns {Promise} - Result of the function
 */
export const retryRequest = async (fn, options = {}) => {
  const { shouldRetry = isRetryableError, ...retryOptions } = options;

  return retryWithBackoff(async () => {
    try {
      return await fn();
    } catch (error) {
      // Only retry if the error is retryable
      if (!shouldRetry(error)) {
        throw error;
      }
      throw error;
    }
  }, retryOptions);
};

export default {
  retryWithBackoff,
  retryRequest,
  isRetryableError,
};
