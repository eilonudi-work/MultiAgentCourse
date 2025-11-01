/**
 * Validation utility functions
 */

/**
 * Validate URL format
 * @param {string} url - URL to validate
 * @returns {object} { valid: boolean, error: string }
 */
export const validateUrl = (url) => {
  if (!url || url.trim() === '') {
    return { valid: false, error: 'URL is required' };
  }

  try {
    const urlObj = new URL(url);

    if (!['http:', 'https:'].includes(urlObj.protocol)) {
      return { valid: false, error: 'URL must start with http:// or https://' };
    }

    return { valid: true, error: null };
  } catch (error) {
    return { valid: false, error: 'Please enter a valid URL' };
  }
};

/**
 * Validate API key
 * @param {string} apiKey - API key to validate
 * @returns {object} { valid: boolean, error: string }
 */
export const validateApiKey = (apiKey) => {
  if (!apiKey || apiKey.trim() === '') {
    return { valid: false, error: 'API key is required' };
  }

  if (apiKey.length < 8) {
    return { valid: false, error: 'API key must be at least 8 characters' };
  }

  if (apiKey.length > 256) {
    return { valid: false, error: 'API key is too long (max 256 characters)' };
  }

  // Check for valid characters (alphanumeric, dash, underscore)
  if (!/^[a-zA-Z0-9\-_]+$/.test(apiKey)) {
    return { valid: false, error: 'API key contains invalid characters' };
  }

  return { valid: true, error: null };
};

/**
 * Validate form fields
 * @param {object} formData - Form data object
 * @returns {object} Errors object
 */
export const validateConfigForm = (formData) => {
  const errors = {};

  // Validate Ollama URL
  const urlValidation = validateUrl(formData.ollamaUrl);
  if (!urlValidation.valid) {
    errors.ollamaUrl = urlValidation.error;
  }

  // Validate API key
  const apiKeyValidation = validateApiKey(formData.apiKey);
  if (!apiKeyValidation.valid) {
    errors.apiKey = apiKeyValidation.error;
  }

  return errors;
};

/**
 * Sanitize user input to prevent XSS
 * @param {string} input - User input
 * @returns {string} Sanitized input
 */
export const sanitizeInput = (input) => {
  if (typeof input !== 'string') return input;

  return input
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
    .replace(/\//g, '&#x2F;');
};
