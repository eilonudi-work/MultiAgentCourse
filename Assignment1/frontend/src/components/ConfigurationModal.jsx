import { useState } from 'react';
import useAuthStore from '../store/authStore';
import authService from '../services/authService';
import modelsService from '../services/modelsService';
import { validateConfigForm } from '../utils/validation';
import { formatApiError, retryWithBackoff, logError } from '../utils/errorHandler';
import LoadingSpinner from './LoadingSpinner';

/**
 * Configuration Modal for initial setup
 * Handles API key and Ollama URL configuration
 */
const ConfigurationModal = ({ onSuccess }) => {
  const { setApiKey, setOllamaUrl, ollamaUrl: storedOllamaUrl } = useAuthStore();

  const [formData, setFormData] = useState({
    apiKey: '',
    ollamaUrl: storedOllamaUrl || 'http://localhost:11434',
  });

  const [connectionStatus, setConnectionStatus] = useState({
    tested: false,
    success: false,
    message: '',
    modelsCount: 0,
  });

  const [loading, setLoading] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);
  const [errors, setErrors] = useState({});
  const [retryCount, setRetryCount] = useState(0);

  // Handle input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    // Clear errors on change
    setErrors((prev) => ({ ...prev, [name]: '' }));
    // Reset connection status when inputs change
    setConnectionStatus({ tested: false, success: false, message: '', modelsCount: 0 });
  };

  // Validate form
  const validateForm = () => {
    const newErrors = validateConfigForm(formData);
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Test connection to backend and Ollama with retry
  const handleTestConnection = async () => {
    if (!validateForm()) {
      return;
    }

    setLoading(true);
    setConnectionStatus({ tested: false, success: false, message: '', modelsCount: 0 });
    setRetryCount(0);

    try {
      // Step 1: Setup API key with backend (with retry)
      await retryWithBackoff(
        () => authService.setup(formData.apiKey, formData.ollamaUrl),
        3,
        1000
      );

      // Step 1.5: Save API key to store/localStorage so subsequent requests can use it
      setApiKey(formData.apiKey);
      setOllamaUrl(formData.ollamaUrl);

      // Step 2: Try to fetch models to verify Ollama connection (with retry)
      const modelsResponse = await retryWithBackoff(
        () => modelsService.list(),
        3,
        1000
      );

      const modelsCount = modelsResponse.models?.length || 0;

      setConnectionStatus({
        tested: true,
        success: true,
        message: modelsCount > 0
          ? `Connected successfully! Found ${modelsCount} model${modelsCount !== 1 ? 's' : ''}.`
          : 'Connected successfully, but no models found.',
        modelsCount,
      });
    } catch (error) {
      logError(error, 'ConfigurationModal.handleTestConnection');

      const errorMessage = formatApiError(error);

      setConnectionStatus({
        tested: true,
        success: false,
        message: errorMessage,
        modelsCount: 0,
      });
    } finally {
      setLoading(false);
    }
  };

  // Retry connection test
  const handleRetry = () => {
    setRetryCount((prev) => prev + 1);
    handleTestConnection();
  };

  // Save configuration and continue
  const handleSaveAndContinue = () => {
    if (!connectionStatus.success) {
      return;
    }

    // Save to Zustand store (persisted to localStorage)
    setApiKey(formData.apiKey);
    setOllamaUrl(formData.ollamaUrl);

    // Call success callback
    if (onSuccess) {
      onSuccess();
    }
  };

  // Handle keyboard shortcuts
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !loading) {
      if (connectionStatus.success) {
        handleSaveAndContinue();
      } else if (formData.apiKey && formData.ollamaUrl) {
        handleTestConnection();
      }
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div
        className="bg-white rounded-xl shadow-2xl w-full max-w-md p-8"
        role="dialog"
        aria-labelledby="modal-title"
        aria-modal="true"
      >
        {/* Header */}
        <div className="text-center mb-6">
          <div className="text-4xl mb-3">🤖</div>
          <h2 id="modal-title" className="text-2xl font-semibold text-text-primary">
            Ollama Configuration
          </h2>
          <p className="text-sm text-text-secondary mt-2">
            Configure your connection to get started
          </p>
        </div>

        {/* Ollama URL Field */}
        <div className="mb-6">
          <label htmlFor="ollamaUrl" className="block text-sm font-medium text-text-primary mb-2">
            Ollama API URL
          </label>
          <input
            type="text"
            id="ollamaUrl"
            name="ollamaUrl"
            value={formData.ollamaUrl}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            className={`input-field w-full ${errors.ollamaUrl ? 'border-red-500' : ''}`}
            placeholder="http://localhost:11434"
            disabled={loading}
            aria-invalid={!!errors.ollamaUrl}
            aria-describedby={errors.ollamaUrl ? 'ollamaUrl-error' : undefined}
          />
          {errors.ollamaUrl && (
            <p id="ollamaUrl-error" className="mt-1 text-sm text-red-500" role="alert">
              {errors.ollamaUrl}
            </p>
          )}
          <p className="mt-1 text-xs text-text-tertiary">
            Default: http://localhost:11434
          </p>
        </div>

        {/* API Key Field */}
        <div className="mb-6">
          <label htmlFor="apiKey" className="block text-sm font-medium text-text-primary mb-2">
            API Key <span className="text-red-500" aria-label="required">*</span>
          </label>
          <div className="relative">
            <input
              type={showApiKey ? 'text' : 'password'}
              id="apiKey"
              name="apiKey"
              value={formData.apiKey}
              onChange={handleChange}
              onKeyDown={handleKeyDown}
              className={`input-field w-full pr-12 ${errors.apiKey ? 'border-red-500' : ''}`}
              placeholder="Enter your API key"
              disabled={loading}
              aria-invalid={!!errors.apiKey}
              aria-describedby={errors.apiKey ? 'apiKey-error' : 'apiKey-help'}
              autoComplete="off"
            />
            <button
              type="button"
              onClick={() => setShowApiKey(!showApiKey)}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-text-tertiary hover:text-text-secondary transition-colors"
              disabled={loading}
              aria-label={showApiKey ? 'Hide API key' : 'Show API key'}
            >
              {showApiKey ? '🙈' : '👁️'}
            </button>
          </div>
          {errors.apiKey && (
            <p id="apiKey-error" className="mt-1 text-sm text-red-500" role="alert">
              {errors.apiKey}
            </p>
          )}
          <p id="apiKey-help" className="mt-1 text-xs text-text-tertiary flex items-start gap-1">
            <span>ℹ️</span>
            <span>Minimum 8 characters, alphanumeric with dashes/underscores</span>
          </p>
        </div>

        {/* Connection Status */}
        {connectionStatus.tested && (
          <div
            className={`mb-6 p-4 rounded-lg border ${
              connectionStatus.success
                ? 'bg-green-50 border-green-200'
                : 'bg-red-50 border-red-200'
            }`}
            role="status"
            aria-live="polite"
          >
            <p className={`text-sm font-medium ${
              connectionStatus.success ? 'text-green-800' : 'text-red-800'
            }`}>
              {connectionStatus.success ? '✓' : '✗'} {connectionStatus.message}
            </p>
            {!connectionStatus.success && (
              <button
                onClick={handleRetry}
                className="mt-2 text-xs text-red-700 hover:text-red-900 underline"
                disabled={loading}
              >
                Retry Connection
              </button>
            )}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="mb-6 flex justify-center">
            <LoadingSpinner size="md" message="Testing connection..." />
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-3">
          <button
            onClick={handleTestConnection}
            disabled={loading || !formData.apiKey || !formData.ollamaUrl}
            className="btn-secondary flex-1"
            aria-label="Test connection to backend and Ollama"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <span className="animate-spin">⏳</span>
                Testing...
              </span>
            ) : (
              'Test Connection'
            )}
          </button>

          <button
            onClick={handleSaveAndContinue}
            disabled={!connectionStatus.success || loading}
            className="btn-primary flex-1"
            aria-label="Save configuration and continue to chat"
          >
            Save & Continue
          </button>
        </div>

        {/* Keyboard Shortcut Hint */}
        <p className="mt-3 text-xs text-center text-text-tertiary">
          Press <kbd className="px-1 py-0.5 bg-bg-tertiary rounded border border-border-color">Enter</kbd> to {connectionStatus.success ? 'continue' : 'test'}
        </p>

        {/* Help Link */}
        <div className="mt-6 pt-6 border-t border-border-color">
          <p className="text-sm text-text-secondary">
            📖 Need help? Check the{' '}
            <a
              href="https://github.com/ollama/ollama"
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent-primary hover:underline"
            >
              Ollama documentation
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default ConfigurationModal;
