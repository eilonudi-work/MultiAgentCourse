import { useNavigate } from 'react-router-dom';

/**
 * Error Screen Component
 * Displays specific error screens for different error types
 */
const ErrorScreen = ({ type = 'general', error, onRetry }) => {
  const navigate = useNavigate();

  const errorConfigs = {
    invalidApiKey: {
      title: 'Invalid API Key',
      icon: '🔑',
      description: 'The API key you provided is invalid or has expired. Please check your credentials and try again.',
      actions: [
        { label: 'Update API Key', onClick: () => navigate('/setup'), primary: true },
        { label: 'Retry', onClick: onRetry, primary: false },
      ],
    },
    ollamaOffline: {
      title: 'Ollama Server Offline',
      icon: '🔌',
      description: 'Unable to connect to Ollama server. Please ensure Ollama is running on your machine.',
      help: [
        'Check if Ollama is running: ollama serve',
        'Verify the server URL is correct',
        'Check your firewall settings',
      ],
      actions: [
        { label: 'Retry Connection', onClick: onRetry, primary: true },
        { label: 'Update Settings', onClick: () => navigate('/setup'), primary: false },
      ],
    },
    networkError: {
      title: 'Network Connection Error',
      icon: '📡',
      description: 'Unable to connect to the server. Please check your internet connection and try again.',
      help: [
        'Check your internet connection',
        'Verify the backend server is running',
        'Check for firewall or VPN issues',
      ],
      actions: [
        { label: 'Retry', onClick: onRetry, primary: true },
        { label: 'Reload Page', onClick: () => window.location.reload(), primary: false },
      ],
    },
    notFound: {
      title: 'Page Not Found',
      icon: '🔍',
      description: 'The page you are looking for does not exist or has been moved.',
      actions: [
        { label: 'Go to Home', onClick: () => navigate('/'), primary: true },
        { label: 'Go Back', onClick: () => navigate(-1), primary: false },
      ],
    },
    serverError: {
      title: 'Server Error',
      icon: '⚠️',
      description: 'The server encountered an unexpected error. Please try again later.',
      actions: [
        { label: 'Retry', onClick: onRetry, primary: true },
        { label: 'Go to Home', onClick: () => navigate('/'), primary: false },
      ],
    },
    general: {
      title: 'Something Went Wrong',
      icon: '😕',
      description: 'An unexpected error occurred. Please try again or contact support if the problem persists.',
      actions: [
        { label: 'Retry', onClick: onRetry, primary: true },
        { label: 'Go to Home', onClick: () => navigate('/'), primary: false },
      ],
    },
  };

  const config = errorConfigs[type] || errorConfigs.general;

  return (
    <div className="min-h-screen bg-bg-secondary flex items-center justify-center p-4">
      <div className="bg-bg-primary rounded-xl shadow-lg max-w-2xl w-full p-8 border border-border-color">
        <div className="text-center mb-6">
          <div
            className="text-6xl mb-4"
            role="img"
            aria-label={`${config.title} icon`}
          >
            {config.icon}
          </div>
          <h1 className="text-2xl font-bold text-text-primary mb-2">
            {config.title}
          </h1>
          <p className="text-text-secondary">
            {config.description}
          </p>
        </div>

        {/* Error details (in development) */}
        {import.meta.env.DEV && error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-6">
            <h3 className="text-sm font-semibold text-red-800 dark:text-red-300 mb-2">
              Error Details (Development Mode)
            </h3>
            <pre className="text-xs text-red-700 dark:text-red-400 overflow-auto max-h-48 whitespace-pre-wrap break-words">
              {error.message || String(error)}
              {error.stack && '\n\n' + error.stack}
            </pre>
          </div>
        )}

        {/* Help list */}
        {config.help && (
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 mb-6">
            <h3 className="text-sm font-semibold text-blue-800 dark:text-blue-300 mb-2">
              Troubleshooting Steps
            </h3>
            <ul className="text-sm text-blue-700 dark:text-blue-400 space-y-1">
              {config.help.map((item, index) => (
                <li key={index}>• {item}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Action buttons */}
        <div className="flex gap-3 justify-center flex-wrap">
          {config.actions.map((action, index) => (
            <button
              key={index}
              onClick={action.onClick}
              className={action.primary ? 'btn-primary' : 'btn-secondary'}
              disabled={!action.onClick}
            >
              {action.label}
            </button>
          ))}
        </div>

        {/* Support information */}
        <div className="mt-8 pt-6 border-t border-border-color text-center">
          <p className="text-sm text-text-tertiary">
            Need help? Check the documentation or report an issue.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ErrorScreen;
