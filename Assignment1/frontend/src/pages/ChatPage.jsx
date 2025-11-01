import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import useAuthStore from '../store/authStore';

/**
 * Main chat page (placeholder for Phase 2)
 */
const ChatPage = () => {
  const navigate = useNavigate();
  const { isAuthenticated, checkAuth, logout, ollamaUrl } = useAuthStore();

  useEffect(() => {
    // Check authentication on mount
    if (!checkAuth()) {
      navigate('/setup');
    }
  }, [checkAuth, navigate]);

  const handleLogout = () => {
    logout();
    navigate('/setup');
  };

  const handleReconfigure = () => {
    navigate('/setup');
  };

  if (!isAuthenticated) {
    return null; // Will redirect
  }

  return (
    <div className="min-h-screen bg-bg-primary">
      {/* Header */}
      <header className="bg-white border-b border-border-color">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="text-2xl">🤖</div>
            <div>
              <h1 className="text-xl font-semibold text-text-primary">
                Ollama Web GUI
              </h1>
              <p className="text-xs text-text-tertiary">
                Phase 1 - Setup Complete
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleReconfigure}
              className="btn-secondary text-sm"
              aria-label="Reconfigure settings"
            >
              ⚙️ Settings
            </button>
            <button
              onClick={handleLogout}
              className="btn-secondary text-sm"
              aria-label="Logout"
            >
              🚪 Logout
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 py-12">
        <div className="text-center">
          <div className="text-6xl mb-6">🎉</div>
          <h2 className="text-3xl font-bold text-text-primary mb-4">
            Setup Complete!
          </h2>
          <p className="text-lg text-text-secondary mb-8">
            Phase 1 foundation is ready. Chat interface coming in Phase 2.
          </p>

          {/* Success Card */}
          <div className="bg-green-50 border border-green-200 rounded-xl p-8 max-w-2xl mx-auto mb-8">
            <div className="flex items-start gap-4">
              <div className="text-3xl">✓</div>
              <div className="flex-1 text-left">
                <h3 className="text-lg font-semibold text-green-800 mb-3">
                  Configuration Saved Successfully
                </h3>
                <ul className="space-y-2 text-sm text-green-700">
                  <li className="flex items-center gap-2">
                    <span className="text-green-600">●</span>
                    <span>API key securely stored</span>
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="text-green-600">●</span>
                    <span>Ollama URL configured: <code className="bg-white px-2 py-1 rounded text-xs">{ollamaUrl}</code></span>
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="text-green-600">●</span>
                    <span>Backend connection verified</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>

          {/* What's Next Section */}
          <div className="bg-white border border-border-color rounded-xl p-8 max-w-2xl mx-auto">
            <h3 className="text-xl font-semibold text-text-primary mb-4">
              What's Next in Phase 2?
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-left">
              <div className="flex items-start gap-3">
                <span className="text-2xl">💬</span>
                <div>
                  <h4 className="font-medium text-text-primary text-sm">Chat Interface</h4>
                  <p className="text-xs text-text-secondary">Full messaging with AI models</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-2xl">⚡</span>
                <div>
                  <h4 className="font-medium text-text-primary text-sm">Real-time Streaming</h4>
                  <p className="text-xs text-text-secondary">Live response generation</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-2xl">📝</span>
                <div>
                  <h4 className="font-medium text-text-primary text-sm">Markdown Support</h4>
                  <p className="text-xs text-text-secondary">Code highlighting & formatting</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-2xl">💾</span>
                <div>
                  <h4 className="font-medium text-text-primary text-sm">Chat History</h4>
                  <p className="text-xs text-text-secondary">Save & manage conversations</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-2xl">🎨</span>
                <div>
                  <h4 className="font-medium text-text-primary text-sm">Model Selection</h4>
                  <p className="text-xs text-text-secondary">Choose from available models</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-2xl">🌙</span>
                <div>
                  <h4 className="font-medium text-text-primary text-sm">Dark Mode</h4>
                  <p className="text-xs text-text-secondary">Theme customization</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="fixed bottom-0 left-0 right-0 bg-white border-t border-border-color py-3">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-xs text-text-tertiary">
            Powered by Ollama • Built with React + Vite + Tailwind CSS
          </p>
        </div>
      </footer>
    </div>
  );
};

export default ChatPage;
