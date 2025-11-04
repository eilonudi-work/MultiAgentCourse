import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { lazy, Suspense, useState } from 'react';
import useAuthStore from './store/authStore';
import useToastStore from './store/toastStore';
import ProtectedRoute from './components/ProtectedRoute';
import ErrorBoundary from './components/ErrorBoundary';
import { ToastContainer } from './components/Toast';
import NetworkStatus from './components/NetworkStatus';
import OnboardingTour from './components/OnboardingTour';
import HelpModal from './components/HelpModal';
import KeyboardShortcutsModal from './components/KeyboardShortcutsModal';
import LoadingSpinner from './components/LoadingSpinner';
import useKeyboardShortcuts from './hooks/useKeyboardShortcuts';

// Lazy load pages for better performance
const SetupPage = lazy(() => import('./pages/SetupPage'));
const ChatPage = lazy(() => import('./pages/ChatPage'));

/**
 * Main App component with routing
 * Enhanced with accessibility, keyboard shortcuts, and performance optimizations
 */
function App() {
  const { checkAuth } = useAuthStore();
  const { toasts, removeToast } = useToastStore();
  const [showHelp, setShowHelp] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);

  // Global keyboard shortcuts
  useKeyboardShortcuts([
    {
      key: '?',
      shift: true,
      callback: () => setShowShortcuts(true),
      allowInInput: false,
    },
    {
      key: '/',
      ctrl: true,
      callback: () => setShowHelp(true),
      allowInInput: false,
    },
    {
      key: 'Escape',
      callback: () => {
        setShowHelp(false);
        setShowShortcuts(false);
      },
      allowInInput: true,
    },
  ]);

  return (
    <ErrorBoundary>
      <Router>
        {/* Skip to main content link for accessibility */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-accent-primary focus:text-white focus:rounded-md"
        >
          Skip to main content
        </a>

        <Suspense
          fallback={
            <LoadingSpinner
              size="lg"
              message="Loading application..."
              fullScreen
            />
          }
        >
          <main id="main-content">
            <Routes>
              {/* Default route - redirect based on auth status */}
              <Route
                path="/"
                element={checkAuth() ? <Navigate to="/chat" replace /> : <Navigate to="/setup" replace />}
              />

              {/* Setup route */}
              <Route path="/setup" element={<SetupPage />} />

              {/* Protected chat route */}
              <Route
                path="/chat"
                element={
                  <ProtectedRoute>
                    <ChatPage />
                  </ProtectedRoute>
                }
              />

              {/* 404 - redirect to home */}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
        </Suspense>

        {/* Global UI Components */}
        <ToastContainer toasts={toasts} removeToast={removeToast} />
        <NetworkStatus />
        <OnboardingTour />
        <HelpModal isOpen={showHelp} onClose={() => setShowHelp(false)} />
        <KeyboardShortcutsModal
          isOpen={showShortcuts}
          onClose={() => setShowShortcuts(false)}
        />
      </Router>
    </ErrorBoundary>
  );
}

export default App;
