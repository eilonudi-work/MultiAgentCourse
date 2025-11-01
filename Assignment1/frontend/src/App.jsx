import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import useAuthStore from './store/authStore';
import SetupPage from './pages/SetupPage';
import ChatPage from './pages/ChatPage';
import ProtectedRoute from './components/ProtectedRoute';
import ErrorBoundary from './components/ErrorBoundary';

/**
 * Main App component with routing
 */
function App() {
  const { checkAuth } = useAuthStore();

  return (
    <ErrorBoundary>
      <Router>
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
      </Router>
    </ErrorBoundary>
  );
}

export default App;
