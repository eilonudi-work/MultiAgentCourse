import { Navigate } from 'react-router-dom';
import useAuthStore from '../store/authStore';

/**
 * Protected route component
 * Redirects to setup if not authenticated
 */
const ProtectedRoute = ({ children }) => {
  const { checkAuth } = useAuthStore();

  if (!checkAuth()) {
    return <Navigate to="/setup" replace />;
  }

  return children;
};

export default ProtectedRoute;
