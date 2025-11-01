import { useNavigate } from 'react-router-dom';
import ConfigurationModal from '../components/ConfigurationModal';

/**
 * Setup page for first-time configuration
 */
const SetupPage = () => {
  const navigate = useNavigate();

  const handleSuccess = () => {
    // Navigate to chat page after successful setup
    navigate('/chat');
  };

  return (
    <div className="min-h-screen bg-bg-secondary flex items-center justify-center">
      <ConfigurationModal onSuccess={handleSuccess} />
    </div>
  );
};

export default SetupPage;
