import apiClient from './api';

/**
 * Chat service for sending messages and handling streaming responses
 */
const chatService = {
  /**
   * Send a message and get streaming response via SSE
   * @param {Object} data - Message data
   * @param {string} data.conversation_id - Conversation ID
   * @param {string} data.message - User message
   * @param {string} data.model_name - Model to use
   * @param {string} data.system_prompt - System prompt
   * @param {Function} onToken - Callback for each token received
   * @param {Function} onComplete - Callback when stream completes
   * @param {Function} onError - Callback on error
   * @returns {Object} EventSource instance for controlling the stream
   */
  async streamMessage(data, onToken, onComplete, onError) {
    try {
      // Get API key from localStorage for auth
      const apiKey = localStorage.getItem('ollama_api_key');
      if (!apiKey) {
        throw new Error('API key not found. Please reconfigure.');
      }

      // Build query parameters
      const params = new URLSearchParams({
        conversation_id: data.conversation_id || '',
        message: data.message,
        model_name: data.model_name,
        system_prompt: data.system_prompt || 'You are a helpful assistant.',
      });

      // Create EventSource for SSE
      const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
      const url = `${apiBaseUrl}/api/chat/stream?${params.toString()}`;

      const eventSource = new EventSource(url);

      // Handle incoming messages
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'token') {
            // Streaming token
            onToken(data.content);
          } else if (data.type === 'done') {
            // Stream complete
            eventSource.close();
            onComplete(data);
          } else if (data.type === 'error') {
            // Error during streaming
            eventSource.close();
            onError(new Error(data.message || 'Streaming error occurred'));
          }
        } catch (parseError) {
          console.error('Failed to parse SSE data:', parseError);
        }
      };

      // Handle connection errors
      eventSource.onerror = (error) => {
        console.error('SSE connection error:', error);
        eventSource.close();
        onError(new Error('Connection to server lost. Please try again.'));
      };

      // Return EventSource for external control (e.g., stopping generation)
      return {
        eventSource,
        stop: () => {
          eventSource.close();
        },
      };
    } catch (error) {
      onError(this.handleError(error));
      return null;
    }
  },

  /**
   * Send a non-streaming message (fallback)
   * @param {Object} data - Message data
   * @returns {Promise} Response
   */
  async sendMessage(data) {
    try {
      const response = await apiClient.post('/api/chat/send', data);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  },

  /**
   * Handle API errors
   * @param {Error} error - Error object
   * @returns {Error} Formatted error
   */
  handleError(error) {
    if (error.response) {
      const { status, data } = error.response;

      if (status === 401) {
        return new Error('Unauthorized. Please verify your API key.');
      } else if (status === 400) {
        return new Error(data.detail || 'Invalid message data.');
      } else if (status === 500) {
        return new Error('Server error. Please try again later.');
      } else if (status === 503) {
        return new Error('Ollama service is not available. Please ensure Ollama is running.');
      } else {
        return new Error(data.detail || data.message || 'Failed to send message.');
      }
    } else if (error.request) {
      return new Error('Cannot connect to server. Please check if the backend is running.');
    } else {
      return new Error(error.message || 'An unexpected error occurred.');
    }
  },
};

export default chatService;
