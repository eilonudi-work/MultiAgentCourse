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

      // Handle conversation created event
      eventSource.addEventListener('conversation_created', (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('Conversation created:', data.conversation_id);
        } catch (parseError) {
          console.error('Failed to parse conversation_created event:', parseError);
        }
      });

      // Handle message created event
      eventSource.addEventListener('message_created', (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('Message created:', data.message_id);
        } catch (parseError) {
          console.error('Failed to parse message_created event:', parseError);
        }
      });

      // Handle streaming tokens
      eventSource.addEventListener('token', (event) => {
        try {
          const data = JSON.parse(event.data);
          onToken(data.content);
        } catch (parseError) {
          console.error('Failed to parse token event:', parseError);
        }
      });

      // Handle completion
      eventSource.addEventListener('done', (event) => {
        try {
          const data = JSON.parse(event.data);
          eventSource.close();
          onComplete(data);
        } catch (parseError) {
          console.error('Failed to parse done event:', parseError);
          eventSource.close();
          onComplete({});
        }
      });

      // Handle errors
      eventSource.addEventListener('error', (event) => {
        try {
          const data = JSON.parse(event.data);
          eventSource.close();
          onError(new Error(data.error || 'Streaming error occurred'));
        } catch (parseError) {
          console.error('Failed to parse error event:', parseError);
        }
      });

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
