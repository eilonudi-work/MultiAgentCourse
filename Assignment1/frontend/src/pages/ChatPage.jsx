import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import useAuthStore from '../store/authStore';
import useConfigStore from '../store/configStore';
import useChatStore from '../store/chatStore';
import useConversationStore from '../store/conversationStore';
import chatService from '../services/chatService';
import ConversationSidebar from '../components/ConversationSidebar';
import ChatMessages from '../components/ChatMessages';
import ChatInput from '../components/ChatInput';
import ModelSelectorModal from '../components/ModelSelectorModal';
import SettingsModal from '../components/SettingsModal';

/**
 * Main chat page with full interface
 * Includes sidebar, chat area, and header with controls
 */
const ChatPage = () => {
  const navigate = useNavigate();
  const { isAuthenticated, checkAuth, logout } = useAuthStore();
  const {
    theme,
    toggleTheme,
    sidebarCollapsed,
    toggleSidebar,
    systemPrompt,
    initializeTheme,
  } = useConfigStore();

  const {
    messages,
    setMessages,
    addMessage,
    startStreaming,
    appendStreamToken,
    stopStreaming,
    setError,
    clearMessages,
    isStreaming,
    stopGeneration,
  } = useChatStore();

  const {
    conversations,
    currentConversation,
    selectedModel,
    setSelectedModel,
    loadConversations,
    createConversation,
    selectConversation,
    updateConversation,
    setCurrentConversation,
  } = useConversationStore();

  const [showModelSelector, setShowModelSelector] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);

  // Initialize theme on mount
  useEffect(() => {
    initializeTheme();
  }, []);

  // Check authentication on mount
  useEffect(() => {
    if (!checkAuth()) {
      navigate('/setup');
    } else {
      // Load conversations on mount
      loadConversations();
      // Set default model if none selected
      if (!selectedModel) {
        setSelectedModel('llama3.2:1b');
      }
    }
  }, [checkAuth, navigate]);

  // Handle conversation selection
  const handleSelectConversation = async (conversationId) => {
    setIsLoadingMessages(true);
    try {
      const conversation = await selectConversation(conversationId);
      // Load messages into chat store
      setMessages(conversation.messages || []);
    } catch (error) {
      console.error('Failed to load conversation:', error);
      setError(error.message);
    } finally {
      setIsLoadingMessages(false);
    }
  };

  // Handle new chat
  const handleNewChat = async () => {
    try {
      const conversation = await createConversation({
        title: 'New Chat',
        model_name: selectedModel || 'llama3.2:1b',
        system_prompt: systemPrompt,
      });
      clearMessages();
      setCurrentConversation(conversation);
    } catch (error) {
      console.error('Failed to create conversation:', error);
      setError(error.message);
    }
  };

  // Handle send message
  const handleSendMessage = async (content) => {
    if (!content.trim() || isStreaming) return;

    // Ensure we have a conversation
    let conversation = currentConversation;
    if (!conversation) {
      try {
        conversation = await createConversation({
          title: content.substring(0, 50),
          model_name: selectedModel || 'llama3.2:1b',
          system_prompt: systemPrompt,
        });
      } catch (error) {
        console.error('Failed to create conversation:', error);
        setError(error.message);
        return;
      }
    }

    // Add user message
    const userMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };
    addMessage(userMessage);

    // Create placeholder for assistant message
    const assistantMessageId = `assistant-${Date.now()}`;
    const assistantMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      isStreaming: true,
    };
    addMessage(assistantMessage);

    // Start streaming
    startStreaming(assistantMessageId);

    // Stream the response
    try {
      const stream = await chatService.streamMessage(
        {
          conversation_id: conversation.id,
          message: content,
          model_name: selectedModel || 'llama3.2:1b',
          system_prompt: systemPrompt,
        },
        // onToken callback
        (token) => {
          appendStreamToken(token);
        },
        // onComplete callback
        (data) => {
          stopStreaming(data);
          // Update conversation title if it's the first message
          if (messages.length === 0 && conversation) {
            updateConversation(conversation.id, {
              title: content.substring(0, 50),
            });
          }
        },
        // onError callback
        (error) => {
          console.error('Streaming error:', error);
          setError(error.message);
          stopStreaming();
        }
      );

      // Store EventSource for potential cancellation
      if (stream) {
        useChatStore.setState({ eventSource: stream.eventSource });
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      setError(error.message);
      stopStreaming();
    }
  };

  // Handle model selection
  const handleModelSelect = (model) => {
    setSelectedModel(model);
    // Update current conversation model if exists
    if (currentConversation) {
      updateConversation(currentConversation.id, { model_name: model });
    }
  };

  // Handle logout
  const handleLogout = () => {
    logout();
    navigate('/setup');
  };

  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="h-screen flex flex-col bg-white dark:bg-gray-900 transition-colors">
      {/* Header */}
      <header className="border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 z-10">
        <div className="flex items-center justify-between px-4 py-3">
          {/* Left: Menu toggle and title */}
          <div className="flex items-center gap-3">
            <button
              onClick={toggleSidebar}
              className="btn-icon"
              aria-label="Toggle sidebar"
              title="Toggle sidebar"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>

            <div className="hidden lg:block">
              <h1 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Ollama Web GUI
              </h1>
            </div>
          </div>

          {/* Center: Model selector and conversation title */}
          <div className="flex items-center gap-2 flex-1 justify-center max-w-md">
            <button
              onClick={() => setShowModelSelector(true)}
              className="btn-secondary text-sm flex items-center gap-2"
              aria-label="Select model"
              title="Select AI model"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
              <span className="hidden sm:inline">{selectedModel || 'Select Model'}</span>
            </button>

            {currentConversation && (
              <span className="text-sm text-gray-600 dark:text-gray-400 truncate max-w-xs hidden md:inline">
                {currentConversation.title}
              </span>
            )}
          </div>

          {/* Right: Controls */}
          <div className="flex items-center gap-2">
            {/* Stop generation button */}
            {isStreaming && (
              <button
                onClick={stopGeneration}
                className="btn-secondary text-sm flex items-center gap-2 text-red-600 dark:text-red-400 border-red-300 dark:border-red-700"
                aria-label="Stop generation"
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <rect x="6" y="6" width="12" height="12" />
                </svg>
                <span className="hidden sm:inline">Stop</span>
              </button>
            )}

            {/* Theme toggle */}
            <button
              onClick={toggleTheme}
              className="btn-icon"
              aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
              title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
            >
              {theme === 'light' ? (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
              )}
            </button>

            {/* Settings */}
            <button
              onClick={() => setShowSettings(true)}
              className="btn-icon"
              aria-label="Settings"
              title="Settings"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </button>

            {/* Logout */}
            <button
              onClick={handleLogout}
              className="btn-icon"
              aria-label="Logout"
              title="Logout"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
            </button>
          </div>
        </div>
      </header>

      {/* Main content area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        {!sidebarCollapsed && (
          <ConversationSidebar
            onSelectConversation={handleSelectConversation}
            onNewChat={handleNewChat}
          />
        )}

        {/* Chat area */}
        <div className="flex-1 flex flex-col">
          <ChatMessages
            messages={messages}
            isStreaming={isStreaming}
            isLoading={isLoadingMessages}
          />
          <ChatInput
            onSend={handleSendMessage}
            disabled={isStreaming}
            placeholder="Type your message..."
          />
        </div>
      </div>

      {/* Modals */}
      <ModelSelectorModal
        isOpen={showModelSelector}
        onClose={() => setShowModelSelector(false)}
        currentModel={selectedModel}
        onSelectModel={handleModelSelect}
      />

      <SettingsModal
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
      />
    </div>
  );
};

export default ChatPage;
