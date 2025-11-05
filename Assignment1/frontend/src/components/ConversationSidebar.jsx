import { useState, useEffect } from 'react';
import useConversationStore from '../store/conversationStore';
import useConfigStore from '../store/configStore';
import LoadingSpinner from './LoadingSpinner';

/**
 * ConversationSidebar component
 * Displays list of conversations with search, create, and delete functionality
 */
const ConversationSidebar = ({ onSelectConversation, onNewChat }) => {
  const {
    conversations,
    currentConversation,
    searchQuery,
    setSearchQuery,
    getFilteredConversations,
    deleteConversation,
    isLoading,
  } = useConversationStore();

  const { sidebarCollapsed, setSidebarCollapsed } = useConfigStore();
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(null);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 1024);
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const filteredConversations = getFilteredConversations();

  const handleDelete = async (id, e) => {
    e.stopPropagation();
    setShowDeleteConfirm(id);
  };

  const confirmDelete = async (id, e) => {
    e.stopPropagation();
    try {
      await deleteConversation(id);
      setShowDeleteConfirm(null);
    } catch (error) {
      console.error('Failed to delete conversation:', error);
    }
  };

  const cancelDelete = (e) => {
    e.stopPropagation();
    setShowDeleteConfirm(null);
  };

  const handleConversationSelect = (id) => {
    onSelectConversation(id);
    // On mobile, close sidebar after selection
    if (isMobile) {
      setSidebarCollapsed(true);
    }
  };

  // Format date for display
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now - date;
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    return date.toLocaleDateString();
  };

  // Truncate text helper
  const truncate = (text, maxLength = 50) => {
    if (!text) return '';
    return text.length > maxLength ? `${text.substring(0, maxLength)}...` : text;
  };

  if (sidebarCollapsed) {
    return null;
  }

  return (
    <>
      {/* Mobile overlay */}
      {isMobile && !sidebarCollapsed && (
        <div
          className="sidebar-overlay"
          onClick={() => setSidebarCollapsed(true)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <div
        className={`
          w-80 border-r border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 flex flex-col h-full
          ${isMobile ? 'sidebar-mobile animate-slideIn' : ''}
        `}
      >
        {/* Sidebar header */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Conversations
            </h2>
            {isMobile && (
              <button
                onClick={() => setSidebarCollapsed(true)}
                className="btn-icon"
                aria-label="Close sidebar"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>

          <button
            onClick={onNewChat}
            className="w-full btn-primary flex items-center justify-center gap-2"
            aria-label="Start new chat"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            New Chat
          </button>

          {/* Search input */}
          <div className="mt-3 relative">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search conversations..."
              className="w-full input-field pl-10 py-2 text-sm"
              aria-label="Search conversations"
            />
            <svg
              className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 dark:text-gray-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
          </div>
        </div>

        {/* Conversations list */}
        <div className="flex-1 overflow-y-auto custom-scrollbar">
          {isLoading && conversations.length === 0 ? (
            <div className="flex justify-center py-8">
              <LoadingSpinner size="sm" />
            </div>
          ) : filteredConversations.length === 0 ? (
            <div className="p-4 text-center text-gray-500 dark:text-gray-400 text-sm">
              {searchQuery ? 'No conversations found' : 'No conversations yet'}
            </div>
          ) : (
            <div className="p-2">
              {filteredConversations.map((conversation) => (
                <div
                  key={conversation.id}
                  onClick={() => handleConversationSelect(conversation.id)}
                  className={`
                    relative group cursor-pointer rounded-lg p-3 mb-2 transition-all
                    ${
                      currentConversation?.id === conversation.id
                        ? 'bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700'
                        : 'hover:bg-gray-50 dark:hover:bg-gray-700 border border-transparent'
                    }
                  `}
                  role="button"
                  tabIndex={0}
                  aria-label={`Select conversation: ${conversation.title}`}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      handleConversationSelect(conversation.id);
                    }
                  }}
                >
                  {/* Delete confirmation overlay */}
                  {showDeleteConfirm === conversation.id && (
                    <div className="absolute inset-0 bg-white dark:bg-gray-800 rounded-lg flex flex-col items-center justify-center gap-2 z-10 p-3">
                      <p className="text-sm text-gray-700 dark:text-gray-300 text-center">
                        Delete this conversation?
                      </p>
                      <div className="flex gap-2">
                        <button
                          onClick={(e) => confirmDelete(conversation.id, e)}
                          className="px-3 py-1 text-xs bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
                        >
                          Delete
                        </button>
                        <button
                          onClick={cancelDelete}
                          className="px-3 py-1 text-xs bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-200 rounded hover:bg-gray-300 dark:hover:bg-gray-500 transition-colors"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Conversation title */}
                  <h4 className="font-medium text-sm text-gray-900 dark:text-gray-100 mb-1 pr-6">
                    {truncate(conversation.title, 40)}
                  </h4>

                  {/* Model badge and date */}
                  <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                    <span className="bg-gray-100 dark:bg-gray-700 px-2 py-0.5 rounded">
                      {conversation.model_name || 'llama3.2:1b'}
                    </span>
                    <span>{formatDate(conversation.updated_at || conversation.created_at)}</span>
                  </div>

                  {/* Last message preview */}
                  {conversation.last_message && (
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 line-clamp-2">
                      {truncate(conversation.last_message, 80)}
                    </p>
                  )}

                  {/* Delete button */}
                  <button
                    onClick={(e) => handleDelete(conversation.id, e)}
                    className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-red-50 dark:hover:bg-red-900/20 rounded"
                    aria-label="Delete conversation"
                    title="Delete conversation"
                  >
                    <svg
                      className="w-4 h-4 text-red-600 dark:text-red-400"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                      />
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default ConversationSidebar;
