import { useEffect, useRef } from 'react';

/**
 * Help Modal Component
 * Displays documentation and help information
 */
const HelpModal = ({ isOpen, onClose }) => {
  const modalRef = useRef(null);
  const closeButtonRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      // Focus the close button when modal opens
      closeButtonRef.current?.focus();

      // Prevent body scroll
      document.body.style.overflow = 'hidden';

      // Handle Escape key
      const handleEscape = (e) => {
        if (e.key === 'Escape') {
          onClose();
        }
      };

      document.addEventListener('keydown', handleEscape);

      return () => {
        document.body.style.overflow = 'unset';
        document.removeEventListener('keydown', handleEscape);
      };
    }
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const sections = [
    {
      title: 'Getting Started',
      items: [
        { label: 'Set up your API key', description: 'Configure your Ollama connection in the setup page' },
        { label: 'Select a model', description: 'Choose from available AI models for your conversation' },
        { label: 'Start chatting', description: 'Type your message and press Enter to send' },
      ],
    },
    {
      title: 'Features',
      items: [
        { label: 'Markdown support', description: 'Messages support full markdown formatting including code blocks' },
        { label: 'Conversation history', description: 'All conversations are automatically saved and can be accessed from the sidebar' },
        { label: 'Export/Import', description: 'Export your conversations to JSON for backup or import them later' },
        { label: 'Dark mode', description: 'Toggle between light and dark themes in settings' },
        { label: 'Multiple models', description: 'Switch between different AI models for different tasks' },
      ],
    },
    {
      title: 'Tips & Tricks',
      items: [
        { label: 'Use keyboard shortcuts', description: 'Press ? to view all available shortcuts' },
        { label: 'Edit conversations', description: 'Rename or delete conversations from the sidebar' },
        { label: 'Code highlighting', description: 'Code blocks are automatically highlighted with syntax' },
        { label: 'Streaming responses', description: 'AI responses stream in real-time as they\'re generated' },
      ],
    },
    {
      title: 'Troubleshooting',
      items: [
        { label: 'Connection errors', description: 'Ensure Ollama is running on your machine (ollama serve)' },
        { label: 'Invalid API key', description: 'Check your API key in the setup page' },
        { label: 'Slow responses', description: 'Try a smaller model or check your internet connection' },
        { label: 'Missing conversations', description: 'Check browser storage and ensure cookies are enabled' },
      ],
    },
  ];

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="help-modal-title"
    >
      <div
        ref={modalRef}
        className="bg-bg-primary dark:bg-gray-800 rounded-xl shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden border border-border-color"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border-color">
          <h2
            id="help-modal-title"
            className="text-2xl font-bold text-text-primary"
          >
            Help & Documentation
          </h2>
          <button
            ref={closeButtonRef}
            onClick={onClose}
            className="btn-icon"
            aria-label="Close help modal"
          >
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
              <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)] custom-scrollbar">
          {sections.map((section, index) => (
            <div key={index} className="mb-8 last:mb-0">
              <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
                <span className="w-8 h-8 bg-accent-primary text-white rounded-full flex items-center justify-center text-sm font-bold">
                  {index + 1}
                </span>
                {section.title}
              </h3>
              <div className="space-y-3 ml-10">
                {section.items.map((item, itemIndex) => (
                  <div key={itemIndex} className="border-l-2 border-accent-primary pl-4">
                    <h4 className="font-medium text-text-primary text-sm mb-1">
                      {item.label}
                    </h4>
                    <p className="text-text-secondary text-sm">
                      {item.description}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          ))}

          {/* Version info */}
          <div className="mt-8 pt-6 border-t border-border-color text-center">
            <p className="text-sm text-text-tertiary">
              Ollama Web GUI v1.0.0
            </p>
            <p className="text-xs text-text-tertiary mt-1">
              Built with React, Vite, and Tailwind CSS
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-border-color flex justify-end gap-3">
          <button
            onClick={() => {
              localStorage.removeItem('onboarding_completed');
              onClose();
              window.location.reload();
            }}
            className="btn-secondary"
          >
            Restart Tour
          </button>
          <button onClick={onClose} className="btn-primary">
            Got it!
          </button>
        </div>
      </div>
    </div>
  );
};

export default HelpModal;
