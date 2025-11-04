import { useEffect, useRef } from 'react';

/**
 * Keyboard Shortcuts Modal Component
 * Displays all available keyboard shortcuts
 */
const KeyboardShortcutsModal = ({ isOpen, onClose }) => {
  const modalRef = useRef(null);
  const closeButtonRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      closeButtonRef.current?.focus();
      document.body.style.overflow = 'hidden';

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

  const shortcuts = [
    {
      category: 'General',
      items: [
        { keys: ['?'], description: 'Show keyboard shortcuts' },
        { keys: ['Ctrl', 'K'], description: 'Focus search/command palette' },
        { keys: ['Escape'], description: 'Close modal or cancel action' },
      ],
    },
    {
      category: 'Navigation',
      items: [
        { keys: ['Ctrl', 'N'], description: 'New conversation' },
        { keys: ['Ctrl', 'B'], description: 'Toggle sidebar' },
        { keys: ['Ctrl', ','], description: 'Open settings' },
        { keys: ['Ctrl', '/'], description: 'Open help' },
      ],
    },
    {
      category: 'Chat',
      items: [
        { keys: ['Enter'], description: 'Send message' },
        { keys: ['Shift', 'Enter'], description: 'New line in message' },
        { keys: ['Ctrl', 'E'], description: 'Export conversation' },
        { keys: ['Ctrl', 'D'], description: 'Delete conversation' },
      ],
    },
    {
      category: 'Editing',
      items: [
        { keys: ['Ctrl', 'C'], description: 'Copy selected text' },
        { keys: ['Ctrl', 'A'], description: 'Select all' },
        { keys: ['Ctrl', 'Z'], description: 'Undo' },
        { keys: ['Ctrl', 'Shift', 'Z'], description: 'Redo' },
      ],
    },
    {
      category: 'Accessibility',
      items: [
        { keys: ['Tab'], description: 'Navigate forward through elements' },
        { keys: ['Shift', 'Tab'], description: 'Navigate backward through elements' },
        { keys: ['Space'], description: 'Activate focused element' },
        { keys: ['Enter'], description: 'Activate focused button' },
      ],
    },
  ];

  const KeyBadge = ({ keyName }) => (
    <kbd className="px-2 py-1 text-xs font-semibold text-text-primary bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded">
      {keyName}
    </kbd>
  );

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
      aria-labelledby="shortcuts-modal-title"
    >
      <div
        ref={modalRef}
        className="bg-bg-primary dark:bg-gray-800 rounded-xl shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden border border-border-color"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border-color">
          <h2
            id="shortcuts-modal-title"
            className="text-2xl font-bold text-text-primary"
          >
            Keyboard Shortcuts
          </h2>
          <button
            ref={closeButtonRef}
            onClick={onClose}
            className="btn-icon"
            aria-label="Close keyboard shortcuts modal"
          >
            <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
              <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)] custom-scrollbar">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {shortcuts.map((category, index) => (
              <div key={index}>
                <h3 className="text-lg font-semibold text-text-primary mb-3 flex items-center gap-2">
                  <span className="w-6 h-6 bg-accent-primary text-white rounded flex items-center justify-center text-xs font-bold">
                    {index + 1}
                  </span>
                  {category.category}
                </h3>
                <div className="space-y-3">
                  {category.items.map((item, itemIndex) => (
                    <div
                      key={itemIndex}
                      className="flex items-center justify-between gap-4 p-2 rounded hover:bg-bg-secondary dark:hover:bg-gray-700/50 transition-colors"
                    >
                      <span className="text-sm text-text-secondary flex-1">
                        {item.description}
                      </span>
                      <div className="flex gap-1 flex-shrink-0">
                        {item.keys.map((key, keyIndex) => (
                          <span key={keyIndex} className="flex items-center">
                            {keyIndex > 0 && (
                              <span className="mx-1 text-text-tertiary text-xs">+</span>
                            )}
                            <KeyBadge keyName={key} />
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Pro tip */}
          <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
            <div className="flex items-start gap-3">
              <svg className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
              <div>
                <h4 className="font-medium text-blue-800 dark:text-blue-300 text-sm mb-1">
                  Pro Tip
                </h4>
                <p className="text-sm text-blue-700 dark:text-blue-400">
                  Press <KeyBadge keyName="?" /> at any time to quickly view these shortcuts. You can also customize shortcuts in the settings.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-border-color flex justify-end">
          <button onClick={onClose} className="btn-primary">
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default KeyboardShortcutsModal;
