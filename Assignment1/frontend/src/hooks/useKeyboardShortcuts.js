import { useEffect } from 'react';

/**
 * Custom hook for keyboard shortcuts
 * Handles global keyboard shortcuts with accessibility support
 */
const useKeyboardShortcuts = (shortcuts) => {
  useEffect(() => {
    const handleKeyDown = (event) => {
      // Don't trigger shortcuts when typing in input fields
      const target = event.target;
      const isInput = target.tagName === 'INPUT' ||
                      target.tagName === 'TEXTAREA' ||
                      target.isContentEditable;

      // Check each shortcut
      for (const shortcut of shortcuts) {
        const { key, ctrl, shift, alt, meta, callback, allowInInput = false } = shortcut;

        // Skip if typing in input and shortcut doesn't allow it
        if (isInput && !allowInInput) {
          continue;
        }

        // Check if all modifier keys match
        const ctrlMatch = ctrl === undefined || event.ctrlKey === ctrl || event.metaKey === ctrl;
        const shiftMatch = shift === undefined || event.shiftKey === shift;
        const altMatch = alt === undefined || event.altKey === alt;
        const metaMatch = meta === undefined || event.metaKey === meta;

        // Check if the key matches (case insensitive)
        const keyMatch = event.key.toLowerCase() === key.toLowerCase();

        if (keyMatch && ctrlMatch && shiftMatch && altMatch && metaMatch) {
          event.preventDefault();
          callback(event);
          break;
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [shortcuts]);
};

export default useKeyboardShortcuts;
