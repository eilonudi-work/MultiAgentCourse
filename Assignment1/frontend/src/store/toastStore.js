import { create } from 'zustand';

/**
 * Toast Store
 * Manages toast notifications globally
 */
const useToastStore = create((set) => ({
  toasts: [],

  // Add a new toast
  addToast: (message, type = 'info', duration = 5000) => {
    const id = Date.now() + Math.random();
    set((state) => ({
      toasts: [...state.toasts, { id, message, type, duration }],
    }));
    return id;
  },

  // Remove a toast by ID
  removeToast: (id) => {
    set((state) => ({
      toasts: state.toasts.filter((toast) => toast.id !== id),
    }));
  },

  // Clear all toasts
  clearToasts: () => {
    set({ toasts: [] });
  },

  // Convenience methods for different types
  success: (message, duration = 5000) => {
    return useToastStore.getState().addToast(message, 'success', duration);
  },

  error: (message, duration = 7000) => {
    return useToastStore.getState().addToast(message, 'error', duration);
  },

  warning: (message, duration = 6000) => {
    return useToastStore.getState().addToast(message, 'warning', duration);
  },

  info: (message, duration = 5000) => {
    return useToastStore.getState().addToast(message, 'info', duration);
  },
}));

export default useToastStore;
