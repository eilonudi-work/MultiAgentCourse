import { create } from 'zustand';
import { persist } from 'zustand/middleware';

/**
 * Authentication state store using Zustand
 * Persists API key and authentication status to localStorage
 */
const useAuthStore = create(
  persist(
    (set, get) => ({
      // State
      apiKey: null,
      isAuthenticated: false,
      ollamaUrl: import.meta.env.VITE_OLLAMA_DEFAULT_URL || 'http://localhost:11434',

      // Actions
      setApiKey: (apiKey) => {
        set({ apiKey, isAuthenticated: !!apiKey });
        // Also store in localStorage for API interceptor
        if (apiKey) {
          localStorage.setItem('ollama_api_key', apiKey);
        } else {
          localStorage.removeItem('ollama_api_key');
        }
      },

      setOllamaUrl: (ollamaUrl) => {
        set({ ollamaUrl });
      },

      setAuthenticated: (isAuthenticated) => {
        set({ isAuthenticated });
      },

      logout: () => {
        set({ apiKey: null, isAuthenticated: false });
        localStorage.removeItem('ollama_api_key');
        localStorage.removeItem('ollama_config');
      },

      // Helper to check if user is authenticated
      checkAuth: () => {
        const { apiKey } = get();
        return !!apiKey;
      },
    }),
    {
      name: 'ollama-auth-storage', // localStorage key
      partialize: (state) => ({
        // Only persist these fields
        apiKey: state.apiKey,
        isAuthenticated: state.isAuthenticated,
        ollamaUrl: state.ollamaUrl,
      }),
    }
  )
);

export default useAuthStore;
