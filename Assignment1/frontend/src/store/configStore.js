import { create } from 'zustand';
import { persist } from 'zustand/middleware';

/**
 * Configuration state store using Zustand
 * Manages application settings and preferences
 */
const useConfigStore = create(
  persist(
    (set, get) => ({
      // State
      theme: 'light', // 'light' | 'dark'
      temperature: 0.7,
      maxTokens: 2000,
      systemPrompt: 'You are a helpful assistant.',
      sidebarCollapsed: false,

      // Actions
      setTheme: (theme) => {
        set({ theme });
        // Apply theme to document
        if (theme === 'dark') {
          document.documentElement.classList.add('dark');
        } else {
          document.documentElement.classList.remove('dark');
        }
      },

      toggleTheme: () => {
        const { theme } = get();
        const newTheme = theme === 'light' ? 'dark' : 'light';
        get().setTheme(newTheme);
      },

      setTemperature: (temperature) => set({ temperature }),

      setMaxTokens: (maxTokens) => set({ maxTokens }),

      setSystemPrompt: (systemPrompt) => set({ systemPrompt }),

      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),

      toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),

      // Reset to defaults
      resetConfig: () => set({
        theme: 'light',
        temperature: 0.7,
        maxTokens: 2000,
        systemPrompt: 'You are a helpful assistant.',
        sidebarCollapsed: false,
      }),

      // Initialize theme on app load
      initializeTheme: () => {
        const { theme } = get();
        if (theme === 'dark') {
          document.documentElement.classList.add('dark');
        } else {
          document.documentElement.classList.remove('dark');
        }
      },
    }),
    {
      name: 'ollama-config-storage', // localStorage key
    }
  )
);

export default useConfigStore;
