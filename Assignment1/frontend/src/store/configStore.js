import { create } from 'zustand';
import { persist } from 'zustand/middleware';

/**
 * Configuration state store using Zustand
 * Manages application settings and preferences
 */
const useConfigStore = create(
  persist(
    (set) => ({
      // State
      theme: 'light', // 'light' | 'dark' | 'system'
      temperature: 0.7,
      maxTokens: 2000,
      systemPrompt: 'You are a helpful assistant.',
      sidebarCollapsed: false,

      // Actions
      setTheme: (theme) => set({ theme }),

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
    }),
    {
      name: 'ollama-config-storage', // localStorage key
    }
  )
);

export default useConfigStore;
