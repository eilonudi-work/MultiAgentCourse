import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { visualizer } from 'rollup-plugin-visualizer';

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    // Bundle analyzer (only in build mode with ANALYZE env var)
    mode === 'production' && process.env.ANALYZE && visualizer({
      open: true,
      filename: 'dist/bundle-analysis.html',
      gzipSize: true,
      brotliSize: true,
    }),
  ].filter(Boolean),

  // Build optimizations
  build: {
    // Output directory
    outDir: 'dist',

    // Generate source maps for production debugging
    sourcemap: mode === 'production' ? 'hidden' : true,

    // Minification
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: mode === 'production', // Remove console.log in production
        drop_debugger: true,
      },
    },

    // Chunk splitting strategy
    rollupOptions: {
      output: {
        manualChunks: {
          // Vendor chunks
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'state-vendor': ['zustand'],
          'markdown-vendor': ['marked', 'highlight.js'],
          // Add more chunks as needed
        },
        // Naming strategy
        chunkFileNames: 'assets/js/[name]-[hash].js',
        entryFileNames: 'assets/js/[name]-[hash].js',
        assetFileNames: 'assets/[ext]/[name]-[hash].[ext]',
      },
    },

    // Target browsers
    target: 'es2015',

    // Chunk size warnings
    chunkSizeWarningLimit: 1000, // 1MB

    // CSS code splitting
    cssCodeSplit: true,

    // Report compressed size
    reportCompressedSize: true,
  },

  // Development server
  server: {
    port: 5173,
    strictPort: true,
    host: true,
    open: true,
  },

  // Preview server
  preview: {
    port: 4173,
    strictPort: true,
    host: true,
  },

  // Path aliases
  resolve: {
    alias: {
      '@': '/src',
      '@components': '/src/components',
      '@pages': '/src/pages',
      '@services': '/src/services',
      '@store': '/src/store',
      '@utils': '/src/utils',
      '@hooks': '/src/hooks',
    },
  },

  // Optimize dependencies
  optimizeDeps: {
    include: ['react', 'react-dom', 'react-router-dom', 'zustand', 'axios', 'marked', 'highlight.js'],
  },

  // Environment variables
  envPrefix: 'VITE_',
}));
