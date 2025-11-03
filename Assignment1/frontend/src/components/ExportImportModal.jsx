import { useState } from 'react';
import useConversationStore from '../store/conversationStore';
import LoadingSpinner from './LoadingSpinner';

/**
 * ExportImportModal component
 * Handles conversation export and import functionality
 */
const ExportImportModal = ({ isOpen, onClose, conversationId = null }) => {
  const { exportConversation, importConversations } = useConversationStore();
  const [activeTab, setActiveTab] = useState('export');
  const [exportFormat, setExportFormat] = useState('json');
  const [isExporting, setIsExporting] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [importFile, setImportFile] = useState(null);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const handleExport = async () => {
    if (!conversationId) {
      setError('No conversation selected for export');
      return;
    }

    setIsExporting(true);
    setError(null);
    try {
      const data = await exportConversation(conversationId, exportFormat);

      // Create download
      const blob = new Blob(
        [exportFormat === 'json' ? JSON.stringify(data, null, 2) : data],
        { type: exportFormat === 'json' ? 'application/json' : 'text/markdown' }
      );
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `conversation-${conversationId}.${exportFormat}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setSuccess('Conversation exported successfully!');
      setTimeout(() => {
        onClose();
      }, 2000);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsExporting(false);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImportFile(file);
      setError(null);
    }
  };

  const handleImport = async () => {
    if (!importFile) {
      setError('Please select a file to import');
      return;
    }

    setIsImporting(true);
    setError(null);
    try {
      const fileContent = await importFile.text();
      const data = JSON.parse(fileContent);

      await importConversations(data);
      setSuccess('Conversations imported successfully!');
      setImportFile(null);
      setTimeout(() => {
        onClose();
      }, 2000);
    } catch (err) {
      if (err instanceof SyntaxError) {
        setError('Invalid JSON file. Please upload a valid conversation export.');
      } else {
        setError(err.message);
      }
    } finally {
      setIsImporting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="export-import-modal-title"
    >
      <div
        className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl max-w-lg w-full"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h2
              id="export-import-modal-title"
              className="text-2xl font-semibold text-gray-900 dark:text-gray-100"
            >
              Export / Import
            </h2>
            <button
              onClick={onClose}
              className="btn-icon"
              aria-label="Close modal"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Tabs */}
          <div className="flex gap-4 border-b border-gray-200 dark:border-gray-700">
            <button
              onClick={() => setActiveTab('export')}
              className={`pb-2 px-1 font-medium transition-colors ${
                activeTab === 'export'
                  ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400'
                  : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
              }`}
            >
              Export
            </button>
            <button
              onClick={() => setActiveTab('import')}
              className={`pb-2 px-1 font-medium transition-colors ${
                activeTab === 'import'
                  ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400'
                  : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
              }`}
            >
              Import
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6">
          {error && (
            <div className="mb-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3 text-sm text-red-700 dark:text-red-400">
              {error}
            </div>
          )}

          {success && (
            <div className="mb-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-3 text-sm text-green-700 dark:text-green-400">
              {success}
            </div>
          )}

          {activeTab === 'export' ? (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Export Format
                </label>
                <div className="flex gap-3">
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="radio"
                      name="format"
                      value="json"
                      checked={exportFormat === 'json'}
                      onChange={(e) => setExportFormat(e.target.value)}
                      className="mr-2"
                    />
                    <span className="text-sm text-gray-700 dark:text-gray-300">JSON</span>
                  </label>
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="radio"
                      name="format"
                      value="markdown"
                      checked={exportFormat === 'markdown'}
                      onChange={(e) => setExportFormat(e.target.value)}
                      className="mr-2"
                    />
                    <span className="text-sm text-gray-700 dark:text-gray-300">Markdown</span>
                  </label>
                </div>
              </div>

              <p className="text-sm text-gray-600 dark:text-gray-400">
                {conversationId
                  ? `Export the current conversation as ${exportFormat.toUpperCase()}`
                  : 'Please select a conversation to export'}
              </p>

              <button
                onClick={handleExport}
                disabled={!conversationId || isExporting}
                className="w-full btn-primary"
              >
                {isExporting ? (
                  <span className="flex items-center justify-center gap-2">
                    <LoadingSpinner size="sm" />
                    Exporting...
                  </span>
                ) : (
                  'Export Conversation'
                )}
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Import File
                </label>
                <input
                  type="file"
                  accept=".json"
                  onChange={handleFileSelect}
                  className="w-full text-sm text-gray-700 dark:text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-medium file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 dark:file:bg-blue-900/20 dark:file:text-blue-400"
                />
              </div>

              {importFile && (
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3 text-sm">
                  <p className="text-gray-700 dark:text-gray-300">
                    <span className="font-medium">Selected file:</span> {importFile.name}
                  </p>
                  <p className="text-gray-500 dark:text-gray-400 text-xs mt-1">
                    Size: {(importFile.size / 1024).toFixed(2)} KB
                  </p>
                </div>
              )}

              <p className="text-sm text-gray-600 dark:text-gray-400">
                Upload a JSON file exported from this application to import conversations.
              </p>

              <button
                onClick={handleImport}
                disabled={!importFile || isImporting}
                className="w-full btn-primary"
              >
                {isImporting ? (
                  <span className="flex items-center justify-center gap-2">
                    <LoadingSpinner size="sm" />
                    Importing...
                  </span>
                ) : (
                  'Import Conversations'
                )}
              </button>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-gray-200 dark:border-gray-700 flex justify-end">
          <button onClick={onClose} className="btn-secondary">
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default ExportImportModal;
