import { useState, useEffect } from 'react';
import modelsService from '../services/modelsService';
import LoadingSpinner from './LoadingSpinner';

/**
 * ModelSelectorModal component
 * Modal for selecting AI models
 */
const ModelSelectorModal = ({ isOpen, onClose, currentModel, onSelectModel }) => {
  const [models, setModels] = useState([]);
  const [filteredModels, setFilteredModels] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedModel, setSelectedModel] = useState(currentModel);

  useEffect(() => {
    if (isOpen) {
      loadModels();
    }
  }, [isOpen]);

  useEffect(() => {
    // Filter models based on search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      const filtered = models.filter(
        (model) =>
          model.name?.toLowerCase().includes(query) ||
          model.description?.toLowerCase().includes(query)
      );
      setFilteredModels(filtered);
    } else {
      setFilteredModels(models);
    }
  }, [searchQuery, models]);

  const loadModels = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await modelsService.list();
      setModels(data.models || data || []);
      setFilteredModels(data.models || data || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelect = (model) => {
    setSelectedModel(model.name);
  };

  const handleConfirm = () => {
    if (selectedModel) {
      onSelectModel(selectedModel);
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
    >
      <div
        className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl max-w-2xl w-full max-h-[80vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h2
              id="modal-title"
              className="text-2xl font-semibold text-gray-900 dark:text-gray-100"
            >
              Select Model
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

          {/* Search input */}
          <div className="relative">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search models..."
              className="w-full input-field pl-10"
              aria-label="Search models"
            />
            <svg
              className="w-5 h-5 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 dark:text-gray-500"
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

        {/* Models list */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-6">
          {isLoading ? (
            <div className="flex justify-center py-12">
              <LoadingSpinner size="md" message="Loading models..." />
            </div>
          ) : error ? (
            <div className="text-center py-12">
              <div className="text-red-500 dark:text-red-400 mb-4">
                <svg className="w-12 h-12 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p className="font-medium">{error}</p>
              </div>
              <button onClick={loadModels} className="btn-primary">
                Retry
              </button>
            </div>
          ) : filteredModels.length === 0 ? (
            <div className="text-center py-12 text-gray-500 dark:text-gray-400">
              {searchQuery ? 'No models found matching your search' : 'No models available'}
            </div>
          ) : (
            <div className="space-y-2">
              {filteredModels.map((model) => (
                <div
                  key={model.name}
                  onClick={() => handleSelect(model)}
                  className={`
                    cursor-pointer rounded-lg p-4 border-2 transition-all
                    ${
                      selectedModel === model.name
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'
                    }
                  `}
                  role="button"
                  tabIndex={0}
                  aria-label={`Select ${model.name}`}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      handleSelect(model);
                    }
                  }}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-1">
                        {model.name}
                      </h3>
                      {model.description && (
                        <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                          {model.description}
                        </p>
                      )}
                      <div className="flex items-center gap-3 text-xs text-gray-500 dark:text-gray-400">
                        {model.size && (
                          <span className="bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                            {model.size}
                          </span>
                        )}
                        {model.parameters && (
                          <span className="bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                            {model.parameters}
                          </span>
                        )}
                      </div>
                    </div>
                    {selectedModel === model.name && (
                      <div className="ml-3">
                        <svg className="w-6 h-6 text-blue-500" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
                        </svg>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-gray-200 dark:border-gray-700 flex items-center justify-between">
          <div className="text-sm text-gray-600 dark:text-gray-400">
            {selectedModel && (
              <span>
                Selected: <span className="font-medium text-gray-900 dark:text-gray-100">{selectedModel}</span>
              </span>
            )}
          </div>
          <div className="flex gap-3">
            <button onClick={onClose} className="btn-secondary">
              Cancel
            </button>
            <button
              onClick={handleConfirm}
              disabled={!selectedModel}
              className="btn-primary"
            >
              Confirm
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelSelectorModal;
