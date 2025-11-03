import { useState, useEffect } from 'react';
import useConfigStore from '../store/configStore';
import promptsService from '../services/promptsService';
import LoadingSpinner from './LoadingSpinner';

/**
 * SettingsModal component
 * Modal for system prompt editing and other settings
 */
const SettingsModal = ({ isOpen, onClose }) => {
  const {
    systemPrompt,
    setSystemPrompt,
    temperature,
    setTemperature,
    maxTokens,
    setMaxTokens,
  } = useConfigStore();

  const [localPrompt, setLocalPrompt] = useState(systemPrompt);
  const [templates, setTemplates] = useState([]);
  const [isLoadingTemplates, setIsLoadingTemplates] = useState(false);
  const [activeTab, setActiveTab] = useState('prompt'); // 'prompt' | 'parameters'

  useEffect(() => {
    if (isOpen) {
      setLocalPrompt(systemPrompt);
      loadTemplates();
    }
  }, [isOpen, systemPrompt]);

  const loadTemplates = async () => {
    setIsLoadingTemplates(true);
    try {
      const data = await promptsService.getTemplates();
      setTemplates(data.templates || data || []);
    } catch (error) {
      console.error('Failed to load templates:', error);
      // Use fallback templates
      setTemplates([
        { name: 'Default', content: 'You are a helpful assistant.' },
        { name: 'Code Assistant', content: 'You are an expert programming assistant. Provide clear, concise code examples and explanations.' },
        { name: 'Creative Writer', content: 'You are a creative writing assistant. Help with storytelling, poetry, and creative content.' },
        { name: 'Teacher', content: 'You are a patient and knowledgeable teacher. Explain concepts clearly with examples.' },
      ]);
    } finally {
      setIsLoadingTemplates(false);
    }
  };

  const handleSave = () => {
    setSystemPrompt(localPrompt);
    onClose();
  };

  const handleReset = () => {
    const defaultPrompt = 'You are a helpful assistant.';
    setLocalPrompt(defaultPrompt);
    setSystemPrompt(defaultPrompt);
  };

  const handleTemplateSelect = (template) => {
    setLocalPrompt(template.content);
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="settings-modal-title"
    >
      <div
        className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl max-w-3xl w-full max-h-[85vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <h2
              id="settings-modal-title"
              className="text-2xl font-semibold text-gray-900 dark:text-gray-100"
            >
              Settings
            </h2>
            <button
              onClick={onClose}
              className="btn-icon"
              aria-label="Close settings"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Tabs */}
          <div className="flex gap-4 mt-4 border-b border-gray-200 dark:border-gray-700">
            <button
              onClick={() => setActiveTab('prompt')}
              className={`pb-2 px-1 font-medium transition-colors ${
                activeTab === 'prompt'
                  ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400'
                  : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
              }`}
            >
              System Prompt
            </button>
            <button
              onClick={() => setActiveTab('parameters')}
              className={`pb-2 px-1 font-medium transition-colors ${
                activeTab === 'parameters'
                  ? 'text-blue-600 dark:text-blue-400 border-b-2 border-blue-600 dark:border-blue-400'
                  : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
              }`}
            >
              Parameters
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto custom-scrollbar p-6">
          {activeTab === 'prompt' ? (
            <div className="space-y-4">
              {/* Templates */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Prompt Templates
                </label>
                {isLoadingTemplates ? (
                  <div className="flex justify-center py-4">
                    <LoadingSpinner size="sm" />
                  </div>
                ) : (
                  <div className="grid grid-cols-2 gap-2">
                    {templates.map((template, index) => (
                      <button
                        key={index}
                        onClick={() => handleTemplateSelect(template)}
                        className="btn-secondary text-left text-sm"
                      >
                        {template.name}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Prompt editor */}
              <div>
                <label
                  htmlFor="system-prompt"
                  className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
                >
                  System Prompt
                </label>
                <textarea
                  id="system-prompt"
                  value={localPrompt}
                  onChange={(e) => setLocalPrompt(e.target.value)}
                  rows={10}
                  className="textarea-field w-full"
                  placeholder="Enter system prompt..."
                  aria-label="System prompt editor"
                />
                <div className="flex items-center justify-between mt-2">
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {localPrompt.length} characters
                  </span>
                  <button
                    onClick={handleReset}
                    className="text-xs text-blue-600 dark:text-blue-400 hover:underline"
                  >
                    Reset to default
                  </button>
                </div>
              </div>

              {/* Preview */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Preview
                </label>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 text-sm text-gray-700 dark:text-gray-300">
                  {localPrompt || <span className="text-gray-400 dark:text-gray-500 italic">No prompt set</span>}
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Temperature */}
              <div>
                <label
                  htmlFor="temperature"
                  className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
                >
                  Temperature: {temperature}
                </label>
                <input
                  id="temperature"
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full"
                  aria-label="Temperature slider"
                />
                <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                  <span>More focused</span>
                  <span>More creative</span>
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                  Controls randomness in responses. Lower values make outputs more deterministic.
                </p>
              </div>

              {/* Max Tokens */}
              <div>
                <label
                  htmlFor="max-tokens"
                  className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
                >
                  Max Tokens: {maxTokens}
                </label>
                <input
                  id="max-tokens"
                  type="range"
                  min="100"
                  max="4000"
                  step="100"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  className="w-full"
                  aria-label="Max tokens slider"
                />
                <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                  <span>100</span>
                  <span>4000</span>
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                  Maximum length of the generated response.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-gray-200 dark:border-gray-700 flex justify-end gap-3">
          <button onClick={onClose} className="btn-secondary">
            Cancel
          </button>
          <button onClick={handleSave} className="btn-primary">
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
