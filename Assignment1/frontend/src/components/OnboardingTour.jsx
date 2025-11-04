import { useState, useEffect } from 'react';

/**
 * Onboarding Tour Component
 * Provides a guided walkthrough for new users
 */
const OnboardingTour = ({ onComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Check if user has completed onboarding
    const hasCompletedOnboarding = localStorage.getItem('onboarding_completed');
    if (!hasCompletedOnboarding) {
      // Small delay before showing the tour
      setTimeout(() => setIsVisible(true), 500);
    }
  }, []);

  const steps = [
    {
      title: 'Welcome to Ollama Web GUI!',
      content: 'Let\'s take a quick tour to help you get started with the application.',
      target: null,
    },
    {
      title: 'Start a Conversation',
      content: 'Click the "New Chat" button to begin a conversation with an AI model.',
      target: 'new-chat-button',
    },
    {
      title: 'Select Your Model',
      content: 'Choose from available AI models. Different models have different capabilities and speeds.',
      target: 'model-selector',
    },
    {
      title: 'Chat Interface',
      content: 'Type your messages here and press Enter or click Send to chat with the AI.',
      target: 'chat-input',
    },
    {
      title: 'Conversation History',
      content: 'All your conversations are saved here. Click any conversation to continue it.',
      target: 'conversation-sidebar',
    },
    {
      title: 'Settings & Export',
      content: 'Access settings, export conversations, and customize your experience from these buttons.',
      target: 'settings-button',
    },
  ];

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      handleComplete();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSkip = () => {
    handleComplete();
  };

  const handleComplete = () => {
    localStorage.setItem('onboarding_completed', 'true');
    setIsVisible(false);
    if (onComplete) {
      onComplete();
    }
  };

  if (!isVisible) return null;

  const currentStepData = steps[currentStep];

  return (
    <>
      {/* Overlay */}
      <div
        className="fixed inset-0 bg-black bg-opacity-50 z-40"
        onClick={handleSkip}
        aria-hidden="true"
      />

      {/* Tour Card */}
      <div
        className="fixed bottom-8 right-8 bg-bg-primary dark:bg-gray-800 rounded-xl shadow-2xl border border-border-color z-50 max-w-md w-full mx-4"
        role="dialog"
        aria-labelledby="onboarding-title"
        aria-describedby="onboarding-description"
      >
        <div className="p-6">
          {/* Header */}
          <div className="flex items-start justify-between mb-4">
            <div className="flex-1">
              <h2
                id="onboarding-title"
                className="text-xl font-bold text-text-primary mb-2"
              >
                {currentStepData.title}
              </h2>
              <p
                id="onboarding-description"
                className="text-text-secondary text-sm"
              >
                {currentStepData.content}
              </p>
            </div>
            <button
              onClick={handleSkip}
              className="text-text-tertiary hover:text-text-primary transition-colors ml-2"
              aria-label="Skip tour"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </button>
          </div>

          {/* Progress */}
          <div className="mb-6">
            <div className="flex items-center justify-between text-xs text-text-tertiary mb-2">
              <span>Step {currentStep + 1} of {steps.length}</span>
              <span>{Math.round(((currentStep + 1) / steps.length) * 100)}%</span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className="bg-accent-primary h-2 rounded-full transition-all duration-300"
                style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
                role="progressbar"
                aria-valuenow={currentStep + 1}
                aria-valuemin={0}
                aria-valuemax={steps.length}
              />
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-3 justify-between">
            <button
              onClick={handleSkip}
              className="btn-secondary flex-1"
            >
              Skip Tour
            </button>
            <div className="flex gap-2 flex-1">
              {currentStep > 0 && (
                <button
                  onClick={handlePrevious}
                  className="btn-secondary flex-1"
                >
                  Previous
                </button>
              )}
              <button
                onClick={handleNext}
                className="btn-primary flex-1"
              >
                {currentStep === steps.length - 1 ? 'Finish' : 'Next'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default OnboardingTour;
