/**
 * Skeleton Loader Components
 * Displays loading placeholders for async content
 */

/**
 * Base Skeleton Component
 */
export const Skeleton = ({ width = '100%', height = '1rem', className = '', variant = 'rounded' }) => {
  const variantClasses = {
    rounded: 'rounded-md',
    circular: 'rounded-full',
    rectangular: 'rounded-none',
  };

  return (
    <div
      className={`
        bg-gray-200 dark:bg-gray-700
        animate-pulse
        ${variantClasses[variant]}
        ${className}
      `}
      style={{ width, height }}
      role="status"
      aria-label="Loading"
    >
      <span className="sr-only">Loading...</span>
    </div>
  );
};

/**
 * Skeleton for conversation list items
 */
export const ConversationSkeleton = () => {
  return (
    <div className="p-3 border-b border-border-color" role="status" aria-label="Loading conversation">
      <div className="flex items-start gap-3">
        <Skeleton width="40px" height="40px" variant="circular" />
        <div className="flex-1 space-y-2">
          <Skeleton width="60%" height="16px" />
          <Skeleton width="90%" height="14px" />
          <Skeleton width="40%" height="12px" />
        </div>
      </div>
    </div>
  );
};

/**
 * Skeleton for conversation sidebar
 */
export const ConversationsSkeleton = ({ count = 5 }) => {
  return (
    <div className="space-y-0" role="status" aria-label="Loading conversations">
      {Array.from({ length: count }).map((_, index) => (
        <ConversationSkeleton key={index} />
      ))}
    </div>
  );
};

/**
 * Skeleton for message bubbles
 */
export const MessageSkeleton = ({ isUser = false }) => {
  return (
    <div
      className={`flex gap-3 mb-4 ${isUser ? 'justify-end' : 'justify-start'}`}
      role="status"
      aria-label="Loading message"
    >
      {!isUser && <Skeleton width="32px" height="32px" variant="circular" />}
      <div className={`max-w-3xl ${isUser ? 'items-end' : 'items-start'} flex flex-col gap-2`}>
        <Skeleton width={isUser ? '200px' : '300px'} height="16px" />
        <Skeleton width={isUser ? '150px' : '250px'} height="16px" />
        <Skeleton width={isUser ? '180px' : '200px'} height="16px" />
      </div>
      {isUser && <Skeleton width="32px" height="32px" variant="circular" />}
    </div>
  );
};

/**
 * Skeleton for chat messages
 */
export const MessagesSkeleton = ({ count = 3 }) => {
  return (
    <div className="space-y-4 p-4" role="status" aria-label="Loading messages">
      {Array.from({ length: count }).map((_, index) => (
        <MessageSkeleton key={index} isUser={index % 2 === 0} />
      ))}
    </div>
  );
};

/**
 * Skeleton for model selector
 */
export const ModelSkeleton = () => {
  return (
    <div className="p-3 border-b border-border-color" role="status" aria-label="Loading model">
      <div className="flex items-center gap-3">
        <Skeleton width="24px" height="24px" variant="circular" />
        <div className="flex-1 space-y-2">
          <Skeleton width="40%" height="16px" />
          <Skeleton width="70%" height="12px" />
        </div>
        <Skeleton width="60px" height="24px" />
      </div>
    </div>
  );
};

/**
 * Skeleton for models list
 */
export const ModelsSkeleton = ({ count = 4 }) => {
  return (
    <div className="space-y-0" role="status" aria-label="Loading models">
      {Array.from({ length: count }).map((_, index) => (
        <ModelSkeleton key={index} />
      ))}
    </div>
  );
};

/**
 * Skeleton for card layouts
 */
export const CardSkeleton = () => {
  return (
    <div className="card p-6" role="status" aria-label="Loading content">
      <Skeleton width="60%" height="24px" className="mb-4" />
      <Skeleton width="100%" height="16px" className="mb-2" />
      <Skeleton width="90%" height="16px" className="mb-2" />
      <Skeleton width="80%" height="16px" className="mb-4" />
      <div className="flex gap-2 mt-4">
        <Skeleton width="80px" height="36px" />
        <Skeleton width="80px" height="36px" />
      </div>
    </div>
  );
};

/**
 * Skeleton for table rows
 */
export const TableRowSkeleton = ({ columns = 3 }) => {
  return (
    <tr role="status" aria-label="Loading table row">
      {Array.from({ length: columns }).map((_, index) => (
        <td key={index} className="p-3">
          <Skeleton width="100%" height="16px" />
        </td>
      ))}
    </tr>
  );
};

/**
 * Skeleton for full page loading
 */
export const PageSkeleton = () => {
  return (
    <div className="min-h-screen bg-bg-secondary p-8" role="status" aria-label="Loading page">
      <div className="max-w-6xl mx-auto">
        <Skeleton width="300px" height="40px" className="mb-8" />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {Array.from({ length: 6 }).map((_, index) => (
            <CardSkeleton key={index} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default Skeleton;
