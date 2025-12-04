import { useState, useCallback } from 'react';
import { api } from '@/lib/api';

interface UseApiOptions {
  onSuccess?: (data: any) => void;
  onError?: (error: string) => void;
  showDebug?: boolean;
}

export function useApi() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const execute = useCallback(
    async <T>(
      apiCall: () => Promise<any>,
      options: UseApiOptions = {}
    ): Promise<T | null> => {
      setLoading(true);
      setError(null);

      if (options.showDebug) {
        console.log('useApi: Starting API call...');
      }

      try {
        const response = await apiCall();
        
        if (options.showDebug) {
          console.log('useApi: Response received:', response);
        }
        
        // Handle the case where response is null or undefined
        if (!response) {
          const errorMsg = 'No response received from server';
          if (options.showDebug) {
            console.log('useApi: No response from server');
          }
          setError(errorMsg);
          if (options.onError) {
            options.onError(errorMsg);
          }
          return null;
        }
        
        // Check if the API returned an error response (success: false)
        if (response.success === false) {
          const errorMsg = response.error || 'Request failed';
          if (options.showDebug) {
            console.log('useApi: API returned error:', errorMsg);
          }
          setError(errorMsg);
          if (options.onError) {
            options.onError(errorMsg);
          }
          return null;
        }

        // This should handle legacy code that might throw errors
        if (!response.success) {
          const errorMsg = response.error || 'Request failed';
          if (options.showDebug) {
            console.log('useApi: Throwing error:', errorMsg);
          }
          throw new Error(errorMsg);
        }

        if (options.onSuccess) {
          if (options.showDebug) {
            console.log('useApi: Calling onSuccess callback');
          }
          options.onSuccess(response);
        }

        const result = response.data || response;
        if (options.showDebug) {
          console.log('useApi: Returning result:', result);
        }
        return result;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        if (options.showDebug) {
          console.log('useApi: Caught error:', err);
        }
        setError(errorMessage);
        
        if (options.onError) {
          options.onError(errorMessage);
        }
        
        return null;
      } finally {
        if (options.showDebug) {
          console.log('useApi: Setting loading to false');
        }
        setLoading(false);
      }
    },
    []
  );

  const executeWithDebug = useCallback(
    async <T>(
      apiCall: () => Promise<any>,
      options: UseApiOptions = {}
    ): Promise<T | null> => {
      return execute(apiCall, { ...options, showDebug: true });
    },
    [execute]
  );

  return {
    loading,
    error,
    execute,
    executeWithDebug,
    resetError: () => setError(null),
  };
}