// hooks/useApi.ts
import { useState, useCallback } from 'react';
import { api } from '@/lib/api';

interface UseApiOptions {
  onSuccess?: (data: any) => void;
  onError?: (error: string) => void;
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

      try {
        const response = await apiCall();
        
        if (!response.success) {
          throw new Error(response.error || 'Request failed');
        }

        if (options.onSuccess) {
          options.onSuccess(response);
        }

        return response.data || response;
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        setError(errorMessage);
        
        if (options.onError) {
          options.onError(errorMessage);
        }
        
        return null;
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return {
    loading,
    error,
    execute,
    resetError: () => setError(null),
  };
}