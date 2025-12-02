// API service for handling all backend communication
// Manages authentication, video translation, and user operations
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Standard response format from all API endpoints
interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  user?: any;
  token?: string;
  job_id?: string;
  download_url?: string;
  remaining_credits?: number;
  requires_verification?: boolean;
}

interface User {
  id: string;
  email: string;
  name: string;
  credits: number;
  verified: boolean;
  created_at?: string;
}

class ApiService {
  // Core request handler for all API calls
  // Handles authentication tokens, error parsing, and response formatting
  private async request<T = any>(
    endpoint: string,
    options: RequestInit = {},
    requiresAuth: boolean = false
  ): Promise<ApiResponse<T>> {
    try {
      const url = `${API_BASE_URL}${endpoint}`;
      
      // Debug logging for development
      console.log(`API Request: ${options.method || 'GET'} ${url}`);
      
      // Get JWT token from localStorage for authenticated requests
      let token: string | null = null;
      if (typeof window !== 'undefined') {
        const userStr = localStorage.getItem('octavia_user');
        if (userStr) {
          try {
            const user = JSON.parse(userStr);
            token = user.token;
          } catch (error) {
            console.error('Failed to parse user token from localStorage:', error);
          }
        }
      }
      
      // Setup headers for the request
      const headers: Record<string, string> = {
        'Accept': 'application/json',
      };
      
      // Add Authorization header for protected endpoints
      if (requiresAuth && token) {
        headers['Authorization'] = `Bearer ${token}`;
      }
      
      // Set Content-Type for JSON payloads (skip for FormData)
      const isFormData = options.body instanceof FormData;
      if (!isFormData && options.body) {
        headers['Content-Type'] = 'application/json';
      }
      
      const response = await fetch(url, {
        ...options,
        headers: {
          ...headers,
          ...options.headers,
        },
      });

      // Handle HTTP errors (4xx, 5xx responses)
      if (!response.ok) {
        let errorData: any = {};
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        
        try {
          // Try to extract error message from response body
          const responseText = await response.text();
          
          if (responseText) {
            try {
              errorData = JSON.parse(responseText);
              
              // Look for error message in common response fields
              errorMessage = errorData.detail || 
                           errorData.error || 
                           errorData.message || 
                           (typeof errorData === 'string' ? errorData : errorMessage);
            } catch {
              // Response wasn't JSON, use text directly
              errorMessage = responseText || errorMessage;
            }
          }
        } catch (parseError) {
          console.error('Failed to parse error response:', parseError);
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        
        throw new Error(errorMessage);
      }

      // Parse successful response
      const responseText = await response.text();
      let data: any = {};
      
      if (responseText) {
        try {
          data = JSON.parse(responseText);
        } catch (parseError) {
          console.error('Server returned invalid JSON:', parseError);
          throw new Error('Server returned an invalid response format');
        }
      }
      
      return data;
    } catch (error) {
      console.error('API request failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'An unexpected error occurred',
      };
    }
  }

  // --- AUTHENTICATION ENDPOINTS ---

  // Register a new user account
  async signup(email: string, password: string, name: string) {
    const body = JSON.stringify({
      email: email,
      password: password,
      name: name
    });

    return this.request('/api/auth/signup', {
      method: 'POST',
      body: body,
    });
  }

  // Login existing user and receive authentication token
  async login(email: string, password: string) {
    const body = JSON.stringify({
      email: email,
      password: password
    });

    return this.request('/api/auth/login', {
      method: 'POST',
      body: body,
    });
  }

  // Logout user and invalidate session
  async logout() {
    const userStr = localStorage.getItem('octavia_user');
    let token: string | undefined;
    
    if (userStr) {
      try {
        const user = JSON.parse(userStr);
        token = user.token;
      } catch (error) {
        console.error('Failed to parse user token for logout:', error);
      }
    }
    
    return this.request('/api/auth/logout', {
      method: 'POST',
      headers: token ? {
        'Authorization': `Bearer ${token}`
      } : {},
    }, false);
  }

  // --- EMAIL VERIFICATION ---

  // Verify email address using token from verification email
  async verifyEmail(token: string) {
    const formData = new FormData();
    formData.append('token', token);

    return this.request('/api/auth/verify', {
      method: 'POST',
      body: formData,
    });
  }

  // Resend verification email to user
  async resendVerification(email: string) {
    const body = JSON.stringify({
      email: email
    });

    return this.request('/api/auth/resend-verification', {
      method: 'POST',
      body: body,
    });
  }

  // --- VIDEO TRANSLATION ---

  // Upload video file for translation to target language
  async translateVideo(
    file: File,
    targetLanguage: string = 'es',
    userEmail: string
  ) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_language', targetLanguage);

    return this.request('/api/translate/video', {
      method: 'POST',
      body: formData,
    }, true);
  }

  // Check status of a translation job
  async getJobStatus(jobId: string) {
    return this.request<{
      job_id: string;
      status: string;
      progress: number;
      status_message?: string;
      download_url?: string;
      original_filename?: string;
      target_language?: string;
      error?: string;
    }>(`/api/jobs/${jobId}/status`, {
      method: 'GET',
    }, true);
  }

  // --- USER PROFILE ---

  // Get current user's profile information
  async getUserProfile() {
    return this.request<{
      user: User;
    }>('/api/user/profile', {
      method: 'GET',
    }, true);
  }

  // Get user's current credit balance
  async getUserCredits() {
    return this.request<{
      credits: number;
      email: string;
    }>('/api/user/credits', {
      method: 'GET',
    }, true);
  }

  // --- FILE DOWNLOAD ---

  // Download translated video file by job ID
  async downloadFile(jobId: string): Promise<Blob> {
    const userStr = localStorage.getItem('octavia_user');
    let token: string = '';
    
    if (userStr) {
      try {
        const user = JSON.parse(userStr);
        token = user.token;
      } catch (error) {
        console.error('Failed to parse user token for download:', error);
      }
    }
    
    const url = `${API_BASE_URL}/api/download/${jobId}`;
    
    const response = await fetch(url, {
      headers: token ? {
        'Authorization': `Bearer ${token}`
      } : {},
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Download failed: ${response.statusText}`);
    }
    
    return await response.blob();
  }

  // --- SYSTEM HEALTH ---

  // Check if backend API is reachable
  async healthCheck() {
    return this.request('/api/health');
  }

  // Login with demo/test account
  async demoLogin() {
    return this.request('/api/auth/demo-login', {
      method: 'POST',
    });
  }

  // Test basic API connectivity
  async testConnection() {
    return this.request('/');
  }

  // Direct signup test (bypasses request wrapper for debugging)
  async testSignupDirect(email: string, password: string, name: string) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/signup`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({ email, password, name }),
      });
      
      const responseText = await response.text();
      
      if (!response.ok) {
        throw new Error(`Signup failed: ${response.status} ${response.statusText}`);
      }
      
      let data = {};
      if (responseText) {
        try {
          data = JSON.parse(responseText);
        } catch (e) {
          console.error('Response was not valid JSON:', e);
          data = { raw: responseText };
        }
      }
      
      return data;
    } catch (error) {
      console.error('Direct signup test error:', error);
      throw error;
    }
  }
}

export const api = new ApiService();