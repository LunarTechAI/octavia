"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';

// User interface matching backend user model
interface User {
  id: string;
  email: string;
  name: string;
  token: string;
  credits: number;
  verified: boolean;
}

// Available methods and state from the user context
interface UserContextType {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<{ success: boolean; error?: string; requiresVerification?: boolean }>;
  signup: (email: string, password: string, name: string) => Promise<{ success: boolean; error?: string; requiresVerification?: boolean; message?: string }>;
  logout: () => Promise<void>;
  updateCredits: (newCredits: number) => void;
  isAuthenticated: boolean;
  setUser: (user: User | null) => void;
  verifyEmail: (token: string) => Promise<{ success: boolean; error?: string }>;
  resendVerification: (email: string) => Promise<{ success: boolean; error?: string; message?: string }>;
  fetchUserProfile: () => Promise<void>;
}

// Create context with undefined default value
const UserContext = createContext<UserContextType | undefined>(undefined);

// Provider component that wraps the app and manages user state
export function UserProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  // Check for existing session on app load
  useEffect(() => {
    const storedUser = localStorage.getItem('octavia_user');
    if (storedUser) {
      try {
        const parsedUser = JSON.parse(storedUser);
        setUser(parsedUser);
        
        // Validate token by fetching fresh user profile
        fetchUserProfile().catch(() => {
          // Token expired or invalid, clear session
          localStorage.removeItem('octavia_user');
          setUser(null);
        });
      } catch (error) {
        console.error('Corrupted user data in localStorage:', error);
        localStorage.removeItem('octavia_user');
      }
    }
    setLoading(false);
  }, []);

  // Fetch latest user data from server
  const fetchUserProfile = async () => {
    if (!user) return;
    
    try {
      const response = await api.getUserProfile();
      if (response.success && response.user) {
        const updatedUser = {
          ...user,
          name: response.user.name,
          credits: response.user.credits,
          verified: response.user.verified
        };
        setUser(updatedUser);
        localStorage.setItem('octavia_user', JSON.stringify(updatedUser));
      }
    } catch (error) {
      console.error('Failed to refresh user profile:', error);
      throw error;
    }
  };

  // Handle user login with email and password
  const login = async (email: string, password: string) => {
    setLoading(true);
    try {
      const response = await api.login(email, password);
      
      if (response.success) {
        const userData: User = {
          id: response.user.id,
          email: response.user.email,
          name: response.user.name,
          token: response.token,
          credits: response.user.credits,
          verified: response.user.verified
        };
        
        setUser(userData);
        localStorage.setItem('octavia_user', JSON.stringify(userData));
        return { success: true };
      } else {
        return { 
          success: false, 
          error: response.error,
          requiresVerification: response.message?.includes('verify') || false
        };
      }
    } catch (error) {
      return { success: false, error: 'Network connection failed' };
    } finally {
      setLoading(false);
    }
  };

  // Create new user account and handle email verification flow
  const signup = async (email: string, password: string, name: string) => {
    setLoading(true);
    try {
      const response = await api.signup(email, password, name);
      
      if (response.success) {
        if (response.requires_verification) {
          // User needs to verify email before login
          localStorage.setItem('pending_verification_email', email);
          
          return { 
            success: true, 
            requiresVerification: true,
            message: response.message
          };
        }
        
        // Account created without verification requirement
        const userData: User = {
          id: response.user?.id || '',
          email: response.user?.email || email,
          name: response.user?.name || name,
          token: response.token || '',
          credits: response.user?.credits || 0,
          verified: response.user?.verified || true
        };
        
        setUser(userData);
        localStorage.setItem('octavia_user', JSON.stringify(userData));
        return { success: true };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      return { success: false, error: 'Network connection failed' };
    } finally {
      setLoading(false);
    }
  };

  // Verify email using token from verification link
  const verifyEmail = async (token: string) => {
    setLoading(true);
    try {
      const response = await api.verifyEmail(token);
      
      if (response.success) {
        const userData: User = {
          id: response.user.id,
          email: response.user.email,
          name: response.user.name,
          token: response.token,
          credits: response.user.credits,
          verified: response.user.verified
        };
        
        setUser(userData);
        localStorage.setItem('octavia_user', JSON.stringify(userData));
        localStorage.removeItem('pending_verification_email');
        
        return { success: true };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      return { success: false, error: 'Network connection failed' };
    } finally {
      setLoading(false);
    }
  };

  // Request new verification email
  const resendVerification = async (email: string) => {
    setLoading(true);
    try {
      const response = await api.resendVerification(email);
      
      if (response.success) {
        return { success: true, message: response.message };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      return { success: false, error: 'Network connection failed' };
    } finally {
      setLoading(false);
    }
  };

  // Logout user from current session
  const logout = async () => {
    try {
      await api.logout();
    } catch (error) {
      console.warn('Server logout failed, clearing client session:', error);
    } finally {
      // Always clear local session data
      setUser(null);
      localStorage.removeItem('octavia_user');
      localStorage.removeItem('pending_verification_email');
      router.push('/login');
    }
  };

  // Update user's credit balance locally (for real-time updates)
  const updateCredits = (newCredits: number) => {
    if (user) {
      const updatedUser = { ...user, credits: newCredits };
      setUser(updatedUser);
      localStorage.setItem('octavia_user', JSON.stringify(updatedUser));
    }
  };

  // Context value provided to consuming components
  const value: UserContextType = {
    user,
    loading,
    login,
    signup,
    logout,
    updateCredits,
    isAuthenticated: !!user,
    setUser,
    verifyEmail,
    resendVerification,
    fetchUserProfile,
  };

  return (
    <UserContext.Provider value={value}>
      {children}
    </UserContext.Provider>
  );
}

// Hook for accessing user context in components
export const useUser = (): UserContextType => {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};