'use client';

import React, { createContext, useContext, useEffect, useState } from 'react';
import { User } from '@/types';
import { getCurrentUser, getToken, loginUser, loginWithGoogleApi, registerUser, setToken, updateCurrentUser, loginAsGuest } from '@/lib/api';

interface AuthContextType {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  authModalOpen: boolean;
  setAuthModalOpen: (open: boolean) => void;
  login: (email: string, pass: string) => Promise<void>;
  register: (email: string, pass: string, name: string) => Promise<void>;
  loginWithGoogle: (email: string, name: string, avatar?: string) => Promise<void>;
  loginGuest: () => Promise<void>;
  updateUser: (name?: string, avatar?: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setTokenState] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [authModalOpen, setAuthModalOpen] = useState(false);

  useEffect(() => {
    const existingToken = getToken();
    if (existingToken) {
      setTokenState(existingToken);
      getCurrentUser()
        .then((userData) => {
          setUser(userData);
          setIsLoading(false);
        })
        .catch(() => {
          // Token expired, smoothly fallback to a guest session
          setToken(null);
          loginAsGuest()
            .then((res) => {
              setTokenState(res.access_token);
              setUser(res.user);
            })
            .catch(() => {
              setTokenState(null);
              setUser(null);
            })
            .finally(() => setIsLoading(false));
        });
    } else {
      // First-time visitor: auto-initialize seamless guest session
      loginAsGuest()
        .then((res) => {
          setTokenState(res.access_token);
          setUser(res.user);
        })
        .catch(() => {
          setTokenState(null);
          setUser(null);
        })
        .finally(() => setIsLoading(false));
    }
  }, []);

  const loginGuest = async () => {
    const res = await loginAsGuest();
    setTokenState(res.access_token);
    setUser(res.user);
    setAuthModalOpen(false);
  };

  const login = async (email: string, pass: string) => {
    const res = await loginUser(email, pass);
    setTokenState(res.access_token);
    setUser(res.user);
    setAuthModalOpen(false);
  };

  const register = async (email: string, pass: string, name: string) => {
    const res = await registerUser(email, pass, name);
    setTokenState(res.access_token);
    setUser(res.user);
    setAuthModalOpen(false);
  };

  const loginWithGoogle = async (email: string, name: string, avatar?: string) => {
    const res = await loginWithGoogleApi(email, name, avatar);
    setTokenState(res.access_token);
    setUser(res.user);
    setAuthModalOpen(false);
  };

  const updateUser = async (name?: string, avatar?: string) => {
    const updated = await updateCurrentUser({ name, avatar_url: avatar });
    setUser(updated);
  };

  const logout = () => {
    setToken(null);
    setTokenState(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isLoading,
        authModalOpen,
        setAuthModalOpen,
        login,
        register,
        loginWithGoogle,
        loginGuest,
        updateUser,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

