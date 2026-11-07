'use client';

import React, { useState } from 'react';
import { useAuth } from '@/context/AuthContext';
import { X, Sparkles, AlertCircle, CheckCircle2, Lock, Mail, User as UserIcon } from 'lucide-react';

export default function AuthModal() {
  const { authModalOpen, setAuthModalOpen, login, register, loginWithGoogle } = useAuth();
  const [tab, setTab] = useState<'signin' | 'signup'>('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  if (!authModalOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      if (tab === 'signin') {
        await login(email, password);
      } else {
        if (!name.trim()) throw new Error('Name is required');
        await register(email, password, name);
      }
    } catch (err: any) {
      setError(err.message || 'Authentication failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleDemo = async () => {
    setError(null);
    setIsLoading(true);
    try {
      await loginWithGoogle('suresh.evaluator@gmail.com', 'Suresh Kumar', 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=100&h=100&fit=crop');
    } catch (err: any) {
      setError(err.message || 'Google login failed');
    } finally {
      setIsLoading(false);
    }
  };

  const fillQuickDemo = (demoEmail: string, demoPass: string) => {
    setEmail(demoEmail);
    setPassword(demoPass);
    if (tab === 'signup') {
      setName('Demo User');
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-xs">
      <div className="relative w-full max-w-md bg-[#161616] border border-zinc-800 rounded-2xl p-6 shadow-2xl animate-fade-in">
        {/* Close button */}
        <button
          onClick={() => setAuthModalOpen(false)}
          className="absolute right-4 top-4 p-1 text-zinc-400 hover:text-white rounded-lg hover:bg-zinc-800 transition"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="text-center mb-6">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center text-black font-black text-lg mx-auto mb-3 shadow-md">
            S
          </div>
          <h2 className="text-xl font-bold text-white">
            {tab === 'signin' ? 'Welcome back' : 'Create your account'}
          </h2>
          <p className="text-xs text-zinc-400 mt-1">
            Access your persistent chats, Groq hosted models & RAG documents
          </p>
        </div>

        {/* Tab Switcher */}
        <div className="flex rounded-xl bg-[#202020] p-1 mb-5 border border-zinc-800">
          <button
            type="button"
            onClick={() => {
              setTab('signin');
              setError(null);
            }}
            className={`flex-1 py-1.5 text-xs font-semibold rounded-lg transition ${
              tab === 'signin' ? 'bg-zinc-800 text-white shadow-xs' : 'text-zinc-400 hover:text-white'
            }`}
          >
            Sign in
          </button>
          <button
            type="button"
            onClick={() => {
              setTab('signup');
              setError(null);
            }}
            className={`flex-1 py-1.5 text-xs font-semibold rounded-lg transition ${
              tab === 'signup' ? 'bg-zinc-800 text-white shadow-xs' : 'text-zinc-400 hover:text-white'
            }`}
          >
            Create account
          </button>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="mb-4 p-3 rounded-xl bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Google OAuth Button */}
        <button
          type="button"
          onClick={handleGoogleDemo}
          disabled={isLoading}
          className="w-full flex items-center justify-center gap-3 px-4 py-2.5 rounded-xl border border-zinc-700 bg-white hover:bg-zinc-100 text-zinc-900 font-medium text-xs transition shadow-sm mb-4"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24">
            <path
              fill="#4285F4"
              d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
            />
            <path
              fill="#34A853"
              d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
            />
            <path
              fill="#FBBC05"
              d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
            />
            <path
              fill="#EA4335"
              d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
            />
          </svg>
          <span>Continue with Google</span>
        </button>

        <div className="flex items-center my-4">
          <div className="flex-1 border-t border-zinc-800" />
          <span className="px-3 text-[11px] text-zinc-500 uppercase tracking-wider">or</span>
          <div className="flex-1 border-t border-zinc-800" />
        </div>

        {/* Email & Password Form */}
        <form onSubmit={handleSubmit} className="space-y-3">
          {tab === 'signup' && (
            <div>
              <label className="block text-xs text-zinc-400 mb-1">Full Name</label>
              <div className="relative">
                <UserIcon className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" />
                <input
                  type="text"
                  required
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Suresh Kumar"
                  className="w-full bg-[#1c1c1c] border border-zinc-800 rounded-xl pl-9 pr-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500 transition"
                />
              </div>
            </div>
          )}

          <div>
            <label className="block text-xs text-zinc-400 mb-1">Email Address</label>
            <div className="relative">
              <Mail className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full bg-[#1c1c1c] border border-zinc-800 rounded-xl pl-9 pr-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500 transition"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs text-zinc-400 mb-1">Password</label>
            <div className="relative">
              <Lock className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full bg-[#1c1c1c] border border-zinc-800 rounded-xl pl-9 pr-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500 transition"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full mt-2 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs transition shadow-md"
          >
            {isLoading ? 'Processing...' : tab === 'signin' ? 'Sign In' : 'Create Account'}
          </button>
        </form>

        {/* Quick Demo Pre-fill */}
        <div className="mt-5 pt-4 border-t border-zinc-800/80 text-center">
          <div className="text-[11px] text-zinc-500 mb-2">1-Click Test Credentials:</div>
          <div className="flex gap-2 justify-center">
            <button
              type="button"
              onClick={() => fillQuickDemo('test@example.com', 'password123')}
              className="px-2.5 py-1 text-[11px] rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition"
            >
              Fill test@example.com
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

