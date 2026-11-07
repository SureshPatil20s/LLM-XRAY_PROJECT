'use client';

import React, { useState } from 'react';
import { Conversation } from '@/types';
import { useAuth } from '@/context/AuthContext';
import {
  MessageSquare,
  Plus,
  Pin,
  Trash2,
  Edit2,
  Check,
  X,
  Search,
  LogOut,
  LogIn,
  Settings,
  Sparkles,
  ChevronLeft,
  ChevronRight,
  Database,
} from 'lucide-react';

interface SidebarProps {
  conversations: Conversation[];
  activeConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
  onDeleteConversation: (id: string) => void;
  onRenameConversation: (id: string, newTitle: string) => void;
  onTogglePin: (id: string, isPinned: boolean) => void;
  onOpenDocuments: () => void;
  isOpen: boolean;
  onToggleOpen: () => void;
}

export default function Sidebar({
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewChat,
  onDeleteConversation,
  onRenameConversation,
  onTogglePin,
  onOpenDocuments,
  isOpen,
  onToggleOpen,
}: SidebarProps) {
  const { user, logout, setAuthModalOpen } = useAuth();
  const [searchQuery, setSearchQuery] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [showSettingsModal, setShowSettingsModal] = useState(false);

  const filtered = conversations.filter((c) =>
    c.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const pinnedConversations = filtered.filter((c) => c.is_pinned);
  const unpinnedConversations = filtered.filter((c) => !c.is_pinned);

  const handleStartRename = (c: Conversation, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(c.id);
    setEditTitle(c.title);
  };

  const handleSaveRename = (c: Conversation, e: React.MouseEvent) => {
    e.stopPropagation();
    if (editTitle.trim()) {
      onRenameConversation(c.id, editTitle.trim());
    }
    setEditingId(null);
  };

  const handleCancelRename = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(null);
  };

  const renderConversationItem = (c: Conversation) => {
    const isActive = c.id === activeConversationId;
    const isEditing = c.id === editingId;

    return (
      <div
        key={c.id}
        onClick={() => onSelectConversation(c.id)}
        className={`group relative flex items-center gap-2 px-3 py-2.5 rounded-xl cursor-pointer text-sm transition-all duration-150 ${
          isActive
            ? 'bg-[#212121] text-white font-medium'
            : 'text-zinc-400 hover:bg-[#1a1a1a] hover:text-zinc-200'
        }`}
      >
        <MessageSquare className="w-4 h-4 shrink-0 text-zinc-500 group-hover:text-zinc-300" />
        
        {isEditing ? (
          <div className="flex items-center gap-1 flex-1 min-w-0" onClick={(e) => e.stopPropagation()}>
            <input
              type="text"
              value={editTitle}
              onChange={(e) => setEditTitle(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSaveRename(c, e as any);
                if (e.key === 'Escape') setEditingId(null);
              }}
              autoFocus
              className="w-full bg-[#111] border border-zinc-700 rounded px-2 py-0.5 text-xs text-white focus:outline-none focus:border-emerald-500"
            />
            <button
              onClick={(e) => handleSaveRename(c, e)}
              className="p-1 hover:text-emerald-400 text-zinc-400"
              title="Save"
            >
              <Check className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={handleCancelRename}
              className="p-1 hover:text-rose-400 text-zinc-400"
              title="Cancel"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        ) : (
          <>
            <span className="truncate flex-1">{c.title}</span>
            <div className="hidden group-hover:flex items-center gap-1">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onTogglePin(c.id, !c.is_pinned);
                }}
                className={`p-1 rounded hover:bg-zinc-800 ${
                  c.is_pinned ? 'text-amber-400' : 'text-zinc-500 hover:text-zinc-300'
                }`}
                title={c.is_pinned ? 'Unpin chat' : 'Pin chat'}
              >
                <Pin className="w-3.5 h-3.5 fill-current" />
              </button>
              <button
                onClick={(e) => handleStartRename(c, e)}
                className="p-1 rounded hover:bg-zinc-800 text-zinc-500 hover:text-zinc-300"
                title="Rename chat"
              >
                <Edit2 className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  if (confirm('Delete this conversation?')) {
                    onDeleteConversation(c.id);
                  }
                }}
                className="p-1 rounded hover:bg-zinc-800 text-zinc-500 hover:text-rose-400"
                title="Delete chat"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          </>
        )}
      </div>
    );
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          onClick={onToggleOpen}
          className="md:hidden fixed inset-0 bg-black/60 z-30 backdrop-blur-xs"
        />
      )}

      {/* Sidebar Panel */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-40 flex flex-col w-64 bg-[#0e0e0e] border-r border-[#212121] transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        }`}
      >
        {/* App Title & New Chat */}
        <div className="p-3 border-b border-[#212121] flex flex-col gap-2">
          <div className="flex items-center justify-between px-1">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg bg-linear-to-tr from-emerald-500 to-teal-400 flex items-center justify-center text-black font-black text-sm shadow-md">
                S
              </div>
              <span className="font-semibold text-white tracking-wide text-sm flex items-center gap-1.5">
                SureshGPT
                <span className="text-[10px] uppercase font-bold px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800/60">
                  Pro
                </span>
              </span>
            </div>
            <button
              onClick={onToggleOpen}
              className="md:hidden p-1.5 text-zinc-400 hover:text-white rounded-lg hover:bg-zinc-800"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
          </div>

          <button
            onClick={onNewChat}
            className="w-full flex items-center justify-between px-3 py-2 rounded-xl bg-zinc-800 hover:bg-zinc-700/80 text-white font-medium text-sm transition-all shadow-xs group"
          >
            <div className="flex items-center gap-2">
              <Plus className="w-4 h-4 text-emerald-400 group-hover:scale-110 transition-transform" />
              <span>New chat</span>
            </div>
            <kbd className="hidden group-hover:inline-block text-[10px] text-zinc-400 bg-zinc-900 px-1.5 py-0.5 rounded border border-zinc-700">
              Ctrl+K
            </kbd>
          </button>

          {/* Search bar */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" />
            <input
              type="text"
              placeholder="Search chats..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-[#181818] border border-zinc-800 text-xs text-white placeholder-zinc-500 pl-8 pr-3 py-1.5 rounded-lg focus:outline-none focus:border-zinc-600 transition"
            />
          </div>
        </div>

        {/* Knowledge Base Shortcut */}
        <div className="px-3 py-2 border-b border-[#212121]">
          <button
            onClick={onOpenDocuments}
            className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs font-medium text-emerald-300 bg-emerald-950/30 border border-emerald-800/40 hover:bg-emerald-900/40 hover:border-emerald-700 transition"
          >
            <Database className="w-4 h-4 text-emerald-400 shrink-0" />
            <div className="flex flex-col text-left">
              <span>Knowledge Base (RAG)</span>
              <span className="text-[10px] text-emerald-500/70 font-normal">PDF, DOCX, TXT, CSV, JSON</span>
            </div>
          </button>
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto px-2 py-2 space-y-4 scrollbar-thin scrollbar-thumb-zinc-800">
          {pinnedConversations.length > 0 && (
            <div>
              <div className="px-3 pb-1 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider flex items-center gap-1.5">
                <Pin className="w-3 h-3 text-amber-500" />
                <span>Pinned</span>
              </div>
              <div className="space-y-0.5">
                {pinnedConversations.map(renderConversationItem)}
              </div>
            </div>
          )}

          <div>
            <div className="px-3 pb-1 text-[11px] font-semibold text-zinc-500 uppercase tracking-wider">
              Recent Chats
            </div>
            {unpinnedConversations.length === 0 && pinnedConversations.length === 0 ? (
              <div className="px-3 py-6 text-center text-xs text-zinc-500">
                {user?.provider === 'guest' ? (
                  <span>
                    Guest session is ephemeral.<br />
                    <span className="text-zinc-400">Chats are not saved to database.</span>
                  </span>
                ) : (
                  'No chats yet. Start a conversation!'
                )}
              </div>
            ) : (
              <div className="space-y-0.5">
                {unpinnedConversations.map(renderConversationItem)}
              </div>
            )}
          </div>
        </div>

        {/* User Card / Auth Footer */}
        <div className="p-3 border-t border-[#212121] bg-[#0c0c0c]">
          {user && user.provider !== 'guest' ? (
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5 min-w-0">
                {user.avatar_url ? (
                  <img
                    src={user.avatar_url}
                    alt={user.name}
                    className="w-8 h-8 rounded-full border border-zinc-700 shrink-0 object-cover"
                  />
                ) : (
                  <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-purple-500 to-indigo-500 text-white flex items-center justify-center font-bold text-xs shrink-0">
                    {user.name.charAt(0).toUpperCase()}
                  </div>
                )}
                <div className="flex flex-col min-w-0">
                  <span className="text-xs font-medium text-white truncate">{user.name}</span>
                  <span className="text-[10px] text-zinc-500 truncate">{user.email}</span>
                </div>
              </div>
              <div className="flex items-center gap-1">
                <button
                  onClick={logout}
                  className="p-1.5 text-zinc-400 hover:text-rose-400 hover:bg-zinc-800 rounded-lg transition"
                  title="Sign out"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            </div>
          ) : user && user.provider === 'guest' ? (
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-7 h-7 rounded-full bg-zinc-800 border border-zinc-700 flex items-center justify-center text-xs text-zinc-300 font-bold">
                    G
                  </div>
                  <div className="flex flex-col">
                    <span className="text-xs font-medium text-zinc-300">Guest Mode</span>
                    <span className="text-[10px] text-zinc-500">Unsaved Session</span>
                  </div>
                </div>
              </div>
              <button
                onClick={() => setAuthModalOpen(true)}
                className="w-full flex items-center justify-center gap-2 px-3 py-1.5 rounded-xl bg-emerald-600/90 hover:bg-emerald-500 text-white font-medium text-xs transition"
              >
                <LogIn className="w-3.5 h-3.5" />
                <span>Sign In to Save History</span>
              </button>
            </div>
          ) : (
            <button
              onClick={() => setAuthModalOpen(true)}
              className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs transition"
            >
              <LogIn className="w-4 h-4" />
              <span>Sign in / Sign up</span>
            </button>
          )}
        </div>
      </aside>
    </>
  );
}

