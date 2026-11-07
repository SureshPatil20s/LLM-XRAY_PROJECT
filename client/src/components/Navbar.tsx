'use client';

import React from 'react';
import { ModelOption } from '@/types';
import {
  Menu,
  ChevronDown,
  Cpu,
  FileText,
  Sparkles,
  ToggleLeft,
  ToggleRight,
  Database,
  ExternalLink,
} from 'lucide-react';

interface NavbarProps {
  onToggleSidebar: () => void;
  models: ModelOption[];
  selectedModel: string;
  onSelectModel: (modelId: string) => void;
  useRag: boolean;
  onToggleRag: () => void;
  docCount: number;
  onOpenDocuments: () => void;
}

export default function Navbar({
  onToggleSidebar,
  models,
  selectedModel,
  onSelectModel,
  useRag,
  onToggleRag,
  docCount,
  onOpenDocuments,
}: NavbarProps) {
  const currentModel = models.find((m) => m.id === selectedModel) || {
    id: selectedModel,
    name: selectedModel.replace(/-/g, ' ').toUpperCase(),
    provider: 'Groq',
  };

  return (
    <header className="h-14 border-b border-[#212121] bg-[#121212] px-4 flex items-center justify-between z-20 shrink-0">
      {/* Left: Sidebar Hamburger + Model Selector */}
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          className="p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition"
          aria-label="Toggle navigation"
        >
          <Menu className="w-5 h-5" />
        </button>

        {/* Model Dropdown Selector */}
        <div className="relative group">
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-[#1c1c1c] border border-zinc-800 hover:border-zinc-700 cursor-pointer text-xs text-white transition">
            <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
            <span className="font-semibold">{currentModel.name}</span>
            <span className="text-[10px] text-zinc-400 font-mono">({currentModel.provider})</span>
            <ChevronDown className="w-3.5 h-3.5 text-zinc-400 ml-1" />
          </div>

          {/* Model dropdown menu */}
          <div className="absolute left-0 mt-1 w-64 bg-[#1a1a1a] border border-zinc-800 rounded-xl shadow-2xl py-1 hidden group-hover:block z-50">
            <div className="px-3 py-1.5 text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">
              Hosted Inference (Groq)
            </div>
            {models.map((model) => (
              <button
                key={model.id}
                onClick={() => onSelectModel(model.id)}
                className={`w-full text-left px-3 py-2 text-xs flex flex-col gap-0.5 hover:bg-zinc-800/80 transition ${
                  selectedModel === model.id ? 'bg-zinc-800 text-white font-medium' : 'text-zinc-300'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-white">{model.name}</span>
                  <span className="text-[10px] px-1.5 py-0.2 rounded bg-zinc-700/60 text-zinc-300 font-mono">
                    {model.provider}
                  </span>
                </div>
                <span className="text-[10px] text-zinc-400 leading-tight">{model.description}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Right: RAG toggle & Documents Modal Trigger */}
      <div className="flex items-center gap-2">
        {/* RAG Toggle */}
        <button
          onClick={onToggleRag}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium border transition-all ${
            useRag
              ? 'bg-emerald-950/60 border-emerald-600/70 text-emerald-300 shadow-sm shadow-emerald-950'
              : 'bg-[#181818] border-zinc-800 text-zinc-400 hover:text-zinc-200'
          }`}
          title="Enable or disable document context retrieval"
        >
          <Database className={`w-3.5 h-3.5 ${useRag ? 'text-emerald-400' : 'text-zinc-500'}`} />
          <span>RAG</span>
          {useRag ? (
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse ml-0.5" />
          ) : (
            <span className="w-2 h-2 rounded-full bg-zinc-600 ml-0.5" />
          )}
        </button>

        {/* Knowledge Base Modal Button */}
        <button
          onClick={onOpenDocuments}
          className="relative flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs bg-[#181818] border border-zinc-800 text-zinc-300 hover:bg-zinc-800 hover:text-white transition"
          title="Manage uploaded documents"
        >
          <FileText className="w-3.5 h-3.5 text-zinc-400" />
          <span>Docs</span>
          {docCount > 0 && (
            <span className="px-1.5 py-0.2 rounded-full text-[10px] font-bold bg-emerald-500 text-black">
              {docCount}
            </span>
          )}
        </button>
      </div>
    </header>
  );
}

