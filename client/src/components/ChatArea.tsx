'use client';

import React, { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Message, RagSource } from '@/types';
import {
  Copy,
  Check,
  RotateCw,
  ThumbsUp,
  ThumbsDown,
  Sparkles,
  FileText,
  ChevronDown,
  ChevronUp,
  User as UserIcon,
  BookOpen,
} from 'lucide-react';

interface ChatAreaProps {
  messages: Message[];
  isStreaming: boolean;
  streamingContent: string;
  streamingSources?: RagSource[];
  onRegenerate: () => void;
  onSelectPrompt: (prompt: string) => void;
  onFeedback: (messageId: string, rating: 'like' | 'dislike') => void;
}

export default function ChatArea({
  messages,
  isStreaming,
  streamingContent,
  streamingSources,
  onRegenerate,
  onSelectPrompt,
  onFeedback,
}: ChatAreaProps) {
  const scrollEndRef = useRef<HTMLDivElement>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({});

  useEffect(() => {
    scrollEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const toggleSourceExpand = (id: string) => {
    setExpandedSources((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const suggestions = [
    { title: '/image A futuristic cyberpunk city in neon rain', subtitle: '🎨 Free high-speed AI image generation (1024×1024)' },
    { title: 'Read and extract text from an uploaded image (OCR)', subtitle: '🔍 Multimodal Llama 3.2 Vision OCR transcription' },
    { title: 'Explain RAG document retrieval', subtitle: '⚡ How verifiable citations prevent AI hallucinations' },
    { title: 'Write an async FastAPI endpoint with Pydantic', subtitle: '💻 Code generation with syntax highlighting & copy' },
  ];

  return (
    <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6 space-y-6">
      {/* Welcome Empty State */}
      {messages.length === 0 && !isStreaming && (
        <div className="max-w-2xl mx-auto h-full flex flex-col items-center justify-center text-center py-12">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center text-black font-black text-2xl shadow-xl shadow-emerald-950/40 mb-4 animate-bounce-slow">
            <Sparkles className="w-7 h-7 text-black" />
          </div>
          <h1 className="text-2xl font-bold text-white mb-2">What can I help you with today?</h1>
          <p className="text-sm text-zinc-400 mb-8 max-w-md">
            Powered by hosted Groq Llama 3.3 70B & Qwen 2.5 with document RAG knowledge base.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full">
            {suggestions.map((s, idx) => (
              <button
                key={idx}
                onClick={() => onSelectPrompt(s.title)}
                className="p-3.5 rounded-xl border border-zinc-800 bg-[#161616] hover:bg-[#202020] hover:border-zinc-700 text-left transition group"
              >
                <div className="text-xs font-semibold text-zinc-200 group-hover:text-emerald-400 transition-colors">
                  {s.title}
                </div>
                <div className="text-[11px] text-zinc-500 mt-1">{s.subtitle}</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Message Feed */}
      <div className="max-w-3xl mx-auto space-y-6">
        {messages.map((msg, index) => {
          const isUser = msg.role === 'user';
          const isLast = index === messages.length - 1;

          return (
            <div
              key={msg.id || index}
              className={`flex items-start gap-4 ${isUser ? 'justify-end' : 'justify-start'}`}
            >
              {/* Assistant Avatar */}
              {!isUser && (
                <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-500 text-black flex items-center justify-center shrink-0 shadow-md">
                  <Sparkles className="w-4 h-4 text-black" />
                </div>
              )}

              {/* Message Body */}
              <div
                className={`flex flex-col max-w-[85%] md:max-w-[78%] ${
                  isUser
                    ? 'bg-[#262626] text-white px-4 py-3 rounded-2xl rounded-tr-xs shadow-sm'
                    : 'text-zinc-100 px-1 py-1'
                }`}
              >
                {isUser ? (
                  <p className="text-sm whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                ) : (
                  <div className="space-y-3">
                    {/* Markdown Content */}
                    <div className="prose prose-invert prose-sm max-w-none text-zinc-200 leading-relaxed break-words">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          code({ node, inline, className, children, ...props }: any) {
                            const match = /language-(\w+)/.exec(className || '');
                            const codeContent = String(children).replace(/\n$/, '');
                            const codeId = `${msg.id}-${Math.random()}`;

                            if (!inline && match) {
                              return (
                                <div className="rounded-xl overflow-hidden my-3 border border-zinc-800 bg-[#0d0d0d]">
                                  <div className="flex items-center justify-between px-3 py-1.5 bg-[#171717] border-b border-zinc-800 text-[11px] font-mono text-zinc-400">
                                    <span>{match[1]}</span>
                                    <button
                                      onClick={() => handleCopy(codeId, codeContent)}
                                      className="flex items-center gap-1 hover:text-white transition"
                                    >
                                      {copiedId === codeId ? (
                                        <>
                                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                                          <span className="text-emerald-400">Copied!</span>
                                        </>
                                      ) : (
                                        <>
                                          <Copy className="w-3.5 h-3.5" />
                                          <span>Copy code</span>
                                        </>
                                      )}
                                    </button>
                                  </div>
                                  <pre className="p-3 text-xs font-mono overflow-x-auto text-zinc-200 bg-[#0a0a0a]">
                                    <code>{children}</code>
                                  </pre>
                                </div>
                              );
                            }
                            return (
                              <code
                                className="px-1.5 py-0.5 rounded bg-zinc-800 text-emerald-300 font-mono text-xs"
                                {...props}
                              >
                                {children}
                              </code>
                            );
                          },
                        }}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    </div>

                    {/* RAG Source Citations */}
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-zinc-800/80">
                        <button
                          onClick={() => toggleSourceExpand(msg.id)}
                          className="flex items-center gap-1.5 text-xs font-semibold text-emerald-400 hover:text-emerald-300 transition"
                        >
                          <BookOpen className="w-3.5 h-3.5" />
                          <span>
                            {msg.sources.length} Referenced Source{msg.sources.length > 1 ? 's' : ''} (RAG)
                          </span>
                          {expandedSources[msg.id] ? (
                            <ChevronUp className="w-3.5 h-3.5" />
                          ) : (
                            <ChevronDown className="w-3.5 h-3.5" />
                          )}
                        </button>

                        {expandedSources[msg.id] && (
                          <div className="mt-2 space-y-2">
                            {msg.sources.map((src, sIdx) => (
                              <div
                                key={sIdx}
                                className="p-2.5 rounded-lg bg-[#141414] border border-zinc-800 text-xs text-zinc-300"
                              >
                                <div className="flex items-center justify-between font-medium text-emerald-300 mb-1">
                                  <div className="flex items-center gap-1.5 truncate">
                                    <FileText className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                                    <span className="truncate">{src.filename}</span>
                                    {src.page_number && (
                                      <span className="text-[10px] bg-zinc-800 px-1.5 py-0.2 rounded text-zinc-300">
                                        Page {src.page_number}
                                      </span>
                                    )}
                                  </div>
                                  <span className="text-[10px] text-zinc-500 font-mono">
                                    Score: {Math.round(src.score * 100)}%
                                  </span>
                                </div>
                                <p className="text-[11px] text-zinc-400 italic line-clamp-2">
                                  &ldquo;{src.snippet}&rdquo;
                                </p>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Assistant Action Bar */}
                    <div className="flex items-center gap-1 pt-1 text-zinc-500">
                      <button
                        onClick={() => handleCopy(msg.id, msg.content)}
                        className="p-1.5 rounded-md hover:bg-zinc-800 hover:text-zinc-200 transition"
                        title="Copy message"
                      >
                        {copiedId === msg.id ? (
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                      </button>

                      {isLast && (
                        <button
                          onClick={onRegenerate}
                          className="p-1.5 rounded-md hover:bg-zinc-800 hover:text-zinc-200 transition"
                          title="Regenerate response"
                        >
                          <RotateCw className="w-3.5 h-3.5" />
                        </button>
                      )}

                      <button
                        onClick={() => onFeedback(msg.id, 'like')}
                        className={`p-1.5 rounded-md hover:bg-zinc-800 transition ${
                          msg.feedback === 'like' ? 'text-emerald-400 bg-emerald-950/40' : 'hover:text-zinc-200'
                        }`}
                        title="Good response"
                      >
                        <ThumbsUp className="w-3.5 h-3.5" />
                      </button>

                      <button
                        onClick={() => onFeedback(msg.id, 'dislike')}
                        className={`p-1.5 rounded-md hover:bg-zinc-800 transition ${
                          msg.feedback === 'dislike' ? 'text-rose-400 bg-rose-950/40' : 'hover:text-zinc-200'
                        }`}
                        title="Poor response"
                      >
                        <ThumbsDown className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* User Avatar */}
              {isUser && (
                <div className="w-8 h-8 rounded-xl bg-zinc-800 text-zinc-300 flex items-center justify-center shrink-0 border border-zinc-700">
                  <UserIcon className="w-4 h-4" />
                </div>
              )}
            </div>
          );
        })}

        {/* Live Streaming Assistant Message */}
        {isStreaming && (
          <div className="flex items-start gap-4 justify-start">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-500 text-black flex items-center justify-center shrink-0 shadow-md">
              <Sparkles className="w-4 h-4 text-black animate-spin-slow" />
            </div>
            <div className="flex flex-col max-w-[85%] md:max-w-[78%] text-zinc-100 px-1 py-1 space-y-3">
              <div className="prose prose-invert prose-sm max-w-none text-zinc-200 leading-relaxed">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {streamingContent}
                </ReactMarkdown>
                <span className="inline-block w-2 h-4 bg-emerald-400 animate-pulse ml-1 translate-y-0.5" />
              </div>

              {/* Streaming Sources if populated early */}
              {streamingSources && streamingSources.length > 0 && (
                <div className="mt-2 text-xs text-emerald-400 flex items-center gap-1.5">
                  <BookOpen className="w-3.5 h-3.5" />
                  <span>Found {streamingSources.length} relevant document excerpts</span>
                </div>
              )}
            </div>
          </div>
        )}

        <div ref={scrollEndRef} />
      </div>
    </div>
  );
}

