'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import Sidebar from '@/components/Sidebar';
import Navbar from '@/components/Navbar';
import ChatArea from '@/components/ChatArea';
import ChatInput from '@/components/ChatInput';
import AuthModal from '@/components/AuthModal';
import DocumentModal from '@/components/DocumentModal';
import { useAuth } from '@/context/AuthContext';
import {
  createConversation,
  deleteConversation,
  getAvailableModels,
  getConversation,
  getConversations,
  getDocuments,
  streamMessage,
  submitMessageFeedback,
  updateConversation,
  uploadDocument,
  deleteDocument,
} from '@/lib/api';
import { Conversation, DocumentItem, Message, ModelOption, RagSource } from '@/types';

export default function Home() {
  const { user } = useAuth();

  // State
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [models, setModels] = useState<ModelOption[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('llama-3.3-70b-versatile');
  const [useRag, setUseRag] = useState<boolean>(true);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);

  // UI state
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);
  const [docModalOpen, setDocModalOpen] = useState<boolean>(false);

  // Streaming state
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [streamingContent, setStreamingContent] = useState<string>('');
  const [streamingSources, setStreamingSources] = useState<RagSource[]>([]);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Load Models
  useEffect(() => {
    getAvailableModels()
      .then((data) => {
        setModels(data);
        const def = data.find((m) => m.is_default);
        if (def) setSelectedModel(def.id);
      })
      .catch((err) => console.error('Failed to load models:', err));
  }, []);

  // Load Conversations and Documents on mount / user change
  const refreshConversations = useCallback(async () => {
    try {
      const data = await getConversations();
      setConversations(data);
      if (data.length > 0 && !activeConversationId) {
        setActiveConversationId(data[0].id);
      }
    } catch (err) {
      console.error('Failed to load conversations:', err);
    }
  }, [activeConversationId]);

  const refreshDocuments = useCallback(async () => {
    try {
      const data = await getDocuments();
      setDocuments(data);
    } catch (err) {
      console.error('Failed to load documents:', err);
    }
  }, []);

  useEffect(() => {
    refreshConversations();
    refreshDocuments();
  }, [user, refreshConversations, refreshDocuments]);

  // Load messages whenever active conversation changes
  useEffect(() => {
    if (!activeConversationId) {
      setMessages([]);
      return;
    }

    getConversation(activeConversationId)
      .then(({ messages: loadedMessages }) => {
        setMessages(loadedMessages);
      })
      .catch((err) => console.error('Failed to load messages:', err));
  }, [activeConversationId]);

  // Create New Chat
  const handleNewChat = async () => {
    try {
      const newConv = await createConversation('New Chat', selectedModel);
      setConversations((prev) => [newConv, ...prev]);
      setActiveConversationId(newConv.id);
      setMessages([]);
    } catch (err) {
      console.error('Failed to create conversation:', err);
    }
  };

  // Delete Conversation
  const handleDeleteConversation = async (id: string) => {
    try {
      await deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeConversationId === id) {
        const remaining = conversations.filter((c) => c.id !== id);
        setActiveConversationId(remaining.length > 0 ? remaining[0].id : null);
        setMessages([]);
      }
    } catch (err) {
      console.error('Failed to delete conversation:', err);
    }
  };

  // Rename Conversation
  const handleRenameConversation = async (id: string, newTitle: string) => {
    try {
      const updated = await updateConversation(id, { title: newTitle });
      setConversations((prev) => prev.map((c) => (c.id === id ? updated : c)));
    } catch (err) {
      console.error('Failed to rename conversation:', err);
    }
  };

  // Toggle Pin Conversation
  const handleTogglePin = async (id: string, isPinned: boolean) => {
    try {
      const updated = await updateConversation(id, { is_pinned: isPinned });
      setConversations((prev) => prev.map((c) => (c.id === id ? updated : c)));
    } catch (err) {
      console.error('Failed to pin conversation:', err);
    }
  };

  // Upload Document
  const handleUploadDocument = async (file: File) => {
    const uploaded = await uploadDocument(file);
    setDocuments((prev) => [uploaded, ...prev]);
  };

  // Delete Document
  const handleDeleteDocument = async (id: string) => {
    await deleteDocument(id);
    setDocuments((prev) => prev.filter((d) => d.id !== id));
  };

  // Send Message & Stream
  const handleSendMessage = async (content: string, attachedFile?: File, imageDataUrl?: string) => {
    if (!content.trim() && !attachedFile && !imageDataUrl) return;

    let convId = activeConversationId;

    // Auto-create conversation if none is active
    if (!convId) {
      try {
        const titleSnippet = content.trim().slice(0, 30) || (imageDataUrl ? 'Image OCR Analysis' : 'New Chat');
        const newConv = await createConversation(titleSnippet, selectedModel);
        setConversations((prev) => [newConv, ...prev]);
        setActiveConversationId(newConv.id);
        convId = newConv.id;
      } catch (err) {
        console.error('Failed to auto-create conversation:', err);
        return;
      }
    }

    // If file was attached (and not an image), upload to RAG document index
    const isImageFile = attachedFile && attachedFile.name.match(/\.(png|jpg|jpeg|webp)$/i);
    if (attachedFile && !isImageFile) {
      try {
        await handleUploadDocument(attachedFile);
      } catch (err: any) {
        alert(`Failed to index file: ${err.message}`);
      }
    }

    // Add user message optimistically
    const displayContent = imageDataUrl
      ? `![Attached Image](${imageDataUrl})\n\n${content || 'Please extract all text (OCR) from this image and summarize its content.'}`
      : content;

    const userMsg: Message = {
      id: `temp-${Date.now()}`,
      conversation_id: convId,
      role: 'user',
      content: displayContent,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);

    // Prepare streaming
    setIsStreaming(true);
    setStreamingContent('');
    setStreamingSources([]);

    let collectedSources: RagSource[] = [];

    await streamMessage({
      conversationId: convId,
      content,
      model: selectedModel,
      useRag,
      imageData: imageDataUrl,
      onChunk: (chunk) => {
        setStreamingContent((prev) => prev + chunk);
      },
      onSources: (sources) => {
        collectedSources = sources;
        setStreamingSources(sources);
      },
      onDone: (fullContent) => {
        const assistantMsg: Message = {
          id: `msg-${Date.now()}`,
          conversation_id: convId!,
          role: 'assistant',
          content: fullContent,
          model: selectedModel,
          sources: collectedSources,
          created_at: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, assistantMsg]);
        setIsStreaming(false);
        setStreamingContent('');
        setStreamingSources([]);
        refreshConversations();
      },
      onError: (err) => {
        console.error('Streaming error:', err);
        const errMsg: Message = {
          id: `err-${Date.now()}`,
          conversation_id: convId!,
          role: 'assistant',
          content: `⚠️ **Error receiving response:** ${err.message}`,
          model: selectedModel,
          created_at: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, errMsg]);
        setIsStreaming(false);
        setStreamingContent('');
      },
    });
  };

  // Regenerate Response
  const handleRegenerate = async () => {
    if (messages.length === 0 || isStreaming) return;
    const lastUserMsg = [...messages].reverse().find((m) => m.role === 'user');
    if (!lastUserMsg) return;

    // Pop the last assistant message
    setMessages((prev) => {
      const idx = prev.map((m) => m.role).lastIndexOf('assistant');
      if (idx !== -1) {
        return prev.slice(0, idx);
      }
      return prev;
    });

    handleSendMessage(lastUserMsg.content);
  };

  // Feedback (Thumbs Up / Down)
  const handleFeedback = async (messageId: string, rating: 'like' | 'dislike') => {
    try {
      await submitMessageFeedback(messageId, rating);
      setMessages((prev) =>
        prev.map((m) => (m.id === messageId ? { ...m, feedback: rating } : m))
      );
    } catch (err) {
      console.error('Failed to submit feedback:', err);
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#121212] text-zinc-100 font-sans">
      {/* Sidebar */}
      <Sidebar
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelectConversation={setActiveConversationId}
        onNewChat={handleNewChat}
        onDeleteConversation={handleDeleteConversation}
        onRenameConversation={handleRenameConversation}
        onTogglePin={handleTogglePin}
        onOpenDocuments={() => setDocModalOpen(true)}
        isOpen={sidebarOpen}
        onToggleOpen={() => setSidebarOpen((prev) => !prev)}
      />

      {/* Main Workspace */}
      <div className="flex-1 flex flex-col min-w-0 h-full relative">
        {/* Top Navbar */}
        <Navbar
          onToggleSidebar={() => setSidebarOpen((prev) => !prev)}
          models={models}
          selectedModel={selectedModel}
          onSelectModel={setSelectedModel}
          useRag={useRag}
          onToggleRag={() => setUseRag((prev) => !prev)}
          docCount={documents.length}
          onOpenDocuments={() => setDocModalOpen(true)}
        />

        {/* Chat Feed */}
        <ChatArea
          messages={messages}
          isStreaming={isStreaming}
          streamingContent={streamingContent}
          streamingSources={streamingSources}
          onRegenerate={handleRegenerate}
          onSelectPrompt={(prompt) => handleSendMessage(prompt)}
          onFeedback={handleFeedback}
        />

        {/* Bottom Input Dock */}
        <ChatInput
          onSendMessage={handleSendMessage}
          isStreaming={isStreaming}
          onStopStreaming={() => setIsStreaming(false)}
        />
      </div>

      {/* Modals */}
      <AuthModal />
      <DocumentModal
        isOpen={docModalOpen}
        onClose={() => setDocModalOpen(false)}
        documents={documents}
        onUpload={handleUploadDocument}
        onDelete={handleDeleteDocument}
      />
    </div>
  );
}
