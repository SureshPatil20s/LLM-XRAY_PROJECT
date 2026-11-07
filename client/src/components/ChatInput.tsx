'use client';

import React, { useRef, useState, useEffect } from 'react';
import {
  ArrowUp,
  Paperclip,
  Square,
  X,
  FileText,
  FileSpreadsheet,
  FileCode,
  Image as ImageIcon,
  Loader2,
} from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (content: string, attachedFile?: File, imageDataUrl?: string) => void;
  isStreaming: boolean;
  onStopStreaming: () => void;
  disabled?: boolean;
}

export default function ChatInput({
  onSendMessage,
  isStreaming,
  onStopStreaming,
  disabled = false,
}: ChatInputProps) {
  const [input, setInput] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    if ((!input.trim() && !selectedFile && !imageDataUrl) || isStreaming || disabled) return;
    onSendMessage(input.trim(), selectedFile || undefined, imageDataUrl || undefined);
    setInput('');
    setSelectedFile(null);
    setImageDataUrl(null);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const validExtensions = ['.pdf', '.docx', '.txt', '.csv', '.json', '.png', '.jpg', '.jpeg', '.webp'];
      const fileExt = '.' + file.name.split('.').pop()?.toLowerCase();
      if (!validExtensions.includes(fileExt)) {
        alert('Supported formats: PDF, DOCX, TXT, CSV, JSON, PNG, JPG, WEBP');
        return;
      }

      setSelectedFile(file);

      // If image, convert to data URL for OCR / Vision processing
      if (['.png', '.jpg', '.jpeg', '.webp'].includes(fileExt)) {
        const reader = new FileReader();
        reader.onload = () => {
          setImageDataUrl(reader.result as string);
        };
        reader.readAsDataURL(file);
      } else {
        setImageDataUrl(null);
      }
    }
  };

  const getFileIcon = (fileName: string) => {
    if (fileName.endsWith('.csv')) return <FileSpreadsheet className="w-4 h-4 text-emerald-400" />;
    if (fileName.endsWith('.json')) return <FileCode className="w-4 h-4 text-amber-400" />;
    if (fileName.match(/\.(png|jpg|jpeg|webp)$/i)) return <ImageIcon className="w-4 h-4 text-purple-400" />;
    return <FileText className="w-4 h-4 text-blue-400" />;
  };

  return (
    <div className="p-4 bg-gradient-to-t from-[#121212] via-[#121212] to-transparent shrink-0">
      <div className="max-w-3xl mx-auto">
        {/* Attached file chip preview */}
        {selectedFile && (
          <div className="mb-2 inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-[#212121] border border-zinc-700 text-xs text-zinc-200">
            {getFileIcon(selectedFile.name)}
            <span className="font-medium truncate max-w-xs">{selectedFile.name}</span>
            <span className="text-[10px] text-zinc-400">
              ({(selectedFile.size / 1024).toFixed(1)} KB)
            </span>
            <button
              onClick={() => setSelectedFile(null)}
              className="p-0.5 hover:text-rose-400 text-zinc-400 rounded-full"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* Input Pill Box */}
        <div className="relative flex items-end gap-2 bg-[#212121] border border-zinc-800 rounded-2xl p-2 focus-within:border-zinc-600 transition-colors shadow-lg">
          {/* File Upload Trigger */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.txt,.csv,.json,.png,.jpg,.jpeg,.webp"
            onChange={handleFileChange}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="p-2 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 rounded-xl transition shrink-0"
            title="Attach file or image for OCR (PDF, DOCX, TXT, CSV, JSON, PNG, JPG)"
          >
            <Paperclip className="w-4 h-4" />
          </button>

          {/* Text Area */}
          <textarea
            ref={textareaRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message SureshGPT... (type /image to generate, or attach an image for OCR)"
            disabled={disabled}
            className="flex-1 bg-transparent border-0 text-sm text-white placeholder-zinc-500 focus:outline-none resize-none py-1.5 max-h-48 scrollbar-thin scrollbar-thumb-zinc-700"
          />

          {/* Send or Stop Button */}
          {isStreaming ? (
            <button
              type="button"
              onClick={onStopStreaming}
              className="p-2 rounded-xl bg-zinc-200 hover:bg-white text-black transition shrink-0 shadow-md"
              title="Stop generating"
            >
              <Square className="w-4 h-4 fill-current" />
            </button>
          ) : (
            <button
              type="button"
              onClick={handleSend}
              disabled={(!input.trim() && !selectedFile) || disabled}
              className={`p-2 rounded-xl transition shrink-0 shadow-md ${
                (input.trim() || selectedFile) && !disabled
                  ? 'bg-emerald-500 hover:bg-emerald-400 text-black cursor-pointer'
                  : 'bg-zinc-800 text-zinc-600 cursor-not-allowed'
              }`}
              title="Send message"
            >
              <ArrowUp className="w-4 h-4 stroke-[2.5]" />
            </button>
          )}
        </div>

        {/* Bottom Disclaimer */}
        <p className="text-[11px] text-center text-zinc-500 mt-2">
          SureshGPT can make mistakes. Verify critical facts and information. Hosted on Groq Cloud.
        </p>
      </div>
    </div>
  );
}

