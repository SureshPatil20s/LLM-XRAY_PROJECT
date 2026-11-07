'use client';

import React, { useState } from 'react';
import { DocumentItem } from '@/types';
import {
  X,
  Upload,
  FileText,
  FileSpreadsheet,
  FileCode,
  Trash2,
  Database,
  CheckCircle,
  AlertCircle,
  Loader2,
} from 'lucide-react';

interface DocumentModalProps {
  isOpen: boolean;
  onClose: () => void;
  documents: DocumentItem[];
  onUpload: (file: File) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}

export default function DocumentModal({
  isOpen,
  onClose,
  documents,
  onUpload,
  onDelete,
}: DocumentModalProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  if (!isOpen) return null;

  const handleFileDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await processUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      await processUpload(e.target.files[0]);
    }
  };

  const processUpload = async (file: File) => {
    const validExts = ['.pdf', '.docx', '.txt', '.csv', '.json'];
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!validExts.includes(ext)) {
      setError(`Invalid format. Supported: ${validExts.join(', ')}`);
      return;
    }

    setError(null);
    setIsUploading(true);
    try {
      await onUpload(file);
    } catch (err: any) {
      setError(err.message || 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const getDocIcon = (type: string) => {
    if (type.includes('csv')) return <FileSpreadsheet className="w-5 h-5 text-emerald-400" />;
    if (type.includes('json')) return <FileCode className="w-5 h-5 text-amber-400" />;
    return <FileText className="w-5 h-5 text-blue-400" />;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-xs">
      <div className="relative w-full max-w-xl bg-[#161616] border border-zinc-800 rounded-2xl p-6 shadow-2xl animate-fade-in flex flex-col max-h-[85vh]">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute right-4 top-4 p-1 text-zinc-400 hover:text-white rounded-lg hover:bg-zinc-800 transition"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Modal Header */}
        <div className="flex items-center gap-3 mb-4">
          <div className="w-9 h-9 rounded-xl bg-emerald-950 border border-emerald-800/60 flex items-center justify-center text-emerald-400">
            <Database className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">RAG Knowledge Base</h2>
            <p className="text-xs text-zinc-400">
              Upload documents to ground your AI answers with verifiable citations
            </p>
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="mb-4 p-3 rounded-xl bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Upload Dropzone */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleFileDrop}
          className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition mb-5 ${
            dragOver
              ? 'border-emerald-500 bg-emerald-950/20'
              : 'border-zinc-800 bg-[#121212] hover:border-zinc-700'
          }`}
          onClick={() => document.getElementById('modal-file-upload')?.click()}
        >
          <input
            id="modal-file-upload"
            type="file"
            accept=".pdf,.docx,.txt,.csv,.json"
            onChange={handleFileInput}
            className="hidden"
            disabled={isUploading}
          />
          <div className="flex flex-col items-center gap-2">
            {isUploading ? (
              <Loader2 className="w-8 h-8 text-emerald-400 animate-spin" />
            ) : (
              <div className="w-10 h-10 rounded-full bg-zinc-800/80 flex items-center justify-center text-zinc-400">
                <Upload className="w-5 h-5 text-emerald-400" />
              </div>
            )}
            <div className="text-xs font-semibold text-zinc-200">
              {isUploading ? 'Chunking & indexing document...' : 'Click or drag files here to upload'}
            </div>
            <p className="text-[11px] text-zinc-500">
              Supports PDF, DOCX, TXT, CSV, JSON (up to 25 MB)
            </p>
          </div>
        </div>

        {/* Document List Header */}
        <div className="flex items-center justify-between text-xs font-semibold text-zinc-400 px-1 mb-2">
          <span>Uploaded Files ({documents.length})</span>
          <span className="text-[11px] text-zinc-500">Indexed for Semantic Search</span>
        </div>

        {/* Document List */}
        <div className="flex-1 overflow-y-auto space-y-2 pr-1 scrollbar-thin scrollbar-thumb-zinc-800">
          {documents.length === 0 ? (
            <div className="text-center py-8 text-xs text-zinc-500">
              No documents uploaded yet. Upload a PDF or file to query it via RAG.
            </div>
          ) : (
            documents.map((doc) => (
              <div
                key={doc.id}
                className="flex items-center justify-between p-3 rounded-xl bg-[#1b1b1b] border border-zinc-800 hover:border-zinc-700 transition"
              >
                <div className="flex items-center gap-3 min-w-0">
                  {getDocIcon(doc.file_type)}
                  <div className="flex flex-col min-w-0">
                    <span className="text-xs font-medium text-white truncate max-w-sm">
                      {doc.filename}
                    </span>
                    <div className="flex items-center gap-2 text-[10px] text-zinc-400 mt-0.5">
                      <span>{formatFileSize(doc.file_size)}</span>
                      <span>•</span>
                      <span className="text-emerald-400">{doc.chunk_count} chunks indexed</span>
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => onDelete(doc.id)}
                  className="p-1.5 text-zinc-500 hover:text-rose-400 hover:bg-zinc-800 rounded-lg transition"
                  title="Delete document"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

