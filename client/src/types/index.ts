export interface User {
  id: string;
  email: string;
  name: string;
  avatar_url?: string;
  provider: string;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface RagSource {
  document_id: string;
  filename: string;
  chunk_index: number;
  page_number?: number | null;
  snippet: string;
  score: number;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  model?: string;
  sources?: RagSource[];
  feedback?: 'like' | 'dislike' | null;
  created_at: string;
}

export interface Conversation {
  id: string;
  user_id: string;
  title: string;
  model: string;
  is_pinned: boolean;
  created_at: string;
  updated_at: string;
}

export interface DocumentItem {
  id: string;
  user_id: string;
  filename: string;
  file_type: string;
  file_size: number;
  chunk_count: number;
  created_at: string;
}

export interface ModelOption {
  id: string;
  name: string;
  provider: string;
  description: string;
  is_default?: boolean;
}

