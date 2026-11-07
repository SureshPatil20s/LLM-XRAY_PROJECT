import { AuthResponse, Conversation, DocumentItem, Message, ModelOption, RagSource, User } from '@/types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export function getToken(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem('suresh_gpt_token');
}

export function setToken(token: string | null) {
  if (typeof window === 'undefined') return;
  if (token) {
    localStorage.setItem('suresh_gpt_token', token);
  } else {
    localStorage.removeItem('suresh_gpt_token');
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers = new Headers(options.headers || {});
  
  if (!headers.has('Content-Type') && !(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  const res = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    if (res.status === 401) {
      setToken(null);
    }
    let errorDetail = 'An error occurred';
    try {
      const errorJson = await res.json();
      errorDetail = errorJson.detail || errorDetail;
    } catch {
      errorDetail = `Request failed: ${res.statusText}`;
    }
    throw new Error(errorDetail);
  }

  return res.json() as Promise<T>;
}

// Auth APIs
export async function registerUser(email: string, password: string, name: string): Promise<AuthResponse> {
  const res = await request<AuthResponse>('/auth/register', {
    method: 'POST',
    body: JSON.stringify({ email, password, name }),
  });
  setToken(res.access_token);
  return res;
}

export async function loginUser(email: string, password: string): Promise<AuthResponse> {
  const res = await request<AuthResponse>('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
  setToken(res.access_token);
  return res;
}

export async function loginWithGoogleApi(email: string, name: string, avatar_url?: string): Promise<AuthResponse> {
  const res = await request<AuthResponse>('/auth/google', {
    method: 'POST',
    body: JSON.stringify({ email, name, avatar_url }),
  });
  setToken(res.access_token);
  return res;
}

export async function loginAsGuest(): Promise<AuthResponse> {
  const res = await request<AuthResponse>('/auth/guest', {
    method: 'POST',
  });
  setToken(res.access_token);
  return res;
}

export async function getCurrentUser(): Promise<User> {
  return request<User>('/auth/me');
}

export async function updateCurrentUser(data: { name?: string; avatar_url?: string }): Promise<User> {
  return request<User>('/auth/me', {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

// Conversation APIs
export async function getConversations(): Promise<Conversation[]> {
  const token = getToken();
  if (!token) return [];
  try {
    return await request<Conversation[]>('/conversations');
  } catch {
    return [];
  }
}

export async function createConversation(title = 'New Chat', model = 'llama-3.3-70b-versatile'): Promise<Conversation> {
  return request<Conversation>('/conversations', {
    method: 'POST',
    body: JSON.stringify({ title, model }),
  });
}

export async function getConversation(id: string): Promise<{ conversation: Conversation; messages: Message[] }> {
  return request<{ conversation: Conversation; messages: Message[] }>(`/conversations/${id}`);
}

export async function updateConversation(id: string, updates: { title?: string; is_pinned?: boolean }): Promise<Conversation> {
  return request<Conversation>(`/conversations/${id}`, {
    method: 'PUT',
    body: JSON.stringify(updates),
  });
}

export async function deleteConversation(id: string): Promise<void> {
  await request<void>(`/conversations/${id}`, { method: 'DELETE' });
}

// Message Feedback
export async function submitMessageFeedback(messageId: string, rating: 'like' | 'dislike'): Promise<void> {
  await request<void>(`/messages/${messageId}/feedback`, {
    method: 'POST',
    body: JSON.stringify({ rating }),
  });
}

// Document / RAG APIs
export async function getDocuments(): Promise<DocumentItem[]> {
  const token = getToken();
  if (!token) return [];
  try {
    return await request<DocumentItem[]>('/documents');
  } catch {
    return [];
  }
}

export async function uploadDocument(file: File): Promise<DocumentItem> {
  const token = getToken();
  const formData = new FormData();
  formData.append('file', file);

  const headers: HeadersInit = {};
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: 'POST',
    headers,
    body: formData,
  });

  if (!res.ok) {
    const errorJson = await res.json().catch(() => ({}));
    throw new Error(errorJson.detail || 'Upload failed');
  }

  return res.json();
}

export async function deleteDocument(id: string): Promise<void> {
  await request<void>(`/documents/${id}`, { method: 'DELETE' });
}

// Models API
export async function getAvailableModels(): Promise<ModelOption[]> {
  return request<ModelOption[]>('/models');
}

// Real-Time SSE Streaming API
export async function streamMessage({
  conversationId,
  content,
  model,
  useRag,
  imageData,
  onChunk,
  onSources,
  onDone,
  onError,
}: {
  conversationId: string;
  content: string;
  model: string;
  useRag: boolean;
  imageData?: string;
  onChunk: (chunk: string) => void;
  onSources?: (sources: RagSource[]) => void;
  onDone: (fullContent: string) => void;
  onError: (err: Error) => void;
}) {
  const token = getToken();
  try {
    const res = await fetch(`${API_BASE}/messages/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({
        conversation_id: conversationId,
        content,
        model,
        use_rag: useRag,
        image_data: imageData,
      }),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(errText || `Server responded with ${res.status}`);
    }

    if (!res.body) {
      throw new Error('No readable response body');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let accumulated = '';
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data: ')) continue;
        const payload = trimmed.slice(6);
        if (payload === '[DONE]') {
          onDone(accumulated);
          return;
        }

        try {
          const parsed = JSON.parse(payload);
          if (parsed.delta) {
            accumulated += parsed.delta;
            onChunk(parsed.delta);
          }
          if (parsed.sources && onSources) {
            onSources(parsed.sources);
          }
          if (parsed.error) {
            onError(new Error(parsed.error));
            return;
          }
        } catch {
          // Plain string chunk fallback
          accumulated += payload;
          onChunk(payload);
        }
      }
    }

    onDone(accumulated);
  } catch (err) {
    onError(err instanceof Error ? err : new Error(String(err)));
  }
}

