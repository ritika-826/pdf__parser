const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return await res.json();
  } catch (error) {
    return {
      status: 'error',
      ollama_connected: false,
      available_models: [],
      active_model: 'unknown',
      embedding_model: 'unknown',
      error: error.message
    };
  }
}

export async function uploadPdf(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE_URL}/upload-pdf`, {
    method: 'POST',
    body: formData,
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Upload failed with status ${res.status}`);
  }
  return data;
}

export async function askQuestion(documentId, question, sessionId = 'default', topK = null) {
  const payload = {
    document_id: documentId,
    question: question.trim(),
    session_id: sessionId,
  };
  if (topK) payload.top_k = Number(topK);

  const res = await fetch(`${API_BASE_URL}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Question answering failed with status ${res.status}`);
  }
  return data;
}

export async function getDocuments() {
  const res = await fetch(`${API_BASE_URL}/documents`);
  if (!res.ok) throw new Error(`Failed to fetch documents: ${res.statusText}`);
  return await res.json();
}

export async function deleteDocument(documentId) {
  const res = await fetch(`${API_BASE_URL}/documents/${encodeURIComponent(documentId)}`, {
    method: 'DELETE',
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Failed to delete document: ${res.statusText}`);
  return data;
}

export async function clearChat(sessionId = 'default') {
  const res = await fetch(`${API_BASE_URL}/chat/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Failed to clear chat: ${res.statusText}`);
  return data;
}
