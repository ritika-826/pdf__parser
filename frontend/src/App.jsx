import React, { useState, useEffect, useRef } from 'react';
import {
  FileText,
  UploadCloud,
  CheckCircle2,
  Circle,
  Loader2,
  AlertCircle,
  Trash2,
  Cpu,
  Database,
  RefreshCw,
  Send,
  User,
  Sparkles,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  BookOpen,
  Bot,
  X
} from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const MAX_SIZE_MB = 20;
const STAGE_LABELS = [
  'PDF uploaded',
  'Text extracted',
  'Text chunked',
  'Embeddings generated',
  'Vectors stored in ChromaDB',
  'Document ready for questions'
];

// ==========================================
// 1. API Service Layer
// ==========================================
async function apiCheckHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) throw new Error('Health check failed');
    return await res.json();
  } catch (e) {
    return { status: 'error', ollama_connected: false, active_model: 'offline' };
  }
}

async function apiUploadPdf(file) {
  const data = new FormData();
  data.append('file', file);
  const res = await fetch(`${API_BASE_URL}/upload-pdf`, { method: 'POST', body: data });
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(json.detail || 'Upload failed');
  return json;
}

async function apiAskQuestion(documentId, question, sessionId) {
  const res = await fetch(`${API_BASE_URL}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_id: documentId, question, session_id: sessionId })
  });
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(json.detail || 'Query failed');
  return json;
}

async function apiGetDocuments() {
  const res = await fetch(`${API_BASE_URL}/documents`);
  if (!res.ok) throw new Error('Failed to fetch documents');
  return await res.json();
}

async function apiDeleteDocument(docId) {
  const res = await fetch(`${API_BASE_URL}/documents/${encodeURIComponent(docId)}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Delete failed');
  return await res.json();
}

async function apiClearChat(sessionId) {
  const res = await fetch(`${API_BASE_URL}/chat/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
  return await res.json();
}

// ==========================================
// 2. Sub-Components
// ==========================================

// Source References Accordion
function SourceReferences({ sources }) {
  const [open, setOpen] = useState(false);
  const [copiedIdx, setCopiedIdx] = useState(null);

  if (!sources || sources.length === 0) return null;

  const copyText = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(null), 1500);
  };

  return (
    <div className="source-references-container">
      <button type="button" className="source-toggle-btn" onClick={() => setOpen(!open)}>
        <div className="source-toggle-left">
          <BookOpen size={15} className="source-icon" />
          <span>Retrieved Sources ({sources.length} excerpt{sources.length > 1 ? 's' : ''})</span>
        </div>
        {open ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
      </button>

      {open && (
        <div className="sources-list">
          {sources.map((src, i) => (
            <div key={i} className="source-card">
              <div className="source-card-header">
                <div className="source-badges">
                  <span className="badge badge-page">Page {src.page}</span>
                  {src.score !== null && (
                    <span className="badge badge-score">Match: {Math.round(src.score * 100)}%</span>
                  )}
                  <span className="chunk-tag">{src.chunk_id}</span>
                </div>
                <button type="button" className="icon-btn-subtle" onClick={() => copyText(src.text, i)}>
                  {copiedIdx === i ? <Check size={13} className="text-success" /> : <Copy size={13} />}
                </button>
              </div>
              <div className="source-excerpt">
                <blockquote>"{src.text}"</blockquote>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Chat Message Row
function ChatMessage({ message }) {
  const isUser = message.role === 'user';
  const [copied, setCopied] = useState(false);

  const copyAnswer = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className={`chat-message-row ${isUser ? 'user-row' : 'assistant-row'}`}>
      <div className={`avatar ${isUser ? 'user-avatar' : 'assistant-avatar'}`}>
        {isUser ? <User size={16} /> : <Sparkles size={16} />}
      </div>

      <div className="message-content-wrapper">
        <div className="message-header">
          <span className="sender-name">{isUser ? 'You' : 'PDF_Parser AI'}</span>
          <div className="message-actions">
            {!isUser && (
              <button type="button" className="icon-btn-subtle" onClick={copyAnswer}>
                {copied ? <Check size={13} className="text-success" /> : <Copy size={13} />}
              </button>
            )}
            <span className="timestamp">{message.timestamp || 'Just now'}</span>
          </div>
        </div>

        <div className={`message-bubble ${isUser ? 'user-bubble' : 'assistant-bubble'}`}>
          <p>{message.content}</p>
        </div>

        {!isUser && message.sources && message.sources.length > 0 && (
          <SourceReferences sources={message.sources} />
        )}
      </div>
    </div>
  );
}

// ==========================================
// 3. Main Application Component
// ==========================================
export default function App() {
  const [documents, setDocuments] = useState([]);
  const [activeDoc, setActiveDoc] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [stages, setStages] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  const [sessionId] = useState(() => `sess_${Math.random().toString(36).substring(2, 8)}`);

  const [health, setHealth] = useState({ ollama_connected: false, active_model: 'Checking...' });
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isAsking]);

  // Load telemetry and existing documents
  const loadInitialData = async () => {
    const h = await apiCheckHealth();
    setHealth(h);
    try {
      const docs = await apiGetDocuments();
      if (docs?.documents) {
        setDocuments(docs.documents);
        if (!activeDoc && docs.documents.length > 0) {
          setActiveDoc(docs.documents[0]);
        }
      }
    } catch (e) {
      console.warn(e);
    }
  };

  useEffect(() => {
    loadInitialData();
  }, []);

  // File Upload Handling
  const handleSelectFile = (file) => {
    setErrorMsg(null);
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setErrorMsg('Only PDF files (.pdf) are allowed.');
      return;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setErrorMsg(`File exceeds ${MAX_SIZE_MB}MB size limit.`);
      return;
    }
    setSelectedFile(file);
  };

  const handleUpload = async () => {
    if (!selectedFile || isProcessing) return;
    setIsProcessing(true);
    setErrorMsg(null);
    setStages(STAGE_LABELS.map((name, i) => ({ name, status: i === 0 ? 'processing' : 'pending' })));

    try {
      const res = await apiUploadPdf(selectedFile);
      if (res.stages) setStages(res.stages);
      if (res.document) {
        setDocuments((prev) => [res.document, ...prev.filter((d) => d.document_id !== res.document.document_id)]);
        setActiveDoc(res.document);
        setMessages([]);
      }
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (err) {
      setErrorMsg(err.message || 'Upload failed.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Delete Document
  const handleDeleteDoc = async (e, docId) => {
    e.stopPropagation();
    if (!window.confirm('Delete this document and all vectors?')) return;
    try {
      await apiDeleteDocument(docId);
      const updated = documents.filter((d) => d.document_id !== docId);
      setDocuments(updated);
      if (activeDoc?.document_id === docId) {
        setActiveDoc(updated.length > 0 ? updated[0] : null);
        setMessages([]);
      }
    } catch (err) {
      alert(`Delete error: ${err.message}`);
    }
  };

  // Question Answering
  const handleAsk = async (textToSend) => {
    const q = textToSend || question;
    if (!q.trim() || !activeDoc || isAsking) return;

    const userMessage = {
      role: 'user',
      content: q.trim(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion('');
    setIsAsking(true);

    try {
      const res = await apiAskQuestion(activeDoc.document_id, q.trim(), sessionId);
      const aiMessage = {
        role: 'assistant',
        content: res.answer,
        sources: res.sources || [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Error: ${err.message}`,
          sources: [],
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]);
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-logo-container">
          <div className="brand-icon-box">
            <FileText size={22} />
          </div>
          <div>
            <div className="brand-title-row">
              <h1 className="brand-title">PDF_Parser</h1>
              <span className="badge-rag">RAG Engine</span>
            </div>
            <p className="brand-subtitle">AI-Powered Document Intelligence</p>
          </div>
        </div>

        <div className="telemetry-bar">
          <div className="telemetry-item">
            <span className={`status-dot ${health.ollama_connected ? 'dot-online' : 'dot-offline'}`} />
            <Cpu size={14} className="telemetry-icon" />
            <span className="telemetry-label">Ollama:</span>
            <span className="telemetry-val">{health.ollama_connected ? health.active_model : 'Offline'}</span>
          </div>

          <div className="telemetry-item">
            <Database size={14} className="telemetry-icon" />
            <span className="telemetry-label">ChromaDB:</span>
            <span className="telemetry-val">all-MiniLM-L6-v2</span>
          </div>

          <button type="button" className="icon-btn-subtle" onClick={loadInitialData} title="Refresh connection">
            <RefreshCw size={14} />
          </button>
        </div>
      </header>

      {/* Main Grid */}
      <main className="app-main-content">
        {/* Sidebar: Upload & Documents */}
        <aside className="sidebar-col">
          {/* Upload Card */}
          <section className="card-section">
            <h2 className="section-heading">Upload PDF Document</h2>

            <div
              className={`dropzone ${isProcessing ? 'disabled' : ''}`}
              onClick={() => !isProcessing && fileInputRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                if (e.dataTransfer.files?.length > 0) handleSelectFile(e.dataTransfer.files[0]);
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,application/pdf"
                style={{ display: 'none' }}
                onChange={(e) => e.target.files?.length > 0 && handleSelectFile(e.target.files[0])}
              />
              <div className="dropzone-content">
                <div className="upload-icon-circle">
                  <UploadCloud size={24} className="text-primary" />
                </div>
                <div className="dropzone-text">
                  <p className="dropzone-primary-text">
                    <span className="browse-link">Click to browse</span> or drag PDF here
                  </p>
                  <p className="dropzone-subtext">Up to {MAX_SIZE_MB}MB</p>
                </div>
              </div>
            </div>

            {selectedFile && (
              <div className="selected-file-card">
                <div className="file-info-col">
                  <FileText size={20} className="text-primary flex-shrink-0" />
                  <div className="file-meta">
                    <span className="file-name">{selectedFile.name}</span>
                    <span className="file-size">{(selectedFile.size / 1024).toFixed(0)} KB</span>
                  </div>
                </div>
                <div className="file-actions-col">
                  <button type="button" className="icon-btn-danger" onClick={() => setSelectedFile(null)}>
                    <X size={15} />
                  </button>
                  <button type="button" className="btn btn-primary btn-sm" onClick={handleUpload} disabled={isProcessing}>
                    {isProcessing ? <Loader2 size={14} className="spin" /> : 'Process PDF'}
                  </button>
                </div>
              </div>
            )}

            {errorMsg && (
              <div className="upload-error-alert">
                <AlertCircle size={15} className="flex-shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}
          </section>

          {/* 6-Stage Pipeline Progress */}
          {(isProcessing || stages) && (
            <section className="card-section status-card">
              <h2 className="section-heading">Processing Stages</h2>
              <div className="stage-steps-list">
                {STAGE_LABELS.map((name, idx) => {
                  const s = stages?.find((st) => st.name === name);
                  const isDone = s?.status === 'completed';
                  const isRunning = s?.status === 'processing';
                  return (
                    <div key={idx} className={`stage-step-item ${isRunning ? 'stage-active' : ''}`}>
                      <div className="stage-icon-col">
                        {isDone ? (
                          <CheckCircle2 size={16} className="text-success" />
                        ) : isRunning ? (
                          <Loader2 size={16} className="spin text-primary" />
                        ) : (
                          <Circle size={16} className="text-muted" />
                        )}
                      </div>
                      <div className="stage-text-col">
                        <span className="stage-name">{name}</span>
                        {s?.detail && <span className="stage-detail">{s.detail}</span>}
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {/* Document Library */}
          <section className="card-section document-library-section">
            <div className="section-header-compact">
              <h2 className="section-heading">Documents ({documents.length})</h2>
            </div>
            {documents.length === 0 ? (
              <p className="empty-library-note">No documents uploaded yet.</p>
            ) : (
              <div className="doc-items-list">
                {documents.map((doc) => {
                  const active = activeDoc?.document_id === doc.document_id;
                  return (
                    <div
                      key={doc.document_id}
                      className={`doc-item-card ${active ? 'doc-item-active' : ''}`}
                      onClick={() => {
                        setActiveDoc(doc);
                        setMessages([]);
                      }}
                    >
                      <div className="doc-item-main">
                        <FileText size={17} className={active ? 'text-primary' : 'text-muted'} />
                        <div className="doc-item-details">
                          <span className="doc-filename">{doc.filename}</span>
                          <span className="doc-specs">
                            {doc.page_count}p • {doc.chunk_count} chunks
                          </span>
                        </div>
                      </div>
                      <div className="doc-item-actions">
                        {active && <span className="active-indicator-pill">Active</span>}
                        <button
                          type="button"
                          className="icon-btn-danger-subtle"
                          onClick={(e) => handleDeleteDoc(e, doc.document_id)}
                          title="Delete document"
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        </aside>

        {/* Chat Area */}
        <section className="chat-col">
          <div className="chat-window-card">
            {/* Header */}
            <div className="chat-window-header">
              <div className="active-doc-tag">
                <FileText size={16} className="text-primary flex-shrink-0" />
                <span className="active-doc-name">
                  {activeDoc ? activeDoc.filename : 'No document selected'}
                </span>
              </div>
              {messages.length > 0 && (
                <button
                  type="button"
                  className="btn btn-secondary-subtle btn-sm"
                  onClick={() => {
                    apiClearChat(sessionId);
                    setMessages([]);
                  }}
                >
                  <Trash2 size={13} /> Clear Chat
                </button>
              )}
            </div>

            {/* Scrollable Messages */}
            <div className="chat-messages-container">
              {messages.length === 0 ? (
                <div className="chat-empty-state">
                  <div className="empty-state-icon">
                    <Bot size={32} className="text-primary" />
                  </div>
                  <h3 className="empty-state-title">
                    {activeDoc ? `Ask about "${activeDoc.filename}"` : 'Welcome to PDF_Parser'}
                  </h3>
                  <p className="empty-state-sub">
                    {activeDoc
                      ? 'Ask questions grounded strictly in your uploaded PDF.'
                      : 'Upload a PDF on the left to begin.'}
                  </p>
                  {activeDoc && (
                    <div className="suggestions-list">
                      {[
                        'What is the main summary of this document?',
                        'What are the key points mentioned?',
                        'Are there any specific recommendations or numbers?'
                      ].map((s, idx) => (
                        <button
                          key={idx}
                          type="button"
                          className="suggestion-chip"
                          onClick={() => handleAsk(s)}
                          disabled={isAsking}
                        >
                          <Sparkles size={13} className="text-primary" /> {s}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="messages-stream">
                  {messages.map((m, idx) => (
                    <ChatMessage key={idx} message={m} />
                  ))}
                  {isAsking && (
                    <div className="chat-message-row assistant-row">
                      <div className="avatar assistant-avatar">
                        <Sparkles size={16} />
                      </div>
                      <div className="message-bubble assistant-bubble loading-bubble">
                        <Loader2 size={15} className="spin text-primary" />
                        <span>Searching ChromaDB & generating grounded answer...</span>
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            {/* Input Bar */}
            <form
              className="chat-input-area"
              onSubmit={(e) => {
                e.preventDefault();
                handleAsk();
              }}
            >
              <div className="input-box-wrapper">
                <textarea
                  className="chat-textarea"
                  placeholder={activeDoc ? 'Ask a question about this document...' : 'Upload a PDF to ask questions...'}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleAsk();
                    }
                  }}
                  disabled={!activeDoc || isAsking}
                  rows={2}
                />
                <button
                  type="submit"
                  className="send-button"
                  disabled={!activeDoc || !question.trim() || isAsking}
                >
                  {isAsking ? <Loader2 size={16} className="spin" /> : <Send size={16} />}
                </button>
              </div>
            </form>
          </div>
        </section>
      </main>
    </div>
  );
}
