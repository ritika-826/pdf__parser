import React, { useState, useEffect } from 'react';
import {
  FileText,
  Layers,
  Database,
  Cpu,
  RefreshCw,
  Trash2,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  ExternalLink
} from 'lucide-react';
import FileUpload from './components/FileUpload';
import DocumentStatus from './components/DocumentStatus';
import ChatWindow from './components/ChatWindow';
import {
  checkHealth,
  uploadPdf,
  askQuestion,
  getDocuments,
  deleteDocument,
  clearChat
} from './api';

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [activeDocument, setActiveDocument] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadStages, setUploadStages] = useState(null);
  const [uploadError, setUploadError] = useState(null);

  const [messages, setMessages] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [sessionId] = useState(() => `session_${Math.random().toString(36).substring(2, 9)}`);

  const [health, setHealth] = useState({
    status: 'checking',
    ollama_connected: false,
    available_models: [],
    active_model: 'llama3:latest',
    embedding_model: 'sentence-transformers/all-MiniLM-L6-v2'
  });

  // Fetch initial documents and system health
  const refreshStatusAndDocs = async () => {
    try {
      const healthData = await checkHealth();
      setHealth(healthData);

      const docsData = await getDocuments();
      if (docsData && docsData.documents) {
        setDocuments(docsData.documents);
        // If no active doc and docs exist, pick the first one
        if (!activeDocument && docsData.documents.length > 0) {
          setActiveDocument(docsData.documents[0]);
        }
      }
    } catch (err) {
      console.error('Initial load error:', err);
    }
  };

  useEffect(() => {
    refreshStatusAndDocs();
  }, []);

  // Handle PDF upload flow
  const handleStartUpload = () => {
    setIsProcessing(true);
    setUploadError(null);
    setUploadStages([
      { name: 'PDF uploaded', status: 'processing', detail: 'Reading file...' },
      { name: 'Text extracted', status: 'pending' },
      { name: 'Text chunked', status: 'pending' },
      { name: 'Embeddings generated', status: 'pending' },
      { name: 'Vectors stored in ChromaDB', status: 'pending' },
      { name: 'Document ready for questions', status: 'pending' }
    ]);
  };

  const handleUploadSuccess = async (file) => {
    try {
      const res = await uploadPdf(file);
      if (res.stages) {
        setUploadStages(res.stages);
      }
      if (res.document) {
        setDocuments((prev) => [res.document, ...prev.filter(d => d.document_id !== res.document.document_id)]);
        setActiveDocument(res.document);
        // Clear previous conversation for the newly active document
        setMessages([]);
      }
      setIsProcessing(false);
    } catch (err) {
      setIsProcessing(false);
      setUploadError(err.message || 'An error occurred during upload.');
      throw err;
    }
  };

  // Switch active document
  const handleSelectDocument = (doc) => {
    if (activeDocument?.document_id === doc.document_id) return;
    setActiveDocument(doc);
    setMessages([]);
    setUploadStages(null);
    setUploadError(null);
  };

  // Delete document
  const handleDeleteDocument = async (e, docId) => {
    e.stopPropagation();
    if (!window.confirm('Are you sure you want to delete this document and all its ChromaDB vectors?')) {
      return;
    }
    try {
      await deleteDocument(docId);
      const updated = documents.filter((d) => d.document_id !== docId);
      setDocuments(updated);
      if (activeDocument?.document_id === docId) {
        setActiveDocument(updated.length > 0 ? updated[0] : null);
        setMessages([]);
      }
    } catch (err) {
      alert(`Failed to delete document: ${err.message}`);
    }
  };

  // Send question in chat
  const handleSendMessage = async (questionText) => {
    if (!activeDocument) return;

    const userMsg = {
      role: 'user',
      content: questionText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsGenerating(true);

    try {
      const res = await askQuestion(activeDocument.document_id, questionText, sessionId);
      const aiMsg = {
        role: 'assistant',
        content: res.answer,
        sources: res.sources || [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages((prev) => [...prev, aiMsg]);
    } catch (err) {
      const errorAiMsg = {
        role: 'assistant',
        content: `Error: ${err.message}`,
        sources: [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages((prev) => [...prev, errorAiMsg]);
    } finally {
      setIsGenerating(false);
    }
  };

  // Clear chat
  const handleClearChat = async () => {
    try {
      await clearChat(sessionId);
    } catch (err) {
      console.warn('Backend clear chat warning:', err);
    }
    setMessages([]);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <div className="brand-logo-container">
            <div className="brand-icon-box">
              <FileText className="brand-icon" size={24} />
            </div>
            <div>
              <div className="brand-title-row">
                <h1 className="brand-title">PDF_Parser</h1>
                <span className="badge badge-rag">RAG Engine</span>
              </div>
              <p className="brand-subtitle">AI-Powered Document Intelligence</p>
            </div>
          </div>
        </div>

        {/* System telemetry & model status */}
        <div className="header-right">
          <div className="telemetry-bar">
            {/* Ollama status */}
            <div className="telemetry-item" title="Local LLM Runtime status">
              <span className={`status-dot ${health.ollama_connected ? 'dot-online' : 'dot-offline'}`} />
              <Cpu size={14} className="telemetry-icon" />
              <span className="telemetry-label">Ollama:</span>
              <span className="telemetry-val">
                {health.ollama_connected ? health.active_model : 'Offline (localhost:11434)'}
              </span>
            </div>

            {/* ChromaDB / Embedding info */}
            <div className="telemetry-item" title="Vector Database and Embedding Model">
              <Database size={14} className="telemetry-icon" />
              <span className="telemetry-label">ChromaDB:</span>
              <span className="telemetry-val">all-MiniLM-L6-v2</span>
            </div>

            <button
              type="button"
              className="icon-btn-subtle"
              title="Refresh connection status"
              onClick={refreshStatusAndDocs}
            >
              <RefreshCw size={15} />
            </button>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="app-main-content">
        {/* Left Sidebar: Ingestion & Document Library */}
        <aside className="sidebar-col">
          {/* File Upload Section */}
          <section className="card-section">
            <div className="section-header-compact">
              <h2 className="section-heading">Upload PDF Document</h2>
            </div>
            <FileUpload
              onUploadSuccess={handleUploadSuccess}
              isProcessing={isProcessing}
              onStartUpload={handleStartUpload}
            />
          </section>

          {/* Document Processing Status */}
          {(isProcessing || uploadStages || uploadError) && (
            <section className="card-section">
              <DocumentStatus
                stages={uploadStages}
                isProcessing={isProcessing}
                error={uploadError}
              />
            </section>
          )}

          {/* Document Library */}
          <section className="card-section document-library-section">
            <div className="section-header-compact">
              <div className="heading-with-badge">
                <h2 className="section-heading">Document Library</h2>
                <span className="counter-pill">{documents.length}</span>
              </div>
            </div>

            {documents.length === 0 ? (
              <div className="empty-library-note">
                <p>No documents uploaded yet. Upload a PDF above to store chunks and start querying.</p>
              </div>
            ) : (
              <div className="doc-items-list">
                {documents.map((doc) => {
                  const isSelected = activeDocument?.document_id === doc.document_id;
                  return (
                    <div
                      key={doc.document_id}
                      className={`doc-item-card ${isSelected ? 'doc-item-active' : ''}`}
                      onClick={() => handleSelectDocument(doc)}
                      role="button"
                      tabIndex={0}
                    >
                      <div className="doc-item-main">
                        <FileText size={18} className={isSelected ? 'text-primary' : 'text-muted'} />
                        <div className="doc-item-details">
                          <span className="doc-filename" title={doc.filename}>{doc.filename}</span>
                          <span className="doc-specs">
                            {doc.page_count} page{doc.page_count > 1 ? 's' : ''} • {doc.chunk_count} chunks • {(doc.file_size_bytes / 1024).toFixed(0)} KB
                          </span>
                        </div>
                      </div>

                      <div className="doc-item-actions">
                        {isSelected && <span className="active-indicator-pill">Active</span>}
                        <button
                          type="button"
                          className="icon-btn-danger-subtle"
                          title="Delete document and vectors"
                          onClick={(e) => handleDeleteDocument(e, doc.document_id)}
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        </aside>

        {/* Right Area: Interactive Conversational QA Window */}
        <section className="chat-col">
          <ChatWindow
            activeDocument={activeDocument}
            messages={messages}
            isLoading={isGenerating}
            onSendMessage={handleSendMessage}
            onClearChat={handleClearChat}
          />
        </section>
      </main>
    </div>
  );
}
