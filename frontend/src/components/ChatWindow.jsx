import React, { useState, useRef, useEffect } from 'react';
import { Send, Trash2, Bot, MessageSquare, Loader2, Sparkles, AlertCircle, FileText } from 'lucide-react';
import ChatMessage from './ChatMessage';

const SUGGESTED_QUESTIONS = [
  "What is the main topic of this document?",
  "Can you summarize the key findings or takeaways?",
  "What are the conclusions or recommendations made?"
];

export default function ChatWindow({
  activeDocument,
  messages,
  isLoading,
  onSendMessage,
  onClearChat
}) {
  const [inputQuestion, setInputQuestion] = useState('');
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll on new message or loading change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSubmit = (e) => {
    e?.preventDefault();
    if (!inputQuestion.trim() || isLoading || !activeDocument) return;
    onSendMessage(inputQuestion.trim());
    setInputQuestion('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSelectSuggestion = (q) => {
    if (isLoading || !activeDocument) return;
    onSendMessage(q);
  };

  return (
    <div className="chat-window-card">
      {/* Chat Header */}
      <div className="chat-window-header">
        <div className="chat-doc-info">
          {activeDocument ? (
            <div className="active-doc-tag">
              <FileText size={16} className="text-primary flex-shrink-0" />
              <div className="active-doc-meta">
                <span className="active-doc-name" title={activeDocument.filename}>
                  {activeDocument.filename}
                </span>
                <span className="active-doc-stats">
                  {activeDocument.page_count} page{activeDocument.page_count > 1 ? 's' : ''} • {activeDocument.chunk_count} chunks indexed
                </span>
              </div>
            </div>
          ) : (
            <div className="no-doc-tag">
              <AlertCircle size={16} className="text-muted flex-shrink-0" />
              <span>No document selected. Upload a PDF to begin asking questions.</span>
            </div>
          )}
        </div>

        {messages.length > 0 && (
          <button
            type="button"
            className="btn btn-secondary-subtle btn-sm"
            onClick={onClearChat}
            disabled={isLoading}
            title="Clear current session chat history"
          >
            <Trash2 size={14} />
            <span>Clear Chat</span>
          </button>
        )}
      </div>

      {/* Messages Scroll Area */}
      <div className="chat-messages-container">
        {messages.length === 0 ? (
          <div className="chat-empty-state">
            <div className="empty-state-icon">
              <Bot size={36} className="text-primary" />
            </div>
            <h3 className="empty-state-title">
              {activeDocument ? `Ask anything about "${activeDocument.filename}"` : 'Welcome to PDF_Parser'}
            </h3>
            <p className="empty-state-sub">
              {activeDocument
                ? 'Your document has been indexed in ChromaDB. Ask specific questions and receive verified, context-grounded answers.'
                : 'Upload a document from the left panel to generate embeddings and unlock intelligent document querying.'}
            </p>

            {activeDocument && (
              <div className="suggestions-container">
                <span className="suggestions-label">Try asking:</span>
                <div className="suggestions-list">
                  {SUGGESTED_QUESTIONS.map((q, i) => (
                    <button
                      key={i}
                      type="button"
                      className="suggestion-chip"
                      onClick={() => handleSelectSuggestion(q)}
                      disabled={isLoading}
                    >
                      <Sparkles size={13} className="text-primary" />
                      <span>{q}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="messages-stream">
            {messages.map((msg, index) => (
              <ChatMessage key={index} message={msg} />
            ))}

            {isLoading && (
              <div className="chat-message-row assistant-row">
                <div className="avatar assistant-avatar">
                  <Sparkles size={18} />
                </div>
                <div className="message-content-wrapper">
                  <div className="message-bubble assistant-bubble loading-bubble">
                    <Loader2 size={16} className="spin text-primary" />
                    <span>Searching vectors & generating grounded answer...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Chat Input Bar */}
      <form className="chat-input-area" onSubmit={handleSubmit}>
        <div className="input-box-wrapper">
          <textarea
            ref={textareaRef}
            className="chat-textarea"
            placeholder={
              !activeDocument
                ? 'Upload and select a PDF document first...'
                : isLoading
                ? 'Generating answer...'
                : 'Ask a question about this document... (Press Enter to send, Shift+Enter for newline)'
            }
            value={inputQuestion}
            onChange={(e) => setInputQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={!activeDocument || isLoading}
            rows={2}
          />

          <button
            type="submit"
            className="send-button"
            disabled={!activeDocument || !inputQuestion.trim() || isLoading}
            title="Send question"
          >
            {isLoading ? <Loader2 size={18} className="spin" /> : <Send size={18} />}
          </button>
        </div>

        <div className="input-footer-note">
          <span>Answers are strictly grounded in document context via ChromaDB & Ollama.</span>
        </div>
      </form>
    </div>
  );
}
