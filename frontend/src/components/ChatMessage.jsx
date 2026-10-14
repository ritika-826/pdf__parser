import React, { useState } from 'react';
import { User, Sparkles, Copy, Check } from 'lucide-react';
import SourceReferences from './SourceReferences';

export default function ChatMessage({ message }) {
  const isUser = message.role === 'user';
  const [copied, setCopied] = useState(false);

  const handleCopyAnswer = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Basic markdown-like paragraph and list rendering
  const renderFormattedText = (text) => {
    if (!text) return null;
    const paragraphs = text.split('\n\n');
    return paragraphs.map((para, pIdx) => {
      // Check for bullet lists
      if (para.includes('\n- ') || para.startsWith('- ') || para.includes('\n* ') || para.startsWith('* ')) {
        const lines = para.split('\n');
        return (
          <ul key={pIdx} className="chat-list">
            {lines.map((line, lIdx) => {
              const cleanLine = line.replace(/^[-*]\s+/, '').trim();
              return cleanLine ? <li key={lIdx}>{formatInline(cleanLine)}</li> : null;
            })}
          </ul>
        );
      }
      return <p key={pIdx} className="chat-paragraph">{formatInline(para)}</p>;
    });
  };

  // Format bold text and citations
  const formatInline = (str) => {
    // Regex for bold **text**
    const parts = str.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, idx) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={idx}>{part.slice(2, -2)}</strong>;
      }
      // Highlight page citations like [Page 1]
      const subParts = part.split(/(\[Page\s+\d+\])/gi);
      return subParts.map((sub, sIdx) => {
        if (/^\[Page\s+\d+\]$/i.test(sub)) {
          return <span key={`${idx}-${sIdx}`} className="inline-page-citation">{sub}</span>;
        }
        return sub;
      });
    });
  };

  return (
    <div className={`chat-message-row ${isUser ? 'user-row' : 'assistant-row'}`}>
      <div className={`avatar ${isUser ? 'user-avatar' : 'assistant-avatar'}`}>
        {isUser ? <User size={18} /> : <Sparkles size={18} />}
      </div>

      <div className="message-content-wrapper">
        <div className="message-header">
          <span className="sender-name">{isUser ? 'You' : 'PDF_Parser AI'}</span>
          <div className="message-actions">
            {!isUser && (
              <button
                type="button"
                className="icon-btn-subtle"
                title="Copy response"
                onClick={handleCopyAnswer}
              >
                {copied ? <Check size={14} className="text-success" /> : <Copy size={14} />}
              </button>
            )}
            <span className="timestamp">{message.timestamp || 'Just now'}</span>
          </div>
        </div>

        <div className={`message-bubble ${isUser ? 'user-bubble' : 'assistant-bubble'}`}>
          {renderFormattedText(message.content)}
        </div>

        {!isUser && message.sources && message.sources.length > 0 && (
          <SourceReferences sources={message.sources} />
        )}
      </div>
    </div>
  );
}
