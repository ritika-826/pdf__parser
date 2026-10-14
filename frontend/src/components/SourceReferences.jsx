import React, { useState } from 'react';
import { BookOpen, ChevronDown, ChevronUp, Copy, Check } from 'lucide-react';

export default function SourceReferences({ sources }) {
  const [isOpen, setIsOpen] = useState(false);
  const [copiedIdx, setCopiedIdx] = useState(null);

  if (!sources || sources.length === 0) return null;

  const handleCopy = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(null), 2000);
  };

  return (
    <div className="source-references-container">
      <button
        type="button"
        className="source-toggle-btn"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
      >
        <div className="source-toggle-left">
          <BookOpen size={16} className="source-icon" />
          <span>
            Retrieved Sources ({sources.length} excerpt{sources.length > 1 ? 's' : ''})
          </span>
        </div>
        {isOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>

      {isOpen && (
        <div className="sources-list">
          {sources.map((src, idx) => (
            <div key={src.chunk_id || idx} className="source-card">
              <div className="source-card-header">
                <div className="source-badges">
                  <span className="badge badge-page">Page {src.page}</span>
                  {src.score !== null && src.score !== undefined && (
                    <span className="badge badge-score">
                      Relevance: {Math.round(src.score * 100)}%
                    </span>
                  )}
                  <span className="chunk-tag">{src.chunk_id}</span>
                </div>
                <button
                  type="button"
                  className="icon-btn-subtle"
                  title="Copy excerpt"
                  onClick={() => handleCopy(src.text, idx)}
                >
                  {copiedIdx === idx ? <Check size={14} className="text-success" /> : <Copy size={14} />}
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
