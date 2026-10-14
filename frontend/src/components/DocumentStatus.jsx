import React from 'react';
import { CheckCircle2, Circle, Loader2, AlertCircle } from 'lucide-react';

const STAGE_LABELS = [
  'PDF uploaded',
  'Text extracted',
  'Text chunked',
  'Embeddings generated',
  'Vectors stored in ChromaDB',
  'Document ready for questions'
];

export default function DocumentStatus({ stages, isProcessing, error }) {
  if (!stages && !isProcessing && !error) return null;

  // Build stage list aligning with the 6 standard stages
  const stageMap = {};
  if (stages && Array.isArray(stages)) {
    stages.forEach(s => {
      stageMap[s.name] = s;
    });
  }

  const completedCount = STAGE_LABELS.filter(name => stageMap[name]?.status === 'completed').length;
  const progressPercent = Math.round((completedCount / STAGE_LABELS.length) * 100);

  return (
    <div className="status-card">
      <div className="status-card-header">
        <div className="status-title-group">
          <h3 className="section-title">Processing Pipeline</h3>
          <span className="stage-counter">{completedCount}/{STAGE_LABELS.length} Complete</span>
        </div>
        {isProcessing && (
          <div className="processing-pill">
            <Loader2 size={14} className="spin" />
            <span>Processing</span>
          </div>
        )}
      </div>

      {/* Progress Bar */}
      <div className="progress-bar-track">
        <div 
          className={`progress-bar-fill ${error ? 'bg-danger' : completedCount === STAGE_LABELS.length ? 'bg-success' : 'bg-primary'}`}
          style={{ width: `${error ? 100 : progressPercent}%` }}
        />
      </div>

      {/* Stage Items */}
      <div className="stage-steps-list">
        {STAGE_LABELS.map((stageName, idx) => {
          const currentStage = stageMap[stageName];
          const status = currentStage?.status || (isProcessing && idx === completedCount ? 'processing' : 'pending');
          const detail = currentStage?.detail;

          let icon = <Circle size={16} className="stage-icon text-muted" />;
          let statusClass = 'stage-pending';

          if (status === 'completed') {
            icon = <CheckCircle2 size={16} className="stage-icon text-success" />;
            statusClass = 'stage-completed';
          } else if (status === 'processing') {
            icon = <Loader2 size={16} className="stage-icon spin text-primary" />;
            statusClass = 'stage-active';
          } else if (status === 'failed' || (error && idx === completedCount)) {
            icon = <AlertCircle size={16} className="stage-icon text-danger" />;
            statusClass = 'stage-failed';
          }

          return (
            <div key={stageName} className={`stage-step-item ${statusClass}`}>
              <div className="stage-icon-col">{icon}</div>
              <div className="stage-text-col">
                <span className="stage-name">{stageName}</span>
                {detail && <span className="stage-detail">{detail}</span>}
              </div>
            </div>
          );
        })}
      </div>

      {error && (
        <div className="stage-error-banner">
          <AlertCircle size={16} className="text-danger flex-shrink-0" />
          <div className="error-text">{error}</div>
        </div>
      )}
    </div>
  );
}
