import React, { useState, useRef } from 'react';
import { UploadCloud, File, AlertCircle, CheckCircle2, Loader2, X } from 'lucide-react';

const MAX_SIZE_MB = 20;

export default function FileUpload({ onUploadSuccess, isProcessing, onStartUpload }) {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [clientError, setClientError] = useState(null);
  const fileInputRef = useRef(null);

  const validateAndSetFile = (file) => {
    setClientError(null);
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
      setClientError('Only PDF files (.pdf) are supported.');
      return;
    }

    const sizeMb = file.size / (1024 * 1024);
    if (sizeMb > MAX_SIZE_MB) {
      setClientError(`File size (${sizeMb.toFixed(1)} MB) exceeds the maximum limit of ${MAX_SIZE_MB} MB.`);
      return;
    }

    setSelectedFile(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isProcessing) setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    if (isProcessing) return;

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      validateAndSetFile(files[0]);
    }
  };

  const handleFileChange = (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      validateAndSetFile(files[0]);
    }
  };

  const handleUploadClick = async () => {
    if (!selectedFile || isProcessing) return;
    setClientError(null);
    onStartUpload();
    try {
      await onUploadSuccess(selectedFile);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (err) {
      // handled by parent or captured here
      setClientError(err.message || 'Upload failed');
    }
  };

  const clearSelected = (e) => {
    e.stopPropagation();
    setSelectedFile(null);
    setClientError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="upload-container">
      <div
        className={`dropzone ${dragOver ? 'drag-active' : ''} ${isProcessing ? 'disabled' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isProcessing && fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          style={{ display: 'none' }}
          onChange={handleFileChange}
          disabled={isProcessing}
        />

        <div className="dropzone-content">
          <div className="upload-icon-circle">
            <UploadCloud size={28} className="upload-icon text-primary" />
          </div>

          <div className="dropzone-text">
            <p className="dropzone-primary-text">
              <span className="browse-link">Click to browse</span> or drag and drop your PDF here
            </p>
            <p className="dropzone-subtext">Supports PDF documents up to {MAX_SIZE_MB} MB</p>
          </div>
        </div>
      </div>

      {/* Selected File Card */}
      {selectedFile && (
        <div className="selected-file-card">
          <div className="file-info-col">
            <File size={22} className="text-primary flex-shrink-0" />
            <div className="file-meta">
              <span className="file-name" title={selectedFile.name}>{selectedFile.name}</span>
              <span className="file-size">{formatFileSize(selectedFile.size)}</span>
            </div>
          </div>

          <div className="file-actions-col">
            {!isProcessing && (
              <button
                type="button"
                className="icon-btn-danger"
                title="Remove file"
                onClick={clearSelected}
              >
                <X size={16} />
              </button>
            )}

            <button
              type="button"
              className="btn btn-primary"
              disabled={isProcessing}
              onClick={handleUploadClick}
            >
              {isProcessing ? (
                <>
                  <Loader2 size={16} className="spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <UploadCloud size={16} />
                  <span>Process Document</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {clientError && (
        <div className="upload-error-alert">
          <AlertCircle size={16} className="flex-shrink-0" />
          <span>{clientError}</span>
        </div>
      )}
    </div>
  );
}
