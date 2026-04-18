import React, { useState, useRef } from 'react'
import './UploadPanel.css'

export default function UploadPanel({ onSubmit, loading }) {
  const [file, setFile]           = useState(null)
  const [model, setModel]         = useState('fagat')
  const [threshold, setThreshold] = useState(0.5)
  const [drag, setDrag]           = useState(false)
  const inputRef = useRef()

  function handleDrop(e) {
    e.preventDefault(); setDrag(false)
    const f = e.dataTransfer.files[0]
    if (f?.name.endsWith('.pdb')) setFile(f)
  }

  function handleSubmit(e) {
    e.preventDefault()
    if (file) onSubmit(file, model, threshold)
  }

  return (
    <form className="upload-panel" onSubmit={handleSubmit}>
      <div className="panel-title">Upload Protein</div>

      <div
        className={`drop-zone ${drag ? 'drag' : ''} ${file ? 'has-file' : ''}`}
        onDragOver={e => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdb"
          hidden
          onChange={e => setFile(e.target.files[0])}
        />
        {file ? (
          <>
            <span className="file-icon">📄</span>
            <span className="file-name">{file.name}</span>
            <span className="file-size">{(file.size / 1024).toFixed(1)} KB</span>
          </>
        ) : (
          <>
            <span className="upload-icon">⬆</span>
            <span>Drop a .pdb file or click to browse</span>
          </>
        )}
      </div>

      <div className="field">
        <label>Model</label>
        <select value={model} onChange={e => setModel(e.target.value)}>
          <option value="fagat">FA-GAT</option>
          <option value="gcn">GCN Baseline</option>
        </select>
      </div>

      <div className="field">
        <label>Threshold — {threshold}</label>
        <input
          type="range" min="0.1" max="0.9" step="0.05"
          value={threshold}
          onChange={e => setThreshold(parseFloat(e.target.value))}
        />
      </div>

      <button className="predict-btn" type="submit" disabled={!file || loading}>
        {loading ? <span className="spinner" /> : '🔍 Predict Pockets'}
      </button>
    </form>
  )
}
