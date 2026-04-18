import React, { useState } from 'react'
import UploadPanel from './components/UploadPanel'
import ResultsPanel from './components/ResultsPanel'
import Viewer3D from './components/Viewer3D'
import AnalysisPanel from './components/AnalysisPanel'
import ModelPerformance from './components/ModelPerformance'
import './App.css'

export default function App() {
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [pdbText, setPdbText] = useState(null)
  const [selected, setSelected] = useState(null)
  const [activeTab, setActiveTab] = useState('viewer')

  async function handleUpload(file, model, threshold) {
    setLoading(true)
    setError(null)
    setResult(null)
    setPdbText(null)
    setActiveTab('viewer')

    // read raw text for 3D viewer
    const text = await file.text()
    setPdbText(text)

    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch(`http://localhost:8000/predict?model=${model}&threshold=${threshold}`, {
        method: 'POST',
        body: form,
      })
      if (!res.ok) {
        let msg = `Server error ${res.status}`
        try { const err = await res.json(); msg = err.detail || msg } catch {}
        throw new Error(msg)
      }
      const data = await res.json()
      setResult(data)
      setSelected(data.pockets[0] ?? null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="logo">⬡ FA-GAT</div>
      </header>

      <main className="main">
        <aside className="sidebar">
          <UploadPanel onSubmit={handleUpload} loading={loading} />
          {error && <div className="error-box">⚠ {error}</div>}
          {result && (
            <ResultsPanel
              result={result}
              selected={selected}
              onSelect={setSelected}
            />
          )}
        </aside>

        <section className="viewer-section">
          {pdbText && result && (
            <div className="view-tabs">
              <button 
                className={`tab-btn ${activeTab === 'viewer' ? 'active' : ''}`}
                onClick={() => setActiveTab('viewer')}
              >
                3D Viewer
              </button>
              <button 
                className={`tab-btn ${activeTab === 'analysis' ? 'active' : ''}`}
                onClick={() => setActiveTab('analysis')}
              >
                Prediction Analysis
              </button>
            </div>
          )}

          {pdbText ? (
            <div className="tab-content" style={{ height: '100%', flex: 1, display: 'flex', flexDirection: 'column' }}>
              {activeTab === 'viewer' ? (
                <Viewer3D
                  pdbText={pdbText}
                  residues={result?.residues}
                  selected={selected}
                />
              ) : (
                <AnalysisPanel 
                  result={result} 
                  selected={selected} 
                />
              )}
            </div>
          ) : (
            <ModelPerformance />
          )}
        </section>
      </main>
    </div>
  )
}

