import React from 'react'
import './ResultsPanel.css'

export default function ResultsPanel({ result, selected, onSelect }) {
  const { filename, numResidues, numPockets, pockets } = result

  return (
    <div className="results-panel">
      <div className="panel-title">Results</div>

      <div className="stats-row">
        <Stat label="Residues" value={numResidues} />
        <Stat label="Pockets"  value={numPockets} color="#34d399" />
      </div>

      <div className="filename">{filename}</div>

      {pockets.length === 0 ? (
        <div className="no-pockets">No pockets detected above threshold</div>
      ) : (
        <div className="pocket-list">
          {pockets.map((p, i) => (
            <div
              key={i}
              className={`pocket-card ${selected?.index === p.index ? 'active' : ''}`}
              onClick={() => onSelect(p)}
            >
              <div className="pocket-header">
                <span className="pocket-num">Pocket {i + 1}</span>
                <span className="pocket-conf">{(p.meanProb * 100).toFixed(1)}%</span>
              </div>
              <div className="pocket-meta">
                <span>{p.size} residues</span>
                <span>({p.center.map(v => v.toFixed(1)).join(', ')})</span>
              </div>
              <div className="conf-bar">
                <div className="conf-fill" style={{ width: `${p.meanProb * 100}%` }} />
              </div>
            </div>
          ))}
        </div>
      )}

      {selected && (
        <div className="residue-list">
          <div className="panel-title" style={{ padding: '0 0 8px' }}>
            Pocket {pockets.findIndex(p => p.index === selected.index) + 1} Residues
          </div>
          <div className="residue-grid">
            {selected.residues.map((r, i) => (
              <span key={i} className="residue-tag">
                {r.chain}:{r.resSeq}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function Stat({ label, value, color }) {
  return (
    <div className="stat">
      <span className="stat-value" style={color ? { color } : {}}>{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  )
}
