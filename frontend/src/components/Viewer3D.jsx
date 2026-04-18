import React, { useEffect, useRef, useState } from 'react'
import './Viewer3D.css'

export default function Viewer3D({ pdbText, residues, selected }) {
  const containerRef  = useRef()
  const stageRef      = useRef()
  const compRef       = useRef()
  const [ready, setReady]           = useState(false)
  const [isSpinning, setIsSpinning] = useState(false)

  // ── init NGL stage ────────────────────────────────────────────────────────
  useEffect(() => {
    let stage
    let ro

    import('ngl').then(NGL => {
      stage = new NGL.Stage(containerRef.current, {
        backgroundColor: '#0a0d16',
        tooltip: true,
      })
      stageRef.current = stage
      setReady(true)

      ro = new ResizeObserver(() => stage.handleResize())
      ro.observe(containerRef.current)
    })

    return () => {
      ro?.disconnect()
      stage?.dispose()
      stageRef.current = null
      compRef.current  = null
    }
  }, [])

  // ── load structure ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!ready || !pdbText || !stageRef.current) return

    const stage = stageRef.current
    stage.removeAllComponents()
    compRef.current = null
    stage.setSpin(false)

    import('ngl').then(NGL => {
      const blob = new Blob([pdbText], { type: 'text/plain' })
      stage.loadFile(blob, { ext: 'pdb', name: 'protein' }).then(comp => {
        compRef.current = comp
        comp.addRepresentation('cartoon', {
          colorScheme: 'chainname',
          opacity: 0.9,
        })
        comp.autoView(500)

        // Read the CURRENT spin state (functional update reads live value)
        setIsSpinning(current => {
          if (current) stage.setSpin(true)
          return current
        })

        if (selected) highlightPocket(comp, selected)
      })
    })
  }, [pdbText, ready])

  // ── highlight pocket ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!compRef.current || !selected) return
    highlightPocket(compRef.current, selected)
  }, [selected])

  // ── spin toggle ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (!stageRef.current) return
    if (isSpinning) {
      stageRef.current.setSpin(true)
    } else {
      stageRef.current.setSpin(false)
    }
  }, [isSpinning])

  // ── render ────────────────────────────────────────────────────────────────
  return (
    <div className="viewer-wrap" style={{ position: 'relative' }}>
      <div ref={containerRef} className="ngl-canvas" />

      {/* Legend Container */}
      <div style={{
        position: 'absolute',
        top: '16px',
        right: '16px',
        background: 'rgba(15, 23, 42, 0.8)',
        border: '1px solid rgba(52, 211, 153, 0.2)',
        padding: '12px',
        borderRadius: '8px',
        backdropFilter: 'blur(4px)',
        zIndex: 10,
        color: '#e2e8f0',
        fontSize: '12px',
        boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
        minWidth: '150px'
      }}>
        {/* Ribbon Colors Legend */}
        <div style={{ fontWeight: '600', marginBottom: '8px', color: '#34d399', textTransform: 'uppercase', letterSpacing: '0.05em', fontSize: '10px' }}>
          Structure Legend
        </div>
        <table style={{ borderCollapse: 'collapse', width: '100%', marginBottom: selected && selected.residues?.length > 0 ? '12px' : '0' }}>
          <tbody>
            <tr>
              <td style={{ padding: '4px 8px 4px 0' }}>
                 <div style={{ width: '12px', height: '12px', background: 'linear-gradient(135deg, #3050f8, #c8c8c8, #ff0d0d, #ffff30)', borderRadius: '2px' }} />
              </td>
              <td style={{ padding: '4px 0' }}>Protein Chains (Ribbons)</td>
            </tr>
          </tbody>
        </table>

        {/* Atom Color Legend (Only when pocket is selected) */}
        {selected && selected.residues?.length > 0 && (
          <>
            <div style={{ fontWeight: '600', marginBottom: '8px', color: '#34d399', textTransform: 'uppercase', letterSpacing: '0.05em', fontSize: '10px', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '8px' }}>
              Atom Legend (Pocket)
            </div>
            <table style={{ borderCollapse: 'collapse', width: '100%' }}>
              <tbody>
                <tr>
                  <td style={{ padding: '4px 8px 4px 0' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#c8c8c8' }} /></td>
                  <td style={{ padding: '4px 0' }}>Carbon (C)</td>
                </tr>
                <tr>
                  <td style={{ padding: '4px 8px 4px 0' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#ff0d0d' }} /></td>
                  <td style={{ padding: '4px 0' }}>Oxygen (O)</td>
                </tr>
                <tr>
                  <td style={{ padding: '4px 8px 4px 0' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#3050f8' }} /></td>
                  <td style={{ padding: '4px 0' }}>Nitrogen (N)</td>
                </tr>
                <tr>
                  <td style={{ padding: '4px 8px 4px 0' }}><div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#ffff30' }} /></td>
                  <td style={{ padding: '4px 0' }}>Sulfur (S)</td>
                </tr>
              </tbody>
            </table>
          </>
        )}
      </div>

      {!ready && (
        <div className="viewer-loading">Loading 3D viewer…</div>
      )}
      <div className="viewer-controls" style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
        <span>🖱 Rotate · Scroll zoom · Right-drag pan</span>
        <button
          onClick={() => setIsSpinning(prev => !prev)}
          style={{
            background: isSpinning ? 'rgba(52, 211, 153, 0.2)' : 'rgba(255,255,255,0.1)',
            color: isSpinning ? '#34d399' : '#fff',
            border: `1px solid ${isSpinning ? '#34d399' : 'rgba(255,255,255,0.2)'}`,
            borderRadius: '6px', cursor: 'pointer', padding: '4px 12px', transition: '0.2s'
          }}
        >
          {isSpinning ? '⏸ Stop Rotation' : '▶ Auto-Rotate'}
        </button>
      </div>
    </div>
  )
}

function highlightPocket(comp, selected) {
  comp.reprList
    .filter(r => r.name === 'pocket-hl' || r.name === 'pocket-surf')
    .forEach(r => comp.removeRepresentation(r))

  if (!selected?.residues?.length) return

  const sele = selected.residues
    .map(r => `(${r.resSeq} and :${r.chain})`)
    .join(' or ')

  comp.addRepresentation('ball+stick', {
    sele,
    colorScheme: 'element',
    opacity: 1.0,
    name: 'pocket-hl',
  })

  comp.addRepresentation('surface', {
    sele,
    colorValue: '#34d399',
    opacity: 0.3,
    name: 'pocket-surf',
  })

  comp.autoView(sele, 1000)
}
