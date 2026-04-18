import React, { useMemo, useState } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import './AnalysisPanel.css';

export default function AnalysisPanel({ result, selected }) {
  const [threshold, setThreshold] = useState(50); // percentage

  if (!result || !result.residues) return null;

  // Prepare data for the chart
  const data = useMemo(() => {
    return result.residues.map((r) => ({
      name: `${r.resName} ${r.resSeq}`,
      chain: r.chain,
      resSeq: r.resSeq,
      probability: r.prob,
      isPocket: selected?.residues?.some(sr => sr.resSeq === r.resSeq && sr.chain === r.chain) ? 1 : 0
    }));
  }, [result, selected]);

  const avgProb = useMemo(() => {
    if (data.length === 0) return 0;
    const sum = data.reduce((acc, curr) => acc + curr.probability, 0);
    return (sum / data.length).toFixed(3);
  }, [data]);

  const thresholdDecimal = threshold / 100;
  
  // Filter and sort for the confidence ranking table
  const highRiskResidues = useMemo(() => {
    return [...data]
      .filter(r => r.probability >= thresholdDecimal)
      .sort((a, b) => b.probability - a.probability);
  }, [data, thresholdDecimal]);

  const downloadCSV = () => {
    if (highRiskResidues.length === 0) return;
    const headers = ["Chain", "Residue Sequence", "Residue Name", "Probability"];
    const rows = highRiskResidues.map(r => 
      `${r.chain},${r.resSeq},${r.name.split(' ')[0]},${r.probability.toFixed(4)}`
    );
    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows].join("\n");
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `pocket_predictions_th${threshold}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="analysis-panel">
      <div className="analysis-header">
        <h2>Prediction Analysis</h2>
        <div className="analysis-stats">
          <span>Total Residues: {data.length}</span>
          <span>Avg Probability: {avgProb}</span>
        </div>
      </div>
      
      <div className="chart-container">
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
          <h3>Binding Probability per Residue</h3>
          <div className="threshold-control">
            <label>Sensitivity Threshold: {threshold}%</label>
            <input 
              type="range" 
              min="1" max="99" 
              value={threshold} 
              onChange={(e) => setThreshold(e.target.value)}
              className="threshold-slider" 
            />
          </div>
        </div>
        
        <ResponsiveContainer width="100%" height={250}>
          <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="colorProb" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" opacity={0.2} vertical={false} />
            <XAxis dataKey="resSeq" stroke="#888" fontSize={12} tickLine={false} axisLine={false} />
            <YAxis stroke="#888" fontSize={12} tickLine={false} axisLine={false} domain={[0, 1]} />
            <Tooltip 
              contentStyle={{ backgroundColor: 'rgba(17, 24, 39, 0.9)', border: '1px solid #374151', borderRadius: '8px', color: '#fff' }}
              itemStyle={{ color: '#fff' }}
              formatter={(val) => [(val * 100).toFixed(2) + '%', 'Probability']}
              labelFormatter={(label, payload) => payload?.[0]?.payload?.name || label}
            />
            
            {/* Dynamic Interactive Threshold Line */}
            <ReferenceLine y={thresholdDecimal} stroke="#f59e0b" strokeDasharray="3 3" label={{ position: 'top', value: `Target Thresh (${threshold}%)`, fill: '#f59e0b', fontSize: 10 }} />
            
            {selected && (
              <ReferenceLine y={selected.meanProb} stroke="#10b981" strokeDasharray="3 3" label={{ position: 'bottom', value: 'Avg Pocket Prob', fill: '#10b981', fontSize: 10 }} />
            )}
            
            <Area type="monotone" dataKey="probability" stroke="#3b82f6" fillOpacity={1} fill="url(#colorProb)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="analysis-grid">
        <div className="pocket-distribution">
          <h3>Pocket Analysis</h3>
          <p>
            The dynamic threshold lets you tune sensitivity. Residues scoring above this line are considered highest risk for binding interactions.
            {selected && (
              <span style={{color: '#10b981', display: 'block', marginTop: '10px'}}>
                Currently highlighting Pocket {result.pockets.findIndex(p => p.index === selected.index) + 1} with an average probability score of {(selected.meanProb * 100).toFixed(1)}%.
              </span>
            )}
          </p>
        </div>

        <div className="confidence-ranking">
          <div className="ranking-header">
            <h3>Top Active Residues (&gt; {threshold}%)</h3>
            <button onClick={downloadCSV} className="btn-download" disabled={highRiskResidues.length === 0}>
              💾 Export CSV
            </button>
          </div>
          
          <div className="table-container">
            <table className="residue-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>ID</th>
                  <th>Chain</th>
                  <th>Score</th>
                </tr>
              </thead>
              <tbody>
                {highRiskResidues.length > 0 ? highRiskResidues.map((r, i) => (
                  <tr key={i} className={r.isPocket ? "highlight-row" : ""}>
                    <td>#{i + 1}</td>
                    <td>{r.name}</td>
                    <td>{r.chain}</td>
                    <td className="score-cell">{(r.probability * 100).toFixed(1)}%</td>
                  </tr>
                )) : (
                  <tr>
                    <td colSpan="4" className="empty-state">No residues above threshold.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
