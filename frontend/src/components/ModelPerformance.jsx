import React, { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import './ModelPerformance.css';

export default function ModelPerformance() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchMetrics() {
      try {
        const res = await fetch('http://localhost:8000/api/metrics/fagat');
        if (!res.ok) throw new Error('Failed to fetch metrics');
        const metrics = await res.json();
        setData(metrics);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchMetrics();
  }, []);

  if (loading) return <div className="metrics-loading">Loading metrics...</div>;
  if (error) return <div className="metrics-error">Error loading metrics: {error}</div>;
  if (!data || data.length === 0) return <div className="metrics-error">No training metrics found.</div>;

  return (
    <div className="model-performance">
      <div className="perf-header">
        <h2>Feature-Augmented Graph Attention Network (FA-GAT) Model Evaluation</h2>
      </div>

      <div className="perf-charts">
        <div className="perf-chart-box">
          <h3>Accuracy Metrics (F1, Precision, Recall)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.1} vertical={false} />
              <XAxis dataKey="epoch" stroke="#888" tickLine={false} axisLine={false} />
              <YAxis domain={[0, 1]} stroke="#888" tickLine={false} axisLine={false} />
              <Tooltip 
                contentStyle={{ backgroundColor: 'rgba(17, 24, 39, 0.9)', border: '1px solid #374151', borderRadius: '8px' }}
                itemStyle={{ color: '#fff' }}
              />
              <Legend verticalAlign="top" height={36}/>
              <Line type="monotone" dataKey="f1" name="F1 Score" stroke="#10b981" strokeWidth={3} dot={false} activeDot={{ r: 6 }} />
              <Line type="monotone" dataKey="precision" name="Precision" stroke="#3b82f6" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="recall" name="Recall" stroke="#f59e0b" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="perf-chart-box">
          <h3>Training Loss</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.1} vertical={false} />
              <XAxis dataKey="epoch" stroke="#888" tickLine={false} axisLine={false} />
              <YAxis stroke="#888" tickLine={false} axisLine={false} />
              <Tooltip 
                contentStyle={{ backgroundColor: 'rgba(17, 24, 39, 0.9)', border: '1px solid #374151', borderRadius: '8px' }}
              />
              <Line type="monotone" dataKey="loss" name="Focal Loss" stroke="#ef4444" strokeWidth={3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
