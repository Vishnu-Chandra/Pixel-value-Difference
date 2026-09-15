import React from 'react';

export default function MetricsCard({ label, value, hint, unit, accent = 'cyan' }) {
  const accentColors = {
    cyan: '#38bdf8',
    emerald: '#34d399',
    amber: '#fbbf24',
    rose: '#fb7185'
  };

  return (
    <div className="metric-card">
      <span className="metric-label">{label}</span>
      <div className="metric-val" style={{ color: accentColors[accent] || '#38bdf8' }}>
        {value} {unit && <span style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-muted)' }}>{unit}</span>}
      </div>
      {hint && <span className="metric-hint">{hint}</span>}
    </div>
  );
}
