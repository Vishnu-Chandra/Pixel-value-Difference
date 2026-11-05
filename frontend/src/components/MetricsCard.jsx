export default function MetricsCard({ label, value, hint, unit, accent = 'cyan' }) {
  const accentColors = {
    cyan: '#00D9FF',
    purple: '#7C3AED',
    secondary: '#7C3AED',
    emerald: '#22C55E',
    success: '#22C55E',
    amber: '#F59E0B',
    warning: '#F59E0B',
    rose: '#EF4444'
  };

  return (
    <div className="metric-card">
      <span className="metric-label">{label}</span>
      <div className="metric-val" style={{ color: accentColors[accent] || 'var(--primary)' }}>
        {value} {unit && <span style={{ fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-muted)' }}>{unit}</span>}
      </div>
      {hint && <span className="metric-hint">{hint}</span>}
    </div>
  );
}
