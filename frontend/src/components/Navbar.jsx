export default function Navbar({ activeTab, setActiveTab, serverOnline }) {
  const tabs = [
    { id: 'embed', label: 'Embed Studio', icon: '🔒' },
    { id: 'extract', label: 'Extract Studio', icon: '🔓' },
    { id: 'security', label: 'Key Sensitivity Lab', icon: '🛡️' }
  ];

  return (
    <header className="navbar">
      <div className="nav-brand">
        <span className="brand-icon">⚡</span>
        <div>
          <div className="brand-title">AI-EPVD Studio</div>
          <div className="brand-subtitle">Adaptive Steganography with Chaotic Cryptography</div>
        </div>
      </div>

      <nav className="nav-tabs">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </nav>

      <div className="server-badge">
        <span className={`status-dot ${serverOnline ? '' : 'offline'}`} style={{ backgroundColor: serverOnline ? '#22C55E' : '#EF4444' }} />
        <span>{serverOnline ? 'Flask API: Online' : 'Connecting API...'}</span>
      </div>
    </header>
  );
}
