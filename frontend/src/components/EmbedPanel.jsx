import { useState } from 'react';

export default function EmbedPanel() {
  const [cover, setCover] = useState('lena.png');
  const [message, setMessage] = useState('');
  const [mode, setMode] = useState('ai');
  const [secretKey, setSecretKey] = useState('AI_EPVD_2026');

  return (
    <div className="glass-card">
      <h2>Stego Embedding Studio</h2>
      <form>
        <div className="form-group">
          <label>Cover Image</label>
          <input
            type="text"
            className="form-control"
            value={cover}
            onChange={(e) => setCover(e.target.value)}
            placeholder="e.g. lena.png"
          />
        </div>

        <div className="form-group">
          <label>Mode</label>
          <select
            className="form-control"
            value={mode}
            onChange={(e) => setMode(e.target.value)}
          >
            <option value="standard">Standard EPVD</option>
            <option value="ai">AI-Adaptive EPVD</option>
            <option value="secure">Secure AI EPVD</option>
          </select>
        </div>

        {mode === 'secure' && (
          <div className="form-group">
            <label>Secret Key</label>
            <input
              type="text"
              className="form-control"
              value={secretKey}
              onChange={(e) => setSecretKey(e.target.value)}
              placeholder="e.g. AI_EPVD_2026"
            />
          </div>
        )}

        <div className="form-group">
          <label>Secret Message</label>
          <textarea
            className="form-control"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Enter secret message..."
          />
        </div>

        <button type="submit" className="btn-primary">
          Embed Message
        </button>
      </form>
    </div>
  );
}
