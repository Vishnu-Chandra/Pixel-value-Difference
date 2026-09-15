import React, { useState } from 'react';

export default function KeySensitivity() {
  const [correctKey, setCorrectKey] = useState('AI_EPVD_2026');
  const [testKey, setTestKey] = useState('WRONG_KEY_2026');

  return (
    <div className="glass-card">
      <h2>Key Sensitivity Lab</h2>
      <p>Test cryptographic avalanche effect on chaotic block ordering.</p>

      <div className="form-group">
        <label>Correct Key</label>
        <input
          type="text"
          className="form-control"
          value={correctKey}
          onChange={(e) => setCorrectKey(e.target.value)}
        />
      </div>

      <div className="form-group">
        <label>Perturbed / Test Key</label>
        <input
          type="text"
          className="form-control"
          value={testKey}
          onChange={(e) => setTestKey(e.target.value)}
        />
      </div>

      <button type="button" className="btn-primary">
        Run Sensitivity Analysis
      </button>
    </div>
  );
}
