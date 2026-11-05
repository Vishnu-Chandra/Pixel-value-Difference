import { useState } from "react";
import axios from "axios";
import "./App.css";

const API = "http://127.0.0.1:5000";

function App() {
  const [activeTab, setActiveTab] = useState("embed");

  // Sender state
  const [cover, setCover] = useState("lena.png");
  const [uploadedCover, setUploadedCover] = useState(null);
  const [uploadingCover, setUploadingCover] = useState(false);
  const [message, setMessage] = useState("Hello Secure AI EPVD");
  const [secretKey, setSecretKey] = useState("AI_EPVD_2026");

  const [stego, setStego] = useState("");
  const [embedResult, setEmbedResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Receiver state (Blind Extraction)
  const [extractStego, setExtractStego] = useState("");
  const [uploadedStego, setUploadedStego] = useState(null);
  const [uploadingStego, setUploadingStego] = useState(false);
  const [extractKey, setExtractKey] = useState("AI_EPVD_2026");
  const [extractResult, setExtractResult] = useState(null);

  // --------------------------------------------------------
  // Sender: Embed Message
  // --------------------------------------------------------
  const embedMessage = async () => {
    try {
      setLoading(true);
      setError("");
      setEmbedResult(null);

      const form = new URLSearchParams();
      form.append("cover", cover);
      form.append("message", message);
      form.append("mode", "secure");
      form.append("secret_key", secretKey);

      const response = await axios.post(`${API}/api/embed`, form);

      setEmbedResult(response.data);
      setStego(response.data.stego);

      // Pre-fill Extraction Lab with generated stego
      setExtractStego(response.data.stego);
      setExtractKey(secretKey);
      setUploadedStego({
        filename: response.data.stego,
        url: `${API}/api/files/upload/${response.data.stego}`,
        width: "Generated",
        height: "Stego PNG"
      });
    } catch (err) {
      setError(
        err.response?.data?.error ||
        "Unable to connect to the Flask backend."
      );
    } finally {
      setLoading(false);
    }
  };

  // --------------------------------------------------------
  // Receiver: Extract Message (Receiver-Only)
  // --------------------------------------------------------
  const extractMessage = async () => {
    try {
      setLoading(true);
      setError("");
      setExtractResult(null);

      const form = new FormData();
      form.append("stego", extractStego);
      form.append("secret_key", extractKey);

      const response = await axios.post(`${API}/api/extract`, form);

      setExtractResult(response.data);
    } catch (err) {
      setError(
        err.response?.data?.error ||
        "Extraction failed."
      );
    } finally {
      setLoading(false);
    }
  };

  // --------------------------------------------------------
  // Download Stego Image
  // --------------------------------------------------------
  const downloadStego = async () => {
    if (!stego) return;

    try {
      const response = await axios.get(
        `${API}/api/files/upload/${stego}`,
        { responseType: "blob" }
      );

      const url = window.URL.createObjectURL(
        new Blob([response.data])
      );

      const link = document.createElement("a");
      link.href = url;
      link.download = stego;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch {
      setError("Unable to download stego image.");
    }
  };

  // --------------------------------------------------------
  // Handle Cover Upload
  // --------------------------------------------------------
  const handleCoverUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      setUploadingCover(true);
      setError("");

      const form = new FormData();
      form.append("file", file);

      const response = await axios.post(
        `${API}/api/upload-cover`,
        form
      );

      setUploadedCover({
        filename: response.data.filename,
        url: `${API}${response.data.image_url}`,
        width: response.data.width,
        height: response.data.height
      });

      setCover(response.data.filename);
    } catch (err) {
      setError(
        err.response?.data?.error ||
        "Unable to upload image."
      );
    } finally {
      setUploadingCover(false);
    }
  };

  // --------------------------------------------------------
  // Handle Stego Upload (Receiver)
  // --------------------------------------------------------
  const handleStegoUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      setUploadingStego(true);
      setError("");

      const form = new FormData();
      form.append("file", file);

      const response = await axios.post(
        `${API}/api/upload-stego`,
        form
      );

      setUploadedStego({
        filename: response.data.filename,
        url: `${API}${response.data.image_url}`,
        width: response.data.width,
        height: response.data.height
      });

      setExtractStego(response.data.filename);
    } catch (err) {
      setError(
        err.response?.data?.error ||
        "Unable to upload stego image."
      );
    } finally {
      setUploadingStego(false);
    }
  };

  return (
    <div className="app">

      {/* TOP BAR */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">AI</div>
          <div>
            <h1>AI-EPVD</h1>
            <span>Secure Image Steganography System</span>
          </div>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          Backend Connected
        </div>
      </header>

      <main>

        {/* HERO */}
        <section className="hero">
          <div>
            <div className="eyebrow">
              ARTIFICIAL INTELLIGENCE • STEGANOGRAPHY • SECURITY
            </div>

            <h2>
              Intelligent Pixel Hiding
              <br />
              <span>with Adaptive Security</span>
            </h2>

            <p>
              AI-driven Enhanced Pixel Value Differencing using
              Random Forest adaptive bit allocation and chaotic
              block selection in YCbCr color space.
            </p>
          </div>

          <div className="hero-stat">
            <strong>YCbCr</strong>
            <span>Color Space</span>
          </div>
        </section>

        {/* NAVIGATION TABS */}
        <nav className="tabs">
          <button
            className={activeTab === "embed" ? "active" : ""}
            onClick={() => setActiveTab("embed")}
          >
            🔐 Stego Studio
          </button>

          <button
            className={activeTab === "extract" ? "active" : ""}
            onClick={() => setActiveTab("extract")}
          >
            🔓 Extraction Lab
          </button>
        </nav>

        {error && (
          <div className="error-box">
            {error}
          </div>
        )}

        {/* TAB 1: STEGO STUDIO (SENDER) */}
        {activeTab === "embed" && (
          <section className="workspace">

            {/* SENDER INPUT PANEL */}
            <div className="panel">
              <div className="panel-title">
                <div>
                  <span className="section-number">01</span>
                  <h3>Embed & Secure Message</h3>
                </div>
                <span className="badge">ENCODE</span>
              </div>

              <label>Cover Image</label>
              <div className="upload-box">
                <input
                  id="cover-upload"
                  type="file"
                  accept=".png,.jpg,.jpeg"
                  onChange={handleCoverUpload}
                  hidden
                />
                <label htmlFor="cover-upload" className="upload-label">
                  <div className="upload-icon">
                    {uploadingCover ? "..." : "↑"}
                  </div>
                  <strong>
                    {uploadingCover
                      ? "Uploading image..."
                      : "Upload Cover Image"}
                  </strong>
                  <span>PNG, JPG or JPEG</span>
                </label>
              </div>

              {uploadedCover && (
                <div className="uploaded-preview">
                  <img src={uploadedCover.url} alt="Uploaded cover" />
                  <div>
                    <strong>{uploadedCover.filename}</strong>
                    <span>{uploadedCover.width} × {uploadedCover.height}</span>
                  </div>
                </div>
              )}

              <div className="preset-row">
                <span>Or use test image:</span>
                <select
                  value={cover}
                  onChange={(e) => {
                    setCover(e.target.value);
                    setUploadedCover(null);
                  }}
                >
                  <option value="lena.png">Lena</option>
                  <option value="baboon.jpeg">Baboon</option>
                  <option value="jet.jpeg">Jet</option>
                  <option value="peppers.jpeg">Peppers</option>
                  <option value="sailboat.jpeg">Sailboat</option>
                  <option value="car-house.jpeg">Car House</option>
                </select>
              </div>

              <label>Secret Message</label>
              <textarea
                rows="4"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Enter your confidential message here..."
              />

              <label>Secret Key</label>
              <input
                value={secretKey}
                onChange={(e) => setSecretKey(e.target.value)}
                placeholder="Cryptographic key for chaotic permutation..."
              />

              <label>Steganography Algorithm</label>
              <div className="method-card active-card">
                <div className="method-icon">AI</div>
                <div>
                  <strong>AI + Chaotic EPVD</strong>
                  <small>Random Forest 5–7 bits • Logistic Map Chaotic Traversal • YCbCr</small>
                </div>
                <span className="method-status">ACTIVE</span>
              </div>

              <button
                className="primary-button"
                onClick={embedMessage}
                disabled={loading || !message}
              >
                {loading ? "Embedding..." : "🔐 Embed & Secure Message"}
              </button>
            </div>

            {/* SENDER RESULT PANEL */}
            <div className="panel result-panel">
              <div className="panel-title">
                <div>
                  <span className="section-number">02</span>
                  <h3>Generated Stego Image</h3>
                </div>
              </div>

              {!embedResult ? (
                <div className="empty-state">
                  <div className="empty-icon">⌘</div>
                  <h4>Ready to Embed</h4>
                  <p>
                    Enter a secret message and cryptographic key, then click
                    Embed & Secure Message.
                  </p>
                </div>
              ) : (
                <>
                  <div className="image-preview">
                    <img
                      src={`${API}${embedResult.stego_image}`}
                      alt="Stego"
                    />
                  </div>

                  <div className="metrics-grid">
                    <Metric
                      label="PSNR"
                      value={`${embedResult.psnr} dB`}
                    />
                    <Metric
                      label="SSIM"
                      value={embedResult.ssim}
                    />
                    <Metric
                      label="MSE"
                      value={embedResult.mse}
                    />
                    <Metric
                      label="Embedded Bits"
                      value={embedResult.embedded_bits}
                    />
                    <Metric
                      label="BPP"
                      value={embedResult.bpp}
                    />
                    <Metric
                      label="Execution"
                      value={`${embedResult.execution_time}s`}
                    />
                  </div>

                  <button
                    className="secondary-button"
                    onClick={downloadStego}
                  >
                    ↓ Download Stego Image
                  </button>
                </>
              )}
            </div>

          </section>
        )}

        {/* TAB 2: EXTRACTION LAB (RECEIVER ONLY) */}
        {activeTab === "extract" && (
          <section className="workspace">

            {/* RECEIVER INPUT PANEL */}
            <div className="panel">
              <div className="panel-title">
                <div>
                  <span className="section-number">01</span>
                  <h3>Extract Secret Message</h3>
                </div>
                <span className="badge">DECODE</span>
              </div>

              <label>Stego Image</label>
              <div className="upload-box">
                <input
                  id="stego-upload"
                  type="file"
                  accept=".png"
                  onChange={handleStegoUpload}
                  hidden
                />
                <label htmlFor="stego-upload" className="upload-label">
                  <div className="upload-icon">
                    {uploadingStego ? "..." : "↑"}
                  </div>
                  <strong>
                    {uploadingStego
                      ? "Uploading stego image..."
                      : "Upload Stego PNG"}
                  </strong>
                  <span>PNG • Lossless format required</span>
                </label>
              </div>

              {uploadedStego && (
                <div className="uploaded-preview">
                  <img
                    src={uploadedStego.url}
                    alt="Uploaded stego"
                  />
                  <div>
                    <strong>{uploadedStego.filename}</strong>
                    <span>
                      {uploadedStego.width} × {uploadedStego.height}
                    </span>
                  </div>
                </div>
              )}

              <label>Secret Key</label>
              <input
                value={extractKey}
                onChange={(e) => setExtractKey(e.target.value)}
                placeholder="Enter the secret key..."
              />

              <button
                className="primary-button"
                onClick={extractMessage}
                disabled={loading || !extractStego || !extractKey}
              >
                {loading ? "Extracting..." : "🔓 Extract Message →"}
              </button>
            </div>

            {/* RECEIVER RESULT PANEL */}
            <div className="panel result-panel">
              <div className="panel-title">
                <div>
                  <span className="section-number">02</span>
                  <h3>Recovered Message</h3>
                </div>
              </div>

              {!extractResult ? (
                <div className="empty-state">
                  <div className="empty-icon">⌁</div>
                  <h4>Waiting for Stego Image</h4>
                  <p>
                    Upload the received stego PNG, enter your secret key,
                    and click Extract Message.
                  </p>
                </div>
              ) : (
                <div className="extracted-message">
                  <div className="message-icon">✓</div>
                  <span>MESSAGE RECOVERED</span>

                  <div className="message-box">
                    {extractResult.extracted_message}
                  </div>

                  <div className="extract-status">
                    <div>
                      <strong>{extractResult.extracted_bits}</strong>
                      <span>Bits Extracted</span>
                    </div>

                    <div>
                      <strong>{extractResult.blocks_used}</strong>
                      <span>Blocks Used</span>
                    </div>

                    <div>
                      <strong>{extractResult.execution_time}s</strong>
                      <span>Execution</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

          </section>
        )}

      </main>

      {/* FOOTER */}
      <footer>
        <span>AI-Driven Adaptive EPVD • Steganography System</span>
        <span>YCbCr • Random Forest • Logistic Map</span>
      </footer>

    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;