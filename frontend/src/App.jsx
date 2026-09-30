import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [querying, setQuerying] = useState(false);
  const [uploadedName, setUploadedName] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [chunksCount, setChunksCount] = useState(0);
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = 'auto';
      ta.style.height = Math.min(ta.scrollHeight, 140) + 'px';
    }
  }, [query]);

  const handleFileChange = (f) => {
    if (!f) return;
    setFile(f);
    setUploadedName('');
    setChunksCount(0);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await axios.post('http://127.0.0.1:8000/upload', formData);
      setUploadedName(file.name);
      setChunksCount(res.data.chunks_indexed);
      setMessages([{
        role: 'system',
        text: `✓ **${file.name}** indexed — ${res.data.chunks_indexed} chunks ready. Ask me anything!`,
      }]);
    } catch (e) {
      const detail = e.response?.data?.detail || 'Upload failed. Please try again.';
      setMessages([{ role: 'system', text: `✗ ${detail}` }]);
    }
    setUploading(false);
  };

  const handleQuery = async () => {
    if (!query.trim() || querying) return;

    const userMsg = { role: 'user', text: query.trim() };
    const updatedMessages = [...messages, userMsg];
    setMessages(updatedMessages);
    setQuery('');
    setQuerying(true);

    // Build history (exclude system messages)
    const history = updatedMessages
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .slice(-10); // last 5 exchanges

    try {
      const res = await axios.post('http://127.0.0.1:8000/query', {
        query: userMsg.text,
        history,
      });
      setMessages(prev => [...prev, { role: 'assistant', text: res.data.response }]);
    } catch (e) {
      const detail = e.response?.data?.detail || 'Something went wrong. Please try again.';
      setMessages(prev => [...prev, { role: 'assistant', text: `⚠️ ${detail}` }]);
    }
    setQuerying(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuery();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="shell">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="logo">
          <span className="logo-icon">◈</span>
          <span className="logo-text">RAG<em>mind</em></span>
        </div>

        <div
          className="upload-zone"
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={e => { e.preventDefault(); setDragOver(false); handleFileChange(e.dataTransfer.files[0]); }}
          onClick={() => fileInputRef.current.click()}
          data-drag={dragOver}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.txt,.md,.csv,.xlsx,.pptx,.html,.htm"
            style={{ display: 'none' }}
            onChange={e => handleFileChange(e.target.files[0])}
          />
          <div className="upload-icon">{file ? '📄' : '⊕'}</div>
          <p className="upload-label">{file ? file.name : 'Drop any document'}</p>
          <p className="upload-sub">{file ? `${(file.size / 1024).toFixed(1)} KB` : 'PDF, DOCX, TXT, CSV, XLSX, PPTX, HTML, MD'}</p>
        </div>

        <button className="upload-btn" onClick={handleUpload} disabled={!file || uploading}>
          {uploading ? <span className="spin">⟳</span> : '↑'}
          {uploading ? ' Indexing…' : ' Index Document'}
        </button>

        {uploadedName && (
          <div className="indexed-badge">
            <span>●</span>
            <div>
              <div>{uploadedName}</div>
              <div className="chunks-info">{chunksCount} chunks indexed</div>
            </div>
          </div>
        )}

        {messages.length > 1 && (
          <button className="clear-btn" onClick={clearChat}>
            ✕ Clear Chat
          </button>
        )}

        <div className="sidebar-footer">
          <p>Agentic RAG · Query Rewriting</p>
          <p>Gemini 2.5 Flash · FAISS</p>
          <p>HuggingFace Embeddings</p>
        </div>
      </aside>

      {/* ── Chat Panel ── */}
      <main className="chat-panel">
        <header className="chat-header">
          <h1>Document Intelligence</h1>
          <p>Upload a document, then ask anything about it</p>
        </header>

        <div className="chat-body">
          {messages.length === 0 && (
            <div className="empty-state">
              <div className="empty-icon">◈</div>
              <p>No conversation yet.<br />Upload a document to begin.</p>
              <div className="feature-pills">
                <span>🔍 Semantic Search</span>
                <span>🧠 Query Rewriting</span>
                <span>💬 Chat Memory</span>
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`bubble-row ${msg.role}`}>
              {(msg.role === 'assistant') && <div className="avatar">AI</div>}
              {(msg.role === 'system') && <div className="avatar sys">✓</div>}
              <div className={`bubble ${msg.role}`}>
                <ReactMarkdown>{msg.text}</ReactMarkdown>
              </div>
              {msg.role === 'user' && <div className="avatar user">You</div>}
            </div>
          ))}

          {querying && (
            <div className="bubble-row assistant">
              <div className="avatar">AI</div>
              <div className="bubble assistant typing">
                <span /><span /><span />
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        <div className="input-bar">
          <textarea
            ref={textareaRef}
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={uploadedName ? 'Ask about your document… (Enter to send)' : 'Upload a document first…'}
            rows={1}
            disabled={!uploadedName || querying}
          />
          <button onClick={handleQuery} disabled={!query.trim() || querying || !uploadedName} className="send-btn">
            ↑
          </button>
        </div>
      </main>
    </div>
  );
}

export default App;