// frontend/src/pages/PipelinePage.jsx
import React, { useState } from 'react';
import { 
  GitBranch, 
  Layers, 
  Cpu, 
  ArrowRight, 
  CheckCircle2, 
  AlertTriangle, 
  FileText, 
  Sliders, 
  Sparkles,
  Zap,
  ShieldAlert,
  HelpCircle
} from 'lucide-react';

export default function PipelinePage() {
  const [activeStage, setActiveStage] = useState('stage2');

  return (
    <div style={{ maxWidth: '1280px', margin: '0 auto', padding: '2rem 1.5rem' }}>
      {/* Title */}
      <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.5rem',
          padding: '0.35rem 0.85rem',
          borderRadius: '9999px',
          background: 'rgba(6, 182, 212, 0.1)',
          border: '1px solid rgba(6, 182, 212, 0.25)',
          color: '#38bdf8',
          fontSize: '0.8rem',
          fontWeight: 600,
          marginBottom: '1rem'
        }}>
          <GitBranch size={14} />
          End-to-End Deep Learning Architecture
        </div>
        <h1 style={{ fontSize: '2.4rem', marginBottom: '0.75rem' }}>
          System Pipeline & <span className="gradient-text">Algorithmic Design</span>
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.05rem', maxWidth: '750px', margin: '0 auto' }}>
          A decoupled Two-Stage Aspect-Based Sentiment Analysis pipeline combining sequence tagging (BiLSTM-CRF) with distance-decay attention classification (PA-BiLSTM).
        </p>
      </div>

      {/* Visual Pipeline Flow Overview */}
      <div className="glass-panel" style={{ padding: '2rem', marginBottom: '2.5rem' }}>
        <h3 style={{ fontSize: '1.25rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Layers size={18} color="#06b6d4" />
          Two-Stage Inference Workflow
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem', position: 'relative' }}>
          {/* Step 1: Raw Review */}
          <div className="glass-card" style={{ borderTop: '3px solid #38bdf8' }}>
            <div style={{ fontSize: '0.75rem', color: '#38bdf8', fontWeight: 700, marginBottom: '0.4rem', textTransform: 'uppercase' }}>
              Input Layer
            </div>
            <h4 style={{ fontSize: '1.05rem', marginBottom: '0.5rem' }}>Raw Product Review</h4>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
              Free-text customer sentence from SemEval-2014 Task 4 laptop domain.
            </p>
            <div style={{ padding: '0.5rem', backgroundColor: 'rgba(0,0,0,0.3)', borderRadius: '6px', fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: '#e2e8f0' }}>
              "The screen is gorgeous but the battery dies in two hours."
            </div>
          </div>

          {/* Step 2: Stage 1 ATE */}
          <div className="glass-card" style={{ borderTop: '3px solid #6366f1' }}>
            <div style={{ fontSize: '0.75rem', color: '#a5b4fc', fontWeight: 700, marginBottom: '0.4rem', textTransform: 'uppercase' }}>
              Stage 1: ATE Engine
            </div>
            <h4 style={{ fontSize: '1.05rem', marginBottom: '0.5rem' }}>BiLSTM-CRF Sequence Tagger</h4>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
              spaCy tokenization + POS (30d) + Word embeddings (100d) decoded via Linear-Chain CRF.
            </p>
            <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
              <span className="badge badge-cyan" style={{ fontSize: '0.7rem' }}>B-ASP: "screen"</span>
              <span className="badge badge-cyan" style={{ fontSize: '0.7rem' }}>B-ASP: "battery"</span>
            </div>
          </div>

          {/* Step 3: Context Window & Relative Distance */}
          <div className="glass-card" style={{ borderTop: '3px solid #f59e0b' }}>
            <div style={{ fontSize: '0.75rem', color: '#fbbf24', fontWeight: 700, marginBottom: '0.4rem', textTransform: 'uppercase' }}>
              Feature Engineering
            </div>
            <h4 style={{ fontSize: '1.05rem', marginBottom: '0.5rem' }}>Distance-Aware Window</h4>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
              Crops k=10 symmetric context window and maps relative distance: <code style={{ color: '#fbbf24' }}>d_i = i - mid_asp</code>.
            </p>
            <div style={{ padding: '0.5rem', backgroundColor: 'rgba(0,0,0,0.3)', borderRadius: '6px', fontSize: '0.75rem', color: '#fbbf24', fontFamily: 'var(--font-mono)' }}>
              penalty_i = -0.18 · |d_i|
            </div>
          </div>

          {/* Step 4: Stage 2 ASC */}
          <div className="glass-card" style={{ borderTop: '3px solid #10b981' }}>
            <div style={{ fontSize: '0.75rem', color: '#34d399', fontWeight: 700, marginBottom: '0.4rem', textTransform: 'uppercase' }}>
              Stage 2: ASC Engine
            </div>
            <h4 style={{ fontSize: '1.05rem', marginBottom: '0.5rem' }}>PA-BiLSTM-Attn Classifier</h4>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
              Frozen GloVe 100d + BiLSTM (128) + Distance-Decay Attention + Softmax.
            </p>
            <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
              <span className="badge badge-positive" style={{ fontSize: '0.7rem' }}>screen: POSITIVE</span>
              <span className="badge badge-negative" style={{ fontSize: '0.7rem' }}>battery: NEGATIVE</span>
            </div>
          </div>
        </div>
      </div>

      {/* Technical Diagnostics & Solutions (Addressing User Report) */}
      <div className="glass-panel" style={{ padding: '2rem', marginBottom: '2.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
          <div style={{
            width: '36px',
            height: '36px',
            borderRadius: '10px',
            backgroundColor: 'rgba(244, 63, 94, 0.15)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <ShieldAlert size={20} color="#f43f5e" />
          </div>
          <div>
            <h3 style={{ fontSize: '1.3rem' }}>
              Diagnostics: Why Baseline Models Failed & How We Fixed Them
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Detailed diagnosis and solutions based on our comprehensive model audit and retraining
            </p>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '1.5rem' }}>
          {/* Issue 1 */}
          <div className="glass-card" style={{ backgroundColor: 'rgba(244, 63, 94, 0.04)', borderColor: 'rgba(244, 63, 94, 0.2)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.6rem' }}>
              <AlertTriangle size={16} color="#fb7185" />
              <h4 style={{ fontSize: '0.95rem', color: '#fb7185' }}>Failure Mode 1: Cross-Clause Attention Bleed</h4>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '0.75rem' }}>
              <strong>Symptom:</strong> In contrastive sentences like <em>"The screen is gorgeous but the battery dies in two hours"</em>, the baseline marked <code>screen</code> as <strong>negative</strong>.
            </p>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '0.75rem' }}>
              <strong>Root Cause:</strong> The baseline dot-product attention computed global attention scores over the entire review without any distance penalty. A high-energy polarity word (<em>"dies"</em>) dominated attention weights regardless of where the aspect term was located.
            </p>
            <div style={{ padding: '0.65rem 0.85rem', backgroundColor: 'rgba(16, 185, 129, 0.08)', borderRadius: '8px', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#34d399', marginBottom: '0.2rem' }}>
                ✓ Our Solution: Distance-Decay Penalty
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                score_i = (h_i^T W v_a) / sqrt(d) - α · |i - mid_a|
              </div>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>
                With α=0.18, words further away from the aspect suffer an exponential attention drop, isolating sentiment to its own clause.
              </p>
            </div>
          </div>

          {/* Issue 2 */}
          <div className="glass-card" style={{ backgroundColor: 'rgba(245, 158, 11, 0.04)', borderColor: 'rgba(245, 158, 11, 0.2)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.6rem' }}>
              <AlertTriangle size={16} color="#fbbf24" />
              <h4 style={{ fontSize: '0.95rem', color: '#fbbf24' }}>Failure Mode 2: Aspect Frequency & Polarity Bias</h4>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '0.75rem' }}>
              <strong>Symptom:</strong> The baseline classified <em>"The build quality is terrible"</em> as <strong>positive</strong> with 71% confidence!
            </p>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '0.75rem' }}>
              <strong>Root Cause:</strong> Fine-tuning/unfreezing 100d GloVe embeddings on a small 1,622-pair dataset caused frequent aspect tokens like <em>"quality"</em> to absorb positive bias, drifting into positive space.
            </p>
            <div style={{ padding: '0.65rem 0.85rem', backgroundColor: 'rgba(16, 185, 129, 0.08)', borderRadius: '8px', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#34d399', marginBottom: '0.2rem' }}>
                ✓ Our Solution: Frozen GloVe + Modifiers
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                Embeddings are frozen (<code>requires_grad = False</code>). The model is strictly forced to learn sentiment from contextual adjectives and adverbs (<em>"terrible"</em>, <em>"gorgeous"</em>) rather than memorizing the aspect noun.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Hyperparameters & Specifications */}
      <div className="glass-panel" style={{ padding: '2rem' }}>
        <h3 style={{ fontSize: '1.25rem', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Sliders size={18} color="#06b6d4" />
          Technical Specifications & Hyperparameters
        </h3>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.88rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: 'var(--text-secondary)', fontSize: '0.78rem', textTransform: 'uppercase' }}>
                <th style={{ padding: '0.75rem 1rem' }}>Component</th>
                <th style={{ padding: '0.75rem 1rem' }}>Hyperparameter</th>
                <th style={{ padding: '0.75rem 1rem' }}>Value / Setting</th>
                <th style={{ padding: '0.75rem 1rem' }}>Design Rationale</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.04)' }}>
                <td style={{ padding: '0.75rem 1rem', fontWeight: 600, color: '#38bdf8' }}>Stage 1 (ATE)</td>
                <td style={{ padding: '0.75rem 1rem' }}>Word Embedding</td>
                <td style={{ padding: '0.75rem 1rem', fontFamily: 'var(--font-mono)' }}>100d (Learnable)</td>
                <td style={{ padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>Captures domain-specific laptop vocabulary</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.04)' }}>
                <td style={{ padding: '0.75rem 1rem', fontWeight: 600, color: '#38bdf8' }}>Stage 1 (ATE)</td>
                <td style={{ padding: '0.75rem 1rem' }}>POS Embedding</td>
                <td style={{ padding: '0.75rem 1rem', fontFamily: 'var(--font-mono)' }}>30d (Universal POS)</td>
                <td style={{ padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>Aspect terms are heavily concentrated in NOUN / PROPN tags</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.04)' }}>
                <td style={{ padding: '0.75rem 1rem', fontWeight: 600, color: '#38bdf8' }}>Stage 1 (ATE)</td>
                <td style={{ padding: '0.75rem 1rem' }}>Sequence Decoder</td>
                <td style={{ padding: '0.75rem 1rem', fontFamily: 'var(--font-mono)' }}>Linear-Chain CRF</td>
                <td style={{ padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>Prevents invalid transitions (e.g. O -&gt; I-ASP without B-ASP)</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.04)' }}>
                <td style={{ padding: '0.75rem 1rem', fontWeight: 600, color: '#34d399' }}>Stage 2 (ASC)</td>
                <td style={{ padding: '0.75rem 1rem' }}>Word Vectors</td>
                <td style={{ padding: '0.75rem 1rem', fontFamily: 'var(--font-mono)' }}>GloVe 6B 100d (Frozen)</td>
                <td style={{ padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>Generalizes out-of-domain words and eliminates aspect bias</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.04)' }}>
                <td style={{ padding: '0.75rem 1rem', fontWeight: 600, color: '#34d399' }}>Stage 2 (ASC)</td>
                <td style={{ padding: '0.75rem 1rem' }}>BiLSTM Hidden Dim</td>
                <td style={{ padding: '0.75rem 1rem', fontFamily: 'var(--font-mono)' }}>128 (Bidirectional = 256)</td>
                <td style={{ padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>Optimal capacity without overfitting on 1.6k training pairs</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.04)' }}>
                <td style={{ padding: '0.75rem 1rem', fontWeight: 600, color: '#34d399' }}>Stage 2 (ASC)</td>
                <td style={{ padding: '0.75rem 1rem' }}>Distance Decay α</td>
                <td style={{ padding: '0.75rem 1rem', fontFamily: 'var(--font-mono)' }}>0.18</td>
                <td style={{ padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>Calibrated penalty for spatial separation of clauses</td>
              </tr>
              <tr>
                <td style={{ padding: '0.75rem 1rem', fontWeight: 600, color: '#34d399' }}>Stage 2 (ASC)</td>
                <td style={{ padding: '0.75rem 1rem' }}>Loss Function</td>
                <td style={{ padding: '0.75rem 1rem', fontFamily: 'var(--font-mono)' }}>Class-Weighted CrossEntropy</td>
                <td style={{ padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>Counteracts class imbalance (Pos: 56.4%, Neg: 29.8%, Neu: 13.8%)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
