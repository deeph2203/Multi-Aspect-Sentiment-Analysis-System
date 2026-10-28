// frontend/src/pages/AnalyzerPage.jsx
import React, { useState } from 'react';
import { 
  Sparkles, 
  Send, 
  CheckCircle2, 
  XCircle, 
  MinusCircle, 
  Clock, 
  Sliders, 
  Eye, 
  Layers, 
  HelpCircle,
  TrendingUp,
  Cpu,
  Zap,
  ArrowRight
} from 'lucide-react';

const PRESETS = [
  {
    label: "Contrastive Mixed Sentiment",
    text: "The screen is gorgeous but the battery dies in two hours.",
    desc: "Screen: Positive | Battery: Negative (Key test for cross-clause attention leakage)"
  },
  {
    label: "Quality Bias Test",
    text: "The build quality is terrible, but the display is magnificent.",
    desc: "Build quality: Negative | Display: Positive (Tests frozen vs unfrozen aspect drift)"
  },
  {
    label: "Multi-Aspect Positive",
    text: "Great battery life and amazing screen quality.",
    desc: "Battery life: Positive | Screen quality: Positive"
  },
  {
    label: "Software vs Hardware",
    text: "Windows 8 is fast, but the keyboard feels cheap.",
    desc: "Windows 8: Positive | Keyboard: Negative"
  },
  {
    label: "Performance vs Support",
    text: "Great performance, but customer service was awful.",
    desc: "Performance: Positive | Customer service: Negative"
  },
  {
    label: "Single Negative",
    text: "The trackpad is unresponsive and frustrating.",
    desc: "Trackpad: Negative"
  }
];

export default function AnalyzerPage({ onAnalyze, loading, result, selectedModel, setSelectedModel }) {
  const [inputText, setInputText] = useState("The screen is gorgeous but the battery dies in two hours.");
  const [selectedAspect, setSelectedAspect] = useState(null);

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    if (!inputText.trim()) return;
    onAnalyze(inputText, selectedModel);
  };

  const handlePreset = (preset) => {
    setInputText(preset.text);
    onAnalyze(preset.text, selectedModel);
  };

  const getSentimentColor = (sentiment) => {
    if (sentiment === 'positive') return { text: '#34d399', bg: 'rgba(16, 185, 129, 0.12)', border: 'rgba(16, 185, 129, 0.35)', icon: CheckCircle2 };
    if (sentiment === 'negative') return { text: '#fb7185', bg: 'rgba(244, 63, 94, 0.12)', border: 'rgba(244, 63, 94, 0.35)', icon: XCircle };
    return { text: '#fbbf24', bg: 'rgba(245, 158, 11, 0.12)', border: 'rgba(245, 158, 11, 0.35)', icon: MinusCircle };
  };

  return (
    <div style={{ maxWidth: '1280px', margin: '0 auto', padding: '2rem 1.5rem' }}>
      {/* Hero Header */}
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
          <Sparkles size={14} />
          Two-Stage Deep Learning ABSA Engine
        </div>
        <h1 style={{ fontSize: '2.6rem', marginBottom: '0.75rem', lineHeight: 1.2 }}>
          Multi-Aspect <span className="gradient-text">Sentiment Analysis</span>
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.05rem', maxWidth: '720px', margin: '0 auto' }}>
          Extract product aspects via <strong>BiLSTM-CRF</strong> sequence tagging and classify granular sentiment via 
          <strong> Position-Aware Distance-Decay Attention</strong> without sentiment bleed across contrastive clauses.
        </p>
      </div>

      {/* Main Grid: Input & Model Config */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '1.75rem', marginBottom: '2.5rem' }}>
        {/* Model Switcher Box */}
        <div className="glass-panel" style={{ padding: '1.25rem 1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Cpu size={18} color="#06b6d4" />
              <span style={{ fontWeight: 600, fontSize: '0.95rem' }}>Active Classification Model</span>
            </div>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              Switch models to inspect contrastive sentiment behavior
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
            {/* Position Aware Card */}
            <div
              onClick={() => setSelectedModel('position_aware')}
              style={{
                padding: '1rem 1.25rem',
                borderRadius: '12px',
                cursor: 'pointer',
                border: selectedModel === 'position_aware' ? '1.5px solid #06b6d4' : '1px solid rgba(255, 255, 255, 0.08)',
                background: selectedModel === 'position_aware' ? 'rgba(6, 182, 212, 0.08)' : 'rgba(255, 255, 255, 0.02)',
                boxShadow: selectedModel === 'position_aware' ? '0 0 20px rgba(6, 182, 212, 0.15)' : 'none',
                transition: 'all 0.2s ease'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
                <span style={{ fontWeight: 700, color: '#38bdf8', fontSize: '0.95rem' }}>
                  Position-Aware BiLSTM-Attn
                </span>
                <span className="badge badge-cyan" style={{ fontSize: '0.65rem' }}>Recommended</span>
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                Applies relative distance decay <code style={{ color: '#38bdf8' }}>-α·|i - mid|</code> to attention. Frozen GloVe embeddings eliminate aspect bias. Solves cross-clause sentiment bleed.
              </p>
              <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.6rem', fontSize: '0.75rem' }}>
                <span style={{ color: '#34d399' }}>✓ Accuracy: <strong>64.1%</strong></span>
                <span style={{ color: '#38bdf8' }}>✓ Macro-F1: <strong>0.617</strong></span>
                <span style={{ color: '#a5b4fc' }}>✓ Contrastive Pass: <strong>60%</strong></span>
              </div>
            </div>

            {/* Baseline Card */}
            <div
              onClick={() => setSelectedModel('original')}
              style={{
                padding: '1rem 1.25rem',
                borderRadius: '12px',
                cursor: 'pointer',
                border: selectedModel === 'original' ? '1.5px solid #6366f1' : '1px solid rgba(255, 255, 255, 0.08)',
                background: selectedModel === 'original' ? 'rgba(99, 102, 241, 0.08)' : 'rgba(255, 255, 255, 0.02)',
                boxShadow: selectedModel === 'original' ? '0 0 20px rgba(99, 102, 241, 0.15)' : 'none',
                transition: 'all 0.2s ease'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
                <span style={{ fontWeight: 700, color: '#a5b4fc', fontSize: '0.95rem' }}>
                  Baseline BiLSTM-Attn
                </span>
                <span className="badge badge-indigo" style={{ fontSize: '0.65rem' }}>Baseline</span>
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                Classic dot-product attention without distance decay. Suffers attention leakage when opposite polarities exist in different clauses.
              </p>
              <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.6rem', fontSize: '0.75rem' }}>
                <span style={{ color: 'var(--text-muted)' }}>Accuracy: 60.4%</span>
                <span style={{ color: 'var(--text-muted)' }}>Macro-F1: 0.565</span>
                <span style={{ color: '#fb7185' }}>✗ Contrastive Pass: 0%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Input Box */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <form onSubmit={handleSubmit}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.65rem' }}>
              <label style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Send size={14} color="#06b6d4" />
                Input Customer Review
              </label>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                {inputText.length} characters
              </span>
            </div>

            <div style={{ position: 'relative' }}>
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Enter a laptop review (e.g., 'The screen is crisp and bright, but the battery life is disappointingly short...')"
                rows={3}
                style={{
                  width: '100%',
                  padding: '1rem 1.25rem',
                  borderRadius: '12px',
                  backgroundColor: 'rgba(7, 10, 18, 0.7)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  color: '#ffffff',
                  fontSize: '1rem',
                  fontFamily: 'var(--font-sans)',
                  lineHeight: 1.5,
                  outline: 'none',
                  resize: 'vertical',
                  transition: 'border-color 0.2s, box-shadow 0.2s',
                  boxShadow: 'inset 0 2px 4px rgba(0, 0, 0, 0.4)'
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = 'rgba(6, 182, 212, 0.6)';
                  e.target.style.boxShadow = '0 0 15px rgba(6, 182, 212, 0.2), inset 0 2px 4px rgba(0, 0, 0, 0.4)';
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = 'rgba(255, 255, 255, 0.1)';
                  e.target.style.boxShadow = 'inset 0 2px 4px rgba(0, 0, 0, 0.4)';
                }}
              />
            </div>

            {/* Presets Row */}
            <div style={{ marginTop: '1rem', marginBottom: '1.25rem' }}>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.5rem', fontWeight: 500 }}>
                Try benchmark test sentences:
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                {PRESETS.map((p, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() => handlePreset(p)}
                    className="btn btn-secondary"
                    style={{
                      fontSize: '0.78rem',
                      padding: '0.35rem 0.75rem',
                      borderRadius: '8px'
                    }}
                    title={p.desc}
                  >
                    <span>{p.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Submit Button */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '1rem' }}>
              <button
                type="submit"
                disabled={loading || !inputText.trim()}
                className="btn btn-primary"
                style={{
                  minWidth: '180px',
                  padding: '0.8rem 1.5rem',
                  fontSize: '0.95rem',
                  opacity: loading ? 0.7 : 1
                }}
              >
                {loading ? (
                  <>
                    <Zap size={16} className="animate-spin" />
                    <span>Analyzing Aspect Terms...</span>
                  </>
                ) : (
                  <>
                    <Sparkles size={16} />
                    <span>Analyze Aspects</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Results Section */}
      {result && (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
          {/* Sentence Summary Header */}
          <div className="glass-panel" style={{ padding: '1.25rem 1.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <div style={{
                  width: '36px',
                  height: '36px',
                  borderRadius: '10px',
                  backgroundColor: 'rgba(6, 182, 212, 0.15)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <Layers size={18} color="#06b6d4" />
                </div>
                <div>
                  <div style={{ fontWeight: 700, fontSize: '1.1rem' }}>
                    Extraction Summary
                  </div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    Evaluated with <strong>{result.model_name}</strong>
                  </div>
                </div>
              </div>

              {/* Status Pills */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                <span className="badge badge-cyan" style={{ fontSize: '0.8rem', padding: '0.35rem 0.75rem' }}>
                  {result.summary.total_aspects} {result.summary.total_aspects === 1 ? 'Aspect' : 'Aspects'} Found
                </span>

                {result.summary.positive > 0 && (
                  <span className="badge badge-positive" style={{ fontSize: '0.8rem', padding: '0.35rem 0.75rem' }}>
                    +{result.summary.positive} Positive
                  </span>
                )}
                {result.summary.negative > 0 && (
                  <span className="badge badge-negative" style={{ fontSize: '0.8rem', padding: '0.35rem 0.75rem' }}>
                    -{result.summary.negative} Negative
                  </span>
                )}
                {result.summary.neutral > 0 && (
                  <span className="badge badge-neutral" style={{ fontSize: '0.8rem', padding: '0.35rem 0.75rem' }}>
                    ~{result.summary.neutral} Neutral
                  </span>
                )}

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: 'var(--text-muted)', fontSize: '0.8rem', marginLeft: '0.5rem' }}>
                  <Clock size={14} />
                  <span>{result.elapsed_ms} ms</span>
                </div>
              </div>
            </div>

            {/* Interactive Sentence Highlighter */}
            <div style={{
              marginTop: '1.25rem',
              padding: '1.1rem 1.25rem',
              backgroundColor: 'rgba(0, 0, 0, 0.35)',
              borderRadius: '12px',
              border: '1px solid rgba(255, 255, 255, 0.06)',
              fontSize: '1.05rem',
              lineHeight: 1.8
            }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Sentence Token Grounding (Click aspect to highlight card)
              </div>
              <div style={{ wordBreak: 'break-word' }}>
                {renderHighlightedSentence(result.text, result.aspects, selectedAspect, setSelectedAspect)}
              </div>
            </div>
          </div>

          {/* Aspect Cards Grid */}
          <div>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <TrendingUp size={18} color="#06b6d4" />
              Aspect-Level Sentiment Classifications
            </h3>

            {result.aspects.length === 0 ? (
              <div className="glass-panel" style={{ padding: '2.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                <p style={{ fontSize: '1rem', marginBottom: '0.5rem' }}>No explicit aspect terms were detected in this review.</p>
                <p style={{ fontSize: '0.85rem' }}>Try mentioning specific hardware or software features (e.g. "battery", "display", "trackpad", "OS", "price").</p>
              </div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '1.25rem' }}>
                {result.aspects.map((asp, idx) => {
                  const sColor = getSentimentColor(asp.sentiment);
                  const Icon = sColor.icon;
                  const isSelected = selectedAspect === asp.aspect;

                  return (
                    <div
                      key={idx}
                      className="glass-card"
                      style={{
                        borderColor: isSelected ? '#06b6d4' : sColor.border,
                        boxShadow: isSelected ? '0 0 25px rgba(6, 182, 212, 0.25)' : 'none',
                        transition: 'all 0.25s ease'
                      }}
                      onMouseEnter={() => setSelectedAspect(asp.aspect)}
                      onMouseLeave={() => setSelectedAspect(null)}
                    >
                      {/* Card Header */}
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.85rem' }}>
                        <div>
                          <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.15rem' }}>
                            "{asp.aspect}"
                          </div>
                          <span style={{
                            fontSize: '0.72rem',
                            padding: '0.15rem 0.5rem',
                            borderRadius: '6px',
                            backgroundColor: 'rgba(255, 255, 255, 0.08)',
                            color: 'var(--text-secondary)',
                            textTransform: 'capitalize'
                          }}>
                            {asp.category}
                          </span>
                        </div>

                        {/* Sentiment Badge */}
                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.35rem',
                          padding: '0.35rem 0.75rem',
                          borderRadius: '9999px',
                          backgroundColor: sColor.bg,
                          border: `1px solid ${sColor.border}`,
                          color: sColor.text,
                          fontWeight: 700,
                          fontSize: '0.85rem',
                          textTransform: 'uppercase'
                        }}>
                          <Icon size={16} />
                          <span>{asp.sentiment}</span>
                        </div>
                      </div>

                      {/* Confidence Meter */}
                      <div style={{ marginBottom: '1rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.3rem' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>Model Confidence</span>
                          <span style={{ fontWeight: 700, color: sColor.text }}>
                            {(asp.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div style={{
                          height: '6px',
                          backgroundColor: 'rgba(255, 255, 255, 0.08)',
                          borderRadius: '9999px',
                          overflow: 'hidden'
                        }}>
                          <div style={{
                            height: '100%',
                            width: `${asp.confidence * 100}%`,
                            backgroundColor: sColor.text,
                            borderRadius: '9999px',
                            transition: 'width 0.5s cubic-bezier(0.16, 1, 0.3, 1)'
                          }} />
                        </div>
                      </div>

                      {/* Probability Distribution */}
                      <div style={{
                        padding: '0.75rem',
                        backgroundColor: 'rgba(0, 0, 0, 0.25)',
                        borderRadius: '8px',
                        marginBottom: '1rem',
                        fontSize: '0.75rem'
                      }}>
                        <div style={{ color: 'var(--text-muted)', marginBottom: '0.4rem', fontWeight: 600 }}>
                          Class Probability Distribution:
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                          {['positive', 'neutral', 'negative'].map((cat) => {
                            const p = asp.probabilities[cat] || 0;
                            const barColor = cat === 'positive' ? '#10b981' : cat === 'negative' ? '#f43f5e' : '#f59e0b';
                            return (
                              <div key={cat} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <span style={{ width: '55px', textTransform: 'capitalize', color: 'var(--text-secondary)' }}>
                                  {cat}:
                                </span>
                                <div style={{ flex: 1, height: '4px', backgroundColor: 'rgba(255, 255, 255, 0.06)', borderRadius: '2px', overflow: 'hidden' }}>
                                  <div style={{ width: `${p * 100}%`, height: '100%', backgroundColor: barColor }} />
                                </div>
                                <span style={{ width: '38px', textAlign: 'right', fontWeight: 600, color: barColor }}>
                                  {(p * 100).toFixed(1)}%
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </div>

                      {/* Attended Context Keywords */}
                      {asp.key_words && asp.key_words.length > 0 && (
                        <div>
                          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '0.35rem', fontWeight: 600 }}>
                            Attended Context Keywords (Attention Peaks):
                          </div>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
                            {asp.key_words.map((kw, ki) => (
                              <span
                                key={ki}
                                style={{
                                  fontSize: '0.75rem',
                                  padding: '0.2rem 0.5rem',
                                  borderRadius: '6px',
                                  backgroundColor: 'rgba(6, 182, 212, 0.1)',
                                  border: '1px solid rgba(6, 182, 212, 0.25)',
                                  color: '#38bdf8',
                                  fontFamily: 'var(--font-mono)'
                                }}
                              >
                                {kw}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Helper: Highlights aspect terms in sentence text with clickable sentiment pills
function renderHighlightedSentence(text, aspects, selectedAspect, setSelectedAspect) {
  if (!aspects || aspects.length === 0) return <span>{text}</span>;

  // Sort aspects by start_char
  const sorted = [...aspects].sort((a, b) => a.start_char - b.start_char);
  const elements = [];
  let lastIdx = 0;

  sorted.forEach((asp, i) => {
    if (asp.start_char > lastIdx) {
      elements.push(<span key={`text-${i}`}>{text.slice(lastIdx, asp.start_char)}</span>);
    }

    const isPos = asp.sentiment === 'positive';
    const isNeg = asp.sentiment === 'negative';
    const bg = isPos ? 'rgba(16, 185, 129, 0.2)' : isNeg ? 'rgba(244, 63, 94, 0.2)' : 'rgba(245, 158, 11, 0.2)';
    const color = isPos ? '#34d399' : isNeg ? '#fb7185' : '#fbbf24';
    const border = isPos ? '#10b981' : isNeg ? '#f43f5e' : '#f59e0b';
    const isSelected = selectedAspect === asp.aspect;

    elements.push(
      <span
        key={`asp-${i}`}
        onClick={() => setSelectedAspect(isSelected ? null : asp.aspect)}
        style={{
          display: 'inline-block',
          backgroundColor: bg,
          color: color,
          border: `1.5px solid ${border}`,
          borderRadius: '6px',
          padding: '0.1rem 0.45rem',
          margin: '0 0.15rem',
          fontWeight: 700,
          cursor: 'pointer',
          boxShadow: isSelected ? `0 0 12px ${color}` : 'none',
          transform: isSelected ? 'scale(1.05)' : 'scale(1)',
          transition: 'all 0.15s ease'
        }}
        title={`Aspect: ${asp.aspect} | Sentiment: ${asp.sentiment} (${(asp.confidence * 100).toFixed(1)}%)`}
      >
        {text.slice(asp.start_char, asp.end_char)}
        <span style={{ fontSize: '0.65rem', marginLeft: '0.3rem', opacity: 0.85 }}>
          [{asp.sentiment}]
        </span>
      </span>
    );

    lastIdx = asp.end_char;
  });

  if (lastIdx < text.length) {
    elements.push(<span key="text-end">{text.slice(lastIdx)}</span>);
  }

  return elements;
}
