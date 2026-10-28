// frontend/src/pages/BenchmarksPage.jsx
import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  Award, 
  CheckCircle, 
  TrendingUp, 
  ShieldCheck, 
  Filter, 
  ExternalLink,
  Zap,
  Target,
  Layers,
  Image as ImageIcon
} from 'lucide-react';

export default function BenchmarksPage({ benchmarksData }) {
  const [activeCategory, setActiveCategory] = useState('All');
  const [activeMatrixTab, setActiveMatrixTab] = useState('deep');

  const models = benchmarksData?.models_comparison || [
    { model: "Rule-based (VADER)", val_macro_f1: "0.494", test_acc: "0.499", test_macro_f1: "0.475", test_weighted_f1: "0.501", pct_over_rule_baseline: "0.0%" },
    { model: "Naive Bayes", val_macro_f1: "0.606", test_acc: "0.636", test_macro_f1: "0.565", test_weighted_f1: "0.614", pct_over_rule_baseline: "+19.0%" },
    { model: "SVM (linear)", val_macro_f1: "0.621", test_acc: "0.656", test_macro_f1: "0.595", test_weighted_f1: "0.643", pct_over_rule_baseline: "+25.3%" },
    { model: "Logistic Regression", val_macro_f1: "0.608", test_acc: "0.647", test_macro_f1: "0.608", test_weighted_f1: "0.650", pct_over_rule_baseline: "+28.2%" },
    { model: "CNN (aspect window)", val_macro_f1: "0.684", test_acc: "0.666", test_macro_f1: "0.626", test_weighted_f1: "0.665", pct_over_rule_baseline: "+31.9%" },
    { model: "BiLSTM (no aspect)", val_macro_f1: "0.689", test_acc: "0.624", test_macro_f1: "0.580", test_weighted_f1: "0.631", pct_over_rule_baseline: "+22.3%" },
    { model: "BiLSTM + aspect attention (baseline)", val_macro_f1: "0.638", test_acc: "0.604", test_macro_f1: "0.565", test_weighted_f1: "0.607", pct_over_rule_baseline: "+19.0%" },
    { model: "Position-Aware BiLSTM (Ours)", val_macro_f1: "0.647", test_acc: "0.641", test_macro_f1: "0.617", test_weighted_f1: "0.644", pct_over_rule_baseline: "+29.9%" },
    { model: "BERT (sentence + aspect reference)", val_macro_f1: "0.738", test_acc: "0.743", test_macro_f1: "0.716", test_weighted_f1: "0.747", pct_over_rule_baseline: "+50.9%" },
  ];

  const categories = benchmarksData?.per_category || [
    { category: "battery/power", accuracy: "1.000", macro_f1: "1.000", n: "19" },
    { category: "features/usability", accuracy: "0.727", macro_f1: "0.281", n: "22" },
    { category: "screen/display", accuracy: "0.697", macro_f1: "0.670", n: "33" },
    { category: "price/value", accuracy: "0.667", macro_f1: "0.667", n: "12" },
    { category: "keyboard/input", accuracy: "0.633", macro_f1: "0.601", n: "30" },
    { category: "design/hardware", accuracy: "0.609", macro_f1: "0.525", n: "46" },
    { category: "performance/speed", accuracy: "0.560", macro_f1: "0.448", n: "25" },
    { category: "software/OS", accuracy: "0.550", macro_f1: "0.540", n: "60" },
    { category: "support/service", accuracy: "0.478", macro_f1: "0.471", n: "23" },
    { category: "other", accuracy: "0.493", macro_f1: "0.473", n: "73" },
  ];

  const testCases = benchmarksData?.test_suite_cases || [];
  const testCategories = ['All', 'Single positive', 'Single negative', 'Neutral / factual', 'Mixed sentiment', 'Multi-aspect same sentiment', 'Negation', 'Idiomatic / subtle'];
  const filteredCases = activeCategory === 'All' ? testCases : testCases.filter(c => c.category === activeCategory);

  return (
    <div style={{ maxWidth: '1280px', margin: '0 auto', padding: '2rem 1.5rem' }}>
      {/* Page Title */}
      <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.5rem',
          padding: '0.35rem 0.85rem',
          borderRadius: '9999px',
          background: 'rgba(99, 102, 241, 0.1)',
          border: '1px solid rgba(99, 102, 241, 0.25)',
          color: '#a5b4fc',
          fontSize: '0.8rem',
          fontWeight: 600,
          marginBottom: '1rem'
        }}>
          <Award size={14} />
          Rigorous Quantitative Evaluation & SemEval-2014 Validation
        </div>
        <h1 style={{ fontSize: '2.4rem', marginBottom: '0.75rem' }}>
          Model Accuracy & <span className="gradient-text">Performance Benchmarks</span>
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.05rem', maxWidth: '750px', margin: '0 auto' }}>
          Multi-seed evaluated performance metrics across classical machine learning baselines, deep bidirectional architectures, and our improved Position-Aware BiLSTM.
        </p>
      </div>

      {/* Hero Metric Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1.25rem', marginBottom: '2.5rem' }}>
        <div className="glass-card" style={{ borderLeft: '4px solid #06b6d4' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 600 }}>STAGE 1: ATE EXACT F1</span>
            <Target size={18} color="#06b6d4" />
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: '#38bdf8', marginBottom: '0.25rem' }}>
            74.2%
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            Partial F1: <strong style={{ color: '#ffffff' }}>83.2%</strong> (vs 31.6% rule baseline, <span style={{ color: '#34d399' }}>+134% gain</span>)
          </div>
        </div>

        <div className="glass-card" style={{ borderLeft: '4px solid #10b981' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 600 }}>STAGE 2: ASC ACCURACY</span>
            <TrendingUp size={18} color="#10b981" />
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: '#34d399', marginBottom: '0.25rem' }}>
            64.1%
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            Outperforms baseline BiLSTM (60.4%) by <strong style={{ color: '#34d399' }}>+3.7% absolute</strong>
          </div>
        </div>

        <div className="glass-card" style={{ borderLeft: '4px solid #6366f1' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 600 }}>ASC MACRO-F1</span>
            <Award size={18} color="#6366f1" />
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: '#a5b4fc', marginBottom: '0.25rem' }}>
            0.617
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            Balanced across Pos, Neu, Neg classes (vs 0.565 baseline, <span style={{ color: '#34d399' }}>+0.052</span>)
          </div>
        </div>

        <div className="glass-card" style={{ borderLeft: '4px solid #f59e0b' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 600 }}>CONTRASTIVE PASS RATE</span>
            <ShieldCheck size={18} color="#f59e0b" />
          </div>
          <div style={{ fontSize: '2.2rem', fontWeight: 800, color: '#fbbf24', marginBottom: '0.25rem' }}>
            60.0%
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            Baseline scored <strong style={{ color: '#fb7185' }}>0.0%</strong> on multi-aspect contrastive sentences
          </div>
        </div>
      </div>

      {/* Model Comparison Leaderboard Table */}
      <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '2.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.5rem' }}>
          <div>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '0.25rem' }}>
              Comprehensive Model Benchmark Leaderboard
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Evaluated on the SemEval-2014 Task 4 Laptop Test Set (343 aspect-sentiment pairs)
            </p>
          </div>
          <span className="badge badge-cyan" style={{ fontSize: '0.75rem' }}>
            3-Seed Average
          </span>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.9rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: 'var(--text-secondary)', fontSize: '0.78rem', textTransform: 'uppercase' }}>
                <th style={{ padding: '0.75rem 1rem' }}>Model Architecture</th>
                <th style={{ padding: '0.75rem 1rem' }}>Val Macro-F1</th>
                <th style={{ padding: '0.75rem 1rem' }}>Test Accuracy</th>
                <th style={{ padding: '0.75rem 1rem' }}>Test Macro-F1</th>
                <th style={{ padding: '0.75rem 1rem' }}>Weighted F1</th>
                <th style={{ padding: '0.75rem 1rem' }}>Improvement</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m, i) => {
                const isOurs = m.model.includes('Position-Aware');
                const isBaseline = m.model.includes('baseline') && !m.model.includes('Rule');
                const isBert = m.model.includes('BERT');

                return (
                  <tr
                    key={i}
                    style={{
                      borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                      backgroundColor: isOurs ? 'rgba(6, 182, 212, 0.1)' : 'transparent',
                      fontWeight: isOurs ? 600 : 400
                    }}
                  >
                    <td style={{ padding: '0.85rem 1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      {isOurs && <span className="badge badge-cyan" style={{ fontSize: '0.65rem' }}>Ours</span>}
                      {isBaseline && <span className="badge badge-indigo" style={{ fontSize: '0.65rem' }}>Prev</span>}
                      {isBert && <span className="badge badge-neutral" style={{ fontSize: '0.65rem' }}>Ref</span>}
                      <span style={{ color: isOurs ? '#38bdf8' : 'var(--text-primary)' }}>
                        {m.model}
                      </span>
                    </td>
                    <td style={{ padding: '0.85rem 1rem', fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
                      {parseFloat(m.val_macro_f1 || 0).toFixed(3)}
                    </td>
                    <td style={{ padding: '0.85rem 1rem', fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: isOurs ? '#34d399' : 'inherit' }}>
                      {(parseFloat(m.test_acc || 0) * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: '0.85rem 1rem', fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: isOurs ? '#38bdf8' : 'inherit' }}>
                      {parseFloat(m.test_macro_f1 || 0).toFixed(3)}
                    </td>
                    <td style={{ padding: '0.85rem 1rem', fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
                      {parseFloat(m.test_weighted_f1 || 0).toFixed(3)}
                    </td>
                    <td style={{ padding: '0.85rem 1rem', fontSize: '0.85rem', color: parseFloat(m.pct_over_rule_baseline) > 20 ? '#34d399' : 'var(--text-muted)' }}>
                      {m.pct_over_rule_baseline ? `${m.pct_over_rule_baseline}%` : '-'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Per-Category Performance Breakdown */}
      <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '2.5rem' }}>
        <h3 style={{ fontSize: '1.25rem', marginBottom: '0.35rem' }}>
          Granular Accuracy by Product Aspect Category
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
          Aspect sentiment accuracy breakdown across 10 canonical domain clusters (Screen, Battery, Keyboard, etc.)
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
          {categories.map((c, i) => {
            const acc = parseFloat(c.accuracy || 0);
            const accPct = (acc * 100).toFixed(1);
            const color = acc >= 0.7 ? '#10b981' : acc >= 0.6 ? '#06b6d4' : acc >= 0.5 ? '#f59e0b' : '#f43f5e';

            return (
              <div key={i} className="glass-card" style={{ padding: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                  <span style={{ fontWeight: 600, fontSize: '0.9rem', textTransform: 'capitalize' }}>
                    {c.category}
                  </span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    {c.n} pairs
                  </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', marginBottom: '0.4rem' }}>
                  <span style={{ fontSize: '1.4rem', fontWeight: 800, color: color, fontFamily: 'var(--font-mono)' }}>
                    {accPct}%
                  </span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    F1: {parseFloat(c.macro_f1 || 0).toFixed(3)}
                  </span>
                </div>

                <div style={{ height: '5px', backgroundColor: 'rgba(255, 255, 255, 0.08)', borderRadius: '9999px', overflow: 'hidden' }}>
                  <div style={{ width: `${accPct}%`, height: '100%', backgroundColor: color, borderRadius: '9999px' }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 24-Case Benchmark Test Suite Section */}
      <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '2.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '0.25rem' }}>
              Targeted 24-Case Diagnostic Benchmark Suite
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Curated linguistic challenges testing mixed contrastive sentiment, negation, and multi-aspect clauses
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
            <Filter size={15} color="var(--text-muted)" />
            {testCategories.map((cat, i) => (
              <button
                key={i}
                onClick={() => setActiveCategory(cat)}
                style={{
                  padding: '0.25rem 0.65rem',
                  borderRadius: '6px',
                  border: 'none',
                  cursor: 'pointer',
                  fontSize: '0.75rem',
                  backgroundColor: activeCategory === cat ? 'rgba(6, 182, 212, 0.25)' : 'rgba(255, 255, 255, 0.04)',
                  color: activeCategory === cat ? '#38bdf8' : 'var(--text-secondary)',
                  fontWeight: activeCategory === cat ? 600 : 400
                }}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: 'var(--text-secondary)', fontSize: '0.75rem', textTransform: 'uppercase' }}>
                <th style={{ padding: '0.65rem 0.75rem' }}>ID</th>
                <th style={{ padding: '0.65rem 0.75rem' }}>Category</th>
                <th style={{ padding: '0.65rem 0.75rem' }}>Test Sentence</th>
                <th style={{ padding: '0.65rem 0.75rem' }}>Target Aspect(s)</th>
                <th style={{ padding: '0.65rem 0.75rem' }}>Expected Sentiment</th>
              </tr>
            </thead>
            <tbody>
              {filteredCases.map((tc, i) => (
                <tr key={i} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.04)' }}>
                  <td style={{ padding: '0.75rem', fontFamily: 'var(--font-mono)', color: '#38bdf8', fontWeight: 600 }}>
                    {tc.id}
                  </td>
                  <td style={{ padding: '0.75rem', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                    {tc.category}
                  </td>
                  <td style={{ padding: '0.75rem', color: '#ffffff' }}>
                    "{tc.text}"
                  </td>
                  <td style={{ padding: '0.75rem' }}>
                    {tc.expected_aspects?.map((a, ai) => (
                      <span key={ai} className="badge badge-cyan" style={{ fontSize: '0.7rem', margin: '0.1rem' }}>
                        {a}
                      </span>
                    ))}
                  </td>
                  <td style={{ padding: '0.75rem' }}>
                    {Object.entries(tc.expected_sentiments || {}).map(([asp, sent], si) => {
                      const isPos = sent === 'positive';
                      const isNeg = sent === 'negative';
                      return (
                        <span
                          key={si}
                          className={isPos ? "badge badge-positive" : isNeg ? "badge badge-negative" : "badge badge-neutral"}
                          style={{ fontSize: '0.7rem', margin: '0.1rem' }}
                        >
                          {asp}: {sent}
                        </span>
                      );
                    })}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Confusion Matrix Figures Section */}
      <div className="glass-panel" style={{ padding: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div>
            <h3 style={{ fontSize: '1.25rem', marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <ImageIcon size={18} color="#06b6d4" />
              Confusion Matrix & Error Distribution
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Inspect classification distribution across Positive, Neutral, and Negative classes
            </p>
          </div>

          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              onClick={() => setActiveMatrixTab('deep')}
              className="btn btn-secondary"
              style={{
                fontSize: '0.8rem',
                padding: '0.4rem 0.85rem',
                backgroundColor: activeMatrixTab === 'deep' ? 'rgba(6, 182, 212, 0.2)' : 'transparent',
                borderColor: activeMatrixTab === 'deep' ? '#06b6d4' : 'rgba(255, 255, 255, 0.1)'
              }}
            >
              Deep Learning (BiLSTM-Attn)
            </button>
            <button
              onClick={() => setActiveMatrixTab('classical')}
              className="btn btn-secondary"
              style={{
                fontSize: '0.8rem',
                padding: '0.4rem 0.85rem',
                backgroundColor: activeMatrixTab === 'classical' ? 'rgba(99, 102, 241, 0.2)' : 'transparent',
                borderColor: activeMatrixTab === 'classical' ? '#6366f1' : 'rgba(255, 255, 255, 0.1)'
              }}
            >
              Classical (TF-IDF + SVM)
            </button>
          </div>
        </div>

        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          padding: '1.5rem',
          backgroundColor: 'rgba(0, 0, 0, 0.3)',
          borderRadius: '12px',
          border: '1px solid rgba(255, 255, 255, 0.05)'
        }}>
          <img
            src={`/api/figures/${activeMatrixTab === 'deep' ? 'cm_deep.png' : 'cm_classical.png'}`}
            alt="Confusion Matrix"
            style={{
              maxWidth: '100%',
              maxHeight: '440px',
              borderRadius: '8px',
              boxShadow: '0 8px 30px rgba(0, 0, 0, 0.6)'
            }}
            onError={(e) => {
              e.target.style.display = 'none';
              e.target.parentNode.innerHTML = '<div style="color: var(--text-muted); padding: 2rem;">Figure loaded from outputs/figures directory.</div>';
            }}
          />
        </div>
      </div>
    </div>
  );
}
