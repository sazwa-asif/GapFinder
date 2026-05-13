import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  Upload, Brain, TrendingUp, FileText,
  Download, Loader2, CheckCircle2, ChevronDown, ChevronUp,
  Sparkles, Target, Search, ArrowRight, Database, HelpCircle, Layers
} from 'lucide-react';
import type { Paper, ResearchGap } from './types';
import { jsPDF } from 'jspdf';

const CONFIDENCE_EXPLANATION =
  "Confidence score is computed from normalized semantic distance (0.30), " +
  "concept prominence (0.20), absence in target domain (0.20), " +
  "distinctiveness (0.10), evidence ratio (0.10), and gap magnitude (0.10).";

// Backend helpers

const extractTextFromBackend = async (file: File, domain: string): Promise<string> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('domain', domain);

  const res = await fetch(`/api/extract`, { method: 'POST', body: formData });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(`Text extraction failed for "${file.name}": ${err.detail}`);
  }
  const data = await res.json();
  if (!data.text || data.sentence_count === 0) {
    throw new Error(
      `No readable text found in "${file.name}". ` +
      `Make sure it is a text-based PDF (not a scanned image).`
    );
  }
  return data.text;
};

// ── Stage config ──────────────────────────────────────────────────────────────

type StageKey = 'extraction' | 'embedding' | 'clustering' | 'divergence' | 'dedup' | 'complete';

const STAGES: { key: StageKey; label: string; icon: React.ElementType; pctRange: [number, number] }[] = [
  { key: 'extraction',  label: 'Text Extraction',              icon: FileText,   pctRange: [0,  15] },
  { key: 'embedding',   label: 'Semantic Embedding',           icon: Brain,      pctRange: [15, 45] },
  { key: 'clustering',  label: 'Concept Discovery',            icon: TrendingUp, pctRange: [45, 65] },
  { key: 'divergence',  label: 'Cross-Domain Divergence',      icon: Search,     pctRange: [65, 85] },
  { key: 'dedup',       label: 'Gap Detection & Deduplication',icon: Layers,     pctRange: [85, 100] },
];

// ── App ───────────────────────────────────────────────────────────────────────

const App: React.FC = () => {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [gaps, setGaps] = useState<ResearchGap[]>([]);
  const [showAllGaps, setShowAllGaps] = useState(false);
  const [activeTab, setActiveTab] = useState<'upload' | 'analysis' | 'gaps'>('upload');
  const [expandedGaps, setExpandedGaps] = useState<Set<string>>(new Set());
  const [processingStage, setProcessingStage] = useState('');
  const [progressPct, setProgressPct] = useState(0);
  const [currentStageKey, setCurrentStageKey] = useState<StageKey | ''>('');
  const [analysisError, setAnalysisError] = useState<string>('');
  const [isDraggingA, setIsDraggingA] = useState(false);
  const [isDraggingB, setIsDraggingB] = useState(false);

  const analysisSectionRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (activeTab === 'analysis' && analysisSectionRef.current) {
      analysisSectionRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [activeTab]);

  const handleFileUpload = useCallback((files: FileList | null, domain: string) => {
    if (!files) return;
    const newPapers: Paper[] = Array.from(files).map(file => ({
      id: `${Date.now()}-${Math.random()}`,
      file,
      domain,
      status: 'pending'
    }));
    setPapers(prev => [...prev, ...newPapers]);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent, domain: string) => {
    e.preventDefault();
    e.stopPropagation();
    if (domain === 'Domain A') setIsDraggingA(false);
    if (domain === 'Domain B') setIsDraggingB(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) handleFileUpload(files, domain);
  }, [handleFileUpload]);

  const toggleExpandGap = useCallback((gapId: string) => {
    setExpandedGaps(prev => {
      const next = new Set(prev);
      next.has(gapId) ? next.delete(gapId) : next.add(gapId);
      return next;
    });
  }, []);

  const removePaper = useCallback((paperId: string) => {
    setPapers(prev => prev.filter(p => p.id !== paperId));
  }, []);

  const clearAllPapers = useCallback(() => setPapers([]), []);

  // ── Main analysis flow ──────────────────────────────────────────────────────

  const runAnalysis = async () => {
    if (papers.length < 2) {
      setAnalysisError('Please upload at least 2 papers from different domains');
      return;
    }
    const domains = new Set(papers.map(p => p.domain));
    if (domains.size < 2) {
      setAnalysisError('Please upload papers from at least 2 different domains');
      return;
    }

    setIsAnalyzing(true);
    setAnalysisError('');
    setActiveTab('analysis');
    setGaps([]);
    setShowAllGaps(false);
    setExpandedGaps(new Set());
    setProgressPct(0);
    setCurrentStageKey('extraction');

    try {
      // ── Step 1: Extract text from each PDF (parallel) ──────────────────────
      setProcessingStage('Extracting text from PDFs…');
      const papersWithText = await Promise.all(
        papers.map(async (paper) => {
          setPapers(prev => prev.map(p =>
            p.id === paper.id ? { ...p, status: 'processing' as const } : p
          ));
          const text = await extractTextFromBackend(paper.file, paper.domain);
          return { id: paper.id, domain: paper.domain, text };
        })
      );
      setProgressPct(15);

      // ── Step 2: Stream /api/analyze/stream via SSE ─────────────────────────
      setCurrentStageKey('embedding');
      setProcessingStage('Connecting to analysis pipeline…');

      const res = await fetch('/api/analyze/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(papersWithText),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Analysis failed');
      }

      // Read the SSE stream line by line
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let finalResult: any = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE lines look like:  data: {...}\n\n
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';   // keep incomplete last chunk

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const json = line.slice(6).trim();
          if (!json) continue;

          let event: any;
          try { event = JSON.parse(json); } catch { continue; }

          if (event.type === 'progress') {
            setProcessingStage(event.message);
            setProgressPct(event.pct ?? progressPct);
            // Map SSE stage key → our StageKey
            const stageMap: Record<string, StageKey> = {
              embedding:  'embedding',
              clustering: 'clustering',
              divergence: 'divergence',
              dedup:      'dedup',
              complete:   'complete',
            };
            if (event.stage && stageMap[event.stage]) {
              setCurrentStageKey(stageMap[event.stage]);
            }
          } else if (event.type === 'result') {
            finalResult = event;
          } else if (event.type === 'error') {
            throw new Error(event.message);
          }
        }
      }

      if (!finalResult) throw new Error('No result received from analysis pipeline.');
      if (!finalResult.gaps || finalResult.gaps.length === 0) {
        throw new Error(
          'No research gaps were detected. This can happen if the papers are too similar, ' +
          'too short, or HDBSCAN could not form enough clusters. ' +
          `Sentences processed: ${finalResult.papers?.map((p: any) => p.num_sentences).join(', ')}.`
        );
      }

      // ── Step 3: Map backend snake_case → frontend camelCase ────────────────
      const mappedGaps: ResearchGap[] = finalResult.gaps.map((gap: any) => ({
        id: gap.id,
        title: gap.title,
        confidence: gap.confidence,
        gapStrength: gap.gap_strength ?? gap.confidence,
        gapStrengthLabel: gap.gap_strength_label ?? 'Exploratory Signal',
        noveltyScore: gap.novelty_score ?? 0.5,
        compatibilityScore: gap.compatibility_score,
        confidenceMin: gap.confidence_min ?? gap.confidence,
        confidenceMax: gap.confidence_max ?? gap.confidence,
        confidenceRange: gap.confidence_range ?? `${gap.confidence.toFixed(2)}–${gap.confidence.toFixed(2)}`,
        clusterCount: gap.cluster_count ?? 1,
        missingIn: gap.missing_in,
        presentIn: gap.present_in,
        explanation: gap.explanation,
        semanticDistance: gap.semantic_distance,
        semanticDistanceMin: gap.semantic_distance_min ?? gap.semantic_distance,
        semanticDistanceMax: gap.semantic_distance_max ?? gap.semantic_distance,
        semanticDistanceRange: gap.semantic_distance_range ?? gap.semantic_distance.toFixed(3),
        evidenceSentences: (gap.evidence_sentences || []).map((e: any) => ({
          domain: e.domain,
          sentences: e.sentences,
        })),
        conceptOverlap: {
          shared: gap.concept_overlap?.shared ?? [],
          uniqueToA: gap.concept_overlap?.unique_a ?? [],
          uniqueToB: gap.concept_overlap?.unique_b ?? [],
        },
        futureSuggestion: gap.future_suggestion,
        whyDetected: gap.why_detected,
      }));

      setPapers(prev => prev.map(p => ({ ...p, status: 'completed' as const })));
      setGaps(mappedGaps);
      setActiveTab('gaps');
      setProcessingStage('Analysis complete!');
      setProgressPct(100);
      setCurrentStageKey('complete');

    } catch (error: any) {
      console.error('Analysis failed:', error);
      setAnalysisError(`Analysis failed: ${error.message || 'Unknown error'}`);
      setProcessingStage('');
      setProgressPct(0);
      setPapers(prev => prev.map(p => ({ ...p, status: 'pending' as const })));
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ── Export ──────────────────────────────────────────────────────────────────

  const exportResults = () => {
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    const margin = 20;
    const maxWidth = pageWidth - 2 * margin;
    let yPos = margin;

    const checkNewPage = (space: number) => {
      if (yPos + space > doc.internal.pageSize.getHeight() - margin) {
        doc.addPage(); yPos = margin;
      }
    };

    doc.setFontSize(24); doc.setTextColor(79, 70, 229);
    doc.text('GapFinder', margin, yPos); yPos += 10;
    doc.setFontSize(12); doc.setTextColor(100, 116, 139);
    doc.text('Research Gap Analysis Report', margin, yPos); yPos += 15;
    doc.setDrawColor(229, 231, 235);
    doc.line(margin, yPos, pageWidth - margin, yPos); yPos += 15;
    doc.setFontSize(10); doc.setTextColor(100, 116, 139);
    doc.text(`Generated: ${new Date().toLocaleDateString()}`, margin, yPos); yPos += 8;
    doc.text(`Total Unique Gaps: ${gaps.length}`, margin, yPos); yPos += 15;
    doc.setFontSize(14); doc.setTextColor(31, 41, 55);
    doc.text('Detected Research Gaps', margin, yPos); yPos += 10;

    const exportGaps = showAllGaps ? gaps : gaps.slice(0, 5);
    exportGaps.forEach((gap, index) => {
      checkNewPage(60);
      doc.setFontSize(12); doc.setTextColor(79, 70, 229);
      doc.text(`${index + 1}. ${gap.title}`, margin, yPos); yPos += 8;
      doc.setFontSize(9); doc.setTextColor(100, 116, 139);
      doc.text(
        `Confidence: ${gap.confidenceRange || `${Math.round(gap.confidence * 100)}%`} | Clusters: ${gap.clusterCount || 1}`,
        margin, yPos
      ); yPos += 6;
      doc.setTextColor(75, 85, 99);
      const expl = gap.explanation.length > 150 ? gap.explanation.substring(0, 150) + '...' : gap.explanation;
      const explLines = doc.splitTextToSize(expl, maxWidth);
      doc.text(explLines, margin, yPos); yPos += explLines.length * 5 + 4;
      doc.text(`Missing in: ${gap.missingIn} | Present in: ${gap.presentIn}`, margin, yPos); yPos += 15;
      doc.setDrawColor(229, 231, 235);
      doc.line(margin, yPos - 5, pageWidth - margin, yPos - 5);
    });

    const pageCount = (doc.internal as any).pages.length - 1;
    for (let i = 1; i <= pageCount; i++) {
      doc.setPage(i);
      doc.setFontSize(8); doc.setTextColor(156, 163, 175);
      doc.text(`Page ${i} of ${pageCount}`, pageWidth / 2, doc.internal.pageSize.getHeight() - 10, { align: 'center' });
    }
    doc.save('research-gaps-report.pdf');
  };

  // ── Render ──────────────────────────────────────────────────────────────────

  const displayGaps = showAllGaps ? gaps : gaps.slice(0, 5);

  // Determine which stages are done based on currentStageKey
  const stageOrder: StageKey[] = ['extraction', 'embedding', 'clustering', 'divergence', 'dedup', 'complete'];
  const currentStageIdx = stageOrder.indexOf(currentStageKey as StageKey);
  const isStageComplete = (key: StageKey) =>
    currentStageIdx > stageOrder.indexOf(key);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">

      {/* Navigation */}
      <nav className="border-b border-slate-800 bg-slate-950/95 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-5">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-xl blur-lg opacity-40"></div>
                <div className="relative p-3.5 bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 rounded-xl">
                  <Brain className="w-6 h-6 text-white" />
                </div>
              </div>
              <div className="border-l border-slate-700 pl-5">
                <h1 className="text-2xl font-bold text-white tracking-tight">GapFinder</h1>
                <p className="text-xs text-slate-400 font-medium tracking-wide uppercase mt-0.5">
                  AI-Driven Cross-Domain Research Gap Analysis
                </p>
              </div>
            </div>
            <div className="hidden md:flex items-center gap-2 text-sm text-slate-400">
              <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span>
              <span>Ready for Analysis</span>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-6 py-4">

        {/* Tab Navigation */}
        <div className="flex gap-2 mb-8">
          {(['upload', 'analysis', 'gaps'] as const).map((tab) => (
            <button key={tab} onClick={() => setActiveTab(tab)}
              className={`relative px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                activeTab === tab ? 'text-white' : 'text-slate-400 hover:text-white'
              }`}>
              {activeTab === tab && (
                <div className="absolute inset-0 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl"></div>
              )}
              <div className="relative flex items-center gap-2">
                {tab === 'upload'   && <Upload className="w-4 h-4" />}
                {tab === 'analysis' && <Search className="w-4 h-4" />}
                {tab === 'gaps'     && <Target className="w-4 h-4" />}
                <span className="capitalize">
                  {tab === 'upload'   ? 'Upload Papers' :
                   tab === 'analysis' ? 'AI Processing' :
                   `Research Gaps (${gaps.length})`}
                </span>
              </div>
            </button>
          ))}
        </div>

        {/* ── Upload Tab ── */}
        {activeTab === 'upload' && (
          <div className="space-y-6">
            <div className="relative overflow-hidden bg-gradient-to-br from-indigo-600/20 via-purple-600/10 to-slate-900 rounded-3xl border border-slate-800 p-8">
              <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-indigo-500/20 to-purple-500/20 rounded-full blur-3xl"></div>
              <div className="relative">
                <h2 className="text-3xl font-bold text-white mb-3">Upload Research Papers</h2>
                <p className="text-slate-400 max-w-xl">
                  Upload PDFs from two different research domains. The AI pipeline will extract text,
                  generate embeddings, cluster concepts, and detect cross-domain gaps.
                </p>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              {/* Domain A */}
              <div
                className={`group relative overflow-hidden bg-gradient-to-br from-slate-900 to-slate-800/50 rounded-2xl border transition-all duration-300 p-8 ${
                  isDraggingA ? 'border-indigo-500 bg-indigo-500/10' : 'border-slate-700/50 hover:border-indigo-500/50'
                }`}
                onDragOver={(e) => { e.preventDefault(); setIsDraggingA(true); }}
                onDragLeave={() => setIsDraggingA(false)}
                onDrop={(e) => handleDrop(e, 'Domain A')}
              >
                <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <div className="relative">
                  <div className="w-14 h-14 bg-gradient-to-br from-indigo-500 to-indigo-600 rounded-2xl flex items-center justify-center mb-6">
                    <Upload className="w-7 h-7 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">Domain A Papers</h3>
                  <p className="text-slate-400 text-sm mb-6">Drag & drop PDFs here or click to browse</p>
                  <input type="file" id="domain-a" multiple accept=".pdf" className="hidden"
                    onChange={(e) => handleFileUpload(e.target.files, 'Domain A')} />
                  <label htmlFor="domain-a"
                    className="inline-flex items-center gap-2 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-medium transition-colors cursor-pointer">
                    <FileText className="w-4 h-4" />Select PDFs
                  </label>
                </div>
              </div>

              {/* Domain B */}
              <div
                className={`group relative overflow-hidden bg-gradient-to-br from-slate-900 to-slate-800/50 rounded-2xl border transition-all duration-300 p-8 ${
                  isDraggingB ? 'border-purple-500 bg-purple-500/10' : 'border-slate-700/50 hover:border-purple-500/50'
                }`}
                onDragOver={(e) => { e.preventDefault(); setIsDraggingB(true); }}
                onDragLeave={() => setIsDraggingB(false)}
                onDrop={(e) => handleDrop(e, 'Domain B')}
              >
                <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <div className="relative">
                  <div className="w-14 h-14 bg-gradient-to-br from-purple-500 to-purple-600 rounded-2xl flex items-center justify-center mb-6">
                    <Upload className="w-7 h-7 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">Domain B Papers</h3>
                  <p className="text-slate-400 text-sm mb-6">Drag & drop PDFs here or click to browse</p>
                  <input type="file" id="domain-b" multiple accept=".pdf" className="hidden"
                    onChange={(e) => handleFileUpload(e.target.files, 'Domain B')} />
                  <label htmlFor="domain-b"
                    className="inline-flex items-center gap-2 px-5 py-2.5 bg-purple-600 hover:bg-purple-500 text-white rounded-lg font-medium transition-colors cursor-pointer">
                    <FileText className="w-4 h-4" />Select PDFs
                  </label>
                </div>
              </div>
            </div>

            {/* Paper list */}
            {papers.length > 0 && (
              <div className="bg-slate-900/50 rounded-2xl border border-slate-800 p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                    <Database className="w-5 h-5 text-indigo-400" />Uploaded Papers ({papers.length})
                  </h3>
                  <button onClick={clearAllPapers}
                    className="flex items-center gap-2 px-4 py-2 bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 rounded-lg text-red-400 transition-colors">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                    Clear All
                  </button>
                </div>
                <div className="space-y-3">
                  {papers.map((paper) => (
                    <div key={paper.id} className="flex items-center justify-between p-4 bg-slate-800/50 rounded-xl">
                      <div className="flex items-center gap-4">
                        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                          paper.status === 'completed' ? 'bg-green-500/20' :
                          paper.status === 'processing' ? 'bg-indigo-500/20' : 'bg-slate-700'
                        }`}>
                          <FileText className={`w-5 h-5 ${
                            paper.status === 'completed' ? 'text-green-400' :
                            paper.status === 'processing' ? 'text-indigo-400' : 'text-slate-400'
                          }`} />
                        </div>
                        <div>
                          <p className="text-white font-medium">{paper.file.name}</p>
                          <p className="text-sm text-slate-500">{paper.domain}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className={`text-xs px-3 py-1 rounded-full ${
                          paper.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                          paper.status === 'processing' ? 'bg-indigo-500/20 text-indigo-400' :
                          'bg-slate-700 text-slate-400'
                        }`}>{paper.status}</span>
                        <button onClick={() => removePaper(paper.id)}
                          className="p-2 hover:bg-red-500/20 rounded-lg text-slate-400 hover:text-red-400 transition-colors">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Run button */}
            <button onClick={runAnalysis} disabled={papers.length < 2 || isAnalyzing} className="w-full relative group">
              <div className={`absolute inset-0 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl blur-lg opacity-50 group-hover:opacity-75 transition-opacity ${papers.length < 2 || isAnalyzing ? 'opacity-25' : ''}`}></div>
              <div className={`relative px-8 py-5 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl font-semibold text-white flex items-center justify-center gap-3 transition-all ${papers.length < 2 || isAnalyzing ? 'opacity-50 cursor-not-allowed' : 'hover:from-indigo-500 hover:to-purple-500'}`}>
                {isAnalyzing
                  ? <><Loader2 className="w-5 h-5 animate-spin" />Running AI Analysis…</>
                  : <><Sparkles className="w-5 h-5" />Run AI Gap Analysis<ArrowRight className="w-5 h-5" /></>
                }
              </div>
            </button>
          </div>
        )}

        {/* ── Analysis Tab ── */}
        {activeTab === 'analysis' && (
          <div ref={analysisSectionRef} className="bg-gradient-to-br from-slate-900 to-slate-800/50 rounded-3xl border border-slate-800 p-8">
            <h2 className="text-2xl font-bold text-white mb-8">AI Processing Pipeline</h2>

            <div className="space-y-6">
              {STAGES.map(({ key, label, icon: Icon }, i) => {
                const done    = isStageComplete(key);
                const active  = currentStageKey === key && isAnalyzing;
                return (
                  <div key={key} className="flex items-center gap-6">
                    <div className={`relative w-16 h-16 rounded-2xl flex items-center justify-center transition-all duration-500 ${
                      done   ? 'bg-gradient-to-br from-green-500/20 to-emerald-500/20 border border-green-500/30' :
                      active ? 'bg-gradient-to-br from-indigo-500/20 to-purple-500/20 border border-indigo-500/30 animate-pulse' :
                               'bg-slate-800 border border-slate-700'
                    }`}>
                      {done
                        ? <CheckCircle2 className="w-8 h-8 text-green-400" />
                        : active
                          ? <Loader2 className="w-8 h-8 text-indigo-400 animate-spin" />
                          : <Icon className="w-8 h-8 text-slate-600" />
                      }
                      {i < STAGES.length - 1 && (
                        <div className={`absolute -right-3 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                          done ? 'bg-green-500 text-white' : 'bg-slate-700 text-slate-400'
                        }`}>{i + 1}</div>
                      )}
                    </div>
                    <div className="flex-1">
                      <p className={`font-medium text-lg transition-colors ${done || active ? 'text-white' : 'text-slate-500'}`}>{label}</p>
                      <p className={`text-sm transition-colors ${
                        done   ? 'text-green-400' :
                        active ? 'text-indigo-400' :
                                 'text-slate-600'
                      }`}>
                        {done ? 'Completed successfully' : active ? 'Processing…' : 'Pending'}
                      </p>
                    </div>
                    {i < STAGES.length - 1 && (
                      <div className={`w-12 h-0.5 transition-colors ${done ? 'bg-gradient-to-r from-green-500 to-transparent' : 'bg-slate-800'}`}></div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Progress bar + live message */}
            {isAnalyzing && (
              <div className="mt-8 p-6 bg-indigo-500/10 border border-indigo-500/30 rounded-2xl">
                <div className="flex items-center gap-4 mb-4">
                  <div className="relative">
                    <Loader2 className="w-8 h-8 text-indigo-400 animate-spin" />
                    <div className="absolute inset-0 bg-indigo-400 blur-lg opacity-30 animate-pulse"></div>
                  </div>
                  <div>
                    <p className="text-indigo-400 font-medium">{processingStage}</p>
                    <p className="text-sm text-slate-500">{progressPct}% complete</p>
                  </div>
                </div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-500 ease-out"
                    style={{ width: `${progressPct}%` }}
                  />
                </div>
              </div>
            )}

            {analysisError && (
              <div className="mt-8 p-6 bg-red-500/10 border border-red-500/30 rounded-2xl">
                <div className="flex items-start gap-4">
                  <svg className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                  <p className="text-red-400 font-medium">{analysisError}</p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Gaps Tab ── */}
        {activeTab === 'gaps' && (
          <div className="space-y-6">
            {gaps.length === 0 ? (
              <div className="bg-gradient-to-br from-slate-900 to-slate-800/50 rounded-3xl border border-slate-800 p-16 text-center">
                <div className="w-20 h-20 bg-slate-800 rounded-2xl flex items-center justify-center mx-auto mb-6">
                  <Search className="w-10 h-10 text-slate-600" />
                </div>
                <p className="text-slate-400 text-lg">No gaps detected yet. Run the analysis first.</p>
              </div>
            ) : (
              <>
                <div className="flex justify-between items-start">
                  <div>
                    <h2 className="text-2xl font-bold text-white">Detected Research Gaps</h2>
                    <p className="text-slate-400 mt-1">{gaps.length} unique gaps identified</p>
                  </div>
                  <div className="flex items-center gap-4">
                    <button
                      onClick={() => {
                        if (window.confirm('This will reset all papers and analysis. Continue?')) {
                          setActiveTab('upload'); setGaps([]); setPapers([]); setExpandedGaps(new Set());
                        }
                      }}
                      className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-slate-400 hover:text-white transition-colors">
                      <Upload className="w-4 h-4" />Back to Upload
                    </button>
                    <div className="group relative">
                      <button className="flex items-center gap-2 px-3 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-400 transition-colors">
                        <HelpCircle className="w-4 h-4" /><span className="text-sm">Confidence Score</span>
                      </button>
                      <div className="absolute right-0 top-full mt-2 w-80 p-4 bg-slate-800 border border-slate-700 rounded-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
                        <p className="text-sm text-slate-300">{CONFIDENCE_EXPLANATION}</p>
                      </div>
                    </div>
                    <button onClick={exportResults}
                      className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:from-indigo-500 hover:to-purple-500 transition-all">
                      <Download className="w-4 h-4" />Export Report
                    </button>
                  </div>
                </div>

                {gaps.length > 5 && (
                  <div className="flex gap-3">
                    <button onClick={() => setExpandedGaps(new Set(gaps.map(g => g.id)))}
                      className="px-4 py-2 bg-slate-800/50 hover:bg-slate-800 border border-slate-700 rounded-xl text-slate-400 hover:text-white transition-colors text-sm font-medium">
                      Expand All
                    </button>
                    <button onClick={() => setExpandedGaps(new Set())}
                      className="px-4 py-2 bg-slate-800/50 hover:bg-slate-800 border border-slate-700 rounded-xl text-slate-400 hover:text-white transition-colors text-sm font-medium">
                      Collapse All
                    </button>
                  </div>
                )}

                {displayGaps.map((gap) => (
                  <div key={gap.id} className="bg-gradient-to-br from-slate-900 to-slate-800/50 rounded-3xl border border-slate-800 overflow-hidden hover:border-slate-700 transition-all">
                    <div className="p-8">
                      <div className="flex items-start justify-between mb-6 flex-wrap gap-4">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-3 mb-3 flex-wrap">
                            <span className={`px-3 py-1 rounded-full text-xs font-medium whitespace-nowrap ${
                              gap.gapStrength >= 0.75 ? 'bg-green-500/20 text-green-400' :
                              gap.gapStrength >= 0.50 ? 'bg-indigo-500/20 text-indigo-400' :
                              'bg-yellow-500/20 text-yellow-400'
                            }`}>{gap.gapStrengthLabel}</span>
                            {gap.noveltyScore !== undefined && (
                              <span className="px-3 py-1 rounded-full text-xs font-medium bg-cyan-500/20 text-cyan-400 whitespace-nowrap">
                                Novelty: {(gap.noveltyScore * 100).toFixed(0)}%
                              </span>
                            )}
                            {gap.compatibilityScore !== undefined && (
                              <span className="px-3 py-1 rounded-full text-xs font-medium bg-indigo-500/20 text-indigo-300 whitespace-nowrap">
                                Compatibility: {(gap.compatibilityScore * 100).toFixed(0)}%
                              </span>
                            )}
                            {gap.clusterCount > 1 && (
                              <span className="px-3 py-1 rounded-full text-xs font-medium bg-purple-500/20 text-purple-400 flex items-center gap-1 whitespace-nowrap">
                                <Layers className="w-3 h-3" />{gap.clusterCount} clusters
                              </span>
                            )}
                            {gap.confidenceRange && (
                              <span className="text-slate-500 text-xs whitespace-nowrap">{gap.confidenceRange}</span>
                            )}
                          </div>
                          <h3 className="text-lg sm:text-xl font-bold text-white mb-3 break-words">{gap.title}</h3>
                          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-2 sm:gap-6 text-sm">
                            <div className="flex items-center gap-2">
                              <span className="w-5 h-5 rounded-full bg-red-500/20 flex items-center justify-center flex-shrink-0">
                                <span className="w-2 h-2 bg-red-400 rounded-full"></span>
                              </span>
                              <span className="text-slate-400">Missing in:</span>
                              <span className="text-white font-medium">{gap.missingIn}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="w-5 h-5 rounded-full bg-green-500/20 flex items-center justify-center flex-shrink-0">
                                <span className="w-2 h-2 bg-green-400 rounded-full"></span>
                              </span>
                              <span className="text-slate-400">Present in:</span>
                              <span className="text-white font-medium">{gap.presentIn}</span>
                            </div>
                          </div>
                        </div>
                        <div className="relative w-20 h-20 sm:w-24 sm:h-24 flex-shrink-0">
                          <svg className="transform -rotate-90 w-20 h-20 sm:w-24 sm:h-24">
                            <circle cx="40" cy="40" r="36" stroke="currentColor" strokeWidth="5" fill="transparent" className="text-slate-800" />
                            <circle cx="40" cy="40" r="36" stroke="currentColor" strokeWidth="5" fill="transparent"
                              strokeDasharray={`${2 * Math.PI * 36}`}
                              strokeDashoffset={`${2 * Math.PI * 36 * (1 - gap.confidence)}`}
                              className={gap.confidence >= 0.85 ? 'text-green-500' : gap.confidence >= 0.75 ? 'text-indigo-500' : 'text-yellow-500'}
                            />
                          </svg>
                          <div className="absolute inset-0 flex items-center justify-center">
                            <span className="text-xl sm:text-2xl font-bold text-white">{Math.round(gap.confidence * 100)}%</span>
                          </div>
                        </div>
                      </div>
                      <p className="text-slate-300 leading-relaxed">{gap.explanation}</p>
                    </div>

                    <button onClick={() => toggleExpandGap(gap.id)}
                      className="w-full px-8 py-4 flex items-center justify-between bg-slate-800/50 hover:bg-slate-800 transition-colors border-t border-slate-800">
                      <span className="text-white font-medium">View Explainability Analysis</span>
                      {expandedGaps.has(gap.id) ? <ChevronUp className="w-5 h-5 text-slate-400" /> : <ChevronDown className="w-5 h-5 text-slate-400" />}
                    </button>

                    {expandedGaps.has(gap.id) && (
                      <div className="px-6 pb-6 space-y-4 border-t border-slate-800 pt-4 animate-in fade-in slide-in-from-top-2 duration-300">
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-white font-medium text-sm">Semantic Distance</span>
                            <div className="flex items-center gap-2">
                              {gap.semanticDistanceRange && <span className="text-slate-500 text-xs">{gap.semanticDistanceRange}</span>}
                              <span className="text-slate-400 text-xs">{gap.semanticDistance.toFixed(3)}</span>
                            </div>
                          </div>
                          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                            <div className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
                              style={{ width: `${gap.semanticDistance * 100}%` }} />
                          </div>
                          <div className="flex justify-between text-xs text-slate-500 mt-1">
                            <span>0 (identical)</span><span>1 (unrelated)</span>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                          <div className="p-3 bg-green-500/10 border border-green-500/20 rounded-xl">
                            <p className="text-green-400 font-medium text-xs mb-1">Shared</p>
                            <p className="text-slate-300 text-xs">{gap.conceptOverlap.shared.join(', ') || 'None'}</p>
                          </div>
                          <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-xl">
                            <p className="text-red-400 font-medium text-xs mb-1">Only in {gap.missingIn}</p>
                            <p className="text-slate-300 text-xs">{gap.conceptOverlap.uniqueToA.join(', ') || 'None'}</p>
                          </div>
                          <div className="p-3 bg-blue-500/10 border border-blue-500/20 rounded-xl">
                            <p className="text-blue-400 font-medium text-xs mb-1">Only in {gap.presentIn}</p>
                            <p className="text-slate-300 text-xs">{gap.conceptOverlap.uniqueToB.join(', ') || 'None'}</p>
                          </div>
                        </div>

                        {gap.evidenceSentences?.length > 0 && (
                          <div className="p-4 bg-slate-800/50 border border-slate-700 rounded-xl">
                            <h4 className="text-slate-300 font-medium text-sm mb-3 flex items-center gap-1">
                              <FileText className="w-4 h-4" />Evidence from Papers
                            </h4>
                            {gap.evidenceSentences.map((ev, idx) => (
                              <div key={idx} className="mb-3">
                                <p className="text-indigo-400 text-sm font-medium mb-2">{ev.domain}</p>
                                {ev.sentences.map((s, si) => (
                                  <p key={si} className="text-slate-300 text-sm mb-2 pl-3 border-l border-slate-600">"{s}"</p>
                                ))}
                              </div>
                            ))}
                          </div>
                        )}

                        <div className="p-4 bg-indigo-500/10 border border-indigo-500/20 rounded-xl">
                          <h4 className="text-indigo-400 font-medium text-sm mb-2 flex items-center gap-1">
                            <Brain className="w-4 h-4" />Why This Gap Was Detected
                          </h4>
                          <p className="text-slate-300 text-sm">{gap.whyDetected}</p>
                        </div>

                        <div className="p-4 bg-purple-500/10 border border-purple-500/20 rounded-xl">
                          <h4 className="text-purple-400 font-medium text-sm mb-2 flex items-center gap-1">
                            <Target className="w-4 h-4" />Future Research Suggestion
                          </h4>
                          <p className="text-slate-300 text-sm">{gap.futureSuggestion}</p>
                        </div>
                      </div>
                    )}
                  </div>
                ))}

                {gaps.length > 5 && (
                  <button onClick={() => setShowAllGaps(!showAllGaps)}
                    className="w-full py-3 bg-slate-800/50 hover:bg-slate-800 border border-slate-700 rounded-xl text-slate-400 hover:text-white transition-colors flex items-center justify-center gap-2">
                    {showAllGaps
                      ? <><ChevronUp className="w-4 h-4" />Show Less</>
                      : <><ChevronDown className="w-4 h-4" />Show All {gaps.length} Gaps</>}
                  </button>
                )}
              </>
            )}
          </div>
        )}
      </div>

      <footer className="border-t border-slate-800 mt-8">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                <Brain className="w-4 h-4 text-white" />
              </div>
              <div>
                <h3 className="text-white font-semibold text-sm">GapFinder</h3>
                <p className="text-xs text-slate-500">AI-Driven Research Gap Analysis</p>
              </div>
            </div>
            <p className="text-xs text-slate-500">© 2026 GapFinder</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;