export interface Paper {
  id: string;
  file: File;
  domain: string;
  text?: string;
  embeddings?: number[][];
  concepts?: Concept[];
  status: 'pending' | 'processing' | 'completed' | 'error';
}

export interface Concept {
  label: string;
  keywords: string[];
  centroid: number[];
  prominence: number;
}

export interface ResearchGap {
  id: string;
  title: string;
  confidence: number;
  gapStrength?: number;
  gapStrengthLabel?: string;
  noveltyScore?: number;
  // Compatibility score between the two selected domains
  compatibilityScore?: number;
  // New deduplication fields
  confidenceMin?: number;
  confidenceMax?: number;
  confidenceRange?: string;
  clusterCount?: number;
  semanticDistance: number;
  semanticDistanceMin?: number;
  semanticDistanceMax?: number;
  semanticDistanceRange?: string;
  missingIn: string;
  presentIn: string;
  explanation: string;
  evidenceSentences: {
    domain: string;
    sentences: string[];
  }[];
  conceptOverlap: {
    shared: string[];
    uniqueToA: string[];
    uniqueToB: string[];
  };
  futureSuggestion: string;
  whyDetected: string;
  // Supporting gaps IDs (for internal tracking)
  supportingGaps?: string[];
}


export interface AnalyzeResponse {
  papers: Paper[];
  gaps: ResearchGap[];
  totalDetected: number;
  totalUnique: number;
  confidenceExplanation: string;
}

