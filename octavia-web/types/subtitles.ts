export interface SubtitleSegment {
  id: number;
  start: number;
  end: number;
  text: string;
  confidence?: number;
}

export interface SubtitleJob {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  file_name: string;
  format: string;
  language: string;
  segments?: SubtitleSegment[];
  created_at: string;
  download_url?: string;
}

export interface SubtitleFormat {
  id: 'srt' | 'vtt' | 'ass';
  name: string;
  extension: string;
  description: string;
}

export const SUBTITLE_FORMATS: SubtitleFormat[] = [
  { id: 'srt', name: 'SRT', extension: '.srt', description: 'Most common subtitle format' },
  { id: 'vtt', name: 'WebVTT', extension: '.vtt', description: 'Web video text tracks' },
  { id: 'ass', name: 'ASS', extension: '.ass', description: 'Advanced SubStation Alpha' },
];