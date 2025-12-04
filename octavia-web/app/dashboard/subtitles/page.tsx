"use client";

import { motion } from "framer-motion";
import { FileVideo, Captions, Loader2 } from "lucide-react";
import { useState } from "react";
import { useApi } from "@/hooks/useApi";
import { useUser } from "@/contexts/UserContext";
import { api } from "@/lib/api";
import { useRouter } from 'next/navigation';

export default function SubtitleGenerationPage() {
  const router = useRouter();
  const { user } = useUser();
  const { loading, error, execute } = useApi();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [language, setLanguage] = useState("en");
  const [format, setFormat] = useState("srt");

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && (file.type.startsWith('video/') || file.type.startsWith('audio/'))) {
      setSelectedFile(file);
    }
  };

  const handleGenerateSubtitles = async () => {
    if (!selectedFile || !user?.email) return;

    const response = await execute(() =>
      api.generateSubtitles(selectedFile, format, user.email, language)
    );

    if (response && response.success) {
      // Store the job ID for progress tracking
      if (response.job_id) {
        localStorage.setItem('current_subtitle_job', response.job_id);
        // Redirect to progress page WITH the job ID
        router.push(`/dashboard/subtitles/progress?jobId=${response.job_id}`);
      } else if (response.download_url) {
        // If immediate download available
        alert('Subtitles generated!');
      }
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col gap-2">
        <h1 className="font-display text-3xl font-black text-white text-glow-purple">Subtitle Generation</h1>
        <p className="text-slate-400 text-sm">Generate accurate subtitles from your video or audio files</p>
      </div>

      {/* Error Display */}
      {error && (
        <div className="glass-panel border-red-500/30 bg-red-500/10 p-4">
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      {/* Upload Zone */}
      <motion.div
        whileHover={{ scale: 1.01 }}
        className="glass-panel glass-panel-high relative border-2 border-dashed border-primary-purple/30 hover:border-primary-purple/50 transition-all cursor-pointer group mb-6 overflow-hidden"
      >
        <div className="glass-shine" />
        <div className="glow-purple" style={{ width: "300px", height: "300px", top: "50%", left: "50%", transform: "translate(-50%, -50%)", zIndex: 1 }} />

        <div className="relative z-20 py-12 px-6">
          <input
            type="file"
            id="media-upload"
            accept="video/*,audio/*"
            onChange={handleFileSelect}
            className="hidden"
          />
          <label htmlFor="media-upload" className="flex flex-col items-center justify-center gap-3 text-center cursor-pointer">
            <div className="flex items-center justify-center w-16 h-16 rounded-2xl bg-primary-purple/10 border border-primary-purple/30 shadow-glow group-hover:scale-110 transition-transform">
              {loading ? <Loader2 className="w-8 h-8 text-primary-purple-bright animate-spin" /> : <FileVideo className="w-8 h-8 text-primary-purple-bright" />}
            </div>
            <div>
              <h3 className="text-white text-lg font-bold mb-1 text-glow-purple">
                {selectedFile ? selectedFile.name : 'Drop your media here'}
              </h3>
              <p className="text-slate-400 text-sm">or click to browse files • Video or Audio supported</p>
            </div>
          </label>
        </div>
      </motion.div>

      {/* Configuration */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="glass-card p-4">
          <label className="text-white text-sm font-semibold mb-2 block">Audio Language</label>
          <select 
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="glass-select w-full"
            disabled={loading}
          >
            <option value="en">English</option>
            <option value="es">Spanish</option>
            <option value="fr">French</option>
            <option value="de">German</option>
          </select>
        </div>
        <div className="glass-card p-4">
          <label className="text-white text-sm font-semibold mb-2 block">Subtitle Format</label>
          <select 
            value={format}
            onChange={(e) => setFormat(e.target.value)}
            className="glass-select w-full"
            disabled={loading}
          >
            <option value="srt">SRT</option>
            <option value="vtt">VTT</option>
            <option value="ass">ASS</option>
          </select>
        </div>
      </div>

      {/* Start Button */}
      <button 
        onClick={handleGenerateSubtitles}
        disabled={!selectedFile || loading || !user}
        className="btn-border-beam w-full group disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-4 text-base">
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Generating...</span>
            </>
          ) : (
            <>
              <Captions className="w-5 h-5" />
              <span>Generate Subtitles</span>
            </>
          )}
        </div>
      </button>
    </div>
  );
}