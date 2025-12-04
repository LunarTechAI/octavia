"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { CheckCircle2, Loader2, Clock, AlertCircle, XCircle, Download } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { api } from "@/lib/api";

interface JobStatus {
  job_id: string;
  status: "pending" | "processing" | "completed" | "failed";
  progress: number;
  status_message?: string;
  language?: string;
  segment_count?: number;
  download_url?: string;
  error?: string;
  estimated_time?: string;
  format?: string;
  original_filename?: string;
  target_language?: string;
}

export default function SubtitleGenerationProgressPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const jobId = searchParams.get("jobId") || localStorage.getItem("current_subtitle_job");
  
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Poll for job status
  useEffect(() => {
    if (!jobId) {
      setError("No job ID found. Please start a subtitle generation first.");
      setLoading(false);
      return;
    }

    const pollJobStatus = async () => {
      try {
        
        const response = await api.getSubtitleJobStatus(jobId);
        
        if (response.success) {
          setJobStatus({
            job_id: response.job_id,
            status: response.status,
            progress: response.progress,
            status_message: response.status_message || response.error,
            language: response.language,
            segment_count: response.segment_count,
            download_url: response.download_url,
            error: response.error,
            format: response.format
          });
          
          // If job is completed or failed, stop polling
          if (response.status === "completed" || response.status === "failed") {
            if (pollingInterval) {
              clearInterval(pollingInterval);
              setPollingInterval(null);
            }
            
            // If completed, automatically redirect to review page after 2 seconds
            if (response.status === "completed") {
              setTimeout(() => {
                router.push(`/dashboard/subtitles/review?jobId=${jobId}`);
              }, 2000);
            }
          }
        } else {
          setError(response.error || "Failed to fetch job status");
        }
      } catch (err) {
        setError("Network error while checking job status");
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    // Initial fetch
    pollJobStatus();

    // Set up polling every 3 seconds
    const interval = setInterval(pollJobStatus, 3000);
    setPollingInterval(interval);

    // Cleanup on unmount
    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [jobId]);

  const getProcessingStep = () => {
    if (!jobStatus) return 0;
    
    if (jobStatus.status === "pending") return 1;
    if (jobStatus.status === "processing") {
      if (jobStatus.progress < 30) return 2; // Audio extraction
      if (jobStatus.progress < 70) return 3; // Speech recognition
      return 4; // Format export
    }
    return 5; // Completed
  };

  const getEstimatedTime = () => {
    if (!jobStatus) return "Calculating...";
    
    if (jobStatus.estimated_time) return jobStatus.estimated_time;
    
    // Calculate based on progress
    const remainingPercent = 100 - (jobStatus.progress || 0);
    const secondsRemaining = Math.max(10, Math.round(remainingPercent * 0.3)); // Rough estimate
    const minutes = Math.floor(secondsRemaining / 60);
    const seconds = secondsRemaining % 60;
    
    if (minutes > 0) {
      return `~${minutes}m ${seconds}s`;
    }
    return `~${seconds}s`;
  };

  const getStatusMessage = () => {
    if (!jobStatus) return "Initializing job...";
    
    if (jobStatus.status_message) return jobStatus.status_message;
    
    switch (jobStatus.status) {
      case "pending":
        return "Job queued, waiting to start...";
      case "processing":
        if (jobStatus.progress < 30) return "Extracting audio from video...";
        if (jobStatus.progress < 70) {
          const estimatedSegments = jobStatus.segment_count || 18;
          const currentSegment = Math.floor((jobStatus.progress - 30) / 40 * estimatedSegments) + 1;
          return `Transcribing audio (segment ${currentSegment} of ${estimatedSegments})...`;
        }
        return "Formatting subtitles and generating output...";
      case "completed":
        return "Subtitles generated successfully! Redirecting to review...";
      case "failed":
        return jobStatus.error || "Subtitle generation failed. Please try again.";
      default:
        return "Processing...";
    }
  };

  if (error) {
    return (
      <div className="space-y-8">
        <div className="flex flex-col gap-1 border-b border-white/10 pb-6">
          <h1 className="font-display text-3xl font-black text-white text-glow-purple">Error</h1>
          <p className="text-slate-400 text-sm">Unable to process subtitle generation</p>
        </div>
        
        <div className="glass-panel p-8">
          <div className="flex items-center gap-4 mb-4">
            <XCircle className="w-8 h-8 text-red-400" />
            <div>
              <h3 className="text-xl font-bold text-white mb-1">Job Processing Error</h3>
              <p className="text-slate-400">{error}</p>
            </div>
          </div>
          
          <div className="mt-6 flex gap-3">
            <button 
              onClick={() => router.push("/dashboard/subtitles")}
              className="btn-border-beam"
            >
              <div className="btn-border-beam-inner px-6 py-2">
                Back to Generate
              </div>
            </button>
            <button 
              onClick={() => window.location.reload()}
              className="px-6 py-2 rounded-lg border border-white/10 hover:bg-white/5 text-sm text-slate-300 hover:text-white transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (loading && !jobStatus) {
    return (
      <div className="space-y-8">
        <div className="flex flex-col gap-1 border-b border-white/10 pb-6">
          <h1 className="font-display text-3xl font-black text-white text-glow-purple">Loading...</h1>
          <p className="text-slate-400 text-sm">Fetching job status</p>
        </div>
        
        <div className="glass-panel p-8 flex items-center justify-center">
          <Loader2 className="w-12 h-12 text-primary-purple-bright animate-spin" />
        </div>
      </div>
    );
  }

  const currentStep = getProcessingStep();

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col gap-1 border-b border-white/10 pb-6">
        <div className="flex items-center justify-between">
          <h1 className="font-display text-3xl font-black text-white text-glow-purple">
            {jobStatus?.status === "completed" ? "Generation Complete!" :
             jobStatus?.status === "failed" ? "Generation Failed" :
             "Generating Subtitles..."}
          </h1>
          {jobId && (
            <div className="text-xs text-slate-500 font-mono bg-white/5 px-3 py-1 rounded">
              Job: {jobId.substring(0, 8)}...
            </div>
          )}
        </div>
        <p className="text-slate-400 text-sm">{getStatusMessage()}</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="flex flex-col gap-6">
          {/* Overall Progress */}
          <div className="glass-panel p-6">
            <div className="flex items-center justify-between gap-6 mb-3">
              <p className="text-base font-medium text-white">Overall Progress</p>
              <p className="text-2xl font-bold text-primary-purple-bright text-glow-purple">
                {jobStatus?.progress || 0}%
              </p>
            </div>
            <div className="w-full bg-white/5 rounded-full h-2.5 mb-3 overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${jobStatus?.progress || 0}%` }}
                transition={{ duration: 0.5, ease: "easeOut" }}
                className={`h-2.5 rounded-full shadow-glow ${
                  jobStatus?.status === "failed" ? "bg-red-500" :
                  jobStatus?.status === "completed" ? "bg-green-500" :
                  "bg-primary-purple"
                }`}
              />
            </div>
            <p className="text-sm text-slate-400">{getStatusMessage()}</p>
            
            {jobStatus?.segment_count && (
              <div className="mt-4 text-xs text-slate-500">
                {jobStatus.segment_count} subtitle segments detected
              </div>
            )}
          </div>

          {/* Pipeline Steps */}
          <div className="glass-panel p-6">
            <h2 className="text-lg font-bold text-white mb-4">Processing Steps</h2>
            <div className="space-y-4">
              {/* Step 1: Job Queued */}
              <div className={`flex items-center gap-4 ${currentStep >= 1 ? "" : "opacity-50"}`}>
                <div className={`flex size-10 items-center justify-center rounded-full ${
                  currentStep >= 1 ? "bg-green-500/20 text-green-400 shadow-glow" : "bg-white/5 text-slate-500"
                }`}>
                  {currentStep > 1 ? (
                    <CheckCircle2 className="w-5 h-5" />
                  ) : currentStep === 1 ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Clock className="w-5 h-5" />
                  )}
                </div>
                <div className="flex-1">
                  <p className={`text-sm font-medium ${currentStep >= 1 ? "text-white" : "text-slate-400"}`}>
                    Job Initialization
                  </p>
                  <p className="text-xs text-slate-400">
                    {currentStep > 1 ? "Completed" : currentStep === 1 ? "In Progress" : "Pending"}
                  </p>
                </div>
              </div>

              {/* Step 2: Audio Extraction */}
              <div className={`flex items-center gap-4 ${currentStep >= 2 ? "" : "opacity-50"}`}>
                <div className={`flex size-10 items-center justify-center rounded-full ${
                  currentStep >= 2 ? "bg-green-500/20 text-green-400 shadow-glow" :
                  currentStep === 2 ? "bg-primary-purple/20 text-primary-purple-bright shadow-glow" :
                  "bg-white/5 text-slate-500"
                }`}>
                  {currentStep > 2 ? (
                    <CheckCircle2 className="w-5 h-5" />
                  ) : currentStep === 2 ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Clock className="w-5 h-5" />
                  )}
                </div>
                <div className="flex-1">
                  <p className={`text-sm font-medium ${currentStep >= 2 ? "text-white" : "text-slate-400"}`}>
                    Audio Extraction
                  </p>
                  <p className="text-xs text-slate-400">
                    {currentStep > 2 ? "Completed" : currentStep === 2 ? "In Progress" : "Pending"}
                  </p>
                </div>
              </div>

              {/* Step 3: Speech Recognition */}
              <div className={`flex items-center gap-4 ${currentStep >= 3 ? "" : "opacity-50"}`}>
                <div className={`flex size-10 items-center justify-center rounded-full ${
                  currentStep >= 3 ? "bg-green-500/20 text-green-400 shadow-glow" :
                  currentStep === 3 ? "bg-primary-purple/20 text-primary-purple-bright shadow-glow" :
                  "bg-white/5 text-slate-500"
                }`}>
                  {currentStep > 3 ? (
                    <CheckCircle2 className="w-5 h-5" />
                  ) : currentStep === 3 ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Clock className="w-5 h-5" />
                  )}
                </div>
                <div className="flex-1">
                  <p className={`text-sm font-medium ${currentStep >= 3 ? "text-white" : "text-slate-400"}`}>
                    Speech Recognition
                  </p>
                  <p className="text-xs text-slate-400">
                    {currentStep > 3 ? "Completed" : currentStep === 3 ? "In Progress" : "Pending"}
                  </p>
                </div>
              </div>

              {/* Step 4: Format Export */}
              <div className={`flex items-center gap-4 ${currentStep >= 4 ? "" : "opacity-50"}`}>
                <div className={`flex size-10 items-center justify-center rounded-full ${
                  currentStep >= 4 ? "bg-green-500/20 text-green-400 shadow-glow" :
                  currentStep === 4 ? "bg-primary-purple/20 text-primary-purple-bright shadow-glow" :
                  "bg-white/5 text-slate-500"
                }`}>
                  {currentStep > 4 ? (
                    <CheckCircle2 className="w-5 h-5" />
                  ) : currentStep === 4 ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Clock className="w-5 h-5" />
                  )}
                </div>
                <div className="flex-1">
                  <p className={`text-sm font-medium ${currentStep >= 4 ? "text-white" : "text-slate-400"}`}>
                    Format Export
                  </p>
                  <p className="text-xs text-slate-400">
                    {currentStep > 4 ? "Completed" : currentStep === 4 ? "In Progress" : "Pending"}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-6">
          {/* Status Overview */}
          <div className="glass-panel p-6">
            <h3 className="text-lg font-bold text-white mb-4">Status Overview</h3>
            <div className="flex flex-col gap-3">
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                <p className="text-sm text-slate-400">Estimated Time</p>
                <p className="text-sm font-bold text-white">{getEstimatedTime()}</p>
              </div>
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                <p className="text-sm text-slate-400">Detected Language</p>
                <p className="text-sm font-bold text-white">{jobStatus?.language || "Auto-detecting..."}</p>
              </div>
              <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                <p className="text-sm text-slate-400">Job Status</p>
                <p className={`text-sm font-bold ${
                  jobStatus?.status === "completed" ? "text-green-400" :
                  jobStatus?.status === "failed" ? "text-red-400" :
                  jobStatus?.status === "processing" ? "text-blue-400" :
                  "text-yellow-400"
                }`}>
                  {jobStatus?.status?.toUpperCase() || "UNKNOWN"}
                </p>
              </div>
              {jobStatus?.format && (
                <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                  <p className="text-sm text-slate-400">Format</p>
                  <p className="text-sm font-bold text-white">{jobStatus.format.toUpperCase()}</p>
                </div>
              )}
            </div>
          </div>

          {/* Actions */}
          <div className="glass-panel p-6">
            <h3 className="text-lg font-bold text-white mb-4">Actions</h3>
            <div className="space-y-3">
              {jobStatus?.status === "completed" && jobStatus.download_url ? (
                <button 
                  onClick={() => {
                    api.downloadFileByUrl(jobStatus.download_url!, `subtitles_${jobId}.${jobStatus.format || 'srt'}`);
                  }}
                  className="w-full py-3 rounded-lg bg-green-500/10 hover:bg-green-500/20 text-green-400 font-medium transition-colors border border-green-500/20 flex items-center justify-center gap-2"
                >
                  <Download className="w-5 h-5" />
                  Download Subtitles
                </button>
              ) : null}
              
              <button 
                onClick={() => router.push("/dashboard/subtitles")}
                className="w-full py-2 rounded-lg border border-white/10 hover:bg-white/5 text-sm text-slate-300 hover:text-white transition-colors"
              >
                Start New Generation
              </button>
              
              <button 
                onClick={() => router.push("/dashboard/history")}
                className="w-full py-2 rounded-lg border border-white/10 hover:bg-white/5 text-sm text-slate-300 hover:text-white transition-colors"
              >
                View All Jobs
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}