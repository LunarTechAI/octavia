"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Search, Filter, Download, ExternalLink, Clock, CheckCircle2, AlertCircle, Loader2, RefreshCw } from "lucide-react";
import { useUser } from "@/contexts/UserContext";
import { api, type Transaction, type SubtitleJobResponse } from "@/lib/api";

interface JobHistoryItem {
  id: string;
  name: string;
  type: string;
  lang: string;
  status: string;
  date: string;
  duration: string;
  description?: string;
  download_url?: string;
  created_at: string;
}

export default function JobHistoryPage() {
  const { user } = useUser();
  const [jobs, setJobs] = useState<JobHistoryItem[]>([]);
  const [filteredJobs, setFilteredJobs] = useState<JobHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterType, setFilterType] = useState<string>("all");
  const [refreshing, setRefreshing] = useState(false);

  // Fetch user's work history
  useEffect(() => {
    if (user) {
      fetchUserHistory();
    }
  }, [user]);

  // Filter jobs when search term or filter changes
  useEffect(() => {
    let filtered = jobs;

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(job =>
        job.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        job.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        job.type.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Apply type filter
    if (filterType !== "all") {
      filtered = filtered.filter(job =>
        job.type.toLowerCase().includes(filterType.toLowerCase())
      );
    }

    setFilteredJobs(filtered);
  }, [jobs, searchTerm, filterType]);

  const fetchUserHistory = async () => {
    if (!user) return;

    setLoading(true);
    try {
      // Fetch transaction history (credit purchases)
      const [transactionsResponse, subtitleJobs, translationJobs] = await Promise.all([
        api.getTransactionHistory(),
        fetchSubtitleJobs(),
        fetchTranslationJobs()
      ]);

      const historyItems: JobHistoryItem[] = [];

      // Add credit purchase transactions
      if (transactionsResponse.success && transactionsResponse.data) {
        const transactions = transactionsResponse.data.transactions || transactionsResponse.data;
        if (transactions && Array.isArray(transactions)) {
          transactions.forEach((tx: Transaction) => {
            historyItems.push({
              id: `TXN-${tx.id.substring(0, 8)}`,
              name: tx.description || "Credit Purchase",
              type: "Credit Purchase",
              lang: "N/A",
              status: tx.status === "completed" ? "Completed" :
                tx.status === "pending" ? "Processing" : "Failed",
              date: formatTimeAgo(tx.created_at),
              duration: "Instant",
              description: tx.credits ? `Added ${tx.credits} credits` : undefined,
              created_at: tx.created_at
            });
          });
        }
      }

      // Add translation jobs (video/audio) from Supabase
      translationJobs.forEach((job: any) => {
        // Handle both old and new schema fields
        const targetLang = job.target_language || job.target_lang || job.language;
        const sourceLang = job.source_language || job.source_lang;
        const filename = job.original_filename || job.filename;

        let lang = "Unknown";
        if (sourceLang && targetLang) {
          lang = `${sourceLang.toUpperCase()} → ${targetLang.toUpperCase()}`;
        } else if (targetLang) {
          lang = `EN → ${targetLang.toUpperCase()}`;
        }

        // Determine job type display name
        let typeDisplay = "Translation";
        if (job.type === "video" || job.type === "video_enhanced") {
          typeDisplay = job.type === "video_enhanced" ? "Video Translation (Enhanced)" : "Video Translation";
        } else if (job.type === "audio") {
          typeDisplay = "Audio Translation";
        } else if (job.type === "subtitles" || job.type === "subtitle") {
          typeDisplay = "Subtitle Generation";
        }

        // Map status
        const status = job.status === "completed" ? "Completed" :
          job.status === "processing" || job.status === "pending" ? "Processing" :
            job.status === "failed" ? "Failed" : "Unknown";

        historyItems.push({
          id: `JOB-${job.id?.substring(0, 8) || "UNKNOWN"}`,
          name: filename || `${typeDisplay} Job`,
          type: typeDisplay,
          lang: lang,
          status: status,
          date: formatTimeAgo(job.created_at),
          duration: job.processing_time_seconds ? `${job.processing_time_seconds.toFixed(1)}s` : "N/A",
          description: job.message || (job.processed_chunks && job.total_chunks ? `${job.processed_chunks}/${job.total_chunks} chunks` : undefined),
          download_url: job.output_path ? `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/download/video/${job.id}` : undefined,
          created_at: job.created_at
        });
      });

      // Add subtitle jobs
      subtitleJobs.forEach((job: any) => {
        historyItems.push({
          id: `JOB-${job.job_id?.substring(0, 8) || "UNKNOWN"}`,
          name: job.original_filename || "Subtitle Generation",
          type: "Subtitle Generation",
          lang: job.language || "Auto",
          status: job.status === "completed" ? "Completed" :
            job.status === "processing" ? "Processing" : "Failed",
          date: formatTimeAgo(job.created_at),
          duration: job.segment_count ? `${job.segment_count} segments` : "N/A",
          download_url: job.download_url,
          created_at: job.created_at
        });
      });

      // Sort by creation date (newest first)
      historyItems.sort((a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );

      setJobs(historyItems);
    } catch (error) {
      console.error("Failed to fetch user history:", error);
      // Use demo data as fallback
      setJobs(getDemoJobs());
    } finally {
      setLoading(false);
    }
  };

  const fetchTranslationJobs = async () => {
    // Fetch user's job history from Supabase (for non-demo users)
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        console.warn("No auth token found");
        return [];
      }

      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/jobs/history?limit=100`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (data.success && data.jobs) {
        console.log(`Fetched ${data.jobs.length} jobs from Supabase`);
        return data.jobs;
      }

      return [];
    } catch (error) {
      console.error("Failed to fetch translation jobs from Supabase:", error);
      return [];
    }
  };

  const fetchSubtitleJobs = async () => {
    // In a real implementation, you would fetch from /api/translate/subtitles/history
    // For now, check localStorage for recent subtitle jobs
    const jobs: any[] = [];

    // Check localStorage for stored subtitle jobs
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key?.startsWith('subtitle_job_')) {
        try {
          const jobData = JSON.parse(localStorage.getItem(key) || '{}');
          if (jobData.job_id) {
            jobs.push(jobData);
          }
        } catch (e) {
          console.warn("Failed to parse stored job:", e);
        }
      }
    }

    return jobs;
  };

  const formatTimeAgo = (dateString: string): string => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffDays > 0) {
      return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    } else if (diffHours > 0) {
      return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    } else if (diffMins > 0) {
      return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
    } else {
      return "Just now";
    }
  };

  const getDemoJobs = (): JobHistoryItem[] => {
    return [
      {
        id: "JOB-1024",
        name: "Product Demo Video",
        type: "Video Translation",
        lang: "EN → ES",
        status: "Completed",
        date: formatTimeAgo(new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString()),
        duration: "4:30",
        created_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString()
      },
      {
        id: "JOB-1023",
        name: "Marketing Podcast Ep. 4",
        type: "Audio Translation",
        lang: "EN → FR",
        status: "Processing",
        date: formatTimeAgo(new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString()),
        duration: "24:15",
        created_at: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString()
      },
      {
        id: "JOB-1022",
        name: "Tutorial Series - Part 1",
        type: "Subtitle Generation",
        lang: "EN",
        status: "Completed",
        date: formatTimeAgo(new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()),
        duration: "12:00",
        created_at: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()
      },
      {
        id: "TXN-abc12345",
        name: "Pro Credits Purchase",
        type: "Credit Purchase",
        lang: "N/A",
        status: "Completed",
        date: formatTimeAgo(new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString()),
        duration: "Instant",
        description: "Added 250 credits",
        created_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString()
      },
      {
        id: "JOB-1021",
        name: "Keynote Speech",
        type: "Video Translation",
        lang: "EN → DE",
        status: "Failed",
        date: formatTimeAgo(new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString()),
        duration: "45:00",
        created_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString()
      },
    ];
  };

  const handleDownload = async (jobId: string, downloadUrl?: string) => {
    if (!downloadUrl) {
      alert("No download available for this item");
      return;
    }

    try {
      // Download the file
      const blob = await api.downloadFileByUrl(downloadUrl);
      // Extract filename from URL or use jobId
      const filename = `octavia_job_${jobId}.zip`;
      api.saveFile(blob, filename);
    } catch (error) {
      console.error("Download failed:", error);
      alert("Failed to download file");
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await fetchUserHistory();
    setRefreshing(false);
  };

  const handleFilterChange = (type: string) => {
    setFilterType(type);
  };

  if (!user) {
    return (
      <div className="space-y-8">
        <div>
          <h1 className="font-display text-3xl font-black text-white mb-2 text-glow-purple">Job History</h1>
          <p className="text-slate-400 text-sm">Please log in to view your job history</p>
        </div>
        <div className="glass-panel p-8 text-center">
          <Clock className="w-12 h-12 text-slate-400 mx-auto mb-4" />
          <h3 className="text-white text-lg font-bold mb-2">No User Found</h3>
          <p className="text-slate-400">Please log in to access your job history.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="font-display text-3xl font-black text-white mb-2 text-glow-purple">Job History</h1>
          <p className="text-slate-400 text-sm">View and manage your past translation tasks</p>
        </div>

        <div className="flex items-center gap-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
            <input
              type="text"
              placeholder="Search jobs..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 pr-4 py-2 rounded-lg bg-white/5 border border-white/10 text-sm text-white placeholder:text-slate-500 focus:outline-none focus:border-primary-purple/50 w-64 transition-all"
            />
          </div>
          <div className="flex items-center gap-1 bg-white/5 border border-white/10 rounded-lg p-1">
            {["all", "translation", "subtitle", "credit"].map((type) => (
              <button
                key={type}
                onClick={() => handleFilterChange(type)}
                className={`px-3 py-1 rounded text-xs font-medium transition-colors ${filterType === type
                  ? "bg-primary-purple text-white"
                  : "text-slate-400 hover:text-white hover:bg-white/5"
                  }`}
              >
                {type === "all" ? "All" :
                  type === "translation" ? "Translation" :
                    type === "subtitle" ? "Subtitles" : "Credits"}
              </button>
            ))}
          </div>
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="p-2 rounded-lg bg-white/5 border border-white/10 text-slate-400 hover:text-white hover:bg-white/10 transition-colors disabled:opacity-50"
            title="Refresh"
          >
            {refreshing ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <RefreshCw className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {/* Jobs List */}
      <div className="glass-panel overflow-hidden">
        {loading ? (
          <div className="p-8 flex items-center justify-center">
            <Loader2 className="w-8 h-8 text-primary-purple-bright animate-spin" />
          </div>
        ) : filteredJobs.length === 0 ? (
          <div className="p-8 text-center">
            <Clock className="w-12 h-12 text-slate-400 mx-auto mb-4" />
            <h3 className="text-white text-lg font-bold mb-2">No Jobs Found</h3>
            <p className="text-slate-400 mb-4">
              {searchTerm || filterType !== "all"
                ? "No jobs match your search criteria"
                : "You haven't completed any jobs yet"}
            </p>
            {searchTerm && (
              <button
                onClick={() => {
                  setSearchTerm("");
                  setFilterType("all");
                }}
                className="text-primary-purple-bright hover:text-primary-purple text-sm font-medium"
              >
                Clear filters
              </button>
            )}
          </div>
        ) : (
          <>
            <div class
