"use client";

import { useRef, useState, useEffect, useCallback } from 'react';
import { Play, Pause, Volume2, VolumeX, Maximize } from 'lucide-react';

interface SideBySideVideoPlayerProps {
  originalVideoUrl?: string;
  translatedVideoUrl: string;
  className?: string;
}

export default function SideBySideVideoPlayer({
  originalVideoUrl,
  translatedVideoUrl,
  className = ""
}: SideBySideVideoPlayerProps) {
  const [originalVideoAvailable, setOriginalVideoAvailable] = useState<boolean>(false);
  const originalVideoRef = useRef<HTMLVideoElement>(null);
  const translatedVideoRef = useRef<HTMLVideoElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [originalVolume, setOriginalVolume] = useState(1);
  const [translatedVolume, setTranslatedVolume] = useState(1);
  const [videosLoaded, setVideosLoaded] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [activeAudio, setActiveAudio] = useState<'A' | 'B'>('A'); // A for left/original, B for right/translated

  // Sync videos when metadata loads
  useEffect(() => {
    const handleLoadedMetadata = () => {
      const originalVideo = originalVideoRef.current;
      const translatedVideo = translatedVideoRef.current;

      // Use the translated video's duration as the primary duration
      if (translatedVideo) {
        setDuration(translatedVideo.duration);
        setVideosLoaded(true);
        setVideoError(null);
      }
    };

    const handleError = () => {
      setVideoError('Failed to load video');
      setVideosLoaded(false);
    };

    const originalVideo = originalVideoRef.current;
    const translatedVideo = translatedVideoRef.current;

    // Reset states when URLs change
    setVideosLoaded(false);
    setVideoError(null);

    if (originalVideo) {
      originalVideo.addEventListener('loadedmetadata', handleLoadedMetadata);
      originalVideo.addEventListener('error', handleError);
    }
    if (translatedVideo) {
      translatedVideo.addEventListener('loadedmetadata', handleLoadedMetadata);
      translatedVideo.addEventListener('error', handleError);
    }

    return () => {
      if (originalVideo) {
        originalVideo.removeEventListener('loadedmetadata', handleLoadedMetadata);
        originalVideo.removeEventListener('error', handleError);
      }
      if (translatedVideo) {
        translatedVideo.removeEventListener('loadedmetadata', handleLoadedMetadata);
        translatedVideo.removeEventListener('error', handleError);
      }
    };
  }, [originalVideoUrl, translatedVideoUrl]);

  // Check if original video is available
  useEffect(() => {
    if (originalVideoUrl) {
      // Try to check if original video exists
      fetch(originalVideoUrl, { method: 'HEAD' })
        .then(response => {
          setOriginalVideoAvailable(response.ok);
          console.log(`[SideBySide] Original video available: ${response.ok}`);
        })
        .catch(() => {
          setOriginalVideoAvailable(false);
          console.log(`[SideBySide] Original video not available`);
        });
    } else {
      setOriginalVideoAvailable(false);
    }
  }, [originalVideoUrl]);

  // Initialize audio tracks when videos load
  useEffect(() => {
    const originalVideo = originalVideoRef.current;
    const translatedVideo = translatedVideoRef.current;

    if (!videosLoaded) return;

    // Set initial volume for active track
    const currentVolume = isMuted ? 0 : volume;

    if (activeAudio === 'A') {
      if (originalVideo) originalVideo.volume = currentVolume;
      if (translatedVideo) translatedVideo.volume = 0;
    } else {
      if (translatedVideo) translatedVideo.volume = currentVolume;
      if (originalVideo) originalVideo.volume = 0;
    }

    console.log(`[SideBySide] Initial setup complete for ${activeAudio}`);
  }, [videosLoaded]); // Only run when videos first load

  // Handle audio track switching
  useEffect(() => {
    const originalVideo = originalVideoRef.current;
    const translatedVideo = translatedVideoRef.current;

    if (!videosLoaded) return;

    const currentVolume = isMuted ? 0 : volume;
    console.log(`[SideBySide] Switching to ${activeAudio}, volume: ${currentVolume}, muted: ${isMuted}`);

    // Since original video/audio is not available, both videos have the same audio
    // We'll simulate the switching for UI purposes and future enhancement
    if (activeAudio === 'A') {
      // "Original" audio experience (currently same as translated)
      if (originalVideo) {
        originalVideo.volume = currentVolume;
        console.log(`[SideBySide] Left video (Original) volume set to: ${originalVideo.volume}`);
      }
      if (translatedVideo) {
        translatedVideo.volume = 0;
        console.log(`[SideBySide] Right video volume set to: ${translatedVideo.volume}`);
      }
    } else {
      // "Translated" audio experience
      if (translatedVideo) {
        translatedVideo.volume = currentVolume;
        console.log(`[SideBySide] Right video (Translated) volume set to: ${translatedVideo.volume}`);
      }
      if (originalVideo) {
        originalVideo.volume = 0;
        console.log(`[SideBySide] Left video volume set to: ${originalVideo.volume}`);
      }
    }
  }, [activeAudio, videosLoaded, isMuted, volume]);

  // Sync time updates and keep videos synchronized
  useEffect(() => {
    let isUpdatingTime = false; // Prevent infinite loops

    const handleTimeUpdate = (event: Event) => {
      if (isUpdatingTime) return;

      const targetVideo = event.target as HTMLVideoElement;
      const originalVideo = originalVideoRef.current;
      const translatedVideo = translatedVideoRef.current;

      if (targetVideo) {
        setCurrentTime(targetVideo.currentTime);

        // Sync the other video to stay in sync
        const otherVideo = targetVideo === originalVideo ? translatedVideo : originalVideo;
        if (otherVideo) {
          const timeDiff = Math.abs(targetVideo.currentTime - otherVideo.currentTime);

          // Only sync if there's a significant difference (more than 0.2 seconds to avoid jitter)
          if (timeDiff > 0.2) {
            isUpdatingTime = true;
            otherVideo.currentTime = targetVideo.currentTime;
            setTimeout(() => { isUpdatingTime = false; }, 100);
          }
        }
      }
    };

    const handlePlay = () => {
      console.log('[SideBySide] Video play event detected');
    };

    const handlePause = () => {
      console.log('[SideBySide] Video pause event detected');
    };

    const originalVideo = originalVideoRef.current;
    const translatedVideo = translatedVideoRef.current;

    if (originalVideo) {
      originalVideo.addEventListener('timeupdate', handleTimeUpdate);
      originalVideo.addEventListener('play', handlePlay);
      originalVideo.addEventListener('pause', handlePause);
    }
    if (translatedVideo) {
      translatedVideo.addEventListener('timeupdate', handleTimeUpdate);
      translatedVideo.addEventListener('play', handlePlay);
      translatedVideo.addEventListener('pause', handlePause);
    }

    return () => {
      if (originalVideo) {
        originalVideo.removeEventListener('timeupdate', handleTimeUpdate);
        originalVideo.removeEventListener('play', handlePlay);
        originalVideo.removeEventListener('pause', handlePause);
      }
      if (translatedVideo) {
        translatedVideo.removeEventListener('timeupdate', handleTimeUpdate);
        translatedVideo.removeEventListener('play', handlePlay);
        translatedVideo.removeEventListener('pause', handlePause);
      }
    };
  }, [videosLoaded]);

  const togglePlayPause = useCallback(() => {
    const originalVideo = originalVideoRef.current;
    const translatedVideo = translatedVideoRef.current;

    if (isPlaying) {
      originalVideo?.pause();
      translatedVideo?.pause();
      setIsPlaying(false);
    } else {
      // Ensure both videos are at the same time before playing
      if (originalVideo && translatedVideo) {
        const syncTime = Math.max(originalVideo.currentTime, translatedVideo.currentTime);
        originalVideo.currentTime = syncTime;
        translatedVideo.currentTime = syncTime;
      }

      // Start both videos but only the active audio track will be heard
      const playPromises = [originalVideo?.play(), translatedVideo?.play()].filter(Boolean);
      Promise.all(playPromises).then(() => {
        setIsPlaying(true);
        console.log('[SideBySide] Both videos started playing synchronously');
      }).catch(error => {
        console.error('[SideBySide] Error playing videos:', error);
      });
    }
  }, [isPlaying]);

  const handleSeek = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    const progressBar = progressRef.current;
    if (!progressBar || duration === 0) return;

    const rect = progressBar.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const percentage = clickX / rect.width;
    const newTime = percentage * duration;

    const originalVideo = originalVideoRef.current;
    const translatedVideo = translatedVideoRef.current;

    console.log(`[SideBySide] Seeking to ${newTime}s`);

    // Seek both videos to the same time
    if (originalVideo) {
      originalVideo.currentTime = newTime;
      console.log(`[SideBySide] Left video seeked to ${originalVideo.currentTime}s`);
    }
    if (translatedVideo) {
      translatedVideo.currentTime = newTime;
      console.log(`[SideBySide] Right video seeked to ${translatedVideo.currentTime}s`);
    }

    setCurrentTime(newTime);
  }, [duration]);

  const toggleMute = useCallback(() => {
    setIsMuted(!isMuted);
    // The useEffect will handle the actual volume changes
  }, [isMuted]);

  const handleVolumeChange = useCallback((newVolume: number) => {
    setVolume(newVolume);
    // The useEffect will handle applying the volume to the correct track
  }, []);

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Video Container */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left Video - Shows original if available, otherwise translated */}
        <div className="space-y-2">
          <div className="glass-panel p-1">
            <div className="relative">
              {(!videosLoaded || videoError) && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-lg z-10">
                  <div className="text-white text-sm text-center">
                    {videoError ? (
                      <div>
                        <div className="text-red-400 mb-2">⚠️ {videoError}</div>
                        <div className="text-xs text-slate-400">Try refreshing the page</div>
                      </div>
                    ) : (
                      'Loading video...'
                    )}
                  </div>
                </div>
              )}
              <video
                ref={originalVideoRef}
                src={originalVideoUrl || translatedVideoUrl}
                className="w-full h-full aspect-video rounded-lg bg-black object-contain"
                preload="metadata"
                playsInline
              >
                Your browser does not support the video tag.
              </video>
              <div className="absolute top-2 left-2 bg-primary-purple/70 text-white text-xs px-2 py-1 rounded z-20">
                {activeAudio === 'A' ? 'Active Audio' : 'Translated'}
              </div>
            </div>
          </div>

          {/* A/B Toggle Button */}
          <button
            onClick={() => setActiveAudio('A')}
            className={`w-full py-2 px-4 rounded-lg font-semibold text-sm transition-all ${
              activeAudio === 'A'
                ? 'bg-primary-purple text-white shadow-lg shadow-primary-purple/50'
                : 'bg-white/5 text-slate-400 hover:bg-white/10 hover:text-white'
            }`}
            title="Switch to left video audio"
          >
            Audio A (Left)
          </button>
        </div>

        {/* Right Video - Always translated */}
        <div className="space-y-2">
          <div className="glass-panel p-1">
            <div className="relative">
              {(!videosLoaded || videoError) && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-lg z-10">
                  <div className="text-white text-sm text-center">
                    {videoError ? (
                      <div>
                        <div className="text-red-400 mb-2">⚠️ {videoError}</div>
                        <div className="text-xs text-slate-400">Try refreshing the page</div>
                      </div>
                    ) : (
                      'Loading video...'
                    )}
                  </div>
                </div>
              )}
              <video
                ref={translatedVideoRef}
                src={translatedVideoUrl}
                className="w-full h-full aspect-video rounded-lg bg-black object-contain"
                preload="metadata"
                playsInline
              >
                Your browser does not support the video tag.
              </video>
            <div className="absolute top-2 left-2 bg-primary-purple/70 text-white text-xs px-2 py-1 rounded z-20">
              {activeAudio === 'B' ? 'Active Audio' : 'Translated'}
            </div>
            </div>
          </div>

          {/* A/B Toggle Button */}
          <button
            onClick={() => setActiveAudio('B')}
            className={`w-full py-2 px-4 rounded-lg font-semibold text-sm transition-all ${
              activeAudio === 'B'
                ? 'bg-primary-purple text-white shadow-lg shadow-primary-purple/50'
                : 'bg-white/5 text-slate-400 hover:bg-white/10 hover:text-white'
            }`}
            title="Switch to right video audio"
          >
            Audio B (Right)
          </button>
        </div>
      </div>

      {/* Controls */}
      <div className="glass-card p-4">
        {/* Progress Bar */}
        <div className="mb-4">
          <div
            ref={progressRef}
            className="h-2 bg-white/10 rounded-full cursor-pointer relative overflow-hidden"
            onClick={handleSeek}
          >
            <div
              className="h-full bg-gradient-to-r from-primary-purple to-primary-purple-bright rounded-full transition-all duration-100"
              style={{ width: `${duration > 0 ? (currentTime / duration) * 100 : 0}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-slate-400 mt-1">
            <span>{formatTime(currentTime)}</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>

        {/* Control Buttons */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            {/* Play/Pause */}
            <button
              onClick={togglePlayPause}
              className="flex items-center justify-center w-10 h-10 rounded-full bg-primary-purple/20 hover:bg-primary-purple/30 text-primary-purple-bright transition-colors"
            >
              {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-1" />}
            </button>

            {/* Volume */}
            <div className="flex items-center gap-2">
              <button
                onClick={toggleMute}
                className="flex items-center justify-center w-8 h-8 rounded-full hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
                title={isMuted ? 'Unmute' : 'Mute'}
              >
                {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
              </button>
              <div className="flex flex-col">
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={volume}
                  onChange={(e) => handleVolumeChange(parseFloat(e.target.value))}
                  className="w-20 h-1 bg-white/10 rounded-lg appearance-none cursor-pointer slider"
                  title={`Volume for ${activeAudio === 'A' ? 'Original' : 'Translated'} audio`}
                />
                <span className="text-xs text-slate-500 mt-1">
                  {activeAudio === 'A' ? 'A' : 'B'}
                </span>
              </div>
            </div>
          </div>

          {/* Additional Info */}
          <div className="text-xs text-slate-400">
            Active audio: {activeAudio === 'A' ? 'Left Video (A)' : 'Right Video (B)'}
            {!originalVideoAvailable && ' • Original audio not available - both sides play translated audio'}
          </div>
        </div>
      </div>
    </div>
  );
}