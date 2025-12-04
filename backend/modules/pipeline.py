"""
Complete video translation pipeline with subtitle generation
Meeting technical assessment requirements
Supports Russian → English and English → German translations
"""

import os
import sys
import json
import uuid
import shutil
import logging
import tempfile
import subprocess
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

import whisper
import torch
from transformers import MarianMTModel, MarianTokenizer
import edge_tts
import asyncio
from pydub import AudioSegment
import numpy as np

# Import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from modules.audio_translator import AudioTranslator, TranslationConfig, TranslationResult
    from modules.instrumentation import MetricsCollector
    from modules.subtitle_generator import SubtitleGenerator
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    logging.warning("Local modules not available, running in simplified mode")

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the video translation pipeline"""
    chunk_size: int = 30
    max_chunk_size: int = 120
    min_chunk_size: int = 10
    timing_tolerance_ms: int = 200
    max_condensation_ratio: float = 1.2
    target_lufs: float = -16.0
    max_peak_db: float = -1.0
    device: str = "cpu"
    max_workers: int = 2
    temp_dir: str = "temp"
    output_dir: str = "outputs"
    generate_subtitles: bool = True
    subtitle_formats: List[str] = None  # ["srt", "vtt", "ass"]
    bilingual_subtitles: bool = True

@dataclass
class VideoInfo:
    """Video metadata"""
    path: str
    duration: float
    width: int
    height: int
    codec: str
    audio_codec: str
    frame_rate: float
    bitrate: int

@dataclass
class ChunkInfo:
    """Audio chunk information"""
    id: int
    path: str
    start_ms: float
    end_ms: float
    duration_ms: float
    has_speech: bool = True

@dataclass
class ProcessingMetrics:
    """Processing metrics for a chunk"""
    chunk_id: int
    original_duration_ms: float
    translated_duration_ms: float
    duration_diff_ms: float
    condensation_ratio: float
    speed_adjustment: float
    processing_time_ms: float
    success: bool
    subtitle_generated: bool = False
    error: Optional[str] = None

class VideoTranslationPipeline:
    """Main video translation pipeline with subtitle support"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        if self.config.subtitle_formats is None:
            self.config.subtitle_formats = ["srt", "vtt"]
        
        self.whisper_model = None
        self.translator = None
        self.subtitle_generator = None
        self.metrics_collector = None
        self.temp_dir = None
        self.output_dir = None
        
        # Create directories
        os.makedirs(self.config.temp_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join("artifacts", "pipeline.log")
        os.makedirs("artifacts", exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def load_models(self, source_lang: str = "en", target_lang: str = "de"):
        """Load all required models including subtitle generator"""
        try:
            logger.info("Loading Whisper model...")
            self.whisper_model = whisper.load_model("base")
            
            logger.info("Loading subtitle generator...")
            self.subtitle_generator = SubtitleGenerator(model_size="base")
            
            logger.info(f"Loading translation models for {source_lang}→{target_lang}...")
            
            # Setup translator
            translator_config = TranslationConfig(
                source_lang=source_lang,
                target_lang=target_lang,
                auto_detect=True
            )
            self.translator = AudioTranslator(translator_config)
            
            # Load models in translator
            if not self.translator.load_models():
                raise Exception("Failed to load translation models")
            
            logger.info("All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def analyze_video(self, video_path: str) -> VideoInfo:
        """Analyze video file and extract metadata"""
        try:
            logger.info(f"Analyzing video: {video_path}")
            
            # Use ffprobe to get video info
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"FFprobe failed: {result.stderr}")
            
            data = json.loads(result.stdout)
            
            # Extract video stream info
            video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
            audio_stream = next((s for s in data['streams'] if s['codec_type'] == 'audio'), None)
            
            if not video_stream:
                raise Exception("No video stream found")
            
            # Get duration from format or stream
            duration = float(data['format'].get('duration', 0))
            if duration == 0 and audio_stream:
                duration = float(audio_stream.get('duration', 0))
            
            video_info = VideoInfo(
                path=video_path,
                duration=duration,
                width=int(video_stream.get('width', 0)),
                height=int(video_stream.get('height', 0)),
                codec=video_stream.get('codec_name', 'unknown'),
                audio_codec=audio_stream.get('codec_name', 'unknown') if audio_stream else 'none',
                frame_rate=eval(video_stream.get('avg_frame_rate', '0/1')) if '/' in video_stream.get('avg_frame_rate', '0/1') else 0,
                bitrate=int(data['format'].get('bit_rate', 0))
            )
            
            logger.info(f"Video analyzed: {duration:.1f}s, {video_info.width}x{video_info.height}, {video_info.codec}")
            return video_info
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            raise
    
    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        """Extract audio from video file with subtitle generation"""
        try:
            logger.info(f"Extracting audio from {video_path}")
            
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '44100', '-ac', '2',
                '-loglevel', 'error',
                audio_path, '-y'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Audio extraction failed: {result.stderr}")
                return False
            
            # Verify audio file was created
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                logger.info(f"Audio extracted successfully: {audio_path}")
                return True
            else:
                logger.error("Audio extraction failed: Output file is empty")
                return False
                
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return False
    
    def generate_original_subtitles(self, video_path: str, output_base: str) -> Dict[str, Any]:
        """Generate subtitles for original video"""
        try:
            if not self.subtitle_generator:
                self.subtitle_generator = SubtitleGenerator()
            
            logger.info("Generating original subtitles...")
            
            # Extract audio for subtitle generation
            audio_path = os.path.join(self.config.temp_dir, f"subtitle_audio_{uuid.uuid4()}.wav")
            if not self.extract_audio(video_path, audio_path):
                return {"success": False, "error": "Audio extraction failed"}
            
            # Generate subtitles
            result = self.subtitle_generator.process_file(
                audio_path,
                output_format="srt",
                language=None,  # Auto-detect
                generate_all=True
            )
            
            # Cleanup temp audio
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            if not result["success"]:
                return {"success": False, "error": result.get("error", "Subtitle generation failed")}
            
            # Move subtitle files to output directory
            subtitle_files = {}
            for format_name, file_path in result["output_files"].items():
                new_path = f"{output_base}_original.{format_name}"
                if os.path.exists(file_path):
                    shutil.move(file_path, new_path)
                    subtitle_files[format_name] = new_path
            
            logger.info(f"Generated original subtitles: {list(subtitle_files.keys())}")
            
            return {
                "success": True,
                "files": subtitle_files,
                "segments": result["segments"],
                "language": result["language"]
            }
            
        except Exception as e:
            logger.error(f"Original subtitle generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def chunk_audio(self, audio_path: str, chunk_size: int = None) -> List[ChunkInfo]:
        """Split audio into chunks with speech detection"""
        try:
            chunk_size = chunk_size or self.config.chunk_size
            logger.info(f"Chunking audio into {chunk_size}s segments")
            
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            duration_ms = len(audio)
            
            chunks = []
            chunk_id = 0
            
            for start_ms in range(0, duration_ms, chunk_size * 1000):
                end_ms = min(start_ms + chunk_size * 1000, duration_ms)
                
                # Extract chunk
                chunk = audio[start_ms:end_ms]
                
                # Save chunk
                chunk_path = os.path.join(self.config.temp_dir, f"chunk_{chunk_id:04d}.wav")
                chunk.export(chunk_path, format="wav")
                
                # Check if chunk has speech
                has_speech = self._has_speech(chunk)
                
                chunk_info = ChunkInfo(
                    id=chunk_id,
                    path=chunk_path,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=end_ms - start_ms,
                    has_speech=has_speech
                )
                
                chunks.append(chunk_info)
                chunk_id += 1
            
            logger.info(f"Created {len(chunks)} audio chunks ({sum(1 for c in chunks if c.has_speech)} with speech)")
            return chunks
            
        except Exception as e:
            logger.error(f"Audio chunking failed: {e}")
            return []
    
    def _has_speech(self, audio_segment: AudioSegment, threshold: float = 0.02) -> bool:
        """Improved speech detection using energy and zero-crossing rate"""
        try:
            # Convert to numpy array
            samples = np.array(audio_segment.get_array_of_samples())
            
            # Normalize
            samples = samples.astype(np.float32) / (2**15)
            
            # Calculate RMS energy
            rms = np.sqrt(np.mean(samples**2))
            
            # Calculate zero-crossing rate
            zero_crossings = np.sum(np.diff(np.sign(samples)) != 0) / len(samples)
            
            # Speech typically has higher energy and moderate zero-crossing rate
            return rms > threshold and zero_crossings > 0.01
            
        except:
            return True  # Assume speech by default
    
    def process_chunk_with_subtitles(self, chunk: ChunkInfo, target_lang: str = "de") -> Tuple[Optional[str], ProcessingMetrics, List[Dict]]:
        """Process a single audio chunk with subtitle generation"""
        start_time = datetime.now()
        subtitle_segments = []
        
        try:
            logger.info(f"Processing chunk {chunk.id} ({chunk.duration_ms:.0f}ms)")
            
            # Skip chunks without speech
            if not chunk.has_speech:
                logger.info(f"Chunk {chunk.id} has no speech, skipping")
                
                # Create silent audio for this chunk
                output_path = os.path.join(self.config.temp_dir, f"translated_chunk_{chunk.id:04d}.wav")
                silent_audio = AudioSegment.silent(duration=chunk.duration_ms)
                silent_audio.export(output_path, format="wav")
                
                metrics = ProcessingMetrics(
                    chunk_id=chunk.id,
                    original_duration_ms=chunk.duration_ms,
                    translated_duration_ms=chunk.duration_ms,
                    duration_diff_ms=0,
                    condensation_ratio=1.0,
                    speed_adjustment=1.0,
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    success=True,
                    subtitle_generated=False
                )
                
                return output_path, metrics, []
            
            # Update translator target language
            self.translator.config.target_lang = target_lang
            
            # Process audio chunk
            result = self.translator.process_audio(chunk.path)
            
            # Calculate processing time
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            if result.success:
                # Move output to temp directory
                new_output_path = os.path.join(self.config.temp_dir, f"translated_chunk_{chunk.id:04d}.wav")
                shutil.move(result.output_path, new_output_path)
                
                # Calculate condensation ratio
                original_words = len(result.original_text.split())
                translated_words = len(result.translated_text.split())
                condensation_ratio = translated_words / original_words if original_words > 0 else 1.0
                
                # Extract subtitle segments
                subtitle_segments = result.timing_segments if result.timing_segments else []
                
                metrics = ProcessingMetrics(
                    chunk_id=chunk.id,
                    original_duration_ms=result.original_duration_ms,
                    translated_duration_ms=result.translated_duration_ms,
                    duration_diff_ms=abs(result.translated_duration_ms - result.original_duration_ms),
                    condensation_ratio=condensation_ratio,
                    speed_adjustment=result.speed_adjustment,
                    processing_time_ms=processing_time_ms,
                    success=True,
                    subtitle_generated=bool(subtitle_segments)
                )
                
                logger.info(f"Chunk {chunk.id} processed successfully")
                logger.info(f"  Duration match: {result.duration_match_percent:.1f}%")
                logger.info(f"  Condensation ratio: {condensation_ratio:.2f}")
                logger.info(f"  Speed adjustment: {result.speed_adjustment:.2f}x")
                logger.info(f"  Subtitles: {len(subtitle_segments)} segments")
                
                return new_output_path, metrics, subtitle_segments
            else:
                logger.error(f"Chunk {chunk.id} processing failed: {result.error}")
                
                # Create fallback silent audio
                output_path = os.path.join(self.config.temp_dir, f"translated_chunk_{chunk.id:04d}.wav")
                silent_audio = AudioSegment.silent(duration=chunk.duration_ms)
                silent_audio.export(output_path, format="wav")
                
                metrics = ProcessingMetrics(
                    chunk_id=chunk.id,
                    original_duration_ms=chunk.duration_ms,
                    translated_duration_ms=chunk.duration_ms,
                    duration_diff_ms=0,
                    condensation_ratio=1.0,
                    speed_adjustment=1.0,
                    processing_time_ms=processing_time_ms,
                    success=False,
                    subtitle_generated=False,
                    error=result.error
                )
                
                return output_path, metrics, []
                
        except Exception as e:
            logger.error(f"Chunk {chunk.id} processing failed with exception: {e}")
            
            # Create fallback silent audio
            output_path = os.path.join(self.config.temp_dir, f"translated_chunk_{chunk.id:04d}.wav")
            silent_audio = AudioSegment.silent(duration=chunk.duration_ms)
            silent_audio.export(output_path, format="wav")
            
            processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            metrics = ProcessingMetrics(
                chunk_id=chunk.id,
                original_duration_ms=chunk.duration_ms,
                translated_duration_ms=chunk.duration_ms,
                duration_diff_ms=0,
                condensation_ratio=1.0,
                speed_adjustment=1.0,
                processing_time_ms=processing_time_ms,
                success=False,
                subtitle_generated=False,
                error=str(e)
            )
            
            return output_path, metrics, []
    
    def generate_final_subtitles(self, original_subtitles: Dict, translated_segments: List[List[Dict]], 
                                 video_info: VideoInfo, output_base: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """Generate final bilingual subtitles"""
        try:
            if not self.config.generate_subtitles:
                return {"success": False, "error": "Subtitle generation disabled"}
            
            logger.info("Generating final bilingual subtitles...")
            
            # Flatten translated segments
            all_translated_segments = []
            for segment_list in translated_segments:
                all_translated_segments.extend(segment_list)
            
            # Generate bilingual subtitles
            if self.config.bilingual_subtitles and self.subtitle_generator:
                subtitle_result = self.subtitle_generator.generate_subtitles_for_translation(
                    video_path=video_info.path,
                    original_segments=original_subtitles.get("segments", []),
                    translated_segments=all_translated_segments,
                    original_lang=source_lang,
                    target_lang=target_lang,
                    output_base=output_base
                )
                
                if subtitle_result.get("success"):
                    logger.info(f"Generated bilingual subtitles: {list(subtitle_result.keys())}")
                    return subtitle_result
            
            # Fallback: generate only translated subtitles
            translated_srt = self.subtitle_generator.format_to_srt(all_translated_segments)
            translated_path = f"{output_base}_{target_lang}.srt"
            
            with open(translated_path, 'w', encoding='utf-8') as f:
                f.write(translated_srt)
            
            logger.info(f"Generated translated subtitles: {translated_path}")
            
            return {
                "success": True,
                "translated": translated_path,
                "bilingual": None
            }
            
        except Exception as e:
            logger.error(f"Final subtitle generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def merge_audio_chunks(self, chunk_paths: List[str], output_path: str) -> bool:
        """Merge audio chunks into single audio file"""
        try:
            logger.info(f"Merging {len(chunk_paths)} audio chunks")
            
            if not chunk_paths:
                logger.error("No audio chunks to merge")
                return False
            
            # Start with first chunk
            merged_audio = AudioSegment.from_file(chunk_paths[0])
            
            # Append remaining chunks
            for chunk_path in chunk_paths[1:]:
                if os.path.exists(chunk_path):
                    chunk_audio = AudioSegment.from_file(chunk_path)
                    merged_audio += chunk_audio
            
            # Export merged audio
            merged_audio.export(output_path, format="wav")
            
            logger.info(f"Merged audio saved: {output_path} ({len(merged_audio):.0f}ms)")
            return True
            
        except Exception as e:
            logger.error(f"Audio merging failed: {e}")
            return False
    
    def embed_subtitles_in_video(self, video_path: str, subtitle_path: str, output_path: str) -> bool:
        """Embed subtitles into video file"""
        try:
            if not os.path.exists(subtitle_path):
                logger.warning(f"Subtitle file not found: {subtitle_path}")
                return False
            
            logger.info(f"Embedding subtitles into video: {subtitle_path}")
            
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', subtitle_path,
                '-c:v', 'copy',
                '-c:a', 'copy',
                '-c:s', 'mov_text',  # MP4 compatible subtitles
                '-metadata:s:s:0', f'language={os.path.splitext(subtitle_path)[0][-2:]}',
                '-loglevel', 'error',
                output_path, '-y'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Subtitle embedding failed: {result.stderr}")
                return False
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Video with embedded subtitles: {output_path}")
                return True
            else:
                logger.error("Subtitle embedding failed: Output file is empty")
                return False
                
        except Exception as e:
            logger.error(f"Subtitle embedding failed: {e}")
            return False
    
    def merge_audio_with_video(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Merge translated audio with original video"""
        try:
            logger.info(f"Merging audio with video: {video_path}")
            
            cmd = [
                'ffmpeg',
                '-i', video_path,  # Original video
                '-i', audio_path,  # Translated audio
                '-c:v', 'copy',  # Copy video stream
                '-c:a', 'aac',  # Encode audio as AAC
                '-b:a', '192k',  # Audio bitrate
                '-map', '0:v:0',  # Use video from first input
                '-map', '1:a:0',  # Use audio from second input
                '-shortest',  # End when shortest stream ends
                '-loglevel', 'error',
                output_path, '-y'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Video/audio merge failed: {result.stderr}")
                return False
            
            # Verify output file
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Final video created: {output_path}")
                return True
            else:
                logger.error("Video/audio merge failed: Output file is empty")
                return False
                
        except Exception as e:
            logger.error(f"Video/audio merge failed: {e}")
            return False
    
    def verify_duration_match(self, original_path: str, translated_path: str, tolerance_ms: int = 100) -> Tuple[bool, float]:
        """Verify that translated video duration matches original"""
        try:
            # Get original duration
            cmd_original = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', original_path
            ]
            
            result_original = subprocess.run(cmd_original, capture_output=True, text=True)
            original_duration = float(result_original.stdout.strip())
            
            # Get translated duration
            cmd_translated = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', translated_path
            ]
            
            result_translated = subprocess.run(cmd_translated, capture_output=True, text=True)
            translated_duration = float(result_translated.stdout.strip())
            
            # Calculate difference
            duration_diff_ms = abs(translated_duration - original_duration) * 1000
            within_tolerance = duration_diff_ms <= tolerance_ms
            
            logger.info(f"Duration verification:")
            logger.info(f"  Original: {original_duration:.3f}s")
            logger.info(f"  Translated: {translated_duration:.3f}s")
            logger.info(f"  Difference: {duration_diff_ms:.0f}ms")
            logger.info(f"  Within tolerance ({tolerance_ms}ms): {'✓' if within_tolerance else '✗'}")
            
            return within_tolerance, duration_diff_ms
            
        except Exception as e:
            logger.error(f"Duration verification failed: {e}")
            return False, 0
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.config.temp_dir):
                for file in os.listdir(self.config.temp_dir):
                    file_path = os.path.join(self.config.temp_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except:
                        pass
                
                logger.info(f"Cleaned up temp directory: {self.config.temp_dir}")
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def process_video(self, video_path: str, target_lang: str = "de") -> Dict[str, Any]:
        """Complete video translation pipeline with subtitle generation"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting video translation pipeline")
            logger.info(f"  Input: {video_path}")
            logger.info(f"  Target language: {target_lang}")
            logger.info(f"  Generate subtitles: {self.config.generate_subtitles}")
            logger.info(f"  Temp directory: {self.config.temp_dir}")
            
            # Step 1: Load models
            logger.info("\n[1/8] Loading AI models...")
            if not self.load_models(target_lang=target_lang):
                return {
                    "success": False,
                    "error": "Failed to load AI models",
                    "processing_time_s": 0
                }
            
            # Step 2: Analyze video
            logger.info("\n[2/8] Analyzing video...")
            video_info = self.analyze_video(video_path)
            
            # Step 3: Generate original subtitles
            original_subtitles = None
            if self.config.generate_subtitles:
                logger.info("\n[3/8] Generating original subtitles...")
                original_subtitles = self.generate_original_subtitles(
                    video_path, 
                    os.path.join(self.config.output_dir, os.path.basename(video_path).rsplit('.', 1)[0])
                )
                if original_subtitles.get("success"):
                    logger.info(f"Original subtitles generated: {original_subtitles.get('language', 'unknown')}")
                else:
                    logger.warning(f"Original subtitle generation failed: {original_subtitles.get('error')}")
            
            # Step 4: Extract audio
            logger.info("\n[4/8] Extracting audio...")
            audio_path = os.path.join(self.config.temp_dir, "original_audio.wav")
            if not self.extract_audio(video_path, audio_path):
                return {
                    "success": False,
                    "error": "Failed to extract audio from video",
                    "processing_time_s": (datetime.now() - start_time).total_seconds()
                }
            
            # Step 5: Chunk audio
            logger.info("\n[5/8] Chunking audio...")
            chunks = self.chunk_audio(audio_path)
            
            if not chunks:
                return {
                    "success": False,
                    "error": "Failed to chunk audio",
                    "processing_time_s": (datetime.now() - start_time).total_seconds()
                }
            
            # Step 6: Process each chunk with subtitles
            logger.info(f"\n[6/8] Processing {len(chunks)} audio chunks...")
            translated_chunk_paths = []
            all_metrics = []
            all_subtitle_segments = []
            detected_source_lang = "en"  # Default
            
            for chunk in chunks:
                translated_path, metrics, subtitle_segments = self.process_chunk_with_subtitles(chunk, target_lang)
                
                if translated_path:
                    translated_chunk_paths.append(translated_path)
                
                all_metrics.append(metrics)
                all_subtitle_segments.append(subtitle_segments)
                
                # Update detected source language from first successful chunk
                if metrics.success and not detected_source_lang:
                    detected_source_lang = self.translator.config.source_lang
                
                # Log progress
                progress = (chunk.id + 1) / len(chunks) * 100
                logger.info(f"  Progress: {progress:.0f}% ({chunk.id + 1}/{len(chunks)})")
            
            # Calculate overall metrics
            successful_chunks = sum(1 for m in all_metrics if m.success)
            success_rate = successful_chunks / len(all_metrics) if all_metrics else 0
            
            avg_duration_diff = np.mean([m.duration_diff_ms for m in all_metrics if m.success]) if successful_chunks > 0 else 0
            avg_condensation = np.mean([m.condensation_ratio for m in all_metrics if m.success]) if successful_chunks > 0 else 0
            
            # Step 7: Merge audio chunks
            logger.info("\n[7/8] Merging audio chunks...")
            merged_audio_path = os.path.join(self.config.temp_dir, "merged_audio.wav")
            
            if not self.merge_audio_chunks(translated_chunk_paths, merged_audio_path):
                return {
                    "success": False,
                    "error": "Failed to merge audio chunks",
                    "processing_time_s": (datetime.now() - start_time).total_seconds(),
                    "metrics": {
                        "total_chunks": len(chunks),
                        "successful_chunks": successful_chunks,
                        "success_rate": success_rate,
                        "avg_duration_diff_ms": avg_duration_diff,
                        "avg_condensation_ratio": avg_condensation
                    }
                }
            
            # Step 8: Merge with video
            logger.info("\n[8/8] Merging audio with video...")
            output_filename = f"translated_{os.path.basename(video_path)}"
            output_base = os.path.join(self.config.output_dir, output_filename.rsplit('.', 1)[0])
            output_path = f"{output_base}.mp4"
            
            if not self.merge_audio_with_video(video_path, merged_audio_path, output_path):
                return {
                    "success": False,
                    "error": "Failed to merge audio with video",
                    "processing_time_s": (datetime.now() - start_time).total_seconds(),
                    "metrics": {
                        "total_chunks": len(chunks),
                        "successful_chunks": successful_chunks,
                        "success_rate": success_rate,
                        "avg_duration_diff_ms": avg_duration_diff,
                        "avg_condensation_ratio": avg_condensation
                    }
                }
            
            # Step 9: Generate final subtitles
            subtitle_files = {}
            if self.config.generate_subtitles:
                logger.info("\n[9/9] Generating final subtitles...")
                subtitle_result = self.generate_final_subtitles(
                    original_subtitles if original_subtitles else {},
                    all_subtitle_segments,
                    video_info,
                    output_base,
                    detected_source_lang,
                    target_lang
                )
                
                if subtitle_result.get("success"):
                    subtitle_files = {k: v for k, v in subtitle_result.items() if k != "success" and v}
                    
                    # Optionally embed subtitles into video
                    if "translated" in subtitle_files:
                        embedded_video_path = f"{output_base}_with_subs.mp4"
                        if self.embed_subtitles_in_video(output_path, subtitle_files["translated"], embedded_video_path):
                            output_path = embedded_video_path
                            logger.info(f"Created video with embedded subtitles: {output_path}")
            
            # Step 10: Verify duration match
            logger.info("\n[10/10] Verifying duration match...")
            duration_match, duration_diff = self.verify_duration_match(
                video_path, 
                output_path,
                tolerance_ms=100
            )
            
            # Calculate total processing time
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare result
            result = {
                "success": True,
                "input_video": video_path,
                "output_video": output_path,
                "target_language": target_lang,
                "source_language": detected_source_lang,
                "processing_time_s": total_time,
                "duration_match": duration_match,
                "duration_diff_ms": duration_diff,
                "subtitle_files": subtitle_files,
                "video_info": {
                    "duration": video_info.duration,
                    "resolution": f"{video_info.width}x{video_info.height}",
                    "codec": video_info.codec,
                    "frame_rate": video_info.frame_rate
                },
                "metrics": {
                    "total_chunks": len(chunks),
                    "successful_chunks": successful_chunks,
                    "success_rate": success_rate,
                    "avg_duration_diff_ms": avg_duration_diff,
                    "avg_condensation_ratio": avg_condensation,
                    "within_tolerance_percentage": (sum(1 for m in all_metrics if m.success and m.duration_diff_ms <= self.config.timing_tolerance_ms) / len(all_metrics)) * 100 if all_metrics else 0,
                    "subtitle_segments": sum(len(seg) for seg in all_subtitle_segments)
                }
            }
            
            # Log summary
            logger.info(f"\n{'='*60}")
            logger.info("TRANSLATION COMPLETED SUCCESSFULLY")
            logger.info(f"{'='*60}")
            logger.info(f"Input: {video_path}")
            logger.info(f"Output: {output_path}")
            logger.info(f"Source → Target: {detected_source_lang} → {target_lang}")
            logger.info(f"Processing time: {total_time:.1f}s")
            logger.info(f"Duration match: {'✓' if duration_match else '✗'} ({duration_diff:.0f}ms diff)")
            logger.info(f"Chunks: {successful_chunks}/{len(chunks)} successful ({success_rate:.1%})")
            logger.info(f"Subtitles: {len(subtitle_files)} files generated")
            logger.info(f"Avg duration diff: {avg_duration_diff:.1f}ms")
            logger.info(f"Avg condensation: {avg_condensation:.2f}x")
            logger.info(f"{'='*60}")
            
            # Cleanup
            self.cleanup_temp_files()
            
            return result
            
        except Exception as e:
            logger.error(f"Video translation pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Cleanup on error
            self.cleanup_temp_files()
            
            return {
                "success": False,
                "error": str(e),
                "processing_time_s": (datetime.now() - start_time).total_seconds() if 'start_time' in locals() else 0
            }

# Command-line interface
def main():
    """Command-line interface for video translation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Octavia Video Translator")
    parser.add_argument("--input", "-i", required=True, help="Input video file")
    parser.add_argument("--output", "-o", help="Output video file (optional)")
    parser.add_argument("--target", "-t", default="de", help="Target language (en, de, ru, es, fr)")
    parser.add_argument("--chunk-size", "-c", type=int, default=30, help="Chunk size in seconds")
    parser.add_argument("--subtitles", "-s", action="store_true", default=True, help="Generate subtitles")
    parser.add_argument("--no-subtitles", action="store_false", dest="subtitles", help="Don't generate subtitles")
    parser.add_argument("--bilingual", "-b", action="store_true", default=True, help="Generate bilingual subtitles")
    parser.add_argument("--cleanup", action="store_true", help="Clean up temporary files")
    
    args = parser.parse_args()
    
    # Configure pipeline
    config = PipelineConfig(
        chunk_size=args.chunk_size,
        temp_dir="temp",
        output_dir="outputs",
        generate_subtitles=args.subtitles,
        bilingual_subtitles=args.bilingual
    )
    
    pipeline = VideoTranslationPipeline(config)
    
    print(f"\n{'='*60}")
    print(f"Octavia Video Translator")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Target language: {args.target}")
    print(f"Chunk size: {args.chunk_size}s")
    print(f"Generate subtitles: {args.subtitles}")
    print(f"Bilingual subtitles: {args.bilingual}")
    print(f"{'='*60}\n")
    
    # Process video
    result = pipeline.process_video(args.input, args.target)
    
    # Print result
    print(f"\n{'='*60}")
    print("RESULT:")
    print(f"{'='*60}")
    
    if result["success"]:
        print(f"✓ SUCCESS")
        print(f"  Output: {result['output_video']}")
        print(f"  Processing time: {result['processing_time_s']:.1f}s")
        print(f"  Duration match: {'Yes' if result['duration_match'] else 'No'} ({result['duration_diff_ms']:.0f}ms)")
        print(f"  Successful chunks: {result['metrics']['successful_chunks']}/{result['metrics']['total_chunks']}")
        print(f"  Average condensation: {result['metrics']['avg_condensation_ratio']:.2f}x")
        
        if result.get('subtitle_files'):
            print(f"  Subtitles generated:")
            for name, path in result['subtitle_files'].items():
                if path:
                    print(f"    - {name}: {os.path.basename(path)}")
    else:
        print(f"✗ FAILED")
        print(f"  Error: {result.get('error', 'Unknown error')}")
    
    print(f"{'='*60}\n")
    
    # Cleanup if requested
    if args.cleanup:
        pipeline.cleanup_temp_files()
    
    return 0 if result["success"] else 1

if __name__ == "__main__":
    sys.exit(main())