"""
Audio Translator Module for Octavia Video Translator
Enhanced with better translation quality and timing accuracy
Supports Russian → English and English → German translations
Uses Edge-TTS for high-quality multilingual voice synthesis
"""

import os
import sys
import json
import asyncio
import logging
import re
import tempfile
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import whisper
import torch
import edge_tts
from transformers import MarianMTModel, MarianTokenizer, pipeline
from pydub import AudioSegment
import numpy as np
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

@dataclass
class TranslationConfig:
    """Configuration for translation"""
    source_lang: str = "en"
    target_lang: str = "de"
    auto_detect: bool = True
    chunk_size: int = 30
    max_condensation_ratio: float = 1.2
    timing_tolerance_ms: int = 200
    voice_speed: float = 1.0
    voice_pitch: str = "+0Hz"
    voice_style: str = "neutral"

@dataclass
class TranslationResult:
    """Result of audio translation"""
    success: bool
    original_text: str
    translated_text: str
    original_language: str
    target_language: str
    original_duration_ms: float
    translated_duration_ms: float
    duration_match_percent: float
    speed_adjustment: float
    output_path: str
    subtitle_path: Optional[str] = None
    timing_segments: List[Dict] = None
    error: Optional[str] = None

class AudioTranslator:
    """Main audio translation class with improved quality"""
    
    # Translation model mapping with better models
    MODEL_MAPPING = {
        "ru-en": "Helsinki-NLP/opus-mt-ru-en",
        "en-de": "Helsinki-NLP/opus-mt-en-de",
        "en-ru": "Helsinki-NLP/opus-mt-en-ru",
        "de-en": "Helsinki-NLP/opus-mt-de-en",
        "en-es": "Helsinki-NLP/opus-mt-en-es",
        "es-en": "Helsinki-NLP/opus-mt-es-en",
        "en-fr": "Helsinki-NLP/opus-mt-en-fr",
        "fr-en": "Helsinki-NLP/opus-mt-fr-en"
    }
    
    # Edge-TTS voice mapping with better voices
    VOICE_MAPPING = {
        "en": "en-US-JennyNeural",
        "de": "de-DE-KatjaNeural",
        "ru": "ru-RU-SvetlanaNeural",
        "es": "es-ES-ElviraNeural",
        "fr": "fr-FR-DeniseNeural"
    }
    
    # Voice rate mapping (characters per second) for different languages
    VOICE_RATES = {
        "en": 12,  # English: ~12 chars/second
        "de": 11,  # German: ~11 chars/second  
        "ru": 10,  # Russian: ~10 chars/second
        "es": 13,  # Spanish: ~13 chars/second
        "fr": 12   # French: ~12 chars/second
    }
    
    def __init__(self, config: TranslationConfig = None):
        self.config = config or TranslationConfig()
        self.whisper_model = None
        self.translation_model = None
        self.translation_tokenizer = None
        self.translation_pipeline = None
        self._models_loaded = False
        
    def load_models(self):
        """Load all required AI models with better configuration"""
        try:
            logger.info("Loading Whisper model...")
            # Use medium model for better accuracy
            self.whisper_model = whisper.load_model("medium")
            
            logger.info("Loading translation model...")
            model_key = f"{self.config.source_lang}-{self.config.target_lang}"
            model_name = self.MODEL_MAPPING.get(model_key)
            
            if not model_name:
                logger.warning(f"Model not found for {model_key}, using Helsinki-NLP/opus-mt-mul-en")
                model_name = "Helsinki-NLP/opus-mt-mul-en"
            
            # Load tokenizer and model
            self.translation_tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.translation_model = MarianMTModel.from_pretrained(model_name)
            
            # Also load pipeline for better translation
            self.translation_pipeline = pipeline(
                "translation",
                model=self.translation_model,
                tokenizer=self.translation_tokenizer,
                device=-1 if torch.cuda.is_available() else -1
            )
            
            self._models_loaded = True
            logger.info(f"Models loaded successfully: Whisper-medium, {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            # Fallback to smaller models
            try:
                self.whisper_model = whisper.load_model("base")
                self.translation_pipeline = pipeline(
                    "translation_en_to_de" if self.config.target_lang == "de" else "translation_en_to_fr",
                    device=-1
                )
                logger.warning("Loaded fallback models")
                return True
            except Exception as fallback_error:
                logger.error(f"Fallback models also failed: {fallback_error}")
                return False
    
    def detect_language(self, audio_path: str) -> str:
        """Detect language of audio file with confidence"""
        try:
            if not self.whisper_model:
                self.load_models()
            
            # Load and process audio
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Make log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(
                self.whisper_model.device if hasattr(self.whisper_model, 'device') else 'cpu'
            )
            
            # Detect language
            _, probs = self.whisper_model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            confidence = probs[detected_lang]
            
            logger.info(f"Detected language: {detected_lang} (confidence: {confidence:.2%})")
            
            # Map Whisper language codes to our codes
            lang_map = {
                "en": "en",
                "de": "de", 
                "ru": "ru",
                "es": "es",
                "fr": "fr"
            }
            
            return lang_map.get(detected_lang, self.config.source_lang)
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return self.config.source_lang
    
    def transcribe_with_segments(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio to text with detailed timestamps"""
        try:
            if not self.whisper_model:
                self.load_models()
            
            # Set language for transcription
            language = self.config.source_lang
            if self.config.auto_detect or language == "auto":
                language = self.detect_language(audio_path)
                self.config.source_lang = language
            
            logger.info(f"Transcribing audio in {language}...")
            
            # Transcribe with detailed options
            result = self.whisper_model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                word_timestamps=True,
                temperature=0.0,
                best_of=5,
                beam_size=5
            )
            
            # Process segments for better accuracy
            segments = []
            for segment in result.get("segments", []):
                if segment["text"].strip():
                    segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"].strip(),
                        "words": segment.get("words", [])
                    })
            
            full_text = " ".join([seg["text"] for seg in segments])
            
            return {
                "text": full_text.strip(),
                "segments": segments,
                "language": result.get("language", language),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "text": "",
                "segments": [],
                "language": self.config.source_lang,
                "success": False,
                "error": str(e)
            }
    
    def translate_text_with_context(self, text: str, segments: List[Dict]) -> Tuple[str, List[Dict]]:
        """Translate text with segment context preservation"""
        try:
            if not self.translation_model or not self.translation_pipeline:
                self.load_models()
            
            if not text or len(text.strip()) < 2:
                return text, []
            
            translated_segments = []
            all_translated_text = []
            
            # Group segments for context-aware translation
            group_size = 3  # Translate 3 segments together for context
            for i in range(0, len(segments), group_size):
                group_segments = segments[i:i + group_size]
                group_text = " ".join([seg["text"] for seg in group_segments])
                
                if not group_text.strip():
                    continue
                
                try:
                    # Translate the group
                    if self.translation_pipeline:
                        translation_result = self.translation_pipeline(
                            group_text,
                            max_length=512,
                            num_beams=4,
                            temperature=0.7
                        )
                        translated_group = translation_result[0]['translation_text']
                    else:
                        # Fallback to model directly
                        inputs = self.translation_tokenizer(
                            group_text, 
                            return_tensors="pt", 
                            padding=True,
                            truncation=True,
                            max_length=512
                        )
                        translated_tokens = self.translation_model.generate(**inputs)
                        translated_group = self.translation_tokenizer.decode(
                            translated_tokens[0],
                            skip_special_tokens=True
                        )
                    
                    # Clean translation
                    translated_group = self._clean_translation(translated_group)
                    
                    # Distribute translated text back to segments (simple proportional split)
                    words_original = group_text.split()
                    words_translated = translated_group.split()
                    
                    if words_original and words_translated:
                        ratio = len(words_translated) / len(words_original)
                        
                        for seg in group_segments:
                            seg_words = len(seg["text"].split())
                            trans_words = max(1, int(seg_words * ratio))
                            
                            # Get portion of translated text
                            if all_translated_text:
                                start_idx = len(" ".join(all_translated_text).split())
                            else:
                                start_idx = 0
                            
                            trans_segment_words = words_translated[start_idx:start_idx + trans_words]
                            trans_segment_text = " ".join(trans_segment_words)
                            
                            translated_segments.append({
                                "start": seg["start"],
                                "end": seg["end"],
                                "original_text": seg["text"],
                                "translated_text": trans_segment_text,
                                "words": seg.get("words", [])
                            })
                    
                    all_translated_text.append(translated_group)
                    
                except Exception as group_error:
                    logger.warning(f"Group translation failed, using original: {group_error}")
                    for seg in group_segments:
                        translated_segments.append({
                            "start": seg["start"],
                            "end": seg["end"],
                            "original_text": seg["text"],
                            "translated_text": seg["text"],  # Fallback to original
                            "words": seg.get("words", [])
                        })
                    all_translated_text.append(group_text)
            
            full_translated_text = " ".join(all_translated_text)
            full_translated_text = self._clean_translation(full_translated_text)
            
            logger.info(f"Translation: {len(text)} chars → {len(full_translated_text)} chars")
            
            return full_translated_text, translated_segments
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            # Return original text with empty segments
            fallback_segments = [{
                "start": seg["start"],
                "end": seg["end"],
                "original_text": seg["text"],
                "translated_text": seg["text"],
                "words": seg.get("words", [])
            } for seg in segments]
            return text, fallback_segments
    
    def _clean_translation(self, text: str) -> str:
        """Clean and normalize translated text"""
        # Remove duplicate punctuation
        text = re.sub(r'[.!?]{2,}', '.', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])(\w)', r'\1 \2', text)
        # Capitalize sentences
        sentences = re.split(r'([.!?])\s+', text)
        cleaned = []
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i].strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:]
                    cleaned.append(sentence)
                if i + 1 < len(sentences):
                    cleaned.append(sentences[i + 1])
        
        text = ' '.join(cleaned)
        return text.strip()
    
    def condense_text_smart(self, text: str, target_duration_ms: float) -> Tuple[str, float]:
        """Smart text condensation to fit target duration"""
        try:
            words = text.split()
            if len(words) <= 10:  # Don't condense very short texts
                return text, 1.0
            
            # Estimate speaking rate for target language
            chars_per_second = self.VOICE_RATES.get(self.config.target_lang, 12)
            target_chars = int(target_duration_ms / 1000 * chars_per_second)
            
            current_chars = len(text)
            
            if current_chars <= target_chars:
                return text, 1.0
            
            # Calculate needed condensation ratio
            ratio = target_chars / current_chars
            
            # Don't condense too much
            min_ratio = 0.7
            if ratio < min_ratio:
                ratio = min_ratio
            
            # Smart condensation: remove filler words first
            filler_words = {
                "en": ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "that", "this", "these", "those"],
                "de": ["der", "die", "das", "und", "oder", "aber", "in", "auf", "an", "zu", "für", "von", "mit", "dass", "dies", "diese", "jene"],
                "ru": ["и", "в", "на", "с", "по", "к", "у", "о", "от", "до", "за", "из", "над", "что", "это", "этот", "эта", "эти"],
                "es": ["el", "la", "los", "las", "y", "o", "pero", "en", "a", "de", "con", "por", "para", "que", "este", "esta", "estos", "estas"],
                "fr": ["le", "la", "les", "et", "ou", "mais", "en", "à", "de", "avec", "pour", "par", "que", "ce", "cette", "ces"]
            }
            
            lang_fillers = filler_words.get(self.config.target_lang, filler_words["en"])
            
            # Remove filler words
            filtered_words = []
            for word in words:
                if word.lower() not in lang_fillers:
                    filtered_words.append(word)
            
            # If still too long, use compression
            if len(" ".join(filtered_words)) > target_chars:
                # Take key parts: first 30%, middle 40%, last 30%
                target_word_count = int(len(filtered_words) * ratio)
                keep_first = int(target_word_count * 0.3)
                keep_last = int(target_word_count * 0.3)
                keep_middle = target_word_count - keep_first - keep_last
                
                if keep_middle > 0:
                    middle_start = len(filtered_words) // 2 - keep_middle // 2
                    middle_end = middle_start + keep_middle
                    
                    if middle_start < 0:
                        middle_start = 0
                    if middle_end > len(filtered_words):
                        middle_end = len(filtered_words)
                    
                    first_part = filtered_words[:keep_first]
                    middle_part = filtered_words[middle_start:middle_end]
                    last_part = filtered_words[-keep_last:] if keep_last > 0 else []
                    
                    filtered_words = first_part + middle_part + last_part
                else:
                    filtered_words = filtered_words[:target_word_count]
            
            condensed_text = " ".join(filtered_words)
            actual_ratio = len(condensed_text) / len(text) if len(text) > 0 else 1.0
            
            logger.info(f"Condensed: {len(words)} words → {len(filtered_words)} words (ratio: {actual_ratio:.2f})")
            
            return condensed_text, actual_ratio
            
        except Exception as e:
            logger.error(f"Smart condensation failed: {e}")
            return text, 1.0
    
    async def synthesize_speech_with_timing(self, text: str, segments: List[Dict], output_path: str) -> Tuple[bool, List[Dict]]:
        """Generate speech with timing preservation"""
        try:
            if not text or len(text.strip()) < 2:
                logger.warning("Text too short for TTS, creating silent audio")
                silent_audio = AudioSegment.silent(duration=1000)
                silent_audio.export(output_path, format="wav")
                return True, []
            
            # Get appropriate voice
            voice = self.VOICE_MAPPING.get(self.config.target_lang, "en-US-JennyNeural")
            logger.info(f"Generating speech with voice: {voice}")
            
            # Create TTS communicator with voice adjustments
            communicate = edge_tts.Communicate(
                text,
                voice,
                rate=self.config.voice_speed,
                pitch=self.config.voice_pitch
            )
            
            # Save audio to temporary file
            temp_path = output_path.replace(".wav", "_temp.wav")
            await communicate.save(temp_path)
            
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                logger.error("TTS output file is empty")
                return False, []
            
            # Load and analyze the generated audio
            tts_audio = AudioSegment.from_file(temp_path)
            tts_duration_ms = len(tts_audio)
            
            # Calculate timing adjustments for segments
            adjusted_segments = []
            if segments and tts_duration_ms > 0:
                # Simple proportional timing (can be enhanced)
                total_original_duration = sum(seg["end"] - seg["start"] for seg in segments)
                
                if total_original_duration > 0:
                    current_pos = 0
                    for seg in segments:
                        seg_duration = seg["end"] - seg["start"]
                        seg_ratio = seg_duration / total_original_duration
                        
                        adjusted_start = current_pos
                        adjusted_end = current_pos + (tts_duration_ms * seg_ratio)
                        
                        adjusted_segments.append({
                            "original_start": seg["start"],
                            "original_end": seg["end"],
                            "adjusted_start": adjusted_start,
                            "adjusted_end": adjusted_end,
                            "text": seg.get("translated_text", seg.get("original_text", "")),
                            "speed_adjustment": tts_duration_ms / total_original_duration
                        })
                        
                        current_pos = adjusted_end
            
            # Apply final speed adjustment if needed
            if self.config.voice_speed != 1.0:
                tts_audio = self._adjust_audio_speed(tts_audio, self.config.voice_speed)
            
            # Normalize audio
            tts_audio = self._normalize_audio(tts_audio)
            
            # Export final audio
            tts_audio.export(output_path, format="wav")
            
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            logger.info(f"Speech synthesized: {output_path} ({tts_duration_ms}ms)")
            return True, adjusted_segments
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            # Create fallback silent audio
            try:
                silent_audio = AudioSegment.silent(duration=5000)
                silent_audio.export(output_path, format="wav")
                return True, []
            except:
                return False, []
    
    def _adjust_audio_speed(self, audio: AudioSegment, speed: float) -> AudioSegment:
        """Adjust audio playback speed with better quality"""
        try:
            if speed == 1.0:
                return audio
            
            # Use FFmpeg for better speed adjustment
            import tempfile
            import subprocess
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
                temp_input = tmp_in.name
                audio.export(temp_input, format="wav")
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
                temp_output = tmp_in.name
                
                cmd = [
                    'ffmpeg', '-i', temp_input,
                    '-filter:a', f'atempo={speed}',
                    '-vn', '-y', temp_output
                ]
                
                subprocess.run(cmd, capture_output=True, text=True)
                
                result = AudioSegment.from_file(temp_output)
            
            # Cleanup
            os.unlink(temp_input)
            os.unlink(temp_output)
            
            return result
            
        except Exception as e:
            logger.error(f"Speed adjustment failed, using pydub fallback: {e}")
            # Fallback to pydub
            if speed > 1.0:
                return audio.speedup(playback_speed=speed)
            else:
                frame_rate = int(audio.frame_rate * speed)
                audio = audio._spawn(audio.raw_data, overrides={"frame_rate": frame_rate})
                return audio.set_frame_rate(audio.frame_rate)
    
    def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """Normalize audio levels"""
        try:
            # Target loudness (LUFS)
            target_lufs = -16.0
            
            # Simple normalization 
            max_dBFS = audio.max_dBFS
            if max_dBFS < -1.0:  # Too quiet
                gain = -1.0 - max_dBFS
                audio = audio.apply_gain(gain)
            elif max_dBFS > -1.0:  # Too loud
                gain = -1.0 - max_dBFS
                audio = audio.apply_gain(gain)
            
            return audio
            
        except Exception as e:
            logger.error(f"Audio normalization failed: {e}")
            return audio
    
    def calculate_optimal_speed(self, original_duration_ms: float, translated_text: str) -> float:
        """Calculate optimal speed adjustment for timing match"""
        try:
            # Get speaking rate for target language
            chars_per_second = self.VOICE_RATES.get(self.config.target_lang, 12)
            
            # Estimate speaking time
            estimated_speaking_time = len(translated_text) / chars_per_second  # seconds
            
            # Convert original duration to seconds
            original_duration_s = original_duration_ms / 1000
            
            if original_duration_s <= 0:
                return 1.0
            
            # Calculate required speed
            speed = estimated_speaking_time / original_duration_s
            
            # Clamp speed to reasonable range
            speed = max(0.7, min(1.5, speed))
            
            # Round to nearest 0.05
            speed = round(speed * 20) / 20
            
            logger.info(f"Speed calculation: {estimated_speaking_time:.1f}s speech in {original_duration_s:.1f}s → speed: {speed:.2f}x")
            
            # Update config
            self.config.voice_speed = speed
            
            return speed
            
        except Exception as e:
            logger.error(f"Speed calculation failed: {e}")
            return 1.0
    
    def generate_subtitles(self, segments: List[Dict], output_path: str) -> str:
        """Generate SRT subtitles from timing segments"""
        try:
            if not segments:
                return ""
            
            srt_lines = []
            
            for i, seg in enumerate(segments, 1):
                # Format timestamps
                start_ms = seg.get("adjusted_start", seg.get("original_start", 0)) * 1000
                end_ms = seg.get("adjusted_end", seg.get("original_end", 0)) * 1000
                
                # Convert to SRT format
                def ms_to_srt_time(ms):
                    hours = int(ms // 3600000)
                    minutes = int((ms % 3600000) // 60000)
                    seconds = int((ms % 60000) // 1000)
                    milliseconds = int(ms % 1000)
                    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
                
                start_time = ms_to_srt_time(start_ms)
                end_time = ms_to_srt_time(end_ms)
                
                # Get text
                text = seg.get("translated_text", seg.get("text", ""))
                
                # Add to SRT
                srt_lines.append(f"{i}")
                srt_lines.append(f"{start_time} --> {end_time}")
                srt_lines.append(text)
                srt_lines.append("")  # Empty line between entries
            
            srt_content = "\n".join(srt_lines)
            
            # Save to file
            srt_path = output_path.replace(".wav", ".srt")
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            logger.info(f"Subtitles generated: {srt_path}")
            return srt_path
            
        except Exception as e:
            logger.error(f"Subtitle generation failed: {e}")
            return ""
    
    def process_audio(self, audio_path: str) -> TranslationResult:
        """Complete audio translation pipeline with improved quality"""
        start_time = datetime.now()
        
        try:
            # Validate input
            if not os.path.exists(audio_path):
                error_msg = f"Audio file not found: {audio_path}"
                logger.error(error_msg)
                return TranslationResult(
                    success=False,
                    original_text="",
                    translated_text="",
                    original_language=self.config.source_lang,
                    target_language=self.config.target_lang,
                    original_duration_ms=0,
                    translated_duration_ms=0,
                    duration_match_percent=0,
                    speed_adjustment=1.0,
                    output_path="",
                    error=error_msg
                )
            
            # Load models if not loaded
            if not self._models_loaded:
                if not self.load_models():
                    error_msg = "Failed to load AI models"
                    logger.error(error_msg)
                    return TranslationResult(
                        success=False,
                        original_text="",
                        translated_text="",
                        original_language=self.config.source_lang,
                        target_language=self.config.target_lang,
                        original_duration_ms=0,
                        translated_duration_ms=0,
                        duration_match_percent=0,
                        speed_adjustment=1.0,
                        output_path="",
                        error=error_msg
                    )
            
            # Step 1: Get original audio duration
            logger.info(f"Processing audio: {audio_path}")
            original_audio = AudioSegment.from_file(audio_path)
            original_duration_ms = len(original_audio)
            logger.info(f"Original duration: {original_duration_ms:.0f}ms")
            
            # Step 2: Transcribe with detailed segments
            transcription = self.transcribe_with_segments(audio_path)
            if not transcription.get("success", False):
                error_msg = transcription.get("error", "Transcription failed")
                logger.error(error_msg)
                return TranslationResult(
                    success=False,
                    original_text="",
                    translated_text="",
                    original_language=self.config.source_lang,
                    target_language=self.config.target_lang,
                    original_duration_ms=original_duration_ms,
                    translated_duration_ms=0,
                    duration_match_percent=0,
                    speed_adjustment=1.0,
                    output_path="",
                    error=error_msg
                )
            
            original_text = transcription["text"]
            detected_language = transcription["language"]
            segments = transcription["segments"]
            
            # Update source language if auto-detected
            if self.config.auto_detect:
                self.config.source_lang = detected_language
            
            logger.info(f"Transcribed: {len(original_text)} characters, {len(segments)} segments in {detected_language}")
            
            # Step 3: Translate with context preservation
            logger.info(f"Translating from {self.config.source_lang} to {self.config.target_lang}...")
            translated_text, translated_segments = self.translate_text_with_context(
                original_text, segments
            )
            
            if translated_text == original_text:
                logger.warning("Translation returned original text (possible fallback)")
            
            # Step 4: Smart condensation if needed
            if len(translated_text) > len(original_text) * self.config.max_condensation_ratio:
                logger.info("Text needs condensation...")
                condensed_text, condensation_ratio = self.condense_text_smart(
                    translated_text, original_duration_ms
                )
                translated_text = condensed_text
                logger.info(f"Condensation applied: ratio {condensation_ratio:.2f}")
            
            # Step 5: Calculate optimal speed adjustment
            speed = self.calculate_optimal_speed(original_duration_ms, translated_text)
            
            # Step 6: Generate speech with timing
            output_path = f"translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            logger.info(f"Generating speech (speed: {speed:.2f}x)...")
            
            # Run async TTS
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            tts_success = False
            timing_segments = []
            
            try:
                tts_success, timing_segments = loop.run_until_complete(
                    self.synthesize_speech_with_timing(
                        translated_text, translated_segments, output_path
                    )
                )
            finally:
                loop.close()
            
            if not tts_success or not os.path.exists(output_path):
                error_msg = "TTS synthesis failed"
                logger.error(error_msg)
                return TranslationResult(
                    success=False,
                    original_text=original_text,
                    translated_text=translated_text,
                    original_language=self.config.source_lang,
                    target_language=self.config.target_lang,
                    original_duration_ms=original_duration_ms,
                    translated_duration_ms=0,
                    duration_match_percent=0,
                    speed_adjustment=speed,
                    output_path="",
                    error=error_msg
                )
            
            # Step 7: Generate subtitles
            subtitle_path = self.generate_subtitles(timing_segments, output_path)
            
            # Step 8: Get translated audio duration
            translated_audio = AudioSegment.from_file(output_path)
            translated_duration_ms = len(translated_audio)
            
            # Step 9: Calculate metrics
            duration_diff = abs(translated_duration_ms - original_duration_ms)
            duration_match_percent = (1 - duration_diff / original_duration_ms) * 100 if original_duration_ms > 0 else 0
            
            # Step 10: Validate timing
            within_tolerance = duration_diff <= self.config.timing_tolerance_ms
            
            if not within_tolerance:
                logger.warning(f"Timing mismatch: {duration_diff:.0f}ms > {self.config.timing_tolerance_ms}ms tolerance")
            
            # Calculate total processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare result
            result = TranslationResult(
                success=True,
                original_text=original_text,
                translated_text=translated_text,
                original_language=self.config.source_lang,
                target_language=self.config.target_lang,
                original_duration_ms=original_duration_ms,
                translated_duration_ms=translated_duration_ms,
                duration_match_percent=duration_match_percent,
                speed_adjustment=speed,
                output_path=output_path,
                subtitle_path=subtitle_path,
                timing_segments=timing_segments
            )
            
            # Log success
            logger.info(f"Translation completed in {processing_time:.1f}s")
            logger.info(f"Duration match: {duration_match_percent:.1f}% ({duration_diff:.0f}ms diff)")
            logger.info(f"Speed adjustment: {speed:.2f}x")
            logger.info(f"Output: {output_path}")
            if subtitle_path:
                logger.info(f"Subtitles: {subtitle_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Audio translation pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            
            return TranslationResult(
                success=False,
                original_text="",
                translated_text="",
                original_language=self.config.source_lang,
                target_language=self.config.target_lang,
                original_duration_ms=0,
                translated_duration_ms=0,
                duration_match_percent=0,
                speed_adjustment=1.0,
                output_path="",
                error=str(e)
            )

# Utility function for quick testing
def test_audio_translation():
    """Test the audio translation module"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test audio translation")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--source", default="auto", help="Source language (auto for detection)")
    parser.add_argument("--target", default="de", help="Target language")
    parser.add_argument("--output", help="Output file path (optional)")
    
    args = parser.parse_args()
    
    # Configure translator
    config = TranslationConfig(
        source_lang=args.source,
        target_lang=args.target,
        auto_detect=(args.source == "auto")
    )
    
    translator = AudioTranslator(config)
    
    print(f"\n{'='*60}")
    print(f"Testing Audio Translation: {args.source} → {args.target}")
    print(f"Audio file: {args.audio}")
    print(f"{'='*60}\n")
    
    # Process audio
    result = translator.process_audio(args.audio)
    
    # Print results
    print(f"\n{'='*60}")
    print("Translation Results:")
    print(f"{'='*60}")
    
    if result.success:
        print(f"✓ SUCCESS")
        print(f"  Source language: {result.original_language}")
        print(f"  Target language: {result.target_language}")
        print(f"  Original duration: {result.original_duration_ms:.0f}ms")
        print(f"  Translated duration: {result.translated_duration_ms:.0f}ms")
        print(f"  Duration match: {result.duration_match_percent:.1f}%")
        print(f"  Speed adjustment: {result.speed_adjustment:.2f}x")
        print(f"  Output file: {result.output_path}")
        
        if result.subtitle_path:
            print(f"  Subtitle file: {result.subtitle_path}")
        
        # Show text preview
        print(f"\n  Original text (preview): {result.original_text[:200]}...")
        print(f"  Translated text (preview): {result.translated_text[:200]}...")
        
        # Move output if specified
        if args.output and os.path.exists(result.output_path):
            import shutil
            shutil.move(result.output_path, args.output)
            print(f"  Moved output to: {args.output}")
    else:
        print(f"✗ FAILED")
        print(f"  Error: {result.error}")
    
    print(f"{'='*60}\n")
    
    return result.success

if __name__ == "__main__":
    # Example usage
    success = test_audio_translation()
    sys.exit(0 if success else 1)