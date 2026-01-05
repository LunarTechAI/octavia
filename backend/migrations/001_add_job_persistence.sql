-- Migration script to add persistent job storage fields to translation_jobs table
-- Run this in Supabase SQL editor or via CLI

-- Add version column for optimistic locking
ALTER TABLE public.translation_jobs 
ADD COLUMN IF NOT EXISTS version integer DEFAULT 0;

-- Add ETA tracking
ALTER TABLE public.translation_jobs 
ADD COLUMN IF NOT EXISTS eta_seconds numeric NULL;

-- Add metrics storage (JSONB for flexibility)
ALTER TABLE public.translation_jobs 
ADD COLUMN IF NOT EXISTS metrics jsonb NULL DEFAULT '{}'::jsonb;

-- Add chunk tracking fields
ALTER TABLE public.translation_jobs 
ADD COLUMN IF NOT EXISTS processed_chunks integer NULL DEFAULT 0;
ALTER TABLE public.translation_jobs 
ADD COLUMN IF NOT EXISTS total_chunks integer NULL DEFAULT 0;
ALTER TABLE public.translation_jobs 
ADD COLUMN IF NOT EXISTS chunk_size integer NULL;
ALTER TABLE public.translation_jobs 
ADD COLUMN IF NOT EXISTS available_chunks jsonb NULL DEFAULT '[]'::jsonb;

-- Add processing metrics
ALTER TABLE public.translation_jobs 
ADD COLUMN IF NOT EXISTS processing_time_seconds numeric NULL;

-- Add additional fields for subtitle-to-audio jobs
ALTER TABLE public.translation_jobs 
ADD COLUMN IF NOT EXISTS source_lang text NULL;
ALTER TABLE public.translation_jobs 
ADD COLUMN IF NOT EXISTS voice text NULL;
ALTER TABLE public.translation_jobs 
ADD COLUMN IF NOT EXISTS output_format text NULL;
ALTER TABLE public.translation_jobs 
ADD COLUMN IF NOT EXISTS segment_count integer NULL;

-- Add index for faster queries on status
CREATE INDEX IF NOT EXISTS idx_translation_jobs_status ON public.translation_jobs(status) TABLESPACE pg_default;

-- Add index for faster queries on user_id and created_at
CREATE INDEX IF NOT EXISTS idx_translation_jobs_user_created ON public.translation_jobs(user_id, created_at desc) TABLESPACE pg_default;
