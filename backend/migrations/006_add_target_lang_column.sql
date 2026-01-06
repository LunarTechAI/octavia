-- Migration to add target_lang column to translation_jobs table
-- This resolves the "Could not find the 'target_lang' column" error

-- Add the target_lang column if it doesn't exist
ALTER TABLE public.translation_jobs 
ADD COLUMN IF NOT EXISTS target_lang text;

-- Copy data from target_language to target_lang (if target_language has data)
UPDATE public.translation_jobs 
SET target_lang = target_language 
WHERE target_language IS NOT NULL 
AND target_lang IS NULL;

-- Create index for better performance (optional)
CREATE INDEX IF NOT EXISTS idx_translation_jobs_target_lang 
ON public.translation_jobs USING btree (target_lang);

-- Add comment for documentation
COMMENT ON COLUMN public.translation_jobs.target_lang IS 'Target language code (e.g., en, es, fr) - for code compatibility';

-- This migration ensures compatibility with code expecting target_lang column
-- while preserving existing data in target_language column
