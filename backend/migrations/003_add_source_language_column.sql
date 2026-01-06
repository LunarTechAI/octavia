-- Migration to add source_language column to translation_jobs table
-- This resolves the "Could not find the 'source_language' column" error

-- Add the source_language column if it doesn't exist
ALTER TABLE public.translation_jobs 
ADD COLUMN IF NOT EXISTS source_language text;

-- Copy data from source_lang to source_language (if source_lang has data)
UPDATE public.translation_jobs 
SET source_language = source_lang 
WHERE source_lang IS NOT NULL 
AND source_language IS NULL;

-- Create index for better performance (optional)
CREATE INDEX IF NOT EXISTS idx_translation_jobs_source_language 
ON public.translation_jobs USING btree (source_language);

-- Add comment for documentation
COMMENT ON COLUMN public.translation_jobs.source_language IS 'Source language code (e.g., en, es, fr)';

-- This migration ensures compatibility with code expecting source_language column
-- while preserving existing data in source_lang column
