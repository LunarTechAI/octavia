# Critical Errors

## 0. Demo Account Limitation
-   **Issue**: In `DEMO_MODE`, the system skips the actual AI pipeline and just copies the input file, resulting in no translation.
-   **Impact**: Users testing the demo don't see the actual product capabilities.
-   **Fix**: Modify `process_video_job` in `translation_routes.py` to allow real processing even for demo users (already partially applied manually).

## 1. Dependency Management Fragility
-   **Issue**: Extreme instability between `numpy` (Conda vs Pip) and `torch` versions.
-   **Impact**: Backend crashes with DLL load errors (`_multiarray_umath`). `transformers` library fails to import due to `scikit-learn` / `numpy` binary mismatch.
-   **Fix**: Standardize on a strict `requirements.txt` using Pip-only or Conda-only packages. Do not mix.

## 2. Server Process Management
-   **Issue**: Multiple `uvicorn` instances were fighting for port 8000.
-   **Impact**: Uploads hanging at 10%, API timeout errors.
-   **Fix**: Implement a PID check/lock file or a startup script that kills stale processes before starting.

## 3. Log File Growth
-   **Issue**: `backend_debug.log` grew to 50MB quickly.
-   **Impact**: Disk space usage, git large file warnings, slow grep/read operations.
-   **Fix**: Implement log rotation (e.g., `RotatingFileHandler`) in `app.py`.

## 4. Hardcoded Duration Logic (Fixed)
-   **Issue**: Frontend calculated duration based on file size * 2.
-   **Status**: Fixed in `page.tsx` using `video.duration`.

## 5. CPU-Only Limitations
-   **Issue**: Current setup uses `torch 2.9.1+cpu`.
-   **Impact**: Slow translation processing, unable to use modern efficient models.
-   **Fix**: Reinstall Torch with CUDA support (pending).
