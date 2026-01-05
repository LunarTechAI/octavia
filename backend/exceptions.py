from fastapi import HTTPException, status

class OctaviaException(HTTPException):
    """Base exception for Octavia application"""

    def __init__(self, status_code: int, detail: str, error_code: str = None):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code or self.__class__.__name__

class ValidationError(OctaviaException):
    """Validation error"""

    def __init__(self, detail: str, field: str = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="VALIDATION_ERROR"
        )
        self.field = field

class AuthenticationError(OctaviaException):
    """Authentication error"""

    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="AUTHENTICATION_ERROR"
        )

class AuthorizationError(OctaviaException):
    """Authorization error"""

    def __init__(self, detail: str = "Access denied"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code="AUTHORIZATION_ERROR"
        )

class NotFoundError(OctaviaException):
    """Resource not found error"""

    def __init__(self, resource: str, resource_id: str = None):
        detail = f"{resource} not found"
        if resource_id:
            detail += f": {resource_id}"
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            error_code="NOT_FOUND_ERROR"
        )
        self.resource = resource
        self.resource_id = resource_id

class InsufficientCreditsError(ValidationError):
    """Insufficient credits error"""

    def __init__(self, required: int, available: int):
        super().__init__(
            detail=f"Insufficient credits. Required: {required}, Available: {available}",
            field="credits"
        )
        self.required = required
        self.available = available

class JobProcessingError(OctaviaException):
    """Job processing error"""

    def __init__(self, job_id: str, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Job {job_id} processing failed: {detail}",
            error_code="JOB_PROCESSING_ERROR"
        )
        self.job_id = job_id

class TranslationError(OctaviaException):
    """Translation processing error"""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {detail}",
            error_code="TRANSLATION_ERROR"
        )

class FileProcessingError(OctaviaException):
    """File processing error"""

    def __init__(self, detail: str, filename: str = None):
        message = f"File processing failed: {detail}"
        if filename:
            message += f" (file: {filename})"
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
            error_code="FILE_PROCESSING_ERROR"
        )
        self.filename = filename