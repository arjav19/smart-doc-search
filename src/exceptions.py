class SmartDocSearchError(Exception):
    """Base class for every error raised by this application."""

class UnsupportedFileError(SmartDocSearchError):
    """Raised for anything that is not .pdf."""

class NoTextLayerError(SmartDocSearchError):
    """Raised ehen .pdf yeilds to extraactable text (scanned image,OCR out of scope).""" 

class ConfigurationError(SmartDocSearchError):
    """Raised when required settings (API keys, provder names) are missing or invalid."""

class LLMError(SmartDocSearchError):
    """Raised when the LLM provider call fails (network, quota, auth)."""
