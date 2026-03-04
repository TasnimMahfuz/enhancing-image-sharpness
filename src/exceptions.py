"""Custom exception classes for image sharpness enhancement system.

This module defines custom exceptions that provide descriptive error messages
with context to help users understand what went wrong and how to fix it.
"""


class ValidationError(ValueError):
    """Raised when input validation fails.
    
    Inherits from ValueError for backward compatibility with existing code.
    
    Used for:
    - Invalid parameter ranges (sigma <= 0, t not in [0,1], omega <= 0)
    - Mismatched image dimensions
    - Invalid image formats or data types
    - Invalid configuration values
    
    Error messages should include:
    - What parameter/input is invalid
    - What the valid range/format is
    - What value was provided
    """
    pass


class FileLoadError(IOError):
    """Raised when file I/O operations fail.
    
    Inherits from IOError for backward compatibility with existing code.
    
    Used for:
    - File not found
    - Permission denied
    - Corrupted image files
    - Unsupported image formats
    - Empty images
    
    Error messages should include:
    - File path that caused the error
    - Specific reason for failure
    - Suggestions for resolution if applicable
    """
    pass


class ProcessingError(RuntimeError):
    """Raised when image processing operations fail.
    
    Inherits from RuntimeError for backward compatibility with existing code.
    
    Used for:
    - Convolution failures
    - Memory allocation errors
    - Processing timeouts
    - General processing failures
    
    Error messages should include:
    - Which processing step failed
    - Context about the operation
    - Input parameters if relevant
    """
    pass


class NumericalError(ArithmeticError):
    """Raised when numerical instability is detected.
    
    Inherits from ArithmeticError for backward compatibility with existing code.
    
    Used for:
    - NaN (Not a Number) values
    - Inf (Infinity) values
    - Division by zero
    - Numerical overflow/underflow
    - Invalid mathematical operations
    
    Error messages should include:
    - Where the numerical issue occurred
    - What values caused the issue
    - Which parameters might need adjustment
    """
    pass
