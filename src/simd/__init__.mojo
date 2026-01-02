"""
SIMD-optimized string operations.
"""

from .whitespace import skip_whitespace_simd, count_whitespace_simd, is_whitespace, trim_whitespace
from .special import find_char_simd, find_any_char_simd, count_char_simd, find_special_token_boundary
