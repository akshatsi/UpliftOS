"""Structured API error, used everywhere instead of raw exceptions or
FastAPI's default `{"detail": ...}` shape, so every failure response
matches `{"error": ..., "code": ...}` with no stack trace exposed.
"""

from __future__ import annotations


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)
