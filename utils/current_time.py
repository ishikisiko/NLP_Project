from datetime import datetime

def get_current_time_str() -> str:
    """Returns the current system time as a formatted string (YYYY-MM-DD HH:MM:SS)."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_current_date_str() -> str:
    """Returns the current system date as a formatted string (YYYY-MM-DD)."""
    return datetime.now().strftime("%Y-%m-%d")

def get_current_year() -> int:
    """Returns the current year."""
    return datetime.now().year
