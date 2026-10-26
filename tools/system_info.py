"""System information and environment tool."""
from datetime import datetime
import os
import platform
import shutil


def get_system_status() -> str:
    """Retrieve the current local system operating status, OS version, disk storage, and timestamp.

    Returns:
        Formatted summary of system resources and timestamp.
    """
    total, used, free = shutil.disk_usage("/")
    total_gb = total // (2**30)
    used_gb = used // (2**30)
    free_gb = free // (2**30)
    cpu_count = os.cpu_count() or 1
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S (%Z)")

    return (
        f"System Diagnostics:\n"
        f"- Current Timestamp: {now}\n"
        f"- OS: {platform.system()} {platform.release()} ({platform.machine()})\n"
        f"- Python Version: {platform.python_version()}\n"
        f"- CPU Logical Cores: {cpu_count}\n"
        f"- Root Disk Usage: {used_gb} GB used / {total_gb} GB total ({free_gb} GB free)"
    )
