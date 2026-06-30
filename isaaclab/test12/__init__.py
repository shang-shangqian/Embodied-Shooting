"""Target direction estimation with OSTrack single-object tracking."""

from .ostrack_tracker import OSTrackBoxTracker, TrackedBox

__all__ = [
    "OSTrackBoxTracker",
    "TrackedBox",
]
