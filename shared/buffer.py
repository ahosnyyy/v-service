#!/usr/bin/env python3
"""
Shared Frame Buffer

Thread-safe frame queue for communication between recorder and detector services.
"""

import threading
import time
from collections import deque
from typing import Optional, Any, Union
from queue import Queue, Empty


class FrameBuffer:
    """Thread-safe frame buffer with configurable size limit."""
    
    def __init__(self, maxlen: int = 100, timeout: float = 1.0):
        """
        Initialize the frame buffer.
        
        Args:
            maxlen: Maximum number of frames to store
            timeout: Timeout for blocking operations (seconds)
        """
        self.maxlen = maxlen
        self.timeout = timeout
        self._buffer = deque(maxlen=maxlen)
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        
        # Statistics
        self._total_pushed = 0
        self._total_popped = 0
        self._dropped_frames = 0
    
    def put(self, frame: Any, frame_name: str = None, block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Add a frame to the buffer.
        
        Args:
            frame: Frame data to add
            frame_name: Optional name for the frame (for JSON naming)
            block: Whether to block if buffer is full
            timeout: Timeout for blocking operation
            
        Returns:
            True if frame was added, False otherwise
        """
        if timeout is None:
            timeout = self.timeout
            
        with self._lock:
            if block:
                # Wait for space to become available
                end_time = time.time() + timeout
                while len(self._buffer) >= self.maxlen:
                    remaining_time = end_time - time.time()
                    if remaining_time <= 0:
                        return False
                    self._not_full.wait(remaining_time)
            
            # Add frame with name as tuple
            frame_data = (frame, frame_name) if frame_name else frame
            if len(self._buffer) < self.maxlen:
                self._buffer.append(frame_data)
                self._total_pushed += 1
                self._not_empty.notify()
                return True
            else:
                # Buffer is full, drop frame
                self._dropped_frames += 1
                return False
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Get a frame from the buffer.
        
        Args:
            block: Whether to block if buffer is empty
            timeout: Timeout for blocking operation
            
        Returns:
            Frame data or None if timeout/empty
        """
        if timeout is None:
            timeout = self.timeout
            
        with self._lock:
            if block:
                # Wait for frame to become available
                end_time = time.time() + timeout
                while len(self._buffer) == 0:
                    remaining_time = end_time - time.time()
                    if remaining_time <= 0:
                        return None
                    self._not_empty.wait(remaining_time)
            
            # Get frame
            if len(self._buffer) > 0:
                frame = self._buffer.popleft()
                self._total_popped += 1
                self._not_full.notify()
                return frame
            else:
                return None
    
    def peek(self) -> Optional[Any]:
        """Peek at the next frame without removing it."""
        with self._lock:
            if len(self._buffer) > 0:
                return self._buffer[0]
            return None
    
    def get_latest(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Get the most recent frame from the buffer (LIFO - Last In, First Out).
        This ensures we always process the freshest frame.
        
        Args:
            block: Whether to block if buffer is empty
            timeout: Timeout for blocking operation
            
        Returns:
            Most recent frame data or None if timeout/empty
        """
        if timeout is None:
            timeout = self.timeout
            
        with self._lock:
            if block:
                # Wait for frame to become available
                end_time = time.time() + timeout
                while len(self._buffer) == 0:
                    remaining_time = end_time - time.time()
                    if remaining_time <= 0:
                        return None
                    self._not_empty.wait(remaining_time)
            
            # Get most recent frame (from the right end of deque)
            if len(self._buffer) > 0:
                frame = self._buffer.pop()  # Get last (newest) frame
                self._total_popped += 1
                self._not_full.notify()
                return frame
            else:
                return None
    
    def clear(self) -> None:
        """Clear all frames from the buffer."""
        with self._lock:
            self._buffer.clear()
            self._not_full.notify_all()
    
    def empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return len(self._buffer) == 0
    
    def full(self) -> bool:
        """Check if buffer is full."""
        with self._lock:
            return len(self._buffer) >= self.maxlen
    
    def __len__(self) -> int:
        """Get current number of frames in buffer."""
        with self._lock:
            return len(self._buffer)
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        with self._lock:
            return {
                "current_size": len(self._buffer),
                "max_size": self.maxlen,
                "total_pushed": self._total_pushed,
                "total_popped": self._total_popped,
                "dropped_frames": self._dropped_frames,
                "utilization": len(self._buffer) / self.maxlen if self.maxlen > 0 else 0
            }
    
    def resize(self, new_maxlen: int) -> None:
        """
        Resize the buffer.
        
        Args:
            new_maxlen: New maximum size
        """
        if new_maxlen <= 0:
            raise ValueError("Buffer size must be positive")
            
        with self._lock:
            old_maxlen = self.maxlen
            self.maxlen = new_maxlen
            
            # If new size is smaller, remove excess frames
            while len(self._buffer) > new_maxlen:
                self._buffer.popleft()
                self._dropped_frames += 1
            
            # Notify waiting threads
            if new_maxlen > old_maxlen:
                self._not_full.notify_all()


class AsyncFrameBuffer(FrameBuffer):
    """Asynchronous frame buffer using Queue for better performance."""
    
    def __init__(self, maxlen: int = 100, timeout: float = 1.0):
        """
        Initialize the async frame buffer.
        
        Args:
            maxlen: Maximum number of frames to store
            timeout: Timeout for blocking operations (seconds)
        """
        super().__init__(maxlen, timeout)
        self._queue = Queue(maxsize=maxlen)
        self._stats_lock = threading.Lock()
    
    def put(self, frame: Any, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Add a frame to the buffer."""
        try:
            if timeout is None:
                timeout = self.timeout
                
            if block:
                self._queue.put(frame, timeout=timeout)
            else:
                self._queue.put_nowait(frame)
                
            with self._stats_lock:
                self._total_pushed += 1
            return True
            
        except (Queue.Full, Empty):
            with self._stats_lock:
                self._dropped_frames += 1
            return False
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Any]:
        """Get a frame from the buffer."""
        try:
            if timeout is None:
                timeout = self.timeout
                
            if block:
                frame = self._queue.get(timeout=timeout)
            else:
                frame = self._queue.get_nowait()
                
            with self._stats_lock:
                self._total_popped += 1
            return frame
            
        except Empty:
            return None
    
    def empty(self) -> bool:
        """Check if buffer is empty."""
        return self._queue.empty()
    
    def full(self) -> bool:
        """Check if buffer is full."""
        return self._queue.full()
    
    def __len__(self) -> int:
        """Get current number of frames in buffer."""
        return self._queue.qsize()
    
    def clear(self) -> None:
        """Clear all frames from the buffer."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break
