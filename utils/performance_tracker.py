# core/performance_tracker.py
"""
Performance tracking utility for system components
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

class PerformanceTracker:
    """
    Track performance metrics for system components
    """
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.logger = logging.getLogger(f'performance_{component_name}')
        
        # Metrics storage
        self.metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_duration': 0.0,
            'operations_today': 0,
            'errors_today': [],
            'last_operation_time': None
        }
        
        # Timing storage
        self.operation_times = []
        
    def record_success(self, duration: float, metadata: Optional[Dict] = None):
        """
        Record a successful operation
        
        Args:
            duration: Operation duration in seconds
            metadata: Optional metadata about the operation
        """
        
        self.metrics['total_operations'] += 1
        self.metrics['successful_operations'] += 1
        self.metrics['operations_today'] += 1
        self.metrics['total_duration'] += duration
        self.metrics['last_operation_time'] = datetime.now()
        
        self.operation_times.append({
            'timestamp': datetime.now(),
            'duration': duration,
            'success': True,
            'metadata': metadata
        })
        
        # Keep only last 1000 operations
        if len(self.operation_times) > 1000:
            self.operation_times = self.operation_times[-1000:]
        
        self.logger.debug(f"Operation successful: {duration:.2f}s")
    
    def record_failure(self, error: str, duration: Optional[float] = None):
        """
        Record a failed operation
        
        Args:
            error: Error message
            duration: Operation duration if available
        """
        
        self.metrics['total_operations'] += 1
        self.metrics['failed_operations'] += 1
        self.metrics['operations_today'] += 1
        
        if duration:
            self.metrics['total_duration'] += duration
        
        self.metrics['errors_today'].append({
            'timestamp': datetime.now(),
            'error': error
        })
        
        # Keep only last 100 errors
        if len(self.metrics['errors_today']) > 100:
            self.metrics['errors_today'] = self.metrics['errors_today'][-100:]
        
        self.operation_times.append({
            'timestamp': datetime.now(),
            'duration': duration,
            'success': False,
            'error': error
        })
        
        self.logger.error(f"Operation failed: {error}")
    
    def get_daily_summary(self) -> Dict:
        """
        Get daily performance summary
        
        Returns:
            Dictionary with performance statistics
        """
        
        # Calculate statistics
        success_rate = 0.0
        avg_duration = 0.0
        
        if self.metrics['total_operations'] > 0:
            success_rate = (self.metrics['successful_operations'] / 
                          self.metrics['total_operations']) * 100
            avg_duration = self.metrics['total_duration'] / self.metrics['total_operations']
        
        # Get recent operations (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_operations = [
            op for op in self.operation_times 
            if op['timestamp'] > cutoff_time
        ]
        
        recent_successes = sum(1 for op in recent_operations if op['success'])
        recent_failures = len(recent_operations) - recent_successes
        
        return {
            'component': self.component_name,
            'total_operations': self.metrics['total_operations'],
            'successful_operations': self.metrics['successful_operations'],
            'failed_operations': self.metrics['failed_operations'],
            'success_rate': success_rate,
            'avg_processing_time': avg_duration,
            'operations_24h': len(recent_operations),
            'successes_24h': recent_successes,
            'failures_24h': recent_failures,
            'errors_today': len(self.metrics['errors_today']),
            'last_operation': self.metrics['last_operation_time'],
            'total_analyses': self.metrics['total_operations']  # Alias for compatibility
        }
    
    def reset_daily_metrics(self):
        """Reset daily metrics"""
        
        self.metrics['operations_today'] = 0
        self.metrics['errors_today'] = []
        
        # Clean up old operation times (keep last 7 days)
        cutoff_time = datetime.now() - timedelta(days=7)
        self.operation_times = [
            op for op in self.operation_times 
            if op['timestamp'] > cutoff_time
        ]
    
    def get_metrics(self) -> Dict:
        """
        Get all metrics
        
        Returns:
            Dictionary with all performance metrics
        """
        
        return self.metrics.copy()