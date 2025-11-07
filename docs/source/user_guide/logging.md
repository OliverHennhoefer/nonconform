# Logging and Progress Control

nonconform uses Python's standard logging framework to control progress bars and informational output. This provides flexible, fine-grained control over the library's output that integrates seamlessly with existing logging infrastructure.

## Overview

The library replaces the previous `silent` parameter with logger-based progress control, offering several advantages:

- **Standard Integration**: Uses Python's built-in logging framework
- **Granular Control**: Configure different loggers independently
- **Hierarchical Configuration**: Control output at package, module, or class level
- **Production Ready**: Easy to integrate with application logging

## Logger Hierarchy

nonconform uses a hierarchical logger structure:

```
nonconform                    # Root logger
├── estimation                # Detector logging
│   ├── standard_conformal    # Standard detector
│   └── weighted_conformal    # Weighted detector
├── strategy                  # Strategy logging
│   ├── bootstrap             # Bootstrap strategy
│   ├── cross_val             # Cross-validation strategy
│   ├── jackknife             # Jackknife strategy
│   └── split                 # Split strategy
└── utils                     # Utility logging
    ├── data                  # Data loading
    ├── func                  # Function utilities
    └── stat                  # Statistical functions
```

## Basic Configuration

### Show Progress Bars (Development)

```python
import logging

# Enable progress bars and informational messages
logging.getLogger('nonconform').setLevel(logging.INFO)

from nonconform.detection import ConformalDetector
from nonconform.strategy import CrossValidation
from pyod.models.iforest import IForest

detector = ConformalDetector(
    detector=IForest(),
    strategy=CrossValidation(k=5)
)
detector.fit(X_train)  # Shows "CV fold training (5 folds)" progress bar
predictions = detector.predict(X_test)  # Shows "Aggregating 1 models" progress bar
```

### Hide Progress Bars (Production)

```python
import logging

# Hide progress bars, show only warnings and errors
logging.getLogger('nonconform').setLevel(logging.WARNING)

detector = ConformalDetector(
    detector=IForest(),
    strategy=CrossValidation(k=5)
)
detector.fit(X_train)  # No progress bars
predictions = detector.predict(X_test)  # No progress bars
```

### Debug Mode (Troubleshooting)

```python
import logging

# Maximum verbosity for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from nonconform.detection import ConformalDetector
from nonconform.strategy import Bootstrap

detector = ConformalDetector(
    detector=IForest(),
    strategy=Bootstrap(n_calib=100, resampling_ratio=0.8)
)
detector.fit(X_train)  # Shows detailed debug information and warnings
```

## Logging Levels

nonconform uses standard Python logging levels:

| Level | Purpose | What's Shown |
|-------|---------|--------------|
| `DEBUG` | Detailed debugging | All internal operations, parameter details |
| `INFO` | Progress monitoring | Progress bars, general information |
| `WARNING` | Important notices | Fallback behaviors, configuration warnings |
| `ERROR` | Error conditions | Failures, exceptions |
| `CRITICAL` | Critical failures | System-level errors |

### Level Examples

```python
import logging

# Show everything (very verbose)
logging.getLogger('nonconform').setLevel(logging.DEBUG)

# Show progress and warnings (recommended for development)
logging.getLogger('nonconform').setLevel(logging.INFO)

# Show only warnings and errors (recommended for production)
logging.getLogger('nonconform').setLevel(logging.WARNING)

# Show only critical errors
logging.getLogger('nonconform').setLevel(logging.ERROR)
```

## Advanced Configuration

### Selective Module Logging

Control specific modules independently:

```python
import logging

# Configure different modules at different levels
logging.getLogger('nonconform').setLevel(logging.INFO)  # General info
logging.getLogger('nonconform.strategy.bootstrap').setLevel(logging.WARNING)  # Hide bootstrap details
logging.getLogger('nonconform.detection').setLevel(logging.DEBUG)  # Debug detector issues

# This configuration will:
# - Show progress bars for CV and aggregation
# - Hide detailed bootstrap configuration messages
# - Show debug information for all detectors
```

### Custom Formatters

Create custom log formatting for better readability:

```python
import logging
import sys

class CustomFormatter(logging.Formatter):
    """Custom formatter with emojis and colors."""

    def format(self, record):
        if record.levelno >= logging.ERROR:
            prefix = "❌"
        elif record.levelno >= logging.WARNING:
            prefix = "⚠️ "
        elif record.levelno >= logging.INFO:
            prefix = "ℹ️ "
        else:
            prefix = "🔍"

        return f"{prefix} {record.getMessage()}"

# Apply custom formatter
logger = logging.getLogger('nonconform')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(CustomFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

### Integration with Application Logging

Integrate with existing application logging:

```python
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'anomaly_detection.log',
        },
    },
    'loggers': {
        'nonconform': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'myapp': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        },
    }
}

logging.config.dictConfig(LOGGING_CONFIG)

# Now both your app and nonconform logs go to console and file
```

## Progress Bar Descriptions

nonconform provides descriptive progress bars that indicate the specific operation being performed:

| Operation | Description | When It Appears |
|-----------|-------------|-----------------|
| `"CV fold training (N folds)"` | Cross-validation training | During `CrossValidation.fit_calibrate()` |
| `"Bootstrap training (N folds)"` | Bootstrap resampling | During `Bootstrap.fit_calibrate()` |
| `"Aggregating N models"` | Model ensemble aggregation | During `predict()` with multiple models |

Example output:
```
CV fold training (5 folds): 100%|██████████| 5/5 [00:02<00:00,  2.31it/s]
Aggregating 3 models: 100%|██████████| 3/3 [00:00<00:00, 156.78it/s]
```

## Common Scenarios

### Jupyter Notebook Development

```python
import logging

# Configure for interactive development
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s: %(message)s'  # Simplified format for notebooks
)

# Your nonconform code here
detector = ConformalDetector(...)
detector.fit(X)  # Shows progress bars
```

### Server/Production Environment

```python
import logging
import logging.handlers

# Production logging setup
logger = logging.getLogger('nonconform')
logger.setLevel(logging.WARNING)  # Hide progress bars

# Log to rotating file
handler = logging.handlers.RotatingFileHandler(
    'anomaly_detection.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Your detection code runs silently
detector.fit(X_train)
predictions = detector.predict(X_test)
```

### Docker/Containerized Environment

```python
import logging
import sys

# Container-friendly logging (JSON format)
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage()
        }
        return json.dumps(log_entry)

# Configure JSON logging for containers
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logging.getLogger('nonconform').addHandler(handler)
logging.getLogger('nonconform').setLevel(logging.INFO)
```

### Testing Environment

```python
import logging

# Disable all nonconform logging during tests
logging.getLogger('nonconform').setLevel(logging.CRITICAL)

# Or capture logs for testing
import io
log_stream = io.StringIO()
handler = logging.StreamHandler(log_stream)
logging.getLogger('nonconform').addHandler(handler)
logging.getLogger('nonconform').setLevel(logging.DEBUG)

# Run your test
detector.fit(X)

# Check logged output
log_output = log_stream.getvalue()
assert "Bootstrap Configuration" in log_output
```

## Migration from Silent Parameter

If you're migrating from code that used the `silent` parameter:

**Before:**
```python
# Old API with silent parameter
detector = ConformalDetector(
    detector=IForest(),
    strategy=Split(n_calib=0.2),
    silent=True  # Hide progress bars
)
```

**After:**
```python
import logging

# New API with logging control
logging.getLogger('nonconform').setLevel(logging.WARNING)  # Hide progress bars

detector = ConformalDetector(
    detector=IForest(),
    strategy=Split(n_calib=0.2)
    # No silent parameter needed
)
```

## Performance Considerations

Logging configuration has minimal performance impact:

- **Progress bars**: Small overhead only when displayed (INFO level)
- **Log level checks**: Very fast, optimized by Python
- **File logging**: Asynchronous handlers available for high-throughput applications

For maximum performance in production, use `logging.WARNING` level to disable all progress output.

## See Also

- [Troubleshooting Guide](troubleshooting.md) - Logging configuration for debugging
- [Best Practices](best_practices.md) - Production logging recommendations
- [Python Logging Documentation](https://docs.python.org/3/library/logging.html) - Complete logging reference
