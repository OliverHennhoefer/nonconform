"""
Logging Configuration Examples for nonconform

This example demonstrates how to configure logging for different scenarios
when using nonconform. The library uses Python's standard logging framework
to control progress bars and informational output.

Key Points:
- Progress bars are shown at logging level INFO and below
- Warnings (like EVT fallbacks) are shown at WARNING and below  
- All output can be controlled through standard Python logging
"""

import logging
import sys
from scipy.stats import false_discovery_control

from nonconform.estimation import StandardConformalDetector, ExtremeConformalDetector
from nonconform.strategy import Bootstrap, CrossValidation
from nonconform.utils.data import load_wbc
from nonconform.utils.stat import false_discovery_rate, statistical_power
from pyod.models.iforest import IForest


def configure_logging(level=logging.INFO, format_string=None):
    """Configure logging for nonconform with custom settings."""
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_string,
        stream=sys.stdout,
        force=True  # Override any existing configuration
    )


def example_production_setup():
    """Example: Production setup with minimal output."""
    print("=== Production Setup (WARNING level - no progress bars) ===")
    configure_logging(level=logging.WARNING)
    
    x_train, x_test, y_test = load_wbc(setup=True)
    
    # Create detector
    detector = StandardConformalDetector(
        detector=IForest(behaviour="new"),
        strategy=Bootstrap(n_calib=500, resampling_ratio=0.95)
    )
    
    print("Fitting detector (no progress bars should appear)...")
    detector.fit(x_train)
    
    estimates = detector.predict(x_test)
    decisions = false_discovery_control(estimates, method="bh") <= 0.2
    
    print(f"Results: FDR={false_discovery_rate(y=y_test, y_hat=decisions):.3f}, "
          f"Power={statistical_power(y=y_test, y_hat=decisions):.3f}")


def example_development_setup():
    """Example: Development setup with progress bars."""
    print("\n=== Development Setup (INFO level - shows progress bars) ===")
    configure_logging(level=logging.INFO)
    
    x_train, x_test, y_test = load_wbc(setup=True)
    
    # Create detector with cross-validation strategy
    detector = StandardConformalDetector(
        detector=IForest(behaviour="new", n_estimators=50),
        strategy=CrossValidation(k=3)
    )
    
    print("Fitting detector with progress bars...")
    detector.fit(x_train)
    
    estimates = detector.predict(x_test)
    print(f"Generated {len(estimates)} predictions")


def example_debug_setup():
    """Example: Debug setup with maximum verbosity."""
    print("\n=== Debug Setup (DEBUG level - shows everything) ===")
    configure_logging(
        level=logging.DEBUG,
        format_string='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    x_train, x_test, y_test = load_wbc(setup=True)
    
    # Use ExtremeConformalDetector which has additional logging
    detector = ExtremeConformalDetector(
        detector=IForest(behaviour="new"),
        strategy=Bootstrap(n_calib=200, resampling_ratio=0.8),
        evt_threshold_method="percentile",
        evt_threshold_value=0.95
    )
    
    print("Fitting extreme detector with debug output...")
    detector.fit(x_train)
    
    estimates = detector.predict(x_test)
    print(f"Generated {len(estimates)} predictions")


def example_selective_logging():
    """Example: Selective logging - show progress but hide specific modules."""
    print("\n=== Selective Logging (custom logger configuration) ===")
    
    # Configure root nonconform logger
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
    
    # Show progress bars but hide bootstrap configuration details
    bootstrap_logger = logging.getLogger('nonconform.strategy.bootstrap')
    bootstrap_logger.setLevel(logging.WARNING)
    
    x_train, x_test, y_test = load_wbc(setup=True)
    
    detector = StandardConformalDetector(
        detector=IForest(behaviour="new"),
        strategy=Bootstrap(n_calib=1000, resampling_ratio=0.9)
    )
    
    print("Fitting with selective logging (progress bars but no bootstrap details)...")
    detector.fit(x_train)
    
    estimates = detector.predict(x_test)
    print(f"Generated {len(estimates)} predictions")


def example_custom_formatter():
    """Example: Custom logging format for better readability."""
    print("\n=== Custom Formatter Example ===")
    
    # Create custom formatter
    class CustomFormatter(logging.Formatter):
        """Custom formatter with colors for different levels."""
        
        def format(self, record):
            if record.levelno >= logging.ERROR:
                prefix = "❌ ERROR"
            elif record.levelno >= logging.WARNING:
                prefix = "⚠️  WARNING"
            elif record.levelno >= logging.INFO:
                prefix = "ℹ️  INFO"
            else:
                prefix = "🔍 DEBUG"
            
            return f"{prefix}: {record.getMessage()}"
    
    # Configure with custom formatter
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers and add custom one
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)
    
    x_train, x_test, y_test = load_wbc(setup=True)
    
    # Use extreme detector to trigger warnings
    detector = ExtremeConformalDetector(
        detector=IForest(behaviour="new"),
        strategy=Bootstrap(n_calib=50, resampling_ratio=0.8)  # Small dataset to trigger EVT warnings
    )
    
    print("Fitting with custom formatter...")
    detector.fit(x_train)
    
    estimates = detector.predict(x_test)
    print(f"Generated {len(estimates)} predictions")


if __name__ == "__main__":
    print("Demonstrating logging configuration options for nonconform\n")
    
    # Run examples in sequence
    example_production_setup()
    example_development_setup() 
    example_debug_setup()
    example_selective_logging()
    example_custom_formatter()
    
    print("\n" + "="*60)
    print("Summary:")
    print("- Use WARNING level in production to hide progress bars")
    print("- Use INFO level in development to see progress")  
    print("- Use DEBUG level for troubleshooting")
    print("- Configure specific loggers for fine-grained control")
    print("- All nonconform loggers use the 'nonconform.*' hierarchy")
    print("="*60)