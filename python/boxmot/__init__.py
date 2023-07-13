__version__ = '10.0.15'

from pathlib import Path

from boxmot.deepocsort.ocsort import OCSort as DeepOCSORT

from boxmot.tracker_zoo import create_tracker, get_tracker_config


__all__ = '__version__',\
          'StrongSORT', 'OCSORT', 'BYTETracker', 'BoTSORT', 'DeepOCSORT'
