"""
# ==============================================================================
# Colab AI Agent Setup Script (SetupLibraries.py)
# ==============================================================================
# This script is designed to streamline the initialization of a Colab notebook
# intended for testing and development of an AI agent.
#
# It performs two main functions:
# 1. **Installs required Python packages** (e.g., via `pip install`).
# 2. **Imports necessary libraries and modules** into the notebook's environment.
#
# To use this file, execute it in your Colab notebook using the following
# pattern after cloning the repository:
#
#   %run FloodAgent/SetupLibraries.py
#
# This ensures a clean and efficient start for your agent's testing session.
# ==============================================================================
"""
#%% Install libraries
!pip install pgmpy rasterio
#%%Load libraries
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from rasterio.warp import transform
import os
import sys
# Removed: import subprocess (no longer needed)

# List of core external packages required for the project
EXTERNAL_PACKAGES = [
    'pandas',
    'numpy',
    'scipy',
    'google-genai',
]

# --- 1. DEFENSIVE IMPORT AND INSTALLATION BLOCK ---
try:
    # Attempt to import all packages
    import pandas as pd
    import numpy as np
    from scipy.spatial import cKDTree
    import google.generativeai as genai
    import re

    print("Success: All core packages were already installed and imported.")

except ImportError as e:
    # An external package is missing. Install all required packages.
    missing_package_name = str(e).split()[-1]
    print(f"Package '{missing_package_name}' not found. Attempting installation of all required packages...")

    # Use 'sys.executable -m pip' to ensure we use the correct pip associated
    # with the running Python kernel. This is the most reliable method.
    install_command = f"{sys.executable} -m pip install -q {' '.join(EXTERNAL_PACKAGES)}"

    # Execute installation command using os.system
    if os.system(install_command) == 0:
        print(f"Successfully installed: {', '.join(EXTERNAL_PACKAGES)}")

        # --- Retry Imports After Installation ---
        try:
            import pandas as pd
            import numpy as np
            from scipy.spatial import cKDTree
            import google.generativeai as genai
            import re
            print("Success: Packages installed and imported after retry.")

        except Exception as e_retry:
            print(f"Fatal Error: Failed to import packages even after installation. Check logs: {e_retry}")
            sys.exit(1)
    else:
        print("Installation failed. Please check your environment or network connection.")
        sys.exit(1)

except Exception as e:
    # Catch any other unexpected error (e.g., file system, permissions)
    print(f"An unexpected error occurred during setup: {e}")
    sys.exit(1)


# --- 2. FINAL VERIFICATION ---
# Verify that a core package like numpy is available
print(f"Numpy version: {np.__version__}")
print("Setup script finished execution")
