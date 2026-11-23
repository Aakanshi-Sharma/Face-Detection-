"""Point the app at a throwaway data folder before it is imported."""
import os
import tempfile

os.environ["FACEID_DATA_DIR"] = tempfile.mkdtemp(prefix="faceid-test-")
os.environ.pop("FACEID_DEV", None)
