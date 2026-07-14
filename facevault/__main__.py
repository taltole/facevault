"""
Entry point: python -m facevault [--demo]
"""
import argparse
from facevault.pipeline import run

parser = argparse.ArgumentParser(description="FaceVault Retail Intelligence")
parser.add_argument("--demo", action="store_true",
                    help="Run in demo mode using webcam instead of OAK-D")
args = parser.parse_args()
run(demo_mode=args.demo)
