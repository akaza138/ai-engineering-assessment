"""
Grid07 AI Engineering Assignment — Main Runner
===============================================
Runs all three phases sequentially and saves execution logs.

Usage:
    python main.py

Make sure to copy .env.example → .env and fill in your API keys first.
"""

import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))


def run_all():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'═'*65}")
    print(f"  GRID07 — AI Engineering Assignment")
    print(f"  Cognitive Routing & RAG")
    print(f"  Run at: {timestamp}")
    print(f"{'═'*65}\n")

    # ── Phase 1 ────────────────────────────────────────
    print("\n▶  Starting Phase 1: Vector-Based Persona Matching")
    from persona_router import run_phase1_demo
    collection = run_phase1_demo()

    # ── Phase 2 ────────────────────────────────────────
    print("\n▶  Starting Phase 2: Autonomous Content Engine")
    try:
        from content_engine import run_phase2_demo
        posts = run_phase2_demo()
    except Exception as e:
        print(f"  [Phase 2 Error] {e}")
        print("  → Check your .env file has a valid LLM_PROVIDER and API key.")
        posts = []

    # ── Phase 3 ────────────────────────────────────────
    print("\n▶  Starting Phase 3: Combat Engine + RAG")
    try:
        from combat_engine import run_phase3_demo
        result1, result2 = run_phase3_demo()
    except Exception as e:
        print(f"  [Phase 3 Error] {e}")
        print("  → Check your .env file has a valid LLM_PROVIDER and API key.")
        result1, result2 = {}, {}

    print(f"\n{'═'*65}")
    print("  All phases complete.")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    run_all()
