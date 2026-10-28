# scripts/run_full_pipeline.py - Master Pipeline Orchestrator
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.features.build_features import build_all_features
from src.models.train_ate import train_ate_model
from src.models.train_asc import train_asc_model
from src.evaluation.error_analysis import run_error_analysis
from src.evaluation.insights import generate_business_insights
from tests.test_pipeline import run_benchmark

if __name__ == '__main__':
    print("=======================================================")
    print("🚀 RUNNING END-TO-END ABSA SYSTEM PIPELINE")
    print("=======================================================\n")

    print("[1/4] Building Features & Neural Encodings...")
    build_all_features()

    print("\n[2/4] Training Position-Aware BiLSTM (ASC)...")
    train_asc_model(model_type='position_aware')

    print("\n[3/4] Running Error Analysis & Generating Charts...")
    run_error_analysis()
    generate_business_insights()

    print("\n[4/4] Executing Benchmark Verification Suite...")
    run_benchmark(model_mode='position_aware')

    print("\n=======================================================")
    print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=======================================================")
