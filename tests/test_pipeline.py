# tests/test_pipeline.py - Comprehensive 24-case benchmark test suite
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.inference.pipeline import AspectAnalyzer

TEST_CASES = [
    # 1. Single positive
    {"id": "TC01", "text": "The battery life is excellent and lasts all day.", "expected_aspects": ["battery life"], "expected_sentiments": {"battery life": "positive"}, "category": "Single positive"},
    {"id": "TC02", "text": "The screen is bright, crisp, and gorgeous.", "expected_aspects": ["screen"], "expected_sentiments": {"screen": "positive"}, "category": "Single positive"},
    {"id": "TC03", "text": "The keyboard is very comfortable to type on.", "expected_aspects": ["keyboard"], "expected_sentiments": {"keyboard": "positive"}, "category": "Single positive"},
    {"id": "TC04", "text": "Super fast performance and boots in seconds.", "expected_aspects": ["performance"], "expected_sentiments": {"performance": "positive"}, "category": "Single positive"},

    # 2. Single negative
    {"id": "TC05", "text": "The trackpad is unresponsive and frustrating.", "expected_aspects": ["trackpad"], "expected_sentiments": {"trackpad": "negative"}, "category": "Single negative"},
    {"id": "TC06", "text": "The battery dies in less than two hours.", "expected_aspects": ["battery"], "expected_sentiments": {"battery": "negative"}, "category": "Single negative"},
    {"id": "TC07", "text": "The price is too high for what you get.", "expected_aspects": ["price"], "expected_sentiments": {"price": "negative"}, "category": "Single negative"},
    {"id": "TC08", "text": "The speakers sound tinny and terrible.", "expected_aspects": ["speakers"], "expected_sentiments": {"speakers": "negative"}, "category": "Single negative"},

    # 3. Single neutral / factual
    {"id": "TC09", "text": "It comes with 8GB RAM and Windows 10.", "expected_aspects": ["8GB RAM", "Windows 10"], "expected_sentiments": {"8GB RAM": "neutral", "Windows 10": "neutral"}, "category": "Neutral / factual"},
    {"id": "TC10", "text": "The laptop has two USB ports and an HDMI output.", "expected_aspects": ["USB ports", "HDMI output"], "expected_sentiments": {"USB ports": "neutral", "HDMI output": "neutral"}, "category": "Neutral / factual"},
    {"id": "TC11", "text": "The power adapter is included in the box.", "expected_aspects": ["power adapter"], "expected_sentiments": {"power adapter": "neutral"}, "category": "Neutral / factual"},

    # 4. Mixed sentiments in one sentence (Key challenge from report)
    {"id": "TC12", "text": "The screen is gorgeous but the battery dies in two hours.", "expected_aspects": ["screen", "battery"], "expected_sentiments": {"screen": "positive", "battery": "negative"}, "category": "Mixed sentiment"},
    {"id": "TC13", "text": "The battery life is great but the screen scratches easily.", "expected_aspects": ["battery life", "screen"], "expected_sentiments": {"battery life": "positive", "screen": "negative"}, "category": "Mixed sentiment"},
    {"id": "TC14", "text": "Windows 8 is fast, but the keyboard feels cheap.", "expected_aspects": ["Windows 8", "keyboard"], "expected_sentiments": {"Windows 8": "positive", "keyboard": "negative"}, "category": "Mixed sentiment"},
    {"id": "TC15", "text": "Great performance, but customer service was awful.", "expected_aspects": ["performance", "customer service"], "expected_sentiments": {"performance": "positive", "customer service": "negative"}, "category": "Mixed sentiment"},
    {"id": "TC16", "text": "The touchpad is difficult to use, though the display is stunning.", "expected_aspects": ["touchpad", "display"], "expected_sentiments": {"touchpad": "negative", "display": "positive"}, "category": "Mixed sentiment"},

    # 5. Multi-aspect same sentiment
    {"id": "TC17", "text": "Great battery life and amazing screen quality.", "expected_aspects": ["battery life", "screen quality"], "expected_sentiments": {"battery life": "positive", "screen quality": "positive"}, "category": "Multi-aspect same sentiment"},
    {"id": "TC18", "text": "Poor build quality, weak wifi signal, and sluggish processor.", "expected_aspects": ["build quality", "wifi signal", "processor"], "expected_sentiments": {"build quality": "negative", "wifi signal": "negative", "processor": "negative"}, "category": "Multi-aspect same sentiment"},

    # 6. Negation sentences
    {"id": "TC19", "text": "The battery does not last long at all.", "expected_aspects": ["battery"], "expected_sentiments": {"battery": "negative"}, "category": "Negation"},
    {"id": "TC20", "text": "The keyboard is not bad for typing.", "expected_aspects": ["keyboard"], "expected_sentiments": {"keyboard": "positive"}, "category": "Negation"},
    {"id": "TC21", "text": "I never had any issue with the operating system.", "expected_aspects": ["operating system"], "expected_sentiments": {"operating system": "positive"}, "category": "Negation"},

    # 7. Subtle / Idiomatic expressions
    {"id": "TC22", "text": "The price burned a hole in my pocket.", "expected_aspects": ["price"], "expected_sentiments": {"price": "negative"}, "category": "Idiomatic / subtle"},
    {"id": "TC23", "text": "The aluminum casing is a feast for the eyes.", "expected_aspects": ["aluminum casing"], "expected_sentiments": {"aluminum casing": "positive"}, "category": "Idiomatic / subtle"},
    {"id": "TC24", "text": "Customer support left much to be desired.", "expected_aspects": ["customer support"], "expected_sentiments": {"customer support": "negative"}, "category": "Idiomatic / subtle"}
]


def run_benchmark(model_mode='auto'):
    analyzer = AspectAnalyzer(model_mode=model_mode)
    print(f"\nRunning Benchmark with: {analyzer.model_name}")
    rows = []
    for tc in TEST_CASES:
        preds = analyzer.analyze(tc["text"])
        pred_aspects = [p["aspect"] for p in preds]
        pred_sentiments = {p["aspect"]: p["sentiment"] for p in preds}

        matched_aspects = []
        for ea in tc["expected_aspects"]:
            matched = any(ea.lower() in pa.lower() or pa.lower() in ea.lower() for pa in pred_aspects)
            matched_aspects.append(matched)
        ate_ok = all(matched_aspects) and len(pred_aspects) > 0

        asc_results = {}
        for ea, es in tc["expected_sentiments"].items():
            actual_pred_s = None
            for pa, ps in pred_sentiments.items():
                if ea.lower() in pa.lower() or pa.lower() in ea.lower():
                    actual_pred_s = ps
                    break
            asc_results[ea] = (es, actual_pred_s, es == actual_pred_s)

        asc_ok = all(m[2] for m in asc_results.values()) if asc_results else False
        rows.append({
            "id": tc["id"],
            "category": tc["category"],
            "text": tc["text"],
            "ate_ok": ate_ok,
            "asc_ok": asc_ok,
            "overall_ok": ate_ok and asc_ok
        })

    df = pd.DataFrame(rows)
    print(f"ATE Pass Rate: {df['ate_ok'].mean():.1%}")
    print(f"ASC Pass Rate: {df['asc_ok'].mean():.1%}")
    print(f"Overall Pass Rate: {df['overall_ok'].mean():.1%}")
    return df


if __name__ == '__main__':
    run_benchmark()
