"""
CassandraBot Validation Script

Runs the bot against resolved Metaculus binary questions to measure
accuracy (Brier score) and compare against the community prediction.

Usage:
  poetry run python validate.py                    # Default: 20 questions
  poetry run python validate.py --num_questions 50 # More questions (costs more)
  poetry run python validate.py --sweep            # Sweep extremization factors

Output:
  - Brier scores for your bot vs community prediction
  - Per-question breakdown
  - Extremization factor sweep results (with --sweep)
  - Results saved to validate_results.json
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone

import dotenv
dotenv.load_dotenv()

from forecasting_tools import (
    MetaculusApi,
    MetaculusClient,
    BinaryQuestion,
)

# Import the bot (same main.py)
from main import CassandraBot, ForesightLlm, extremize

logger = logging.getLogger(__name__)


def brier_score(predicted: float, actual: float) -> float:
    """Brier score: lower is better. Range [0, 2]."""
    return (predicted - actual) ** 2


def log_score(predicted: float, actual: float) -> float:
    """Log score: higher (less negative) is better."""
    predicted = max(0.001, min(0.999, predicted))
    if actual == 1.0:
        return math.log(predicted)
    else:
        return math.log(1.0 - predicted)


async def run_validation(num_questions: int = 20, sweep: bool = False):
    """
    Main validation loop:
    1. Fetch resolved binary benchmark questions from Metaculus
    2. Run CassandraBot on each (without submitting predictions)
    3. Score predictions vs actual outcomes
    4. Compare against community prediction baseline
    """

    logger.info(f"Fetching {num_questions} benchmark questions from Metaculus...")

    try:
        questions = MetaculusApi.get_benchmark_questions(
            num_of_questions_to_return=num_questions,
        )
    except Exception as e:
        logger.error(f"Failed to fetch benchmark questions: {e}")
        logger.info("Falling back to manual question fetching...")
        # Fallback: fetch from a known resolved tournament or question set
        raise

    # Filter to binary questions only (the benchmarker currently returns binary)
    binary_questions = [q for q in questions if isinstance(q, BinaryQuestion)]
    logger.info(f"Got {len(binary_questions)} binary benchmark questions")

    if not binary_questions:
        logger.error("No binary benchmark questions found!")
        sys.exit(1)

    # Initialize the bot in validation mode
    foresight = ForesightLlm(temperature=0.3, max_tokens=4000, timeout=180)
    bot = CassandraBot(
        foresight=foresight,
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,  # IMPORTANT: don't submit during validation
        skip_previously_forecasted_questions=False,  # Always forecast for eval
        extra_metadata_in_explanation=False,
        llms={
            "default": "openrouter/openai/gpt-4o-mini",
            "summarizer": "openrouter/openai/gpt-4o-mini",
            "researcher": "openrouter/openai/gpt-4o-mini",
            "parser": "openrouter/openai/gpt-4o-mini",
        },
    )

    # Run predictions
    results = []
    for i, question in enumerate(binary_questions):
        logger.info(f"\n{'='*60}")
        logger.info(f"Question {i+1}/{len(binary_questions)}: {question.question_text[:80]}...")
        logger.info(f"URL: {question.page_url}")

        # Get the resolution (actual outcome)
        resolution = question.resolution
        if resolution is None:
            logger.warning(f"  Skipping - no resolution available")
            continue

        # resolution is typically 1.0 (Yes) or 0.0 (No)
        actual = float(resolution)
        logger.info(f"  Actual resolution: {actual}")

        # Get community prediction if available
        community_pred = None
        if hasattr(question, 'community_prediction') and question.community_prediction is not None:
            community_pred = question.community_prediction
            if hasattr(community_pred, 'latest'):
                community_pred = community_pred.latest
            if isinstance(community_pred, (int, float)):
                community_pred = float(community_pred)
            else:
                community_pred = None

        try:
            # Run research
            research = await bot.run_research(question)

            # Run the ensemble forecast
            prediction_result = await bot._run_forecast_on_binary(question, research)
            bot_pred = prediction_result.prediction_value

            logger.info(f"  Bot prediction: {bot_pred:.3f}")
            if community_pred is not None:
                logger.info(f"  Community prediction: {community_pred:.3f}")

            result = {
                "question_id": question.id_of_post if hasattr(question, 'id_of_post') else str(i),
                "question_text": question.question_text[:200],
                "url": str(question.page_url) if question.page_url else "",
                "resolution": actual,
                "bot_prediction": bot_pred,
                "community_prediction": community_pred,
                "bot_brier": brier_score(bot_pred, actual),
                "community_brier": brier_score(community_pred, actual) if community_pred is not None else None,
                "bot_log": log_score(bot_pred, actual),
                "community_log": log_score(community_pred, actual) if community_pred is not None else None,
            }
            results.append(result)

            logger.info(f"  Bot Brier: {result['bot_brier']:.4f}")
            if result['community_brier'] is not None:
                logger.info(f"  Community Brier: {result['community_brier']:.4f}")

        except Exception as e:
            logger.error(f"  Failed to forecast: {e}")
            continue

    if not results:
        logger.error("No results! All questions failed.")
        sys.exit(1)

    # ============================================================
    # SUMMARY STATISTICS
    # ============================================================
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 70)

    bot_briers = [r["bot_brier"] for r in results]
    avg_bot_brier = sum(bot_briers) / len(bot_briers)
    print(f"\nQuestions evaluated: {len(results)}")
    print(f"Bot average Brier score: {avg_bot_brier:.4f}")

    community_briers = [r["community_brier"] for r in results if r["community_brier"] is not None]
    if community_briers:
        avg_comm_brier = sum(community_briers) / len(community_briers)
        print(f"Community average Brier score: {avg_comm_brier:.4f}")
        diff = avg_bot_brier - avg_comm_brier
        better_or_worse = "BETTER" if diff < 0 else "WORSE"
        print(f"Difference (bot - community): {diff:+.4f} ({better_or_worse})")
    else:
        print("Community predictions not available for comparison.")

    # Bot log scores
    bot_logs = [r["bot_log"] for r in results]
    avg_bot_log = sum(bot_logs) / len(bot_logs)
    print(f"\nBot average log score: {avg_bot_log:.4f}")

    # Per-question breakdown
    print(f"\n{'='*70}")
    print("PER-QUESTION BREAKDOWN")
    print(f"{'='*70}")
    print(f"{'Question':<50} {'Actual':>6} {'Bot':>6} {'Brier':>7}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["bot_brier"], reverse=True):
        q_text = r["question_text"][:48]
        print(f"{q_text:<50} {r['resolution']:>6.0f} {r['bot_prediction']:>6.2f} {r['bot_brier']:>7.4f}")

    # ============================================================
    # EXTREMIZATION SWEEP (optional)
    # ============================================================
    if sweep:
        print(f"\n{'='*70}")
        print("EXTREMIZATION FACTOR SWEEP")
        print(f"{'='*70}")
        print(f"{'Factor':>8} {'Avg Brier':>12} {'vs Community':>14}")
        print("-" * 36)

        # We need raw (pre-extremized) predictions to sweep properly.
        # The bot already applies extremization, so we need to reverse it.
        # Reverse: given extremized prob and factor, find the original median.
        # Since extremize(p, f) = odds^f / (1 + odds^f), we can reverse it.
        current_factor = bot.EXTREMIZE_FACTOR
        best_factor = current_factor
        best_brier = avg_bot_brier

        for factor in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
            sweep_briers = []
            for r in results:
                # Reverse the current extremization to get the raw median
                ep = r["bot_prediction"]
                if 0.01 < ep < 0.99 and current_factor != 1.0:
                    # Reverse: raw_odds = extremized_odds ^ (1/current_factor)
                    e_odds = ep / (1 - ep)
                    raw_odds = e_odds ** (1.0 / current_factor)
                    raw_prob = raw_odds / (1 + raw_odds)
                else:
                    raw_prob = ep

                # Apply new factor
                new_prob = extremize(raw_prob, factor)
                sweep_briers.append(brier_score(new_prob, r["resolution"]))

            avg_sweep = sum(sweep_briers) / len(sweep_briers)
            vs_comm = ""
            if community_briers:
                diff = avg_sweep - avg_comm_brier
                vs_comm = f"{diff:+.4f}"

            marker = " <-- current" if factor == current_factor else ""
            if avg_sweep < best_brier:
                best_brier = avg_sweep
                best_factor = factor
                marker += " *BEST*"

            print(f"{factor:>8.1f} {avg_sweep:>12.4f} {vs_comm:>14}{marker}")

        print(f"\nOptimal extremization factor: {best_factor:.1f} (Brier: {best_brier:.4f})")
        if best_factor != current_factor:
            print(f"  -> Consider changing EXTREMIZE_FACTOR from {current_factor} to {best_factor}")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": len(results),
        "avg_bot_brier": avg_bot_brier,
        "avg_bot_log": avg_bot_log,
        "avg_community_brier": sum(community_briers) / len(community_briers) if community_briers else None,
        "extremize_factor": bot.EXTREMIZE_FACTOR,
        "num_models": len(bot.ensemble_models),
        "model_names": [m.name for m in bot.ensemble_models],
        "results": results,
    }

    output_path = "validate_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return output


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    for noisy in ["LiteLLM", "openai.agents", "httpx", "httpcore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Validate CassandraBot on resolved questions")
    parser.add_argument(
        "--num_questions", type=int, default=20,
        help="Number of benchmark questions to evaluate (default: 20)"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Sweep extremization factors to find optimal value"
    )
    args = parser.parse_args()

    asyncio.run(run_validation(
        num_questions=args.num_questions,
        sweep=args.sweep,
    ))
