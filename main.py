"""
CassandraBot v2 - Multi-Model Ensemble Forecasting Bot

Architecture:
1. RESEARCH: AskNews (latest + historical) with Foresight v3 fallback
2. FORECAST: 6-model ensemble via OpenRouter + Lightning Rod direct API
   - Models run in parallel with diverse prompting strategies
   - Outside View / Inside View / Devil's Advocate perspectives
3. AGGREGATE: Median (binary/MC) or mixture (numeric) + extremization
4. SUBMIT: via forecasting-tools to Metaculus

Models in ensemble:
  - Lightning Rod Foresight v3 (direct API, purpose-built forecaster)
  - OpenAI o3 (strong reasoning)
  - OpenAI GPT-5.4 (frontier general)
  - Anthropic Claude Opus 4.6 (strongest complex reasoning)
  - Anthropic Claude Sonnet 4.6 (fast, strong)
  - OpenAI o4-mini (fast reasoning, proven in tournament)
  - DeepSeek R1 (strong open-source reasoning)
"""

import argparse
import asyncio
import logging
import os
import re
import statistics
from datetime import datetime, timezone
from typing import Literal

import dotenv
import numpy as np

dotenv.load_dotenv()

from openai import OpenAI

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    PredictedOption,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
)

logger = logging.getLogger(__name__)


# ============================================================
# LLM WRAPPERS
# ============================================================

class ForesightLlm:
    """Direct wrapper for Lightning Rod's Foresight API (bypasses litellm)."""

    def __init__(self, temperature=0.3, max_tokens=4000, timeout=180):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = "foresight-v3"
        api_key = os.getenv("LIGHTNINGROD_API_KEY")
        if not api_key:
            raise ValueError("LIGHTNINGROD_API_KEY not found!")
        self.client = OpenAI(
            base_url="https://api.lightningrod.ai/api/public/v1/openai",
            api_key=api_key,
            timeout=timeout,
        )

    def _call_sync(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="LightningRodLabs/foresight-v3",
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Empty response from Foresight v3")
        return content

    async def invoke(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                result = await asyncio.to_thread(self._call_sync, prompt)
                return result
            except Exception as e:
                logger.warning(f"Foresight API attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)


class OpenRouterLlm:
    """Wrapper for any model available on OpenRouter via OpenAI-compatible API."""

    def __init__(self, model: str, temperature=0.3, max_tokens=4000, timeout=180):
        self.model = model
        self.name = model.split("/")[-1]  # e.g. "o3" from "openai/o3"
        self.temperature = temperature
        self.max_tokens = max_tokens
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found!")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=timeout,
        )

    def _call_sync(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"Empty response from {self.model}")
        return content

    async def invoke(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                result = await asyncio.to_thread(self._call_sync, prompt)
                return result
            except Exception as e:
                logger.warning(f"{self.name} attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)


# ============================================================
# PARSING HELPERS
# ============================================================

def parse_binary_probability(text: str) -> float:
    """Extract probability from text like 'Probability: 35%'."""
    specific_match = re.search(
        r"(?:Probability|PROBABILITY)\s*:\s*(\d+(?:\.\d+)?)\s*%",
        text
    )
    if specific_match:
        prob = float(specific_match.group(1)) / 100.0
        return max(0.01, min(0.99, prob))

    decimal_match = re.search(
        r"(?:Probability|PROBABILITY)\s*:\s*(\d*\.\d+)",
        text
    )
    if decimal_match:
        prob = float(decimal_match.group(1))
        if prob > 1:
            prob = prob / 100.0
        return max(0.01, min(0.99, prob))

    matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
    if matches:
        prob = float(matches[-1]) / 100.0
        return max(0.01, min(0.99, prob))

    logger.warning("Could not parse probability, defaulting to 0.5")
    return 0.5


def parse_percentiles(text: str) -> dict[float, float]:
    """Extract percentile values from text in various formats."""
    results = {}

    # Format 1: "Percentile 10: 115" or "Percentile 10: **115**"
    pattern1 = r"(?:P|p)ercentile\s*(\d+)\s*[:%]\s*\**\s*[≈~]?\s*(-?\s*[\d,]+(?:\.\d+)?)"
    for match in re.finditer(pattern1, text):
        pct = float(match.group(1))
        val = float(match.group(2).replace(",", "").replace(" ", ""))
        results[pct] = val

    # Format 2: Markdown table "| 10 % | **115** |"
    if len(results) < 4:
        pattern2 = r"\|\s*(\d+)\s*%?\s*\|\s*\**\s*[≈~]?\s*(-?[\d,]+(?:\.\d+)?)"
        for match in re.finditer(pattern2, text):
            pct = float(match.group(1))
            if 5 <= pct <= 95:
                val = float(match.group(2).replace(",", "").replace(" ", ""))
                results[pct] = val

    # Format 3: "10%: 115"
    if len(results) < 4:
        pattern3 = r"(\d+)\s*%\s*[:\|]\s*\**\s*[≈~]?\s*(-?[\d,]+(?:\.\d+)?)"
        for match in re.finditer(pattern3, text):
            pct = float(match.group(1))
            if 5 <= pct <= 95:
                val = float(match.group(2).replace(",", ""))
                results[pct] = val

    # Format 4: Inside code blocks
    if len(results) < 4:
        code_blocks = re.findall(r"```(.*?)```", text, re.DOTALL)
        for block in code_blocks:
            for match in re.finditer(r"(?:P|p)ercentile\s*(\d+)\s*:\s*(-?[\d,]+(?:\.\d+)?)", block):
                pct = float(match.group(1))
                val = float(match.group(2).replace(",", ""))
                results[pct] = val

    return results


def parse_multiple_choice(text: str, options: list[str]) -> dict[str, float]:
    """Extract option probabilities from text."""
    results = {}

    for option in options:
        escaped_option = re.escape(option)
        patterns = [
            rf"{escaped_option}\s*:\s*(\d+(?:\.\d+)?)\s*%",
            rf"{escaped_option}\s*:\s*(\d+(?:\.\d+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results[option] = float(match.group(1))
                break

    if len(results) < len(options):
        all_numbers = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
        if len(all_numbers) >= len(options):
            last_n = all_numbers[-len(options):]
            results = {opt: float(num) for opt, num in zip(options, last_n)}

    if results:
        total = sum(results.values())
        if total > 0:
            results = {k: max(0.01, min(0.99, v / total)) for k, v in results.items()}
            total = sum(results.values())
            results = {k: v / total for k, v in results.items()}

    if not results or len(results) < len(options):
        equal_prob = 1.0 / len(options)
        results = {opt: equal_prob for opt in options}

    return results


# ============================================================
# EXTREMIZATION
# ============================================================

def extremize(prob: float, factor: float = 1.4) -> float:
    """
    Push probabilities away from 0.5 toward 0 or 1.
    Aggregated forecasts are systematically too moderate.
    factor=1.0 means no change, >1.0 extremizes.
    Typical range: 1.2-1.8. Tune via MiniBench.
    """
    if prob <= 0.01 or prob >= 0.99:
        return prob
    odds = prob / (1 - prob)
    extremized_odds = odds ** factor
    result = extremized_odds / (1 + extremized_odds)
    return max(0.01, min(0.99, result))


# ============================================================
# PROMPT TEMPLATES
# ============================================================
# Three distinct perspectives to maximize ensemble diversity.
# Each perspective asks the model to reason differently before
# arriving at a probability.

BINARY_OUTSIDE_VIEW = """You are an expert superforecaster using the OUTSIDE VIEW approach.

Your interview question is:
{question_text}

Question background:
{background_info}

Resolution criteria (not yet satisfied):
{resolution_criteria}

{fine_print}

Your research assistant says:
{research}

Today is {today}.

{conditional_disclaimer}

Use the OUTSIDE VIEW methodology:
(a) What is the base rate for events like this? Find the reference class.
(b) How much time is left until resolution?
(c) What is the status quo outcome if nothing changes? Weight this heavily.
(d) How often do similar predictions/events actually materialize?
(e) Adjust from the base rate only if you have strong specific evidence.

Good forecasters anchor heavily on base rates and the status quo, adjusting only modestly for specific evidence. The world changes slowly most of the time.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""

BINARY_INSIDE_VIEW = """You are an expert superforecaster using the INSIDE VIEW approach.

Your interview question is:
{question_text}

Question background:
{background_info}

Resolution criteria (not yet satisfied):
{resolution_criteria}

{fine_print}

Your research assistant says:
{research}

Today is {today}.

{conditional_disclaimer}

Use the INSIDE VIEW methodology:
(a) How much time remains until resolution?
(b) What specific causal mechanisms would lead to Yes vs No?
(c) What is the current trajectory and momentum of relevant factors?
(d) What recent developments or evidence shift the probability?
(e) Map out the specific steps needed for each outcome and assess their likelihood.

Reason carefully through the specific causal chain. Consider what concrete events would need to happen and their individual probabilities.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""

BINARY_DEVILS_ADVOCATE = """You are an expert superforecaster playing DEVIL'S ADVOCATE.

Your interview question is:
{question_text}

Question background:
{background_info}

Resolution criteria (not yet satisfied):
{resolution_criteria}

{fine_print}

Your research assistant says:
{research}

Today is {today}.

{conditional_disclaimer}

Use the DEVIL'S ADVOCATE methodology:
(a) How much time remains until resolution?
(b) What is the consensus/obvious answer? State it clearly.
(c) Now argue AGAINST the consensus. What are the strongest reasons it could be wrong?
(d) What unexpected scenarios could flip the outcome?
(e) What are forecasters most likely to overlook or underweight?
(f) After considering contrarian arguments, give your honest revised estimate.

Good forecasters consider tail risks and contrarian scenarios seriously. The crowd is often right, but sometimes systematically wrong.

The last thing you write is your final answer as: "Probability: ZZ%", 0-100
"""


MC_PROMPT = """You are a professional forecaster interviewing for a job.

Your interview question is:
{question_text}

The options are: {options}

Background:
{background_info}

{resolution_criteria}

{fine_print}

Your research assistant says:
{research}

Today is {today}.

{view_instruction}

{conditional_disclaimer}

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A description of a scenario that results in an unexpected outcome.

You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

The last thing you write is your final probabilities for the N options in this order {options} as:
Option_A: Probability_A%
Option_B: Probability_B%
...
Option_N: Probability_N%
"""


NUMERIC_PROMPT = """You are a professional forecaster interviewing for a job.

Your interview question is:
{question_text}

Background:
{background_info}

{resolution_criteria}

{fine_print}

Units for answer: {unit_of_measure}

Your research assistant says:
{research}

Today is {today}.

{lower_bound_message}
{upper_bound_message}

{view_instruction}

{conditional_disclaimer}

Formatting Instructions:
- Please notice the units requested and give your answer in these units.
- Never use scientific notation.
- Always start with a smaller number and then increase from there.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The outcome if nothing changed.
(c) The outcome if the current trend continued.
(d) The expectations of experts and markets.
(e) A brief description of an unexpected scenario that results in a low outcome.
(f) A brief description of an unexpected scenario that results in a high outcome.

You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

The last thing you write is your final answer as:
"
Percentile 10: XX (lowest number value)
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX (highest number value)
"
"""

DATE_PROMPT = """You are a professional forecaster interviewing for a job.

Your interview question is:
{question_text}

Background:
{background_info}

{resolution_criteria}

{fine_print}

Your research assistant says:
{research}

Today is {today}.

{lower_bound_message}
{upper_bound_message}

{conditional_disclaimer}

Formatting Instructions:
- This is a date question. Answer in YYYY-MM-DD format.
- Always start with an earlier date at percentile 10 and increase chronologically.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The outcome if nothing changed.
(c) The outcome if the current trend continued.
(d) A brief description of an unexpected scenario that results in an early outcome.
(e) A brief description of an unexpected scenario that results in a late outcome.

You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals.

The last thing you write is your final answer as:
"
Percentile 10: YYYY-MM-DD (earliest date)
Percentile 20: YYYY-MM-DD
Percentile 40: YYYY-MM-DD
Percentile 60: YYYY-MM-DD
Percentile 80: YYYY-MM-DD
Percentile 90: YYYY-MM-DD (latest date)
"
"""


# View instructions for prompt diversity on non-binary question types
VIEW_INSTRUCTIONS = {
    "outside": "Use the OUTSIDE VIEW: anchor on base rates and historical reference classes. What usually happens in situations like this? Adjust only modestly from the base rate.",
    "inside": "Use the INSIDE VIEW: reason through the specific causal mechanisms and current evidence. What does the trajectory of current events suggest?",
    "advocate": "Play DEVIL'S ADVOCATE: consider what the consensus might get wrong. What tail risks or overlooked factors could shift the outcome?",
}


# ============================================================
# THE BOT
# ============================================================

class CassandraBot(ForecastBot):
    """
    CassandraBot v2 - Multi-model ensemble forecasting bot.

    Runs each question through multiple frontier models with diverse
    prompting strategies, then aggregates via median + extremization.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    # Limit concurrent model calls to avoid rate limits
    _model_call_semaphore = asyncio.Semaphore(4)

    # Tune this via MiniBench. Range: 1.0 (no extremization) to 2.0 (aggressive)
    EXTREMIZE_FACTOR = 1.4

    def __init__(self, *args, ensemble_models: list = None, foresight: ForesightLlm = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.foresight = foresight or ForesightLlm()

        if ensemble_models is not None:
            self.ensemble_models = ensemble_models
        else:
            # Build default ensemble from available API keys
            self.ensemble_models = [self.foresight]

            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_key:
                openrouter_models = [
                    "openai/o3",                         # Strong reasoning
                    "openai/gpt-5.4",                    # Frontier general model
                    "anthropic/claude-opus-4.6",         # Strongest complex reasoning
                    "anthropic/claude-sonnet-4.6",       # Fast, strong, good value
                    "openai/o4-mini",                    # Fast reasoning, proven in tournament
                    "deepseek/deepseek-r1",              # Strong open-source reasoning
                ]
                for model_name in openrouter_models:
                    try:
                        self.ensemble_models.append(
                            OpenRouterLlm(model=model_name, temperature=0.3, max_tokens=4000)
                        )
                    except Exception as e:
                        logger.warning(f"Could not initialize {model_name}: {e}")
            else:
                logger.warning("OPENROUTER_API_KEY not set - running Foresight-only (no ensemble)")

            logger.info(
                f"Ensemble initialized with {len(self.ensemble_models)} models: "
                f"{[m.name for m in self.ensemble_models]}"
            )

    # ======================== RESEARCH ========================

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Multi-source research pipeline:
        1. AskNews via forecasting-tools AskNewsSearcher (works with Metaculus free tier)
        2. Foresight v3 reasoning fallback if AskNews unavailable
        """
        async with self._concurrency_limiter:
            research_parts = []

            research_prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the
                most relevant news, including if the question would resolve Yes or No based
                on current information. You do not produce forecasts yourself.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )

            # --- AskNews via forecasting-tools wrapper (works with Metaculus free tier) ---
            asknews_client_id = os.getenv("ASKNEWS_CLIENT_ID")
            asknews_secret = os.getenv("ASKNEWS_SECRET")
            if asknews_client_id and asknews_secret:
                try:
                    searcher = AskNewsSearcher()
                    research = await searcher.call_preconfigured_version(
                        "asknews/news-summaries", research_prompt
                    )
                    if research and len(research.strip()) > 50:
                        research_parts.append(f"=== ASKNEWS RESEARCH ===\n{research}")
                        logger.info(f"AskNews research: got {len(research)} chars")
                    else:
                        logger.warning("AskNews returned empty/short result")
                except Exception as e:
                    logger.warning(f"AskNews research failed: {e}")
            else:
                logger.info("AskNews credentials not found, skipping AskNews research")

            # --- Fallback: if no AskNews results, use Foresight for basic research ---
            if not research_parts:
                logger.info("No AskNews results, falling back to Foresight v3 for research")
                fallback_prompt = clean_indents(
                    f"""
                    You are a research assistant to a superforecaster.
                    Generate a concise but detailed rundown of the most relevant facts,
                    news, and context for this forecasting question. Include base rates
                    and historical precedents where possible. You do not produce forecasts yourself.

                    Question:
                    {question.question_text}

                    Resolution criteria:
                    {question.resolution_criteria}

                    {question.fine_print}
                    """
                )
                fallback_research = await self.foresight.invoke(fallback_prompt)
                research_parts.append(f"=== RESEARCH CONTEXT ===\n{fallback_research}")

            research = "\n\n".join(research_parts)
            logger.info(f"Research for {question.page_url}: {len(research)} chars total")
            return research

    # ======================== ENSEMBLE HELPERS ========================

    async def _call_model_safe(self, model, prompt: str) -> str | None:
        """Call a model with semaphore and error handling. Returns None on failure."""
        async with self._model_call_semaphore:
            try:
                return await model.invoke(prompt)
            except Exception as e:
                logger.warning(f"Model {model.name} failed: {e}")
                return None

    def _assign_perspectives(self, models: list) -> list[tuple]:
        """
        Assign prompt perspectives to models, cycling through
        outside/inside/advocate to maximize diversity.
        """
        perspectives = ["outside", "inside", "advocate"]
        return [
            (model, perspectives[i % len(perspectives)])
            for i, model in enumerate(models)
        ]

    # ======================== BINARY QUESTIONS ========================

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        today = datetime.now().strftime("%Y-%m-%d")
        conditional_disclaimer = self._get_conditional_disclaimer_if_necessary(question)

        prompt_templates = {
            "outside": BINARY_OUTSIDE_VIEW,
            "inside": BINARY_INSIDE_VIEW,
            "advocate": BINARY_DEVILS_ADVOCATE,
        }

        model_assignments = self._assign_perspectives(self.ensemble_models)

        # Fire all model calls in parallel
        tasks = []
        task_labels = []
        for model, perspective in model_assignments:
            prompt = prompt_templates[perspective].format(
                question_text=question.question_text,
                background_info=question.background_info or "",
                resolution_criteria=question.resolution_criteria or "",
                fine_print=question.fine_print or "",
                research=research,
                today=today,
                conditional_disclaimer=conditional_disclaimer,
            )
            tasks.append(self._call_model_safe(model, prompt))
            task_labels.append(f"{model.name}({perspective})")

        responses = await asyncio.gather(*tasks)

        # Parse probabilities from successful responses
        all_probs = []
        all_reasonings = []
        for label, response in zip(task_labels, responses):
            if response is not None:
                prob = parse_binary_probability(response)
                all_probs.append(prob)
                all_reasonings.append(f"### {label}: {prob:.1%}\n{response[:500]}...")
                logger.info(f"  {label} -> {prob:.1%}")

        if not all_probs:
            raise ValueError("All ensemble models failed for binary question")

        # Aggregate: median + extremize
        median_prob = statistics.median(all_probs)
        final_prob = extremize(median_prob, self.EXTREMIZE_FACTOR)
        final_prob = max(0.01, min(0.99, final_prob))

        combined_reasoning = (
            f"## Ensemble Forecast ({len(all_probs)} models)\n"
            f"Individual predictions: {[f'{p:.1%}' for p in all_probs]}\n"
            f"Median: {median_prob:.1%} -> Extremized ({self.EXTREMIZE_FACTOR}): {final_prob:.1%}\n\n"
            + "\n\n".join(all_reasonings)
        )

        logger.info(
            f"Binary ensemble for {question.page_url}: "
            f"preds={[round(p,3) for p in all_probs]} median={median_prob:.3f} final={final_prob:.3f}"
        )
        return ReasonedPrediction(prediction_value=final_prob, reasoning=combined_reasoning)

    # ======================== MULTIPLE CHOICE QUESTIONS ========================

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        today = datetime.now().strftime("%Y-%m-%d")
        conditional_disclaimer = self._get_conditional_disclaimer_if_necessary(question)
        model_assignments = self._assign_perspectives(self.ensemble_models)

        tasks = []
        task_labels = []
        for model, perspective in model_assignments:
            prompt = MC_PROMPT.format(
                question_text=question.question_text,
                options=question.options,
                background_info=question.background_info or "",
                resolution_criteria=question.resolution_criteria or "",
                fine_print=question.fine_print or "",
                research=research,
                today=today,
                view_instruction=VIEW_INSTRUCTIONS[perspective],
                conditional_disclaimer=conditional_disclaimer,
            )
            tasks.append(self._call_model_safe(model, prompt))
            task_labels.append(f"{model.name}({perspective})")

        responses = await asyncio.gather(*tasks)

        all_option_probs = []  # list of dicts
        all_reasonings = []
        for label, response in zip(task_labels, responses):
            if response is not None:
                probs = parse_multiple_choice(response, question.options)
                all_option_probs.append(probs)
                all_reasonings.append(f"### {label}\n{response[:500]}...")
                logger.info(f"  {label} -> {probs}")

        if not all_option_probs:
            raise ValueError("All ensemble models failed for MC question")

        # Aggregate: normalized median per option
        final_probs = {}
        for option in question.options:
            option_values = [d[option] for d in all_option_probs if option in d]
            final_probs[option] = statistics.median(option_values) if option_values else 1.0 / len(question.options)

        # Normalize
        total = sum(final_probs.values())
        if total > 0:
            final_probs = {k: max(0.01, v / total) for k, v in final_probs.items()}
            total = sum(final_probs.values())
            final_probs = {k: v / total for k, v in final_probs.items()}

        predicted_options = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=opt, probability=final_probs[opt])
                for opt in question.options
            ]
        )

        combined_reasoning = (
            f"## Ensemble MC Forecast ({len(all_option_probs)} models)\n"
            f"Final: {final_probs}\n\n"
            + "\n\n".join(all_reasonings)
        )

        logger.info(f"MC ensemble for {question.page_url}: {final_probs}")
        return ReasonedPrediction(prediction_value=predicted_options, reasoning=combined_reasoning)

    # ======================== NUMERIC QUESTIONS ========================

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        today = datetime.now().strftime("%Y-%m-%d")
        conditional_disclaimer = self._get_conditional_disclaimer_if_necessary(question)
        model_assignments = self._assign_perspectives(self.ensemble_models)

        tasks = []
        task_labels = []
        for model, perspective in model_assignments:
            prompt = NUMERIC_PROMPT.format(
                question_text=question.question_text,
                background_info=question.background_info or "",
                resolution_criteria=question.resolution_criteria or "",
                fine_print=question.fine_print or "",
                unit_of_measure=question.unit_of_measure or "Not stated (please infer)",
                research=research,
                today=today,
                lower_bound_message=lower_bound_message,
                upper_bound_message=upper_bound_message,
                view_instruction=VIEW_INSTRUCTIONS[perspective],
                conditional_disclaimer=conditional_disclaimer,
            )
            tasks.append(self._call_model_safe(model, prompt))
            task_labels.append(f"{model.name}({perspective})")

        responses = await asyncio.gather(*tasks)

        # Collect percentile dicts from each model
        all_percentile_sets = []
        all_reasonings = []
        for label, response in zip(task_labels, responses):
            if response is None:
                continue
            pvals = parse_percentiles(response)
            if len(pvals) < 4:
                # LLM fallback re-prompt for extraction
                extraction_prompt = f"""Extract ONLY the percentile forecast values from the text below.
Output EXACTLY in this format with nothing else:
Percentile 10: [number]
Percentile 20: [number]
Percentile 40: [number]
Percentile 60: [number]
Percentile 80: [number]
Percentile 90: [number]

Text to extract from:
{response}"""
                try:
                    extraction = await self.foresight.invoke(extraction_prompt)
                    pvals = parse_percentiles(extraction)
                except Exception:
                    pass

            if len(pvals) >= 4:
                all_percentile_sets.append(pvals)
                all_reasonings.append(f"### {label}\n{response[:500]}...")
                logger.info(f"  {label} -> {pvals}")
            else:
                logger.warning(f"  {label}: only got {len(pvals)} percentiles, skipping")

        if not all_percentile_sets:
            raise ValueError("All ensemble models failed for numeric question")

        # Aggregate: median at each percentile level
        standard_pcts = [10, 20, 40, 60, 80, 90]
        merged = {}
        for pct in standard_pcts:
            values = [ps[pct] for ps in all_percentile_sets if pct in ps]
            if values:
                merged[pct] = statistics.median(values)

        if len(merged) < 2:
            raise ValueError(f"Not enough merged percentiles: {merged}")

        percentile_list = [
            Percentile(percentile=p / 100.0, value=v)
            for p, v in sorted(merged.items())
        ]

        # Ensure strictly increasing
        for i in range(len(percentile_list) - 1):
            if percentile_list[i].value >= percentile_list[i + 1].value:
                percentile_list[i + 1] = Percentile(
                    percentile=percentile_list[i + 1].percentile,
                    value=percentile_list[i].value + 0.001,
                )

        prediction = NumericDistribution.from_question(percentile_list, question)
        combined_reasoning = (
            f"## Ensemble Numeric Forecast ({len(all_percentile_sets)} models)\n"
            f"Merged percentiles: {merged}\n\n"
            + "\n\n".join(all_reasonings)
        )

        logger.info(f"Numeric ensemble for {question.page_url}: {merged}")
        return ReasonedPrediction(prediction_value=prediction, reasoning=combined_reasoning)

    # ======================== DATE QUESTIONS ========================

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        today = datetime.now().strftime("%Y-%m-%d")
        conditional_disclaimer = self._get_conditional_disclaimer_if_necessary(question)
        model_assignments = self._assign_perspectives(self.ensemble_models)

        tasks = []
        task_labels = []
        for model, perspective in model_assignments:
            prompt = DATE_PROMPT.format(
                question_text=question.question_text,
                background_info=question.background_info or "",
                resolution_criteria=question.resolution_criteria or "",
                fine_print=question.fine_print or "",
                research=research,
                today=today,
                lower_bound_message=lower_bound_message,
                upper_bound_message=upper_bound_message,
                conditional_disclaimer=conditional_disclaimer,
            )
            tasks.append(self._call_model_safe(model, prompt))
            task_labels.append(f"{model.name}({perspective})")

        responses = await asyncio.gather(*tasks)

        from datetime import datetime as dt
        date_pattern = r"(?:P|p)ercentile\s*(\d+)\s*:\s*(\d{4}-\d{2}-\d{2})"

        all_date_sets = []  # list of dict[pct -> timestamp]
        all_reasonings = []
        for label, response in zip(task_labels, responses):
            if response is None:
                continue
            date_matches = re.findall(date_pattern, response)
            if len(date_matches) >= 4:
                date_dict = {}
                for pct_str, date_str in date_matches:
                    try:
                        parsed = dt.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        date_dict[float(pct_str)] = parsed.timestamp()
                    except ValueError:
                        pass
                if len(date_dict) >= 4:
                    all_date_sets.append(date_dict)
                    all_reasonings.append(f"### {label}\n{response[:500]}...")
                    logger.info(f"  {label} -> {len(date_dict)} date percentiles")
            else:
                logger.warning(f"  {label}: only {len(date_matches)} date matches, skipping")

        if not all_date_sets:
            # Fallback to single Foresight call
            logger.warning("Date ensemble failed, falling back to single Foresight call")
            prompt = DATE_PROMPT.format(
                question_text=question.question_text,
                background_info=question.background_info or "",
                resolution_criteria=question.resolution_criteria or "",
                fine_print=question.fine_print or "",
                research=research,
                today=today,
                lower_bound_message=lower_bound_message,
                upper_bound_message=upper_bound_message,
                conditional_disclaimer=conditional_disclaimer,
            )
            response = await self.foresight.invoke(prompt)
            date_matches = re.findall(date_pattern, response)
            if len(date_matches) < 2:
                raise ValueError("Could not parse date percentiles even from fallback")
            date_dict = {}
            for pct_str, date_str in date_matches:
                parsed = dt.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                date_dict[float(pct_str)] = parsed.timestamp()
            all_date_sets.append(date_dict)
            all_reasonings.append(f"### foresight-v3 (fallback)\n{response[:500]}...")

        # Aggregate: median timestamps per percentile
        standard_pcts = [10, 20, 40, 60, 80, 90]
        merged = {}
        for pct in standard_pcts:
            values = [ds[pct] for ds in all_date_sets if pct in ds]
            if values:
                merged[pct] = statistics.median(values)

        if len(merged) < 2:
            raise ValueError(f"Not enough date percentiles: {merged}")

        percentile_list = [
            Percentile(percentile=p / 100.0, value=v)
            for p, v in sorted(merged.items())
        ]

        # Ensure strictly increasing
        for i in range(len(percentile_list) - 1):
            if percentile_list[i].value >= percentile_list[i + 1].value:
                percentile_list[i + 1] = Percentile(
                    percentile=percentile_list[i + 1].percentile,
                    value=percentile_list[i].value + 86400,  # +1 day
                )

        prediction = NumericDistribution.from_question(percentile_list, question)
        combined_reasoning = (
            f"## Ensemble Date Forecast ({len(all_date_sets)} models)\n\n"
            + "\n\n".join(all_reasonings)
        )

        logger.info(f"Date ensemble for {question.page_url}: {len(all_date_sets)} models contributed")
        return ReasonedPrediction(prediction_value=prediction, reasoning=combined_reasoning)

    # ======================== CONDITIONAL QUESTIONS ========================

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = f"""
## Parent Question Reasoning
{parent_info.reasoning}
## Child Question Reasoning
{child_info.reasoning}
## Yes Question Reasoning
{yes_info.reasoning}
## No Question Reasoning
{no_info.reasoning}
"""
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,
            child=child_info.prediction_value,
            prediction_yes=yes_info.prediction_value,
            prediction_no=no_info.prediction_value,
        )
        return ReasonedPrediction(reasoning=full_reasoning, prediction_value=full_prediction)

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research

    def _add_reasoning_to_research(
        self, research: str, reasoning: ReasonedPrediction[PredictionTypes], question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer
        question_type = question_type.title()
        return f"""{research}
---
## {question_type} Question Information
You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
"""

    # ======================== HELPERS ========================

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            upper_bound_number = question.nominal_upper_bound or question.upper_bound
            lower_bound_number = question.nominal_lower_bound or question.lower_bound
            unit = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_msg = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit}."
        else:
            upper_msg = f"The outcome can not be higher than {upper_bound_number} {unit}."

        if question.open_lower_bound:
            lower_msg = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit}."
        else:
            lower_msg = f"The outcome can not be lower than {lower_bound_number} {unit}."

        return upper_msg, lower_msg

    def _get_conditional_disclaimer_if_necessary(self, question: MetaculusQuestion) -> str:
        if not hasattr(question, 'conditional_type') or question.conditional_type not in ["yes", "no"]:
            return ""
        return "As you are given a conditional question, only forecast the CHILD question given the parent question's resolution."


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    for noisy in ["LiteLLM", "openai.agents", "httpx", "httpcore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Run CassandraBot v2")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    # ============================================================
    # Initialize ensemble
    # Models are auto-detected from environment variables.
    # Set LIGHTNINGROD_API_KEY for Foresight v3.
    # Set OPENROUTER_API_KEY for all OpenRouter models.
    # Set ASKNEWS_CLIENT_ID + ASKNEWS_SECRET for news research.
    # ============================================================

    foresight = ForesightLlm(temperature=0.3, max_tokens=4000, timeout=180)

    cassandra_bot = CassandraBot(
        foresight=foresight,
        research_reports_per_question=1,
        predictions_per_research_report=1,  # We do our own ensemble internally
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            # The parent ForecastBot uses these for internal operations.
            # We override run_research and all _run_forecast_on_* methods,
            # but the parent still uses "summarizer" for report generation.
            # Use a cheap OpenRouter model so it doesn't error.
            "default": "openrouter/openai/gpt-4o-mini",
            "summarizer": "openrouter/openai/gpt-4o-mini",
            "researcher": "openrouter/openai/gpt-4o-mini",
            "parser": "openrouter/openai/gpt-4o-mini",
        },
    )

    client = MetaculusClient()

    if run_mode == "tournament":
        seasonal_reports = asyncio.run(
            cassandra_bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            cassandra_bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_reports + minibench_reports

    elif run_mode == "metaculus_cup":
        cassandra_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            cassandra_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )

    elif run_mode == "test_questions":
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
        ]
        cassandra_bot.skip_previously_forecasted_questions = False
        cassandra_bot.publish_reports_to_metaculus = False
        questions = [
            client.get_question_by_url(url) for url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            cassandra_bot.forecast_questions(questions, return_exceptions=True)
        )

    cassandra_bot.log_report_summary(forecast_reports)