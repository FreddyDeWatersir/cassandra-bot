"""
CassandraBot - Metaculus Forecasting Bot powered by Lightning Rod's Foresight v3.

This bot bypasses litellm entirely for LLM calls, using the OpenAI-compatible
API from Lightning Rod directly. This avoids litellm's provider routing issues
with custom model names.

The forecasting-tools library is still used for:
- Loading questions from Metaculus
- Submitting predictions to Metaculus
- The ForecastBot orchestration (research -> forecast -> aggregate -> submit)
- NumericDistribution CDF generation
"""

import argparse
import asyncio
import logging
import os
import re
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
# FORESIGHT LLM WRAPPER
# ============================================================
# Direct OpenAI-compatible client for Lightning Rod's Foresight v3.
# Bypasses litellm entirely to avoid provider routing issues.
# ============================================================

class ForesightLlm:
    """Direct wrapper for Lightning Rod's Foresight API."""

    def __init__(self, temperature=0.3, max_tokens=4000, timeout=120):
        self.temperature = temperature
        self.max_tokens = max_tokens
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


# ============================================================
# PARSING HELPERS
# ============================================================
# Since we're not using GeneralLlm, we can't use structure_output
# (which requires a GeneralLlm for parsing). Instead we parse
# the LLM output directly with regex. This is what the
# bot_from_scratch.py template does too.
# ============================================================

def parse_binary_probability(text: str) -> float:
    """Extract probability from text like 'Probability: 35%'."""
    # Look specifically for the "Probability: XX%" pattern first
    specific_match = re.search(
        r"(?:Probability|PROBABILITY)\s*:\s*(\d+(?:\.\d+)?)\s*%",
        text
    )
    if specific_match:
        prob = float(specific_match.group(1)) / 100.0
        return max(0.01, min(0.99, prob))
    
    # Fallback: look for "Probability: 0.XX" (decimal format)
    decimal_match = re.search(
        r"(?:Probability|PROBABILITY)\s*:\s*(\d*\.\d+)",
        text
    )
    if decimal_match:
        prob = float(decimal_match.group(1))
        if prob > 1:  # It was actually a percentage
            prob = prob / 100.0
        return max(0.01, min(0.99, prob))
    
    # Last resort: take the last percentage in the text
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

    # Format 2: Markdown table "| 10 % | **115** |" or "| 10 | 115 |"
    if len(results) < 4:
        pattern2 = r"\|\s*(\d+)\s*%?\s*\|\s*\**\s*[≈~]?\s*(-?[\d,]+(?:\.\d+)?)"
        for match in re.finditer(pattern2, text):
            pct = float(match.group(1))
            if 5 <= pct <= 95:
                val = float(match.group(2).replace(",", "").replace(" ", ""))
                results[pct] = val

    # Format 3: "10%: 115" or "10 %  | **115**"
    if len(results) < 4:
        pattern3 = r"(\d+)\s*%\s*[:\|]\s*\**\s*[≈~]?\s*(-?[\d,]+(?:\.\d+)?)"
        for match in re.finditer(pattern3, text):
            pct = float(match.group(1))
            if 5 <= pct <= 95:
                val = float(match.group(2).replace(",", ""))
                results[pct] = val

    # Format 4: Inside code blocks "Percentile 10: 115"
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

    # Try to find lines with option names and percentages
    for option in options:
        # Escape special regex characters in option name
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

    # If we didn't find all options, try finding the last N numbers
    if len(results) < len(options):
        all_numbers = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
        if len(all_numbers) >= len(options):
            last_n = all_numbers[-len(options):]
            results = {opt: float(num) for opt, num in zip(options, last_n)}

    # Normalize to sum to 1
    if results:
        total = sum(results.values())
        if total > 0:
            results = {k: max(0.01, min(0.99, v / total)) for k, v in results.items()}
            # Re-normalize after clamping
            total = sum(results.values())
            results = {k: v / total for k, v in results.items()}

    # Fallback: equal probabilities
    if not results or len(results) < len(options):
        equal_prob = 1.0 / len(options)
        results = {opt: equal_prob for opt in options}

    return results


# ============================================================
# THE BOT
# ============================================================

class CassandraBot(ForecastBot):
    """
    CassandraBot - Forecasting bot powered by Lightning Rod's Foresight v3.

    Uses direct OpenAI API calls instead of litellm for LLM inference.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, foresight: ForesightLlm = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.foresight = foresight or ForesightLlm()

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            researcher = self.get_llm("researcher")

            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                You do not produce forecasts yourself.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )

            if (
                isinstance(researcher, str)
                and researcher.startswith("asknews/")
            ):
                try:
                    research = await AskNewsSearcher().call_preconfigured_version(
                        researcher, prompt
                    )
                except Exception as e:
                    logger.warning(f"AskNews failed: {e}. Using Foresight for research.")
                    research = await self.foresight.invoke(prompt)
            elif not researcher or researcher == "None" or researcher == "no_research":
                # Use Foresight itself for basic reasoning about the question
                research = await self.foresight.invoke(prompt)
            else:
                research = await self.foresight.invoke(prompt)

            logger.info(f"Found Research for URL {question.page_url}:\n{research[:200]}...")
            return research

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        reasoning = await self.foresight.invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")

        decimal_pred = parse_binary_probability(reasoning)
        logger.info(f"Forecasted URL {question.page_url} with prediction: {decimal_pred}")

        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of a scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A%
            Option_B: Probability_B%
            ...
            Option_N: Probability_N%
            """
        )

        reasoning = await self.foresight.invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")

        probs = parse_multiple_choice(reasoning, question.options)
        predicted_options = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=opt, probability=probs.get(opt, 1.0 / len(question.options)))
                for opt in question.options
            ]
        )

        logger.info(f"Forecasted URL {question.page_url} with prediction: {predicted_options}")
        return ReasonedPrediction(prediction_value=predicted_options, reasoning=reasoning)

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units.
            - Never use scientific notation.
            - Always start with a smaller number and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

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
        )

        reasoning = await self.foresight.invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")

        percentile_values = parse_percentiles(reasoning)
        # Fallback: if regex couldn't parse, ask Foresight to extract just the numbers
        if len(percentile_values) < 4:
            logger.warning(f"Regex parsing got only {len(percentile_values)} percentiles, using LLM fallback")
            extraction_prompt = f"""Extract ONLY the percentile forecast values from the text below.
Output EXACTLY in this format with nothing else:
Percentile 10: [number]
Percentile 20: [number]
Percentile 40: [number]
Percentile 60: [number]
Percentile 80: [number]
Percentile 90: [number]

Text to extract from:
{reasoning}"""
            extraction = await self.foresight.invoke(extraction_prompt)
            percentile_values = parse_percentiles(extraction)
            
            if len(percentile_values) < 2:
                raise ValueError(f"Could not parse enough percentiles even with LLM fallback. Got: {percentile_values}")

        percentile_list = [
            Percentile(percentile=p / 100.0, value=v)
            for p, v in sorted(percentile_values.items())
        ]

        # Ensure values are in increasing order
        for i in range(len(percentile_list) - 1):
            if percentile_list[i].value >= percentile_list[i + 1].value:
                logger.warning(f"Percentile values not in order, adjusting")
                percentile_list[i + 1] = Percentile(
                    percentile=percentile_list[i + 1].percentile,
                    value=percentile_list[i].value + 0.001,
                )

        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}")

        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

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
        )

        reasoning = await self.foresight.invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")

        # Parse dates from percentile lines
        from datetime import datetime as dt
        date_pattern = r"(?:P|p)ercentile\s*(\d+)\s*:\s*(\d{4}-\d{2}-\d{2})"
        date_matches = re.findall(date_pattern, reasoning)

        if len(date_matches) < 2:
            raise ValueError(f"Could not parse enough date percentiles from response")

        percentile_list = []
        for pct_str, date_str in date_matches:
            parsed_date = dt.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            percentile_list.append(
                Percentile(
                    percentile=float(pct_str) / 100.0,
                    value=parsed_date.timestamp(),
                )
            )

        percentile_list.sort(key=lambda p: p.percentile)
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}")

        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### CONDITIONAL QUESTIONS #####################################

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

    ##################################### HELPERS #####################################

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
        if question.conditional_type not in ["yes", "no"]:
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

    # Suppress noisy loggers
    for noisy in ["LiteLLM", "openai.agents", "httpx", "httpcore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Run CassandraBot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    # ============================================================
    # Initialize Foresight v3 - purpose-built forecasting model
    # Beat o3, Grok 4, and Claude Opus on live prediction markets
    # Cost: ~$0.05 per question (5 runs) = ~$25-35 for full tournament
    # ============================================================

    foresight = ForesightLlm(
        temperature=0.3,
        max_tokens=4000,
        timeout=120,
    )

    cassandra_bot = CassandraBot(
        foresight=foresight,
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            # These are needed by the parent ForecastBot class
            # but we override all the methods that use them
            "default": "no_research",
            "summarizer": "no_research",
            "researcher": "no_research",
            "parser": "no_research",
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