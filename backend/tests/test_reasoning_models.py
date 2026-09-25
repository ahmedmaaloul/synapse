# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Ahmed Maaloul <ahmed.maaloul@proton.me>
# Synapse — https://github.com/ahmedmaaloul/synapse
"""
Tests for OpenAI reasoning models (gpt-5, o-series) and pricing honesty.

Hermetic: no model is ever called. ``ChatOpenAI`` is either replaced by a
recorder that captures its constructor kwargs, or constructed for real with a
fake key and asked only for the request payload it WOULD send (no network).
Usage is read from hand-built ``AIMessage``s and from langchain-openai's own
response parser fed a fake Chat Completions body.

Four jobs:

  1. **The provider.** A reasoning model gets ``reasoning_effort`` and no
     temperature on the openai, azure_openai and openai_compatible branches;
     every other model keeps its temperature and gets no effort.
  2. **The dry run.** A reasoning model's estimate carries a per-call
     reasoning allowance, and the printed plan says it is an assumption.
  3. **Price dates.** gpt-5-nano / gpt-5-mini quote their own verification
     date; everything else keeps ``PRICES_CHECKED_ON``.
  4. **The metered cap counts reasoning tokens**, because the provider's
     ``output_tokens`` already includes them.
"""

from __future__ import annotations

import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.config import Settings, get_settings
from app.services import llm_provider
from app.services.llm_provider import (
    get_chat_llm,
    is_reasoning_model,
    reasoning_effort_for,
)
from app.services.procedural_store import load_prior
from benchmarks.procedural import run_procedural as rp
from benchmarks.public import cost

FAKE_KEY = "sk-test-not-a-real-key"
GPT5_CHECKED_ON = "2026-09-24"
#: A gpt-5-nano call that reasoned: 512 of its 600 output tokens were hidden.
REASONING_USAGE = {
    "input_tokens": 100,
    "output_tokens": 600,
    "total_tokens": 700,
    "output_token_details": {"reasoning": 512},
}


# ── Fakes ────────────────────────────────────────────────────────────────────
class RecordingChatModel:
    """Stands in for ChatOpenAI / AzureChatOpenAI: keeps the kwargs it was built with."""

    built: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        RecordingChatModel.built.append(kwargs)


@pytest.fixture
def recorder(monkeypatch):
    """Route every ChatOpenAI-family construction to :class:`RecordingChatModel`."""
    RecordingChatModel.built = []
    monkeypatch.setattr(llm_provider, "_import_chat_openai", lambda: RecordingChatModel)
    langchain_openai = pytest.importorskip("langchain_openai")
    monkeypatch.setattr(langchain_openai, "AzureChatOpenAI", RecordingChatModel)
    return RecordingChatModel


def openai_settings(model: str, **overrides) -> Settings:
    return Settings(
        llm_provider="openai", openai_api_key=FAKE_KEY, openai_chat_model=model, **overrides
    )


def azure_settings(deployment: str) -> Settings:
    return Settings(
        llm_provider="azure_openai",
        azure_openai_api_key=FAKE_KEY,
        azure_openai_endpoint="https://example.openai.azure.com/",
        azure_openai_chat_deployment=deployment,
    )


def compatible_settings(model: str) -> Settings:
    return Settings(
        llm_provider="openai_compatible",
        openai_compatible_base_url="http://localhost:8000/v1",
        openai_compatible_api_key="",
        openai_compatible_chat_model=model,
    )


class ReasoningModel:
    """A chat model whose every answer spent 512 hidden reasoning tokens."""

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages, *args, **kwargs):
        self.calls += 1
        return AIMessage(content="Thought: done.", usage_metadata=dict(REASONING_USAGE))


@pytest.fixture
def no_spending(monkeypatch):
    """Every route to a paid model, an embedder or Neo4j explodes (and is recorded)."""
    calls: list[str] = []

    def explode(name):
        def _raise(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} was called during a run that must not spend")

        return _raise

    from app import neo4j_driver

    for module, attribute in (
        (llm_provider, "get_chat_llm"),
        (llm_provider, "get_embeddings"),
        (neo4j_driver, "get_driver"),
        (neo4j_driver, "verify_connectivity"),
        (rp, "make_models"),
    ):
        monkeypatch.setattr(module, attribute, explode(f"{module.__name__}.{attribute}"))
    return calls


@pytest.fixture
def demo() -> rp.Dataset:
    return rp.load_demo_dataset()


@pytest.fixture
def prior():
    return load_prior("graphrag-navigator")


# ── 1. The provider ──────────────────────────────────────────────────────────
class TestReasoningModelDetection:
    @pytest.mark.parametrize(
        "model",
        ["gpt-5-nano", "gpt-5-mini", "gpt-5", "gpt-5-nano-2025-08-07", "GPT-5-Nano", "o1",
         "o3-mini", "o4-mini", " o3 "],
    )
    def test_reasoning_models_are_recognised(self, model):
        assert is_reasoning_model(model)

    @pytest.mark.parametrize(
        "model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-nano", "omni-moderation", "o", "gpt-oss-120b",
         "openai/gpt-5-nano", "", None],
    )
    def test_everything_else_is_not(self, model):
        # Anchored on purpose: a gateway's "openai/gpt-5-nano" id is not matched.
        assert not is_reasoning_model(model)

    def test_minimal_effort_is_gpt5_only_so_the_o_series_gets_low(self):
        assert reasoning_effort_for("gpt-5-nano", "minimal") == "minimal"
        assert reasoning_effort_for("o3-mini", "minimal") == "low"
        assert reasoning_effort_for("o3-mini", "high") == "high"
        assert reasoning_effort_for("gpt-5-nano", " HIGH ") == "high"
        assert reasoning_effort_for("gpt-5-nano", "") == "minimal"

    def test_the_setting_defaults_to_minimal(self):
        # The declared default, not Settings(): a local .env may override it.
        assert Settings.model_fields["openai_reasoning_effort"].default == "minimal"
        assert Settings(openai_reasoning_effort="low").openai_reasoning_effort == "low"


class TestProviderKwargs:
    """The kwargs ChatOpenAI is CONSTRUCTED with, per model id."""

    def test_gpt5_nano_gets_reasoning_effort_and_no_temperature(self, recorder):
        get_chat_llm(temperature=0, settings=openai_settings("gpt-5-nano"))
        (kwargs,) = recorder.built
        assert kwargs["model"] == "gpt-5-nano"
        assert "temperature" not in kwargs  # even though the caller asked for 0
        assert kwargs["reasoning_effort"] == "minimal"

    def test_gpt4o_mini_keeps_its_temperature_and_gets_no_effort(self, recorder):
        get_chat_llm(temperature=0, settings=openai_settings("gpt-4o-mini"))
        (kwargs,) = recorder.built
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["temperature"] == 0
        assert "reasoning_effort" not in kwargs

    def test_the_default_temperature_still_applies_to_classic_models(self, recorder):
        get_chat_llm(settings=openai_settings("gpt-4o-mini"))
        assert recorder.built[0]["temperature"] == 0.2

    def test_the_effort_comes_from_the_setting(self, recorder):
        get_chat_llm(settings=openai_settings("gpt-5-mini", openai_reasoning_effort="medium"))
        assert recorder.built[0]["reasoning_effort"] == "medium"

    def test_an_o_series_model_is_sent_low_instead_of_minimal(self, recorder):
        get_chat_llm(settings=openai_settings("o3-mini"))
        (kwargs,) = recorder.built
        assert kwargs["reasoning_effort"] == "low" and "temperature" not in kwargs

    def test_json_mode_survives_on_a_reasoning_model(self, recorder):
        get_chat_llm(json_mode=True, temperature=0, settings=openai_settings("gpt-5-nano"))
        (kwargs,) = recorder.built
        assert kwargs["model_kwargs"] == {"response_format": {"type": "json_object"}}
        assert "temperature" not in kwargs

    def test_azure_matches_on_the_deployment_name(self, recorder):
        get_chat_llm(temperature=0, settings=azure_settings("gpt-5-nano"))
        get_chat_llm(temperature=0, settings=azure_settings("gpt-4o-mini"))
        reasoning, classic = recorder.built
        assert reasoning["azure_deployment"] == "gpt-5-nano"
        assert "temperature" not in reasoning and reasoning["reasoning_effort"] == "minimal"
        assert classic["temperature"] == 0 and "reasoning_effort" not in classic

    def test_openai_compatible_applies_only_when_the_id_matches(self, recorder):
        get_chat_llm(temperature=0, settings=compatible_settings("gpt-5-nano"))
        get_chat_llm(temperature=0, settings=compatible_settings("local-model"))
        reasoning, classic = recorder.built
        assert "temperature" not in reasoning and reasoning["reasoning_effort"] == "minimal"
        assert classic["temperature"] == 0 and "reasoning_effort" not in classic

    def test_providers_outside_the_openai_family_are_untouched(self, monkeypatch):
        # Ollama never takes a reasoning effort, whatever its model is called.
        pytest.importorskip("langchain_ollama")
        s = Settings(llm_provider="ollama", ollama_chat_model="gpt-5-nano")
        llm = get_chat_llm(temperature=0, settings=s)
        assert llm.temperature == 0


class TestInstalledLangchainOpenAI:
    """Against the INSTALLED langchain-openai: the supported field, and the wire payload."""

    def test_reasoning_effort_is_a_declared_chat_openai_field(self):
        langchain_openai = pytest.importorskip("langchain_openai")
        fields = langchain_openai.ChatOpenAI.model_fields
        assert "reasoning_effort" in fields
        assert fields["temperature"].default is None  # a None temperature is not sent

    def test_the_gpt5_request_payload_has_no_temperature(self):
        pytest.importorskip("langchain_openai")
        llm = get_chat_llm(temperature=0, settings=openai_settings("gpt-5-nano"))
        payload = llm._get_request_payload([HumanMessage(content="hi")])
        assert "temperature" not in payload
        assert payload["reasoning_effort"] == "minimal"
        assert payload["model"] == "gpt-5-nano"

    def test_the_gpt4o_mini_request_payload_has_its_temperature(self):
        pytest.importorskip("langchain_openai")
        llm = get_chat_llm(temperature=0, settings=openai_settings("gpt-4o-mini"))
        payload = llm._get_request_payload([HumanMessage(content="hi")])
        assert payload["temperature"] == 0
        assert "reasoning_effort" not in payload


# ── 2. The dry run ───────────────────────────────────────────────────────────
class TestDryRunReasoningAllowance:
    def test_the_allowance_applies_when_either_model_reasons(self):
        assert rp.REASONING_TOKENS_ALLOWANCE == 512
        assert rp.reasoning_allowance("gpt-5-nano") == 512
        assert rp.reasoning_allowance("gpt-4o-mini") == 0
        # A deployment alias priced as gpt-5-nano is still sized as one.
        assert rp.reasoning_allowance("my-deployment", "gpt-5-nano") == 512

    def test_every_call_carries_the_allowance_and_nothing_else_moves(self, demo, prior):
        kwargs = {
            "systems": ["no_pg", "pg_gen_local"], "prior": prior, "max_steps": 3,
            "evolve_rounds": 1, "evolve_batch_size": 3,
        }
        plain = rp.estimate_plan(demo, **kwargs)
        reasoning = rp.estimate_plan(demo, reasoning_tokens=512, **kwargs)
        assert plain.reasoning_tokens_per_call == 0
        assert reasoning.reasoning_tokens_per_call == 512
        for before, after in zip(plain.phases, reasoning.phases, strict=True):
            assert after.usage.calls == before.usage.calls
            # Hidden reasoning never re-enters a prompt.
            assert after.usage.prompt_tokens == before.usage.prompt_tokens
            assert after.usage.completion_tokens == (
                before.usage.completion_tokens + 512 * before.usage.calls
            )
            assert after.usage.reasoning_tokens == 512 * before.usage.calls
        assert reasoning.usd("gpt-5-nano") > plain.usd("gpt-5-nano")

    def test_the_printed_plan_says_the_allowance_is_an_assumption(self, demo, prior):
        plan = rp.estimate_plan(
            demo, systems=["no_pg"], prior=prior, max_steps=4, reasoning_tokens=512
        )
        text = "\n".join(
            rp.dry_run_lines(
                plan, demo, systems=["no_pg"], model="gpt-5-nano", max_steps=4, max_usd=2.0
            )
        )
        assert "REASONING model" in text
        assert "512-token reasoning allowance" in text
        assert "ASSUMPTION" in text and "calibrated on a measured run" in text
        assert "Effort sent: 'minimal'" in text
        assert "probably LOW" not in text
        assert f"(incl. {plan.calls * 512:,} ASSUMED reasoning tokens)" in text

    def test_a_higher_effort_is_flagged_as_underestimated(self, demo, prior):
        settings = get_settings().model_copy(update={"openai_reasoning_effort": "high"})
        plan = rp.estimate_plan(
            demo, systems=["no_pg"], prior=prior, max_steps=4, reasoning_tokens=512
        )
        text = "\n".join(
            rp.dry_run_lines(
                plan, demo, systems=["no_pg"], model="gpt-5-nano", max_steps=4, max_usd=2.0,
                settings=settings,
            )
        )
        assert "Effort sent: 'high'" in text and "probably LOW" in text

    def test_a_classic_model_plan_has_no_reasoning_line(self, demo, prior):
        plan = rp.estimate_plan(demo, systems=["no_pg"], prior=prior, max_steps=4)
        text = "\n".join(
            rp.dry_run_lines(
                plan, demo, systems=["no_pg"], model="gpt-4o-mini", max_steps=4, max_usd=2.0
            )
        )
        assert "REASONING" not in text and "reasoning allowance" not in text
        assert "ASSUMED reasoning" not in text

    async def test_a_gpt5_nano_dry_run_is_free_and_sized_with_the_allowance(
        self, no_spending, capsys
    ):
        assert await rp.main(["--dry-run", "--model", "gpt-5-nano"]) == 0
        out = capsys.readouterr().out
        assert "512-token reasoning allowance" in out
        assert f"prices hand-recorded on {GPT5_CHECKED_ON}" in out
        assert no_spending == []

    async def test_a_configured_gpt5_nano_is_sized_without_a_model_flag(
        self, no_spending, monkeypatch, capsys
    ):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-5-nano")
        get_settings.cache_clear()
        assert await rp.main(["--dry-run"]) == 0
        assert "gpt-5-nano is a REASONING model" in capsys.readouterr().out
        assert no_spending == []

    async def test_the_real_run_preflight_counts_the_allowance(
        self, no_spending, monkeypatch, demo, prior, capsys
    ):
        # A cap between the plain and the reasoning-sized bound: the pre-flight
        # must refuse, which it only does if it added the allowance.
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-5-nano")
        get_settings.cache_clear()
        kwargs = {
            "systems": list(rp.BASE_SYSTEMS), "prior": prior,
            "max_steps": get_settings().agent_max_steps,
        }
        plain = rp.estimate_plan(demo, **kwargs).usd("gpt-5-nano")
        sized = rp.estimate_plan(demo, reasoning_tokens=512, **kwargs).usd("gpt-5-nano")
        cap = (plain + sized) / 2
        assert await rp.main(["--max-usd", f"{cap:.6f}"]) == 5
        assert "exceeds --max-usd" in capsys.readouterr().err
        assert no_spending == []


# ── 3. Price dates ───────────────────────────────────────────────────────────
class TestPerModelPriceDates:
    @pytest.mark.parametrize("model", ["gpt-5-nano", "gpt-5-mini", "gpt-5-nano-2025-08-07"])
    def test_the_gpt5_prices_carry_their_own_date(self, model):
        assert cost.price_checked_on(model) == GPT5_CHECKED_ON

    @pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-4.1-nano", "text-embedding-3-small"])
    def test_everything_else_keeps_the_table_date(self, model):
        assert cost.price_checked_on(model) == cost.PRICES_CHECKED_ON

    def test_an_unpriced_model_gets_the_date_of_the_table_it_is_missing_from(self):
        assert cost.price_checked_on("my-local-model") == cost.PRICES_CHECKED_ON

    def test_a_custom_table_is_honoured(self):
        table = {"a": cost.Price(1.0, 2.0), "b": cost.Price(1.0, 2.0, checked_on="2030-01-01")}
        assert cost.price_checked_on("a", table) == cost.PRICES_CHECKED_ON
        assert cost.price_checked_on("b", table) == "2030-01-01"

    def test_the_date_is_provenance_not_price(self):
        assert cost.Price(0.05, 0.40) == cost.PRICES_USD_PER_1M_TOKENS["gpt-5-nano"]
        assert cost.resolve_price("gpt-5-nano") == cost.Price(0.05, 0.40)

    def test_a_ledger_quotes_the_date_of_the_price_it_used(self):
        nano = cost.CostLedger("gpt-5-nano")
        nano.record(cost.Usage(prompt_tokens=1000, calls=1))
        text = "\n".join(nano.lines())
        assert f"hand-recorded on {GPT5_CHECKED_ON}" in text
        assert cost.PRICES_CHECKED_ON not in text
        assert nano.summary()["prices_checked_on"] == GPT5_CHECKED_ON

        mini = cost.CostLedger("gpt-4o-mini")
        mini.record(cost.Usage(prompt_tokens=1000, calls=1))
        assert f"hand-recorded on {cost.PRICES_CHECKED_ON}" in "\n".join(mini.lines())
        assert mini.summary()["prices_checked_on"] == cost.PRICES_CHECKED_ON

    def test_the_harness_cap_message_quotes_the_priced_model_date(self):
        guard = rp.SpendGuard("gpt-5-nano", max_usd=0.0, max_calls=None)
        with pytest.raises(rp.BudgetExceeded, match=f"estimated from {GPT5_CHECKED_ON} prices"):
            guard.check()

    def test_the_harness_provenance_quotes_the_priced_model_date(self, demo, prior):
        args = rp.build_parser().parse_args([])
        settings = get_settings().model_copy(
            update={"llm_provider": "openai", "openai_chat_model": "gpt-5-nano"}
        )
        result = rp.BenchResult(dataset=demo, prior=prior, max_steps=8)
        provenance = rp.build_provenance(
            result, args=args, settings=settings, model="gpt-5-nano", commit="abc",
            date="2026-09-24", priced_as="gpt-5-nano",
        )
        assert provenance["Prices"].startswith(f"`gpt-5-nano` hand-recorded {GPT5_CHECKED_ON}")
        # A reasoning model is not sent a temperature; the provenance must not claim one.
        assert "no temperature sent" in provenance["Model"]
        assert "reasoning effort `minimal`" in provenance["Model"]
        assert "temperature 0.0" not in provenance["Model"]

        classic = rp.build_provenance(
            result, args=args, settings=get_settings(), model="gpt-4o-mini", commit="abc",
            date="2026-09-24", priced_as="gpt-4o-mini",
        )
        assert classic["Prices"].startswith(f"`gpt-4o-mini` hand-recorded {cost.PRICES_CHECKED_ON}")
        assert "temperature" in classic["Model"] and "reasoning" not in classic["Model"]


# ── 4. The metered cap counts reasoning tokens ───────────────────────────────
class TestReasoningTokensAreMetered:
    def test_output_tokens_already_include_the_reasoning_breakdown(self):
        usage = cost.usage_from_response(AIMessage(content="x", usage_metadata=REASONING_USAGE))
        # Not 88 (reasoning dropped) and not 1112 (reasoning counted twice).
        assert usage.completion_tokens == 600
        assert usage.reasoning_tokens == 512
        assert usage.estimated is False

    def test_langchain_openai_puts_reasoning_inside_output_tokens(self):
        """The installed parser, fed a raw Chat Completions body, then our ledger."""
        langchain_openai = pytest.importorskip("langchain_openai")
        llm = langchain_openai.ChatOpenAI(model="gpt-5-nano", api_key=FAKE_KEY)
        body = {
            "id": "chatcmpl-test",
            "model": "gpt-5-nano-2025-08-07",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Paris"},
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 600,
                "total_tokens": 700,
                "completion_tokens_details": {"reasoning_tokens": 512},
            },
        }
        message = llm._create_chat_result(body).generations[0].message
        assert message.usage_metadata["output_tokens"] == 600
        assert message.usage_metadata["output_token_details"]["reasoning"] == 512
        usage = cost.usage_from_response(message)
        assert (usage.prompt_tokens, usage.completion_tokens, usage.reasoning_tokens) == (
            100, 600, 512,
        )

    def test_the_raw_token_usage_block_is_read_the_same_way(self):
        response = AIMessage(
            content="x",
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 600,
                    "completion_tokens_details": {"reasoning_tokens": 512},
                }
            },
        )
        usage = cost.usage_from_response(response)
        assert usage.completion_tokens == 600 and usage.reasoning_tokens == 512

    def test_a_breakdown_larger_than_its_total_is_added_not_dropped(self):
        # Inconsistent, but never allowed to under-report: 512 cannot be inside 10.
        response = AIMessage(
            content="x",
            usage_metadata={
                "input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
                "output_token_details": {"reasoning": 512},
            },
        )
        assert cost.usage_from_response(response).completion_tokens == 522

    def test_usage_arithmetic_carries_the_breakdown(self):
        one = cost.Usage(prompt_tokens=1, completion_tokens=600, calls=1, reasoning_tokens=512)
        assert (one + one).reasoning_tokens == 1024
        assert one.scaled(3).reasoning_tokens == 1536
        ledger = cost.CostLedger("gpt-5-nano")
        ledger.record(one)
        assert "of which 512 hidden reasoning" in ledger.lines()[0]
        assert ledger.summary()["reasoning_tokens"] == 512

    async def test_the_guard_prices_reasoning_as_output(self):
        guard = rp.SpendGuard("gpt-5-nano", max_usd=None, max_calls=None)
        model = guard.wrap(ReasoningModel(), guard.ledger("solver"))
        await model.ainvoke([HumanMessage(content="q")])
        assert guard.usage.completion_tokens == 600
        assert guard.usd() == pytest.approx((100 * 0.05 + 600 * 0.40) / 1_000_000)

    async def test_reasoning_tokens_trip_the_usd_cap(self):
        # Without its 512 reasoning tokens the call is $0.0000402, with them $0.000245:
        # a $0.0001 cap refuses the second call only because reasoning is counted.
        inner = ReasoningModel()
        guard = rp.SpendGuard("gpt-5-nano", max_usd=0.0001, max_calls=None)
        model = guard.wrap(inner, guard.ledger("solver"))
        await model.ainvoke([HumanMessage(content="q")])
        with pytest.raises(rp.BudgetExceeded, match="--max-usd"):
            await model.ainvoke([HumanMessage(content="q")])
        assert inner.calls == 1

    def test_the_report_prints_the_measured_mean_for_calibration(self, demo, prior):
        guard = rp.SpendGuard("gpt-5-nano", max_usd=None, max_calls=None)
        ledger = guard.ledger("solver")
        for _ in range(4):
            ledger.record_response(AIMessage(content="x", usage_metadata=REASONING_USAGE))
        result = rp.BenchResult(dataset=demo, prior=prior, max_steps=8, guard=guard)
        text = "\n".join(rp.cost_lines(result))
        assert "2,048 hidden reasoning tokens" in text
        assert re.search(r"mean of 512 per measured call against the dry run's 512-token", text)
        assert f"estimated from {GPT5_CHECKED_ON} prices" in text

    def test_the_threats_say_a_reasoning_run_has_no_temperature_control(self, demo, prior):
        result = rp.BenchResult(dataset=demo, prior=prior, max_steps=8)
        assert any("TEMPERATURE 0 IS NOT DETERMINISM" in t for t in rp.threats_lines(result))
        result.model, result.reasoning_effort = "gpt-5-nano", "minimal"
        text = "\n".join(rp.threats_lines(result))
        assert "NO TEMPERATURE CONTROL" in text and "effort 'minimal'" in text
        assert "TEMPERATURE 0" not in text
