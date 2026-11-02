"""LangGraph Ask-the-Portfolio agent: retrieve → generate → validate → repair/fallback."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from src.ask_portfolio import answer_local, load_facts
from src.cross_query import run_cross
from src.paths import ROOT

load_dotenv(ROOT / ".env")

GROQ_MODEL_DEFAULT = "openai/gpt-oss-20b"
GEMINI_MODEL_DEFAULT = "gemini-2.0-flash"
OPENAI_MODEL_DEFAULT = "gpt-4o-mini"
MAX_RETRIES = 2

OFF_TOPIC = (
    "gdp",
    "forecast",
    "next year",
    "ceo",
    "ssn",
    "password",
    "why did",
    "stock price",
    "weather",
    "write python",
    "hack",
)


class AskState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    facts: dict[str, Any]
    evidence: dict[str, Any]
    intent: Literal["qa", "decline", "chitchat"]
    draft: str
    declined: bool
    source: str | None
    engine: str
    retries: int
    validation_error: str
    answer: str


def available_llm() -> tuple[str, str] | None:
    """Return (provider, model) for the first configured free/compatible API."""
    if os.getenv("GROQ_API_KEY", "").strip():
        return "groq", os.getenv("GROQ_MODEL", GROQ_MODEL_DEFAULT).strip() or GROQ_MODEL_DEFAULT
    if os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip():
        return "gemini", os.getenv("GEMINI_MODEL", GEMINI_MODEL_DEFAULT).strip() or GEMINI_MODEL_DEFAULT
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai", os.getenv("OPENAI_MODEL", OPENAI_MODEL_DEFAULT).strip() or OPENAI_MODEL_DEFAULT
    return None


def evidence_for(question: str, facts: dict[str, Any]) -> dict[str, Any]:
    """Send only grouped facts (no account rows). Always include book totals."""
    q = question.lower()
    out: dict[str, Any] = {
        "n_accounts": facts.get("n_accounts"),
        "n_customers": facts.get("n_customers"),
        "n_defaults": facts.get("n_defaults"),
        "portfolio_default_rate": facts.get("portfolio_default_rate"),
        "total_exposure": facts.get("total_exposure"),
        "avg_loan_amount": facts.get("avg_loan_amount"),
        "avg_income": facts.get("avg_income"),
        "as_of": facts.get("as_of"),
    }
    dim_map = {
        "loan": "by_loan_type",
        "product": "by_loan_type",
        "mortgage": "by_loan_type",
        "region": "by_region",
        "south": "by_region",
        "north": "by_region",
        "employment": "by_employment_type",
        "unemployed": "by_employment_type",
        "salaried": "by_employment_type",
        "income": "by_income_band",
        "age": "by_age_band",
        "city": "by_city_tier",
        "tier": "by_city_tier",
        "housing": "by_housing_type",
        "channel": "by_disbursal_channel",
        "education": "by_education_level",
    }
    selected = {v for k, v in dim_map.items() if k in q}
    if "cross" in q or ("employment" in q and "loan" in q):
        selected.add("cross_employment_loan_type")
    if ("region" in q) and (
        any(w in q for w in ("salaried", "unemployed", "employment", "employee", "self-employed", "contract", "business owner"))
        or any(w in q for w in ("loan", "mortgage", "business", "personal", "credit card", "auto"))
    ):
        selected.add("cross_region_employment_loan_type")
        selected.add("cross_employment_loan_type")
    if not selected:
        selected = {
            "by_loan_type",
            "by_region",
            "by_employment_type",
            "by_income_band",
            "by_age_band",
            "cross_employment_loan_type",
        }
    for key in selected:
        if key in facts:
            out[key] = facts[key]
    try:
        cross = run_cross(question)
    except FileNotFoundError:
        cross = None
    if cross:
        out["cross_query"] = cross["table"]
        out["cross_spec"] = cross["spec"]
        out["cross_headline"] = cross["headline"]
        out["cross_source"] = cross["source"]
    return out


def allowed_numbers(evidence: dict[str, Any]) -> dict[str, set[str]]:
    pcts: set[str] = set()
    money: set[str] = set()
    ints: set[str] = set()

    def walk(obj: Any, key: str | None = None) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, k)
            return
        if isinstance(obj, list):
            for item in obj:
                walk(item, key)
            return
        if isinstance(obj, bool) or obj is None:
            return
        if isinstance(obj, int):
            ints.add(str(obj))
            ints.add(f"{obj:,}")
            return
        if isinstance(obj, float):
            if key and "rate" in key:
                pcts.add(f"{obj * 100:.1f}")
                pcts.add(f"{obj * 100:.2f}")
                pcts.add(f"{obj * 100:.0f}")
            if key and ("exposure" in (key or "") or "amount" in (key or "") or "income" in (key or "")):
                money.add(f"{obj:,.0f}")
                money.add(str(int(round(obj))))
            ints.add(str(int(round(obj))))

    walk(evidence)
    return {"pcts": pcts, "money": money, "ints": ints}


def validate_answer_numbers(answer: str, evidence: dict[str, Any]) -> str | None:
    """Return an error string if the draft uses figures that are not in evidence."""
    allowed = allowed_numbers(evidence)
    for raw in re.findall(r"(\d+(?:\.\d+)?)%", answer):
        if raw not in allowed["pcts"] and f"{float(raw):.1f}" not in allowed["pcts"]:
            return f"percentage {raw}% is not in the evidence tables"
    for raw in re.findall(r"\$[\s]*([\d,]+(?:\.\d+)?)", answer):
        token = raw.replace(",", "").split(".")[0]
        compact = f"{int(token):,}" if token.isdigit() else raw
        if token not in allowed["money"] and compact not in allowed["money"] and raw not in allowed["money"]:
            # allow $601m style if 601 matches millions of total_exposure
            if not _money_ok(raw, evidence):
                return f"amount ${raw} is not in the evidence tables"
    return None


def _money_ok(raw: str, evidence: dict[str, Any]) -> bool:
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return False
    exposure = float(evidence.get("total_exposure") or 0)
    if abs(value * 1_000_000 - exposure) / max(exposure, 1) < 0.02:
        return True
    if abs(value - exposure) / max(exposure, 1) < 0.02:
        return True
    return False


def _parse_llm_json(text: str) -> dict[str, Any]:
    blob = text.strip()
    if blob.startswith("```"):
        blob = re.sub(r"^```(?:json)?", "", blob).removesuffix("```").strip()
    data = json.loads(blob)
    if not isinstance(data, dict) or "answer" not in data:
        raise ValueError("LLM JSON missing answer")
    return {
        "declined": bool(data.get("declined", False)),
        "answer": str(data["answer"]).strip(),
        "source": data.get("source") or "facts.json via LLM",
    }


def _llm_messages(state: AskState) -> list[dict[str, str]]:
    history_txt = ""
    for turn in (state.get("history") or [])[-6:]:
        history_txt += f"{turn.get('role', 'user')}: {turn.get('content', '')}\n"
    repair = state.get("validation_error") or ""
    repair_block = (
        f"\nYour last draft was rejected: {repair}. Rewrite using only numbers that appear in EVIDENCE.\n"
        if repair
        else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a credit-risk analyst assistant. Answer ONLY from EVIDENCE JSON. "
                "Never invent rates, counts, or dollar amounts. "
                "If EVIDENCE.cross_query is present, you MUST answer the cross-cut from that table "
                "(and cross_headline). Do not decline a join that the table already computed. "
                "If EVIDENCE cannot answer, set declined=true and say you cannot answer from the "
                "dashboard tables. Return JSON only: "
                '{"declined": bool, "answer": str, "source": str}. '
                "Keep the answer to at most 3 sentences. Cite the grouping you used in source."
            ),
        },
        {
            "role": "user",
            "content": (
                f"EVIDENCE:\n{json.dumps(state['evidence'])}\n"
                f"{repair_block}"
                f"RECENT_CHAT:\n{history_txt or '(none)'}\n"
                f"QUESTION:\n{state['question']}"
            ),
        },
    ]


def _complete(provider: str, model: str, messages: list[dict[str, str]]) -> str:
    if provider == "groq":
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"].strip())
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return resp.choices[0].message.content or "{}"
    if provider == "gemini":
        import urllib.request

        key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        )
        body = json.dumps(
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": messages[0]["content"] + "\n\n" + messages[1]["content"]}],
                    }
                ],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
            }
        ).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
        return payload["candidates"][0]["content"]["parts"][0]["text"]
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return resp.choices[0].message.content or "{}"
    raise RuntimeError(f"unknown provider {provider}")


def node_load(state: AskState) -> dict[str, Any]:
    return {"facts": load_facts(), "retries": int(state.get("retries") or 0), "validation_error": ""}


def node_classify(state: AskState) -> dict[str, Any]:
    q = (state.get("question") or "").strip()
    if not q:
        return {"intent": "decline"}
    low = q.lower()
    if re.search(r"\b(hello|hi|hey)\b", low) and len(low.split()) <= 4:
        return {"intent": "chitchat"}
    if any(w in low for w in OFF_TOPIC):
        return {"intent": "decline"}
    if re.search(r"\bACC\d+\b", q) or "customer_id" in low:
        return {"intent": "decline"}
    return {"intent": "qa"}


def node_chitchat(state: AskState) -> dict[str, Any]:
    return {
        "declined": False,
        "answer": "Ask a portfolio question, for example: which loan type has the highest default rate?",
        "source": None,
        "engine": "rules",
    }


def node_decline(state: AskState) -> dict[str, Any]:
    return {
        "declined": True,
        "answer": (
            "I cannot answer that from the dashboard tables. I only know grouped facts from "
            "customers, loans, bureau, repayment and collections — not forecasts, PII, or account-level rows."
        ),
        "source": None,
        "engine": "rules",
    }


def node_retrieve(state: AskState) -> dict[str, Any]:
    return {"evidence": evidence_for(state["question"], state["facts"])}


def _from_cross_evidence(state: AskState) -> dict[str, Any] | None:
    ev = state.get("evidence") or {}
    headline = ev.get("cross_headline")
    if not headline:
        return None
    return {
        "draft": headline,
        "declined": False,
        "source": ev.get("cross_source"),
        "engine": "local_retriever",
        "answer": headline,
    }


def node_generate(state: AskState) -> dict[str, Any]:
    spec = available_llm()
    if spec is None:
        local = answer_local(state["question"], state["facts"])
        return {
            "draft": local["answer"],
            "declined": local["declined"],
            "source": local.get("source"),
            "engine": "local_retriever",
            "answer": local["answer"],
        }
    provider, model = spec
    try:
        raw = _complete(provider, model, _llm_messages(state))
        parsed = _parse_llm_json(raw)
        if parsed["declined"]:
            cross = _from_cross_evidence(state)
            if cross:
                return cross
        return {
            "draft": parsed["answer"],
            "declined": parsed["declined"],
            "source": f"{parsed['source']} ({provider}:{model})",
            "engine": f"{provider}:{model}",
        }
    except Exception as exc:  # noqa: BLE001
        local = answer_local(state["question"], state["facts"])
        prefix = f"(LLM unavailable: {type(exc).__name__}. Using local facts.) "
        return {
            "draft": local["answer"],
            "declined": local["declined"],
            "source": local.get("source"),
            "engine": "local_retriever",
            "answer": prefix + local["answer"],
        }


def node_validate(state: AskState) -> dict[str, Any]:
    if state.get("engine", "").startswith("local") or state.get("declined"):
        return {
            "answer": state.get("answer") or state.get("draft") or "",
            "validation_error": "",
        }
    err = validate_answer_numbers(state.get("draft") or "", state.get("evidence") or {})
    if err:
        return {"validation_error": err, "retries": int(state.get("retries") or 0) + 1}
    return {
        "validation_error": "",
        "answer": state.get("draft") or "",
    }


def node_fallback(state: AskState) -> dict[str, Any]:
    local = answer_local(state["question"], state["facts"])
    note = state.get("validation_error")
    prefix = f"(Grounding check failed: {note}. Showing book figures instead.) " if note else ""
    return {
        "declined": local["declined"],
        "answer": prefix + local["answer"],
        "source": local.get("source"),
        "engine": "local_retriever",
        "validation_error": "",
    }


def _after_classify(state: AskState) -> str:
    return {"qa": "retrieve", "decline": "decline", "chitchat": "chitchat"}[state["intent"]]


def _after_generate(state: AskState) -> str:
    if (state.get("engine") or "").startswith("local") or state.get("answer"):
        return "end"
    return "validate"


def _after_validate(state: AskState) -> str:
    if not state.get("validation_error"):
        return "end"
    if int(state.get("retries") or 0) < MAX_RETRIES and available_llm() is not None:
        return "generate"
    return "fallback"


def build_graph():
    graph = StateGraph(AskState)
    graph.add_node("load", node_load)
    graph.add_node("classify", node_classify)
    graph.add_node("chitchat", node_chitchat)
    graph.add_node("decline", node_decline)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("generate", node_generate)
    graph.add_node("validate", node_validate)
    graph.add_node("fallback", node_fallback)

    graph.add_edge(START, "load")
    graph.add_edge("load", "classify")
    graph.add_conditional_edges(
        "classify",
        _after_classify,
        {"retrieve": "retrieve", "decline": "decline", "chitchat": "chitchat"},
    )
    graph.add_edge("retrieve", "generate")
    graph.add_conditional_edges("generate", _after_generate, {"validate": "validate", "end": END})
    graph.add_conditional_edges("validate", _after_validate, {"generate": "generate", "fallback": "fallback", "end": END})
    graph.add_edge("chitchat", END)
    graph.add_edge("decline", END)
    graph.add_edge("fallback", END)
    return graph.compile()


@lru_cache(maxsize=1)
def compiled_graph():
    return build_graph()


def run_ask(question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    result = compiled_graph().invoke(
        {"question": question, "history": history or [], "retries": 0},
        {"recursion_limit": 12},
    )
    return {
        "declined": bool(result.get("declined")),
        "answer": result.get("answer") or result.get("draft") or "",
        "source": result.get("source"),
        "engine": result.get("engine"),
    }
