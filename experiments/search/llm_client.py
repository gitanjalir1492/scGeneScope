"""Chat Completions transport for an existing server; no landscape access.

The historical Responses clients remain unchanged. No network activity occurs
on import. Only /v1/models and /v1/chat/completions are required. Diagnostic
endpoints provide optional evidence, never a requirement to launch vLLM our way.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


MODELS = {
    "qwen3-8b-thinking": ("Qwen/Qwen3-8B", "qwen3"),
    "qwen3.8-27b": ("Qwen/Qwen3.8-27B", "qwen38"),
    "gpt-oss-20b": ("openai/gpt-oss-20b", "gptoss"),
    "gpt-oss-120b": ("openai/gpt-oss-120b", "gptoss"),
    "gemma-4-31b-it": ("google/gemma-4-31B-it", "gemma4"),
    "glm-4.7-flash": ("zai-org/GLM-4.7-Flash", "glm45"),
}
DEFAULT_SAMPLING = {"qwen3": (0.6, 0.95, 20), "qwen38": (1.0, 0.95, 20),
                    "gptoss": (1.0, 1.0, -1), "gemma4": (1.0, 0.95, 64), "glm45": (1.0, 0.95, -1)}
HEADROOM = 256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def add_arguments(parser):
    parser.add_argument("--llm-api", choices=["responses", "chat"], default="responses",
                        help="chat connects to OPENAI_BASE_URL using the smoke-gated LLM replay protocol.")
    parser.add_argument("--condition", default=None,
                        help="Optional descriptive local-Chat condition name to separate otherwise colliding directories.")
    parser.add_argument("--thinking", choices=["on", "off"], default=None)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh"], default=None)
    parser.add_argument("--decoding-profile", choices=["controlled", "publisher-nonthinking"], default="controlled")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--model-revision", default=None, help="Optional 40-character revision assertion, checked if the server reports one.")
    parser.add_argument("--context-window", type=int, default=None,
                        help="Optional server context-length assertion, labelled user-supplied unless independently reported.")
    parser.add_argument("--smoke-test", action="store_true", help="Validate one complete trajectory per method before production.")
    parser.add_argument("--analysis-inputs", nargs="+", type=Path, help="Explicit protocol directories; requires --analyze-only.")


def settings(args):
    if args.condition is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.condition):
        raise ValueError("--condition must be 1–64 letters, digits, dots, underscores or hyphens, starting with a letter or digit.")
    profiles = {**MODELS, **{model_id: (model_id, family) for model_id, family in MODELS.values()}}
    if args.llm_model not in profiles:
        raise ValueError(f"Unknown local alias {args.llm_model}; supported: {', '.join(MODELS)}")
    model_id, family = profiles[args.llm_model]
    revision = args.model_revision
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("--model-revision must be a 40-character commit, not a mutable branch.")
    if args.context_window is not None and args.context_window < 1:
        raise ValueError("--context-window must be positive.")
    thinking = args.thinking or "on"
    effort = args.reasoning_effort
    if family == "gptoss":
        if args.thinking is not None or effort not in {None, "low", "medium", "high"}:
            raise ValueError("GPT-OSS uses low/medium/high --reasoning-effort, not --thinking.")
        effort = effort or "medium"
    elif family == "qwen38":
        if thinking == "off" and effort is not None:
            raise ValueError("Omit --reasoning-effort when Qwen thinking is off.")
        if effort not in {None, "low", "medium", "xhigh"}:
            raise ValueError("Qwen3.8 supports low/medium/xhigh, not high.")
        effort = (effort or "xhigh") if thinking == "on" else None
    elif effort is not None:
        raise ValueError("This model uses --thinking on/off, not reasoning effort.")
    defaults = (*DEFAULT_SAMPLING[family], 0.0)
    if args.decoding_profile == "publisher-nonthinking":
        if family != "qwen38" or args.thinking != "off":
            raise ValueError("publisher-nonthinking requires Qwen3.8 and explicit --thinking off.")
        defaults = (0.7, 0.8, 20, 1.5)
    values = tuple(default if value is None else value for value, default in
                   zip((args.temperature, args.top_p, args.top_k, args.presence_penalty), defaults))
    if args.decoding_profile == "publisher-nonthinking" and values != defaults:
        raise ValueError("Do not override the named publisher-nonthinking decoding profile.")
    temperature, top_p, top_k, presence = values
    if not all(math.isfinite(v) for v in values) or not 0 < temperature <= 2 or not 0 < top_p <= 1:
        raise ValueError("Use finite stochastic temperature in (0, 2] and top-p in (0, 1].")
    if (top_k != -1 and top_k < 1) or not -2 <= presence <= 2:
        raise ValueError("top-k must be -1 or positive; presence penalty must be in [-2, 2].")
    cap = args.max_output_tokens if args.max_output_tokens is not None else 16384
    if cap < 1:
        raise ValueError("--max-output-tokens must be positive.")
    return dict(model=args.llm_model, model_id=model_id, family=family, revision=revision,
                context_window=args.context_window, thinking=thinking, reasoning_effort=effort, decoding_profile=args.decoding_profile,
                temperature=temperature, top_p=top_p, top_k=top_k, presence_penalty=presence,
                frequency_penalty=0.0, repetition_penalty=1.0, min_p=0.0, max_tokens=cap,
                timeout_seconds=900, selection_attempts=3, transport_attempts_per_invocation=3)


def request_seed(base_seed, method, replicate, trial, attempt):
    # Stable across processes and reasoning arms; never use Python's salted hash().
    # The runner checks the entire requested schedule for collisions before generation.
    return int(digest(["llm-replay-request-v1", base_seed, method, replicate, trial, attempt])[:8], 16)


def template_kwargs(config):
    if config["family"] == "gptoss":
        return {}
    result = {"enable_thinking": config["thinking"] == "on"}
    if config["family"] == "qwen38":
        result["preserve_thinking"] = False  # Each selection is a fresh single-user request.
    return result


def request_body(prompt, config, seed):
    body = dict(model=config["model"], messages=[dict(role="user", content=prompt)], seed=seed,
                max_tokens=config["max_tokens"], temperature=config["temperature"],
                top_p=config["top_p"], presence_penalty=config["presence_penalty"], frequency_penalty=config["frequency_penalty"],
                stream=False, n=1, top_k=config["top_k"], min_p=config["min_p"], repetition_penalty=config["repetition_penalty"],
                chat_template_kwargs=template_kwargs(config))
    if config["reasoning_effort"] is not None:
        body["reasoning_effort"] = config["reasoning_effort"]
    return body


def normalize_response(raw):
    if not isinstance(raw, dict):
        raise ValueError("Expected a Chat response object.")
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Expected exactly one response choice.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("Missing final-answer message.")
    reasoning = message.get("reasoning")
    older = message.get("reasoning_content")
    if reasoning and older and reasoning != older:
        raise ValueError("Conflicting reasoning fields.")
    text = message.get("content")
    reasoning = reasoning or older or ""
    if not isinstance(reasoning, str) or not isinstance(text, str) or not text.strip():
        raise ValueError("Missing final-answer text or invalid reasoning field.")
    if message.get("tool_calls") or choices[0].get("finish_reason") != "stop":
        raise ValueError(f"Incomplete/non-final response: {choices[0].get('finish_reason')}")
    usage = mapping(raw.get("usage"))
    return dict(text=text.strip(), reasoning=reasoning, finish_reason=choices[0]["finish_reason"],
                usage=usage, final_chars=len(text), reasoning_chars=len(reasoning))


def evidence(value=None, source=None, status=None):
    """Verified means directly observed/reported, not an audit of server internals."""
    return dict(status=status or ("verified" if value is not None else "unavailable"), value=value, source=source)


def mapping(value):
    return value if isinstance(value, dict) else {}


def empty_render():
    return dict(count=None, token_sha256=None, text=None, sampling=None)


def check_sampling(sampling, config, seed):
    expected = {k: config[k] for k in ("temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty",
                                      "repetition_penalty", "min_p", "max_tokens")}
    expected.update(seed=seed, n=1)
    if sampling is not None and any(k in sampling and sampling[k] != value for k, value in expected.items()):
        raise ValueError("Engine changed requested sampling settings, seed or output cap; refusing generation.")
    return sampling is not None and all(k in sampling for k in expected)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        # Keep credentials and protocol identity bound to the configured endpoint.
        raise HTTPError(request.full_url, code, "Redirect refused; set the exact OPENAI_BASE_URL", headers, fp)


class LocalClient:
    def __init__(self, config):
        self.config = config
        self.base = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        parsed = urlparse(self.base)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Use an HTTP(S) OPENAI_BASE_URL without credentials, query parameters or fragments.")
        self.root = self.base[:-3] if self.base.endswith("/v1") else self.base
        self.api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        handlers = [NoRedirect()]
        if parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
            handlers.append(ProxyHandler({}))
        self.opener = build_opener(*handlers)
        self.render_available = None

    def http(self, path, body=None, timeout=None):
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        url = self.base + path[3:] if path.startswith("/v1/") else self.root + path
        request = Request(url, data=data,
                          headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        # urllib performs no hidden SDK retry. The replay journal owns retries.
        with self.opener.open(request, timeout=timeout or self.config["timeout_seconds"]) as response:
            return json.load(response)

    def optional(self, path, body=None):
        try:
            result = self.http(path, body, timeout=5)
            return result if isinstance(result, dict) else None
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError, ValueError):
            # No inference is sent here. Missing diagnostic routes never imply
            # missing Chat support; actual Chat errors are journalled and raised.
            return None

    def complete(self, body):
        return self.http("/v1/chat/completions", body)

    def render(self, prompt, config=None, seed=0):
        config = config or self.config
        # Unlike /tokenize, this route uses exactly the Chat renderer, including
        # GPT-OSS's native Harmony preamble and the effective sampling defaults.
        if self.render_available is False:
            return empty_render()
        rendered = self.optional("/v1/chat/completions/render", request_body(prompt, config, seed))
        if rendered is None:
            self.render_available = False
            return empty_render()
        tokens, sampling = rendered.get("token_ids"), rendered.get("sampling_params")
        sampling = sampling if isinstance(sampling, dict) else None
        check_sampling(sampling, config, seed)  # Contradictory evidence is always fatal.
        valid_tokens = isinstance(tokens, list) and tokens and all(type(t) is int and t >= 0 for t in tokens)
        text = None
        if valid_tokens:
            decoded = self.optional("/detokenize", dict(model=config["model"], tokens=tokens))
            text = decoded.get("prompt") if decoded else None
        self.render_available = bool(valid_tokens or sampling is not None)
        return dict(count=len(tokens) if valid_tokens else None, token_sha256=digest(tokens) if valid_tokens else None,
                    text=text if isinstance(text, str) else None, sampling=sampling)

    def preflight(self):
        models = self.http("/v1/models")["data"]
        if not isinstance(models, list):
            raise ValueError("Malformed /v1/models response.")
        cards = [m for m in models if isinstance(m, dict) and m.get("id") == self.config["model"]]
        if len(cards) != 1:
            raise ValueError("Requested served-model alias is absent from /v1/models.")
        card = cards[0]
        raw = self.optional("/server_info?config_format=json") or {}
        engine = mapping(raw.get("vllm_config"))
        mc = mapping(engine.get("model_config"))
        config = self.config
        reported_model = mc.get("model") or card.get("root")
        reported_model = reported_model if isinstance(reported_model, str) else None
        known_model_ids = {value[0] for value in MODELS.values()}
        if reported_model in known_model_ids and reported_model != config["model_id"]:
            raise ValueError("Server reports a different underlying model from the requested profile.")
        reported_revision = mc.get("revision") or card.get("revision")
        reported_revision = reported_revision if isinstance(reported_revision, str) else None
        immutable_revision = reported_revision if isinstance(reported_revision, str) and re.fullmatch(r"[0-9a-f]{40}", reported_revision) else None
        if config["revision"] and immutable_revision and config["revision"] != immutable_revision:
            raise ValueError("Server revision differs from the user-supplied --model-revision.")
        context_values = [v for v in (mc.get("max_model_len"), card.get("max_model_len")) if type(v) is int and v > 0]
        if len(set(context_values)) > 1:
            raise ValueError("Server context-length reports disagree.")
        context = context_values[0] if context_values else config["context_window"]
        if config["context_window"] and context_values and context != config["context_window"]:
            raise ValueError("Server context length differs from --context-window.")
        if context is not None and context <= config["max_tokens"] + HEADROOM:
            raise ValueError("Output cap leaves no prompt headroom. Explicitly choose a smaller --max-output-tokens before smoke; no server restart is required.")
        version = self.optional("/version") or {}
        tokenizer = self.optional("/tokenizer_info") or {}
        # Do not persist arbitrary server config: model_config may contain hf_token.
        engine_fields = dict(
            model_config=("model", "revision", "tokenizer", "tokenizer_revision", "dtype", "quantization",
                          "max_model_len", "generation_config", "override_generation_config", "seed", "enforce_eager"),
            scheduler_config=("max_num_seqs", "max_num_batched_tokens", "async_scheduling"),
            parallel_config=("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"),
            cache_config=("cache_dtype", "enable_prefix_caching", "gpu_memory_utilization"),
        )
        engine_evidence = {section: {k: values[k] for k in fields if k in values}
                           for section, fields in engine_fields.items() if (values := mapping(engine.get(section)))}
        facts = dict(
            endpoint=evidence(self.base, "OPENAI_BASE_URL", "user_supplied"),
            served_model=evidence(card["id"], "GET /v1/models"),
            model_id=evidence(reported_model, "server model metadata"),
            reported_revision=evidence(reported_revision, "server revision field (may be mutable)"),
            revision=evidence(immutable_revision or config["revision"],
                              "server model metadata" if immutable_revision else "--model-revision" if config["revision"] else None,
                              "verified" if immutable_revision else "user_supplied" if config["revision"] else "unavailable"),
            context_window=evidence(context, "server model metadata" if context_values else "--context-window" if context else None,
                                    "verified" if context_values else "user_supplied" if context else "unavailable"),
            server_version=evidence(version.get("version"), "GET /version"),
            tokenizer_class=evidence(tokenizer.get("tokenizer_class"), "GET /tokenizer_info"),
            chat_template_sha256=evidence(digest(tokenizer["chat_template"]) if tokenizer.get("chat_template") else None, "GET /tokenizer_info"),
            system_start_date=evidence(mapping(raw.get("vllm_env")).get("VLLM_SYSTEM_START_DATE"), "GET /server_info"),
            engine=evidence(engine_evidence or None, "GET /server_info; selected non-credential fields"),
        )
        system = mapping(raw.get("system_env"))
        facts["hardware"] = evidence({k: system[k] for k in ("nvidia_gpu_models", "nvidia_driver_version", "torch_version",
                                      "cuda_compiled_version", "python_version") if k in system} or None, "GET /server_info")
        parallel = mapping(engine.get("parallel_config"))
        sizes = [parallel.get(k) for k in ("tensor_parallel_size", "pipeline_parallel_size")]
        facts["inference_gpu_count"] = evidence(math.prod(sizes) if system.get("nvidia_gpu_models") and all(type(v) is int and v > 0 for v in sizes) else None,
                                                "server tensor_parallel_size * pipeline_parallel_size")
        schema = self.optional("/openapi.json")
        requested = request_body("probe", config, 0)
        properties, schema_source = chat_properties(schema, requested)
        fields = [k for k in requested if k not in {"messages", "model", "stream"}]
        facts["request_fields_declared"] = evidence({k: k in properties for k in fields} if properties is not None else None, schema_source)
        probe = "Return a short JSON object containing an answer."
        rendered = self.render(probe)
        sampling_verified = check_sampling(rendered["sampling"], config, 0)
        facts["effective_sampling"] = evidence(rendered["sampling"], "Chat render endpoint",
                                                "verified" if sampling_verified else "unavailable")
        control = None
        if rendered["token_sha256"] is not None:
            opposite = dict(config)
            if config["family"] == "gptoss":
                opposite["reasoning_effort"] = "low" if config["reasoning_effort"] != "low" else "high"
                if rendered["text"] is not None and f"Reasoning: {config['reasoning_effort']}" not in rendered["text"]:
                    raise ValueError("Rendered Harmony prompt does not contain the requested reasoning effort.")
            else:
                opposite["thinking"] = "off" if config["thinking"] == "on" else "on"
                if config["family"] == "qwen38":
                    opposite["reasoning_effort"] = None if opposite["thinking"] == "off" else "xhigh"
            other = self.render(probe, opposite)
            if other["token_sha256"] is not None:
                if rendered["token_sha256"] == other["token_sha256"]:
                    raise ValueError("Reasoning switch did not change rendered control tokens; refusing generation.")
                control = dict(requested=rendered["token_sha256"], alternative=other["token_sha256"])
        if config["family"] == "qwen38" and config["thinking"] == "on" and control is not None:
            alternate = dict(config, reasoning_effort="low" if config["reasoning_effort"] != "low" else "xhigh")
            effort_probe = self.render(probe, alternate)
            if effort_probe["token_sha256"] is None:
                control = None  # On/off alone does not establish effort application.
            elif effort_probe["token_sha256"] == rendered["token_sha256"]:
                raise ValueError("Qwen reasoning effort did not change rendered control tokens.")
            else:
                control["alternative_effort"] = effort_probe["token_sha256"]
        facts["reasoning_control_rendered"] = evidence(control, "Chat render contrast (not a guarantee of model behaviour)")
        warnings = [f"{key}: {facts[key]['status']}" for key in ("revision", "context_window", "effective_sampling", "reasoning_control_rendered")
                    if facts[key]["status"] != "verified"]
        if reported_revision and immutable_revision is None:
            warnings.append("Server revision is not an immutable commit; the exact checkpoint is unverified.")
        if reported_model not in {config["model"], config["model_id"]}:
            warnings.append("The underlying model identifier is unavailable or a custom server path; the alias alone does not verify checkpoint identity.")
        if properties is None or any(k not in properties for k in fields):
            warnings.append("Some request parameters lack API-schema evidence; smoke checks acceptance and outputs, not hidden application.")
        if not sampling_verified:
            warnings.append("Seed and sampling application inside the server are unverified; exact requests will be logged without claiming deterministic inference.")
        if config["reasoning_effort"] is not None and control is None:
            warnings.append("Requested reasoning effort is not independently verified; response length is not evidence that effort was applied.")
        if config["family"] == "gptoss" and facts["system_start_date"]["value"] is None:
            warnings.append("Harmony's server-added date/preamble is unavailable; it may change across restarts or dates.")
        return dict(endpoint=self.base, model=dict(max_model_len=context), provenance=facts, warnings=warnings,
                    render_available=self.render_available, control_probe=rendered)


def chat_properties(schema, body=None):
    """Read a declared Chat request schema if offered; acceptance is checked by smoke."""
    if not isinstance(schema, dict):
        return None, None
    try:
        request = schema["paths"]["/v1/chat/completions"]["post"]["requestBody"]["content"]["application/json"]["schema"]
        if "$ref" in request:
            parts = request["$ref"].removeprefix("#/").split("/")
            request = schema
            for part in parts:
                request = request[part]
        properties = request["properties"]
        if isinstance(properties, dict) and body is not None:
            if request.get("additionalProperties") is False and set(body) - set(properties):
                raise ValueError(f"Chat schema explicitly rejects requested fields: {sorted(set(body) - set(properties))}")
            for name, value in body.items():
                field = mapping(properties.get(name))
                variants = field.get("anyOf", [field])
                if isinstance(variants, list) and variants and all(isinstance(part, dict) and ("enum" in part or part.get("type") == "null") for part in variants):
                    choices = [v for part in variants for v in part.get("enum", [None])]
                    if value not in choices:
                        raise ValueError(f"Chat schema does not support requested {name}={value!r}.")
        return properties if isinstance(properties, dict) else None, "GET /openapi.json Chat request schema"
    except (KeyError, TypeError):
        return None, None
