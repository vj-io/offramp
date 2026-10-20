# Ecosystem scan: closed-LLM call sites in popular public Python repos

Offramp's static scan (`offramp scan`, the same code as the hosted /scan page) over the 30 most-starred public Python repos on GitHub that import `openai` or `anthropic`, run on 2026-10-21 with Offramp at `ba2a29b` under Python 3.14.7. The scan reads each repo's .py files and never imports or runs them, makes no model call and needs no key. Raw data: [ecosystem-scan.json](ecosystem-scan.json). Regenerate with `uv run python scripts/ecosystem_scan.py`.

## Totals

- **184 call sites in 24 of 30 repos**: 166 chat or completion calls and 18 embedding calls (105 openai, 39 langchain, 37 anthropic, 3 litellm).
- **17% of the chat calls (28 of 166) have a classification, JSON or tool-call shape**, the shapes Offramp scores deterministically. The call's own `tools` or `response_format` settles 27 of them; the rest are guessed from the prompt text.
- **16% stream** (21 pass `stream=True`, 6 set `stream` at runtime). Offramp flags streaming calls for manual review.
- **Sunset risk: 66% of the chat calls whose model is named in the code (33 of 50)** use a model its provider has retired (12) or scheduled for shutdown (21), per the provider deprecation pages read on 2026-10-21. The other 116 chat calls set the model at runtime, so a static scan cannot name it.

## Call shapes (chat calls)

| Shape | Sites | Share | Settled by the call's parameters |
|---|---:|---:|---:|
| unknown | 119 | 72% | 119 |
| tool call | 20 | 12% | 20 |
| free text | 19 | 11% | 0 |
| JSON | 8 | 5% | 7 |

`unknown`: the prompt is built at runtime and the call passes no `tools` or `response_format`. Shapes not settled by the parameters are guessed from the prompt text (it asks for JSON, or to classify or pick one of a set); a full Offramp run labels each call with a model before it tests anything.

## Models named in the code

| Model | Chat sites | Sunset risk |
|---|---:|---|
| `gpt-4o` | 9 | deprecated, shutdown 2027-01-14 |
| `claude-opus-4-7` | 7 | - |
| `gpt-5.5` | 6 | - |
| `claude-3-5-sonnet-20241022` | 6 | retired, shutdown 2025-10-28 |
| `gpt-4-turbo` | 5 | deprecated, shutdown 2026-10-23 |
| `claude-3-sonnet-20240229` | 3 | retired, shutdown 2025-07-21 |
| `gpt-4o-mini` | 3 | deprecated, shutdown 2027-01-14 |
| `gpt-5.6-sol` | 3 | - |
| `gpt-4-turbo-preview` | 2 | retired, shutdown 2026-03-26 |
| `gpt-5-mini` | 2 | deprecated, shutdown 2026-12-11 |
| `gpt-4` | 1 | deprecated, shutdown 2026-10-23 |
| `gpt-3.5-turbo` | 1 | deprecated, shutdown 2026-10-23 |
| `claude-opus-4-0` | 1 | retired, shutdown 2026-06-15 |
| `test-credential-check` | 1 | - |

## Repos

| Repo | Stars | Commit scanned | .py files | Chat sites | Embedding sites |
|---|---:|---|---:|---:|---:|
| [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | 148,148 | [`e6d83ef`](https://github.com/langchain-ai/langchain/tree/e6d83ef6a6e2482cd55691023c51e45d7584a94f) | 2,637 | 14 | 1 |
| [xtekky/gpt4free](https://github.com/xtekky/gpt4free) | 66,908 | [`1dbac25`](https://github.com/xtekky/gpt4free/tree/1dbac2541335a2312f53002dfa9d14119af53099) | 358 | 0 | 0 |
| [zylon-ai/private-gpt](https://github.com/zylon-ai/private-gpt) | 57,598 | [`ff410fe`](https://github.com/zylon-ai/private-gpt/tree/ff410fed574b8b2eab4e2f4c53d99e7065e5cdef) | 929 | 0 | 0 |
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | 43,594 | [`a24fc97`](https://github.com/langchain-ai/langgraph/tree/a24fc97778fca67cfac4cd3fc63e508ee1d29f59) | 476 | 13 | 0 |
| [AstrBotDevs/AstrBot](https://github.com/AstrBotDevs/AstrBot) | 42,432 | [`f2404c1`](https://github.com/AstrBotDevs/AstrBot/tree/f2404c191716a64daaa422be7f3c567cace50d39) | 747 | 6 | 2 |
| [PDFMathTranslate/PDFMathTranslate](https://github.com/PDFMathTranslate/PDFMathTranslate) | 37,705 | [`17922e7`](https://github.com/PDFMathTranslate/PDFMathTranslate/tree/17922e7610291d3f2f43098b34478059087549fe) | 33 | 5 | 0 |
| [mukul975/Anthropic-Cybersecurity-Skills](https://github.com/mukul975/Anthropic-Cybersecurity-Skills) | 34,759 | [`54a7988`](https://github.com/mukul975/Anthropic-Cybersecurity-Skills/tree/54a798831d2266a3ca61ce68a7acb80b81160d57) | 1,103 | 1 | 0 |
| [openai/openai-agents-python](https://github.com/openai/openai-agents-python) | 30,180 | [`64cd5df`](https://github.com/openai/openai-agents-python/tree/64cd5dfc5a1579870dcbed0ea42b27aa84033fee) | 1,020 | 7 | 0 |
| [chidiwilliams/buzz](https://github.com/chidiwilliams/buzz) | 22,103 | [`6186d09`](https://github.com/chidiwilliams/buzz/tree/6186d09a9f7c2b51c9d5de89592ace2e9cc0092c) | 226 | 3 | 0 |
| [openai/swarm](https://github.com/openai/swarm) | 22,064 | [`6af0b4c`](https://github.com/openai/swarm/tree/6af0b4caf37dca4526dfd98e9fbd8ce36e7eeb22) | 62 | 2 | 5 |
| [langbot-app/LangBot](https://github.com/langbot-app/LangBot) | 18,317 | [`fc62f2e`](https://github.com/langbot-app/LangBot/tree/fc62f2ee137f5936284fcc462bbaec4807ae36b3) | 1,171 | 2 | 0 |
| [AsyncFuncAI/deepwiki-open](https://github.com/AsyncFuncAI/deepwiki-open) | 18,232 | [`d92819a`](https://github.com/AsyncFuncAI/deepwiki-open/tree/d92819a9c9f3b99416e3580ff235fc9d3adf8b89) | 67 | 4 | 2 |
| [TransformerOptimus/SuperAGI](https://github.com/TransformerOptimus/SuperAGI) | 17,711 | [`c3c1982`](https://github.com/TransformerOptimus/SuperAGI/tree/c3c1982e7bd6a11cfed53c5a193ea502f924b1b6) | 461 | 2 | 0 |
| [Usagi-org/ai-goofish-monitor](https://github.com/Usagi-org/ai-goofish-monitor) | 15,045 | [`77f8d75`](https://github.com/Usagi-org/ai-goofish-monitor/tree/77f8d75869acd65f9dba72731b95c7c49ea67bdf) | 115 | 0 | 0 |
| [livekit/agents](https://github.com/livekit/agents) | 14,868 | [`de40bb2`](https://github.com/livekit/agents/tree/de40bb258b58e1aeb1b6350a24e6b6cca79b25a5) | 1,181 | 4 | 0 |
| [The-PR-Agent/pr-agent](https://github.com/The-PR-Agent/pr-agent) | 13,506 | [`729415c`](https://github.com/The-PR-Agent/pr-agent/tree/729415c68b8ef55df8bcce919a419af7f5cebffb) | 493 | 3 | 1 |
| [simonw/llm](https://github.com/simonw/llm) | 12,647 | [`cbeeb8f`](https://github.com/simonw/llm/tree/cbeeb8f2b93b642107e51c735c7eb55734355da1) | 56 | 8 | 1 |
| [bentoml/OpenLLM](https://github.com/bentoml/OpenLLM) | 12,560 | [`9ca8247`](https://github.com/bentoml/OpenLLM/tree/9ca8247b503f5fb2e90e2a5b1b6480e1bd76c58f) | 12 | 1 | 0 |
| [TheR1D/shell_gpt](https://github.com/TheR1D/shell_gpt) | 12,296 | [`a082bd5`](https://github.com/TheR1D/shell_gpt/tree/a082bd5327ce0c4ef5a0284d9060e833be9444a6) | 28 | 0 | 0 |
| [Arize-ai/phoenix](https://github.com/Arize-ai/phoenix) | 11,927 | [`e554217`](https://github.com/Arize-ai/phoenix/tree/e554217fb39fbc05b6732462dc57371ca64b5e6e) | 1,599 | 56 | 0 |
| [BeehiveInnovations/pal-mcp-server](https://github.com/BeehiveInnovations/pal-mcp-server) | 11,793 | [`7afc7c1`](https://github.com/BeehiveInnovations/pal-mcp-server/tree/7afc7c1cc96e23992c8f105f960132c657883bb1) | 249 | 3 | 0 |
| [Doriandarko/claude-engineer](https://github.com/Doriandarko/claude-engineer) | 11,215 | [`0a9e4b3`](https://github.com/Doriandarko/claude-engineer/tree/0a9e4b309bf6b2eda05dcf073cf5094beb08e0c8) | 21 | 7 | 0 |
| [QuentinFuxa/WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) | 11,156 | [`363e4f6`](https://github.com/QuentinFuxa/WhisperLiveKit/tree/363e4f6d029694d9c81ae548beddd9d3c88a3637) | 139 | 0 | 0 |
| [langchain-ai/open-swe](https://github.com/langchain-ai/open-swe) | 11,047 | [`1f8699a`](https://github.com/langchain-ai/open-swe/tree/1f8699a90ed66f01a717af2022d83ffc1975bb87) | 1,251 | 0 | 0 |
| [OthersideAI/self-operating-computer](https://github.com/OthersideAI/self-operating-computer) | 10,290 | [`eb8165a`](https://github.com/OthersideAI/self-operating-computer/tree/eb8165a2efb1b34d66a28ecebf8ca97ae67cbb79) | 18 | 1 | 0 |
| [LearningCircuit/local-deep-research](https://github.com/LearningCircuit/local-deep-research) | 9,372 | [`683a967`](https://github.com/LearningCircuit/local-deep-research/tree/683a967df20fe98ec0ddc99668041e22b95ff78b) | 3,167 | 4 | 0 |
| [zilliztech/deep-searcher](https://github.com/zilliztech/deep-searcher) | 8,350 | [`ddc36db`](https://github.com/zilliztech/deep-searcher/tree/ddc36db317836bd42302b36f03166bec5324708d) | 156 | 11 | 4 |
| [zilliztech/GPTCache](https://github.com/zilliztech/GPTCache) | 8,213 | [`a74ac65`](https://github.com/zilliztech/GPTCache/tree/a74ac654473f7bf4109118e8576945161616e17a) | 229 | 5 | 0 |
| [RayVentura/ShortGPT](https://github.com/RayVentura/ShortGPT) | 8,053 | [`3df4e0f`](https://github.com/RayVentura/ShortGPT/tree/3df4e0f7a422bf7386565d498bf4521a2544c614) | 68 | 1 | 0 |
| [Upsonic/Upsonic](https://github.com/Upsonic/Upsonic) | 7,960 | [`101f031`](https://github.com/Upsonic/Upsonic/tree/101f0313b0ddb96cd4078354879b2ff57005db29) | 968 | 3 | 2 |

## How the repos were picked

GitHub repository search, most stars first: `openai language:python archived:false fork:false stars:>1000` and `anthropic language:python archived:false fork:false stars:>1000`. Each candidate was downloaded as GitHub's HEAD tarball with /api/scan's caps (25 MB download, 5,000 .py files; 1200 s instead of 20 s, since this ran locally) and kept once a file the scanner reads imports `openai` or `anthropic`, until 30 were kept. Skipped on the way:

- NousResearch/hermes-agent (255,551 stars): The repository archive is over 25 MB.
- microsoft/markitdown (191,295 stars): no file the scanner reads imports openai or anthropic
- Significant-Gravitas/AutoGPT (187,675 stars): The repository archive is over 25 MB.
- langgenius/dify (159,053 stars): The repository archive is over 25 MB.
- open-webui/open-webui (156,288 stars): no file the scanner reads imports openai or anthropic
- vllm-project/vllm (94,394 stars): The repository archive is over 25 MB.
- unslothai/unsloth (78,322 stars): The repository archive is over 25 MB.
- headroomlabs-ai/headroom (76,686 stars): The repository archive is over 25 MB.
- calesthio/OpenMontage (70,265 stars): The repository archive is over 25 MB.
- asgeirtj/system_prompts_leaks (69,939 stars): no file the scanner reads imports openai or anthropic
- BerriAI/litellm (61,669 stars): The repository archive is over 25 MB.
- hesreallyhim/awesome-claude-code (56,097 stars): no file the scanner reads imports openai or anthropic
- Aider-AI/aider (49,665 stars): The repository archive is over 25 MB.
- oobabooga/textgen (47,737 stars): no file the scanner reads imports openai or anthropic
- zhayujie/CowAgent (47,481 stars): no file the scanner reads imports openai or anthropic
- anthropics/claude-plugins-official (43,375 stars): no file the scanner reads imports openai or anthropic
- wshobson/agents (41,242 stars): no file the scanner reads imports openai or anthropic
- mouredev/Hello-Python (37,814 stars): no file the scanner reads imports openai or anthropic
- davila7/claude-code-templates (32,783 stars): The repository archive is over 25 MB.
- openai/openai-python (31,883 stars): the OpenAI SDK itself
- mlflow/mlflow (28,469 stars): The repository archive is over 25 MB.
- SYSTRAN/faster-whisper (25,994 stars): no file the scanner reads imports openai or anthropic
- karpathy/minGPT (24,959 stars): no file the scanner reads imports openai or anthropic
- guillaumemeyer/watermarks-remover (24,543 stars): no file the scanner reads imports openai or anthropic
- comet-ml/opik (22,783 stars): The repository archive is over 25 MB.
- modelscope/FunASR (20,768 stars): The repository archive is over 25 MB.
- openai/gpt-oss (20,479 stars): The repository archive is over 25 MB.
- tanweai/pua (19,722 stars): no file the scanner reads imports openai or anthropic
- openai/tiktoken (19,495 stars): no file the scanner reads imports openai or anthropic
- lss233/kirara-ai (19,121 stars): no file the scanner reads imports openai or anthropic
- emcie-co/parlant (18,305 stars): The repository archive is over 25 MB.
- wanshuiyin/Auto-claude-code-research-in-sleep (17,791 stars): no file the scanner reads imports openai or anthropic
- Evolink-AI/awesome-gpt-image-2-API-Prompts (17,325 stars): The repository archive is over 25 MB.
- cft0808/edict (17,006 stars): The repository archive is over 25 MB.
- openai/baselines (16,773 stars): no file the scanner reads imports openai or anthropic
- kyegomez/OpenMythos (14,908 stars): no file the scanner reads imports openai or anthropic
- e2b-dev/E2B (14,475 stars): no file the scanner reads imports openai or anthropic
- 567-labs/instructor (14,039 stars): The repository archive is over 25 MB.
- DLR-RM/stable-baselines3 (13,914 stars): no file the scanner reads imports openai or anthropic
- xszyou/Fay (13,581 stars): The repository archive is over 25 MB.
- jina-ai/clip-as-service (12,832 stars): no file the scanner reads imports openai or anthropic
- FareedKhan-dev/train-llm-from-scratch (12,391 stars): no file the scanner reads imports openai or anthropic
- dataelement/bisheng (12,060 stars): The repository archive is over 25 MB.
- EmbraceAGI/awesome-chatgpt-zh (11,768 stars): The repository archive is over 25 MB.
- lucidrains/DALLE2-pytorch (11,302 stars): no file the scanner reads imports openai or anthropic
- aden-hive/hive (11,113 stars): The repository archive is over 25 MB.
- xonsh/xonsh (9,680 stars): no file the scanner reads imports openai or anthropic
- anthropics/anthropic-sdk-python (3,985 stars): the Anthropic SDK itself

Limits: the scan finds direct calls through the OpenAI, Anthropic, LiteLLM and LangChain clients. It does not count a client passed in from a file that never imports its SDK, a call through an alias of the method (`completion = client.chat.completions.create`), structured-output `.parse()` calls, the pre-1.0 `openai.ChatCompletion` API, LangChain's `init_chat_model` factory, or notebooks, so these numbers are a lower bound. A repo whose calls all go through another library (LlamaIndex, its own SDK) shows 0 sites. No issues or pull requests were opened on these repos.
