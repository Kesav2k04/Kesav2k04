# Kesav Kumar J

Information Technology graduate, first-author vision-language model evaluation researcher, and open-source patch author building scalable, resource-efficient AI systems. My work connects measurable research, distributed GPU infrastructure, and production-grade software engineering.

[GitHub](https://github.com/Kesav2k04) · [LinkedIn](https://www.linkedin.com/in/kesav-kumar-j-5414aa253/) · [ORCID](https://orcid.org/0009-0006-9909-7851)

## Proof at a glance

| Evidence | Result | Public artifact |
| --- | --- | --- |
| POPE hallucination-evaluation audit | Identified a tokenizer prefix-space artifact that deflated scores by up to 7 F1. Correcting the logit read-out raised the LLaVA-1.5 baseline by 6.13 points to 82.21 F1. | [Source](https://github.com/Kesav2k04/ugaa-research) · [9,000 prediction records](https://huggingface.co/datasets/kesav2k04/pope-audit-records) · [PyPI toolkit](https://pypi.org/project/pope-audit/) · [Zenodo archive](https://doi.org/10.5281/zenodo.20738626) |
| Published distributed-GPU framework | Published **VisualPC: A Lightweight and Measurable Framework for Experimental Evaluation of Distributed GPU Execution** at ICCET 2026, Chennai, ISBN 978-81-986418-1-6. | [Source](https://github.com/Kesav2k04/visualpc) · [Live demo](https://visualpc.vercel.app) · [Zenodo archive](https://doi.org/10.5281/zenodo.20866268) |
| International AI challenge | Runner-Up, IBM SkillsBuild "AI Inside the Match" International Challenge, awarded $1,250 among 3,381 registrants and 307 projects, judged by a 19-member IBM panel. | [DecisionLens source](https://github.com/Kesav2k04/decisionlens-june-2026) |
| National hackathon | National Finalist, Snapdragon Multiverse Hackathon, Bengaluru, July 2026, for the Sankat-Mochan offline disaster-response mesh. | [Sankat-Mochan source](https://github.com/Kesav2k04/Sankat-Mochan) · [Base model](https://huggingface.co/kesav2k04/sahayak-e2b) · [GGUF model](https://huggingface.co/kesav2k04/sahayak-e2b-gguf) |

## Research and publications

### Token-Set Choice Confounds POPE: A Systematic Audit of Yes/No Extraction in VLM Hallucination Evaluation

**Under review, 2026** for the NeurIPS 2026 Workshop on Grounded and Faithful Vision-Language Models, VLM4RWD, Sydney.

- Traced the tokenizer prefix-space artifact in POPE yes/no extraction and corrected the logit read-out, lifting the true LLaVA-1.5 baseline by **6.13 F1 points to 82.21 F1**.
- Re-evaluated six published mitigation methods against the corrected baseline. Several reported gains shrink or reverse after the correction.
- Confirmed the confound across **4 VLMs**, **3 tokenizer families**, **9,000 queries**, and **90 GPU-hours**. Attention probing measured **Cohen's d = -0.46**.
- Released `pope-audit` on PyPI, version 0.1.1, plus 9,000 prediction records on Hugging Face for reproducibility. The source is archived on Zenodo with a DOI.

### VisualPC: A Lightweight and Measurable Framework for Experimental Evaluation of Distributed GPU Execution

**Published, March 2026** at ICCET 2026, Chennai. ISBN: 978-81-986418-1-6. Proceedings are to be indexed in IEEE Xplore and SCOPUS.

- Designed a hybrid cloud-edge Platform-as-a-Service where a FastAPI priority scheduler routes CUDA workloads through a Tailscale VPN mesh to cloud GPUs and Raspberry Pi 4B edge nodes.
- Built a Next.js dashboard that streams live worker telemetry through Server-Sent Events.
- Measured queue wait, execution, and network components across tensor payload sizes, with pytest integration tests, Playwright end-to-end tests, JWT authentication, and Dockerized deployment.

## Submitted open-source patches

| Project | Contribution and validation | Pull request |
| --- | --- | --- |
| LangChain, `langchain-openai` | Submitted a structured-output streaming warning fix that extends a maintainer change to synchronous and asynchronous generators. Validated against 517 tests, Ruff, and mypy. `+8 / -2` | [PR #38625, closed](https://github.com/langchain-ai/langchain/pull/38625) |
| Future-AGI, `agentcc-gateway` in Go | Submitted a founder-invited patch for an MCP schema-validation bypass where empty tool arguments skipped required-field checks, with a test-driven regression guard. `+43 / -1` | [PR #1405, open](https://github.com/future-agi/future-agi/pull/1405) |
| EleutherAI, `lm-evaluation-harness` | Submitted a just-in-time lazy loader using native Python `@property` descriptors to replace eager dataset loading in `lm_eval/api/task.py`, reducing multi-task startup memory without new dependencies. `+49 / -7` | [PR #3862, open](https://github.com/EleutherAI/lm-evaluation-harness/pull/3862) |
| Hugging Face, `lighteval` | Submitted a fail-fast static linter for custom evaluation tasks using Python reflection, `inspect`, and `typing`; refactored past the Ruff C901 complexity gate and added an in-memory pytest suite. `+220` | [PR #1268, open](https://github.com/huggingface/lighteval/pull/1268) |

## Selected systems

### [RepoMind](https://github.com/Kesav2k04/RepoMind) | OpenAI Build Week, July 2026

`FastAPI` `React` `GPT-5.6 Terra` `MCP`

- Built a repository preflight engine that orchestrates four parallel GPT-5.6 processes limited to read-only workspace tools, mapping module boundaries, risk zones, and test coverage before autonomous code execution.
- Implemented an application-layer firewall that silently drops model outputs failing strict file-path and line-range validation. Exposed the verified pipeline through a React dashboard, CLI, and local stdio MCP server.

### [Sankat-Mochan](https://github.com/Kesav2k04/Sankat-Mochan) | Snapdragon Multiverse Hackathon finalist, July 2026

`Kotlin` `BLE mesh` `LoRa` `FastAPI` `On-device LLM`

- Built an off-grid SOS network where Android phones relay compressed voice reports phone-to-phone over Bluetooth Low Energy, then bridge them over kilometres through a 433 MHz LoRa gateway on Raspberry Pi and Arduino.
- Fine-tuned and released Sahayak-E2B, a Gemma-based model, with QLoRA. Exported GGUF for fully offline triage on a Snapdragon NPU and open-sourced both Base and GGUF variants on Hugging Face.

### [DecisionLens](https://github.com/Kesav2k04/decisionlens-june-2026) | IBM SkillsBuild runner-up, June 2026

`RAG` `IBM Granite 3.1` `BM25 + embeddings` `React`

- Developed a hallucination-resistant explainer for football VAR calls. A hybrid BM25-plus-vector retriever over **593** rule chunks feeds IBM Granite 3.1 8B under a strict JSON schema and abstains when evidence is weak.
- Validated it on a 50-question adversarial suite. It abstains on out-of-scope queries instead of fabricating and scored full marks on citation, keyword, and decision-type checks.

### YOLOv10 + CBAM Leaf-Disease Detection | Faculty-invited thesis, June 2026

`PyTorch` `TensorRT`

- Extended a faculty-designed YOLOv10-Nano, **96.2% mAP@0.5**, and CBAM-MobileNetV2, **93.6% accuracy**, pipeline to **5,010 images**.
- Executed INT8 TensorRT layer fusion for sub-100 ms Jetson Nano latency under 50 MB.

## Awards and research collaboration

- **National Finalist**, Snapdragon Multiverse Hackathon, Bengaluru, July 2026, for Sankat-Mochan.
- **Runner-Up, 2nd place**, IBM SkillsBuild "AI Inside the Match" International Challenge, June 2026. Awarded $1,250 among 3,381 registrants and 307 projects, judged by a 19-member IBM panel.
- **Research Collaborator**, MSME Government Grant, 2025. Faculty-invited to design machine-learning algorithms for a sanctioned social-media fake-ID-detection project worth INR 13.5L, about US$14,000.

## Industry experience

### DevOps Intern | iStudio | June 2025 to August 2025

`MongoDB` `Docker` `Jenkins` `GitHub Actions`

- Automated GitHub Actions deployments, cutting release cycles by **25%** and removing manual steps.
- Built live MongoDB analytics dashboards for demos and ran Docker and CI/CD reliability checks that held **97.8%** test-phase uptime.

### Java Development Intern | NxtLogic Software Solutions | April 2025 to May 2025

`Java` `Spring Boot` `AWS EC2`

- Integrated analytics modules into Java REST APIs and tuned Spring Boot with EC2 deployment strategies.
- Improved backend response time by **18%** and cut server error rates by **12%** under sustained load testing.

## Education, certifications, and affiliations

**B.Tech, Information Technology** | Sri Krishna College of Technology, India | October 2022 to April 2026 | **CGPA: 8.41 / 10**

- Sri Krishna College of Technology is autonomous, NAAC A, and NBA-accredited.
- Awarded **10/10, O, Outstanding** in the Semester 8 capstone, Project Work Phase II, for VisualPC, with a cumulative top-5% department cohort rank.
- Relevant coursework: Artificial Intelligence, Data Visualization, Computer Networks, Cloud Computing and Containerization, Data Structures and Algorithms, and Operating Systems.
- NVIDIA DLI, Deep Learning and AI on Jetson Nano, 2024.
- IEEE Student Member and IEEE Computer Society Member.

## Technical foundation

| Area | Technologies |
| --- | --- |
| Languages | Python, Java, C++, Go, Kotlin, JavaScript/TypeScript, SQL |
| AI and ML | PyTorch, Hugging Face, QLoRA/PEFT, scikit-learn, RAG, LLM evaluation, NF4 and INT8 quantization, TensorRT |
| Cloud and DevOps | AWS, EC2, S3, VPC, GCP, Docker, GitHub Actions, Jenkins, Linux, Tailscale |
| Web and data | React, Next.js, Spring Boot, FastAPI, REST, PostgreSQL, MySQL, MongoDB |
| Testing and tools | pytest, Playwright, JMeter, Selenium, Postman, Git, Ruff, mypy |

## Connect

[GitHub](https://github.com/Kesav2k04) · [LinkedIn](https://www.linkedin.com/in/kesav-kumar-j-5414aa253/) · [ORCID](https://orcid.org/0009-0006-9909-7851)
