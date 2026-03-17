# Data and Code Availability

## Data Availability

In order to ensure maximum reproducibility, transparency, and reusability of the research evaluating video dynamics reasoning in LVLMs, the dataset supporting this study is publicly accessible, governed by the following core components:

**1. Storage Location and Access (Where & How)**
The complete VDR-Bench dataset, comprising 2,700 curated videos and 5,400 high-quality question-answer pairs, is housed within our project's primary Git repository. Specifically, the hierarchical datasets can be found under the `qwen2/` and `enlarge_result/` directories. To guarantee persistent long-term access and stable citation for the scientific community, each official release of the VDR-Bench schema will also be published in established open-science repositories (such as Zenodo or HuggingFace Datasets) with an assigned Digital Object Identifier (DOI).

**2. Data Formats and Technical Standards (Formats & Standards)**
To maximize interoperability across different machine learning frameworks and ease of parsing, the dataset is provided in the widely supported JSON Lines (`.jsonl`) format. The data is systematically separated by task temporal dimension (sequence, reason, prediction, pacing, and change) and video duration (short, medium, long) (e.g., `change_short.jsonl`, `Prediction_long_result.jsonl`). Each object in the schema rigorously encapsulates essential multi-modal evaluation metadata: the main evaluation question, corresponding choices, verifiable ground-truth answers, a structurally linked logical sub-question (true/false) to detect hallucinations, and detailed video captions. 

**3. Tools and Infrastructure (Tools & Infrastructure)**
VDR-Bench is fully supported by an ecosystem of processing and infrastructure tools designed to automate LVLM evaluation. The toolkit, available in the repository, includes comprehensive evaluation scripts located in the `eval/` directory (such as `cal_ac.py`, `llava_eval.py`, `score.py`, and domain-specific tests for Qwen and VideoLLaMA3). These scripts handle automated inference, model prediction logging, and the calculation of our proposed Logical Hallucination Index (LHI). Standardized templates utilized during dataset expansion and evaluation are also provided, enabling direct plug-and-play evaluation.

**4. Data Provenance and Processing (Provenance & Processing)**
The video corpora compiled for VDR-Bench originate from publicly available benchmarks including Video-MME and Google-DeepMind, as well as distinct clips collected from platforms like YouTube (e.g., Friends TV show, SpongeBob SquarePants, and TED talks) to cover diverse, open-domain dynamic scenarios. 
*Data Processing and Curation:* We applied a rigorous three-stage pipeline to transform and extend the data. Initial expert annotations were executed on 500 sets to map temporal logic exactly to visual cues. These were subsequently scaled systematically via GPT-4o using base64 frame sampling and strict template guidance. Crucially, a human-in-the-loop quality review further purged the data of questions exhibiting "information leakage", stereotypical biases, or text-only solvability, strictly enforcing the need for spatio-temporal video comprehension.

**5. Licensing and Versioning (Licensing & Versioning)**
The annotated data and associated VDR-Bench structural schemas are shared under the Creative Commons Attribution 4.0 International (CC BY 4.0) license (or an equivalent open-access license), permitting unrestricted reuse, adaptation, and sharing for both academic and commercial research, provided proper attribution is maintained. The complete dataset curation history, modification trails, and structural pipeline reviews are tracked utilizing Git version control, with detailed documentation maintained in the repository's `README.md`.

---

## Code Availability

As an extension of the primary evaluation dataset, the custom codebase and infrastructure supporting the Video Dynamics Analysis Tree (VDAT) inference framework and hallucination-mitigation methodology are accessible in the local main repository (`E:\desk\TEST\论文\mm-veu\2025.8_aaai`).

The codebase is organized into key functional operations:
*   **Evaluation Pipeline (`eval/`):** Contains algorithms to execute VDAT, assess multi-dimensional reasoning accuracy, and track logical hallucination. Dedicated testing scripts for frontier models (VideoLLaMA-3, Qwen2.5-VL, VILA1.5, LLaVA-OneVision) are integrated seamlessly.
*   **Model Fine-tuning (`finetune/`):** Includes scripts to replicate the Quantized Low-Rank Adaptation (QLoRA) and Supervised Fine-Tuning (SFT) pipeline, ensuring models can be trained to improve semantic consistency as tested in the study.
*   **System Environment (`E:\desk\llm`):** Foundational Large Language Model deployment hooks, hardware inference utilities, and shared dependencies have been developed alongside this local auxiliary workspace.

All code will be provided under an open-source license (e.g., MIT/Apache 2.0). Execution instructions, environment requirements (`requirements.txt`), and step-by-step documentation for reproduction and benchmark deployment are maintained as part of the repository history.