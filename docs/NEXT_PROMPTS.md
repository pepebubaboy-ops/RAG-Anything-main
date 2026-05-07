# Next prompts

Use these prompts after applying the fixed repository baseline.

## Prompt A: apply this fixed zip to the GitHub repo

```text
You are working in pepebubaboy-ops/RAG-Anything-main.

Goal: replace the currently broken one-line baseline with the fixed repository contents from `genealogy-rag-core-fixed.zip`.

Create branch:
fix/apply-restored-baseline-zip

Steps:
1. Unzip `genealogy-rag-core-fixed.zip` outside the repository.
2. Copy the contents of `genealogy-rag-core-fixed/` into the repository root.
3. Do not copy a nested `genealogy-rag-core-fixed/` directory into the repo.
4. Remove generated artifacts if present:
   - chatgpt_pipeline_review_package/
   - chatgpt_pipeline_review_package.zip
   - outputs/
   - output*/
   - *.pdf, *.zip, generated JSONL/HTML/DOT artifacts outside tests/fixtures
5. Run:
   python -m compileall -q raganything tests
   python -m pytest -q
   ruff check .
   ruff format --check .
6. Create PR with title:
   fix: restore valid local-first genealogy baseline

Do not add new features in this PR.
```

## Prompt B: claim lifecycle after baseline is green

```text
You are working in pepebubaboy-ops/RAG-Anything-main.

Create branch:
feat/claim-lifecycle-status

Goal:
Add an explicit lifecycle for genealogy claims so low-confidence or unsupported model outputs cannot enter the accepted graph.

Do not change OCR, vector search, chat UI, or persistence.

Tasks:
1. Add a claim status enum or constants:
   - accepted
   - pending
   - rejected
   - conflict
   - needs_review
2. Add a minimum accepted confidence threshold, default 0.55.
3. In LLM claim validation:
   - evidence quote must be supported by the source chunk;
   - confidence < threshold must become pending or rejected_low_confidence;
   - malformed claim payloads must become rejected;
   - only accepted claims can be passed to graph build.
4. Preserve audit artifacts:
   - llm_claim_candidates.jsonl
   - llm_rejected_claims.jsonl
   - claims.jsonl
   - evidences.jsonl
5. Add tests:
   - confidence 0.0 does not enter relationships.json;
   - unsupported evidence quote is rejected;
   - valid claim with supported quote and confidence >= threshold is accepted;
   - pending/rejected rows include reason.
6. Update docs/ARCHITECTURE.md and docs/ROADMAP.md minimally.
7. Run:
   python -m compileall -q raganything tests
   python -m pytest -q
   ruff check .
   ruff format --check .
```

## Prompt C: Russian name normalization

```text
You are working in pepebubaboy-ops/RAG-Anything-main.

Create branch:
feat/russian-name-normalization

Goal:
Add a small, deterministic Russian genealogy name normalization layer backed by tests.

Do not implement entity resolution yet.

Tasks:
1. Add `raganything/genealogy/russian_normalization.py`.
2. Implement:
   - normalize_old_russian_orthography(text)
   - strip_genealogy_titles(text)
   - normalize_person_name(text)
   - generate_name_variants(text)
3. Cover at least:
   - final hard sign: Иванъ -> Иван
   - ѣ -> е, і -> и, ѳ -> ф
   - ё/е variants
   - князь, кн., император, царь, царевич titles
   - simple genitive patronymic/name cases used in Romanov-like text
4. Do not auto-merge people based only on normalized names.
5. Add tests with Russian and old-orthography examples.
6. Wire normalization into retrieval/query resolution only as variants, not as canonical merge.
7. Run:
   python -m compileall -q raganything tests
   python -m pytest -q
   ruff check .
   ruff format --check .
```
