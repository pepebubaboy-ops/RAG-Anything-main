# Next prompts

## Prompt 1: replace broken baseline with this zip

Use this prompt after unzipping the archive into a clean working tree.

```text
You are working in pepebubaboy-ops/RAG-Anything-main.

Goal: replace the broken one-line baseline with the fixed local-first baseline from the provided zip.

Do not add new features. Do not change genealogy algorithms. This PR only restores valid Python/TOML/YAML/Markdown, removes generated artifacts, and makes tests pass.

Steps:
1. Create branch `fix/apply-fixed-baseline-zip`.
2. Copy the zip contents into the repository root.
3. Ensure these paths are absent: `chatgpt_pipeline_review_package/`, `chatgpt_pipeline_review_package.zip`, `.github/workflows/linting.yaml`, `outputs/`, `output*/`, generated JSONL/JSON/DOT/HTML artifacts outside `tests/fixtures`.
4. Ensure legacy runtime files are absent: `raganything/base.py`, `raganything/batch.py`, `raganything/config.py`, `raganything/modalprocessors.py`, `raganything/processor.py`, `raganything/prompt.py`, `raganything/query.py`, `raganything/raganything.py`, `raganything/utils.py`.
5. Run:
   - `python -m compileall -q raganything tests`
   - `python -m pytest -q`
   - `python -c "import raganything; print(raganything.__version__)"`
   - `genealogy-rag --help`
   - `genealogy-rag genealogy --help`
6. If ruff is installed, run:
   - `ruff check .`
   - `ruff format --check .`
7. Add PR description with test results and state that no feature logic was changed.
```

## Prompt 2: claim lifecycle and confidence gate

Use after the baseline PR is green.

```text
You are working in pepebubaboy-ops/RAG-Anything-main.

Goal: add a small claim lifecycle layer without changing the overall genealogy pipeline behavior.

Do not add vector search, chat API, name normalization, database persistence, or UI.

Implement:
1. Introduce claim statuses: accepted, pending, rejected, conflict, needs_review.
2. Add a constant such as `MIN_ACCEPTED_CLAIM_CONFIDENCE = 0.55` in the claim validation module.
3. A claim with confidence below the threshold must not enter the accepted graph. It should become pending or rejected_low_confidence.
4. Preserve existing artifacts for accepted claims, and add/keep rejected or pending artifacts for audit.
5. Update tests so claims with `confidence=0.0` never appear in accepted `claims.jsonl`, accepted relationships, or final people/families graph.
6. Add tests for:
   - accepted high-confidence parent_child claim;
   - low-confidence parent_child claim kept out of graph;
   - unsupported evidence quote rejected;
   - invalid JSON repaired or rejected without crashing.
7. Keep file-based JSON/JSONL baseline. No DB adapters.
8. Run `python -m pytest -q`, `python -m compileall -q raganything tests`, and ruff checks if installed.
```

## Prompt 3: basic person-name normalization baseline

Use after claim lifecycle is merged.

```text
You are working in pepebubaboy-ops/RAG-Anything-main.

Goal: add a tested basic person-name normalization module for ordinary genealogy data.

Scope:
- This is ordinary genealogy data.
- Keep normalization generic and conservative.
- Do not add special historical, social-status, or honorific rules.
- Do not change entity resolution yet.
- Do not automatically merge people using normalized names.

Implement `raganything/genealogy/name_normalization.py` with:
1. `normalize_name_text(value: str) -> str`
   - trim whitespace;
   - collapse repeated whitespace;
   - normalize dash/quote variants;
   - remove obvious surrounding punctuation;
   - preserve meaningful hyphenated names.

2. `split_person_name(value: str) -> PersonNameParts`
   - return optional surname, given_name, patronymic, suffix/notes where possible;
   - be conservative: unknown parts should remain `None` rather than guessed aggressively.

3. `normalize_person_name(value: str) -> str`
   - return a display-safe normalized full name;
   - support common forms such as `Фамилия Имя Отчество`, `Имя Отчество Фамилия`, and initials;
   - preserve initials when the full name is not available.

4. `generate_name_variants(value: str) -> list[str]`
   - include the original cleaned form;
   - include safe forms with and without initials;
   - include surname-first and given-name-first variants when all parts are known;
   - remove duplicates while preserving order.

5. Optional small dictionaries are allowed for common modern given-name and patronymic case variants, but keep them conservative and test-backed.

Add `tests/test_name_normalization.py` with examples:
- `"  Иван   Петрович   Сидоров  "` -> `"Иван Петрович Сидоров"`.
- `"Сидоров И. П."` keeps a safe initials variant.
- `"Сидоров Иван Петрович"` variants include `"Иван Петрович Сидоров"` and `"Сидоров Иван Петрович"`.
- `"Анна-Мария Иванова"` preserves the hyphenated name.
- `"А. С. Иванов"` preserves initials.
- `"Михаила Федоровича"` normalizes only if an explicit test-backed mapping is added.

Do not wire this into automatic person merging yet. It may be used only by query parsing/retrieval helpers as a non-authoritative variant generator.

Run:
- `python -m compileall -q raganything tests`
- `python -m pytest -q`
- `ruff check .`
- `ruff format --check .`

PR title:
feat: add basic genealogy person-name normalization helpers
```

## Prompt 4: mention-candidate-canonical boundary

Use after basic name normalization is merged.

```text
You are working in pepebubaboy-ops/RAG-Anything-main.

Goal: introduce a clear data-model boundary between mentions, person candidates, and canonical people.

Do not implement aggressive auto-merge. This PR should make the pipeline safer, not more eager.

Implement:
1. `Mention` represents one source-local occurrence with evidence.
2. `PersonCandidate` represents a possible person assembled from one or more mentions.
3. `CanonicalPerson` represents a reviewed or high-confidence merged person.
4. Claims should refer to mention/candidate data where possible and should not silently merge same-name people.
5. Add tests showing two people with the same full name remain separate when evidence is insufficient.
6. Add tests showing a merge can happen only when strong evidence exists, for example same name plus same spouse/parent/source context.
7. Keep file-based artifacts. No DB persistence yet.
```
