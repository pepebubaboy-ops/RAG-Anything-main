from __future__ import annotations

from raganything.genealogy.name_normalization import (
    PersonNameParts,
    generate_name_variants,
    normalize_name_text,
    normalize_person_name,
    split_person_name,
)


def test_normalize_name_text_trims_and_collapses_whitespace() -> None:
    assert (
        normalize_name_text("  Иван   Петрович   Сидоров  ") == "Иван Петрович Сидоров"
    )
    assert (
        normalize_person_name("  Иван   Петрович   Сидоров  ")
        == "Иван Петрович Сидоров"
    )


def test_normalize_name_text_normalizes_quotes_and_dashes() -> None:
    assert normalize_name_text(" «Анна–Мария» ") == "Анна-Мария"


def test_surname_first_initials_keep_safe_variants() -> None:
    parts = split_person_name("Сидоров И. П.")

    assert parts == PersonNameParts(
        surname="Сидоров",
        given_name="И.",
        patronymic="П.",
    )
    assert normalize_person_name("Сидоров И. П.") == "Сидоров И. П."

    variants = generate_name_variants("Сидоров И. П.")
    assert variants[0] == "Сидоров И. П."
    assert "И. П. Сидоров" in variants


def test_surname_first_full_name_generates_safe_order_variants() -> None:
    parts = split_person_name("Сидоров Иван Петрович")

    assert parts == PersonNameParts(
        surname="Сидоров",
        given_name="Иван",
        patronymic="Петрович",
    )
    assert normalize_person_name("Сидоров Иван Петрович") == "Иван Петрович Сидоров"

    variants = generate_name_variants("Сидоров Иван Петрович")
    assert variants[0] == "Сидоров Иван Петрович"
    assert "Иван Петрович Сидоров" in variants
    assert "Сидоров Иван Петрович" in variants
    assert "И. П. Сидоров" in variants
    assert "Сидоров И. П." in variants
    assert "Иван Сидоров" in variants


def test_hyphenated_given_name_is_preserved() -> None:
    assert normalize_person_name("Анна-Мария Иванова") == "Анна-Мария Иванова"
    assert "Анна-Мария Иванова" in generate_name_variants("Анна-Мария Иванова")


def test_given_first_initials_are_preserved() -> None:
    parts = split_person_name("А. С. Иванов")

    assert parts == PersonNameParts(
        surname="Иванов",
        given_name="А.",
        patronymic="С.",
    )
    assert normalize_person_name("А. С. Иванов") == "А. С. Иванов"
    assert "Иванов А. С." in generate_name_variants("А. С. Иванов")


def test_test_backed_case_mapping_normalizes_given_name_and_patronymic() -> None:
    parts = split_person_name("Михаила Федоровича")

    assert parts == PersonNameParts(given_name="Михаил", patronymic="Федорович")
    assert normalize_person_name("Михаила Федоровича") == "Михаил Федорович"


def test_split_person_name_keeps_trailing_notes_when_clear() -> None:
    parts = split_person_name("Иван Сидоров (ветка А)")

    assert parts == PersonNameParts(
        surname="Сидоров",
        given_name="Иван",
        notes="ветка А",
    )


def test_unknown_two_token_name_is_not_aggressively_split() -> None:
    parts = split_person_name("Неизвестная Строка")

    assert parts == PersonNameParts()
    assert normalize_person_name("Неизвестная Строка") == "Неизвестная Строка"
    assert generate_name_variants("Неизвестная Строка") == ["Неизвестная Строка"]
