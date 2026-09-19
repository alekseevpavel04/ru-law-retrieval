from rlr.gen.filters import check_article_ref, check_copy, check_length, exact_duplicates, ngrams, run_filters, words

SOURCE = (
    "Работодатель обязан предупредить работника о предстоящем увольнении персонально "
    "и под роспись не менее чем за два месяца"
)


def test_article_references_are_rejected():
    for q in [
        "Что говорит статья 81 про увольнение?",
        "ст. 80 ТК про увольнение",
        "Какие условия предусмотрены в данной статье?",
        "Согласно статье, когда можно уволить?",
        "как применяется ч. 2 про отпуск",
    ]:
        assert check_article_ref(q) == "article_ref", q


def test_normal_questions_pass_article_ref():
    for q in ["Меня сократили, сколько должны заплатить?", "срок предупреждения о сокращении штата"]:
        assert check_article_ref(q) is None


def test_copy_ngram():
    src = ngrams(words(SOURCE), 6)
    assert (
        check_copy("Правда что предупредить работника о предстоящем увольнении персонально нужно?", src) == "copy_ngram"
    )
    assert check_copy("За сколько меня должны предупредить о сокращении?", src) is None


def test_length_and_search_word_count():
    assert check_length("коротко", "legal") == "length"
    assert check_length("срок предупреждения о сокращении", "search") is None
    assert check_length("сокращение", "search") == "length"
    assert check_length(" ".join(["слово"] * 12), "search") == "length"


def test_run_filters_order_and_pass():
    assert run_filters("", "legal", SOURCE) == "empty"
    assert run_filters("За какой срок работодатель уведомляет о сокращении штата?", "legal", SOURCE) is None


def test_exact_duplicates_normalized():
    assert exact_duplicates(["Ёлка, срок?", "елка срок", "другое"]) == [False, True, False]
