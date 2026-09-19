from rlr.data.parse import ARTICLE_RE, Block, article_from_page, article_status, blocks_from_full_page, parse_blocks


def P(text: str) -> Block:
    return Block("p", text)


def test_article_numbers_with_dots_and_dashes():
    # regression: tk-rf-rag read "341.1-1" as "341" and dropped "22.1"-like numbers
    for header, number, title in [
        ("Статья 81. Расторжение трудового договора", "81", "Расторжение трудового договора"),
        ("Статья 22.1. Понятие", "22.1", "Понятие"),
        ("Статья 341.1-1. Особенности", "341.1-1", "Особенности"),
        ("Статья 5.27.1. Нарушение требований охраны труда", "5.27.1", "Нарушение требований охраны труда"),
        ("Статья 14.1.1. Незаконные азартные игры", "14.1.1", "Незаконные азартные игры"),
    ]:
        m = ARTICLE_RE.match(header)
        assert m is not None
        assert m.group(1) == number
        assert m.group(2) == title


def test_numbers_are_strings_and_ids_unique():
    blocks = [P("Статья 341.1. A"), P("text a"), P("Статья 341.1-1. B"), P("text b"), P("Статья 341. C"), P("c")]
    arts = parse_blocks(blocks, "tk")
    assert [a.doc_id for a in arts] == ["tk-341.1", "tk-341.1-1", "tk-341"]
    assert [a.text for a in arts] == ["text a", "text b", "c"]


def test_repealed_article_with_body_is_marked():
    # regression: repealed articles have a body with the repealing law reference
    blocks = [
        P("Статья 7. Утратила силу. - Федеральный закон от 30.06.2006 N 90-ФЗ."),
        P("Статья 8. Утратила силу"),
        P("Утратила силу с 1 января 2020 года. - Федеральный закон от 01.01.2019 N 1-ФЗ."),
        P("Статья 9. Нормальная"),
        P("Текст."),
    ]
    arts = parse_blocks(blocks, "tk")
    assert [a.status for a in arts] == ["repealed", "repealed", "active"]
    assert article_status("Название", "") == "empty"


def test_chapters_and_centered_headings():
    blocks = [
        Block("center", "Раздел I. ОБЩИЕ ПОЛОЖЕНИЯ"),
        Block("center", "Глава 1. ОСНОВНЫЕ НАЧАЛА"),
        P("Статья 1. Цели"),
        P("Текст 1."),
        Block("center", "Глава 49.1. ДИСТАНЦИОННЫЕ"),
        Block("center", "§ 2. Параграф"),
        P("Статья 312.1. Общие положения"),
        P("Текст 2."),
    ]
    arts = parse_blocks(blocks, "tk")
    assert (arts[0].chapter, arts[0].chapter_title) == ("1", "ОСНОВНЫЕ НАЧАЛА")
    assert (arts[1].chapter, arts[1].text) == ("49.1", "Текст 2.")


def test_stops_at_signature_and_related_documents():
    # regression: legalacts appends related documents after the signature
    blocks = [
        P("Статья 424. Последняя"),
        P("Текст."),
        Block("right", "Президент"),
        P("Москва, Кремль"),
        Block("stop", ""),
        P("1.6. ТК РФ, иные нормативные правовые акты"),
        P("Статья 5 чего-то постороннего"),
    ]
    arts = parse_blocks(blocks, "tk")
    assert len(arts) == 1
    assert arts[0].text == "Текст."


def test_signature_before_first_article_is_ignored():
    blocks = [Block("right", "Принят"), Block("right", "Государственной Думой"), P("Статья 1. A"), P("x")]
    assert [a.number for a in parse_blocks(blocks, "tk")] == ["1"]


def test_blocks_from_full_page_html():
    html = """<html><body><div class="main-center-block">
      <p class="pCenter">Глава 1. ОБЩЕЕ</p>
      <p class="pBoth">Статья 1. Первая</p>
      <p class="pBoth">Абзац&nbsp;с неразрывным пробелом <a href="#">и ссылкой</a>.</p>
      <p class="pRight">Президент</p>
      <div class="pb-2">related</div><p class="pBoth">Статья 2. Лишняя</p>
    </div></body></html>"""
    arts = parse_blocks(blocks_from_full_page(html), "zozpp")
    assert len(arts) == 1
    assert arts[0].text == "Абзац с неразрывным пробелом и ссылкой ."


def test_repeal_lines_inside_article_are_dropped():
    blocks = [
        P("Статья 11.17. Нарушение правил"),
        P("1. Нарушение правил поведения."),
        P("2. Утратил силу. - Федеральный закон от 26.06.2007 N 118-ФЗ."),
        P("3 - 4. Утратили силу. - Федеральный закон от 31.07.2020 N 262-ФЗ."),
        P("Статьи 11.18 - 11.19. Утратили силу. - Федеральный закон от 01.07.2021 N 283-ФЗ."),
        P("5. Лицо, утратившее силу доверенности, не является..."),
    ]
    (art,) = parse_blocks(blocks, "koap")
    assert art.text == "1. Нарушение правил поведения.\n5. Лицо, утратившее силу доверенности, не является..."


def test_roman_chapter_numbers():
    blocks = [Block("center", "Глава II. ЗАЩИТА ПРАВ ПОТРЕБИТЕЛЕЙ ПРИ ПРОДАЖЕ ТОВАРОВ"), P("Статья 18. Права"), P("x")]
    (art,) = parse_blocks(blocks, "zozpp")
    assert art.chapter == "II"


def test_footnote_only_article_is_empty():
    assert article_status("Товарный знак", '<*> Буква "Т" в окружности.\n<**> Буква "Т" в квадрате.') == "empty"


def test_article_page_stops_at_signature():
    html = """<html><body><h1>Статья 170. Последняя</h1>
    <div class="main-center-block-article-text">
      <p class="pBoth">Текст статьи.</p>
      <p class="pRight">Президент</p><p class="pBoth">Москва, Кремль</p>
    </div></body></html>"""
    art = article_from_page(html, "sk", "/kodeks/SK-RF/razdel-viii/glava-22/statja-170/")
    assert art is not None
    assert (art.doc_id, art.chapter, art.text) == ("sk-170", "22", "Текст статьи.")


def test_merged_repealed_header():
    assert article_status(", статья 150. Утратили силу", "") == "repealed"
    assert article_status("Признание утратившими силу отдельных актов", "Текст.") == "active"
