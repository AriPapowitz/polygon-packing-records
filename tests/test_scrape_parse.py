"""Parser tests for the record-table scraper (no network access: a crafted
HTML fragment in the site's markup dialect, including a range row)."""
from polypack.scrape_tables import parse, parse_caption

PAGE = """
<html><body>
<TABLE border=1><tr>
<td align=center><img src="5.gif"><br><font size=+3>5.</font><br>
2.79323+<br>Found by Some Body<br>in May 2026</td>
<td align=center><img src="6.gif"><br><font size=+3>6.-7.</font><br>
3.11+<br>Found by Other Person<br>in June 2026</td>
</tr></TABLE>
</body></html>
"""


def test_parse_rows_and_range_expansion():
    rows = parse(PAGE)
    assert (5, "2.79323+", "Some Body", "May 2026") in rows
    # "6.-7." means one packing covers both n = 6 and n = 7
    assert (6, "3.11+", "Other Person", "June 2026") in rows
    assert (7, "3.11+", "Other Person", "June 2026") in rows
    assert len(rows) == 3


def test_parse_caption_rejects_spacer_cells():
    assert parse_caption("&nbsp;") is None
    assert parse_caption('<img src="x.gif">') is None
