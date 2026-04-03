"""mkdocs hook: strip TOC entries whose headings are absent from the page HTML.

mkdocs-jupyter registers all headings from the raw notebook in page.toc before
cell-tag filtering (remove_cell_tags) removes those cells from the rendered HTML.
This hook aligns page.toc with the actual headings present after filtering.
"""

import re


def on_page_content(html, page, config, files):
    heading_ids = set(re.findall(r'<h[1-6][^>]* id="([^"]*)"', html))

    def _filter(items):
        result = []
        for item in items:
            item.children = _filter(item.children)
            if item.id in heading_ids:
                result.append(item)
        return result

    page.toc = _filter(list(page.toc))
    return html
