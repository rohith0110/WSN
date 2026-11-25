"""Registry of the app's pages so any page can link to another with ``st.page_link``."""

from __future__ import annotations

from streamlit.navigation.page import StreamlitPage


class PageRegistry:
    def __init__(self) -> None:
        self._pages: dict[str, StreamlitPage] = {}

    def register(self, key: str, page: StreamlitPage) -> StreamlitPage:
        self._pages[key] = page
        return page

    def get(self, key: str) -> StreamlitPage:
        if key not in self._pages:
            raise KeyError(f"Page '{key}' is not registered")
        return self._pages[key]


PAGES = PageRegistry()
