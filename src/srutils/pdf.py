"""Work with PDF files."""
from __future__ import annotations

# Standard library
import dataclasses as dc
import os
from typing import Sequence

# Third-party
from pypdf import PdfReader
from pypdf import PdfWriter
from pypdf._page import PageObject
from pypdf.errors import PdfReadError


@dc.dataclass
class MultiPagePDF:
    """A multi-page PDF file composed of multiple individual PDF files."""

    pages: list[PageObject]

    def write(self, path: str) -> None:
        """Write the file to disk."""
        writer = PdfWriter()
        for page in self.pages:
            writer.add_page(page)
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(path, "wb") as fo:
            writer.write(fo)

    @staticmethod
    def _read_pages(paths: Sequence[str]) -> list[PageObject]:
        """Read pages from disk."""
        pages: list[PageObject] = []
        for path_i in paths:
            try:
                file = PdfReader(path_i)
            except (ValueError, TypeError, PdfReadError) as e:
                # Occur sporadically; likely a file system issue
                raise PdfReadError(path_i) from e
            page = file.pages[0]
            pages.append(page)
        return pages

    @classmethod
    def from_files(cls, paths: Sequence[str]) -> MultiPagePDF:
        pages = cls._read_pages(paths)
        return cls(pages)
