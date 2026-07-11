import asyncio
import logging
import os
import re

import markdown
from weasyprint import HTML, CSS, default_url_fetcher
from weasyprint.text.fonts import FontConfiguration

logger = logging.getLogger(__name__)

# Reuse a single FontConfiguration to avoid repeated fontconfig/Pango C-level allocations
_font_config = FontConfiguration()


# Pattern to strip HTML tags that could be injected
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

# Pattern to match emoji characters (comprehensive range)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport and Map
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended (includes 🟠🟢🟥)
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U00002702-\U000027B0"  # Dingbats
    "\U00002600-\U000026FF"  # Misc symbols (includes ✨)
    "\U000024C2-\U000024FF"  # Enclosed Alphanumerics (includes Ⓜ)
    "\U0000FE00-\U0000FE0F"  # Variation Selectors
    "\U0000200D"             # Zero Width Joiner
    "]+",
    flags=re.UNICODE,
)


def _safe_url_fetcher(url: str, timeout: int = 10, ssl_context=None):
    """
    Custom URL fetcher that blocks all external resources to prevent SSRF.

    Only allows data: URIs for inline content.
    Blocks file://, http://, https://, and any other schemes.
    """
    if url.startswith("data:"):
        # Allow data URIs (inline content)
        return default_url_fetcher(url, timeout, ssl_context)

    # Block all other URLs to prevent SSRF and local file access
    logger.warning(f"Blocked attempt to fetch external resource: {url}")
    # Return empty content instead of raising an error
    return {
        "string": b"",
        "mime_type": "text/plain",
    }


def _strip_emojis(text: str) -> str:
    """Remove emoji characters from text for PDF rendering."""
    return EMOJI_PATTERN.sub("", text)


def _sanitize_html(html_content: str) -> str:
    """
    Remove potentially dangerous HTML elements from content.

    This is a defense-in-depth measure - the url_fetcher also blocks resources.
    """
    # Remove script tags and their content
    html_content = re.sub(
        r"<script[^>]*>.*?</script>",
        "",
        html_content,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove style tags with @import or url()
    html_content = re.sub(
        r"<style[^>]*>.*?</style>",
        "",
        html_content,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove iframe, object, embed, frame tags
    html_content = re.sub(
        r"<(iframe|object|embed|frame|link)[^>]*>",
        "",
        html_content,
        flags=re.IGNORECASE,
    )

    # Remove event handlers (onclick, onerror, etc.)
    html_content = re.sub(
        r"\s+on\w+\s*=\s*[\"'][^\"']*[\"']",
        "",
        html_content,
        flags=re.IGNORECASE,
    )

    # Remove src/href attributes pointing to file:// or javascript:
    html_content = re.sub(
        r"\s+(src|href)\s*=\s*[\"'](file:|javascript:)[^\"']*[\"']",
        "",
        html_content,
        flags=re.IGNORECASE,
    )

    return html_content

DEFAULT_CSS = """
@page {
    size: A4;
    margin: 2cm 2.5cm;

    @bottom-center {
        content: "- " counter(page) " -";
        font-family: "Sarasa Gothic SC", "Noto Sans CJK SC", "PingFang SC",
                     "Microsoft YaHei", sans-serif;
        font-size: 9pt;
        color: #808080;
    }
}

body {
    font-family: "Sarasa Gothic SC", "Noto Sans CJK SC", "PingFang SC",
                 "Microsoft YaHei", sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
    text-align: left;
    word-break: break-word;
    overflow-wrap: break-word;
}

h1 {
    font-size: 20pt;
    color: #1a1a1a;
    border-bottom: 1px solid #c8c8c8;
    padding-bottom: 0.3em;
    margin-top: 0;
    margin-bottom: 0.3em;
}

h2 {
    font-size: 14pt;
    color: #2c3e50;
    margin-top: 1.2em;
    margin-bottom: 0.5em;
}

h3 {
    font-size: 12pt;
    color: #34495e;
    margin-top: 1em;
    margin-bottom: 0.4em;
}

p {
    margin: 0.5em 0;
    text-align: left;
}

ul, ol {
    margin: 0.5em 0;
    padding-left: 2em;
}

li {
    margin: 0.3em 0;
}

a {
    color: #0066cc;
    text-decoration: none;
}

hr {
    border: none;
    border-top: 1px solid #c8c8c8;
    margin: 1em 0;
}

code {
    font-family: "Fira Code", "Source Code Pro", "Consolas", monospace;
    background-color: #f5f5f5;
    padding: 0.1em 0.3em;
    border-radius: 3px;
    font-size: 0.9em;
}

pre {
    background-color: #f5f5f5;
    padding: 1em;
    border-radius: 5px;
    overflow-x: auto;
    font-size: 0.9em;
}

blockquote {
    border-left: 4px solid #ddd;
    margin: 1em 0;
    padding-left: 1em;
    color: #666;
}

strong {
    font-weight: bold;
}

em {
    font-style: italic;
}
"""


async def generate_pdf(markdown_content: str, output_path: str) -> str:
    """
    Generate a PDF from Markdown content.

    Args:
        markdown_content: The Markdown content to convert
        output_path: Path to save the PDF file

    Returns:
        Path to the generated PDF file
    """
    logger.info("Generating PDF from Markdown...")

    # Strip emojis before PDF rendering (fonts don't support them)
    markdown_content = _strip_emojis(markdown_content)

    # Convert Markdown to HTML
    html_content = markdown.markdown(
        markdown_content,
        extensions=["tables", "fenced_code", "codehilite", "toc"],
    )

    # Sanitize HTML to remove potentially dangerous elements
    html_content = _sanitize_html(html_content)

    # Wrap in HTML document structure
    full_html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transcript</title>
</head>
<body>
{html_content}
</body>
</html>
"""

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Generate PDF in thread pool (WeasyPrint is synchronous)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: _generate_pdf_sync(full_html, output_path)
    )

    logger.info(f"PDF generated: {output_path}")
    return output_path


def _generate_pdf_sync(html_content: str, output_path: str) -> None:
    """Generate PDF synchronously (for thread pool execution)."""
    # Use custom url_fetcher to block all external resources (SSRF prevention)
    html = HTML(string=html_content, url_fetcher=_safe_url_fetcher)
    css = CSS(string=DEFAULT_CSS, font_config=_font_config)
    html.write_pdf(output_path, stylesheets=[css], font_config=_font_config)
