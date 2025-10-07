import random
from typing import Dict, Any, Tuple, List


STYLES = {
    "clean_grid": """
        table { border-collapse: collapse; width: 100%%; font-family: Arial, sans-serif; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    """,
    "minimalist": """
        table { border-collapse: collapse; width: 100%%; font-family: Helvetica, sans-serif; border-spacing: 0; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e0e0; }
        th { font-weight: bold; }
    """,
    "academic_paper": """
        table { border-collapse: collapse; width: 100%%; font-family: 'Times New Roman', serif; }
        th, td { padding: 8px; text-align: left; }
        thead th { border-bottom: 2px solid black; }
        tbody td { border-bottom: 1px solid #ccc; }
        tfoot td { border-top: 2px solid black; }
    """,
    "excel_default": """
        table { border-collapse: collapse; width: 100%%; font-family: Calibri, sans-serif; border: 1px solid #c0c0c0; }
        th, td { border: 1px solid #c0c0c0; padding: 5px; }
        th { background-color: #e7e6e6; font-weight: bold; text-align: center; }
    """,
    "financial_report_blue": """
        table { border-collapse: collapse; width: 100%%; font-family: 'Georgia', serif; }
        th { background-color: #004a99; color: white; padding: 10px; text-align: left; }
        td { padding: 10px; border-bottom: 1px solid #d4d4d4; }
        tr:nth-child(even) { background-color: #f8f8f8; }
    """,
    "dark_mode_tech": """
        body { background-color: #1e1e1e; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Consolas', monospace; color: #d4d4d4; }
        th { background-color: #333333; padding: 10px; border-bottom: 2px solid #555; }
        td { padding: 10px; border-bottom: 1px solid #444; }
    """,
    "vintage_paper": """
        body { background-color: #fdf5e6; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Courier New', monospace; color: #5a4b3c; }
        th, td { padding: 8px; text-align: left; border: 1px solid #c8bba8; }
        th { background-color: #e9e2d5; }
    """,
    "spreadsheet_green": """
        table { border-collapse: collapse; width: 100%%; font-family: Arial, sans-serif; border: 1px solid #a9a9a9; }
        th, td { border: 1px solid #a9a9a9; padding: 4px; }
        th { background-color: #1e6c43; color: white; font-weight: bold; text-align: center; }
    """,
    "bold_header_gray": """
        table { border-collapse: collapse; width: 100%%; font-family: 'Segoe UI', sans-serif; }
        th { background-color: #6c757d; color: white; padding: 12px; font-weight: bold; }
        td { padding: 12px; border-bottom: 1px solid #dee2e6; }
    """,
    # This is the one that uses the %s formatter, so it's the most important to fix.
    "zebra_stripes_random": """
        table { border-collapse: collapse; width: 100%%; font-family: Verdana, sans-serif; }
        th, td { padding: 9px; text-align: left; }
        th { background-color: #555; color: white; }
        tr:nth-child(even) { background-color: %s; }
    """
}

def generate_html_with_style(table_data: Dict[str, Any], base_font_size: str) -> Tuple[str, str]:
    style_name, css_template = random.choice(list(STYLES.items()))
    
    if style_name == "zebra_stripes_random":
        random_color = random.choice(["#f2f2f2", "#e8f4f8", "#f8f8e8", "#f8e8e8", "#e8f8e8"])
        css = css_template % random_color
    else:
        css = css_template.replace('%%', '%')

    
    from html import escape
    headers = "".join(f'<th id="cell-h-{i}">{escape(str(h))}</th>' for i, h in enumerate(table_data.get('columns', [])))
    
    rows_html = []
    for r, row in enumerate(table_data.get('data', [])):
        cells = "".join(f'<td id="cell-r{r}-c{c}">{escape(str(cell))}</td>' for c, cell in enumerate(row))
        rows_html.append(f"<tr>{cells}</tr>")
    body = "".join(rows_html)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-size: {base_font_size}; }}
            {css}
        </style>
    </head>
    <body>
        <table>
            <thead><tr>{headers}</tr></thead>
            <tbody>{body}</tbody>
        </table>
    </body>
    </html>
    """
    return html_content, style_name