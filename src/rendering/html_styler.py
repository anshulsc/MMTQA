import random
from typing import Dict, Any, Tuple, List


STYLES = {
    # ==================== BASIC DOCUMENT STYLES ====================
    "standard_word_doc": """
        body { background-color: #ffffff; margin: 40px; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Calibri', Arial, sans-serif; 
                font-size: 11pt; }
        th, td { border: 1px solid #000000; padding: 6px 8px; text-align: left; }
        th { background-color: #d9d9d9; font-weight: bold; }
    """,
    
    "google_docs_default": """
        body { background-color: #f9fbfd; padding: 30px; }
        table { border-collapse: collapse; width: 100%%; font-family: Arial, sans-serif; 
                background: white; border: 1px solid #c7c7c7; }
        th, td { border: 1px solid #c7c7c7; padding: 8px 10px; text-align: left; }
        th { background-color: #f3f3f3; font-weight: 500; }
    """,
    
    "plain_text_editor": """
        body { background-color: #ffffff; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Courier New', monospace; 
                font-size: 10pt; }
        th, td { border: 1px solid #808080; padding: 4px 6px; text-align: left; }
        th { font-weight: bold; }
    """,
    
    "notebook_paper": """
        body { background: repeating-linear-gradient(#fefefe, #fefefe 31px, #93b7d9 31px, #93b7d9 32px); 
               padding: 40px 60px; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Trebuchet MS', sans-serif; 
                background: transparent; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #d0d0d0; }
        th { font-weight: bold; border-bottom: 2px solid #666; }
    """,
    
    "old_book_scan": """
        body { background-color: #f5f1e8; padding: 50px; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Garamond', 'Georgia', serif; 
                font-size: 10pt; color: #2a2826; }
        th, td { padding: 6px 10px; text-align: left; border: none; }
        th { border-bottom: 2px solid #6b5d4f; font-weight: 600; padding-bottom: 8px; }
        td { border-bottom: 1px solid #d4c9b8; }
    """,
    
    # ==================== SPREADSHEET STYLES ====================
    "excel_2019": """
        table { border-collapse: collapse; width: 100%%; font-family: Calibri, Arial, sans-serif; 
                border: 1px solid #9bc2e6; font-size: 11pt; }
        th, td { border: 1px solid #9bc2e6; padding: 4px 8px; }
        th { background-color: #4472c4; color: white; font-weight: 600; text-align: center; }
        td { background-color: #ffffff; }
    """,
    
    "google_sheets_default": """
        table { border-collapse: collapse; width: 100%%; font-family: Arial, sans-serif; 
                background: #ffffff; }
        th, td { border: 1px solid #d3d3d3; padding: 5px 10px; }
        th { background-color: #f3f3f3; color: #000000; font-weight: 500; text-align: left; }
    """,
    
    "numbers_mac": """
        body { background-color: #fafafa; }
        table { border-collapse: separate; border-spacing: 0; width: 100%%; 
                font-family: -apple-system, 'Helvetica Neue', sans-serif; 
                border: 1px solid #d1d1d6; border-radius: 8px; overflow: hidden; }
        th { background-color: #ffffff; color: #1d1d1f; padding: 10px 12px; 
             border-bottom: 2px solid #d1d1d6; font-weight: 600; text-align: left; }
        td { padding: 8px 12px; border-bottom: 1px solid #e5e5ea; background: #ffffff; }
    """,
    
    "libreoffice_calc": """
        table { border-collapse: collapse; width: 100%%; font-family: 'Liberation Sans', Arial, sans-serif; 
                border: 1px solid #808080; }
        th, td { border: 1px solid #808080; padding: 3px 6px; }
        th { background-color: #cccccc; font-weight: bold; text-align: center; }
    """,
    
    # ==================== ACADEMIC PAPERS ====================
    "latex_booktabs": """
        body { background-color: #ffffff; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Computer Modern', 'Latin Modern', serif; 
                font-size: 10pt; }
        th { padding: 8px 10px; border-top: 1.5pt solid #000000; border-bottom: 1pt solid #000000; 
             font-weight: 500; text-align: left; }
        td { padding: 6px 10px; text-align: left; }
        tbody tr:last-child td { border-bottom: 1.5pt solid #000000; }
    """,
    
    "apa_style_table": """
        body { background-color: #ffffff; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Times New Roman', Georgia, serif; 
                font-size: 12pt; }
        th { padding: 8px 6px; border-top: 2px solid #000000; border-bottom: 2px solid #000000; 
             font-weight: 400; font-style: italic; text-align: left; }
        td { padding: 6px; text-align: left; border: none; }
        tbody tr:last-child td { border-bottom: 2px solid #000000; }
    """,
    
    "chicago_manual": """
        body { background-color: #ffffff; }
        table { border-collapse: collapse; width: 100%%; font-family: Georgia, 'Times New Roman', serif; 
                font-size: 11pt; }
        th { padding: 10px 8px; border-bottom: 1px solid #000000; font-weight: 600; text-align: left; }
        td { padding: 8px; text-align: left; border-bottom: 1px solid #cccccc; }
    """,
    
    "ieee_paper": """
        table { border-collapse: collapse; width: 100%%; font-family: 'Times New Roman', serif; 
                font-size: 9pt; }
        th { padding: 6px 4px; border-top: 1.5pt solid #000000; border-bottom: 1pt solid #000000; 
             font-weight: bold; text-align: center; font-size: 8pt; text-transform: uppercase; }
        td { padding: 5px 4px; text-align: center; }
        tbody tr:last-child td { border-bottom: 1.5pt solid #000000; }
    """,
    
    "springer_journal": """
        body { background-color: #ffffff; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Minion Pro', Georgia, serif; 
                font-size: 9pt; }
        th { padding: 8px; border-bottom: 1.5px solid #333333; font-weight: 600; text-align: left; 
             color: #333333; }
        td { padding: 7px 8px; border-bottom: 0.5px solid #cccccc; }
        tr:nth-child(even) { background-color: #f9f9f9; }
    """,
    
    # ==================== FINANCIAL DOCUMENTS ====================
    "annual_report": """
        body { background-color: #fafafa; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Garamond', Georgia, serif; 
                border-top: 3px solid #1e3a5f; border-bottom: 3px solid #1e3a5f; }
        th { padding: 12px 10px; border-bottom: 2px solid #1e3a5f; font-weight: 600; 
             text-align: right; color: #1e3a5f; }
        td { padding: 10px; border-bottom: 1px solid #d0d0d0; text-align: right; }
        td:first-child, th:first-child { text-align: left; font-weight: 600; }
    """,
    
    "bank_statement": """
        body { background-color: #ffffff; }
        table { border-collapse: collapse; width: 100%%; font-family: Arial, sans-serif; 
                font-size: 9pt; border: 1px solid #003366; }
        th { background-color: #003366; color: white; padding: 8px; font-weight: 600; 
             text-align: left; font-size: 8pt; }
        td { padding: 6px 8px; border-bottom: 1px solid #cccccc; }
        tr:nth-child(even) { background-color: #f5f7fa; }
    """,
    
    "invoice_standard": """
        body { background-color: #ffffff; }
        table { border-collapse: collapse; width: 100%%; font-family: Arial, Helvetica, sans-serif; 
                font-size: 10pt; }
        th { background-color: #2c3e50; color: white; padding: 10px; font-weight: 600; 
             text-align: left; }
        td { padding: 8px 10px; border-bottom: 1px solid #dddddd; }
        tfoot td { background-color: #ecf0f1; font-weight: bold; border-top: 2px solid #2c3e50; }
    """,
    
    "tax_form": """
        body { background-color: #ffffff; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Courier New', monospace; 
                font-size: 10pt; border: 2px solid #000000; }
        th, td { border: 1px solid #000000; padding: 4px 6px; text-align: left; }
        th { background-color: #e0e0e0; font-weight: bold; }
    """,
    
    # ==================== NEWSPAPER & MAGAZINE ====================
    "broadsheet_newspaper": """
        body { background-color: #f8f8f6; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Georgia', 'Times New Roman', serif; 
                font-size: 9pt; border-top: 2px solid #000000; }
        th { padding: 8px 6px; border-bottom: 1px solid #000000; font-weight: 700; text-align: left; 
             font-size: 9pt; text-transform: uppercase; }
        td { padding: 6px; border-bottom: 1px solid #d5d5d5; }
    """,
    
    "tabloid_layout": """
        body { background-color: #ffffff; }
        table { border-collapse: collapse; width: 100%%; font-family: Arial, Helvetica, sans-serif; 
                font-size: 11pt; border: 3px solid #cc0000; }
        th { background-color: #cc0000; color: white; padding: 10px; font-weight: bold; 
             text-align: center; font-size: 13pt; }
        td { padding: 8px; border: 1px solid #dddddd; }
        tr:nth-child(even) { background-color: #fff5f5; }
    """,
    
    "magazine_feature": """
        body { background-color: #fafafa; padding: 40px; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Helvetica', Arial, sans-serif; 
                font-size: 10pt; }
        th { padding: 14px 10px; border-bottom: 3px solid #333333; font-weight: 300; 
             text-align: left; font-size: 14pt; letter-spacing: 1px; }
        td { padding: 10px; border-bottom: 1px solid #e0e0e0; }
    """,
    
    # ==================== TECHNICAL DOCUMENTATION ====================
    "api_documentation": """
        body { background-color: #f6f8fa; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Roboto Mono', 'Courier New', monospace; 
                font-size: 13px; background: white; border: 1px solid #d1d5da; border-radius: 6px; }
        th { background-color: #f6f8fa; padding: 10px 12px; border-bottom: 1px solid #d1d5da; 
             font-weight: 600; text-align: left; color: #24292e; }
        td { padding: 10px 12px; border-bottom: 1px solid #e1e4e8; }
    """,
    
    "markdown_table": """
        body { background-color: #ffffff; }
        table { border-collapse: collapse; width: 100%%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
                font-size: 14px; }
        th { padding: 10px 13px; border-bottom: 2px solid #d0d7de; font-weight: 600; 
             text-align: left; }
        td { padding: 10px 13px; border-bottom: 1px solid #d0d7de; }
    """,
    
    "readme_github": """
        body { background-color: #0d1117; padding: 20px; }
        table { border-collapse: collapse; width: 100%%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
                background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; }
        th { padding: 10px 13px; border-bottom: 1px solid #21262d; border-right: 1px solid #21262d; 
             font-weight: 600; text-align: left; background: #161b22; }
        td { padding: 10px 13px; border-bottom: 1px solid #21262d; border-right: 1px solid #21262d; }
    """,
    
    "confluence_wiki": """
        body { background-color: #f4f5f7; }
        table { border-collapse: collapse; width: 100%%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
                background: white; border: 1px solid #c1c7d0; }
        th { background-color: #f4f5f7; padding: 10px; border: 1px solid #c1c7d0; 
             font-weight: 600; text-align: left; }
        td { padding: 8px 10px; border: 1px solid #c1c7d0; }
    """,
    
    # ==================== GOVERNMENT & LEGAL ====================
    "government_form": """
        body { background-color: #ffffff; }
        table { border-collapse: collapse; width: 100%%; font-family: Arial, sans-serif; 
                font-size: 10pt; border: 2px solid #000000; }
        th { background-color: #e8e8e8; padding: 8px; border: 1px solid #000000; 
             font-weight: bold; text-align: center; text-transform: uppercase; }
        td { padding: 6px 8px; border: 1px solid #000000; }
    """,
    
    "court_document": """
        body { background-color: #ffffff; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Century Schoolbook', Georgia, serif; 
                font-size: 11pt; line-height: 1.8; }
        th { padding: 10px; border-top: 1.5pt solid #000000; border-bottom: 1pt solid #000000; 
             font-weight: bold; text-align: left; }
        td { padding: 8px 10px; border-bottom: 0.5pt solid #808080; }
    """,
    
    "legislative_text": """
        body { background-color: #fffff8; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Palatino Linotype', 'Book Antiqua', serif; 
                font-size: 11pt; }
        th { padding: 10px 8px; border-bottom: 2px solid #333333; font-weight: 600; 
             text-align: left; font-variant: small-caps; }
        td { padding: 8px; border-bottom: 1px solid #cccccc; }
    """,
    
    # ==================== MEDICAL & SCIENTIFIC ====================
    "lab_report": """
        body { background-color: #ffffff; }
        table { border-collapse: collapse; width: 100%%; font-family: Arial, sans-serif; 
                font-size: 10pt; border: 1px solid #4a90a4; }
        th { background-color: #d6eaf8; padding: 8px; border: 1px solid #4a90a4; 
             font-weight: 600; text-align: center; }
        td { padding: 6px 8px; border: 1px solid #aed6f1; text-align: center; }
    """,
    
    "clinical_trial": """
        body { background-color: #fafafa; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Calibri', Arial, sans-serif; 
                font-size: 10pt; }
        th { background-color: #1f4e78; color: white; padding: 10px; font-weight: 600; 
             text-align: left; }
        td { padding: 8px 10px; border-bottom: 1px solid #d0d0d0; }
        tr:nth-child(even) { background-color: #f5f5f5; }
    """,
    
    "pathology_report": """
        table { border-collapse: collapse; width: 100%%; font-family: 'Courier New', monospace; 
                font-size: 9pt; border: 2px solid #2c5f2d; }
        th { background-color: #e8f5e9; padding: 8px; border: 1px solid #2c5f2d; 
             font-weight: bold; text-align: left; }
        td { padding: 6px 8px; border: 1px solid #81c784; }
    """,
    
    # ==================== BUSINESS & CORPORATE ====================
    "powerpoint_table": """
        body { background-color: #f0f0f0; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Calibri', Arial, sans-serif; 
                font-size: 18px; background: white; }
        th { background-color: #4472c4; color: white; padding: 16px; font-weight: 600; 
             text-align: center; font-size: 20px; }
        td { padding: 14px 16px; border-bottom: 2px solid #d0d0d0; text-align: center; }
    """,
    
    "executive_summary": """
        body { background-color: #ffffff; }
        table { border-collapse: collapse; width: 100%%; font-family: Georgia, serif; 
                font-size: 11pt; }
        th { padding: 12px 10px; border-bottom: 3px solid #2e4057; font-weight: 600; 
             text-align: left; color: #2e4057; }
        td { padding: 10px; border-bottom: 1px solid #cccccc; }
    """,
    
    "quarterly_earnings": """
        table { border-collapse: collapse; width: 100%%; font-family: Arial, sans-serif; 
                font-size: 10pt; border-top: 2px solid #003d5c; border-bottom: 2px solid #003d5c; }
        th { padding: 10px; border-bottom: 1px solid #003d5c; font-weight: bold; 
             text-align: right; background-color: #f0f4f7; }
        td { padding: 8px 10px; border-bottom: 1px solid #d5d5d5; text-align: right; }
        td:first-child { text-align: left; font-weight: 600; }
    """,
    
    # ==================== EDUCATION ====================
    "gradebook": """
        body { background-color: #fafafa; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Segoe UI', Arial, sans-serif; 
                font-size: 11pt; border: 2px solid #5b9bd5; }
        th { background-color: #5b9bd5; color: white; padding: 10px; font-weight: 600; 
             text-align: center; }
        td { padding: 8px; border: 1px solid #9bc2e6; text-align: center; }
        tr:nth-child(even) { background-color: #deebf7; }
    """,
    
    "syllabus_table": """
        body { background-color: #ffffff; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Times New Roman', serif; 
                font-size: 11pt; }
        th { padding: 10px 8px; border-bottom: 2px solid #000000; font-weight: bold; 
             text-align: left; }
        td { padding: 8px; border-bottom: 1px solid #cccccc; }
    """,
    
    "assignment_rubric": """
        table { border-collapse: collapse; width: 100%%; font-family: Arial, sans-serif; 
                font-size: 10pt; border: 2px solid #70ad47; }
        th { background-color: #70ad47; color: white; padding: 10px; font-weight: 600; 
             text-align: center; border: 1px solid #548235; }
        td { padding: 8px; border: 1px solid #a9d08e; }
    """,
    
    # ==================== VINTAGE & RETRO ====================
    "typewriter_document": """
        body { background-color: #f4f0e8; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Courier Prime', 'Courier New', monospace; 
                font-size: 11pt; color: #2b2b2b; }
        th, td { padding: 6px 10px; text-align: left; border: none; }
        th { border-bottom: 1px solid #2b2b2b; font-weight: bold; }
        td { border-bottom: 1px dotted #8b8b8b; }
    """,
    
    "carbon_copy": """
        body { background-color: #e8e8f0; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Courier New', monospace; 
                font-size: 10pt; color: #1a1a4d; }
        th, td { padding: 6px 8px; text-align: left; }
        th { border-bottom: 1px solid #1a1a4d; font-weight: bold; }
        td { border-bottom: 1px dotted #6666aa; }
    """,
    
    "ledger_paper": """
        body { background-color: #f8f5f0; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Courier New', monospace; 
                font-size: 10pt; color: #2c3e1f; }
        th, td { padding: 6px 10px; border-right: 1px solid #9db389; }
        th { background-color: #e8f0dc; font-weight: bold; text-align: left; 
             border-bottom: 2px solid #5a6f4a; }
        td { border-bottom: 1px solid #d4dcc8; text-align: right; }
        td:first-child { text-align: left; }
    """,
    
    # ==================== MODERN WEB ====================
    "bootstrap_table": """
        body { background-color: #f8f9fa; }
        table { border-collapse: collapse; width: 100%%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
                font-size: 14px; background: white; border: 1px solid #dee2e6; }
        th { background-color: #e9ecef; padding: 12px; border-bottom: 2px solid #dee2e6; 
             font-weight: 600; text-align: left; color: #495057; }
        td { padding: 12px; border-bottom: 1px solid #dee2e6; }
    """,
    
    "tailwind_default": """
        body { background-color: #f9fafb; }
        table { border-collapse: collapse; width: 100%%; font-family: ui-sans-serif, system-ui, sans-serif; 
                font-size: 14px; }
        th { background-color: #f3f4f6; padding: 12px 16px; border-bottom: 1px solid #e5e7eb; 
             font-weight: 600; text-align: left; color: #111827; }
        td { padding: 12px 16px; border-bottom: 1px solid #e5e7eb; color: #374151; }
    """,
    
    "material_design": """
        body { background-color: #fafafa; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Roboto', sans-serif; 
                font-size: 14px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }
        th { background-color: #ffffff; padding: 16px; border-bottom: 1px solid #e0e0e0; 
             font-weight: 500; text-align: left; color: rgba(0,0,0,0.87); }
        td { padding: 14px 16px; border-bottom: 1px solid #e0e0e0; color: rgba(0,0,0,0.87); }
    """,
    
    # ==================== SCANNED DOCUMENTS ====================
    "fax_transmission": """
        body { background-color: #f0f0f0; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Courier New', monospace; 
                font-size: 9pt; color: #1a1a1a; }
        th, td { border: 1px solid #808080; padding: 4px 6px; text-align: left; }
        th { background-color: #d9d9d9; font-weight: bold; }
    """,
    
    "photocopy_document": """
        body { background-color: #f5f5f5; }
        table { border-collapse: collapse; width: 100%%; font-family: Arial, sans-serif; 
                font-size: 10pt; color: #2b2b2b; }
        th, td { border: 1px solid #4d4d4d; padding: 6px 8px; text-align: left; }
        th { background-color: #e0e0e0; font-weight: bold; }
    """,
    
    "scanner_default": """
        body { background-color: #fafafa; }
        table { border-collapse: collapse; width: 100%%; font-family: Arial, sans-serif; 
                font-size: 10pt; }
        th, td { border: 1px solid #999999; padding: 6px 8px; text-align: left; }
        th { background-color: #e8e8e8; font-weight: 600; }
    """,
    
    "watercolor_dream": """
        body { background: linear-gradient(180deg, #ffeaa7 0%%, #fab1a0 50%%, #a29bfe 100%%); padding: 25px; }
        table { border-collapse: separate; border-spacing: 0; width: 100%%; font-family: 'Quicksand', sans-serif; 
                background: rgba(255, 255, 255, 0.95); border-radius: 20px; overflow: hidden; 
                box-shadow: 0 15px 35px rgba(0,0,0,0.2); }
        th { background: linear-gradient(135deg, #ff9a76 0%%, #fd79a8 100%%); color: white; padding: 18px; 
             font-weight: 600; text-align: left; font-size: 16px; }
        td { padding: 15px 18px; border-bottom: 2px solid #fdcb6e; font-size: 14px; }
    """,
    
    "vintage_poster": """
        body { background: #f4e7d7; padding: 30px; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Rockwell', serif; 
                border: 8px solid #8b4513; background: #fff8dc; }
        th { background: #8b4513; color: #f4e7d7; padding: 15px; font-weight: bold; 
             text-align: center; font-size: 18px; text-transform: uppercase; letter-spacing: 3px; 
             border: 3px solid #654321; }
        td { padding: 12px; border: 2px solid #daa520; font-size: 14px; color: #5d4037; }
        tr:nth-child(even) { background-color: #faebd7; }
    """,
    
    "art_deco": """
        body { background: linear-gradient(135deg, #1a1a1a 0%%, #2d2d2d 100%%); padding: 25px; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Poiret One', sans-serif; 
                background: #f0f0f0; border: 5px solid #d4af37; }
        th { background: linear-gradient(180deg, #1a1a1a 0%%, #000000 100%%); color: #d4af37; 
             padding: 16px; font-weight: 600; text-align: center; font-size: 16px; 
             text-transform: uppercase; letter-spacing: 4px; border-bottom: 3px solid #d4af37; }
        td { padding: 14px; border-right: 1px solid #bbb; border-bottom: 1px solid #bbb; }
        td:last-child { border-right: none; }
    """,
    
    # ==================== NEWSPAPER & MAGAZINE ====================
    "new_york_times": """
        body { background-color: #f7f7f7; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Cheltenham', Georgia, serif; 
                border-top: 3px solid #000; }
        th { padding: 12px 8px; border-bottom: 1px solid #000; font-weight: 700; text-align: left; 
             font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
        td { padding: 10px 8px; border-bottom: 1px solid #e0e0e0; font-size: 13px; line-height: 1.6; }
        tr:nth-child(even) { background-color: #fafafa; }
    """,
    
    "vogue_magazine": """
        body { background-color: #ffffff; padding: 40px; }
        table { border-collapse: separate; border-spacing: 0; width: 100%%; 
                font-family: 'Didot', 'Bodoni MT', serif; }
        th { padding: 20px; font-weight: 300; text-align: left; font-size: 24px; 
             border-bottom: 2px solid #000; letter-spacing: 2px; }
        td { padding: 18px 20px; border-bottom: 1px solid #e0e0e0; font-size: 14px; 
             font-weight: 300; }
    """,
    
    "sports_illustrated": """
        body { background: linear-gradient(135deg, #c41e3a 0%%, #000080 100%%); padding: 20px; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Impact', sans-serif; 
                background: white; border: 5px solid #c41e3a; }
        th { background: linear-gradient(90deg, #c41e3a 0%%, #000080 100%%); color: white; 
             padding: 15px; font-weight: 700; text-align: center; font-size: 18px; 
             text-transform: uppercase; letter-spacing: 2px; }
        td { padding: 12px; border-bottom: 2px solid #e0e0e0; font-size: 15px; text-align: center; }
        tr:hover { background-color: #fff3cd; }
    """,
    
    # ==================== TECH & DIGITAL ====================
    "github_dark": """
        body { background-color: #0d1117; padding: 20px; }
        table { border-collapse: collapse; width: 100%%; font-family: 'SFMono-Regular', Consolas, monospace; 
                background: #161b22; border: 1px solid #30363d; border-radius: 6px; }
        th { background-color: #161b22; color: #c9d1d9; padding: 12px; border-bottom: 1px solid #21262d; 
             font-weight: 600; text-align: left; font-size: 13px; }
        td { padding: 10px 12px; color: #c9d1d9; border-bottom: 1px solid #21262d; font-size: 13px; }
        tr:hover { background-color: #1c2128; }
    """,
    
    "apple_human_interface": """
        body { background-color: #f5f5f7; }
        table { border-collapse: separate; border-spacing: 0; width: 100%%; 
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif; 
                background: white; border-radius: 18px; overflow: hidden; 
                box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
        th { background-color: white; color: #1d1d1f; padding: 16px 20px; font-weight: 600; 
             text-align: left; font-size: 15px; border-bottom: 1px solid #d2d2d7; }
        td { padding: 14px 20px; border-bottom: 1px solid #f5f5f7; font-size: 15px; color: #1d1d1f; }
    """,
    
    "android_material": """
        body { background-color: #fafafa; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Roboto', sans-serif; 
                background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
        th { background-color: #6200ee; color: white; padding: 16px; font-weight: 500; 
             text-align: left; font-size: 14px; text-transform: uppercase; letter-spacing: 1.25px; }
        td { padding: 14px 16px; border-bottom: 1px solid #e0e0e0; font-size: 14px; }
        tr:hover { background-color: #f5f5f5; }
    """,
    
    "windows_fluent": """
        body { background: linear-gradient(135deg, #0078d4 0%%, #00bcf2 100%%); padding: 25px; }
        table { border-collapse: separate; border-spacing: 0; width: 100%%; 
                font-family: 'Segoe UI', sans-serif; background: rgba(255,255,255,0.98); 
                border-radius: 8px; backdrop-filter: blur(20px); 
                box-shadow: 0 8px 32px rgba(0,0,0,0.2); }
        th { background: rgba(0,120,212,0.15); color: #0078d4; padding: 14px 18px; 
             font-weight: 600; text-align: left; font-size: 14px; }
        td { padding: 12px 18px; border-bottom: 1px solid #e0e0e0; font-size: 14px; }
    """,
    
    # ==================== RETRO & VINTAGE ====================
    "dos_terminal": """
        body { background-color: #000000; }
        table { border-collapse: collapse; width: 100%%; font-family: 'VT323', monospace; 
                color: #00ff00; background: #000; border: 2px solid #00ff00; }
        th { background-color: #003300; padding: 8px; border: 1px solid #00ff00; 
             font-weight: normal; font-size: 18px; text-transform: uppercase; }
        td { padding: 6px 8px; border: 1px solid #004400; font-size: 16px; }
    """,
    
    "commodore_64": """
        body { background-color: #40318d; }
        table { border-collapse: collapse; width: 100%%; font-family: 'C64 Pro Mono', monospace; 
                color: #7b71d6; background: #40318d; border: 4px solid #7b71d6; }
        th { background-color: #40318d; color: #a5a5ff; padding: 10px; border: 1px solid #7b71d6; 
             font-weight: normal; font-size: 16px; }
        td { padding: 8px; border: 1px solid #5b4bc3; font-size: 14px; }
    """,
    
    "1920s_typewriter": """
        body { background: #f4e8d0; padding: 30px; }
        table { border-collapse: collapse; width: 100%%; font-family: 'Courier Prime', monospace; 
                color: #2b2b2b; background: #faf7f0; border: 1px solid #8b7355; }
        th { padding: 10px; border-bottom: 2px solid #2b2b2b; font-weight: bold; 
             text-transform: uppercase; letter-spacing: 3px; }
        td { padding: 8px; border-bottom: 1px dotted #8b7355; }
    """,
    
    "1970s_psychedelic": """
        body { background: linear-gradient(45deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3); padding: 25px; }
        table { border-collapse: separate; border-spacing: 0; width: 100%%; 
                font-family: 'Cooper Black', serif; background: #fff; border-radius: 15px; 
                border: 5px solid #ff6b6b; box-shadow: 10px 10px 0 #feca57; }
        th { background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb); color: white; 
             padding: 15px; font-weight: bold; font-size: 18px; text-transform: uppercase; }
        td { padding: 12px; border-bottom: 3px solid #feca57; font-size: 14px; }
    """,
}


def generate_html_with_style(table_data: Dict[str, Any], base_font_size: str) -> Tuple[str, str]:
    style_name, css_template = random.choice(list(STYLES.items()))
    
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