import pandas as pd
import arxiv
import requests
from bs4 import BeautifulSoup
import fitz  
import time
import logging

from src.data_collection.base_collector import BaseCollector
from src.configs import collection_config as cfg


logging.basicConfig(level=logging.INFO)
logging.getLogger("arxiv").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


class ArxivTableCollector(BaseCollector):
    def __init__(self, target_count: int):
        super().__init__(source_name="arxiv", target_count=target_count)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def _parse_html_tables(self, html_content: str) -> list[tuple[pd.DataFrame, str]]:
        """Parse HTML tables and extract captions. Returns list of (dataframe, caption) tuples."""
        soup = BeautifulSoup(html_content, 'html.parser')
        tables_with_captions = []
        
        try:
            # Find all table elements
            table_elements = soup.find_all('table')
            
            for table_elem in table_elements:
                # Try to find caption
                caption = ""
                
                # Check for caption tag within table
                caption_tag = table_elem.find('caption')
                if caption_tag:
                    caption = caption_tag.get_text(strip=True)
                else:
                    # Check for preceding figcaption or div with class containing 'caption'
                    prev_sibling = table_elem.find_previous_sibling()
                    if prev_sibling and prev_sibling.name in ['figcaption', 'caption']:
                        caption = prev_sibling.get_text(strip=True)
                    elif prev_sibling and 'caption' in prev_sibling.get('class', []):
                        caption = prev_sibling.get_text(strip=True)
                    
                    # Check parent figure for figcaption
                    parent_figure = table_elem.find_parent('figure')
                    if parent_figure and not caption:
                        figcaption = parent_figure.find('figcaption')
                        if figcaption:
                            caption = figcaption.get_text(strip=True)
                
                # Parse the table
                try:
                    dfs = pd.read_html(str(table_elem), flavor='bs4')
                    for df in dfs:
                        if not df.empty:
                            tables_with_captions.append((df, caption))
                except ValueError:
                    continue
                    
        except Exception as e:
            logging.error(f"Error parsing HTML tables: {e}")
            
        return tables_with_captions
            
    def _parse_pdf_tables(self, pdf_path: str) -> list[tuple[pd.DataFrame, str]]:
        """Parse PDF tables and extract captions. Returns list of (dataframe, caption) tuples."""
        tables_with_captions = []
        
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                tables = page.find_tables()
                
                for table_idx, table in enumerate(tables):
                    try:
                        table_data = table.extract()
                        if not table_data or len(table_data) < 2:
                            continue
                        
                        header = table_data[0]
                        cleaned_header = [str(h).replace('\n', ' ').strip() if h is not None else '' for h in header]
                        
                        df = pd.DataFrame(table_data[1:], columns=cleaned_header)
                        
                        # Extract caption from text near table
                        caption = self._extract_table_caption_from_pdf(page, table, page_num, table_idx)
                        
                        tables_with_captions.append((df, caption))
                    except Exception:
                        continue 
        except Exception as e:
            logging.error(f"Failed to process PDF {pdf_path}: {e}")
            
        return tables_with_captions
    
    def _extract_table_caption_from_pdf(self, page, table, page_num: int, table_idx: int) -> str:
        """Extract caption text near a table in PDF."""
        caption = ""
        
        try:
            # Get table bounding box
            table_bbox = table.bbox
            
            # Search for text above the table (typical caption location)
            text_dict = page.get_text("dict")
            
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                    
                block_bbox = block["bbox"]
                
                # Check if block is above and close to the table
                if (block_bbox[3] <= table_bbox[1] and  # below table top
                    block_bbox[1] >= table_bbox[1] - 100 and  # within 100 points above
                    abs(block_bbox[0] - table_bbox[0]) < 50):  # horizontally aligned
                    
                    # Extract text from block
                    block_text = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            block_text += span.get("text", "") + " "
                    
                    block_text = block_text.strip()
                    
                    # Check if it looks like a caption (starts with "Table" or similar)
                    if any(keyword in block_text.lower()[:20] for keyword in ["table", "tab.", "tab:"]):
                        caption = block_text
                        break
                        
        except Exception as e:
            logging.debug(f"Could not extract caption for table {table_idx} on page {page_num}: {e}")
            
        return caption

    def collect(self):
        query_surveys = 'ti:("survey" OR "review" OR "benchmark") OR abs:("comparative study" OR "analysis of methods" OR "qualitative analysis")'
        query_finance = '(ti:("financial" OR "economic" OR "market") OR abs:("stock prediction" OR "risk analysis"))'
        query_methods = 'abs:("experimental results" OR "dataset analysis")'
        combined_themes = f"({query_surveys}) OR ({query_finance}) OR ({query_methods})"
        categories = "(cat:cs.CL OR cat:cs.AI OR cat:cs.LG OR cat:q-fin.* OR cat:econ.EM)"
        date_range = "submittedDate:[20230101 TO 20250531]"

        # Final Query
        final_query = f"({combined_themes}) AND ({categories}) AND ({date_range})"
        
        search = arxiv.Search(
            query=final_query,
            max_results=self.target_count * 5,  
            sort_by=arxiv.SortCriterion.Relevance
        )

        try:
            for result in search.results():
                if self.collected_count >= self.target_count:
                    break

                html_url = result.entry_id.replace("/abs/", "/html/")
                
                try:
                    # --- HTML First Approach ---
                    response = requests.get(html_url, headers=self.headers, timeout=10)
                    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

                    tables_with_captions = self._parse_html_tables(response.content)
                    parse_method = "html"

                    if not tables_with_captions:
                        logging.info(f"No tables found in HTML for {result.entry_id}. Falling back to PDF.")
                        pdf_path = result.download_pdf(dirpath=str(cfg.RAW_DATA_DIR))
                        tables_with_captions = self._parse_pdf_tables(pdf_path)
                        parse_method = "pdf"

                    if not tables_with_captions:
                        logging.info(f"No tables found for {result.entry_id} in either HTML or PDF.")
                        continue

                    for df, caption in tables_with_captions:
                        if self.collected_count >= self.target_count:
                            break
                        
                        extra_meta = {
                            "source_dataset": "arxiv",
                            "arxiv_id": result.entry_id,
                            "paper_title": result.title,
                            "parse_method": parse_method,
                            "table_caption": caption if caption else None
                        }
                        self._process_and_save(df, extra_meta)
                    
                    time.sleep(1) 

                except requests.exceptions.HTTPError as e:
                    logging.warning(f"HTML version not found for {result.entry_id} (Status {e.response.status_code}). Skipping paper.")
                    continue
                except Exception as e:
                    logging.error(f"An unexpected error occurred for paper {result.entry_id}: {e}")
                    continue
        except arxiv.UnexpectedEmptyPageError as e:
            logging.warning(f"ArXiv API returned empty page. Moving to next query variation.")
            pass
        except Exception as e:
            logging.error(f"Unexpected error. Stopping collection: {e}")
            pass

if __name__ == '__main__':
    print("Running ArxivTableCollector as a standalone script (HTML first)...")
   
    target = cfg.COLLECTION_TARGETS.get("arxiv", 100) 
    collector = ArxivTableCollector(target_count=target)
    collector.run()
    print("Standalone run finished.")