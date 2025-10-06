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

    def _parse_html_tables(self, html_content: str) -> list[pd.DataFrame]:
        soup = BeautifulSoup(html_content, 'html.parser')
        try:
            dfs = pd.read_html(str(soup), flavor='bs4')
            return [df for df in dfs if not df.empty]
        except ValueError:
            return []
            
    def _parse_pdf_tables(self, pdf_path: str) -> list[pd.DataFrame]:
        dfs = []
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                tables = page.find_tables()
                for table in tables:
                    try:
                        table_data = table.extract()
                        if not table_data or len(table_data) < 2:
                            continue
                        
                        header = table_data[0]
                        cleaned_header = [str(h).replace('\n', ' ').strip() if h is not None else '' for h in header]
                        
                        df = pd.DataFrame(table_data[1:], columns=cleaned_header)
                        dfs.append(df)
                    except Exception:
                        continue 
        except Exception as e:
            logging.error(f"Failed to process PDF {pdf_path}: {e}")
        return dfs

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

        for result in search.results():
            if self.collected_count >= self.target_count:
                break

           
            html_url = result.entry_id.replace("/abs/", "/html/")
            
            try:
                # --- HTML First Approach ---
                response = requests.get(html_url, headers=self.headers, timeout=10)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

                dfs = self._parse_html_tables(response.content)
                parse_method = "html"

                if not dfs:
                    logging.info(f"No tables found in HTML for {result.entry_id}. Falling back to PDF.")
                    pdf_path = result.download_pdf(dirpath=str(cfg.RAW_DATA_DIR))
                    dfs = self._parse_pdf_tables(pdf_path)
                    parse_method = "pdf"

                if not dfs:
                    logging.info(f"No tables found for {result.entry_id} in either HTML or PDF.")
                    continue

                for df in dfs:
                    if self.collected_count >= self.target_count:
                        break
                    
                    extra_meta = {
                        "source_dataset": "arxiv",
                        "arxiv_id": result.entry_id,
                        "paper_title": result.title,
                        "parse_method": parse_method
                    }
                    self._process_and_save(df, extra_meta)
                
                time.sleep(1) 

            except requests.exceptions.HTTPError as e:
                logging.warning(f"HTML version not found for {result.entry_id} (Status {e.response.status_code}). Skipping paper.")
                continue
            except Exception as e:
                logging.error(f"An unexpected error occurred for paper {result.entry_id}: {e}")
                continue


if __name__ == '__main__':
    print("Running ArxivTableCollector as a standalone script (HTML first)...")
   
    target = 20
    collector = ArxivTableCollector(target_count=target)
    collector.run()
    print("Standalone run finished.")