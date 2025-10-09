import requests
import pandas as pd
import io
import time
from typing import Optional, Dict, Any, Generator, List, Set
from dataclasses import dataclass
import random
from base64 import b64decode
from termcolor import cprint

from src.data_collection.base_collector import BaseCollector
from src.configs import collection_config as cfg


@dataclass
class SearchConfig:
    min_size: int = 1000  
    max_size: int = 500000 
    min_stars: int = 0
    quality_filters: bool = True

    search_terms: List[str] = None
    
    def __post_init__(self):
        if self.search_terms is None:
            
            self.search_terms = [
                'data',
                'sample', 
                'dataset',
                'test',
                'example',
                'records',
                'list',
                'table',
                'info',
                'stats'
            ]


@dataclass
class TableQualityMetrics:
    total_parsed: int = 0
    total_filtered: int = 0
    parse_failures: int = 0
    too_small: int = 0
    missing_headers: int = 0
    api_errors: int = 0


class GithubCsvCollector(BaseCollector):

    API_BASE = "https://api.github.com"
    RAW_CONTENT_URL = "https://raw.githubusercontent.com"
    
    def __init__(
        self, 
        target_count: int,
        search_config: Optional[SearchConfig] = None,
        delay_range: tuple = (2, 4),
        enable_quality_filters: bool = True,
        github_token: Optional[str] = None
    ):
        super().__init__(source_name="github_csv", target_count=target_count)
        
        self.search_config = search_config or SearchConfig()
        self.delay_range = delay_range
        self.enable_quality_filters = enable_quality_filters
        self.github_token = github_token
        self.page = 1
        self.current_term_idx = 0
        self.session = self._create_session()
        self.collected_urls: Set[str] = set()
        self.quality_metrics = TableQualityMetrics()
        
        cprint("="*60, "cyan")
        cprint("GitHub CSV Collector - API Method", "green", attrs=["bold"])
        cprint("Using GitHub API to search for CSV files", "blue")
        cprint(f"Auth: {'Enabled (higher rate limits)' if github_token else 'Anonymous (60 requests/hour)'}", "yellow")
        cprint("="*60, "cyan")
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'GitHub-CSV-Collector'
        }
        
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        
        session.headers.update(headers)
        return session
    
    def _polite_delay(self):
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)
    
    def _check_rate_limit(self) -> Dict[str, Any]:
        try:
            response = self.session.get(f"{self.API_BASE}/rate_limit", timeout=10)
            if response.status_code == 200:
                data = response.json()
                core = data.get('resources', {}).get('core', {})
                return {
                    'limit': core.get('limit', 0),
                    'remaining': core.get('remaining', 0),
                    'reset': core.get('reset', 0)
                }
        except Exception as e:
            cprint(f"Rate limit check failed: {e}", "red")
        
        return {'limit': 0, 'remaining': 0, 'reset': 0}
    
    def _wait_for_rate_limit(self):
        rate_info = self._check_rate_limit()
        remaining = rate_info.get('remaining', 1)
        
        if remaining < 5:
            reset_time = rate_info.get('reset', time.time() + 60)
            wait_seconds = max(reset_time - time.time(), 60)
            cprint(f"Rate limit low ({remaining} remaining). Waiting {int(wait_seconds)} seconds...", "yellow", attrs=["bold"])
            time.sleep(wait_seconds + 5)
    
    def _build_search_query(self) -> str:
        term = self.search_config.search_terms[self.current_term_idx]
        
        query = f'{term} extension:csv size:{self.search_config.min_size}..{self.search_config.max_size}'
        
        if self.search_config.min_stars > 0:
            query += f' stars:>={self.search_config.min_stars}'
        
        return query
    
    def _rotate_search_term(self):
        self.current_term_idx = (self.current_term_idx + 1) % len(self.search_config.search_terms)
        self.page = 1
        cprint(f"Rotating to search term: '{self.search_config.search_terms[self.current_term_idx]}'", "magenta", attrs=["bold"])
    
    def _search_github_api(self) -> Generator[Dict[str, Any], None, None]:

        query = self._build_search_query()
        consecutive_empty_results = 0
        max_empty_results = 3
        max_pages = 10  
        
        cprint(f"Search query: {query}", "blue", attrs=["bold"])
        
        while self.collected_count < self.target_count:
            self._wait_for_rate_limit()
            
            params = {
                'q': query,
                'per_page': 100,  
                'page': self.page
            }
            
            try:
                cprint(f"API search page {self.page} (term: {self.search_config.search_terms[self.current_term_idx]})...", "cyan")
                
                response = self.session.get(
                    f"{self.API_BASE}/search/code",
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 403:
                    cprint("Rate limit exceeded or access forbidden", "red", attrs=["bold"])
                    self._wait_for_rate_limit()
                    continue
                elif response.status_code == 422:
                    cprint("Query validation failed, rotating search term", "yellow", attrs=["bold"])
                    self._rotate_search_term()
                    query = self._build_search_query()
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                items = data.get('items', [])
                total_count = data.get('total_count', 0)
                
                if not items:
                    consecutive_empty_results += 1
                    cprint(f"No results on page {self.page} ({consecutive_empty_results}/{max_empty_results})", "yellow")
                    
                    if consecutive_empty_results >= max_empty_results or self.page >= max_pages:
                        self._rotate_search_term()
                        query = self._build_search_query()
                        consecutive_empty_results = 0
                        continue
                    
                    self.page += 1
                    self._polite_delay()
                    continue
                
                consecutive_empty_results = 0
                cprint(f"Found {len(items)} CSV files (total: {total_count})", "green", attrs=["bold"])
                
                for item in items:
                    try:
                        processed_item = self._process_api_result(item)
                        if processed_item and processed_item['download_url'] not in self.collected_urls:
                            self.collected_urls.add(processed_item['download_url'])
                            yield processed_item
                    except Exception as e:
                        cprint(f"Error processing result: {e}", "red")
                        continue
                
                self.page += 1
                
                if self.page > max_pages:
                    self._rotate_search_term()
                    query = self._build_search_query()
                
                self._polite_delay()
                
            except requests.exceptions.RequestException as e:
                cprint(f"API request failed: {e}", "red", attrs=["bold"])
                self.quality_metrics.api_errors += 1
                self._polite_delay()
                continue
            except Exception as e:
                cprint(f"Unexpected error: {e}", "red", attrs=["bold"])
                break
    
    def _process_api_result(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            repo_info = item.get('repository', {})
            path = item.get('path', '')
            
            if not path.endswith('.csv'):
                return None
            
            owner = repo_info.get('owner', {}).get('login', '')
            repo_name = repo_info.get('name', '')
            default_branch = repo_info.get('default_branch', 'main')
            
            if not owner or not repo_name:
                return None
            
            download_url = f"{self.RAW_CONTENT_URL}/{owner}/{repo_name}/{default_branch}/{path}"
            
            return {
                'repository': {
                    'full_name': repo_info.get('full_name', f"{owner}/{repo_name}"),
                    'html_url': repo_info.get('html_url', ''),
                    'owner': owner,
                    'name': repo_name,
                    'stars': repo_info.get('stargazers_count', 0),
                    'language': repo_info.get('language', 'Unknown')
                },
                'path': path,
                'branch': default_branch,
                'download_url': download_url,
                'html_url': item.get('html_url', ''),
                'size': item.get('size', 0)
            }
            
        except Exception as e:
            cprint(f"Error processing API result: {e}", "red")
            return None
    
    def _parse_csv_content(self, content: str) -> Optional[pd.DataFrame]:
        try:
            for delimiter in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(
                        io.StringIO(content),
                        delimiter=delimiter,
                        on_bad_lines='skip',
                        low_memory=False,
                        encoding_errors='ignore'
                    )
                    if len(df.columns) > 1:  # Valid parse
                        return df
                except:
                    continue
            
            return None
            
        except Exception as e:
            cprint(f"CSV parsing failed: {e}", "red")
            return None
    
    def _apply_quality_filters(self, df: pd.DataFrame) -> bool:
        if not self.enable_quality_filters:
            return True

        if len(df) < 2 or len(df.columns) < 2:
            self.quality_metrics.too_small += 1
            cprint(f"Filtered: Too small ({len(df)}x{len(df.columns)})", "yellow")
            return False
        
        unnamed_cols = sum(1 for col in df.columns if str(col).startswith('Unnamed'))
        if unnamed_cols > len(df.columns) * 0.5:
            self.quality_metrics.missing_headers += 1
            cprint(f"Filtered: Missing headers ({unnamed_cols}/{len(df.columns)})", "yellow")
            return False
        
        if df.isnull().all().all():
            cprint("Filtered: All empty values", "yellow")
            return False
        
        self.quality_metrics.total_filtered += 1
        return True
    
    def _calculate_table_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=['number']).columns
        text_cols = df.select_dtypes(include=['object']).columns
        
        return {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'num_cells': len(df) * len(df.columns),
            'numeric_columns': len(numeric_cols),
            'text_columns': len(text_cols),
            'numeric_percentage': len(numeric_cols) / len(df.columns) if len(df.columns) > 0 else 0,
            'column_names': list(df.columns)[:20],  # First 20 columns only
            'data_types': {k: str(v) for k, v in df.dtypes.to_dict().items()}
        }
    
    def _fetch_and_parse_csv(self, item: Dict[str, Any]) -> Optional[pd.DataFrame]:
        download_url = item.get('download_url')
        if not download_url:
            return None
        
        try:
            cprint(f"Fetching: {item.get('path', 'unknown')}", "blue")
            
            response = requests.get(download_url, timeout=30)
            response.raise_for_status()
            
            content_length = len(response.content)
            if content_length > 10_000_000:  
                cprint(f"Skipped: File too large ({content_length} bytes)", "yellow")
                return None
            
            try:
                csv_content = response.content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    csv_content = response.content.decode('latin-1')
                except:
                    csv_content = response.content.decode('utf-8', errors='ignore')
            
            # Parse CSV
            df = self._parse_csv_content(csv_content)
            
            if df is None:
                self.quality_metrics.parse_failures += 1
                return None
            
            self.quality_metrics.total_parsed += 1
            
            if not self._apply_quality_filters(df):
                return None
            
            cprint(
                f"✓ Parsed: {item.get('path', 'unknown')[:60]} "
                f"({len(df)} rows × {len(df.columns)} cols)", 
                "green", 
                attrs=["bold"]
            )
            return df
            
        except Exception as e:
            cprint(f"Failed to fetch/parse: {e}", "red")
            return None
    
    def collect(self) -> None:
        cprint("="*60, "cyan")
        cprint("Starting GitHub CSV Collection via API", "green", attrs=["bold"])
        cprint(f"Target: {self.target_count} tables", "blue")
        cprint(f"Quality filters: {'Enabled' if self.enable_quality_filters else 'Disabled'}", "yellow")
        cprint("="*60, "cyan")
        
        # Check rate limit at start
        rate_info = self._check_rate_limit()
        cprint(f"API Rate Limit: {rate_info.get('remaining', '?')}/{rate_info.get('limit', '?')} remaining", "magenta", attrs=["bold"])
        
        successful = 0
        failed = 0
        
        for item in self._search_github_api():
            if self.collected_count >= self.target_count:
                cprint(f"✓ Target of {self.target_count} reached!", "green", attrs=["bold"])
                break
            
            self._polite_delay()
            
            df = self._fetch_and_parse_csv(item)
            
            if df is None:
                failed += 1
                continue
            

            stats = self._calculate_table_statistics(df)
            
            extra_meta = {
                "source_dataset": "github_csv",
                "repository": item['repository']['full_name'],
                "repository_url": item['repository']['html_url'],
                "repository_owner": item['repository']['owner'],
                "repository_name": item['repository']['name'],
                "repository_stars": item['repository'].get('stars', 0),
                "repository_language": item['repository'].get('language', 'Unknown'),
                "file_path": item.get('path', 'unknown'),
                "branch": item.get('branch', 'unknown'),
                "download_url": item['download_url'],
                "file_size_bytes": item.get('size', 0),
                "collection_method": "github_api",
                "table_statistics": stats
            }
            
            self._process_and_save(df, extra_meta)
            successful += 1
        
        self._print_collection_summary(successful, failed)
    
    def _print_collection_summary(self, successful: int, failed: int):
        cprint("="*60, "cyan")
        cprint("Collection Summary", "green", attrs=["bold"])
        cprint("-"*60, "cyan")
        cprint(f"  ✓ Successfully collected: {successful}", "green")
        cprint(f"  ✗ Failed attempts: {failed}", "red")
        cprint(f"  📊 Total in corpus: {self.collected_count}", "blue", attrs=["bold"])
        cprint("-"*60, "cyan")
        cprint("Quality Metrics:", "yellow", attrs=["bold"])
        cprint(f"  • Total parsed: {self.quality_metrics.total_parsed}", "white")
        cprint(f"  • Parse failures: {self.quality_metrics.parse_failures}", "red")
        cprint(f"  • Filtered (too small): {self.quality_metrics.too_small}", "yellow")
        cprint(f"  • Filtered (headers): {self.quality_metrics.missing_headers}", "yellow")
        cprint(f"  • Passed quality filters: {self.quality_metrics.total_filtered}", "green")
        cprint(f"  • API errors: {self.quality_metrics.api_errors}", "red")
        cprint("-"*60, "cyan")
        
        rate_info = self._check_rate_limit()
        cprint(f"Final API Rate Limit: {rate_info.get('remaining', '?')}/{rate_info.get('limit', '?')}", "magenta", attrs=["bold"])
        cprint("="*60, "cyan")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()


def main():
    cprint("Running GitHub CSV Collector via API", "green", attrs=["bold"])
    
    target = cfg.COLLECTION_TARGETS.get("github_csv", 20)

    github_token = "token"
    
    search_config = SearchConfig(
        min_size=1000,        
        max_size=500000,      
        min_stars=0,         
        quality_filters=True,
        search_terms=[
            'data', 'sample', 'dataset', 'test', 'example',
            'records', 'list', 'table', 'export', 'report'
        ]
    )
    
    with GithubCsvCollector(
        target_count=target,
        search_config=search_config,
        delay_range=(2, 4),
        enable_quality_filters=True,
        github_token=github_token
    ) as collector:
        collector.run()
    
    cprint("✓ Collection completed successfully", "green", attrs=["bold"])


if __name__ == '__main__':
    main()