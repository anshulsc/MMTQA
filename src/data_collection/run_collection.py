import time
from src.configs import collection_config as cfg
from src.data_collection.wikitables_collector import WikiTableCollector
from src.data_collection.axriv_collector import ArxivTableCollector
from src.data_collection.finqa_collector import FinqaCollector
from src.data_collection.github_collector import GithubCsvCollector

def main():

    start_time = time.time()
    print("==============================================")
    print("      STARTING PHASE 1: DATA COLLECTION       ")
    print("==============================================")

    collectors = [
        WikiTableCollector(target_count=cfg.COLLECTION_TARGETS["wikisql"]),
        ArxivTableCollector(target_count=cfg.COLLECTION_TARGETS["arxiv"]),
        FinqaCollector(target_count=cfg.COLLECTION_TARGETS["finqa"]),
        GithubCsvCollector
       
    ]

    for collector in collectors:
        collector.run()

    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n==============================================")
    print("      PHASE 1: DATA COLLECTION COMPLETE       ")
    print("==============================================")
    print(f"Total time taken: {total_time:.2f} seconds")

if __name__ == '__main__':
    main()