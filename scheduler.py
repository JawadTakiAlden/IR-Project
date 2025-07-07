# scheduler.py

from apscheduler.schedulers.background import BackgroundScheduler

from services.crawler_service import WikipediaCrawler
from database import SessionLocal
import time
from fastapi import  Depends
from sqlalchemy.orm import Session

crawler = WikipediaCrawler()

WIKI_URLS = "https://en.wikipedia.org/wiki/Special:Random"

def crawl_job():
    db = SessionLocal()
    try:
        print("[INFO] Starting crawling job...")
        result = crawler.crawl_and_store(WIKI_URLS, dataset_name="wikipedia", db=db)
        print("[CRAWLED]", result)
    except Exception as e:
        print("[ERROR]", str(e))
    finally:
        db.close()

# Set up scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(crawl_job, 'interval', minutes=1)
scheduler.start()

# Keep script alive
print("Scheduler started. Running crawl job every hour.")
try:
    while True:
        time.sleep(60)
except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown()
    print("Scheduler stopped.")
