# services/crawler_service.py

import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session
from models.document import Document
from services.processor import TextProcessor
from repositories import document_repo
from uuid import uuid4


class WikipediaCrawler:
    def __init__(self):
        self.processor = TextProcessor()

    def crawl_and_store(self, url: str, dataset_name: str, db: Session):
        try:
            res = requests.get(url)
            soup = BeautifulSoup(res.content, "html.parser")

            title = soup.find("h1").get_text(strip=True)
            content = "\n".join(p.get_text(strip=True) for p in soup.find_all("p"))

            processed_text = self.processor.normalize(content)
            if not processed_text:
                return {"status": "skipped", "reason": "empty after processing"}

            document = Document(
                doc_id=str(uuid4()),
                raw_text=content,
                processed_text=" ".join(processed_text),
                source=dataset_name
            )
            document_repo.upsert_document(db, document)
            document_repo.commit(db)
            return {"status": "success", "title": title}

        except Exception as e:
            return {"status": "error", "message": str(e)}
