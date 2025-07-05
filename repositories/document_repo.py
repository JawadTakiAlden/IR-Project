from sqlalchemy.orm import Session
from models.document import Document
from typing import List

def upsert_document(db: Session, doc: Document):
    db.merge(doc)

def get_documents_by_source(db: Session, source: str) -> List[Document]:
    return db.query(Document).filter(Document.source == source).all()

def commit(db: Session):
    db.commit()