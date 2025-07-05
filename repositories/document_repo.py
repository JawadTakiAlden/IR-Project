from sqlalchemy.orm import Session
from models.document import Document

def upsert_document(db: Session, doc: Document):
    db.merge(doc)

def commit(db: Session):
    db.commit()