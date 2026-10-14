from app.routes.upload import router as upload_router
from app.routes.questions import router as questions_router
from app.routes.documents import router as documents_router

__all__ = ["upload_router", "questions_router", "documents_router"]
