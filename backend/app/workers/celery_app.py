from celery import Celery
from app.config import settings

# Initialize Celery app instance
celery_app = Celery(
    "document_processing_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.workers.tasks"],
)

# Celery application configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes hard limit per task
    worker_prefetch_multiplier=1,  # Prevent worker greediness
)
