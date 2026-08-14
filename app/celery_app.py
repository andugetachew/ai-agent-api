from celery import Celery

from config import settings

celery_app = Celery(
    "agent_tasks",
    broker=settings.celery_broker_url,
    backend=settings.celery_broker_url,
    include=["tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)