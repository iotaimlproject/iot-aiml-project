from databases import Database

from api.config import settings

database = Database(
    f"mysql+asyncmy://{settings.db_user}:{settings.db_password}@"
    f"{settings.db_host}:{settings.db_port}/{settings.db_database}"
)


async def fetch_latest_row():
    query = f"""
        SELECT rpm, availability, performance, quality, oee, downtime_minutes
        FROM {settings.db_table}
        ORDER BY timestamp DESC
        LIMIT 1
    """
    return await database.fetch_one(query)
