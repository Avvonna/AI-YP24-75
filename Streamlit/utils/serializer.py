from datetime import datetime


def serialize_datetime(obj):
    """Small function for serializing datetime into json"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError("Type not serializable")
