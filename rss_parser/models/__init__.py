"""
Models created according to https://www.rssboard.org/rss-specification.

Some types and validation may be a bit custom to account for broken standards in some RSS feeds.
"""
from pydantic import BaseModel

from rss_parser.models.utils import camel_case


class RSSBaseModel(BaseModel):
    # TODO: Override func to be able to input camel

    class Config:
        allow_mutation = False
        alias_generator = camel_case
