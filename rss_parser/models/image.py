from typing import Optional

from rss_parser.models import RSSBaseModel


class Image(RSSBaseModel):
    """https://www.rssboard.org/rss-specification#ltimagegtSubelementOfLtchannelgt."""

    url: str = None
    "The URL of a GIF, JPEG or PNG image that represents the channel."

    title: str = None
    "Describes the image, it's used in the ALT attribute of the HTML <img> tag when the channel is rendered in HTML."

    link: str = None
    "The URL of the site, when the channel is rendered, the image is a link to the site. (Note, in practice the " "image <title> and <link> should have the same value as the channel's <title> and <link>."  # noqa

    width: Optional[int] = None
    "Number, indicating the width of the image in pixels."

    height: Optional[int] = None
    "Number, indicating the height of the image in pixels."

    description: Optional[str] = None
    "Contains text that is included in the TITLE attribute of the link formed around the image in the HTML rendering."
