import warnings
from copy import deepcopy
from json import loads
from math import ceil, floor, trunc
from operator import add, eq, floordiv, ge, gt, index, invert, le, lt, mod, mul, ne, neg, pos, pow, sub, truediv
from typing import Generic, Optional, Type, TypeVar, Union

from pydantic import create_model
from pydantic.generics import GenericModel
from pydantic.json import pydantic_encoder

from rss_parser.models import XMLBaseModel

T = TypeVar("T")


class TagRaw(GenericModel, Generic[T]):
    """
    >>> from rss_parser.models import XMLBaseModel
    >>> class Model(XMLBaseModel):
    ...     number: Tag[int]
    ...     string: Tag[str]
    >>> m = Model(
    ...     number=1,
    ...     string={'@attr': '1', '#text': 'content'},
    ... )
    >>> # Content value is an integer, as per the generic type
    >>> m.number.content
    1
    >>> # But you're still able to use the Tag itself in common operators
    >>> m.number.content + 10  == m.number + 10
    True
    >>> # As it's the case for methods/attributes not found in the Tag itself
    >>> m.number.bit_length()
    1
    >>> # types are NOT the same, however, the interfaces are very similar most of the time
    >>> type(m.number), type(m.number.content)
    (<class 'rss_parser.models.image.Tag[int]'>, <class 'int'>)
    >>> # The attributes are empty by default
    >>> m.number.attributes
    {}
    >>> # But are populated when provided.
    >>> # Note that the @ symbol is trimmed from the beggining, however, camelCase is not converted
    >>> m.string.attributes
    {'attr': '1'}
    >>> # Generic argument types are handled by pydantic - let's try to provide a string for a Tag[int] number
    >>> m = Model(number='not_a_number', string={'@customAttr': 'v', '#text': 'str tag value'})
    Traceback (most recent call last):
        ...
    pydantic.error_wrappers.ValidationError: 1 validation error for Model
    number -> content
      value is not a valid integer (type=type_error.integer)
    """

    # Optional in case of self-closing tags
    content: Optional[T]
    attributes: dict

    def __getattr__(self, item):
        """Forward default getattr for content for simplicity."""
        return getattr(self.content, item)

    def __getitem__(self, key):
        return self.content[key]

    def __setitem__(self, key, value):
        self.content[key] = value

    @classmethod
    def __get_validators__(cls):
        yield cls.pre_convert
        yield cls.validate

    @classmethod
    def pre_convert(cls, v: Union[T, dict], **kwargs):  # noqa
        """Used to split tag's text with other xml attributes."""
        if isinstance(v, dict):
            data = deepcopy(v)
            attributes = {k.lstrip("@"): v for k, v in data.items() if k.startswith("@")}
            content = data.pop("#text", data) if not len(attributes) == len(data) else None
            return {"content": content, "attributes": attributes}
        return {"content": v, "attributes": {}}

    @classmethod
    def flatten_tag_encoder(cls, v):
        """Encoder that translates Tag objects (dict) to plain .content values (T)."""
        bases = v.__class__.__bases__
        if XMLBaseModel in bases:
            # Can't pass encoder to .dict :/
            return loads(v.json_plain())
        if cls in bases:
            return v.content

        return pydantic_encoder(v)


def _make_proxy_operator(operator):
    def f(self, *args):
        return operator(self.content, *args)

    f.__name__ = operator.__name__

    return f


class Tag(TagRaw[T], GenericModel, Generic[T]):
    # Unary
    __pos__ = _make_proxy_operator(pos)
    __neg__ = _make_proxy_operator(neg)
    __abs__ = _make_proxy_operator(abs)
    __invert__ = _make_proxy_operator(invert)
    __round__ = _make_proxy_operator(round)
    __floor__ = _make_proxy_operator(floor)
    __ceil__ = _make_proxy_operator(ceil)
    # Conversion
    __str__ = _make_proxy_operator(str)
    __int__ = _make_proxy_operator(int)
    __float__ = _make_proxy_operator(float)
    __bool__ = _make_proxy_operator(bool)
    __complex__ = _make_proxy_operator(complex)
    __oct__ = _make_proxy_operator(oct)
    __hex__ = _make_proxy_operator(hex)
    __index__ = _make_proxy_operator(index)
    __trunc__ = _make_proxy_operator(trunc)
    # Comparison
    __lt__ = _make_proxy_operator(lt)
    __gt__ = _make_proxy_operator(gt)
    __le__ = _make_proxy_operator(le)
    __eq__ = _make_proxy_operator(eq)
    __ne__ = _make_proxy_operator(ne)
    __ge__ = _make_proxy_operator(ge)
    # Arithmetic
    __add__ = _make_proxy_operator(add)
    __sub__ = _make_proxy_operator(sub)
    __mul__ = _make_proxy_operator(mul)
    __truediv__ = _make_proxy_operator(truediv)
    __floordiv__ = _make_proxy_operator(floordiv)
    __mod__ = _make_proxy_operator(mod)
    __pow__ = _make_proxy_operator(pow)
