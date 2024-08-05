from pydantic import BaseModel


class Coordinate(BaseModel, frozen=True):
    x: float
    y: float
