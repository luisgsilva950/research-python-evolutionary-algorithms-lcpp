from pydantic import BaseModel


class Coordinate(BaseModel):
    x: float
    y: float
