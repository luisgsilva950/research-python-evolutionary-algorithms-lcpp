import math

from pydantic import BaseModel, Field

from algorithms.models.coordinate import Coordinate


class Edge(BaseModel):
    p1: Coordinate
    p2: Coordinate
    distance: float = Field(default=0)

    def __init__(self, **data):
        super().__init__(**data)
        self.distance = math.sqrt((self.p1.x - self.p2.x) ** 2 + (self.p1.y - self.p2.y) ** 2)
