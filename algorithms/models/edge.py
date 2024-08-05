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

    def __hash__(self):
        return hash((frozenset((self.p1, self.p2)), self.distance))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return False

        return (self.p1 == other.p1 and self.p2 == other.p2) or (self.p1 == other.p2 and self.p2 == other.p1)

    def reverse(self):
        self.p1, self.p2 = self.p2, self.p1

    def connected(self, edge: 'Edge') -> bool:
        return self.p2 == edge.p1 or self.p2 == edge.p2 or self.p1 == edge.p1 or self.p1 == edge.p2
