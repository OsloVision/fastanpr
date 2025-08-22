from typing import List, Optional
from pydantic import BaseModel


class NumberPlate(BaseModel):
    det_box: List[int]
    det_conf: float
    rec_poly: Optional[List[List[int]]] = None
    rec_text: Optional[str] = None
    rec_conf: Optional[float] = None

    class Config:
        frozen = True
