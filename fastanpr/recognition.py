from paddleocr import PaddleOCR
from pydantic import BaseModel
from typing import List, Tuple, Optional


class Recognition(BaseModel):
    text: str
    poly: List[List[int]]
    conf: float

    class Config:
        frozen = True


class Recogniser:
    def __init__(self, device: str):
        self.device = device
        self.model = PaddleOCR(lang="en")

    def run(self, image) -> Optional[Recognition]:
        try:
            result = self.model.ocr(image)
            print(f"DEBUG: OCR result type: {type(result)}")
            print(f"DEBUG: OCR result length: {len(result) if result else 0}")

            # PaddleOCR 3.x returns a list containing results
            if isinstance(result, list) and len(result) > 0:
                predictions = result[0]
                print(f"DEBUG: Predictions type: {type(predictions)}")

                # Try to access the OCRResult object attributes directly
                if hasattr(predictions, "rec_texts"):
                    print("DEBUG: OCRResult object detected")
                    texts = getattr(predictions, "rec_texts", [])
                    scores = getattr(predictions, "rec_scores", [])
                    polys = getattr(predictions, "rec_polys", [])

                    print(f"DEBUG: rec_texts: {texts}")
                    print(f"DEBUG: rec_scores: {scores}")
                    print(f"DEBUG: rec_polys: {polys}")

                    # Let's also check if there are other attributes
                    print(
                        f"DEBUG: Available attributes: {[attr for attr in dir(predictions) if not attr.startswith('_')]}"
                    )

                    if texts and len(texts) > 0:
                        # Convert the first valid result
                        clean_text = _clean_text(texts[0])
                        clean_conf = (
                            float(scores[0]) if scores and len(scores) > 0 else 0.0
                        )
                        clean_poly = (
                            polys[0].tolist() if polys and len(polys) > 0 else []
                        )

                        # Convert numpy array polygon to the expected format if needed
                        if clean_poly and len(clean_poly) > 0:
                            try:
                                # Convert to list of [x, y] coordinates
                                formatted_poly = []
                                for point in clean_poly:
                                    if len(point) >= 2:
                                        formatted_poly.append(
                                            [int(float(point[0])), int(float(point[1]))]
                                        )
                                clean_poly = formatted_poly
                            except Exception:
                                clean_poly = []

                        return Recognition(
                            text=clean_text, poly=clean_poly, conf=clean_conf
                        )

                # Check if it's a dictionary format
                elif isinstance(predictions, dict):
                    texts = predictions.get("rec_texts", [])
                    scores = predictions.get("rec_scores", [])
                    polys = predictions.get("rec_polys", [])

                    print(f"DEBUG: Dictionary format - rec_texts: {texts}")
                    print(f"DEBUG: Dictionary format - rec_scores: {scores}")
                    print(f"DEBUG: Dictionary format - rec_polys: {polys}")

                    if texts and len(texts) > 0:
                        # Convert the first valid result
                        clean_text = _clean_text(texts[0])
                        clean_conf = (
                            float(scores[0]) if scores and len(scores) > 0 else 0.0
                        )
                        clean_poly = (
                            polys[0].tolist() if polys and len(polys) > 0 else []
                        )

                        # Convert numpy array polygon to the expected format if needed
                        if clean_poly and len(clean_poly) > 0:
                            try:
                                # Convert to list of [x, y] coordinates
                                formatted_poly = []
                                for point in clean_poly:
                                    if len(point) >= 2:
                                        formatted_poly.append(
                                            [int(float(point[0])), int(float(point[1]))]
                                        )
                                clean_poly = formatted_poly
                            except Exception:
                                clean_poly = []

                        return Recognition(
                            text=clean_text, poly=clean_poly, conf=clean_conf
                        )

                # Unknown format
                else:
                    print(f"DEBUG: Unknown predictions type: {type(predictions)}")
                    print(f"DEBUG: Predictions content: {str(predictions)[:500]}")

            return None
        except Exception as e:
            print(f"DEBUG: Exception in OCR: {e}")
            return None


def _clean_ocr(
    polys: List[List[List[int]]], texts: List[str], confidences: List[float]
) -> Tuple[List[List[int]], str, float]:
    """Clean multiple ocr boxes from recognition model."""

    # Clean recognised texts removing relatively smaller ones
    polys, texts, confidences = _denoise_ocr_boxes(polys, texts, confidences)

    # Merge recognised texts
    return _merge_polys(polys), _merge_texts(texts), _merge_confs(confidences)


def _merge_texts(texts: List[str], delimiter: str = "") -> str:
    """Merge multiple texts into one."""
    return _clean_text(delimiter.join(texts))


def _denoise_ocr_boxes(
    polys: List[List[List[int]]], texts: List[str], confs: List[float]
) -> Tuple[List[List[List[int]]], List[str], List[float]]:
    """Remove noisy ocr boxes detected by the recognition model. Boxes less than half the height of the longest text
    will be removed."""

    # get the longest box
    polys_x_points = [[point[0] for point in poly] for poly in polys]
    poly_length = [
        max(poly_x_points) - min(poly_x_points) for poly_x_points in polys_x_points
    ]
    longest_poly_idx = max(enumerate(poly_length), key=lambda x: x[1])[0]

    # get the height of the longest box
    longest_poly_y_points = [point[1] for point in polys[longest_poly_idx]]
    longest_poly_height = max(longest_poly_y_points) - min(longest_poly_y_points)

    # if it is smaller than half of the height of the longest box, remove the bos
    valid_idx = [
        idx
        for idx, poly in enumerate(polys)
        if (max(point[1] for point in poly) - min(point[1] for point in poly))
        >= longest_poly_height
    ]

    return (
        [polys[i] for i in valid_idx],
        [texts[i] for i in valid_idx],
        [confs[i] for i in valid_idx],
    )


def _merge_polys(polys: List[List[List[int]]]) -> List[List[int]]:
    """Merge multiple boxes into one."""

    # Initialize variables to store min and max coordinates
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    # Iterate through all boxes to find min and max coordinates
    for box in polys:
        for point in box:
            min_x = min(min_x, point[0])
            min_y = min(min_y, point[1])
            max_x = max(max_x, point[0])
            max_y = max(max_y, point[1])

    # Construct the merged box
    merged_box = [
        [int(min_x), int(min_y)],
        [int(max_x), int(min_y)],
        [int(max_x), int(max_y)],
        [int(min_x), int(max_y)],
    ]

    return merged_box


def _merge_confs(confidences: List[float]) -> float:
    """Merge multiple confidences into one."""
    result = 1.0
    for confidence in confidences:
        result *= confidence
    return result


def _clean_text(text: str) -> str:
    """Clean text by removing non-alphanumerics."""
    return "".join([t for t in text if t.isalnum()])
