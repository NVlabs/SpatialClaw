import asyncio
from dataclasses import dataclass
from typing import List, Set

import easyocr
import numpy as np
from PIL import Image
import torch

from spatial_agent.gpu_models.base import AgentTool, AgentToolOutput, AgentContext
ImageLoader = None  # stub: not used (PIL images passed directly)

__ALL__ = ['EasyOCRModel', 'EasyOCRModelOutput']


@dataclass
class EasyOCROutput(AgentContext):
    """
    texts (List[str]): A list of `N` recognized texts.
    boxes (torch.Tensor): Shape `(N, 4)`. Bounding boxes for each recognized text.
    scores (torch.Tensor): Shape `(N,)`. Confidence score for each recognition, ranging from 0.0 to 1.0.
    """
    texts: List[str]
    boxes: torch.Tensor
    scores: torch.Tensor

    def to_message_content(self, top_k: int = 5) -> str:
        if not self.texts:
            return (
                'Failed: No text is found in the image. The target might not '
                'contain text or is illegible.'
            )

        num_found = len(self.texts)
        
        # Combine texts and scores for sorting
        detections = sorted(
            zip(self.texts, self.scores.cpu().tolist()), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        summary_parts = [
            f'Found {num_found} piece(s) of text.'
        ]
        
        # Display the top_k most confident results
        num_to_display = min(num_found, top_k)
        if num_to_display > 0:
            summary_parts.append(f'The top {num_to_display} results are:')
            for text, score in detections[:num_to_display]:
                summary_parts.append(f'- "{text}" (Confidence: {score:.2f})')
        
        return '\n'.join(summary_parts)

    def get_computation_doc(self) -> Set[str]:
        return set(['boxes'])


class EasyOCR(AgentTool):
    CPU_CONSUMED = 0.25
    VRAM_CONSUMED = 5.0
    AUTOSCALING_MIN_REPLICAS = 0
    AUTOSCALING_MAX_REPLICAS = 2

    DEVICE = 'cuda'

    def __init__(self, image_loader: ImageLoader) -> None:
        super().__init__()
        self.model = easyocr.Reader(['ch_sim', 'en'], gpu=(self.DEVICE == 'cuda'))
        self.image_loader = image_loader

    @torch.no_grad()
    def _ocr(self, image: np.ndarray) -> EasyOCROutput:
        results = self.model.readtext(image=image)
        texts, boxes, scores = [], [], []
        for result in results:
            box, text, score = result

            # transform box into [x1, y1, x2, y2] format
            box = torch.tensor([[int(coord) for coord in p] for p in box])
            box = torch.cat([box.min(dim=0).values, box.max(dim=0).values])
            
            texts.append(text)
            boxes.append(box)
            scores.append(float(score))
        
        return EasyOCROutput(
            texts=texts,
            boxes=torch.stack(boxes) if len(boxes) > 0 else torch.empty(0, 4),
            scores=torch.tensor(scores),
        )

    @AgentTool.document_output_class(EasyOCROutput)
    async def ocr(self, image_source: str | Image.Image) -> AgentToolOutput:
        """
        Performs Optical Character Recognition (OCR) on an image.
        Args:
            image_source (Image.Image): The `PIL.Image.Image` object.
        """
        if isinstance(image_source, Image.Image):
            image = image_source
        else:
            image_result = await self.image_loader.load_image.remote(image_source)
            if image_result.err:
                return image_result
            image = image_result.result

        orc_output = await asyncio.to_thread(self._ocr, np.array(image))
        return self.success(result=orc_output)
