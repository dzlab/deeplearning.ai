from typing import List
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import (
    BaseModel,
)
from utils import encode_image, bt_embedding_from_prediction_guard
from tqdm import tqdm

class BridgeTowerEmbeddings(BaseModel, Embeddings):
    """ BridgeTower embedding model """
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using BridgeTower.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = []
        for text in texts:
            embedding = bt_embedding_from_prediction_guard(text, "")
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using BridgeTower.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]

    def embed_image_text_pairs(self, texts: List[str], images: List[str], batch_size=2) -> List[List[float]]:
        """Embed a list of image-text pairs using BridgeTower.

        Args:
            texts: The list of texts to embed.
            images: The list of path-to-images to embed
            batch_size: the batch size to process, default to 2
        Returns:
            List of embeddings, one for each image-text pairs.
        """

        # the length of texts must be equal to the length of images
        assert len(texts)==len(images), "the len of captions should be equal to the len of images"

        embeddings = []
        for path_to_img, text in tqdm(zip(images, texts), total=len(texts)):
            embedding = bt_embedding_from_prediction_guard(text, encode_image(path_to_img))
            embeddings.append(embedding)
        return embeddings