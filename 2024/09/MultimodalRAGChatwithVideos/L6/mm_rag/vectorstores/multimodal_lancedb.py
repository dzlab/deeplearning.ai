from typing import Any, Iterable, List, Optional
from langchain_core.embeddings import Embeddings
import uuid
from langchain_community.vectorstores.lancedb import LanceDB

class MultimodalLanceDB(LanceDB):
    """`LanceDB` vector store to process multimodal data
    
    To use, you should have ``lancedb`` python package installed.
    You can install it with ``pip install lancedb``.

    Args:
        connection: LanceDB connection to use. If not provided, a new connection
                    will be created.
        embedding: Embedding to use for the vectorstore.
        vector_key: Key to use for the vector in the database. Defaults to ``vector``.
        id_key: Key to use for the id in the database. Defaults to ``id``.
        text_key: Key to use for the text in the database. Defaults to ``text``.
        image_path_key: Key to use for the path to image in the database. Defaults to ``image_path``.
        table_name: Name of the table to use. Defaults to ``vectorstore``.
        api_key: API key to use for LanceDB cloud database.
        region: Region to use for LanceDB cloud database.
        mode: Mode to use for adding data to the table. Defaults to ``overwrite``.



    Example:
        .. code-block:: python
            vectorstore = MultimodalLanceDB(uri='/lancedb', embedding_function)
            vectorstore.add_texts(['text1', 'text2'])
            result = vectorstore.similarity_search('text1')
    """
    
    def __init__(
        self,
        connection: Optional[Any] = None,
        embedding: Optional[Embeddings] = None,
        uri: Optional[str] = "/tmp/lancedb",
        vector_key: Optional[str] = "vector",
        id_key: Optional[str] = "id",
        text_key: Optional[str] = "text",
        image_path_key: Optional[str] = "image_path", 
        table_name: Optional[str] = "vectorstore",
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        mode: Optional[str] = "append",
    ):
        super(MultimodalLanceDB, self).__init__(connection, embedding, uri, vector_key, id_key, text_key, table_name, api_key, region, mode)
        self._image_path_key = image_path_key
        
    def add_text_image_pairs(
        self,
        texts: Iterable[str],
        image_paths: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Turn text-image pairs into embedding and add it to the database

        Args:
            texts: Iterable of strings to combine with corresponding images to add to the vectorstore.
            images: Iterable of path-to-images as strings to combine with corresponding texts to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate w    ith the texts.

        Returns:
            List of ids of the added text-image pairs.
        """
        # the length of texts must be equal to the length of images
        assert len(texts)==len(image_paths), "the len of transcripts should be equal to the len of images"

        # Embed texts and create documents
        docs = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        embeddings = self._embedding.embed_image_text_pairs(texts=list(texts), images=list(image_paths))  # type: ignore
        for idx, text in enumerate(texts):
            embedding = embeddings[idx]
            metadata = metadatas[idx] if metadatas else {"id": ids[idx]}
            docs.append(
                {
                    self._vector_key: embedding,
                    self._id_key: ids[idx],
                    self._text_key: text,
                    self._image_path_key : image_paths[idx],
                    "metadata": metadata,
                }
            )

        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = self.mode
        if self._table_name in self._connection.table_names():
            tbl = self._connection.open_table(self._table_name)
            if self.api_key is None:
                tbl.add(docs, mode=mode)
            else:
                tbl.add(docs)
        else:
            self._connection.create_table(self._table_name, data=docs)
        return ids

    @classmethod
    def from_text_image_pairs(
        cls,
        texts: List[str],
        image_paths: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        connection: Any = None,
        vector_key: Optional[str] = "vector",
        id_key: Optional[str] = "id",
        text_key: Optional[str] = "text",
        image_path_key: Optional[str] = "image_path",
        table_name: Optional[str] = "vectorstore",
        **kwargs: Any,
    ):

        instance = MultimodalLanceDB(
            connection=connection,
            embedding=embedding,
            vector_key=vector_key,
            id_key=id_key,
            text_key=text_key,
            image_path_key=image_path_key,
            table_name=table_name,
        )
        instance.add_text_image_pairs(texts, image_paths, metadatas=metadatas, **kwargs)

        return instance