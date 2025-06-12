from typing import AsyncGenerator, Iterator, Optional, Union

from kotaemon.base import BaseComponent, Document


class BasePdfExtractionBlock(BaseComponent):
    """A simple base class for pdf extraction blocks"""

    def stream(self, *args, **kwargs) -> Optional[Iterator[Document]]:
        raise NotImplementedError

    def astream(self, *args, **kwargs) -> Optional[AsyncGenerator[Document, None]]:
        raise NotImplementedError

    def run(
        self, *args, **kwargs
    ) -> Optional[Union[Document, list[Document], Iterator[Document]]]:
        raise NotImplementedError
