import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BatchSizeMixin:
    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size: Optional[int]):
        if batch_size is None:
            logger.debug(
                f"'batch_size' was `None`. Using default batch size (={self.DEFAULT_BATCH_SIZE})"
            )
            batch_size = self.DEFAULT_BATCH_SIZE

        if batch_size < 1:
            raise ValueError(f"'batch_size' cannot be < 1! got: {batch_size}")

        self.__batch_size = batch_size
