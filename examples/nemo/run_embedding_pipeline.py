import asyncio
import logging

from nemo_example.embedding_pipeline import cli

if __name__ == "__main__":
    from morpheus.utils.logger import configure_logging

    configure_logging(logging.DEBUG)

    cli()

    print("Done")
