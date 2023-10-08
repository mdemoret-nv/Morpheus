import asyncio
import logging

from nemo_example.cve_llm_engine import run_engine
from nemo_example.rag_pipeline import run_rag_pipeline

if __name__ == "__main__":
    from morpheus.utils.logger import configure_logging

    configure_logging(logging.DEBUG)

    # asyncio.run(run_rag_pipeline())
    asyncio.run(run_engine())

    print("Done")
