import logging
import os
import src.config as config
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.getLevelNamesMapping().get(os.environ.get("COMP597_LOG_LEVEL", config.default_logging_config.level), logging.WARNING),
    format=config.default_logging_config.format,
    datefmt=config.default_logging_config.datefmt,
    style=config.default_logging_config.style,
)

from typing import Any, Dict, Optional, Tuple
import argparse
import gc
import src.data as data
import src.models as models
import src.trainer as trainer

def setup_logging(conf : config.Config) -> None:
    logging.basicConfig(
        filename=conf.logging.filename,
        filemode=conf.logging.filemode,
        format=conf.logging.format,
        datefmt=conf.logging.datefmt,
        style=conf.logging.style,
        level=conf.logging.level,
        force=True,
    )

def process_conf(conf : config.Config) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
    dataset = data.load_data(conf)
    logger.debug(f"Dataset loaded with {len(dataset)} samples.")

    return models.model_factory(conf, dataset)

def get_conf() -> config.NewConfig:
    parser = argparse.ArgumentParser()

    conf = config.Config()
    conf.add_arguments(parser)
    
    args, _ = parser.parse_known_args()
    conf.parse_arguments(args)
    return conf

def main():
    conf = get_conf()
    logger.debug(f"Configuration: {conf}")
    logger.info(f"available models: {models.get_available_models()}")
    model_trainer, model_kwargs = process_conf(conf)
    model_trainer.train(model_kwargs)

    # This forces garbage collection at process exit. It ensure proper closing of resources.
    del conf
    del model_kwargs
    del model_trainer

if __name__ == "__main__":
    main()
    gc.collect()

