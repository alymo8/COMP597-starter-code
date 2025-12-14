from typing import Any, Dict, Optional, Tuple
import argparse
import gc
import src.data as data
import src.models as models
import src.trainer as trainer
import src.config as config

def process_conf(conf : config.Config) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
    dataset = data.load_data(conf)
    print(f"Dataset loaded with {len(dataset)} samples.")

    return models.model_factory(conf, dataset)

def get_conf() -> config.NewConfig:
    parser = argparse.ArgumentParser()

    conf = config.NewConfig()
    conf.add_arguments(parser)
    
    args, _ = parser.parse_known_args()
    conf.parse_arguments(args)
    return conf

def main():
    conf = get_conf()
    # conf = config.Config(args)
    print(f"available models: {models.get_available_models()}")
    model_trainer, model_kwargs = process_conf(conf)
    # print(conf)
    model_trainer.train(model_kwargs)

    # This forces garbage collection at process exit. It ensure proper closing of resources.
    del conf
    del model_kwargs
    del model_trainer

if __name__ == "__main__":
    main()
    gc.collect()

