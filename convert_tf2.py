import argparse
import logging
import os
import torch
from electra_model_tf2 import TFElectraDis, TFElectraGen
from transformers import ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining, load_tf_weights_in_electra
from shutil import copyfile

logging.basicConfig(level=logging.INFO)


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path, discriminator_or_generator):
    # Initialise PyTorch model
    config = ElectraConfig.from_json_file(config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))

    if discriminator_or_generator == "discriminator":
        model = ElectraForPreTraining(config)
    elif discriminator_or_generator == "generator":
        model = ElectraForMaskedLM(config)
    else:
        raise ValueError("The discriminator_or_generator argument should be either 'discriminator' or 'generator'")

    # Load weights from tf checkpoint
    load_tf_weights_in_electra(
        model, config, tf_checkpoint_path, discriminator_or_generator=discriminator_or_generator
    )

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path",
        default='./model_pretrained/raw_model',
        type=str,
        help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--config_file_gen",
        default='./model_pretrained/config_files/config_gen.json',
        type=str,
        help="The config json file corresponding to the pre-trained model. \n"
             "This specifies the generator model architecture.",
    )
    parser.add_argument(
        "--config_file_dis",
        default='./model_pretrained/config_files/config_dis.json',
        type=str,
        help="The config json file corresponding to the pre-trained model. \n"
             "This specifies the discriminator model architecture.",
    )
    parser.add_argument(
        "--dump_path_dis",
        default='./model_pretrained/dis/',
        type=str,
        help="Path to the output discriminator model."
    )
    parser.add_argument(
        "--dump_path_gen",
        default='./model_pretrained/gen/',
        type=str,
        help="Path to the output generator model."
    )
    parser.add_argument(
        "--discriminator_or_generator",
        default=None,
        type=str,
        help="Whether to export the generator or the discriminator. Should be a string, either 'discriminator' or "
             "'generator'.",
    )
    args = parser.parse_args()

    # Convert tf1 generator to torch generator
    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path,
        args.config_file_gen,
        os.path.join(args.dump_path_gen, 'pytorch_model.bin'),
        "generator"
    )
    # Convert tf1 discriminator to torch discriminator
    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path,
        args.config_file_dis,
        os.path.join(args.dump_path_dis, 'pytorch_model.bin'),
        "discriminator"
    )

    # copy config file
    copyfile(args.config_file_gen, os.path.join(args.dump_path_gen, 'config.json'))
    copyfile(args.config_file_dis, os.path.join(args.dump_path_dis, 'config.json'))

    # Convert torch generator tf2 generator
    tf_generator = TFElectraGen.from_pretrained(args.dump_path_gen, from_pt=True)
    tf_generator.save_pretrained(args.dump_path_gen)

    # Convert torch discriminator tf2 discriminator
    tf_discriminator = TFElectraDis.from_pretrained(args.dump_path_dis, from_pt=True)
    tf_discriminator.save_pretrained(args.dump_path_dis)

    # Remove pytorch file
    os.remove(os.path.join(args.dump_path_gen, 'pytorch_model.bin'))
    os.remove(os.path.join(args.dump_path_dis, 'pytorch_model.bin'))
