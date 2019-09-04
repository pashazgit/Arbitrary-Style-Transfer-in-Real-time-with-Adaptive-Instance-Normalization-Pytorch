
# Copyright (C), Visual Computing Group @ University of Victoria.

import argparse


# ----------------------------------------
# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# ----------------------------------------
# Arguments for the main program
main_arg = add_argument_group("Main")


main_arg.add_argument("--mode", type=str,
                      default="train",
                      choices=["train", "test"],
                      help="Run mode")

# ----------------------------------------
# Arguments for the main program
verb_arg = add_argument_group("Verbosity")


main_arg.add_argument("--printDetails", type=str2bool,
                      default="True",
                      help="If true, print style loss at individual layers")

# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")


train_arg.add_argument("--modelDir", type=str,
                       default="/home/pasha/Styletransfer/models",
                       help="Directory containing models")

train_arg.add_argument("--contentDir", type=str,
                       default="/scratch/pasha/data/coco",
                       help="Directory containing content images for training")

train_arg.add_argument("--contentImage", type=str,
                       default="./data/content/chicago_cropped.jpg",
                       choices=["avril_cropped.jpg",
                                "chicago_cropped.jpg",
                                "lenna_cropped.jpg",
                                "modern_cropped.jpg",
                                "sailboat_cropped.jpg"],
                       help="Containing content images for testing")

train_arg.add_argument("--styleDir", type=str,
                       default="/scratch/pasha/data/wikiart",
                       help="Directory containing style images for training")

train_arg.add_argument("--styleImage", type=str,
                       default="./data/style/ashville_cropped.jpg",
                       choices=["ashville_cropped.jpg",
                                "en_campo_gris_cropped.jpg",
                                "goeritz_cropped.jpg",
                                "sketch_cropped.png"],
                       help="Containing style images for testing")

train_arg.add_argument("--styleInterpWeights", type=str,
                       default="[0.2, 0.8]",
                       help="Weights to interpolate between several styles")

train_arg.add_argument("--interpolate", type=str2bool,
                       default=False,
                       help="If true, the decoder is also trained to reconstruct style images")

train_arg.add_argument("--log_dir", type=str,
                       default="/home/pasha/Styletransfer/logs",
                       help="Directory to save logs and current model")

train_arg.add_argument("--save_dir", type=str,
                       default="",
                       help="Directory to save the best model")

train_arg.add_argument("--contentInfoFile", type=str,
                       default="",
                       help="File containing info about content images for training")

train_arg.add_argument("--styleInfoFile", type=str,
                       default="",
                       help="File containing info about style images for training")

train_arg.add_argument("--finalSize", type=int,
                       default=256,
                       help='Size of images used for training')

train_arg.add_argument("--contentSize", type=int,
                       default=512,
                       help='Size of content images before cropping, keep original size if set to 0')

train_arg.add_argument("--styleSize", type=int,
                       default=512,
                       help='Size of style images before cropping, keep original size if set to 0')

train_arg.add_argument("--crop", type=str2bool,
                       default=True,
                       help='If true, crop training images')

train_arg.add_argument("--learning_rate", type=float,
                       default=1e-4,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--learning_rate_decay", type=float,
                       default=0.5,
                       help="Learning rate decay")

train_arg.add_argument("--momentum", type=float,
                       default=0.9,
                       help="Momentum")

train_arg.add_argument("--weightDecay", type=float,
                       default=0,
                       help="Weight decay")

train_arg.add_argument("--maxIter", type=float,
                       default=160000,
                       help="Maximum number of iterations")

train_arg.add_argument("--targetContentLayer", type=str,
                       default='relu4_1',
                       choices=["relu4_1", "conv4_1"],
                       help="Target content layer used to compute the loss")

train_arg.add_argument("--targetStyleLayers", type=str,
                       default='relu1_1,relu2_1,relu3_1,relu4_1',
                       help="Target style layers used to compute the loss")

train_arg.add_argument("--tvWeight", type=float,
                       default=0,
                       help="Weight of TV loss")

train_arg.add_argument("--styleWeight", type=float,
                       default=4,
                       help="Weight of style loss")

train_arg.add_argument("--alpha", type=float,
                       default=1,
                       help="Content style trade off weight")

train_arg.add_argument("--contentWeight", type=float,
                       default=1,
                       help="Weight of content loss")

train_arg.add_argument("--reconStyle", type=str2bool,
                       default=False,
                       help="If true, the decoder is also trained to reconstruct style images")

train_arg.add_argument("--normalize", type=str2bool,
                       default=False,
                       help="If true, gradients at the loss function are normalized")

train_arg.add_argument("--batch_size", type=int,
                       default=8,
                       help="Size of each training batch")

train_arg.add_argument("--num_epoch", type=int,
                       default=16,
                       help="Number of epochs to train")

train_arg.add_argument("--val_intv", type=int,
                       default=1000,
                       help="Validation interval")

train_arg.add_argument("--rep_intv", type=int,
                       default=1000,
                       help="Report interval")

train_arg.add_argument("--resume", type=str2bool,
                       default=False,
                       help="If true, resume training from the last checkpoint")
# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")

model_arg.add_argument("--l2_reg", type=float,
                       default=1e-4,
                       help="L2 Regularization strength")

model_arg.add_argument("--num_unit", type=int,
                       default=128,
                       help="Number of neurons in the hidden layer")

model_arg.add_argument("--num_hidden", type=int,
                       default=3,
                       help="Number of hidden layers")

model_arg.add_argument("--nchannel_base", type=int,
                       default=8,
                       help="Base number of channels")

model_arg.add_argument("--ksize", type=int,
                       default=3,
                       help="Size of the convolution kernel")

model_arg.add_argument("--num_conv_outer", type=int,
                       default=3,
                       help="Number of outer blocks")

model_arg.add_argument("--num_conv_inner", type=int,
                       default=1,
                       help="Number of convolution in each block")

model_arg.add_argument("--num_class", type=int,
                       default=10,
                       help="Number of classes in the dataset")

model_arg.add_argument("--conv2d", type=str,
                       default="torch",
                       help="Convolution type")

model_arg.add_argument("--pool2d", type=str,
                       default="MaxPool2d",
                       help="Pooling type")

model_arg.add_argument("--activation", type=str,
                       default="ReLU",
                       help="Activation type")


def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()

