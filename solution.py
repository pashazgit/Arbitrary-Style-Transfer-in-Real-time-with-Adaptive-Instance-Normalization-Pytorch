
# Copyright (C), Visual Computing Group @ University of Victoria.


import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm, trange
from config import get_config, print_usage
from tensorboardX import SummaryWriter
import torchvision
from skimage import io, transform
from models import encoder, decoder, adain
from datawrapper import trainloader, testloader
from criterions import content_loss, style_loss
#from criterians import content_criterian, style_criterian
import pandas as pd
from PIL import Image
from torchvision import transforms


def train(config):
    """Training process.

    """
    # create model instances:
    enc = encoder(config)
    ain = adain(enc.noutchannels)
    dec = decoder(config, enc)

    # Move enc to gpu if cuda is available
    if torch.cuda.is_available():
        enc = enc.cuda()
        ain = ain.cuda()
        dec = dec.cuda()

    # set to train
    enc.eval()
    dec.train()

    # # Create loss objects
    # content_loss = content_criterian()
    # style_loss = style_criterian()
    # if torch.cuda.is_available():
    #     content_loss = content_loss.cuda()
    #     style_loss = style_loss.cuda()

    # Create log directory if it does not exist
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    # Create summary writer
    tr_writer = SummaryWriter(log_dir=os.path.join(config.log_dir, "train"))

    # Initialize training
    iter_idx = -1  # make counter start at zero

    # loss_vec = []

    # Training loop
    for epoch in range(config.num_epoch):
        # # For each iteration
        prefix = "Training Epoch {:3d}: ".format(epoch)

        # lr = config.learning_rate/(1 + config.learning_rate_decay * epoch)
        lr = config.learning_rate

        # Create optimizer
        optimizer = optim.Adam(dec.parameters(), lr=lr)

        for (content, style) in tqdm(trainloader(config), desc=prefix):
            # Counter
            iter_idx += 1

            # Send data to GPU if we have one
            if torch.cuda.is_available():
                content = content.cuda()
                style = style.cuda()

            # Apply the model to obtain features (forward pass)
            contentf = enc(content, multiple=False)
            stylef = enc(style, multiple=False)
            targetf = ain(contentf, stylef)

            g = dec(targetf)

            outf1, outf2, outf3, outf4 = enc(g, multiple=True)
            stylef1, stylef2, stylef3, stylef4 = enc(style, multiple=True)

            # Compute the loss
            loss_c = content_loss(outf4, targetf)
            loss_s = style_loss(outf1, stylef1) + style_loss(outf2, stylef2) + style_loss(outf3, stylef3) + style_loss(outf4, stylef4)
            loss = loss_c + config.styleWeight * loss_s
            # Compute gradients
            loss.backward()
            # Update parameters
            optimizer.step()
            # Zero the parameter gradients in the optimizer
            optimizer.zero_grad()

            # Monitor results every report interval
            if iter_idx % config.rep_intv == 0:
                # List to contain all losses and accuracies for all the training batches
                loss_c_test = []
                loss_s_test = []
                loss_test = []
                # Set model for evaluation
                dec = dec.eval()
                for (content, style) in tqdm(testloader(config)):
                    # Send data to GPU if we have one
                    if torch.cuda.is_available():
                        content = content.cuda()
                        style = style.cuda()

                    # Apply forward pass to compute the losses for each of the test batches
                    with torch.no_grad():
                        # Apply the model to obtain features (forward pass)
                        contentf = enc(content, multiple=False)
                        stylef = enc(style, multiple=False)
                        targetf = ain(contentf, stylef)

                        g = dec(targetf)

                        outf1, outf2, outf3, outf4 = enc(g, multiple=True)
                        stylef1, stylef2, stylef3, stylef4 = enc(style, multiple=True)

                        # Compute the loss
                        loss_c_temp = content_loss(outf4, targetf)
                        loss_c_test += [loss_c_temp.cpu().numpy()]
                        loss_s_temp = style_loss(outf1, stylef1) + style_loss(outf2, stylef2) + style_loss(outf3, stylef3) + style_loss(outf4, stylef4)
                        loss_s_test += [loss_s_temp.cpu().numpy()]
                        loss_temp = loss_c_temp + config.styleWeight * loss_s_temp
                        loss_test += [loss_temp.cpu().numpy()]

                # Set model back for training
                dec = dec.train()

                # Take average
                loss_c_test = np.mean(loss_c_test)
                loss_s_test = np.mean(loss_s_test)
                loss_test = np.mean(loss_test)

                # Write loss to tensorboard, using keywords `loss`
                tr_writer.add_scalar("loss_content_test", loss_c_test, global_step=iter_idx)
                tr_writer.add_scalar("loss_style_test", loss_s_test, global_step=iter_idx)
                tr_writer.add_scalar("loss_test", loss_test, global_step=iter_idx)

                torch.save({"model": dec.state_dict()}, os.path.join(config.modelDir, "dec_model.pth"))

                # loss_vec += [loss.item()]

    # # Draw
    # plt.figure()
    # plt.plot(np.arange(len(loss_vec)), np.asarray(loss_vec))
    # plt.show()


def test(config):
    """Test routine"""

    # create model instances:
    enc = encoder(config)
    ain = adain(enc.noutchannels)
    dec = decoder(config, enc)
    load_res = torch.load(os.path.join(config.modelDir, 'dec_model.pth'), map_location="cpu")
    dec.load_state_dict(load_res["model"])

    # Move enc to gpu if cuda is available
    if torch.cuda.is_available():
        enc = enc.cuda()
        ain = ain.cuda()
        dec = dec.cuda()

    # set to eval
    enc.eval()
    dec.eval()

    tt = transforms.ToTensor()
    tp = transforms.ToPILImage()

    fig = plt.figure()

    if not config.interpolate:

        content = Image.open(config.contentImage)
        assert (np.asarray(content).shape[2] == 3)
        content = tt(content)
        imgplot = plt.imshow(content.permute(1, 2, 0))
        plt.show()
        content = content.reshape(1, *content.shape)

        style = Image.open(config.styleImage)
        assert (np.asarray(style).shape[2] == 3)
        style = tt(style)
        imgplot = plt.imshow(style.permute(1, 2, 0))
        plt.show()
        style = style.reshape(1, *style.shape)

        if torch.cuda.is_available():
            content = content.cuda()
            style = style.cuda()

        with torch.no_grad():
            contentf = enc(content, multiple=False)
            stylef = enc(style, multiple=False)
            targetf = ain(contentf, stylef)
            g = dec((1 - config.alpha) * contentf + config.alpha * targetf).squeeze()

        imgplot = plt.imshow(g.permute(1, 2, 0))
        plt.show()

    if config.interpolate:
        targetf = 0

        content = Image.open(config.contentImage)
        assert (np.asarray(content).shape[2] == 3)
        content = tt(content)
        imgplot = plt.imshow(content.permute(1, 2, 0))
        plt.show()
        content = content.reshape(1, *content.shape)

        weights = list(map(float, config.styleInterpWeights.strip('[]').split(',')))

        im_names = config.styleImage.split(',')

        for i, im_name in enumerate(im_names):
            style = Image.open(im_name)
            assert (np.asarray(style).shape[2] == 3)
            style = tt(style)
            imgplot = plt.imshow(style.permute(1, 2, 0))
            plt.show()
            style = style.reshape(1, *style.shape)

            if torch.cuda.is_available():
                content = content.cuda()
                style = style.cuda()

            with torch.no_grad():
                contentf = enc(content, multiple=False)
                stylef = enc(style, multiple=False)
                targetf += weights[i] * ain(contentf, stylef)

        with torch.no_grad():
            g = dec(targetf).squeeze()

            imgplot = plt.imshow(g.permute(1, 2, 0))
            plt.show()


def main(config):
    """The main function."""

    if config.mode == "train":
        train(config)
    elif config.mode == "test":
        test(config)
    else:
        raise ValueError("Unknown run mode \"{}\"".format(config.mode))


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)

