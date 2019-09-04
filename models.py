
# Copyright (C), Visual Computing Group @ University of Victoria.


import torch
from torch import nn
import os
import torchvision


class encoder(nn.Module):

    def __init__(self, config):

        # Run initialization for super class
        super(encoder, self).__init__()

        # Create/save vgg19 instance

        # vgg19 = torchvision.models.vgg19(pretrained=True)
        # vgg19_file = os.path.join(config.modelDir, 'vgg19_model.pth')
        # torch.save({'model': vgg19.state_dict()}, vgg19_file)
        # for param in vgg19.parameters():
        #     param.requires_grad = False
        vgg19 = torchvision.models.vgg19(pretrained=False)
        load_res = torch.load(os.path.join(config.modelDir, 'vgg19_model.pth'))
        vgg19.load_state_dict(load_res["model"])
        for param in vgg19.parameters():
            param.requires_grad = False

        self.noutchannels = vgg19.features[(19)].out_channels

        self.layers = nn.Sequential()
        self.layers1 = nn.Sequential()
        self.layers2 = nn.Sequential()
        self.layers3 = nn.Sequential()
        self.layers4 = nn.Sequential()
        for _i in range(21):
            self.layers.add_module('module_{}'.format(_i), vgg19.features[(_i)])
            if _i <= 1:
                self.layers1.add_module('module_{}'.format(_i), vgg19.features[(_i)])
            if 2 <= _i <= 6:
                self.layers2.add_module('module_{}'.format(_i), vgg19.features[(_i)])
            if 7 <= _i <= 11:
                self.layers3.add_module('module_{}'.format(_i), vgg19.features[(_i)])
            if 12 <= _i <= 20:
                self.layers4.add_module('module_{}'.format(_i), vgg19.features[(_i)])

    def forward(self, x, multiple=False):

        if multiple:
            assert (len(x.shape) == 4)
            x1 = self.layers1(x)
            x2 = self.layers2(x1)
            x3 = self.layers3(x2)
            x4 = self.layers4(x3)
            return x1, x2, x3, x4
        else:
            assert(len(x.shape) == 4)
            x = self.layers(x)
            return x


# adaptive instance normalization module
class adain(nn.Module):

    def __init__(self, noutchannels, disabled=False):

        # Run initialization for super class
        super(adain, self).__init__()

        self.disabled = disabled
        self.noutchannels = noutchannels

    def forward(self, contentf, stylef):
        """
        :param content and style: have shape (N, c, x, y)
        :return:
        """
        if self.disabled:
            self.output = contentf
            return self.output

        assert (len(contentf.shape) == 4)
        assert (len(stylef.shape) == 4)
        assert (contentf.size(0) == stylef.size(0))
        assert (contentf.size(1) == self.noutchannels)
        assert (stylef.size(1) == self.noutchannels)

        N, c, x, y = contentf.shape

        contentView = contentf.view(*contentf.shape[:2], -1)  # contentView.shape = (N, c, x*y)
        contentMean = contentView.mean(-1)  # contentMean.shape = (N, c)
        # contentStd = (contentView.var(-1) + 1e-6).sqrt()  # contentStd.shape = (N, c)
        contentCentered = contentView - contentMean.view(N, c, 1)  # contentCentred.shape = (N, c, x*y)
        contentStd = ((contentCentered ** 2).mean(-1) + 1e-6).sqrt()  # contentStd.shape = (N, c)
        contentMean = contentMean.view(*contentMean.shape[:2],
                                       *((len(contentf.shape) - 2) * [1]))  # contentMean.shape = (N, c, 1, 1)
        contentStd = contentStd.view(*contentStd.shape[:2],
                                     *((len(contentf.shape) - 2) * [1]))  # contentStd.shape = (N, c, 1, 1)

        styleView = stylef.view(*stylef.shape[:2], -1)  # styleView.shape = (N, c, x*y)
        styleMean = styleView.mean(-1)  # styleMean.shape = (N, c)
        # styleStd = (styleView.var(-1) + 1e-6).sqrt()  # styleStd.shape = (N, c)
        styleCentered = styleView - styleMean.view(N, c, 1)  # styleCentred.shape = (N, c, x*y)
        styleStd = ((styleCentered ** 2).mean(-1) + 1e-6).sqrt()  # styleStd.shape = (N, c)
        styleMean = styleMean.view(*styleMean.shape[:2],
                                       *((len(stylef.shape) - 2) * [1]))  # styleMean.shape = (N, c, 1, 1)
        styleStd = styleStd.view(*styleStd.shape[:2],
                                     *((len(stylef.shape) - 2) * [1]))  # styleStd.shape = (N, c, 1, 1)

        out = ((contentf - contentMean) / contentStd) * styleStd + styleMean

        return out


class Identity(nn.Module):
    def __init__(self):

        # Run initialization for super class
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class decoder(nn.Module):

    def __init__(self, config, enc):

        # Run initialization for super class
        super(decoder, self).__init__()

        Activation = getattr(nn, config.activation)

        self.layers = nn.Sequential()
        _i = 0
        for layer in reversed(enc.layers):
            if type(layer) is nn.Conv2d:
                ninchannels, noutchannels = layer.out_channels, layer.in_channels
                self.layers.add_module('module_{}'.format(_i), nn.ReflectionPad2d(1))
                self.layers.add_module('module_{}'.format(_i+1), nn.Conv2d(ninchannels, noutchannels, 3, 1))
                self.layers.add_module('module_{}'.format(_i+2), Activation())
                _i += 3
            elif type(layer) is nn.MaxPool2d:
                self.layers.add_module('module_{}'.format(_i), nn.UpsamplingNearest2d(scale_factor=2))
                _i += 1
        self.layers[(-1)] = Identity()

        # print('decoder')
        # print(self)

    def forward(self, x):

        assert (len(x.shape) == 4)
        x = self.layers(x)

        return x


