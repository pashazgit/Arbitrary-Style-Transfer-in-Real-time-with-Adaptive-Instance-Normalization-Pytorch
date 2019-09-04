
import torch
import torch.nn as nn


# class content_criterian(nn.Module):
#     def __init__(self):
#
#         # Run initialization for super class
#         super(content_criterian, self).__init__()
#
#     def forward(self, outf, targetf):
#         """
#         :param outf and targetf: have shape (N, c, x, y)
#         :return:
#         """
#         assert (len(outf.shape) == 4)
#         assert (len(targetf.shape) == 4)
#         assert (outf.size(0) == targetf.size(0))
#
#         return torch.norm((outf.reshape(outf.size(0), -1) - targetf.reshape(targetf.size(0), -1)), dim=1).mean()


# class style_criterian(nn.Module):
#     def __init__(self):
#
#         # Run initialization for super class
#         super(style_criterian, self).__init__()
#
#     def forward(self, outf, stylef):
#         """
#         :param outf and stylef: have shape (N, c, x, y)
#         :return:
#         """
#         assert (len(outf.shape) == 4)
#         assert (len(stylef.shape) == 4)
#         assert (outf.size(0) == stylef.size(0))
#
#         outfView = outf.view(*outf.shape[:2], -1)  # outfView.shape = (N, c, x*y)
#         outfMean = outfView.mean(-1)  # outfMean.shape = (N, c)
#         outfStd = outfView.std(-1)  # outfStd.shape = (N, c)
#
#         stylefView = stylef.view(*stylef.shape[:2], -1)  # stylefView.shape = (N, c, x*y)
#         stylefMean = stylefView.mean(-1)  # stylefMean.shape = (N, c)
#         stylefStd = stylefView.std(-1)  # stylefStd.shape = (N, c)
#
#         x1 = torch.norm((outfMean - stylefMean), dim=1)  # x1.shape = (N,)
#         x2 = torch.norm((outfStd - stylefStd), dim=1)  # x2.shape = (N,)
#
#         return (x1 + x2).mean()


def content_loss(outf, targetf):
    """
    :param outf and targetf: have shape (N, c, x, y)
    :return:
    """
    assert (len(outf.shape) == 4)
    assert (len(targetf.shape) == 4)
    assert (outf.size(0) == targetf.size(0))

    N, c, x, y = outf.shape

    return torch.sum((outf - targetf) ** 2) / (N*c*x*y)


def style_loss(outf, stylef):
    """
    :param outf and stylef: have shape (N, c, x, y)
    :return:
    """
    assert (len(outf.shape) == 4)
    assert (len(stylef.shape) == 4)
    assert (outf.size(0) == stylef.size(0))

    N, c, x, y = outf.shape

    outfView = outf.view(*outf.shape[:2], -1)  # outfView.shape = (N, c, x*y)
    outfMean = outfView.mean(-1)  # outfMean.shape = (N, c)
    outfCentered = outfView - outfMean.view(N, c, 1)  # outfCentered.shape = (N, c, x*y)
    outfStd = ((outfCentered ** 2).mean(-1) + 1e-6).sqrt()  # outfStd.shape = (N, c)

    stylefView = stylef.view(*stylef.shape[:2], -1)  # stylefView.shape = (N, c, x*y)
    stylefMean = stylefView.mean(-1)  # stylefMean.shape = (N, c)
    stylefCentered = stylefView - stylefMean.view(N, c, 1)  # outfCentered.shape = (N, c, x*y)
    stylefStd = ((stylefCentered ** 2).mean(-1) + 1e-6).sqrt()  # outfStd.shape = (N, c)

    mLoss = torch.sum((outfMean - stylefMean) ** 2) / (N*c)
    sLoss = torch.sum((outfStd - stylefStd) ** 2) / (N*c)

    return mLoss + sLoss

