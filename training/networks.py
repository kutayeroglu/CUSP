# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma

from training.tools import getGaussianKernel

from itertools import chain, zip_longest
import re

# ----------------------------------------------------------------------------


@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


# ----------------------------------------------------------------------------


@misc.profiled_function
def modulated_conv2d(
    x,  # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,  # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,  # Modulation coefficients of shape [batch_size, in_channels].
    noise=None,  # Optional noise tensor to add to the output activations.
    up=1,  # Integer upsampling factor.
    down=1,  # Integer downsampling factor.
    padding=0,  # Padding with respect to the upsampled image.
    resample_filter=None,  # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate=True,  # Apply weight demodulation?
    flip_weight=True,  # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv=True,  # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw])  # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels])  # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (
            1
            / np.sqrt(in_channels * kh * kw)
            / weight.norm(float("inf"), dim=[1, 2, 3], keepdim=True)
        )  # max_Ikk
        styles = styles / styles.norm(float("inf"), dim=1, keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=weight.to(x.dtype),
            f=resample_filter,
            up=up,
            down=down,
            padding=padding,
            flip_weight=flip_weight,
        )
        if demodulate and noise is not None:
            x = fma.fma(
                x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype)
            )
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings():  # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(
        x=x,
        w=w.to(x.dtype),
        f=resample_filter,
        up=up,
        down=down,
        padding=padding,
        groups=batch_size,
        flip_weight=flip_weight,
    )
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(
        self,
        in_features,  # Number of input features.
        out_features,  # Number of output features.
        bias=True,  # Apply additive bias before the activation function?
        activation="linear",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=1,  # Learning rate multiplier.
        bias_init=0,  # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) / lr_multiplier
        )
        self.bias = (
            torch.nn.Parameter(torch.full([out_features], np.float32(bias_init)))
            if bias
            else None
        )
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == "linear" and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        kernel_size,  # Width and height of the convolution kernel.
        bias=True,  # Apply additive bias before the activation function?
        activation="linear",  # Activation function: 'relu', 'lrelu', etc.
        up=1,  # Integer upsampling factor.
        down=1,  # Integer downsampling factor.
        resample_filter=None,  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
        channels_last=False,  # Expect the input to have memory_format=channels_last?
        trainable=True,  # Update the weights of this layer during training?
    ):
        super().__init__()
        if resample_filter is None:
            resample_filter = [1, 3, 3, 1]
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(
            memory_format=memory_format
        )
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer("weight", weight)
            if bias is not None:
                self.register_buffer("bias", bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = self.up == 1  # slightly faster
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=w.to(x.dtype),
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
            flip_weight=flip_weight,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,  # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,  # Intermediate latent (W) dimensionality.
        num_ws,  # Number of intermediate latents to output, None = do not broadcast.
        num_layers=8,  # Number of mapping layers.
        embed_features=None,  # Label embedding dimensionality, None = same as w_dim.
        layer_features=None,  # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=0.01,  # Learning rate multiplier for the mapping layers.
        w_avg_beta=0.995,  # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = (
            [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]
        )

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(
                in_features,
                out_features,
                activation=activation,
                lr_multiplier=lr_multiplier,
            )
            setattr(self, f"fc{idx}", layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer("w_avg", torch.zeros([w_dim]))

    def forward(
        self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False
    ):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function("input"):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f"fc{idx}")
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function("update_w_avg"):
                self.w_avg.copy_(
                    x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta)
                )

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function("broadcast"):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function("truncate"):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(
                        x[:, :truncation_cutoff], truncation_psi
                    )
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        w_dim,  # Intermediate latent (W) dimensionality.
        resolution,  # Resolution of this layer.
        kernel_size=3,  # Convolution kernel size.
        up=1,  # Integer upsampling factor.
        use_noise=True,  # Enable noise input?
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        resample_filter=[
            1,
            3,
            3,
            1,
        ],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last=False,  # Use channels_last format for the weights?
        bias=True,  # Use bias
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias=bias, bias_init=1)
        memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(
                memory_format=memory_format
            )
        )
        if use_noise:
            self.register_buffer("noise_const", torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels])) if bias else None

    def forward(self, x, w, noise_mode="random", fused_modconv=True, gain=1):
        assert noise_mode in ["random", "const", "none"]
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == "random":
            noise = (
                torch.randn(
                    [x.shape[0], 1, self.resolution, self.resolution], device=x.device
                )
                * self.noise_strength
            )
        if self.use_noise and noise_mode == "const":
            noise = self.noise_const * self.noise_strength

        flip_weight = self.up == 1  # slightly faster
        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            noise=noise,
            up=self.up,
            padding=self.padding,
            resample_filter=self.resample_filter,
            flip_weight=flip_weight,
            fused_modconv=fused_modconv,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(
            x,
            b=self.bias.to(x.dtype) if self.bias is not None else self.bias,
            act=self.activation,
            gain=act_gain,
            clamp=act_clamp,
        )
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        kernel_size=1,
        conv_clamp=None,
        channels_last=False,
    ):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(
                memory_format=memory_format
            )
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            demodulate=False,
            fused_modconv=fused_modconv,
        )
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels, 0 = first block.
        out_channels,  # Number of output channels.
        w_dim,  # Intermediate latent (W) dimensionality.
        resolution,  # Resolution of this block.
        img_channels,  # Number of output color channels.
        is_last,  # Is this the last block?
        architecture="skip",  # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter=[
            1,
            3,
            3,
            1,
        ],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16=False,  # Use FP16 for this block?
        fp16_channels_last=False,  # Use channels-last memory format with FP16?
        **layer_kwargs,  # Arguments for SynthesisLayer.
    ):
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = use_fp16 and fp16_channels_last
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(
                torch.randn([out_channels, resolution, resolution])
            )

        if in_channels != 0:
            self.conv0 = SynthesisLayer(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=resolution,
                up=2,
                resample_filter=resample_filter,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
                **layer_kwargs,
            )
            self.num_conv += 1

        self.conv1 = SynthesisLayer(
            out_channels,
            out_channels,
            w_dim=w_dim,
            resolution=resolution,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
            **layer_kwargs,
        )
        self.num_conv += 1

        if is_last or architecture == "skip":
            # torgb always has bias
            self.torgb = ToRGBLayer(
                out_channels,
                img_channels,
                w_dim=w_dim,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
            )
            self.num_torgb += 1

        if in_channels != 0 and architecture == "resnet":
            self.skip = Conv2dLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                up=2,
                resample_filter=resample_filter,
                channels_last=self.channels_last,
            )

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = (
            torch.channels_last
            if self.channels_last and not force_fp32
            else torch.contiguous_format
        )
        if fused_modconv is None:
            with (
                misc.suppress_tracer_warnings()
            ):  # this value will be treated as a constant
                fused_modconv = (not self.training) and (
                    dtype == torch.float32 or int(x.shape[0]) == 1
                )

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(
                x, [None, self.in_channels, self.resolution // 2, self.resolution // 2]
            )
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == "resnet":
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(
                x,
                next(w_iter),
                fused_modconv=fused_modconv,
                gain=np.sqrt(0.5),
                **layer_kwargs,
            )
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(
                img,
                [None, self.img_channels, self.resolution // 2, self.resolution // 2],
            )
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == "skip":
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img


# ----------------------------------------------------------------------------


@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(
        self,
        w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # Output image resolution.
        img_channels,  # Number of color channels.
        channel_base=32768,  # Overall multiplier for the number of channels.
        channel_max=512,  # Maximum number of channels in any layer.
        num_fp16_res=0,  # Use FP16 for the N highest resolutions.
        **block_kwargs,  # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2**i for i in range(2, img_resolution_log2 + 1)]
        channels_dict = {
            res: min(channel_base // res, channel_max) for res in self.block_resolutions
        }
        fp16_resolution = max(2 ** (img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = res >= fp16_resolution
            is_last = res == self.img_resolution
            block = SynthesisBlock(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=res,
                img_channels=img_channels,
                is_last=is_last,
                use_fp16=use_fp16,
                **block_kwargs,
            )
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f"b{res}", block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function("split_ws"):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f"b{res}")
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f"b{res}")
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img


# ----------------------------------------------------------------------------


@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality.
        c_dim,  # Conditioning label (C) dimensionality.
        w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # Output resolution.
        img_channels,  # Number of output color channels.
        mapping_kwargs={},  # Arguments for MappingNetwork.
        synthesis_kwargs={},  # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            **synthesis_kwargs,
        )
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(
            z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs
        )

    def forward(
        self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs
    ):
        ws = self.mapping(
            z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff
        )
        img = self.synthesis(ws, **synthesis_kwargs)
        return img


# ----------------------------------------------------------------------------


@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels, 0 = first block.
        tmp_channels,  # Number of intermediate channels.
        out_channels,  # Number of output channels.
        resolution,  # Resolution of this block.
        img_channels,  # Number of input color channels.
        first_layer_idx,  # Index of the first layer.
        architecture="resnet",  # Architecture: 'orig', 'skip', 'resnet'.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        resample_filter=[
            1,
            3,
            3,
            1,
        ],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16=False,  # Use FP16 for this block?
        fp16_channels_last=False,  # Use channels-last memory format with FP16?
        freeze_layers=0,  # Freeze-D: Number of layers to freeze.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = use_fp16 and fp16_channels_last
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0

        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = layer_idx >= freeze_layers
                self.num_layers += 1
                yield trainable

        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == "skip":
            self.fromrgb = Conv2dLayer(
                img_channels,
                tmp_channels,
                kernel_size=1,
                activation=activation,
                trainable=next(trainable_iter),
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
            )

        self.conv0 = Conv2dLayer(
            tmp_channels,
            tmp_channels,
            kernel_size=3,
            activation=activation,
            trainable=next(trainable_iter),
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        self.conv1 = Conv2dLayer(
            tmp_channels,
            out_channels,
            kernel_size=3,
            activation=activation,
            down=2,
            trainable=next(trainable_iter),
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        if architecture == "resnet":
            self.skip = Conv2dLayer(
                tmp_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                down=2,
                trainable=next(trainable_iter),
                resample_filter=resample_filter,
                channels_last=self.channels_last,
            )

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = (
            torch.channels_last
            if self.channels_last and not force_fp32
            else torch.contiguous_format
        )

        # Input.
        if x is not None:
            misc.assert_shape(
                x, [None, self.in_channels, self.resolution, self.resolution]
            )
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == "skip":
            misc.assert_shape(
                img, [None, self.img_channels, self.resolution, self.resolution]
            )
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = (
                upfirdn2d.downsample2d(img, self.resample_filter)
                if self.architecture == "skip"
                else None
            )

        # Main layers.
        if self.architecture == "resnet":
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img


# ----------------------------------------------------------------------------


@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with (
            misc.suppress_tracer_warnings()
        ):  # as_tensor results are registered as constants
            G = (
                torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N))
                if self.group_size is not None
                else N
            )
        F = self.num_channels
        c = C // F

        y = x.reshape(
            G, -1, F, c, H, W
        )  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        cmap_dim,  # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,  # Resolution of this block.
        img_channels,  # Number of input color channels.
        architecture="resnet",  # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels=1,  # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ["orig", "skip", "resnet"]
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == "skip":
            self.fromrgb = Conv2dLayer(
                img_channels, in_channels, kernel_size=1, activation=activation
            )
        self.mbstd = (
            MinibatchStdLayer(
                group_size=mbstd_group_size, num_channels=mbstd_num_channels
            )
            if mbstd_num_channels > 0
            else None
        )
        self.conv = Conv2dLayer(
            in_channels + mbstd_num_channels,
            in_channels,
            kernel_size=3,
            activation=activation,
            conv_clamp=conv_clamp,
        )
        self.fc = FullyConnectedLayer(
            in_channels * (resolution**2), in_channels, activation=activation
        )
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(
            x, [None, self.in_channels, self.resolution, self.resolution]
        )  # [NCHW]
        _ = force_fp32  # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == "skip":
            misc.assert_shape(
                img, [None, self.img_channels, self.resolution, self.resolution]
            )
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x


# ----------------------------------------------------------------------------

from training.tools import identity_f


@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(
        self,
        c_dim,  # Conditioning label (C) dimensionality.
        img_resolution,  # Input resolution.
        img_channels,  # Number of input color channels.
        architecture="resnet",  # Architecture: 'orig', 'skip', 'resnet'.
        channel_base=32768,  # Overall multiplier for the number of channels.
        channel_max=512,  # Maximum number of channels in any layer.
        num_fp16_res=0,  # Use FP16 for the N highest resolutions.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.
        block_kwargs={},  # Arguments for DiscriminatorBlock.
        mapping_kwargs={},  # Arguments for MappingNetwork.
        epilogue_kwargs={},  # Arguments for DiscriminatorEpilogue.
        cmap_pre=identity_f,  # Class preprocess mapping
        finetune=None,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.cmap_pre = cmap_pre
        self.img_resolution = img_resolution

        ####
        # Parse resolutions
        ####
        i = 0
        while img_resolution % (2 ** (i + 1)) == 0:
            i += 1
        i = int(min(i, np.log2(img_resolution / 4)))
        self.min_resolution = int(img_resolution / (2**i))
        ####

        img_resolution_log2 = int(np.log2(img_resolution / self.min_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [
            self.min_resolution * (2**i) for i in range(img_resolution_log2, 0, -1)
        ]
        channels_dict = {
            res: min(channel_base // res, channel_max)
            for res in self.block_resolutions + [self.min_resolution]
        }
        fp16_resolution = max(2 ** (img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[self.min_resolution]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(
            img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp
        )
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = res >= fp16_resolution
            block = DiscriminatorBlock(
                in_channels,
                tmp_channels,
                out_channels,
                resolution=res,
                first_layer_idx=cur_layer_idx,
                use_fp16=use_fp16,
                **block_kwargs,
                **common_kwargs,
            )
            setattr(self, f"b{res}", block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(
                z_dim=0,
                c_dim=c_dim,
                w_dim=cmap_dim,
                num_ws=None,
                w_avg_beta=None,
                **mapping_kwargs,
            )
        self.b4 = DiscriminatorEpilogue(
            channels_dict[self.min_resolution],
            cmap_dim=cmap_dim,
            resolution=self.min_resolution,
            **epilogue_kwargs,
            **common_kwargs,
        )
        self.finetune = finetune

    def requires_grad_(self, requires_grad: bool = True):
        if self.finetune:
            for p in self.parameters():
                p.requires_grad = requires_grad
            regex = r"b[0-9]+"
            sorted_modules = sorted(
                filter(
                    lambda x: re.search(regex, x[0]) is not None, self.named_children()
                ),
                key=lambda x: -int(x[0][1:]),
            )[: self.finetune]
            for _, m in sorted_modules:
                for p in m.parameters():
                    p.requires_grad = requires_grad

        else:
            for p in self.parameters():
                p.requires_grad = requires_grad
        return self

    def forward(self, img, c, **block_kwargs):
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f"b{res}")
            x, img = block(x, img, **block_kwargs)

        cmap = None

        if self.c_dim > 0:
            cmap = self.mapping(None, self.cmap_pre(c))
        x = self.b4(x, img, cmap)
        return x


# ----------------------------------------------------------------------------


@persistence.persistent_class
class ImageEncoder(torch.nn.Module):
    def __init__(
        self,
        layer_features,
        img_channels=3,
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        use_fp16=False,  # Use FP16 for this block?
        fp16_channels_last=False,  # Use channels-last memory format with FP16?
        store_out=False,
    ):
        super().__init__()

        self.num_layers = len(layer_features)
        self.use_fp16 = use_fp16
        self.channels_last = use_fp16 and fp16_channels_last
        self.block_dims = [img_channels]
        self.store_out = store_out

        for i, features in enumerate(layer_features):
            layer = Conv2dLayer(
                self.block_dims[-1],
                features,
                kernel_size=3,
                down=2,
                activation=activation,
            )
            self.block_dims += [features]
            setattr(self, f"conv{i}", layer)

    def forward(self, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = (
            torch.channels_last
            if self.channels_last and not force_fp32
            else torch.contiguous_format
        )

        x = img.to(dtype=dtype, memory_format=memory_format)
        xout = []
        for i in range(self.num_layers):
            layer = getattr(self, f"conv{i}")
            x = layer(x)
            xout += [x]

        return (x, xout)


# ----------------------------------------------------------------------------


@persistence.persistent_class
class ImageDecoder(torch.nn.Module):
    def __init__(
        self,
        w_dim,  # Intermediate latent (W) dimensionality.
        init_resolution,  # First input resolution
        encoder_channels,  # All enconder output channels
        img_resolution,  # Output image resolution.
        img_channels,  # Number of color channels.
        channel_base=32768,  # Overall multiplier for the number of channels.
        channel_max=512,  # Maximum number of channels in any layer.
        num_fp16_res=0,  # Use FP16 for the N highest resolutions.
        skip_layers=[],
        rgb_attention=False,  # Add RGB attention on skip connection
        **block_kwargs,  # Arguments for SynthesisBlock.
    ):
        # assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        img_resolution_log2 = int(np.log2(img_resolution / init_resolution))
        self.block_resolutions = [
            init_resolution * (2**i) for i in range(1, img_resolution_log2 + 1)
        ]

        channels_dict = {
            res: min(channel_base // res, channel_max) for res in self.block_resolutions
        }
        fp16_resolution = max(
            init_resolution * (2 ** (img_resolution_log2 + 1 - num_fp16_res)), 8
        )
        init_channels = encoder_channels[-1]

        self.num_ws = 0
        skip_layers = [len(self.block_resolutions) - x for x in skip_layers]
        self.skip_bool_list = np.isin(
            np.arange(len(self.block_resolutions)), [0] + skip_layers
        )
        self.rgb_attention = rgb_attention
        for res, do_skip, skip_dim in zip(
            self.block_resolutions, self.skip_bool_list, encoder_channels[::-1]
        ):
            in_channels = channels_dict.get(res // 2, 0) + (skip_dim if do_skip else 0)
            out_channels = channels_dict[res]
            use_fp16 = res >= fp16_resolution
            is_last = res == img_resolution
            block = SynthesisBlock(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=res,
                img_channels=img_channels,
                is_last=is_last,
                use_fp16=use_fp16,
                **block_kwargs,
            )
            self.num_ws += block.num_conv + block.num_torgb
            setattr(self, f"b{res}", block)

            if do_skip and self.rgb_attention:
                m = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
                    nn.Sigmoid(),
                )
                setattr(self, f"rgb_att_{res}", m)

    def forward(self, skip_list, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function("split_ws"):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f"b{res}")
                nws = block.num_conv + block.num_torgb
                block_ws.append(ws.narrow(1, w_idx, nws))
                w_idx += nws

        img = x = None
        assert len(self.block_resolutions) == len(block_ws)
        for res, cur_ws, do_skip, skip in zip_longest(
            self.block_resolutions, block_ws, self.skip_bool_list, skip_list[::-1]
        ):
            block = getattr(self, f"b{res}")
            if do_skip:
                if self.rgb_attention and img is not None:
                    att_layer = getattr(self, f"rgb_att_{res}")
                    skip = att_layer(img) * skip
                x = torch.cat((x, skip), dim=1) if x is not None else skip
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img


def identity(*args):
    return args if len(args) > 1 else args[0]


# def batch_blur(ims,ksize):
#     k1 = torch.tensor(getGaussianKernel(ksize, None)).type_as(ims)
#     k = k1 @ k1.T
#     b,c,w,h = ims.shape
#     blur = torch.nn.functional.conv2d(ims.reshape(b*c,1,w,h),k[None,None,:],padding=(k.size(0)-1)//2).reshape(*ims.shape)
#     return blur
from training.tools import batch_blur_quick as batch_blur


def batch_clamp_norm(t, a_mn, a_mx):
    return (
        (t / t.amax(tuple(torch.arange(1, t.ndim)), keepdims=True)).clamp(a_mn, a_mx)
        - a_mn
    ) / (a_mx - a_mn)


def scale_prop_kernel(ref, tar, v):
    return int((tar / (ref / v)) // 2 * 2 + 1)


def module_no_grad(module):
    for x in module.parameters():
        x.requires_grad = False


from torch.autograd import Function


# Copied from https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/guided_backprop.py
@persistence.persistent_class
class GuidedBackpropReLUFunction(Function):
    @staticmethod
    def forward(self, input_img):
        # positive_mask = (input_img > 0).type_as(input_img)
        # output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        # self.save_for_backward(input_img, output)
        output = F.relu(input_img).detach()
        self.save_for_backward(
            (input_img > 0),
        )
        return output

    @staticmethod
    def backward(self, grad_output):
        # input_img, output = self.saved_tensors
        #
        # positive_mask_1 = (input_img > 0).type_as(grad_output)
        # positive_mask_2 = (grad_output > 0).type_as(grad_output)
        # grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
        #                            torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
        #                                          positive_mask_1), positive_mask_2)
        input_img = self.saved_tensors[0]
        msk = input_img * (grad_output > 0)
        grad_input = grad_output * msk
        return grad_input


class GuidedBackpropReLU(torch.nn.Module):
    def forward(self, inp):
        return GuidedBackpropReLUFunction.apply(inp)


import dnnlib


@persistence.persistent_class
class GradiendtClassifierWrapper(torch.nn.Module):
    def __init__(self, classifier_kwargs, map_type="gb"):
        super().__init__()
        self.model = dnnlib.util.construct_class_by_name(**classifier_kwargs)

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == "ReLU":
                    module_top._modules[idx] = GuidedBackpropReLU()

        recursive_relu_apply(self.model.get_classifier())
        module_no_grad(self.model.get_classifier())

        self.map_type = map_type

        self.gradients = None
        self.activations = None
        self.hooks = []
        self.register_hooks()

    def register_hooks(self):
        [h.remove() for h in self.hooks]
        target_layer = self.model.get_hook()

        self.hooks = [
            target_layer.register_forward_hook(self.save_activation),
            target_layer.register_full_backward_hook(self.save_gradient)
            if "register_full_backward_hook" in dir(target_layer)
            else target_layer.register_backward_hook(self.save_gradient),
        ]

    def save_activation(self, module, input, output):
        activation = output
        self.activations = activation.detach()

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        self.gradients = grad.detach()

    def assert_device(self, x):
        if self.model.device != x.device:
            self.model.to(x.device)
            self.register_hooks()

    def forward(self, x):
        # if self.model.device != x.device:
        #     self.model.to(x.device)
        pred = self.model(x, do_softmax=False)
        return pred

    def __call__(self, x, **kwargs):
        # This is a hack just in case the G is copied / deepcopied (as G_ema does)
        if any(x.grad is not None for x in self.model.classifier.parameters()):
            self.model.zero_grad()

        self.gradients = None
        self.activations = None

        x.requires_grad = True
        x.grad = None
        h, w = x.shape[2:]

        self.assert_device(x)
        pred = super().__call__(x, **kwargs)

        if hasattr(self.model, "topclass"):
            loss = self.model.topclass(pred).sum()
        else:
            loss = pred.sum()
        # loss = pred.sum()
        loss.backward()

        # gb = x.grad
        # gb = torch.clamp((gb / (gb.std() + 1e-5) * 0.1).abs(), 0, 1).mean(1, keepdims=True)
        gb = x.grad.abs().mean(1, keepdims=True)
        gb = (gb / (gb.std((1, 2, 3), keepdims=True) + 1e-15)).detach()

        cam = (
            (self.activations * self.gradients.mean((2, 3), keepdims=True))
            .clamp(min=0)
            .sum(1, keepdims=True)
        )
        cam /= cam.amax((2, 3), keepdims=True) + 1e-15
        cam = F.interpolate(
            cam, size=(h, w), mode="bicubic", align_corners=False
        ).clamp(0, 1)

        if self.map_type != "cam":
            assert self.map_type in ["camgb", "gb"]
            out = gb * cam if (self.map_type == "camgb") else gb

            # TODO: Choose one of the two
            # Gaussian blur
            # blur_out = batch_blur(out,scale_prop_kernel(29, x.size(-1), 224))
            # # blur_out = batch_clamp_norm(blur_out, 0.1, 0.3)

            # Band-pass filter:
            from training.tools import batch_bandpass_quick

            blur_out = batch_bandpass_quick(out, scale_prop_kernel(29, x.size(-1), 224))

            blur_out = blur_out.abs().sum(1, keepdims=True)
            thrs = blur_out.flatten(1).std(1).reshape(-1, 1, 1, 1) * 2 + 1e-15
            blur_out = blur_out.clamp(max=thrs) / thrs

            out = blur_out.detach()
        else:
            out = cam.detach()

        return 1 - out


@persistence.persistent_class
class Image2Image(torch.nn.Module):
    def __init__(
        self,
        img_resolution,
        img_channels,
        n_downsample,
        attr_dim,
        channel_max=512,
        style_dim=512,
        style_layers=8,
        attr_layers=8,
        skip_layers=[],
        skip_kind="linear",  # linear
        mapping_kwargs={},
        synthesis_kwargs={},
        classifier_kwargs={},
        blur_skip=False,
        skip_grad_blur=None,
        learn_mask=None,
        blur_msk=None,
        style_enc=True,
        bottleneck_class=False,
        finetune=None,
    ):
        super().__init__()

        # ToRGB -> ToRGB+M
        if learn_mask in ["rgb", "rgb_cam"]:
            img_channels = img_channels + 1

        self.content_enc = ImageEncoder(
            [min(2 ** (i + 6), channel_max) for i in range(n_downsample)],
            store_out=len(skip_layers) == 0,
        )
        if style_enc:
            self.style_enc = ImageEncoder(
                [min(2 ** (i + 6), channel_max) for i in range(n_downsample)]
            )
            z_dim = self.style_enc.block_dims[-1]
        else:
            self.style_enc = None
            z_dim = self.content_enc.block_dims[-1]

        self.bottleneck_class = bottleneck_class
        if self.bottleneck_class:
            style_num_ws = n_downsample * 3 - 1
            attr_num_ws = 1
        else:
            style_num_ws = n_downsample * 2
            attr_num_ws = n_downsample

        self.style_map = MappingNetwork(
            z_dim=z_dim,
            c_dim=0,
            num_layers=style_layers,
            w_dim=style_dim,
            num_ws=style_num_ws,
        )
        self.attr_map = MappingNetwork(
            z_dim=attr_dim,
            c_dim=0,
            num_layers=attr_layers,
            w_dim=style_dim,
            num_ws=attr_num_ws,
        )

        self.image_dec = ImageDecoder(
            style_dim,
            img_resolution // (2**n_downsample),
            self.content_enc.block_dims,
            img_resolution,
            img_channels,
            skip_layers=skip_layers,
            **synthesis_kwargs,
        )

        # Skip activation blur
        self.blur_skip = blur_skip
        if self.blur_skip:
            im_size = img_resolution / (skip_layers[0] ** 2)

            KSIZE = int((im_size / 3) // 2) * 2 + 1
            N_KERNELS = 1024
            st = im_size / 25

            self.register_buffer(
                "blur_list",
                torch.tensor(
                    [
                        np.dot(
                            getGaussianKernel(KSIZE, i), getGaussianKernel(KSIZE, i).T
                        )
                        for i in np.linspace(1e-20, st, N_KERNELS)
                    ]
                ).float(),
                persistent=True,
            )

        # Skip connection transformers (Identity)
        do_skip_n_channels = list(
            zip(self.image_dec.skip_bool_list, self.content_enc.block_dims[::-1])
        )[1:]

        self.skip_transf = [None]
        for i, (do_skip, nc) in enumerate(do_skip_n_channels):
            if do_skip:
                if skip_kind == "linear":
                    m = identity
                else:
                    raise NotImplementedError
                setattr(self, f"skip{i}", m)
            else:
                m = None
            self.skip_transf += [m]
        self.skip_transf = self.skip_transf[::-1]

        assert not ((skip_grad_blur is not None) and (learn_mask is not None))

        self.skip_grad_blur = skip_grad_blur
        if skip_grad_blur is not None:
            if skip_grad_blur in ["gb", "camgb"]:
                self.skip_grad_blur = GradiendtClassifierWrapper(
                    classifier_kwargs, map_type=skip_grad_blur
                )
            else:
                raise NotImplementedError

        self.learn_mask_random = (
            False if learn_mask is None else learn_mask[-6:] == "random"
        )
        self.learn_mask = learn_mask[:-7] if self.learn_mask_random else learn_mask
        learn_mask = self.learn_mask

        self.learn_mask_random = self.learn_mask_random or (blur_msk == "random")

        if learn_mask is not None:
            # Skip mask
            if learn_mask in ["skip", "skip_cam"]:
                skip_pos = np.nonzero([x[0] for x in do_skip_n_channels])[0]
                if len(skip_pos) > 1:
                    raise NotImplementedError
                else:
                    learn_mask_in_channels = do_skip_n_channels[skip_pos[0]][1]

                self.learn_mask_seq = nn.Sequential(
                    nn.Conv2d(
                        learn_mask_in_channels,
                        1,
                        kernel_size=7,
                        stride=1,
                        padding=3,
                        bias=False,
                    ),
                    nn.Sigmoid(),
                )
            # RGB mask
            elif learn_mask in ["rgb", "rgb_cam"]:
                pass
            else:
                raise NotImplementedError

            if learn_mask in ["rgb_cam", "skip_cam"]:
                self.skip_grad_blur = GradiendtClassifierWrapper(
                    classifier_kwargs, map_type="cam"
                )

        # Fine tune
        self.finetune = finetune

    def requires_grad_(self, requires_grad: bool = True):
        if self.finetune:
            for p in self.parameters():
                p.requires_grad = False

            sorted_modules = sorted(
                self.image_dec.named_children(), key=lambda x: -int(x[0][1:])
            )[: self.finetune]
            for _, m in sorted_modules:
                for p in m.parameters():
                    p.requires_grad = requires_grad

        else:
            for p in self.parameters():
                p.requires_grad = requires_grad

        return self

    def __age_to_matrix__(self, age):
        z = torch.zeros(len(age), 101)
        z[torch.arange(len(age)), age] = 1
        return z

    def __interleave_attr_style__(self, a_out, s_out):
        if self.bottleneck_class:
            res = torch.cat([a_out, s_out], dim=1)
        else:
            res = torch.cat(
                list(
                    chain.from_iterable(
                        zip(a_out.split(1, dim=1), s_out.split(2, dim=1))
                    )
                ),
                dim=1,
            )
        return res

    def _batch_blur(self, act_batch, blur_val=None):
        bsize = act_batch.size(0)
        if blur_val is not None:
            blur_idx = [int(blur_val * (self.blur_list.size(0) - 1))] * bsize
        else:
            blur_idx = np.random.choice(
                np.arange(self.blur_list.size(0)), bsize, replace=True
            )
        blur_w = self.blur_list[blur_idx][:, None, :]
        temp_m, temp_w = (act_batch[None, :], blur_w[None, :].transpose(0, 1))
        pad = temp_w.shape[-1] // 2
        act_batch = torch.nn.functional.conv3d(
            temp_m, temp_w, padding=(0, pad, pad), groups=bsize
        )[0]
        return act_batch

    def forward(
        self,
        img,
        cls,
        truncation_psi=1,
        truncation_cutoff=None,
        blur_val=None,
        return_msk=False,
        style_mixing=False,
    ):
        c_out, c_out_skip = self.content_enc(img)
        if self.style_enc:
            s_out = self.style_enc(img)[0].mean((2, 3))
        else:
            s_out = c_out.mean((2, 3))

        s_out = self.style_map(s_out, None, truncation_psi, truncation_cutoff)
        a_out = self.attr_map(
            cls.to(s_out.device), None, truncation_psi, truncation_cutoff
        )

        # if style_mixing:
        #     cutoff = torch.empty([], dtype=torch.int64, device=s_out.device).random_(1, a_out.shape[1])
        #     perms = torch.randperm(s_out.shape[0])
        #     s_out[:, cutoff*2:] = s_out[perms, cutoff*2:]
        #
        #     #c_out_skip is reversed
        #     c_out_skip = c_out_skip[::-1]
        #     c_out_skip[cutoff:] = [e[perms] for e in c_out_skip[cutoff:]]
        #     c_out_skip = c_out_skip[::-1]

        w = self.__interleave_attr_style__(a_out, s_out)

        # activation blur
        if self.blur_skip:
            skip_counter = 0
            for i, (f, _) in enumerate(zip(self.skip_transf, c_out_skip)):
                if f is not None:
                    c_out_skip[i] = self._batch_blur(
                        c_out_skip[i], blur_val=None if not self.finetune else 0.0
                    )
                    skip_counter += 1
            # assert skip_counter == 1

        # import matplotlib.pyplot as plt
        cam = None

        if self.skip_grad_blur is not None:
            cam = self.skip_grad_blur(img.float().detach())

        msk = (
            cam if self.learn_mask is None else None
        )  # e.i.: "skip_age_blur" in ['gb','camgb']

        if self.learn_mask in ["skip", "skip_cam"]:
            skip_idx = [i for i, x in enumerate(self.skip_transf) if x is not None]
            assert len(skip_idx) == 1
            msk = self.learn_mask_seq(c_out_skip[skip_idx[0]])

        if msk is not None:
            for i, (f, c) in enumerate(zip(self.skip_transf, c_out_skip)):
                if f is not None:
                    im_size = c.size(-1)
                    blur_val = (
                        None
                        if self.learn_mask_random
                        else (1.0 if not self.finetune else 0.8)
                    )
                    blur_c = self._batch_blur(c, blur_val=blur_val)
                    if msk.size(2) != im_size:
                        msk = F.interpolate(msk, size=(im_size, im_size), mode="area")
                    merged_c = c * msk + blur_c * (1 - msk)
                    c_out_skip[i] = merged_c

        c_out_skip = [
            f(c) if f is not None else c for f, c in zip(self.skip_transf, c_out_skip)
        ]
        img_out = self.image_dec(c_out_skip, w)

        if self.learn_mask in ["rgb", "rgb_cam"]:
            img_out, msk = torch.split(img_out, [img_out.size(1) - 1, 1], dim=1)
            msk = torch.sigmoid(msk)
            merged_img = img * msk + img_out * (1 - msk)
            img_out = merged_img

        if return_msk:
            to_return = (
                (img_out, msk, cam)
                if self.learn_mask is not None
                else (img_out, None, None)
            )
        else:
            to_return = img_out

        return to_return


import torch.nn as nn
import torch.nn.functional as F


@persistence.persistent_class
class VGG(torch.nn.Module):
    def __init__(self, pool="max", own_relu=False):
        super().__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.fc6 = nn.Linear(25088, 4096, bias=True)
        self.fc7 = nn.Linear(4096, 4096, bias=True)
        self.fc8_101 = nn.Linear(4096, 101, bias=True)

        self.own_relu = own_relu
        self.relu = nn.ReLU() if self.own_relu else F.relu

        if pool == "max":
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == "avg":
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = {}
        out["r11"] = self.relu(self.conv1_1(x))
        out["r12"] = self.relu(self.conv1_2(out["r11"]))
        out["p1"] = self.pool1(out["r12"])
        out["r21"] = self.relu(self.conv2_1(out["p1"]))
        out["r22"] = self.relu(self.conv2_2(out["r21"]))
        out["p2"] = self.pool2(out["r22"])
        out["r31"] = self.relu(self.conv3_1(out["p2"]))
        out["r32"] = self.relu(self.conv3_2(out["r31"]))
        out["r33"] = self.relu(self.conv3_3(out["r32"]))
        out["p3"] = self.pool3(out["r33"])
        out["r41"] = self.relu(self.conv4_1(out["p3"]))
        out["r42"] = self.relu(self.conv4_2(out["r41"]))
        out["r43"] = self.relu(self.conv4_3(out["r42"]))
        out["p4"] = self.pool4(out["r43"])
        out["r51"] = self.relu(self.conv5_1(out["p4"]))
        out["r52"] = self.relu(self.conv5_2(out["r51"]))
        out["r53"] = self.relu(self.conv5_3(out["r52"]))
        out["p5"] = self.pool5(out["r53"])
        out["p5"] = out["p5"].view(out["p5"].size(0), -1)
        out["fc6"] = self.relu(self.fc6(out["p5"]))
        out["fc7"] = self.relu(self.fc7(out["fc6"]))
        out["fc8"] = self.fc8_101(out["fc7"])
        return out

    def __deepcopy__(self, memodict={}):
        m = VGG(own_relu=self.own_relu)
        m.load_state_dict(self.state_dict())
        return m
