import argparse
import os
from datetime import datetime

import torch
import torch.utils.data
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from model import ReCoNet
from dataset import MonkaaDataset, FlyingThings3DDataset
import custom_transforms
from vgg import Vgg16
from utils import \
    warp_optical_flow, \
    rgb_to_luminance, \
    l2_squared, \
    tensors_sum, \
    resize_optical_flow, \
    occlusion_mask_from_flow, \
    gram_matrix, \
    preprocess_for_reconet, \
    preprocess_for_vgg, \
    postprocess_reconet, \
    RunningLossesContainer, \
    Dummy


def output_temporal_loss(
        input_frame,
        previous_input_frame,
        output_frame,
        previous_output_frame,
        reverse_optical_flow,
        occlusion_mask):
    input_diff = input_frame - warp_optical_flow(previous_input_frame, reverse_optical_flow)
    output_diff = output_frame - warp_optical_flow(previous_output_frame, reverse_optical_flow)
    luminance_input_diff = rgb_to_luminance(input_diff).unsqueeze_(1)

    n, c, h, w = input_frame.shape
    loss = l2_squared(occlusion_mask * (output_diff - luminance_input_diff)) / (h * w)
    return loss


def feature_temporal_loss(
        feature_maps,
        previous_feature_maps,
        reverse_optical_flow,
        occlusion_mask):
    n, c, h, w = feature_maps.shape

    reverse_optical_flow_resized = resize_optical_flow(reverse_optical_flow, h, w)
    occlusion_mask_resized = torch.nn.functional.interpolate(occlusion_mask, size=(h, w), mode='nearest')

    feature_maps_diff = feature_maps - warp_optical_flow(previous_feature_maps, reverse_optical_flow_resized)
    loss = l2_squared(occlusion_mask_resized * feature_maps_diff) / (c * h * w)

    return loss


def content_loss(
        content_feature_maps,
        style_feature_maps):
    n, c, h, w = content_feature_maps.shape

    return l2_squared(content_feature_maps - style_feature_maps) / (c * h * w)


def style_loss(
        content_feature_maps,
        style_gram_matrices):
    loss = 0
    for content_fm, style_gm in zip(content_feature_maps, style_gram_matrices):
        loss += l2_squared(gram_matrix(content_fm) - style_gm)
    return loss


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))


def stylize_image(image, model):
    if isinstance(image, Image.Image):
        image = transforms.ToTensor()(image)
    image = image.cuda().unsqueeze_(0)
    image = preprocess_for_reconet(image)
    styled_image = model(image).squeeze()
    styled_image = postprocess_reconet(styled_image)
    return styled_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("style", help="Path to style image")
    parser.add_argument("--data-dir", default="./data", help="Path to data root dir")
    parser.add_argument("--gpu-device", type=int, default=0, help="GPU device index")
    parser.add_argument("--alpha", type=float, default=1e4, help="Weight of content loss")
    parser.add_argument("--beta", type=float, default=1e5, help="Weight of style loss")
    parser.add_argument("--gamma", type=float, default=1e-5, help="Weight of style loss")
    parser.add_argument("--lambda-f", type=float, default=1e5, help="Weight of feature temporal loss")
    parser.add_argument("--lambda-o", type=float, default=2e5, help="Weight of output temporal loss")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-file", default="./model.pth", help="Output model file path")
    parser.add_argument("--frn", action='store_true', help="Use Filter Response Normalization and TLU")

    args = parser.parse_args()

    running_losses = RunningLossesContainer()
    global_step = 0

    with torch.cuda.device(args.gpu_device):
        transform = transforms.Compose([
            custom_transforms.Resize(640, 360),
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.ToTensor()
        ])
        monkaa = MonkaaDataset(os.path.join(args.data_dir, "monkaa"), transform)
        flyingthings3d = FlyingThings3DDataset(os.path.join(args.data_dir, "flyingthings3d"), transform)
        dataset = monkaa + flyingthings3d
        batch_size = 2
        traindata = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=3,
                                                pin_memory=True,
                                                drop_last=True)

        model = ReCoNet(frn=args.frn).cuda()
        vgg = Vgg16().cuda()

        with torch.no_grad():
            style = Image.open(args.style)
            style = transforms.ToTensor()(style).cuda()
            style = style.unsqueeze_(0)
            style_vgg_features = vgg(preprocess_for_vgg(style))
            style_gram_matrices = [gram_matrix(x) for x in style_vgg_features]
            del style, style_vgg_features

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        writer = SummaryWriter()

        n_epochs = args.epochs
        for epoch in range(n_epochs):
            for sample in traindata:
                optimizer.zero_grad()

                sample = {name: tensor.cuda() for name, tensor in sample.items()}

                occlusion_mask = occlusion_mask_from_flow(
                    sample["optical_flow"],
                    sample["reverse_optical_flow"],
                    sample["motion_boundaries"])

                # Compute ReCoNet features and output

                reconet_input = preprocess_for_reconet(sample["frame"])
                feature_maps = model.encoder(reconet_input)
                output_frame = model.decoder(feature_maps)

                previous_reconet_input = preprocess_for_reconet(sample["previous_frame"])
                previous_feature_maps = model.encoder(previous_reconet_input)
                previous_output_frame = model.decoder(previous_feature_maps)

                # Compute VGG features

                vgg_input_frame = preprocess_for_vgg(sample["frame"])
                vgg_output_frame = preprocess_for_vgg(postprocess_reconet(output_frame))
                input_vgg_features = vgg(vgg_input_frame)
                output_vgg_features = vgg(vgg_output_frame)

                vgg_previous_input_frame = preprocess_for_vgg(sample["previous_frame"])
                vgg_previous_output_frame = preprocess_for_vgg(postprocess_reconet(previous_output_frame))
                previous_input_vgg_features = vgg(vgg_previous_input_frame)
                previous_output_vgg_features = vgg(vgg_previous_output_frame)

                # Compute losses

                alpha = args.alpha
                beta = args.beta
                gamma = args.gamma
                lambda_f = args.lambda_f
                lambda_o = args.lambda_o

                losses = {
                    "content loss": tensors_sum([
                        alpha * content_loss(output_vgg_features[2], input_vgg_features[2]),
                        alpha * content_loss(previous_output_vgg_features[2], previous_input_vgg_features[2]),
                    ]),
                    "style loss": tensors_sum([
                        beta * style_loss(output_vgg_features, style_gram_matrices),
                        beta * style_loss(previous_output_vgg_features, style_gram_matrices),
                    ]),
                    "total variation": tensors_sum([
                        gamma * total_variation(output_frame),
                        gamma * total_variation(previous_output_frame),
                    ]),
                    "feature temporal loss": lambda_f * feature_temporal_loss(feature_maps, previous_feature_maps,
                                                                              sample["reverse_optical_flow"],
                                                                              occlusion_mask),
                    "output temporal loss": lambda_o * output_temporal_loss(reconet_input, previous_reconet_input,
                                                                            output_frame, previous_output_frame,
                                                                            sample["reverse_optical_flow"],
                                                                            occlusion_mask)
                }

                training_loss = tensors_sum(list(losses.values()))
                losses["training loss"] = training_loss

                training_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    running_losses.update(losses)

                    last_iteration = global_step == len(dataset) // batch_size * n_epochs - 1
                    if global_step % 25 == 0 or last_iteration:
                        average_losses = running_losses.get()
                        for key, value in average_losses.items():
                            writer.add_scalar(key, value, global_step)

                        running_losses.reset()

                    if global_step % 100 == 0 or last_iteration:
                        styled_test_image = stylize_image(Image.open("test_image.jpeg"), model)
                        writer.add_image('test image', styled_test_image, global_step)

                        for i in range(0, len(dataset), len(dataset) // 4):
                            sample = dataset[i]
                            styled_train_image_1 = stylize_image(sample["frame"], model)
                            styled_train_image_2 = stylize_image(sample["previous_frame"], model)
                            grid = torchvision.utils.make_grid([styled_train_image_1, styled_train_image_2])
                            writer.add_image(f'train images {i}', grid, global_step)

                    global_step += 1

        torch.save(model.state_dict(), args.output_file)
        writer.close()
