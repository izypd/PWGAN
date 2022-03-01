import torch
import torchvision

import math
from pathlib import Path
import typer


def get_square_weight_matrix(n_channels, img_size, center_rate, device):
    m = torch.ones(n_channels, img_size, img_size).to(device)
    half_not_center_size = int(img_size / 2 * (1 - center_rate))
    if half_not_center_size == 0:
        return m
    for i in range(half_not_center_size - 1, -1, -1):
        weight = 1 - (i - half_not_center_size) ** 2 / (half_not_center_size ** 2)
        m[:, i] = weight
        m[:, :, i] = weight
        m[:, img_size - 1 - i] = weight
        m[:, :, img_size - 1 - i] = weight
    return m


def get_circle_weight_matrix(n_channels, img_size, center_rate, device):
    m = torch.ones(n_channels, img_size, img_size).to(device)
    half_not_center_size = int(img_size / 2 * (1 - center_rate))
    dict_x_weight = {}
    for i in range(half_not_center_size - 1, -1, -1):
        weight = 1 - (i - half_not_center_size) ** 2 / (half_not_center_size ** 2)
        dict_x_weight[i] = weight
    for x in range(img_size):
        for y in range(img_size):
            center_i = img_size // 2 - 1
            x_subtract_center = x - center_i
            y_subtract_center = y - center_i
            if x_subtract_center <= 0:
                corresponding_x = -x_subtract_center
            else:
                corresponding_x = x_subtract_center - 1
            if y_subtract_center <= 0:
                corresponding_y = -y_subtract_center
            else:
                corresponding_y = y_subtract_center - 1
            power_sum_sqrt = math.sqrt(corresponding_x ** 2 + corresponding_y ** 2)
            half_size = img_size // 2
            half_center_size = half_size - half_not_center_size
            if power_sum_sqrt >= half_size - 1:
                m[:, x, y] = 0.0
            elif power_sum_sqrt >= half_center_size:
                m[:, x, y] = dict_x_weight[half_size - 1 - int(power_sum_sqrt)]
    return m


def get_diamond_weight_matrix(n_channels, img_size, center_rate, device):
    m = torch.ones(n_channels, img_size, img_size).to(device)
    half_not_center_size = int(img_size / 2 * (1 - center_rate))
    dict_x_weight = {}
    for i in range(half_not_center_size - 1, -1, -1):
        weight = 1 - (i - half_not_center_size) ** 2 / (half_not_center_size ** 2)
        dict_x_weight[i] = weight
    for x in range(img_size):
        for y in range(img_size):
            center_i = img_size // 2 - 1
            x_subtract_center = x - center_i
            y_subtract_center = y - center_i
            if x_subtract_center <= 0:
                corresponding_x = -x_subtract_center
            else:
                corresponding_x = x_subtract_center - 1
            if y_subtract_center <= 0:
                corresponding_y = -y_subtract_center
            else:
                corresponding_y = y_subtract_center - 1
            corresponding_x_plus_y = corresponding_x + corresponding_y
            first_corresponding_x = img_size // 2 - half_not_center_size
            last_corresponding_x = img_size // 2 - 1
            if corresponding_x_plus_y > last_corresponding_x:
                m[:, x, y] = dict_x_weight[0]
            elif corresponding_x_plus_y >= first_corresponding_x:
                m[:, x, y] = dict_x_weight[last_corresponding_x - corresponding_x_plus_y]
    return m


def loss(input_dir: str, output_dir: str, device='cpu', type='circle', size=192, rate=0.5, img_extension='png'):
    if type == 'circle':
        weight_matrix = get_circle_weight_matrix(3, size, rate, device)
    if type == 'square':
        weight_matrix = get_square_weight_matrix(3, size, rate, device)
    if type == 'diamond':
        weight_matrix = get_diamond_weight_matrix(3, size, rate, device)
    img_loss = torch.nn.MSELoss()

    input_img_paths = sorted(Path(input_dir).glob(f'*.{img_extension}'))
    output_img_paths = sorted(Path(output_dir).glob(f'*.{img_extension}'))
    input_output_paths = zip(input_img_paths, output_img_paths)

    all_weighted_loss = 0.0
    for input_output_path in input_output_paths:
        input_path = str(input_output_path[0])
        output_path = str(input_output_path[1])
        input_img = torchvision.io.read_image(input_path).type_as(torch.FloatTensor()).to(device)
        output_img = torchvision.io.read_image(output_path).type_as(torch.FloatTensor()).to(device)

        weighted_loss = img_loss(input_img * weight_matrix, output_img * weight_matrix)
        all_weighted_loss += float(weighted_loss)

    num = len(input_img_paths)
    avg_weighted_loss = all_weighted_loss / num
    print(avg_weighted_loss)


if __name__ == "__main__":
    typer.run(loss)
