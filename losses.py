import torch
import torch.nn.functional as F

def ssim(prediction, target, window_size=11):
    prediction = torch.clamp(prediction, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    padding = window_size // 2

    mu_x = F.avg_pool2d(prediction, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(prediction * prediction, window_size, stride=1, padding=padding) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=padding) - mu_y_sq
    sigma_xy = F.avg_pool2d(prediction * target, window_size, stride=1, padding=padding) - mu_xy

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    score = numerator / (denominator + 1e-8)
    return score.mean()

def ssim_loss(prediction, target):
    return 1.0 - ssim(prediction, target)
