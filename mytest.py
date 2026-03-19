import torch
from ultralytics.nn.modules.head import AngleAwareDetect, ImageAngle

if __name__ == "__main__":
    # Step 1: 创建 AngleAwareDetect（它内部包含 ImageAngle）
    nc = 11
    ch = (64, 128, 256)  # P3, P4, P5 的通道数

    # 先创建两个模块
    image_angle = ImageAngle(11, (64, 128, 256))
    detect_head = AngleAwareDetect(11, (64, 128, 256, -1))  # 注意 ch 包含 -1

    # 前向
    features = [torch.randn(2, c, h, w) for c, h, w in [(64, 80, 80), (128, 40, 40), (256, 20, 20)]]
    angle_confs = image_angle(features)
    full_input = features + [angle_confs]  # [P3, P4, P5, [conf3, conf4, conf5]]

    outputs = detect_head(full_input)


    head = AngleAwareDetect(nc=nc, ch=ch)
    head.train()

    # Step 2: 构造 dummy neck 输出（可微）
    dummy_features = [
        torch.randn(2, 64, 80, 80, requires_grad=True),
        torch.randn(2, 128, 40, 40, requires_grad=True),
        torch.randn(2, 256, 20, 20, requires_grad=True)
    ]

    # Step 3: 前向
    outputs = head(dummy_features)  # outputs 是 list of tensors

    # Step 4: 构造假 loss
    fake_loss = 0
    for out in outputs:
        cls_part = out[:, -nc:, :, :]  # 最后 nc 通道是分类
        fake_loss += cls_part.mean()

    print("Before backward:")

    for name, param in head.named_parameters():
        if "conf_convs" in name or "image_angle" in name:
            print(f"  {name}: grad is None = {param.grad is None}")

    # Step 5: 反向
    fake_loss.backward()

    print("\nAfter backward:")
    for name, param in head.named_parameters():
        if "conf_convs" in name or "image_angle" in name:
            grad_norm = param.grad.norm().item() if param.grad is not None else 'N/A'
            print(f"  {name}: grad is None = {param.grad is None}, grad norm = {grad_norm}")