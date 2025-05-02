import argparse
import torch
from torchinfo import summary
from torchvision import transforms
import mlflow
import mlflow.pytorch

import data_setup
import engine
import model_builder
import utils

def build_model(args, num_classes: int, device: str, img_size: int):
    return model_builder.ViT(
        img_size=img_size,
        in_channels=3,
        patch_size=args.patch_size,
        num_transformer_layer=args.transformer_layer,
        embedding_dim=768,
        mlp_size=args.mlp_size,
        num_heads=args.num_heads,
        attn_dropout=args.attn_dropout,
        num_classes=num_classes
    ).to(device)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = 224
    print(f"[INFO] Using {device} to train the model")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    train_loader, test_loader, class_names = data_setup.create_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        transform=transform,
        batch_size=args.batch_size
    )

    model = build_model(args, num_classes=len(class_names), device=device, img_size=IMG_SIZE)

    print('=============Model Architecture=============')
    print(summary(model, input_size=(32, 3, IMG_SIZE, IMG_SIZE),
                  col_names=["input_size", "output_size", "num_params", "trainable"],
                  col_width=20, row_settings=['var_names']))

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.mlflow:
        if not args.exp_name or not args.model_architecture:
            raise ValueError("--exp_name and --model_architecture are required with --mlflow")

        mlflow.set_experiment(args.exp_name)
        with mlflow.start_run():
            mlflow.log_params(vars(args))

            results = engine.train(model, train_loader, test_loader, loss_fn, optimizer, args.epochs, device=device)

            for i in range(args.epochs):
                mlflow.log_metric("train_loss", results["train_loss"][i], step=i)
                mlflow.log_metric("test_loss", results["test_loss"][i], step=i)
                mlflow.log_metric("train_accuracy", results["train_acc"][i], step=i)
                mlflow.log_metric("test_accuracy", results["test_acc"][i], step=i)

            mlflow.pytorch.log_model(model, "model")
    else:
        engine.train(model, train_loader, test_loader, loss_fn, optimizer, args.epochs, device)

    utils.save_model(model=model, target_dir=args.save_dir, model_name=args.model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Vision Transformer model on image data")
    parser.add_argument("--train_dir", type=str, default="./data/pizza_steak_sushi/train")
    parser.add_argument("--test_dir", type=str, default="./data/pizza_steak_sushi/test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--model_name", type=str, default="model.pth")

    # Vision Transformer specific
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--transformer_layer", type=int, default=12)
    parser.add_argument("--mlp_size", type=int, default=3072)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--attn_dropout", type=float, default=0.1)

    # MLflow tracking
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--exp_name", type=str, help="MLflow experiment name")
    parser.add_argument("--model_architecture", type=str, help="Model architecture name")

    args = parser.parse_args()
    main(args)
