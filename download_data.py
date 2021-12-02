import torchvision

if __name__ == "__main__":
        print("Downloading CIFAR10 data.")
        torchvision.datasets.CIFAR10(root="data", download=True)
        print("Data download complete.")

        train_dataset = torchvision.datasets.CIFAR10(root="data")
        print(f"Number of train samples:")