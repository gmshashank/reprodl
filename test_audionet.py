import torch
import pytorch_lightning as pl
from hydra.experimental import initialize, compose
from training import AudioNet


def testAudioNet():
    with initialize(config_path="configs"):
        # Load configurations
        cfg = compose(
            config_name="default.yaml",
            overrides=["trainer.max_epochs=250", "trainer.gpus=0"],
        )

        pl.seed_everything(cfg.seed)

        # Create a random batch of data
        data = torch.utils.data.TensorDataset(
            torch.rand(5, 1, 200, 100), torch.randint(0, 10, size=(5,))
        )

        # Initialize the network
        net = AudioNet(cfg.model)

        trainer = pl.Trainer(**cfg.trainer, overfit_batches=1)
        trainer.fit(net, torch.utils.data.DataLoader(data))

        print(trainer.logged_metrics["train_loss"].item())
        assert trainer.logged_metrics["train_loss"].item() <= 0.1
