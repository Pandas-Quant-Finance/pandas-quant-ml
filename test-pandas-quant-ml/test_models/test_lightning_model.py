import os
import tempfile
from typing import Any

import pandas as pd

from pandas_quant_ml.data_generators.train_loop_data_generator import TrainTestLoop
from pandas_quant_ml.data_transformers.generic.selection import Select
from pandas_quant_ml.models.lightning_model import LightningModel
from pandas_quant_ml.utils.serialize import serialize, deserialize

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from unittest import TestCase

from testing_data.data import get_x_or


class TestLightningModel(TestCase):

    def test_simple_no_test(self):
        df = get_x_or()

        class LNet(pl.LightningModule):

            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(2, 1),
                    nn.Linear(1, 1),
                    nn.Sigmoid()
                )

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                return self.net(*args, **kwargs)

            def training_step(self, batch, batch_idx):
                x, y, w = batch
                y_hat = self.forward(x)

                # loss='binary_crossentropy', optimizer=keras.optimizers.Adam()
                loss = nn.functional.binary_cross_entropy(y_hat, y, w)
                # print(loss.detach().numpy())
                self.log('loss', loss, on_step=True, on_epoch=False)
                return loss

            def configure_optimizers(self):
                optimizer = optim.Adam(self.parameters())
                return optimizer

        model = LightningModel(
            TrainTestLoop(Select(0, 1), Select("label")),
            LNet()
        )

        model.fit(df, train_test_split_ratio=1.0, batch_size=6, max_epochs=30)
        print(model.history)
        self.assertEquals(len(model.history['loss']), 30)
        self.assertLess(model.history['loss'][-1], model.history['loss'][0])

        _, pdf = next(model.predict(df, include_labels=True))
        self.assertGreater(len(pdf), 10)

        # Test save and load model
        _, pdf_orig = next(model.predict(df))
        print(pdf)

        with tempfile.TemporaryDirectory() as td:
            serialize(model, os.path.join(td, "lala"))
            model2 = deserialize(os.path.join(td, "lala"))
            _, pdf_restored = next(model2.predict(df))

        pd.testing.assert_frame_equal(pdf_orig, pdf_restored)
        self.assertEquals(len(model.history['loss']), len(model2.history['loss']))

        with self.assertLogs() as captured:
            model2.fit(df, train_test_split_ratio=1.0, batch_size=6, max_epochs=3)

        self.assertIn("reset_pipeline=True", ";".join(captured.output))
        self.assertEquals(len(model2.history['loss']), 33)
