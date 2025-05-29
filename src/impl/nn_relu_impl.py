from logging import getLogger

from src.config.config import Config
from src.impl import IImpl
from src.tensor.interfaces import ITensorHandler


log = getLogger(__name__)


class _NnReluImpl(IImpl):

    def __init__(self, config: Config, tensor_handler: ITensorHandler):
        super().__init__(config, tensor_handler)

        self.m = 0  # Sample count.
        self.n = self.dimensions  # Pixels * 3 (rbg).
        self.h = config.hidden_layer_size  # Hidden layer size.

        self.w1 = self.tensor_handler.zeros((self.h, self.n))
        self.b1 = 0
        self.w2 = self.tensor_handler.zeros(self.h)
        self.b2 = 0

        self.a1 = None
        self.z1 = None
        self.a2 = None
        self.z2 = None

        self.d1 = None
        self.d2 = None

        self.j = None

        self.dj_dw2 = None
        self.dj_db2 = None
        self.dj_dw1 = None
        self.dj_db1 = None

    def _z1(self):
        w1 = self.w1
        b1 = self.b1
        x = self.x
        z1 = self.tensor_handler.multiply(w1, x) + b1
        self.z1 = z1

    def _a1(self):
        z1 = self.z1
        a1 = self.tensor_handler.relu(z1)
        self.a1 = a1

    def _z2(self):
        w2 = self.w2
        b2 = self.b2
        a1 = self.a1
        z2 = self.tensor_handler.multiply(w2, a1) + b2
        self.z2 = z2

    def _a2(self):
        z2 = self.z2
        a2 = 1 / (1 + self.tensor_handler.exp(-z2))
        self.a2 = a2

    def _j(self):
        a2 = self.a2
        y = self.y
        m = self.m
        j = (-1 / m) * (
            self.tensor_handler.multiply(
                y,
                self.tensor_handler.log(a2)) + self.tensor_handler.multiply(
                    1 - y, self.tensor_handler.log(1 - a2)))
        self.j = j

    def _d2(self):
        a2 = self.a2
        y = self.y
        d2 = a2 - y
        self.d2 = d2

    def _d1(self):
        d2 = self.d2
        w2 = self.w2
        z1 = self.z1
        d1 = self.tensor_handler.multiply(d2, w2.T) * self.tensor_handler.d_relu(z1)
        self.d1 = d1

    def _dj_dw2(self):
        a1 = self.a1
        d2 = self.d2
        dj_dw2 = self.tensor_handler.multiply(a1, d2.T)
        self.dj_dw2 = dj_dw2

    def _dj_db2(self):
        d2 = self.d2
        m = self.m
        dj_b2 = (1 / m) * self.tensor_handler.sum(d2)
        self.dj_b2 = dj_b2

    def _dj_dw1(self):
        d1 = self.d1
        x = self.x
        m = self.m
        dj_dw1 = (1 / m) * self.tensor_handler.multiply(d1, x)
        self.dj_dw1 = dj_dw1

    def _dj_db1(self):
        d1 = self.d1
        m = self.m
        dj_db1 = (1 / m) * self.tensor_handler.sum(d1)
        self.dj_db1 = dj_db1

    def _w1(self):
        w1 = self.w1
        alpha = self.alpha
        dj_dw1 = self.dj_dw1
        w1 -= alpha * dj_dw1
        self.w1 = w1

    def _w2(self):
        w2 = self.w2
        alpha = self.alpha
        dj_dw2 = self.dj_dw2
        w2 -= alpha * dj_dw2
        self.w2 = w2

    def _b1(self):
        b1 = self.b1
        alpha = self.alpha
        dj_db1 = self.dj_db1
        b1 -= alpha * dj_db1
        self.b1 = b1

    def _b2(self):
        b2 = self.b2
        alpha = self.alpha
        dj_db2 = self.dj_db2
        b2 -= alpha * dj_db2
        self.b2 = b2

    def train_epoch(self):

        # Forward propagation.
        self._z1()
        self._a1()
        self._z2()
        self._a2()

        # Cost.
        self._j()

        # Backward propagation.
        self._d2()
        self._dj_dw2()
        self._dj_db2()

        self._d1()
        self._dj_dw1()
        self._dj_db1()

        # Update weights and bias.
        self._w1()
        self._b1()
        self._w2()
        self._b2()

        return self.j

    def infer(self, image_path, expected_label):
        self.m = 1
        self.x = self.tensor_handler.zeros((self.dimensions, self.m))
        image_data = self.load_image_data(image_path)

        if self._config.testing:
            # Note we're not normalising the data here since we're using the label itself.
            self.tensor_handler.fill(self.x[:, 0], expected_label)
        else:
            self.x[:, 0] = image_data
            self.normalise_data()

        # Forward propagation.
        self._z1()
        self._a1()
        self._z2()
        self._a2()

        return self.a2
