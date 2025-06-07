from logging import getLogger

from src.config.config import Config
from src.impl import IImpl
from src.tensor.interfaces import ITensorHandler


log = getLogger(__name__)


class _NnReluImpl(IImpl):

    def __init__(self, config: Config, tensor_handler: ITensorHandler):
        super().__init__(config, tensor_handler)
        t = self.tensor_handler

        self.m = 0  # Sample count.
        self.n = self.dimensions  # Pixels * 3 (rbg).
        self.h = config.hidden_layer_size  # Hidden layer size.

        # He (Kaiming) initialisation. Avoids zero learning due to symmetry.
        self.w1 = t.randn((self.n, self.h)) * t.sqrt(t.as_tensor(2 / self.n))
        self.b1 = t.zeros((self.h, 1))
        self.w2 = t.randn((1, self.h)) * t.sqrt(t.as_tensor(2 / self.h))
        self.b2 = t.zeros((1, 1))

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
        z1 = self.tensor_handler.multiply(w1.T, x) + b1
        self.z1 = z1
        assert z1.shape == (self.h, self.m)

    def _a1(self):
        z1 = self.z1
        a1 = self.tensor_handler.relu(z1)
        self.a1 = a1
        assert a1.shape == (self.h, self.m)

    def _z2(self):
        w2 = self.w2
        b2 = self.b2
        a1 = self.a1
        z2 = self.tensor_handler.multiply(w2, a1) + b2
        self.z2 = z2
        assert z2.shape == (1, self.m)

    def _a2(self):
        z2 = self.z2
        a2 = 1 / (1 + self.tensor_handler.exp(-z2))
        self.a2 = a2
        assert a2.shape == (1, self.m)

    def _j(self):
        a2 = self.a2
        y = self.y
        m = self.m

        _log_a2 = self.tensor_handler.log(a2)
        _y_log_a2_T = self.tensor_handler.multiply(y, _log_a2.T)
        _log_1_a2 = self.tensor_handler.log(1 - a2)
        _1_y_log_1_a2_T = self.tensor_handler.multiply(1 - y, _log_1_a2.T)

        j = (-1 / m) * (_y_log_a2_T + _1_y_log_1_a2_T.T)
        self.j = j
        assert j.shape == (1,)

    def _d1(self):
        d2 = self.d2
        w2 = self.w2
        z1 = self.z1

        _d2_w2_T = self.tensor_handler.multiply(d2.T, w2)
        _relu_z1 = self.tensor_handler.d_relu(z1)

        d1 = (_d2_w2_T * _relu_z1.T).T
        self.d1 = d1
        assert d1.shape == (self.h, self.m)

    def _dj_dw1(self):
        d1 = self.d1
        x = self.x
        m = self.m
        dj_dw1 = (1 / m) * self.tensor_handler.multiply(d1, x.T).T
        self.dj_dw1 = dj_dw1
        assert dj_dw1.shape == (self.n, self.h)

    def _dj_db1(self):
        d1 = self.d1
        m = self.m
        dj_db1 = (1 / m) * self.tensor_handler.sum(d1)
        self.dj_db1 = dj_db1
        assert dj_db1.shape == ()

    def _d2(self):
        a2 = self.a2
        y = self.y
        d2 = a2 - y
        self.d2 = d2
        assert d2.shape == (1, self.m)

    def _dj_dw2(self):
        a1 = self.a1
        d2 = self.d2
        dj_dw2 = self.tensor_handler.multiply(a1, d2.T).T
        self.dj_dw2 = dj_dw2
        assert dj_dw2.shape == (1, self.h)

    def _dj_db2(self):
        d2 = self.d2
        m = self.m
        _dj_db2 = (1 / m) * self.tensor_handler.sum(d2)
        self.dj_db2 = _dj_db2
        assert _dj_db2.shape == ()

    def _w1(self):
        w1 = self.w1
        alpha = self.alpha
        dj_dw1 = self.dj_dw1
        w1 -= alpha * dj_dw1
        self.w1 = w1
        assert w1.shape == (self.n, self.h)

    def _b1(self):
        b1 = self.b1
        alpha = self.alpha
        dj_db1 = self.dj_db1
        b1 -= alpha * dj_db1
        self.b1 = b1
        assert b1.shape == (self.h, 1)

    def _b2(self):
        b2 = self.b2
        alpha = self.alpha
        dj_db2 = self.dj_db2
        b2 -= alpha * dj_db2
        self.b2 = b2
        assert b2.shape == (1, 1)

    def _w2(self):
        w2 = self.w2
        alpha = self.alpha
        dj_dw2 = self.dj_dw2
        w2 -= alpha * dj_dw2
        self.w2 = w2
        assert w2.shape == (1, self.h)

    def train_epoch(self):

        # Forward propagation.
        self._z1()
        self._a1()
        self._z2()
        self._a2()

        # Cost.
        self._j()

        # Backward propagation.
        self._d1()
        self._dj_dw1()
        self._dj_db1()

        self._d2()
        self._dj_dw2()
        self._dj_db2()

        # Update weights and bias.
        # Only do this at the end to not
        # cross-contaminate within an epoch.
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
