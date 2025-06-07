from logging import getLogger

from src.config.config import Config
from src.impl import IImpl
from src.tensor.interfaces import ITensorHandler


log = getLogger(__name__)


class _NnReluImpl(IImpl):

    def __init__(self, config: Config, tensor_handler: ITensorHandler):
        super().__init__(config, tensor_handler)
        self._th = self.tensor_handler

        self.m = 0  # Sample count.
        self.n = self.dimensions  # Pixels * 3 (rbg).
        self.h = config.hidden_layer_size  # Hidden layer size.

        # He (Kaiming) initialisation. Avoids zero learning due to symmetry.
        self._X = None
        self._W1 = self._th.randn((self.n, self.h)) * self._th.sqrt(self._th.as_tensor(2 / self.n))
        self._b1 = self._th.zeros((self.h, 1))
        self._w2 = self._th.randn((self.h, 1)) * self._th.sqrt(self._th.as_tensor(2 / self.h))
        self._b2 = self._th.zeros((1, 1))

        self._A1 = None
        self._z1 = None
        self._a2 = None
        self._z2 = None

        self._d1 = None
        self._d2 = None

        self._j = None

        self._dj_dw2 = None
        self._dj_db2 = None
        self._dj_dW1 = None
        self._dj_db1 = None

    def _update_z1(self):
        _W1 = self._W1
        _b1 = self._b1
        _X = self._X
        _z1 = self._th.multiply(_W1.T, _X) + _b1
        self._z1 = _z1
        assert _z1.shape == (self.h, self.m)

    def _update_A1(self):
        _z1 = self._z1
        _a1 = self._th.relu(_z1)
        self._A1 = _a1
        assert _a1.shape == (self.h, self.m)

    def _update_z2(self):
        _w2 = self._w2
        _b2 = self._b2
        _a1 = self._A1
        _z2 = self._th.multiply(_w2.T, _a1) + _b2
        self._z2 = _z2
        assert _z2.shape == (1, self.m)

    def _update_a2(self):
        _z2 = self._z2
        _a2 = 1 / (1 + self._th.exp(-_z2))
        self._a2 = _a2
        assert _a2.shape == (1, self.m)

    def _update_j(self):
        _a2 = self._a2
        _y = self.y
        _m = self.m

        _log_a2 = self._th.log(_a2)
        _y_log_a2_T = self._th.multiply(_y, _log_a2.T)
        _log_1_a2 = self._th.log(1 - _a2)
        _1_y_log_1_a2_T = self._th.multiply(1 - _y, _log_1_a2.T)

        _j = (-1 / _m) * (_y_log_a2_T + _1_y_log_1_a2_T.T)
        self._j = _j
        assert _j.shape == (1,)

    def _update_d2(self):
        _a2 = self._a2
        _y = self.y
        _d2 = _a2 - _y
        self._d2 = _d2
        assert _d2.shape == (1, self.m)

    def _update_dj_dw2(self):
        _A1 = self._A1
        _d2 = self._d2
        _m = self.m
        _dj_dw2 = (1 / _m) * self._th.multiply(_d2, _A1.T).T
        self._dj_dw2 = _dj_dw2
        assert _dj_dw2.shape == (self.h, 1)

    def _update_dj_db2(self):
        _d2 = self._d2
        _m = self.m
        _dj_db2 = (1 / _m) * self._th.sum(_d2)
        self._dj_db2 = _dj_db2
        assert _dj_db2.shape == ()

    def _update_d1(self):
        _d2 = self._d2
        _w2 = self._w2
        _z1 = self._z1

        _j1 = self._th.d_relu(_z1)
        _d1 = (_w2 * _j1)

        self._d1 = _d1
        assert _d1.shape == (self.h, self.m), _d1.shape

    def _update_dj_dW1(self):
        _d2 = self._d2
        _d1 = self._d1
        _X = self._X
        _m = self.m
        _diag_d2 = self._th.diag(_d2.flatten())
        _d1_diag_d2 = self._th.multiply(_d1, _diag_d2)
        _dj_dW1 = (1 / _m) * self._th.multiply(_X, _d1_diag_d2.T)
        self._dj_dW1 = _dj_dW1
        assert _dj_dW1.shape == (self.n, self.h)

    def _update_dj_db1(self):
        _d1 = self._d1
        _m = self.m
        _dj_db1 = (1 / _m) * self._th.sum(_d1)
        self._dj_db1 = _dj_db1
        assert _dj_db1.shape == ()

    def _update_W1(self):
        _W1 = self._W1
        _alpha = self.alpha
        _dj_dW1 = self._dj_dW1
        _W1 -= _alpha * _dj_dW1
        self._W1 = _W1
        assert _W1.shape == (self.n, self.h)

    def _update_b1(self):
        _b1 = self._b1
        _alpha = self.alpha
        _dj_db1 = self._dj_db1
        _b1 -= _alpha * _dj_db1
        self._b1 = _b1
        assert _b1.shape == (self.h, 1)

    def _update_b2(self):
        _b2 = self._b2
        _alpha = self.alpha
        _dj_db2 = self._dj_db2
        _b2 -= _alpha * _dj_db2
        self._b2 = _b2
        assert _b2.shape == (1, 1)

    def _update_w2(self):
        _w2 = self._w2
        _alpha = self.alpha
        _dj_dw2 = self._dj_dw2
        _w2 -= _alpha * _dj_dw2
        self._w2 = _w2
        assert _w2.shape == (self.h, 1)

    def train_epoch(self):

        # Forward propagation.
        self._update_z1()
        self._update_A1()
        self._update_z2()
        self._update_a2()

        # Cost.
        self._update_j()

        # Backward propagation.
        self._update_d2()
        self._update_dj_dw2()
        self._update_dj_db2()

        self._update_d1()
        self._update_dj_dW1()
        self._update_dj_db1()

        # Update weights and bias.
        # Only do this at the end to not
        # cross-contaminate within an epoch.
        self._update_W1()
        self._update_b1()
        self._update_w2()
        self._update_b2()

        return self._j

    def infer(self, image_path, expected_label):
        self.m = 1
        self._X = self._th.zeros((self.dimensions, self.m))
        image_data = self.load_image_data(image_path)

        if self._config.testing:
            # Note we're not normalising the data here since we're using the label itself.
            self._th.fill(self._X[:, 0], expected_label)
        else:
            self._X[:, 0] = image_data
            self.normalise_data()

        # Forward propagation.
        self._update_z1()
        self._update_A1()
        self._update_z2()
        self._update_a2()

        return self._a2
