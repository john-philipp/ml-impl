from logging import getLogger

from src.config.config import Config
from src.impl.interfaces import IImpl
from src.tensor.interfaces import ITensorHandler


log = getLogger(__name__)


class _LogRegImpl(IImpl):

    class State:
        def __init__(self, w, b):
            self.w = w
            self.b = b

    def __init__(self, config: Config, tensor_handler: ITensorHandler):
        super().__init__(config, tensor_handler)

        # Training output.
        self._w = self._th.zeros(self.dimensions)
        self._b = 0

        self._z = None  # Logit (raw score prior to activation: z = w.T dot x + b
        self._a = None  # Predictions. Activation function (sigmoid): 1 / (1 + e^-z)
        self._j = None  # Average cost over all samples.

        # Derivatives of J (cost) wrt to variable.
        self._dl_dz = None
        self._dw = None
        self._db = None

    def _update_z(self):
        self._z = self._th.multiply(self._w, self._X) + self._b

    def _update_a(self):
        self._a = 1 / (1 + self._th.exp(-self._z))

    def _update_j(self):
        self._j = (-1 / self.m) * (
            self._th.multiply(
                self._y,
                self._th.log(self._a)) + self._th.multiply(
                    1 - self._y, self._th.log(1 - self._a)))

    def _update_dl_dz(self):
        self._dl_dz = self._a - self._y

    def _update_dj_dw(self):
        self._dw = (1 / self.m) * self._th.multiply(self._X, self._dl_dz)

    def _update_dj_db(self):
        self._db = (1 / self.m) * self._th.sum(self._dl_dz)

    def _update_w(self):
        self._w -= self.alpha * self._dw

    def _update_b(self):
        self._b -= self.alpha * self._db

    def train_epoch(self):
        # Forward propagation.
        self._update_z()
        self._update_a()
        self._update_j()

        # Backward propagation.
        self._update_dl_dz()
        self._update_dj_db()
        self._update_dj_dw()

        # Update weights and bias.
        self._update_w()
        self._update_b()

        return self._j

    def test(self, image_path, expected_label):
        self.m = 1
        self._X = self._th.zeros((self.dimensions, self.m))
        image_data = self.load_image_data(image_path)

        if self._config.testing:
            # Note we're not normalising the data here since we're using the label itself.
            self._th.fill(self._X[:, 0], expected_label)
        else:
            self._X[:, 0] = image_data
            self.normalise_data()

        self._update_z()
        self._update_a()

        return self._a

    def get_state(self):
        return _LogRegImpl.State(self._w, self._b)

    def from_state(self, state: State):
        self._w = state.w
        self._b = state.b
