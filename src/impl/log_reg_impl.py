from logging import getLogger

from src.config.config import Config
from src.impl.interfaces import IImpl
from src.tensor.interfaces import ITensorHandler


log = getLogger(__name__)


class _LogRegImpl(IImpl):

    def __init__(self, config: Config, tensor_handler: ITensorHandler):
        super().__init__(config, tensor_handler)

        self.w = self.tensor_handler.zeros(self.dimensions)
        self.b = 0

        self.m = 0  # Sample count.
        self.a = None  # Predictions. Activation function (sigmoid): 1 / (1 + e^-z)
        self.x = None  # Inputs.
        self.y = None  # Truth.

        self.j = None  # Average cost over all samples.
        self.z = None  # Logit (raw score prior to activation: z = w.T dot x + b

        # Derivatives of J (cost) wrt to variable.
        self.dl_dz = None
        self.dw = None
        self.db = None

    def _update_z(self):
        self.z = self.tensor_handler.multiply(self.w, self.x) + self.b

    def _update_a(self):
        self.a = 1 / (1 + self.tensor_handler.exp(-self.z))

    def _update_j(self):
        self.j = (-1 / self.m) * (
            self.tensor_handler.multiply(
                self.y,
                self.tensor_handler.log(self.a)) + self.tensor_handler.multiply(
                    1 - self.y, self.tensor_handler.log(1 - self.a)))

    def _update_dl_dz(self):
        self.dl_dz = self.a - self.y

    def _update_dj_dw(self):
        self.dw = (1 / self.m) * self.tensor_handler.multiply(self.x, self.dl_dz)

    def _update_dj_db(self):
        self.db = (1 / self.m) * self.tensor_handler.sum(self.dl_dz)

    def _update_w(self):
        self.w -= self.alpha * self.dw

    def _update_b(self):
        self.b -= self.alpha * self.db

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

        self._update_z()
        self._update_a()

        return self.a
