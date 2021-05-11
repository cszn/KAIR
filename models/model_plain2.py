from models.model_plain import ModelPlain

class ModelPlain2(ModelPlain):
    """Train with two inputs (L, C) and with pixel loss"""

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        self.C = data['C'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed (L, C) to netG and get E
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L, self.C)

