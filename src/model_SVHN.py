# third party import
import torch
import torch.nn as nn
import torch.nn.functional as F


VERBOSE = False
if VERBOSE:

    def verboseprint(*args, **kwargs):
        print(*args, **kwargs)

else:
    verboseprint = lambda *a, **k: None  # do-nothing function


class Encoder(nn.Module):
    def __init__(self, latent_size, num_classes, H, model):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.model = model

        # conv layers
        self.conv_1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=5, padding=2
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=5, padding=2
        )
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=5, padding=2
        )
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_4 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm2d(256)

        self.conv_5 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )
        self.bn5 = nn.BatchNorm2d(256)

        self.l1 = nn.Linear(in_features=4 * 4 * 256, out_features=H)

        # flatten
        self.flatten = nn.Flatten()

        # mlp
        if self.model in ["GFZ", "GBZ", "GFY", "GBY"]:
            self.l2 = nn.Linear(in_features=H + num_classes, out_features=H)
        elif self.model in ["DBX", "DFX", "DFZ"]:
            self.l2 = nn.Linear(in_features=H, out_features=H)

        self.l3 = nn.Linear(in_features=H, out_features=latent_size)
        self.l4 = nn.Linear(in_features=H, out_features=latent_size)

    def forward(self, x, y):
        x = self.pool1(F.relu(self.bn1(self.conv_1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv_2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv_3(x))))
        x = F.relu(self.bn4(self.conv_4(x)))
        x = F.relu(self.bn5(self.conv_5(x)))
        x = self.flatten(x)
        fea = self.l1(x)

        if self.model in ["GFZ", "GBZ", "GFY", "GBY"]:
            x = torch.cat([fea, y], 1)
        elif self.model in ["DBX", "DFX", "DFZ"]:
            x = fea

        x = F.relu(self.l2(x))
        mu = self.l3(x)
        log_var = self.l4(x)

        return mu, log_var, fea


class p_yz(nn.Module):
    def __init__(self, latent_size, num_classes, H):
        super(p_yz, self).__init__()
        # p_yz
        self.l_p_yz_1 = nn.Linear(in_features=latent_size, out_features=H)
        self.l_p_yz_2 = nn.Linear(in_features=H, out_features=num_classes)

    def forward(self, z):
        y_tild = self.l_p_yz_1(z)
        y_tild = self.l_p_yz_2(y_tild)

        return y_tild


class p_zy(nn.Module):
    def __init__(self, latent_size, num_classes, H):
        super(p_zy, self).__init__()
        # p_zy
        self.l1 = nn.Linear(in_features=num_classes, out_features=H)
        self.l_mu = nn.Linear(in_features=H, out_features=latent_size)
        self.l_log_var = nn.Linear(in_features=H, out_features=latent_size)

    def forward(self, y):
        z_tild = self.l1(y)
        mu_tild = self.l_mu(z_tild)
        log_var_tild = self.l_log_var(z_tild)
        return mu_tild, log_var_tild


class p_yxz(nn.Module):
    def __init__(self, latent_size, num_classes, H):
        super(p_yxz, self).__init__()
        # p_yxz
        self.l_p_yz_1 = nn.Linear(in_features=latent_size + H, out_features=H)
        self.l_p_yz_2 = nn.Linear(in_features=H, out_features=num_classes)

    def forward(self, fea, z):
        z = torch.cat([fea, z], 1)
        y_tild = self.l_p_yz_1(z)
        y_tild = self.l_p_yz_2(y_tild)

        return y_tild


class Decoder(nn.Module):
    def __init__(self, latent_size, num_classes, H, model, fea=None):
        super(Decoder, self).__init__()

        self.model = model
        if self.model in ["GFZ", "GBZ", "DBX"]:
            # p_yz
            self.p_yz = p_yz(latent_size, num_classes, H)
        elif self.model in ["GFY", "GBY"]:
            # p_yz
            self.p_zy = p_zy(latent_size, num_classes, H)
        elif self.model in ["DFX", "DFZ"]:
            # p_yxz
            self.p_yxz = p_yxz(latent_size, num_classes, H)

        if self.model in ["GFZ", "GFY"]:
            # p_xyz
            self.l_p_xyz_1 = nn.Linear(
                in_features=latent_size + num_classes, out_features=H
            )
        elif self.model in ["GBZ", "GBY", "DFZ"]:
            # p_xz
            self.l_p_xyz_1 = nn.Linear(in_features=latent_size, out_features=H)

        # reconstruct x
        self.l_p_xyz_2 = nn.Linear(in_features=H, out_features=4 * 4 * 256)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2)
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.deconv5 = nn.ConvTranspose2d(
            64, 3, kernel_size=5, stride=2, padding=0, output_padding=1
        )

    def forward(self, z, y, fea=None):
        verboseprint("0", z.shape, y.shape)
        if self.model in ["GFZ", "GBZ", "DBX"]:
            # p_yz
            temp_tild = self.p_yz(z)
        elif self.model in ["GFY", "GBY"]:
            # p_zy
            temp_tild = self.p_zy(y)
        elif self.model in ["DFX", "DFZ"]:
            # p_yxz
            temp_tild = self.p_yxz(fea, z)

        if self.model in ["GFZ", "GBZ", "GFY", "GBY", "DFZ"]:
            if self.model in ["GFZ", "GFY"]:
                # p_xyz
                x_tild = torch.cat([z, y], dim=1)
            elif self.model in ["GBZ", "GBY", "DFZ"]:
                # p_xy
                x_tild = z

            x_tild = F.relu(self.l_p_xyz_1(x_tild))
            x_tild = F.relu(self.l_p_xyz_2(x_tild))
            x_tild = x_tild.view(-1, 256, 4, 4)
            x_tild = F.relu(self.deconv1(x_tild))
            x_tild = F.relu(self.deconv2(x_tild))
            x_tild = self.deconv3(x_tild)
            x_tild = self.deconv4(x_tild)
            x_tild = self.deconv5(x_tild)

            return x_tild, temp_tild
        else:
            return temp_tild


class CVAE(nn.Module):
    def __init__(self, latent_size, num_classes, H, model):
        super(CVAE, self).__init__()

        self.model = model
        self.encoder = Encoder(latent_size, num_classes, H, model)
        self.decoder = Decoder(latent_size, num_classes, H, model)

    def predict(self, x, y):
        mu, log_var, _ = self.encoder(x, y)
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        x_tild, y_tild = self.decoder(z, y)
        return x_tild, y_tild

    def loss_py_z(self, y_tild, y):
        return -F.binary_cross_entropy_with_logits(y_tild, y, reduction="none").sum(-1)

    def lower_bound_GFZ(self, x, y, it):
        mu, log_var, _ = self.encoder(x, y)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        x_tild, y_tild = self.decoder(z, y)

        ### px_yz ###################
        log_px_yz = -((x_tild - x) ** 2)
        log_px_yz = log_px_yz.sum((1, 2, 3))
        #############################

        ###  py_z ####################
        log_py_z = self.loss_py_z(y_tild, y)
        ##############################

        ### KL divergence ############
        # pz
        pz = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        log_pz = pz.log_prob(z).sum(-1)

        # qz_yx
        q = torch.distributions.Normal(mu, std)
        log_qz_yx = q.log_prob(z).sum(-1)
        ##############################

        beta = 1
        alpha = 21

        bound = log_px_yz * beta + log_py_z * alpha + (log_pz - log_qz_yx)
        return bound

    def lower_bound_GFY(self, x, y, it):
        mu, log_var, _ = self.encoder(x, y)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        x_tild, z_tild = self.decoder(z, y)
        mu_pz_y, log_var_pz_y = z_tild
        std_pz_y = torch.exp(log_var_pz_y / 2)

        ### px_yz ####################
        log_px_yz = -((x_tild - x) ** 2)
        log_px_yz = log_px_yz.sum(dim=(1, 2, 3))
        ##############################

        ### KL divergence ############
        # p_zy
        pz_y = torch.distributions.Normal(mu_pz_y, std_pz_y)
        log_pz_y = pz_y.log_prob(z).sum(-1)
        # qz_yx
        q = torch.distributions.Normal(mu, std)
        log_qz_yx = q.log_prob(z).sum(-1)
        ##############################

        ### py #######################
        log_py = torch.log(torch.Tensor([0.1]))[0]
        ##############################
        beta = 1
        alpha = 21

        bound = log_px_yz * beta + log_py * alpha + (log_pz_y - log_qz_yx)
        return bound

    def lower_bound_DBX(self, x, y, it):
        mu, log_var, _ = self.encoder(x, y)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        y_tild = self.decoder(z, y)

        ###  py_z ####################
        log_py_z = self.loss_py_z(y_tild, y)
        ##############################

        ### KL divergence ############
        # pz
        pz = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        log_pz = pz.log_prob(z).sum(-1)

        # qz_x
        q = torch.distributions.Normal(mu, std)
        log_qz_x = q.log_prob(z).sum(-1)
        ##############################

        bound = 10 * log_py_z + (log_pz - log_qz_x)

        return bound

    def lower_bound_DFX(self, x, y, it):
        mu, log_var, fea = self.encoder(x, y)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        y_tild = self.decoder(z, y, fea)

        ###  py_z ####################
        log_py_z = self.loss_py_z(y_tild, y)
        ##############################

        ### KL divergence ############
        # pz
        pz = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        log_pz = pz.log_prob(z).sum(-1)

        # qz_x
        q = torch.distributions.Normal(mu, std)
        log_qz_x = q.log_prob(z).sum(-1)
        ##############################
        bound = 10 * log_py_z + (log_pz - log_qz_x)

        return bound

    def lower_bound_DFZ(self, x, y, it):
        mu, log_var, fea = self.encoder(x, y)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        x_tild, y_tild = self.decoder(z, y, fea)

        ### px_yz ###################
        log_px_yz = -((x_tild - x) ** 2)
        log_px_yz = log_px_yz.sum((1, 2, 3))
        #############################

        ###  py_xz ####################
        log_py_xz = self.loss_py_z(y_tild, y)
        ##############################

        ### KL divergence ############
        # pz
        pz = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        log_pz = pz.log_prob(z).sum(-1)

        # qz_yx
        q = torch.distributions.Normal(mu, std)
        log_qz_yx = q.log_prob(z).sum(-1)
        ##############################

        beta = 1
        alpha = 21

        bound = log_px_yz * beta + log_py_xz * alpha + (log_pz - log_qz_yx)
        return bound

    def forward(self, x, y, it=1):
        # Generative models
        if self.model in ["GFZ", "GBZ"]:
            # GFZ and GBZ are really similar
            # if self.model = GBZ then y is not used
            bound = self.lower_bound_GFZ(x, y, it)
        elif self.model in ["GFY", "GBY"]:
            # GFY and GBY are really similar
            # if self.model = GBY then y is not used
            bound = self.lower_bound_GFY(x, y, it)
        # Discriminative models
        elif self.model in ["DBX"]:
            bound = self.lower_bound_DBX(x, y, it)
        elif self.model in ["DFX"]:
            bound = self.lower_bound_DFX(x, y, it)
        elif self.model in ["DFZ"]:
            bound = self.lower_bound_DFZ(x, y, it)

        return bound
