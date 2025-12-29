
def forward(self, x, **kwargs):
    b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
    assert n == feature_size, f'number of variable must be {feature_size}'
    t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
    return self._train_loss(x_start=x, t=t, **kwargs)


def _train_loss(self, x_start, t, target=None, noise=None, padding_masks=None):
    noise = default(noise, lambda: torch.randn_like(x_start))
    if target is None:
        target = x_start

    x = self.q_sample(x_start=x_start, t=t, noise=noise)  # noise sample
    model_out = self.output(x, t, padding_masks)

    train_loss = self.loss_fn(model_out, target, reduction='none')

    fourier_loss = torch.tensor([0.])
    if self.use_ff:
        fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
        fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
        fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
        fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none') \
                       + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
        train_loss += self.ff_weight * fourier_loss

    train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
    train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
    return train_loss.mean()
