from quicksom.som import SOM
import numpy as np
import torch
import time


class customSOM(SOM):
    def __init__(self, m, n, dim,
                 alpha=None,
                 sigma=None,
                 niter=2,
                 sched='linear',
                 device='cpu',
                 precompute=True,
                 periodic=False,
                 p_norm=2):
        super().__init__(m, n, dim,
                         alpha,
                         sigma,
                         niter,
                         sched,
                         device,
                         precompute,
                         periodic,
                         p_norm)

    def scheduler(self, it, tot):
        if self.sched == 'linear':
            b = (self.niter - 1) / (100 - 1)
            return b / (it + b)
        # half the lr 20 times
        if self.sched == 'half':
            return 0.5 ** int(20 * it / tot)
        # decay from 1 to exp(-5)
        if self.sched == 'exp':
            return np.exp(- 5 * it / tot)
        raise NotImplementedError('Wrong value of "sched"')

    def __call__(self, x, learning_rate_op, step=0):
        """
        timing info : now most of the time is in pdist ~1e-3s and the rest is 0.2e-3
        :param x: the minibatch
        :param learning_rate_op: the learning rate to apply to the batch
        :return:
        """
        # Make an inference call

        # Compute distances from batch to centroids
        x, batch_size = self.find_batchsize(x)
        dists = torch.cdist(x, self.centroids, p=self.p_norm)

        # Find closest and retrieve the gaussian correlation matrix for each point in the batch
        # bmu_loc is BS, num points
        mindist, bmu_index = torch.min(dists, -1)
        bmu_loc = self.locations[bmu_index].reshape(batch_size, 2)

        # Compute the update

        # Update LR
        alpha_op = self.alpha * learning_rate_op
        sigma_op = self.sigma - step * 4 / 99  # custom
        self.alpha_op = alpha_op if alpha_op >= 0.05 else 0.05  # tuning phase
        self.sigma_op = sigma_op
        if self.precompute:
            bmu_distance_squares = self.distance_mat[bmu_index].reshape(batch_size, self.grid_size)
        else:
            bmu_distance_squares = []
            for loc in bmu_loc:
                bmu_distance_squares.append(self.get_bmu_distance_squares(loc))
            bmu_distance_squares = torch.stack(bmu_distance_squares)
        neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, 2 * sigma_op ** 2 + 1e-5)))
        learning_rate_multiplier = alpha_op * neighbourhood_func

        # Take the difference of centroids with centroids and weight it with gaussian
        # x is (BS,1,dim)
        # self.weights is (grid_size,dim)
        # delta is (BS, grid_size, dim)
        expanded_x = x.expand(-1, self.grid_size, -1)
        expanded_weights = self.centroids.unsqueeze(0).expand((batch_size, -1, -1))
        delta = expanded_x - expanded_weights
        delta = torch.mul(learning_rate_multiplier.reshape(*learning_rate_multiplier.size(), 1).expand_as(delta), delta)

        # Perform the update by taking the mean
        delta = torch.mean(delta, dim=0)
        new_weights = torch.add(self.centroids, delta)
        self.centroids = new_weights
        return bmu_loc, torch.mean(mindist)

    def fit(self, samples, batch_size=20, n_iter=None, print_each=20):
        if self.alpha is None:
            self.alpha = float((self.m * self.n) / samples.shape[0])
        if n_iter is None:
            n_iter = self.niter
        n_steps_periter = len(samples) // batch_size
        total_steps = n_iter * n_steps_periter

        step = 0
        start = time.perf_counter()
        learning_error = list()
        for iter_no in range(n_iter):
            order = np.random.choice(len(samples), size=n_steps_periter, replace=False)
            for counter, index in enumerate(order):
                lr_step = self.scheduler(step, total_steps)
                bmu_loc, error = self.__call__(samples[index:index + batch_size], learning_rate_op=lr_step,
                                               step=iter_no)
                learning_error.append(error)
                if not step % print_each:
                    print(f'{iter_no + 1}/{n_iter}: {batch_size * (counter + 1)}/{len(samples)} '
                          f'| alpha: {self.alpha_op:4f} | sigma: {self.sigma_op:4f} '
                          f'| error: {error:4f} | time {time.perf_counter() - start:4f}')
                step += 1
        self.compute_umat()
        self.compute_all_dists()
        return learning_error

    if __name__ == '__main__':
        pass

