# From FFJORD
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle


# Dataset iterator
def inf_train_gen(data, dim=2, rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X, random_state=rng)

        # Add noise
        X = X + rng.normal(scale=0.08, size=X.shape)

        return X.astype("float32")

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data

    elif data == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    elif data == "square":
        dataset = []
        # n = int(batch_size ** 0.5)
        # for i in range(n):
        #     for j in range(n):
        #         x = (j+0.5)/n
        #         y = (i+0.5)/n
        #         dataset.append([(y-0.5)*2, (x-0.5)*2])
        for i in range(batch_size):
            # point = (rng.rand(2) - 0.5)*2
            # dataset.append(point)

            point = (rng.rand(2) - 0.5)*2
            point[0] -= 1
            point[1] -= 2
            dataset.append(point)
            point = (rng.rand(2) - 0.5)*4
            point[0] -= 1
            point[1] += 2
            dataset.append(point)
            # point = (rng.rand(2) - 0.5)
            # point[1]*=2
            # point[0]*=0.2
            # point[0] -= 1
            # dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        return dataset

    elif data == "1gaussians":
        centers = [(0,0)]

        dataset = []
        values  = []
        sigma = 0.5 # variance
        for i in range(batch_size):
            # point = rng.randn(2) * 1.5
            point = rng.randn(2) * sigma
            dataset.append(point)
            val = (point[0])**2 + (point[1])**2
            values.append(np.exp(-val / (2.0 * sigma**2 ) ))
        dataset = np.array(dataset, dtype="float32")
        values  = np.array(values,  dtype="float32")
        return dataset, values

    elif data == "porous":
        dim = 6
        centers = [(0,0)]
        centers = [np.zeros((dim))]

        dataset = np.zeros((batch_size, dim), dtype="float32")
        values  = np.zeros((batch_size), dtype="float32")
        i = 0
        t0= 0.001

        m = 2
        A = (4 * np.pi * m * t0) ** ((1-m)/m)
        B = (m-1)/(4*m**2*t0)
        # while i < batch_size:
        #     x = (np.random.rand(dim)*2-1)*np.sqrt(A/B)
        #     xnorm2 = np.sum(x**2)
        #     rhoval = (A - B * xnorm2)**(1/(m-1))
        #     r = np.random.random() * A **(1/(m-1)) + 1e-3
        #     if r < rhoval:
        #         dataset[i] = x
        #         values[i]  = rhoval
        #         i+=1

        X = np.zeros((1,dim))
        V = np.zeros(1)
        while X.shape[0] <= batch_size:
            x = (np.random.rand(batch_size,dim)*2-1)*(np.sqrt(A/B))
            xnorm2 = np.sum(x**2,axis=1)
            rhoval = (A - B * xnorm2)**(1/(m-1))
            r = np.random.rand(batch_size) * A **(1/(m-1)) - 1
            I = r < rhoval
            X = np.vstack((X, x[I,:]))
            # print(f'rhoval: {rhoval.shape} V: {V.shape}')
            V = np.hstack((V, rhoval[I]))
        dataset = X[1:(batch_size+1)]
        V  = V[1:(batch_size+1)]
        return dataset, V

    elif data == "2gaussians":
        center1 = np.zeros(dim)
        centers = [center1]

        dataset = []
        values  = []
        i = 0
        idx= 0
        sigma = 0.5 # standard deviation
        while i < batch_size:
            i+=1
            
            point = np.random.randn(dim) * sigma
            center = centers[idx]
            point += center
            dataset.append(point)

            val = 0
            for j in range(len(centers)):
                val += np.exp(- (np.sum((point-centers[j])**2))/(2.0 * sigma**2 ))
            values.append(1.0/(np.sqrt(sigma * np.pi))**2 * val)
            idx = (idx+1)%len(centers)
        dataset = np.array(dataset, dtype="float32")
        values  = np.array(values,  dtype="float32")
        return dataset, values

        

        # dataset = []
        # values  = []
        # i = 0
        # while i < batch_size:
        #     i+=1
        #     point = (rng.rand(2) - 0.5)*2
        #     dataset.append(point)
        #     values.append(0.2)
        # dataset = np.array(dataset, dtype="float32")
        # values  = np.array(values,  dtype="float32")
        # return dataset, values

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    elif data == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    else:
        return inf_train_gen("8gaussians", rng, batch_size)
