import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.patches import Ellipse
import matplotlib
matplotlib.use("QtAGG")


def show_gaussian(mean, cov, ax, text=""):
    u, s, v = np.linalg.svd(cov)
    width, height = np.sqrt(s[0]) * 3, np.sqrt(s[1]) * 3
    angle = np.arctan2(v[0, 1], v[0, 0]) * 360 / (2 * np.pi)
    el = Ellipse((mean[0], mean[1]), width, height, facecolor='none', angle=angle, zorder=8, edgecolor='red')
    el.set_alpha(1)
    el.set_clip_box(ax.bbox)
    ax.add_artist(el)
    ax.plot(mean[0], mean[1], 'r.', markersize=2)
    t = ax.text(mean[0], mean[1], text, fontsize=12)


class GaussianEmissionModel:
    def __init__(self, means, covariances):
        self.means = means
        self.covariances = covariances
        self.embedded_dimension = means.shape[1]
        self.number_of_states = means.shape[0]

    def get_observation(self, bubble, rng):
        return rng.multivariate_normal(self.means[bubble, :], self.covariances[bubble, :, :])

    def log_likelihood_of_obs_given_bubble(self, bubble, obs):
        return stats.multivariate_normal(self.means[bubble, :], self.covariances[bubble, :, :]).logpdf(obs).sum()

    def get_likelihood_across_bubbles(self, obs):
        likelyhoods = np.zeros(self.number_of_states)
        for i in range(self.number_of_states):
            likelyhoods[i] = stats.multivariate_normal(self.means[i, :], self.covariances[i, :, :]).logpdf(obs).sum()
        return np.exp(likelyhoods)

    def plot_onto_axis(self, ax, observations=None):
        assert self.embedded_dimension == 2
        for i in range(self.number_of_states):
            show_gaussian(self.means[i, :], self.covariances[i, :, :], ax, text=f"{i}")
        if observations is not None:
            ax.scatter(observations[:,0], observations[:,1])
        ax.axis("equal")

    @staticmethod
    def estimate_from_obs_and_gamma(observations, gamma, old, axes=None):
        means = np.zeros(old.means.shape)
        variances = np.zeros(old.covariances.shape)

        # g = np.argmax(gamma, axis=1)
        # means[k, :] = np.average(observations[g==k,:], axis=0)
        # variances[k, :, :] =  (g==k) * (observations - means[k, :]).T @ (observations - means[k, :])/(g==k).sum()

        for k in range(means.shape[0]):
            weights = gamma[:, k]
            means[k, :] = np.average(observations, weights=weights, axis=0)
            variances[k, :, :] = (weights * (observations - means[k, :]).T) @ (observations - means[k, :])/weights.sum()



            if axes is not None:
                ax = axes[k]
            else:
                fig, ax = plt.subplots(tight_layout=True)
            ax.scatter(observations[:, 0], observations[:, 1], c=weights)
            show_gaussian(means[k,:], variances[k,:,:], ax)
            ax.axis("equal")
            # plt.show()

        return GaussianEmissionModel(means, variances)
        # return old

    def to_matrix(self):
        return np.hstack((self.covariances.reshape(self.number_of_states, -1), self.means))


def make_p(X):
    "A utility function to convert numpy matrices into probability matrices."
    return X / X.sum(axis=1)[:, None]


class DiscreteEmissionModel:
    def __init__(self, observation_matrix):
        self.observation_matrix = observation_matrix
        self.number_of_states = observation_matrix.shape[0]
        self.number_of_output_characters = observation_matrix.shape[1]
        self.embedded_dimension = 1

    def get_observation(self, state, rng):
        pvec = self.observation_matrix[state, :]
        return rng.choice(self.number_of_output_characters, p=pvec)

    def log_likelihood_of_obs_given_bubble(self, bubble, obs):
        return np.log(self.observation_matrix[bubble, int(obs)])

    def get_likelihood_across_bubbles(self, obs):
        return self.observation_matrix[:, int(obs)]

    def plot_onto_axis(self, ax):
        ax.imshow(self.observation_matrix)

    @staticmethod
    def estimate_from_obs_and_gamma(observations, gamma, old):
        estimated_observation_matrix = np.zeros(old.observation_matrix.shape)
        for k in range(estimated_observation_matrix.shape[1]):
            estimated_observation_matrix[:, k] = gamma[np.squeeze(observations) == k, :].sum(axis=0) / gamma.sum(axis=0)
        return DiscreteEmissionModel(estimated_observation_matrix)

    def to_matrix(self):
        return self.observation_matrix


class HMM:
    def __init__(self, transition_matrix, emission_model, initial_distribution, mutation_function=None):
        """
        mutation_function should have the signature (HMM, time, [states]) -> (transition_matrix, emission_model, initial_distribution)
        """
        self.n_states = transition_matrix.shape[0]
        self.n_symbols = emission_model.number_of_states
        self.transition_matrix = transition_matrix
        self.emission_model = emission_model
        self.initial_distribution = initial_distribution
        self.mutation_function = mutation_function
        self.sanity_check()

    def sanity_check(self):
        # TODO: should this have all the conditions?
        assert np.allclose(self.transition_matrix.sum(axis=1), 1)
        assert np.all(self.transition_matrix >= 0)

    @staticmethod
    def gaussian_clock_hmm(n_states, p1=1., angle=0., variance_scale=1., radius=10):
        transition_matrix = np.eye(n_states)
        diag_indices = list(np.diag_indices(n_states))
        diag_indices[1] = np.roll(diag_indices[1], -1)
        transition_matrix[tuple(diag_indices)] = p1

        transition_matrix[np.diag_indices(n_states)] = 1-p1

        initial_distribution = np.zeros(n_states)
        initial_distribution[0] = 1

        data_dimension = 2
        means = np.zeros((n_states, data_dimension))
        variances = np.stack([np.eye(data_dimension)*variance_scale for _ in range(n_states)])

        for i in range(n_states):
            theta = 2 * np.pi * i / n_states + angle
            means[i, :] = np.array([np.cos(theta), np.sin(theta)]) * radius

        return HMM(transition_matrix, GaussianEmissionModel(means, variances), initial_distribution)


    @staticmethod
    def wandering_gaussian_clock_hmm(n_states, p1=1., angle=0., radius=10, speed=0):
        transition_matrix = np.eye(n_states)
        diag_indices = list(np.diag_indices(n_states))
        diag_indices[1] = np.roll(diag_indices[1], -1)
        transition_matrix[tuple(diag_indices)] = p1

        transition_matrix[np.diag_indices(n_states)] = 1-p1

        initial_distribution = np.zeros(n_states)
        initial_distribution[0] = 1

        data_dimension = 2
        means = np.zeros((n_states, data_dimension))
        variances = np.stack([np.eye(data_dimension) for _ in range(n_states)])

        for i in range(n_states):
            theta = 2 * np.pi * i / n_states + angle
            means[i, :] = np.array([np.cos(theta), np.sin(theta)]) * radius

        def mutation_function(hmm, time, rng):

            means = hmm.emission_model.means
            means = means + rng.normal(size=means.shape) * speed
            em = GaussianEmissionModel(means=means, covariances=hmm.emission_model.covariances)

            return (hmm.transition_matrix, em, hmm.initial_distribution)


        return HMM(transition_matrix, GaussianEmissionModel(means, variances), initial_distribution, mutation_function=mutation_function)

    @staticmethod
    def teetering_gaussian_clock_hmm(n_states, p0=0, p1=1., angle=0., rate=1., radius=10):
        """p1 is the probability of switching"""

        transition_matrix = np.eye(n_states)
        diag_indices = list(np.diag_indices(n_states))
        diag_indices[1] = np.roll(diag_indices[1], -1)
        transition_matrix[tuple(diag_indices)] = p1

        transition_matrix[np.diag_indices(n_states)] = 1-p1

        initial_distribution = np.zeros(n_states)
        initial_distribution[0] = 1

        data_dimension = 2
        means = np.zeros((n_states, data_dimension))
        variances = np.stack([np.eye(data_dimension) for _ in range(n_states)])

        for i in range(n_states):
            theta = 2 * np.pi * i / n_states + angle
            means[i, :] = np.array([np.cos(theta), np.sin(theta)]) * radius

        def mutation_function(hmm, time, rng):
            m_transition_matrix = np.eye(n_states)
            diag_indices = list(np.diag_indices(n_states))
            diag_indices[1] = np.roll(diag_indices[1], -1)

            mixing_v = (np.sin(time*rate)/2 + .5)
            switch_p = mixing_v * p1 + (1-mixing_v) * p0
            m_transition_matrix[tuple(diag_indices)] = switch_p

            m_transition_matrix[np.diag_indices(n_states)] = 1-switch_p

            return (m_transition_matrix, hmm.emission_model, hmm.initial_distribution)


        return HMM(transition_matrix, GaussianEmissionModel(means, variances), initial_distribution, mutation_function=mutation_function)


    @staticmethod
    def inverting_gaussian_clock_hmm(n_states,  mixing_p=1, p1=1, angle=0., rate=1, radius=10):
        "p1 is the probability of switching"
        transition_matrix = np.eye(n_states)
        transition_matrix[np.diag_indices(n_states)] = 1

        initial_distribution = np.zeros(n_states)
        initial_distribution[0] = 1

        data_dimension = 2
        means = np.zeros((n_states, data_dimension))
        variances = np.stack([np.eye(data_dimension) for _ in range(n_states)])

        for i in range(n_states):
            theta = 2 * np.pi * i / n_states + angle
            means[i, :] = np.array([np.cos(theta), np.sin(theta)]) * radius

        def mutation_function(hmm, time, rng):
            m_transition_matrix = np.eye(n_states)

            forward_diag_indices = list(np.diag_indices(n_states))
            forward_diag_indices[1] = np.roll(forward_diag_indices[1], -1)

            backward_diag_indices = list(np.diag_indices(n_states))
            backward_diag_indices[1] = np.roll(backward_diag_indices[1], 1)

            s = np.sin(time * rate) * p1
            v = np.array([s - -1,abs(s), 1-s])
            v = -v
            v = np.exp(v*mixing_p)
            v = v/v.sum()

            m_transition_matrix[tuple(forward_diag_indices)] = v[2]
            m_transition_matrix[tuple(backward_diag_indices)] = v[0]
            m_transition_matrix[np.diag_indices(n_states)] = v[1]


            return (m_transition_matrix, hmm.emission_model, hmm.initial_distribution)


        return HMM(transition_matrix, GaussianEmissionModel(means, variances), initial_distribution, mutation_function=mutation_function)

    @staticmethod
    def discrete_clock_hmm(n_states, p1=1.0):
        transition_matrix = np.eye(n_states)
        diag_indices = list(np.diag_indices(n_states))
        diag_indices[1] = np.roll(diag_indices[1], -1)
        transition_matrix[tuple(diag_indices)] = p1
        transition_matrix = make_p(transition_matrix)

        initial_distribution = np.zeros(n_states)
        initial_distribution[0] = 1

        observation_matrix = make_p(np.eye(n_states) + .2)

        return HMM(transition_matrix, DiscreteEmissionModel(observation_matrix), initial_distribution)

    # @staticmethod
    # def _distance_between(hmms, rng, n_steps=50, n_repeats=100):
    #     # TODO: look at setting the n_steps and repeats values to be reasonable
    #     N = len(hmms)
    #     observations_matrix = np.zeros((N, n_repeats, n_steps), dtype=int)
    #     distances = np.zeros((n_repeats, N, N))
    #     for j in range(N):
    #         for k in range(n_repeats):
    #             observations_matrix[j, k, :] = hmms[j].simulate(n_steps, rng)
    #
    #     for i in tqdm(range(N)):
    #         for j in range(N):
    #             for k in range(n_repeats):
    #                 distances[k, i, j] = hmms[i].log_probability_of_observations(
    #                     observations_matrix[j, k, :]
    #                 )
    #     return distances / n_steps

    # @staticmethod
    # def distance_between(hmms, rng, n_steps=50, n_repeats=100):
    #     return HMM._distance_between(hmms, rng, n_steps=n_steps, n_repeats=n_repeats).mean(axis=0)

    def simulate_with_states(self, n_steps, rng):
        observations = np.zeros((n_steps, self.emission_model.embedded_dimension))
        states = np.zeros(n_steps, dtype=int)

        states[0] = rng.choice(self.transition_matrix.shape[0], p=self.initial_distribution)

        observations[0, :] = self.emission_model.get_observation(states[0], rng)

        for t in range(1, n_steps):
            if self.mutation_function is not None:
                transition_matrix, emission_model, initial_distribution = self.mutation_function(self,t, rng)
                self.transition_matrix = transition_matrix
                self.emission_model = emission_model
                self.initial_distribution = initial_distribution

            pvec = self.transition_matrix[states[t - 1], :]
            states[t] = rng.choice(self.transition_matrix.shape[0], p=pvec)

            observations[t, :] = self.emission_model.get_observation(states[t], rng)
        return states, observations

    def simulate(self, n_steps, rng):
        return self.simulate_with_states(n_steps, rng)[1]

    def log_probability_of_state_sequence(self, states):
        p = np.log(self.initial_distribution[states[0]])
        for i in np.arange(1, len(states)):
            p = p + np.log(self.transition_matrix[states[i - 1], states[i]])
        return p

    def log_probability_of_observations_given_states(self, observations, states):
        p = 0
        for i in range(len(states)):
            p = p + self.emission_model.log_likelihood_of_obs_given_bubble(bubble=states[i], obs=observations[i, :])
        return p

    def log_probability_of_observations(self, observations):
        alpha_hat, alpha_scale = self.forward_algorithm(observations)
        return -np.log(alpha_scale).sum()

    def forward_algorithm(self, observations):
        T = len(observations)
        alpha_hat = np.zeros((T, self.n_states))
        alpha_scale = np.zeros(T)

        # pdb.set_trace()
        alpha_hat[0, :] = (
                self.initial_distribution * self.emission_model.get_likelihood_across_bubbles(
            obs=observations[0, :])
        )

        alpha_scale[0] = 1 / alpha_hat[0, :].sum()
        alpha_hat[0, :] = alpha_hat[0, :] * alpha_scale[0]
        for i in range(1, T):
            alpha_hat[i, :] = (
                    alpha_hat[i - 1, :]
                    @ self.transition_matrix
                    * self.emission_model.get_likelihood_across_bubbles(obs=observations[i, :])
            )
            alpha_scale[i] = 1 / alpha_hat[i, :].sum()
            alpha_hat[i, :] = alpha_hat[i, :] * alpha_scale[i]
        return alpha_hat, alpha_scale

    def backward_algorithm(self, observations):
        # TODO: this one doesn't make as much sense to me yet
        T = len(observations)
        beta_hat = np.zeros((T, self.n_states))
        beta_scale = np.zeros(T)

        beta_hat[T - 1, :] = 1
        beta_scale[T - 1] = 1 / beta_hat[T - 1, :].sum()
        beta_hat[T - 1, :] = beta_hat[T - 1, :] * beta_scale[T - 1]
        for i in range(T - 1, 0, -1):
            beta_hat[i - 1, :] = (
                                         beta_hat[i, :] * self.emission_model.get_likelihood_across_bubbles(
                                     obs=observations[i, :])
                                 ) @ self.transition_matrix.T
            beta_scale[i - 1] = 1 / beta_hat[i - 1, :].sum()
            beta_hat[i - 1, :] = beta_hat[i - 1, :] * beta_scale[i - 1]
        return beta_hat, beta_scale

    def viterbi_algorithm(self, observations):
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        delta_scale = np.zeros(T)
        psi = np.zeros((T, self.n_states), dtype=int)

        delta[0, :] = (
                self.initial_distribution * self.emission_model.get_likelihood_across_bubbles(
            obs=observations[0, :])
        )
        delta_scale[0] = 1 / delta[0, :].sum()
        delta[0, :] = delta[0, :] * delta_scale[0]
        for t in range(1, T):
            for j in range(0, self.n_states):
                probs = delta[t - 1, :] * self.transition_matrix[:, j]
                delta[t, j] = (
                        np.max(probs) * self.emission_model.log_likelihood_of_obs_given_bubble(j, observations[t, :])
                )
                psi[t, j] = np.argmax(probs)
            delta_scale[t] = 1 / delta[t, :].sum()
            delta[t, :] = delta[t, :] * delta_scale[t]

        max_path = np.zeros(len(observations), dtype=int)
        max_path[-1] = np.argmax(delta[-1, :])
        for t in range(T - 2, -1, -1):
            max_path[t] = psi[t + 1, max_path[t + 1]]
        return (delta, delta_scale), psi[1:, :], max_path

    @ staticmethod
    def baum_welch_step(hmm, observations, fig, axs):
        T = len(observations)
        alpha_hat, alpha_scale = hmm.forward_algorithm(observations)
        beta_hat, beta_scale = hmm.backward_algorithm(observations)

        xi = np.zeros((T, hmm.n_states, hmm.n_states))
        for t in range(T - 1):
            xi[t, :, :] = (
                    alpha_hat[t, :, None]
                    @ (
                            beta_hat[t + 1, :]
                            * hmm.emission_model.get_likelihood_across_bubbles(observations[t + 1])[None, :]
                    )
                    * hmm.transition_matrix
            )
            xi[t, :, :] = xi[t, :, :] / xi[t, :, :].sum()
        gamma = xi.sum(axis=2)

        estimated_transition_matrix = xi.sum(axis=0) / gamma.sum(axis=0)[:, None]

        estimated_initial_distribution = gamma[0, :]


        # axes = [axs[1,2], axs[0,1], axs[1,0], axs[2,1]]
        axes = [axs[1,2], axs[0,2], axs[0,1], axs[0,0], axs[1,0], axs[2,0], axs[2,1], axs[2,2]]

        emission_model = hmm.emission_model.estimate_from_obs_and_gamma(observations, gamma, hmm.emission_model, axes)

        new_hmm = HMM(estimated_transition_matrix, emission_model, estimated_initial_distribution)
        # new_hmm.emission_model.plot_onto_axis(axs)
        hmm.emission_model.plot_onto_axis(axs[1,1], observations=observations)

        return new_hmm

    @staticmethod
    def _baum_welch_fit(hmm, observations, tol=1e-5, max_iter=500):
        from matplotlib.animation import HTMLWriter
        fig, axs = plt.subplots(3,3)
        moviewriter = HTMLWriter()
        moviewriter.setup(fig, "a.html", dpi=150)

        old_estimate = estimate = hmm
        for _ in range(max_iter):
            estimate = HMM.baum_welch_step(old_estimate, observations, fig, axs)

            if np.linalg.norm(estimate.to_matrix() - old_estimate.to_matrix()) < tol:
                break

            old_estimate = estimate
            moviewriter.grab_frame()
            for ax in axs.flatten():
                ax.cla()
        moviewriter.grab_frame()

        moviewriter.finish()
        return estimate

    def baum_welch_fit(self, observations, tol=1e-5, max_iter=500):
        HMM._baum_welch_fit(self, observations, tol, max_iter)

    def to_matrix(self):
        return np.hstack((self.initial_distribution[:,None], self.transition_matrix, self.emission_model.to_matrix()))


    def show(self):
        fig, ax = plt.subplots(nrows=1, ncols=3)
        im = ax[0].imshow(self.initial_distribution[:, None], clim=[0, 1])

        ax[1].set_ylabel("from state")
        ax[1].set_xlabel("to state")
        im = ax[1].imshow(self.transition_matrix, clim=[0, 1])

        self.emission_model.plot_onto_axis(ax[2])

        fig.colorbar(im)

def generate_to_save(actually_save=False):
    rng = np.random.default_rng()

    rate = .003
    states, observations = HMM.gaussian_clock_hmm(n_states=8, radius=20, variance_scale=.1, p1=.5,).simulate_with_states(10_000, rng)
    # states, observations = HMM.teetering_gaussian_clock_hmm(n_states=8, radius=12, p0=.1, p1=.9, rate=rate).simulate_with_states(10000, rng)
    # states, observations = HMM.inverting_gaussian_clock_hmm(n_states=8, p1=.5, mixing_p=2, rate=rate, radius=12).simulate_with_states(10_000, rng)
    # states, observations = HMM.wandering_gaussian_clock_hmm(n_states=8, p1=.5, speed=.1, radius=12).simulate_with_states(10_000, rng)

    order = rng.permuted(np.arange(observations.shape[0]))
    observations = observations[order,:]
    states = states[order]

    if not actually_save:
        plt.plot(states, '.')
        plt.plot(np.sin(np.arange(len(states)) * rate))
        plt.xlabel("time")
        plt.ylabel("state")
        plt.show()

    import datetime
    import os
    time_string = datetime.datetime.now().strftime('%m-%d-%H-%M')
    outfile = os.path.join("/home/jgould/Documents/Bubblewrap/generated", f"clock-{time_string}.npz")
    print(outfile)

    if actually_save:
        np.savez(outfile, x=states, y=observations[None,:])

def just_run():
    rng = np.random.default_rng()
    hmm = HMM.gaussian_clock_hmm(8, 0.5, radius=12)

    states, observations = hmm.simulate_with_states(1000, rng)

    plt.plot(states, '.')
    plt.plot(np.sin(np.arange(len(states)) * .02))
    plt.xlabel("time")
    plt.ylabel("state")
    plt.show()

if __name__ == '__main__':
    generate_to_save(True)
