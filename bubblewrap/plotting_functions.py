import numpy as np
import matplotlib
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse
from math import atan2, floor

import os
if os.environ.get("display") is not None:
    matplotlib.use('QtAgg')


def plot_2d(ax, data, bw):
    A = bw.A
    mu = bw.mu
    L = bw.L
    n_obs = np.array(bw.n_obs)
    # todo: remove bw
    ax.cla()
    ax.scatter(data[:,0], data[:,1], s=5, color='#004cff', alpha=np.power(1-bw.eps, np.arange(data.shape[0], 0, -1)))
    for n in np.arange(A.shape[0]):
        if n not in bw.dead_nodes:
            el = np.linalg.inv(L[n])
            sig = el.T @ el
            u,s,v = np.linalg.svd(sig)
            width, height = np.sqrt(s[0])*3, np.sqrt(s[1])*3
            angle = atan2(v[0,1],v[0,0])*360 / (2*np.pi)
            el = Ellipse((mu[n,0], mu[n,1]), width, height, angle=angle, zorder=8)
            el.set_alpha(0.4)
            el.set_clip_box(ax.bbox)
            el.set_facecolor('#ed6713')
            ax.add_artist(el)
            d = min(width,height)
            ax.text(mu[n,0] + .5*d,mu[n,1] +.5*d,str(n), clip_on=True)
        else:
            el = np.linalg.inv(L[n])
            sig = el.T @ el
            u,s,v = np.linalg.svd(sig)
            width, height = np.sqrt(s[0])*3, np.sqrt(s[1])*3
            angle = atan2(v[0,1],v[0,0])*360 / (2*np.pi)
            el = Ellipse((mu[n,0], mu[n,1]), width, height, angle=angle, zorder=8)
            el.set_alpha(0.05)
            el.set_clip_box(ax.bbox)
            el.set_facecolor('#000000')
            ax.add_artist(el)

    mask = np.ones(mu.shape[0], dtype=bool)
    mask[n_obs < .1] = False
    mask[bw.dead_nodes] = False
    ax.scatter(mu[mask, 0], mu[mask, 1], c='k', zorder=10)
    ax.scatter(data[0,0], data[0,1], color="#004cff", s=10)


def plot_current_2d(ax, data, bw):
    A = bw.A
    mu = bw.mu
    L = bw.L
    n_obs = np.array(bw.n_obs)
    # todo: remove bw
    ax.cla()
    ax.scatter(data[:,0], data[:,1], s=5, color='#004cff', alpha=np.power(1-bw.eps, np.arange(data.shape[0], 0, -1)))
    ax.scatter(data[-1,0], data[-1,1], s=10, color='red')

    to_draw = np.argsort(np.array(bw.alpha))[-3:]
    opacities = np.array(bw.alpha)[to_draw]
    opacities = opacities * .5/opacities.max()

    for i, n in enumerate(to_draw):
        el = np.linalg.inv(L[n])
        sig = el.T @ el
        u,s,v = np.linalg.svd(sig)
        width, height = np.sqrt(s[0])*3, np.sqrt(s[1])*3
        angle = atan2(v[0,1],v[0,0])*360 / (2*np.pi)
        el = Ellipse((mu[n,0], mu[n,1]), width, height, angle=angle, zorder=8)
        el.set_alpha(opacities[i])
        el.set_clip_box(ax.bbox)
        el.set_facecolor('#ed6713')
        ax.add_artist(el)
        d = min(width,height)
        ax.text(mu[n,0] + .25*d,mu[n,1] + .25*d,str(n), alpha=min(opacities[i] * 2,1), clip_on=True)


def br_plot_3d(br):
    # TODO: make a plot_3d like above
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # Set of all spherical angles to draw our ellipsoid
    n_points = 10
    theta = np.linspace(0, 2*np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)

    # Get the xyz points for plotting
    # Cartesian coordinates that correspond to the spherical angles:
    X = np.outer(np.cos(theta), np.sin(phi))
    Y = np.outer(np.sin(theta), np.sin(phi)).flatten()
    Z = np.outer(np.ones_like(theta), np.cos(phi)).flatten()
    old_shape = X.shape
    X = X.flatten()


    s = np.load(br.file)
    data = s['y'][0]

    A = br.A
    mu = br.mu
    L = br.L
    n_obs = br.n_obs

    # TODO: make these not lists
    pred = br.pred_list[:,0]
    entropy = br.entropy_list[:,0]

    ax.plot(data[:,0], data[:,1], data[:,2], color='gray', alpha=0.8)
    for n in np.arange(A.shape[0]):
        if n_obs[n] > 1e-4:
            el = np.linalg.inv(L[n]).T
            sig = el @ el.T
            # Find and sort eigenvalues to correspond to the covariance matrix
            eigvals, eigvecs = np.linalg.eigh(sig)
            idx = np.sum(sig,axis=0).argsort()
            eigvals_temp = eigvals[idx]
            idx = eigvals_temp.argsort()
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:,idx]

            # Width, height and depth of ellipsoid
            nstd = 3
            rx, ry, rz = nstd * np.sqrt(eigvals)

            # Rotate ellipsoid for off axis alignment
            a,b,c = np.matmul(eigvecs, np.array([X*rx,Y*ry,Z*rz]))
            a,b,c = a.reshape(old_shape), b.reshape(old_shape), c.reshape(old_shape)

            # Add in offsets for the mean
            a = a + mu[n,0]
            b = b + mu[n,1]
            c = c + mu[n,2]

            ax.plot_surface(a, b, c, color='#ff4400', alpha=0.6)

    ax.view_init(40,23)

    mask = np.ones(mu.shape[0], dtype=bool)
    mask[n_obs<1e-4] = False
    ax.scatter(mu[mask,0], mu[mask,1], mu[mask,2], c='k' , zorder=10)

    ax.set_xticks([200, 600, 1000, 1400])
    ax.set_yticks([-20, -10, 0, 10])
    ax.set_zticks([-1400, -1000, -600, -200])
    in1, in2 = 0, 1
    ax.text(in1, in2, 100, s='b', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    plt.show()


def plot_A_differences(A, end=100):
    fig, ax = plt.subplots()
    last_A = A
    differences = []
    for _ in range(2,end):
        new_A = last_A @ A
        differences.append(np.linalg.norm(last_A - new_A))
        last_A = new_A
    plt.plot(differences)




def show_bubbles(ax, data, bw, params, step, i, keep_every_nth_frame):
    ax.cla()

    d = data[0:i+params["M"]+step]
    plot_2d(ax, d, bw)

    d = data[i + params["M"] - (keep_every_nth_frame - 1) * step:i + params["M"] + step]
    ax.plot(d[:,0], d[:,1], 'k.')
    ax.set_title(f"Observation Model (Bubbles) (i={i})")
    ax.set_xlabel("neuron 1")
    ax.set_ylabel("neuron 2")


def show_inhabited_bubbles(ax, data, bw, params, step, i, keep_every_nth_frame):
    ax.cla()

    d = data[0:i+params["M"]+step]
    plot_current_2d(ax, d, bw)

    d = data[i + params["M"] - (keep_every_nth_frame - 1) * step:i + params["M"] + step]
    ax.set_title(f"Currrent Bubbles (i={i})")
    ax.set_xlabel("neuron 1")
    ax.set_ylabel("neuron 2")

def show_A(ax, bw):
    ax.cla()
    ims = ax.imshow(bw.A, aspect='equal', interpolation='nearest')

    ax.set_title("Transition Matrix (A)")
    ax.set_xlabel("To")
    ax.set_ylabel("From")

    ax.set_xticks(np.arange(bw.N))
    live_nodes = [x for x in np.arange(bw.N) if x not in bw.dead_nodes]
    ax.set_yticks(live_nodes)

def show_D(ax, bw):
    ax.cla()
    ims = ax.imshow(bw.D, aspect='equal', interpolation='nearest')
    ax.set_title("D")

def show_Ct_y(ax, bw):
    old_ylim = ax.get_ylim()
    ax.cla()
    ax.plot(bw.Ct_y, '.-')
    ax.set_title("Ct_y")

    new_ylim = ax.get_ylim()
    ax.set_ylim([min(old_ylim[0], new_ylim[0]), max(old_ylim[1], new_ylim[1])])


def show_alpha(ax, bw):
    ax.cla()
    ims = ax.imshow(np.array(bw.alpha_list[-19:] + [bw.alpha]).T, aspect='auto', interpolation='nearest')

    ax.set_title("State Estimate ($\\alpha$)")
    live_nodes = [x for x in np.arange(bw.N) if x not in bw.dead_nodes]
    ax.set_yticks(live_nodes)
    ax.set_ylabel("bubble")
    ax.set_xlabel("steps (ago)")
    # ax.set_xticks([0.5,5,10,15,20])
    # ax.set_xticklabels([-20, -15, -10, -5, 0])
def show_behavior_variables(ax, bw, obs):
    ax.cla()
    ax.plot(bw.beh_list[-20:])
    ax.plot(obs[-20:])
    ax.set_ylim([-21,21])
    ax.set_title("Behavior prediction")

def show_A_eigenspectrum(ax, bw):
    ax.cla()
    eig = np.sort(np.linalg.eigvals(bw.A))[::-1]
    ax.plot(eig, '.')
    ax.set_title("Eigenspectrum of A")
    ax.set_ylim([0,1])

# TODO: remove this?
def mean_distance(data, shift=1):
    x = data - data.mean(axis=0)
    T = x.shape[0]

    differences = x[0:T - shift] - x[shift:T]
    distances = np.linalg.norm(differences, axis=1)

    return distances.mean()

def show_data_distance(ax, data, end_of_block, max_step=50):
    old_ylim = ax.get_ylim()
    ax.cla()
    start = max(end_of_block-3*max_step, 0)
    d = data[start:end_of_block]
    if d.shape[0] > 10:
        shifts = np.arange(0,min(d.shape[0]//2, max_step))
        distances = [mean_distance(d, shift) for shift in shifts]
        ax.plot(shifts, distances)
    ax.set_xlim([0,max_step])
    new_ylim = ax.get_ylim()
    ax.set_ylim([0, max(old_ylim[1], new_ylim[1])])
    ax.set_title(f"dataset[{start}:{end_of_block}] distances")
    ax.set_xlabel("offset")
    ax.set_ylabel("distance")

def show_nstep_pred_pdf(ax, bw, data, current_index, other_axis, fig, n=0):
    # vmin = np.inf
    # vmax = -np.inf
    if ax.collections:
        vmax = ax.collections[-3].colorbar.vmax
        vmin = ax.collections[-3].colorbar.vmin
        ax.collections[-3].colorbar.remove()
    ax.cla()
    other_axis: plt.Axes

    xlim = other_axis.get_xlim()
    ylim = other_axis.get_ylim()
    density = 50
    x_bins = np.linspace(*xlim, density+1)
    y_bins = np.linspace(*ylim, density+1)
    pdf = np.zeros(shape=(density, density))
    for i in range(density):
        for j in range(density):
            x = np.array([x_bins[i] + x_bins[i+1], y_bins[j] + y_bins[j+1]])/2
            b_values = bw.logB_jax(x, bw.mu, bw.L, bw.L_diag)
            pdf[i, j] = bw.alpha @ np.linalg.matrix_power(bw.A,n) @ np.exp(b_values)
    # cmesh = ax.pcolormesh(x_bins,y_bins,pdf.T, vmin=min(vmin, pdf.min()), vmax=max(vmax, pdf.max()))
    # cmesh = ax.pcolormesh(x_bins,y_bins,pdf.T, vmin=0, vmax=0.03) #log, vmin=-15, vmax=-5
    cmesh = ax.pcolormesh(x_bins,y_bins,pdf.T) #log, vmin=-15, vmax=-5
    fig.colorbar(cmesh)
    if current_index+n < data.shape[0]:
        to_draw = data[current_index+n]
        ax.scatter(to_draw[0], to_draw[1], c='red', alpha=.25)

    to_draw = data[current_index]
    ax.scatter(to_draw[0], to_draw[1], c='red')
    ax.set_title(f"{n}-step pred. at t={current_index}")

def show_w(ax,bw):
    ax.cla()
    ax.plot(bw.D@bw.Ct_y, '.-')
    ax.set_ylim([-1.1, 1.1])
    ax.set_xlabel("bubble #")
    ax.set_ylabel("weight magnitude")

def show_w_sideways(ax,bw, obs):
    ax.cla()
    w = np.array(bw.D @ bw.Ct_y)
    w[bw.dead_nodes] = 0

    a = np.array(bw.alpha)
    a = a / np.max(a)
    ax.plot(w, np.arange(w.size), alpha=0.25)
    ax.scatter(w, np.arange(w.size), alpha=a, c="C0")
    ylim = ax.get_ylim()
    ax.vlines(obs[-1], alpha=.5, ymin=ylim[0], ymax=ylim[1], colors="C1" )
    ax.set_ylabel("bubble #")
    ax.set_xlabel("weight magnitude")
    ax.set_title(r"Weights (times $\alpha$)")
    ax.set_xlim([-21, 21])






# todo: delete
def br_plot_2d(br):
    fig, ax = plt.subplots()
    s = np.load(br.file)
    data = s['y'][0]
    n_obs = np.array(br.n_obs)
    plot_2d(ax, data, br.A, br.mu, br.L, n_obs)
    plt.show()

def plot(fname):
    import pickle
    with open(fname, "rb") as fhan:
        br = pickle.load(fhan)
    if "vdp" in br.file:
        br_plot_2d(br)
    elif "lorenz" in br.file:
        br_plot_3d(br)
    elif "clock" in br.file:
        br_plot_2d(br)
    else:
        raise Exception("Cannot detect trajectory type from saved file.")



if __name__ == '__main__':
    import glob
    files = glob.glob("generated/bubblewrap_runs/bubblewrap_run_2023*")
    files.sort()
    plot(files[-1])
