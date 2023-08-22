import matplotlib.pyplot as plt
import bubblewrap.plotting_functions as bpf



def test_axis_plots(make_br):

    fig, axs = plt.subplots(nrows=1, ncols=2)
    ax = axs[0]
    br = make_br()
    bw = br.bw
    obs, beh = br.data_source.get_history()

    offset = 3
    current_location, current_behavior = br.data_source.get_pair(item="last_seen", offset=0)
    predicted_location, _ = br.data_source.get_pair(item="last_seen", offset=offset)

    ### good ###
    bpf.plot_2d(ax, obs, bw)
    bpf.plot_current_2d(ax, obs, bw)
    bpf.show_A(ax, bw)
    bpf.show_alpha(ax, br)
    bpf.show_behavior_variables(ax, br, beh)
    bpf.show_A_eigenspectrum(ax, bw)
    bpf.show_data_distance(ax, obs, max_step=50)

    bpf.plot_2d(axs[0], obs, bw)
    bpf.show_nstep_pred_pdf(axs[1], bw, axs[0], fig=fig, current_location=current_location, offset_location=predicted_location,  offset=offset)


    ### to fix ###
    # bpf.br_plot_3d(br)
    # bpf.show_Ct_y(ax, regressor=bw.regressor)
    # bpf.show_w_sideways(ax, bw, current_behavior)

def test_comparison_plots(make_br):
    brs = [make_br() for _ in range(3)]
    bpf.compare_metrics(brs, offset=0)