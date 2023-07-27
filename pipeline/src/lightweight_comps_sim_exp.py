from kfp import dsl
from kfp.dsl import HTML, Input, Output, Dataset


@dsl.component(
    packages_to_install=[
        "pandas==1.5.3",
        "s3fs",
        "pyarrow",
        "signals",
        "plotly",
        "scipy",
    ],
    base_image="python:3.9",
)
def simulate_data(
    simulation_data: Output[Dataset],
    underlying_cause_plot: Output[HTML],
    generated_signals_plot: Output[HTML],
    min_phase_length: int = 1500,
    max_phase_length: int = 2500,
    length: int = 500,
):
    import numpy as np
    from scipy.integrate import odeint
    import pandas as pd
    import plotly.subplots as sp
    import plotly.graph_objects as go

    # generate underlying causal factors
    # which are the durations of the states of comp a-c
    min_lenght = min_phase_length
    max_length = max_phase_length
    length = length
    comp_a_durations = np.random.randint(min_lenght, max_length, length)
    comp_b_durations = np.random.randint(min_lenght, max_length, length)

    # generate corresponding signal for causal factor of comp a
    comp_a_signal = np.concatenate(
        [[-1] * i if j % 2 == 0 else [1] * i for j, i in enumerate(comp_a_durations)]
    )
    comp_b_signal = np.concatenate(
        [[-1] * i if j % 2 == 0 else [1] * i for j, i in enumerate(comp_b_durations)]
    )

    # get list of gain values for dynamic system simulation
    def get_comp_signal(comp_duration):
        comp_kp_ls = [-1]
        for i in range(1, len(comp_duration)):
            comp_kp_ls.append(-1 if comp_kp_ls[i - 1] == 1 else 1)
        return np.array(comp_kp_ls)

    comp_a_kp_ls = get_comp_signal(comp_b_durations)

    ## component A signals

    # tau * dy2/dt2 + 2*zeta*tau*dy/dt + y = Kp*u
    tau = 20.0  # time constant
    zeta = 0.3  # damping factor
    theta = 100.0  # no time delay
    du = 1.0  # change in u
    taup = 50

    # (3) ODE Integrator
    def second_order_model(x, t, Kp):
        y = x[0]
        dydt = x[1]
        dy2dt2 = (-2.0 * zeta * tau * dydt - y + Kp * du) / tau**2
        return [dydt, dy2dt2]

    def first_order_model(y, t, Kp):
        u = 1
        return (-y + Kp * u) / taup

    x0_2nd = [-1, 0]
    x0_1st = [-1.1, 0]
    y_2nd_ls = []
    y_1st_ls = []

    for Kp, tmax in zip(comp_a_kp_ls, comp_a_durations):
        t = np.linspace(0, tmax, tmax)
        x_2nd = odeint(second_order_model, x0_2nd, t, (Kp,))
        y_2nd = x_2nd[:, 0]
        y_2nd_ls.append(y_2nd)
        x0_2nd = list(x_2nd[-1, :])

        x_1st = odeint(first_order_model, x0_1st, t, (Kp - 0.1,))
        y_1st = x_1st[:, 0]
        y_1st_ls.append(y_1st)
        x0_1st = list(x_1st[-1, :])

    sig_a1 = comp_a_signal
    sig_a2 = np.concatenate(y_2nd_ls)
    sig_a3 = np.concatenate(y_1st_ls)

    min_len = min([len(sig) for sig in [comp_a_signal, comp_b_signal]])
    df = pd.DataFrame(
        dict(comp_a=comp_a_signal[:min_len], comp_b=comp_b_signal[:min_len])
    )

    t = np.linspace(0, min_len, min_len)
    sig_b1 = comp_b_signal
    sig_b2 = df.comp_b.cumsum() / df.comp_b.cumsum().max()
    sig_b3 = sig_b2 + 0.5 * np.sin(2 * np.pi * 1 / 1000 * t)

    sig_a1 = sig_a1[:min_len] + 0.5 * sig_b2[:min_len]
    sig_b1 = sig_b1[:min_len] + 0.1 * sig_a1[:min_len]

    # write out
    df = pd.DataFrame(
        dict(
            sig_a1=sig_a1[:min_len],
            sig_a2=sig_a2[:min_len],
            sig_a3=sig_a3[:min_len],
            sig_b1=sig_b1[:min_len],
            sig_b2=sig_b2[:min_len],
            sig_b3=sig_b3[:min_len],
        )
    )
    df.to_parquet(simulation_data.path)



    ## generate plots:

    fig = go.Figure()

    fig.add_trace(go.Scatter(y=comp_a_signal[0:15000], mode='lines', name='comp_a_signal'))
    fig.add_trace(go.Scatter(y=comp_b_signal[0:15000], mode='lines', name='comp_b_signal'))
    fig.update_layout(title_text='Underlying factor of change for subsystem a and b')

    fig.write_html(underlying_cause_plot.path)



    start_idx = 0
    end_idx = 6000

    # Create subplots
    fig = sp.make_subplots(rows=2, cols=1)

    # Add traces
    fig.add_trace(go.Scatter(y=sig_a1[start_idx:end_idx], mode='lines', name='sig_a1'), row=1, col=1)
    fig.add_trace(go.Scatter(y=sig_a2[start_idx:end_idx], mode='lines', name='sig_a2'), row=1, col=1)
    fig.add_trace(go.Scatter(y=sig_a3[start_idx:end_idx], mode='lines', name='sig_a3'), row=1, col=1)
    fig.add_trace(go.Scatter(y=sig_b1[start_idx:end_idx], mode='lines', name='sig_b1'), row=2, col=1)
    fig.add_trace(go.Scatter(y=sig_b2[start_idx:end_idx], mode='lines', name='sig_b2'), row=2, col=1)
    fig.add_trace(go.Scatter(y=sig_b3[start_idx:end_idx], mode='lines', name='sig_b3'), row=2, col=1)

    fig.update_layout(title_text='Some random area of the generated signals')
    fig.write_html(generated_signals_plot.path)



@dsl.component(
    packages_to_install=[
        "pandas==1.5.3",
        "pyarrow"
    ],
    base_image="python:3.9",
)
def split_data(
    simulate_data: Input[Dataset],
    train_data: Output[Dataset],
    val_data: Output[Dataset],
    test_data: Output[Dataset],
):
    import pandas as pd
    # read data
    df_data = pd.read_parquet(simulate_data.path)

    total_length = len(df_data)
    start_idx_val = int(0.8*total_length)
    end_idx_val = int(0.9*total_length)

    df_train = df_data[:start_idx_val]
    df_val = df_data[start_idx_val:end_idx_val]
    df_test =df_data[end_idx_val:]

    df_train.to_parquet(train_data.path)
    df_val.to_parquet(val_data.path)
    df_test.to_parquet(test_data.path)


