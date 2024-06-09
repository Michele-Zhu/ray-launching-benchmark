import os
import time
from enum import Enum
from scipy.io import savemat
import tensorflow as tf
import pandas as pd
import numpy as np
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
from pathlib import Path

gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"  # '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
        # Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

tf.random.set_seed(1)  # Set global random seed for reproducibility


def load_positions_from_csv(csv_file_path):
    positions_df = pd.read_csv(csv_file_path)
    return positions_df.to_numpy().tolist()


def set_scene(scene, carrier_frequency_Hz):
    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization="V")

    scene.rx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization="V")

    scene.frequency = carrier_frequency_Hz

    # If set to False, ray tracing will be done per antenna element (slower for large arrays)
    scene.synthetic_array = False


def get_camera():
    return Camera("camera_0", position=[0, 0, 1000], orientation=[0, 0, 0], look_at=[0, 0, 0])


def plot_scene(scene, camera, scene_plot_file_path, rendering_resolution=(600, 400)):
    scene.render_to_file(camera=camera.name, paths=None, show_devices=True, show_paths=True,
                         filename=scene_plot_file_path, resolution=rendering_resolution)


def run_simulation_over_tx_rx_pairs(scene, tx_positions_list, rx_positions_list, ray_tracing_method,
                                    max_paths_depth, rays_num_samples, enable_scattering_flag, paths_limit,
                                    channel_parsing_flag):
    elapsed_time_matrix = np.zeros((len(tx_positions_list), len(rx_positions_list)))
    paths_data_list = []

    for tx_idx, tx_position in enumerate(tx_positions_list):  # for each BS as tx
        set_bs_in_sionna_scene(scene, tx_position)

        for rx_idx, rx_position in enumerate(rx_positions_list):  # for each UE as rx
            set_ue_in_sionna_scene(scene, rx_position)
            print(f"Running Sionna simulation (depth={max_paths_depth}, method={ray_tracing_method}, "
                  f"scattering={enable_scattering_flag}, tx={tx_idx}, rx={rx_idx})...")

            elapsed_time = run_sionna_simulation_with_timing(
                scene, ray_tracing_method, max_paths_depth,
                rays_num_samples, enable_scattering_flag,
                tx_idx, tx_position, rx_idx, rx_position, paths_limit, channel_parsing_flag)
            elapsed_time_matrix[tx_idx, rx_idx] = elapsed_time

    return elapsed_time_matrix


def set_bs_in_sionna_scene(scene, bs_position):
    scene.remove("tx")

    tx = Transmitter(name="tx", position=bs_position, orientation=[0, 0, 0])
    scene.add(tx)


def set_ue_in_sionna_scene(scene, ue_position):
    scene.remove("rx")

    rx = Receiver(name="rx", position=ue_position, orientation=[0, 0, 0])
    scene.add(rx)


def run_sionna_simulation_with_timing(scene, ray_tracing_method, max_paths_depth,
                                      rays_num_samples, enable_scattering_flag):
    sim_start = time.time()

    paths = run_sionna_simulation(scene, ray_tracing_method, max_paths_depth, rays_num_samples, enable_scattering_flag)

    sim_end = time.time()
    elapsed_time = sim_end - sim_start

    return elapsed_time


def run_sionna_simulation(scene, ray_tracing_method, max_paths_depth, rays_num_samples, enable_scattering_flag):
    paths = scene.compute_paths(method=ray_tracing_method, max_depth=max_paths_depth,
                                num_samples=rays_num_samples, scattering=enable_scattering_flag)
    return paths


def save_elapsed_time_matrix(elapsed_time_matrix_file_path, elapsed_time_matrix, save_format='.mat'):
    if save_format == '.mat':
        fm = os.path.splitext(elapsed_time_matrix_file_path)[0] + '.mat'
        # save the file in matlab format
        mymat = {'num': elapsed_time_matrix}
        savemat(fm, mymat)
    else:
        with open(elapsed_time_matrix_file_path, 'wb') as f:
            np.save(f, elapsed_time_matrix)


def run_simulation_over_tx(scene, tx_positions_list, rx_positions_list, ray_tracing_method,
                           max_paths_depth, rays_num_samples, enable_scattering_flag):
    elapsed_time_matrix = np.zeros(len(tx_positions_list))
    rx_name_list = set_rx_list_in_sionna_scene(scene, rx_positions_list)
    for tx_idx, tx_position in enumerate(tx_positions_list):
        tx_name_list = set_tx_list_in_sionna_scene(scene, [tx_position])
        print(f"Running Sionna simulation over GPU (depth={max_paths_depth}, method={ray_tracing_method}, "
              f"scattering={enable_scattering_flag}, tx={tx_idx})...")

        elapsed_time = run_sionna_simulation_with_timing(
            scene, ray_tracing_method, max_paths_depth,
            rays_num_samples, enable_scattering_flag)
        elapsed_time_matrix[tx_idx] = elapsed_time
        remove_tx_list_from_sionna_scene(scene, tx_name_list)

    remove_rx_list_from_sionna_scene(scene, rx_name_list)
    return elapsed_time_matrix


def set_tx_list_in_sionna_scene(scene, tx_list):
    tx_name_list = []
    for tx_index, tx_position in enumerate(tx_list):
        tx_name = "tx_{}".format(tx_index)
        tx = Transmitter(name=tx_name, position=tx_position, orientation=[0, 0, 0])
        scene.add(tx)
        tx_name_list.append(tx_name)
    return tx_name_list


def remove_tx_list_from_sionna_scene(scene, tx_name_list):
    for tx_name in tx_name_list:
        scene.remove(tx_name)


def set_rx_list_in_sionna_scene(scene, rx_list):
    rx_name_list = []
    for rx_index, rx_position in enumerate(rx_list):
        rx_name = "rx_{}".format(rx_index)
        rx = Receiver(name=rx_name, position=rx_position, orientation=[0, 0, 0])
        scene.add(rx)
        rx_name_list.append(rx_name)
    return rx_name_list


def remove_rx_list_from_sionna_scene(scene, rx_name_list):
    for rx_name in rx_name_list:
        scene.remove(rx_name)


def main():
    # Paths
    scene_file_path = r"./sionna_mitsuba_mesh/carla_scenario.xml"  # [xml scenario file path] # ["Scenario1", "Scenario2"]
    scene_plot_file_path = r"./plots/scene_plot.png"
    tx_positions_csv_file_path = r"./tx_rx_positions/tx_positions.csv"  # [Transmitter position csv file path]
    rx_positions_csv_file_path = r"./tx_rx_positions/rx_positions.csv"  # [Receiver position csv file path]
    elapsed_time_matrix_file_path_format_str = r"./elapsed_time/carla/elapsed_time_matrix_smp_{}_r_{}_m_{}_s_{}.npy"
    output_path = Path("elapsed_time/carla")
    output_path.mkdir(parents=True, exist_ok=True)

    # Import tx/rx positions
    tx_positions_list = load_positions_from_csv(tx_positions_csv_file_path)
    rx_positions_list = load_positions_from_csv(rx_positions_csv_file_path)

    # Parameters
    carrier_frequency_Hz = 28 * 1e9  # [Hz]
    rays_num_samples_list = [1e4, 163842, 1e6]
    max_paths_depths_list = list(range(1, 11))
    ray_tracing_methods_list = ["fibonacci"]  # ["fibonacci", "exhaustive"]
    enable_scattering_flag_list = [False, True]

    # Simulation setup
    scene = load_scene(scene_file_path)
    set_scene(scene, carrier_frequency_Hz)
    camera = get_camera()
    scene.add(camera)
    plot_scene(scene, camera, scene_plot_file_path)

    # Simulation run, component warm-up is not present in this code
    print("Running Sionna simulation...")
    for rays_num_samples in rays_num_samples_list:
        rays_num_samples_str = f"{rays_num_samples:.1E}"
        for ray_tracing_method in ray_tracing_methods_list:
            for enable_scattering_flag in enable_scattering_flag_list:
                for max_paths_depth in max_paths_depths_list:
                    print(f"Running Sionna simulation (sampled_rays_num={rays_num_samples_str}, "
                          f"depth={max_paths_depth}, method={ray_tracing_method}, "
                          f"scattering={enable_scattering_flag})...")
                    elapsed_time_matrix = run_simulation_over_tx(scene, tx_positions_list, rx_positions_list,
                                                                 ray_tracing_method,
                                                                 max_paths_depth, rays_num_samples,
                                                                 enable_scattering_flag)
                    elapsed_time_matrix_file_path = elapsed_time_matrix_file_path_format_str.format(
                        rays_num_samples_str, max_paths_depth, ray_tracing_method, enable_scattering_flag)
                    save_elapsed_time_matrix(elapsed_time_matrix_file_path, elapsed_time_matrix)

    print("Sionna simulation completed.")


if __name__ == '__main__':
    main()
