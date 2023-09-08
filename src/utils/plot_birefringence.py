import numpy as np
# import plotly
import plotly.graph_objects as go
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
import stackview


def plot_volume_plotly(optical_info, voxels_in=None, opacity=0.5, colormap='gray', fig=None):
    '''Plots a 3D array with the non-zero voxels shaded.'''
    voxels = voxels_in * 1.0
    # Check if this is a torch tensor
    if not isinstance(voxels_in, np.ndarray):
        try:
            voxels = voxels.detach()
            voxels = voxels.cpu().abs().numpy()
        except:
            pass
    voxels = np.abs(voxels)
    err = ("The set of voxels are expected to have non-zeros values. If the " +
        "BirefringentVolume was cropped to fit into a region, the non-zero values " +
        "may no longer be included.")
    assert voxels.any(), err

    import plotly.graph_objects as go
    volume_shape = optical_info['volume_shape']
    volume_size_um = [optical_info['voxel_size_um'][i] * optical_info['volume_shape'][i] for i in range(3)]
    # Define grid
    coords = np.indices(np.array(voxels.shape)).astype(float)
    # Shift by half a voxel and multiply by voxel size
    coords = [(coords[i]+0.5) * optical_info['voxel_size_um'][i] for i in range(3)]
    if fig is None:
        fig = go.Figure()
    fig.add_volume(
        x=coords[0].flatten(),
        y=coords[1].flatten(),
        z=coords[2].flatten(),
        value=voxels.flatten() / voxels.max(),
        isomin=0,
        isomax=0.1,
        opacity=opacity, # needs to be small to see through all surfaces
        surface_count=20, # needs to be a large number for good volume rendering
        # colorscale=colormap
        )
    camera = {'eye': {'x': 50, 'y': 0, 'z': 0}}
    fig.update_layout(
        scene=dict(
            xaxis = {"nticks": volume_shape[0], "range": [0, volume_size_um[0]]},
            yaxis = {"nticks": volume_shape[1], "range": [0, volume_size_um[1]]},
            zaxis = {"nticks": volume_shape[2], "range": [0, volume_size_um[2]]},
            xaxis_title='Axial dimension',
            aspectratio = {"x": volume_size_um[0], "y": volume_size_um[1], "z": volume_size_um[2]},
            aspectmode = 'manual'
            ),
        scene_camera=camera,
        margin={'r': 0, 'l': 0, 'b': 0, 't': 0},
        autosize=True
        )
    # fig.data = fig.data[::-1]
    # fig.show()
    return fig

def plot_surface(delta_n, fig=None, colormap='gray'):
    volume_shape = [8, 32, 32]
    volume_size_um = [1 * volume_shape[i] for i in range(3)]
    voxels = np.abs(delta_n)
    # Define grid
    coords = np.indices(np.array(voxels.shape)).astype(float)
    # Shift by half a voxel and multiply by voxel size
    coords = [(coords[i]+0.5) * volume_size_um[i] for i in range(3)]
    # coords = np.mgrid[:8, :32, :32]
    # fig = None
    if fig is None:
        fig = go.Figure()
    fig.add_volume(
        x=coords[0].flatten(),
        y=coords[1].flatten(),
        z=coords[2].flatten(),
        value=voxels.flatten() / voxels.max(),
        isomin=0,
        isomax=0.1,
        opacity=0.1, # needs to be small to see through all surfaces
        surface_count=20, # needs to be a large number for good volume rendering
        colorscale=colormap
        )
    camera = {'eye': {'x': 50, 'y': 0, 'z': 0}}
    fig.update_layout(
        scene=dict(
            # xaxis = {"nticks": volume_shape[0], "range": [0, volume_size_um[0]]},
            # yaxis = {"nticks": volume_shape[1], "range": [0, volume_size_um[1]]},
            # zaxis = {"nticks": volume_shape[2], "range": [0, volume_size_um[2]]},
            xaxis_title='Axial dim',
            yaxis_title='',
            zaxis_title='',
            aspectratio = {"x": volume_size_um[0], "y": volume_size_um[1], "z": volume_size_um[2]},
            aspectmode = 'manual'
            ),
        scene_camera=camera,
        margin={'r': 0, 'l': 0, 'b': 0, 't': 0},
        # autosize=True,
        scene_xaxis_showticklabels=False,
        scene_yaxis_showticklabels=False,
        scene_zaxis_showticklabels=False,
        )
    return fig

def plot_lines(delta_n, optic_axis, fig=None, draw_spheres=True):
    # delta_n = delta_n_gt
    # optic_axis = optic_axis_gt
    op_axis_mag = np.linalg.norm(optic_axis, axis=0)
    optic_axis[:, op_axis_mag != 0] = (
            optic_axis[:, op_axis_mag != 0] / op_axis_mag[op_axis_mag != 0]
        )
    colormap='Bluered_r'
    size_scaler=5
    delta_n_ths=0.1
    delta_n /= np.max(np.abs(delta_n))
    delta_n[np.abs(delta_n)<delta_n_ths] = 0

    import plotly.graph_objects as go
    volume_shape = [8, 32, 32]
    volume_size_um = [1 * volume_shape[i] for i in range(3)]
    [dz, dxy, dxy] = volume_size_um
    # Define grid
    coords = np.indices(np.array(delta_n.shape)).astype(float)

    coords_base = [(coords[i] + 0.5) * volume_size_um[i] for i in range(3)]
    coords_tip =  [(coords[i] + 0.5 + optic_axis[i,...] * delta_n * 0.75) * volume_size_um[i] for i in range(3)]

    # Plot single line per voxel, where it's length is delta_n
    z_base, y_base, x_base = coords_base
    z_tip, y_tip, x_tip = coords_tip

    # Don't plot zero values
    mask = delta_n==0
    x_base[mask] = np.NaN
    y_base[mask] = np.NaN
    z_base[mask] = np.NaN
    x_tip[mask] = np.NaN
    y_tip[mask] = np.NaN
    z_tip[mask] = np.NaN

    # Gather all rays in single arrays, to plot them all at once, placing NAN in between them
    array_size = 3 * len(x_base.flatten())
    # Prepare colormap
    all_x = np.empty((array_size))
    all_x[::3] = x_base.flatten()
    all_x[1::3] = x_tip.flatten()
    all_x[2::3] = np.NaN
    all_y = np.empty((array_size))
    all_y[::3] = y_base.flatten()
    all_y[1::3] = y_tip.flatten()
    all_y[2::3] = np.NaN
    all_z = np.empty((array_size))
    all_z[::3] = z_base.flatten()
    all_z[1::3] = z_tip.flatten()
    all_z[2::3] = np.NaN
    # Compute colors
    all_color = np.empty((array_size))
    all_color[::3] =    (x_base-x_tip).flatten() ** 2 + \
                        (y_base-y_tip).flatten() ** 2 + \
                        (z_base-z_tip).flatten() ** 2
    # all_color[::3] =  delta_n.flatten() * 1.0
    all_color[1::3] = all_color[::3]
    all_color[2::3] = 0
    all_color[np.isnan(all_color)] = 0

    err = ("The BirefringentVolume is expected to have non-zeros values. If the " +
        "BirefringentVolume was cropped to fit into a region, the non-zero values " +
        "may no longer be included.")
    assert any(all_color != 0), err

    all_color[all_color!=0] -= all_color[all_color!=0].min()    # twice the length of the nonzero delta_n
    all_color += 0.5
    all_color /= all_color.max()

    if fig is None:
        fig = go.Figure()
    fig.add_scatter3d(z=all_x, y=all_y, x=all_z,
        marker={"color": all_color, "colorscale": colormap, "size": 4},
        line={"color": all_color, "colorscale": colormap, "width": size_scaler},
        connectgaps=False,
        mode='lines'
        )
    if draw_spheres:
        fig.add_scatter3d(z=x_base.flatten(), y=y_base.flatten(), x=z_base.flatten(),
            marker={"color": all_color[::3] - 0.5,
                    "colorscale": colormap,
                    "size": size_scaler * 5 * all_color[::3]},
            line={"color": all_color[::3] - 0.5, "colorscale": colormap, "width": 5},
            mode='markers')
    camera = {'eye': {'x': 50, 'y': 0, 'z': 0}}
    fig.update_layout(
    #     scene=dict(
    #         xaxis = {"nticks": volume_shape[0], "range": [0, volume_size_um[0]]},
    #         yaxis = {"nticks": volume_shape[1], "range": [0, volume_size_um[1]]},
    #         zaxis = {"nticks": volume_shape[2], "range": [0, volume_size_um[2]]},
    #         xaxis_title = 'Axial dimension',
    #         aspectratio = {"x": volume_size_um[0], "y": volume_size_um[1], "z": volume_size_um[2]},
    #         aspectmode = 'manual'
    #         ),
        scene_camera=camera,
        margin={'r': 0, 'l': 0, 'b': 0, 't': 0},
        # layout_showlegend=False,
        )
    fig.update_traces(showlegend=False)
    # fig.data = fig.data[::-1]
    # fig.show()
    return fig

def plot_surface_lines(delta_n, optic_axis):
    fig = plot_surface(delta_n)
    fig = plot_lines(delta_n, optic_axis,
                                fig=fig, draw_spheres=False)
    return fig

def plot_obj_from_file(filename):
    obj = imread(filename)
    delta_n = obj[0, ...]
    optic_axis = obj[1:, ...]
    fig = plot_surface_lines(delta_n, optic_axis)
    return fig


if __name__ == "__main__":
    if False:
        DATA_PATH = "/mnt/efs/shared_data/restorators/spheres"
        obj = imread(DATA_PATH + '/objects/0001_sphere.tiff')
        delta_n = obj[0, ...]
        optic_axis = obj[1:3, ...]
        stackview.orthogonal(delta_n, continuous_update=True)
        plt.imshow(delta_n[4, ...])
        plt.show(block=True)
        plt.pause(0.2)

    # pred_filename = f"inference/round1/pred.tiff"
    # gt_filename = f"inference/round1/gt.tiff"
    # obj_pred = imread(pred_filename)
    # delta_n_pred = obj_pred[0, ...]
    # optic_axis_pred = obj_pred[1:, ...]
    # obj_gt = imread(gt_filename)
    # delta_n_gt = obj_gt[0, ...]
    # optic_axis_gt = obj_gt[1:, ...]

    # # predicted output
    # pred_fig = plot_surface(delta_n_pred)
    # # pred_fig.show()
    # pred_fig_spheres = plot_lines(delta_n_pred, optic_axis_pred,
    #                               fig=pred_fig, draw_spheres=False)
    # pred_fig_spheres.update_layout(
    # title={
    #     'text': "Predicted volume",
    #     # 'y':0.9,
    #     # 'x':0.5,
    #     'xanchor': 'center',
    #     'yanchor': 'top'})
    # pred_fig_spheres.show()
    
    # # ground truth
    # gt_fig = plot_surface(delta_n_gt)
    # gt_fig_spheres = plot_lines(delta_n_gt, optic_axis_gt,
    #                             fig=gt_fig, draw_spheres=False)
    # gt_fig_spheres.show()

    ## plotting predictions of the model
    # pred_filename = f"inference/round1/pred.tiff"
    # fig = plot_obj_from_file(pred_filename)
    # fig.show()

    # pred_filename = f"inference/relu/pred.tiff"
    # lf_filename = f"inference/relu/source.tiff"
    # fig = plot_obj_from_file(pred_filename)
    # fig.show()
    
    # obj_pred = imread(pred_filename)
    # delta_n_pred = obj_pred[0, ...]
    # optic_axis_pred = obj_pred[1:, ...]
    # pred_fig = plot_surface_lines(delta_n_pred, optic_axis_pred)
    # pred_fig.show()
    # obj_gt = imread(gt_filename)
    # delta_n_gt = obj_gt[0, ...]
    # optic_axis_gt = obj_gt[1:, ...]
    
    ob_filename = f"/mnt/efs/shared_data/restorators/spheres_cropped_cube/objects/0000_sphere.tiff"
    fig = plot_obj_from_file(ob_filename)
    fig.show()
